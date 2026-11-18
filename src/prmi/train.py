"""Entraînement du U-Net sur le train (masques corrigés par défaut), avec early stopping sur le val (masques corrigés).

Perte : 0,5 x Dice + 0,5 x BCE. Le Dice est calculé carré par carré, sur les carrés qui contiennent de la racine, pour
qu'une grosse racine n'écrase pas les petites des autres carrés du batch ; les carrés sans racine sont laissés à la BCE.
Option `cldice` (poids a) : la part Dice devient (1 - a) x Dice + a x clDice (Shit et al., CVPR 2021), qui compare les
lignes centrales « douces » (squelettisation par érosions successives, CLDICE_ITERS itérations), carré par carré aussi.
Taux d'apprentissage différencié : encoder_lr pour l'encodeur pré-entraîné, lr pour le décodeur, qui part de zéro.
Les deux décroissent en cosinus jusqu'à LR_MIN sur l'ensemble des époques prévues.

Suivi sur les images entières, à chaque époque :
- global_dice : Dice calculé sur tous les pixels du val, images vides comprises (décide de l'early stopping) ;
- dice : Dice par image, moyenne sur les images avec racine du val ;
- train_dice : même calcul sur N_TRAIN_EVAL images du train avec racine, toujours les mêmes ;
- roots_found : part des racines (morceaux du masque) dont au moins la moitié de la ligne centrale est prédite ;
- ghosts : nombre de taches prédites d'au moins MIN_FP pixels sans contact avec une racine ;
- length_err : erreur relative de longueur du squelette, médiane sur les images avec racine du val ;
- fp_empty : part des images sans racine du val où le modèle détecte au moins MIN_FP pixels ;
- cldice : clDice par image (moyenne sur les images avec racine) : moyenne harmonique de la part du squelette prédit dans
  le masque de référence (précision) et de la part du squelette de référence couverte par la prédiction (sensibilité) ;
- global_cldice : même calcul sur tous les pixels du val, images vides comprises ;
- val_loss : perte standard (Dice + BCE, sans clDice, pour rester comparable d'un essai à l'autre), sur le val.

Grand val (`score_run`) : une fois l'entraînement fini, le meilleur modèle est évalué sur des images du val officiel aux
masques d'origine comblés, lues sur la clé par paquets (rien n'est gardé en mémoire) ; résultat dans LARGE_FILE.
Les premiers scores, contre les masques d'origine bruts, restent dans val_large_images.csv (filled=False).

Sortie : data/processed/runs/<nom>/ avec config.json (réglages, puis meilleure époque à la fin), history.csv,
best.pt (poids de la meilleure époque) et val_images.csv (scores image par image à la meilleure époque, pour comparer
les variantes par paires), puis le tableau du grand val.

Usage (depuis la racine du projet) : PYTHONPATH=src python -m prmi.train --name unet_effb0
"""
import argparse
import json
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage as ndi
from scipy.stats import wilcoxon
from skimage.morphology import skeletonize
from torch.utils.data import DataLoader

from prmi import dataset
from prmi.dataset import RootCrops, load_extra, load_lot, pad_image
from prmi.inventory import ROOT
from prmi.model import build_model, count_used_parameters

RUNS_DIR = ROOT / "data" / "processed" / "runs"
LR_MIN = 1e-6
N_TRAIN_EVAL = 44   # autant que d'images avec racine dans le val
MIN_FP = 100        # pixels ; en dessous, pas de fausse détection (la plupart des morceaux de racine annotés dépassent 500 px)
EIGHT = np.ones((3, 3))     # connexité à 8 voisins
LARGE_FILE = "val_large_comble_images.csv"   # grand val, référence = masques d'origine comblés
CLDICE_ITERS = 10   # itérations de la squelettisation douce : au moins la demi-largeur des plus grosses racines (à ÷ 2)


def dice_loss(logits, target, smooth=1.0):
    """1 - Dice, carré par carré, moyenné sur les carrés qui contiennent de la racine (0 s'il n'y en a aucun)."""
    p, y = torch.sigmoid(logits).flatten(1), target.flatten(1)
    has_root = y.sum(1) > 0
    if not has_root.any():
        return logits.sum() * 0
    dice = (2 * (p * y).sum(1) + smooth) / (p.sum(1) + y.sum(1) + smooth)
    return 1 - dice[has_root].mean()


def soft_erode(x):
    return torch.min(-F.max_pool2d(-x, (3, 1), 1, (1, 0)), -F.max_pool2d(-x, (1, 3), 1, (0, 1)))


def soft_dilate(x):
    return F.max_pool2d(x, 3, 1, 1)


def soft_skeleton(x, iters=CLDICE_ITERS):
    """Squelette « doux » d'une carte de probabilité (N, 1, H, W), dérivable (Shit et al., 2021)."""
    skeleton = F.relu(x - soft_dilate(soft_erode(x)))
    for _ in range(iters):
        x = soft_erode(x)
        delta = F.relu(x - soft_dilate(soft_erode(x)))
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def cldice_loss(logits, target, smooth=1.0):
    """1 - clDice, carré par carré, moyenné sur les carrés qui contiennent de la racine (0 s'il n'y en a aucun)."""
    has_root = target.flatten(1).sum(1) > 0
    if not has_root.any():
        return logits.sum() * 0
    p = torch.sigmoid(logits)
    sp, sy = soft_skeleton(p).flatten(1), soft_skeleton(target).flatten(1)
    p, y = p.flatten(1), target.flatten(1)
    precision = ((sp * y).sum(1) + smooth) / (sp.sum(1) + smooth)
    sensitivity = ((sy * p).sum(1) + smooth) / (sy.sum(1) + smooth)
    cldice = 2 * precision * sensitivity / (precision + sensitivity)
    return 1 - cldice[has_root].mean()


def loss_fn(logits, target, cldice=0.0):
    """0,5 x BCE + 0,5 x [(1 - cldice) x Dice + cldice x clDice]."""
    overlap = dice_loss(logits, target)
    if cldice:
        overlap = (1 - cldice) * overlap + cldice * cldice_loss(logits, target)
    return 0.5 * overlap + 0.5 * F.binary_cross_entropy_with_logits(logits, target)


def count_roots(pred, mask, skeleton):
    """Nombre de morceaux de racine du masque, et nombre de ceux dont au moins la moitié de la ligne centrale est prédite."""
    labels, n = ndi.label(mask, structure=EIGHT)
    if n == 0:
        return 0, 0
    index = range(1, n + 1)
    return n, int((ndi.sum(skeleton & pred, labels, index) >= 0.5 * ndi.sum(skeleton, labels, index)).sum())


def count_ghosts(pred, mask):
    """Nombre de taches prédites d'au moins MIN_FP pixels qui ne touchent aucune racine."""
    labels, n = ndi.label(pred, structure=EIGHT)
    if n == 0:
        return 0
    index = range(1, n + 1)
    return int(((ndi.sum(pred, labels, index) >= MIN_FP) & (ndi.sum(mask, labels, index) == 0)).sum())


def cldice_score(skel_pred_in, skel_pred, skel_ref_in, skel_ref):
    """clDice à partir des comptes de pixels de squelette (0 si rien n'est prédit)."""
    precision = skel_pred_in / skel_pred if skel_pred else 0.0
    sensitivity = skel_ref_in / skel_ref
    return 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0.0


def image_scores(pred, mask):
    """Scores d'une image, à partir de masques booléens. Dice, clDice, erreur de longueur : NaN si l'image n'a pas de racine."""
    skeleton = skeletonize(mask) if mask.any() else mask
    skel_pred = skeletonize(pred) if pred.any() else pred
    roots, found = count_roots(pred, mask, skeleton)
    scores = {"tp": int((pred & mask).sum()), "fp": int((pred & ~mask).sum()), "fn": int((~pred & mask).sum()),
              "roots": roots, "roots_found": found, "ghosts": count_ghosts(pred, mask), "pred_px": int(pred.sum()),
              "skel_pred": int(skel_pred.sum()), "skel_pred_in": int((skel_pred & mask).sum()),
              "skel_ref": int(skeleton.sum()), "skel_ref_in": int((skeleton & pred).sum())}
    if not mask.any():
        return {"dice": np.nan, "cldice": np.nan, "length_err": np.nan, **scores}
    return {"dice": 2 * scores["tp"] / (pred.sum() + mask.sum()),
            "cldice": cldice_score(scores["skel_pred_in"], scores["skel_pred"], scores["skel_ref_in"], scores["skel_ref"]),
            "length_err": abs(scores["skel_pred"] - scores["skel_ref"]) / scores["skel_ref"], **scores}


def summarize(df):
    """Moyennes d'un lot à partir du tableau image par image (colonnes clDice absentes des anciens tableaux : ignorées)."""
    tp, fp, fn = df["tp"].sum(), df["fp"].sum(), df["fn"].sum()
    out = {"dice": df["dice"].mean(), "global_dice": 2 * tp / max(2 * tp + fp + fn, 1),
           "roots_found": df["roots_found"].sum() / max(df["roots"].sum(), 1), "ghosts": int(df["ghosts"].sum()),
           "length_err": df["length_err"].median(), "fp_empty": (df.loc[~df["has_root"], "pred_px"] >= MIN_FP).mean()}
    if "cldice" in df:
        out["cldice"] = df["cldice"].mean()
        out["global_cldice"] = cldice_score(*(df[c].sum() for c in ("skel_pred_in", "skel_pred", "skel_ref_in", "skel_ref")))
    return out


def work_logits(model, item, device):
    """Sortie du modèle sur une image entière, à la résolution de travail."""
    x, y = pad_image(item["image"], item["mask"])
    h, w = y.shape[1:]
    return model(x[None].to(device))[..., :h, :w]


def full_logits(model, item, device):
    """Sortie du modèle ramenée à la résolution d'origine, et masque d'origine en tenseur."""
    full = item["mask_full"]
    logits = F.interpolate(work_logits(model, item, device), size=full.shape, mode="bilinear", align_corners=False)
    return logits, torch.from_numpy(full[None, None].astype(np.float32)).to(device)


@torch.no_grad()
def predict(model, item, device):
    """Masque prédit (booléen) à la résolution de travail, pour l'affichage : probabilité supérieure à 0,5."""
    model.eval()
    return work_logits(model, item, device)[0, 0].cpu().numpy() > 0


@torch.no_grad()
def predict_proba(model, item, device):
    """Carte de probabilité (sigmoïde des logits) à la résolution de travail, pour l'affichage. Seuil de la
    prédiction : 0,5."""
    model.eval()
    return torch.sigmoid(work_logits(model, item, device))[0, 0].cpu().numpy()


@torch.no_grad()
def evaluate(model, items, device):
    """Évalue le modèle sur des images entières. Renvoie les moyennes du lot et le tableau image par image."""
    model.eval()
    rows, losses = [], []
    for it in items:
        logits, y = full_logits(model, it, device)
        losses.append(loss_fn(logits, y).item())
        rows.append({"name": it["name"], "group": it["group"], "type": it["type"], "has_root": bool(it["mask_full"].any()),
                     **image_scores(logits[0, 0].cpu().numpy() > 0, it["mask_full"])})
    df = pd.DataFrame(rows)
    return {**summarize(df), "val_loss": np.mean(losses)}, df


def train_eval_items(items, seed=0):
    """Images du train suivies à chaque époque : N_TRAIN_EVAL images avec racine, tirées une fois pour toutes."""
    rooted = [it for it in items if dataset.work_mask(it).any()]
    return [rooted[i] for i in sorted(np.random.default_rng(seed).choice(len(rooted), N_TRAIN_EVAL, replace=False))]


def train(train_items, val_items, name="unet_effb0", epochs=100, patience=20, batch=8, lr=1e-3, encoder_lr=1e-4,
          weight_decay=1e-4, seed=0, augment=False, crops_per_epoch=None, train_eval=None, cldice=0.0, occlusion=0.0,
          init=None):
    """Entraîne le modèle. Renvoie le modèle (poids de la meilleure époque), l'historique et la meilleure époque.

    crops_per_epoch : nombre de carrés par époque (par défaut, CROPS par image). train_eval : images du train sur
    lesquelles suivre le Dice train (par défaut, N_TRAIN_EVAL images avec racine tirées dans train_items). cldice :
    poids du clDice dans la part Dice de la perte (0 : perte standard). occlusion : probabilité de coller de la terre sur
    les racines d'un carré (`dataset.occlude`, avec augment seulement). init : nom d'un essai dont on reprend les poids
    (meilleure époque) pour l'affiner ; il est évalué avant tout entraînement (époque 0) et reste le meilleur modèle si
    aucune époque ne le dépasse.
    """
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = RUNS_DIR / name
    out.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(RootCrops(train_items, seed, augment, crops_per_epoch, occlusion), batch_size=batch, shuffle=True,
                        drop_last=True)
    train_eval = train_eval or train_eval_items(train_items, seed)
    model = build_model().to(device)
    if init:
        model.load_state_dict(torch.load(RUNS_DIR / init / "best.pt", map_location=device))
    n_params = count_used_parameters(model)
    groups = [{"params": model.encoder.parameters(), "lr": encoder_lr},
              {"params": [*model.decoder.parameters(), *model.segmentation_head.parameters()], "lr": lr}]
    opt = torch.optim.AdamW(groups, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=LR_MIN)
    config = {"name": name, "epochs": epochs, "patience": patience, "batch": batch, "lr": lr, "encoder_lr": encoder_lr,
              "lr_min": LR_MIN, "scale": round(train_items[0]["mask_full"].shape[1] / train_items[0]["mask"].shape[1], 3),
              "weight_decay": weight_decay, "seed": seed, "augment": augment, "crop": dataset.CROP, "crops": dataset.CROPS,
              "forced": dataset.FORCED, "encoder": "efficientnet-b0", "encoder_weights": "imagenet", "parameters": n_params,
              "n_train": len(train_items), "n_val": len(val_items),
              "train_masks": dict(Counter(it.get("masks", "corriges") for it in train_items)),
              "val_masks": val_items[0].get("masks", "corriges"), "crops_per_epoch": crops_per_epoch,
              "cldice": cldice, "cldice_iters": CLDICE_ITERS if cldice else None,
              "occlusion": occlusion, "occlude_range": dataset.OCCLUDE_RANGE if occlusion else None,
              "occlude_min": dataset.OCCLUDE_MIN if occlusion else None, "occlude_max": dataset.OCCLUDE_MAX if occlusion else None,
              "init": init, "stop_metric": "global_dice", "best_epoch": None,
              "best_score": None}
    (out / "config.json").write_text(json.dumps(config, indent=1))
    print(f"{device}, {n_params:,} paramètres utilisés, {len(train_items)} images train, {len(val_items)} images val, "
          f"{len(loader)} batchs par époque", flush=True)

    best, best_epoch, wait, history = -1.0, 0, 0, []
    if init:        # époque 0 : le modèle de départ, à battre
        t0 = time.time()
        scores, per_image = evaluate(model, val_items, device)
        history.append({"epoch": 0, "lr": lr, "lr_encoder": encoder_lr, "train_loss": np.nan,
                        "train_dice": evaluate(model, train_eval, device)[0]["dice"], **scores, "seconds": time.time() - t0})
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        best = scores["global_dice"]
        torch.save(model.state_dict(), out / "best.pt")
        per_image.to_csv(out / "val_images.csv", index=False)
        print(f"  0  départ ({init}) : Dice val {scores['dice']:.3f}  global {best:.3f}", flush=True)
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        lr_encoder, lr_decoder = sched.get_last_lr()
        model.train()
        total = 0.0
        for x, y in loader:
            opt.zero_grad()
            loss = loss_fn(model(x.to(device)), y.to(device), cldice)
            loss.backward()
            opt.step()
            total += loss.item()
        sched.step()
        scores, per_image = evaluate(model, val_items, device)
        train_dice = evaluate(model, train_eval, device)[0]["dice"]
        history.append({"epoch": epoch, "lr": lr_decoder, "lr_encoder": lr_encoder, "train_loss": total / len(loader),
                        "train_dice": train_dice, **scores, "seconds": time.time() - t0})
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        h = history[-1]
        print(f"{epoch:3d}/{epochs}  perte train {h['train_loss']:.3f}  val {h['val_loss']:.3f}  "
              f"Dice train {h['train_dice']:.3f}  val {h['dice']:.3f}  global {h['global_dice']:.3f}  "
              f"clDice {h['cldice']:.3f}  "
              f"racines {h['roots_found']:.0%}  fantômes {h['ghosts']}  longueur {h['length_err']:.2f}  "
              f"images vides touchées {h['fp_empty']:.2f}  lr {lr_encoder:.1e} / {lr_decoder:.1e}  {h['seconds']:.0f} s",
              flush=True)
        if scores["global_dice"] > best:
            best, best_epoch, wait = scores["global_dice"], epoch, 0
            torch.save(model.state_dict(), out / "best.pt")
            per_image.to_csv(out / "val_images.csv", index=False)
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping à l'époque {epoch} (meilleure : {best_epoch})", flush=True)
                break
    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    (out / "config.json").write_text(json.dumps({**config, "best_epoch": best_epoch, "best_score": float(best)}, indent=1))
    print(f"Meilleur Dice global du val : {best:.3f} à l'époque {best_epoch} -> {out}", flush=True)
    return model, pd.DataFrame(history), best_epoch


def run_exists(name):
    """Vrai si un entraînement existe sous ce nom, terminé ou interrompu (au moins une époque enregistrée)."""
    return (RUNS_DIR / name / "history.csv").exists() and (RUNS_DIR / name / "best.pt").exists()


def load_run(name):
    """Relit un entraînement, terminé ou interrompu. Renvoie le modèle, l'historique, la meilleure époque et le tableau du val."""
    out = RUNS_DIR / name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(weights=None).to(device)
    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    history = pd.read_csv(out / "history.csv")
    metric = "global_dice" if "global_dice" in history else "dice"      # essais antérieurs : arrêt sur le Dice par image
    best_epoch = int(history.loc[history[metric].idxmax(), "epoch"])
    return model, history, best_epoch, pd.read_csv(out / "val_images.csv")


@torch.no_grad()
def evaluate_rows(model, rows, device, scale=2, chunk=50, filled=True):
    """Évalue le modèle sur des images lues sur la clé (masques d'origine comblés ou bruts), par paquets : rien n'est
    gardé en mémoire."""
    tables = [evaluate(model, load_extra(rows.iloc[i:i + chunk], scale, filled=filled), device)[1]
              for i in range(0, len(rows), chunk)]
    return pd.concat(tables, ignore_index=True)


def score_run(name, rows, file=LARGE_FILE, filled=True):
    """Évalue le meilleur modèle d'un essai sur des images de la clé et enregistre le tableau (déjà fait : relu)."""
    out = RUNS_DIR / name / file
    if out.exists():
        return pd.read_csv(out)
    model = load_run(name)[0]
    scale = json.loads((RUNS_DIR / name / "config.json").read_text()).get("scale", 1)
    df = evaluate_rows(model, rows, next(model.parameters()).device, scale, filled=filled)
    df.to_csv(out, index=False)
    return df


def score_items(name, items, file):
    """Évalue le meilleur modèle d'un essai sur des images déjà chargées (load_lot) et enregistre le tableau dans `file`
    (déjà fait : relu)."""
    out = RUNS_DIR / name / file
    if out.exists():
        return pd.read_csv(out)
    model = load_run(name)[0]
    df = evaluate(model, items, next(model.parameters()).device)[1]
    df.to_csv(out, index=False)
    return df


def rescore(name, items):
    """Recalcule val_images.csv d'un essai (même modèle, mêmes images) pour y ajouter les colonnes récentes (clDice)."""
    out = RUNS_DIR / name / "val_images.csv"
    old = pd.read_csv(out)
    if "cldice" in old:
        return old
    model = load_run(name)[0]
    df = evaluate(model, items, next(model.parameters()).device)[1]
    assert np.allclose(old["dice"], df["dice"], atol=1e-3, equal_nan=True), f"{name} : scores différents de l'original"
    df.to_csv(out, index=False)
    return df


PRED_DIR = ROOT / "data" / "processed" / "predictions"


def save_pred(pred, path):
    """Masque prédit (booléen, résolution d'origine après prétraitement) en PNG un canal, 0 ou 255."""
    Image.fromarray(pred.astype(np.uint8) * 255).save(path, optimize=True)


@torch.no_grad()
def predict_items(model, items, device, out_dir):
    """Prédit chaque image (résolution d'origine, probabilité supérieure à 0,5), enregistre le masque dans
    out_dir/PRED_<nom>.png et renvoie le tableau des scores image par image."""
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for it in items:
        pred = full_logits(model, it, device)[0][0, 0].cpu().numpy() > 0
        save_pred(pred, out_dir / f"PRED_{it['name']}.png")
        rows.append({"name": it["name"], "group": it["group"], "type": it["type"], "has_root": bool(it["mask_full"].any()),
                     **image_scores(pred, it["mask_full"])})
    return pd.DataFrame(rows)


def prediction_loader(name, rows, lot, filled=True):
    """Pour parcourir un grand jeu déjà prédit (`predict_rows`) : fonction nom -> (image prétraitée, référence,
    prédiction), à la résolution d'origine. L'image et la référence sont relues sur la clé, la prédiction dans
    PRED_DIR/<name>/<lot>/."""
    by_name = rows.set_index("name")

    def get(image_name):
        it = load_extra(by_name.loc[[image_name]].reset_index(), 1, filled=filled)[0]
        pred = np.asarray(Image.open(PRED_DIR / name / lot / f"PRED_{image_name}.png")) > 127
        return it["image"], it["mask_full"], pred

    return get


def predict_rows(name, rows, lot, scale=2, chunk=50, filled=True):
    """Grand test : prédit les images `rows` lues sur la clé, par paquets, avec le meilleur modèle de l'essai `name`.
    Masques prédits dans PRED_DIR/<name>/<lot>/ (pour mesurer longueurs et diamètres sans refaire les prédictions),
    scores dans RUNS_DIR/<name>/<lot>_images.csv (déjà fait : relu)."""
    table = RUNS_DIR / name / f"{lot}_images.csv"
    if table.exists():
        return pd.read_csv(table)
    model = load_run(name)[0]
    device = next(model.parameters()).device
    out_dir = PRED_DIR / name / lot
    parts = []
    for i in range(0, len(rows), chunk):
        part = rows.iloc[i:i + chunk]
        df = predict_items(model, load_extra(part, scale, filled=filled), device, out_dir)
        parts.append(df.assign(tube_id=part["tube_id"].values, position=part["position"].values,
                               date=part["date"].values))
        print(f"\r{min(i + chunk, len(rows))}/{len(rows)} images", end="", flush=True)
    print()
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(table, index=False)
    return df


def key_scores(df):
    """Chiffres clés d'un jeu de test, à partir du tableau image par image : scores médians par image (images avec
    racine), Dice global (tous les pixels), racines trouvées, part des images sans racine touchées."""
    r = df[df["has_root"]]
    s = summarize(df)
    return {"images": len(df), "avec racine": len(r), "Dice médian": r["dice"].median(), "Dice global": s["global_dice"],
            "clDice médian": r["cldice"].median(), "erreur de longueur médiane": r["length_err"].median(),
            "racines trouvées": s["roots_found"], "images vides touchées": s["fp_empty"]}


@torch.no_grad()
def blob_scores(model, items, device):
    """Pour chaque image : scores sans filtre (résolution d'origine) et tableau de toutes ses taches prédites, avec ce
    qu'elles contiennent de vrai (`postprocess.components`)."""
    from prmi.postprocess import components
    model.eval()
    images, blobs = [], []
    for it in items:
        prob = torch.sigmoid(full_logits(model, it, device)[0])[0, 0].cpu().numpy()
        pred, ref = prob > 0.5, it["mask_full"]
        base = {"name": it["name"], "group": it["group"], "type": it["type"], "has_root": bool(ref.any())}
        images.append({**base, **image_scores(pred, ref)})
        comps = components(pred, prob, ref)[0]
        if len(comps):
            blobs.append(comps.assign(**base))
    return pd.DataFrame(images), pd.concat(blobs, ignore_index=True)


def blob_table(name, lot, items=None, rows=None, scale=2, chunk=50):
    """`blob_scores` sur des images chargées (`items`) ou lues sur la clé par paquets (`rows`, masques PRMI comblés),
    avec le meilleur modèle de l'essai `name`. Enregistré dans RUNS_DIR/<name>/taches_<lot>_{images,taches}.csv
    (déjà fait : relu)."""
    out = [RUNS_DIR / name / f"taches_{lot}_{k}.csv" for k in ("images", "taches")]
    if all(f.exists() for f in out):
        return tuple(pd.read_csv(f) for f in out)
    model = load_run(name)[0]
    device = next(model.parameters()).device
    if items is not None:
        parts = [blob_scores(model, items, device)]
    else:
        parts = [blob_scores(model, load_extra(rows.iloc[i:i + chunk], scale), device)
                 for i in range(0, len(rows), chunk)]
    tables = tuple(pd.concat([p[k] for p in parts], ignore_index=True) for k in range(2))
    for t, f in zip(tables, out):
        t.to_csv(f, index=False)
    return tables


@torch.no_grad()
def blob_crops(model, items, blobs, device, half=48):
    """Vignettes centrées sur des taches (`blobs`, avec leur colonne name) : image, probabilité, tache et référence, à
    la résolution d'origine. `items` : {nom : image chargée (load_lot ou load_extra, résolution de travail)}."""
    from prmi.postprocess import components
    model.eval()
    out = []
    for _, b in blobs.iterrows():
        it = items[b["name"]]
        prob = torch.sigmoid(full_logits(model, it, device)[0])[0, 0].cpu().numpy()
        labels = components(prob > 0.5, prob)[1]
        img = preprocess_full(it)
        h, w = prob.shape
        cy, cx = int(b["y"]), int(b["x"])
        y0, x0 = np.clip(cy - half, 0, max(h - 2 * half, 0)), np.clip(cx - half, 0, max(w - 2 * half, 0))
        sl = (slice(y0, y0 + 2 * half), slice(x0, x0 + 2 * half))
        out.append({"image": img[sl], "prob": prob[sl], "tache": labels[sl] == b["label"], "vraie": bool(b.get("vraie")),
                    "ref": it["mask_full"][sl], "titre": f"{b['px']} px | p {b['p_moy']:.2f} | allong. "
                                                         f"{b['allongement']:.1f}".replace(".", ",")})
    return out


def preprocess_full(item):
    """Image d'un élément remise à la résolution d'origine (pour l'affichage), entre 0 et 1."""
    h, w = item["mask_full"].shape
    img = item["image"].astype(np.float32) / (255 if item["image"].dtype == np.uint8 else 1)
    return np.asarray(Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR))


@torch.no_grad()
def postprocess_scores(model, items, device, variants):
    """Scores image par image de chaque variante de post-traitement, à la résolution d'origine. variants : {nom :
    fonction prob -> masque booléen}. Le modèle ne passe qu'une fois par image."""
    model.eval()
    rows = []
    for it in items:
        prob = torch.sigmoid(full_logits(model, it, device)[0])[0, 0].cpu().numpy()
        ref = it["mask_full"]
        base = {"name": it["name"], "group": it["group"], "type": it["type"], "has_root": bool(ref.any())}
        rows += [{"variante": v, **base, **image_scores(f(prob), ref)} for v, f in variants.items()]
    return pd.DataFrame(rows)


def postprocess_study(name, lot, variants, items=None, rows=None, scale=2, chunk=50):
    """`postprocess_scores` sur des images chargées (`items`) ou lues sur la clé par paquets (`rows`, masques PRMI
    comblés). Enregistré dans RUNS_DIR/<name>/postprocess_<lot>.csv (déjà fait : relu ; à supprimer si les variantes
    changent)."""
    out = RUNS_DIR / name / f"postprocess_{lot}.csv"
    if out.exists():
        return pd.read_csv(out)
    model = load_run(name)[0]
    device = next(model.parameters()).device
    if items is not None:
        df = postprocess_scores(model, items, device, variants)
    else:
        df = pd.concat([postprocess_scores(model, load_extra(rows.iloc[i:i + chunk], scale), device, variants)
                        for i in range(0, len(rows), chunk)], ignore_index=True)
    df.to_csv(out, index=False)
    return df


def traits(df):
    """Traits par image avec racine, à partir du tableau image par image : longueur (pixels de squelette, résolution
    d'origine) et diamètre moyen (surface / longueur), sur la référence et sur la prédiction."""
    r = df[df["has_root"]].copy()
    r["longueur réf."], r["longueur prédite"] = r["skel_ref"], r["skel_pred"]
    r["diamètre réf."] = (r["tp"] + r["fn"]) / r["skel_ref"]
    r["diamètre prédit"] = r["pred_px"] / r["skel_pred"].where(r["skel_pred"] > 0)
    return r


def trait_scores(df):
    """Erreur relative médiane par image et biais d'ensemble (somme prédite / somme de référence - 1) de la longueur
    et du diamètre moyen (pondéré par la longueur)."""
    t = traits(df)
    out = {"images avec racine": len(t)}
    for k in ("longueur", "diamètre"):
        ref, pred = t[f"{k} réf."], t[f"{k} prédite" if k == "longueur" else f"{k} prédit"]
        out[f"{k} : erreur médiane par image"] = ((pred - ref).abs() / ref).median()
    out["longueur : biais d'ensemble"] = t["longueur prédite"].sum() / t["longueur réf."].sum() - 1
    out["diamètre : biais d'ensemble"] = (t["pred_px"].sum() / t["skel_pred"].sum()) / (
        (t["tp"] + t["fn"]).sum() / t["skel_ref"].sum()) - 1
    return out


def largest_blobs(df, pred_dir):
    """Taille (px) de la plus grosse tache prédite sur chaque image sans racine (0 si rien n'est prédit)."""
    sizes = []
    for name in df.loc[~df["has_root"], "name"]:
        pred = np.asarray(Image.open(pred_dir / f"PRED_{name}.png")) > 127
        labels, n = ndi.label(pred, structure=EIGHT)
        sizes.append(ndi.sum(pred, labels, range(1, n + 1)).max() if n else 0)
    return np.array(sizes)


@torch.no_grad()
def timing(model, rows, device, scale=2, n=50):
    """Vitesse sur n images lues sur la clé : lecture et prétraitement, puis modèle seul (secondes par image)."""
    model.eval()
    t0 = time.perf_counter()
    items = load_extra(rows.head(n), scale, filled=False)
    t1 = time.perf_counter()
    for it in items:
        full_logits(model, it, device)
    t2 = time.perf_counter()
    return {"lecture et prétraitement (s / image)": (t1 - t0) / n, "modèle (s / image)": (t2 - t1) / n,
            "images par minute": 60 * n / (t2 - t0), "processeur": f"{torch.get_num_threads()} fils, {device}"}


def compare_runs(a, b, metric="dice", file="val_images.csv"):
    """Compare deux essais sur un val, à leur meilleure époque : moyennes du lot, puis `metric` image par image (b moins a).

    Pour l'erreur de longueur, plus bas est meilleur : « b meilleur » compte alors les images où l'écart est négatif.
    """
    tables = {n: pd.read_csv(RUNS_DIR / n / file) for n in (a, b)}
    summary = pd.DataFrame({n: summarize(t) for n, t in tables.items()}).T
    pa, pb = (t.set_index("name")[metric] for t in tables.values())
    delta = (pb - pa).dropna()
    sign = -1 if metric == "length_err" else 1
    paired = {"images": len(delta), "écart moyen": delta.mean(), "écart médian": delta.median(),
              "b meilleur": int((sign * delta > 0).sum()), "b moins bon": int((sign * delta < 0).sum()),
              "p (Wilcoxon)": wilcoxon(delta).pvalue}
    return summary, pd.Series(paired), delta


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="unet_effb0")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--scale", type=float, default=1)
    args = p.parse_args()
    train(load_lot("train", args.scale), load_lot("val", args.scale), args.name, args.epochs, args.patience)
