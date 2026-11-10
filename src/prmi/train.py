"""Entraînement du U-Net sur les masques corrigés (train), avec early stopping sur le val.

Perte : 0,5 x Dice + 0,5 x BCE. Le Dice est calculé carré par carré, sur les carrés qui contiennent de la racine, pour
qu'une grosse racine n'écrase pas les petites des autres carrés du batch ; les carrés sans racine sont laissés à la BCE.
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
- val_loss : même perte qu'à l'entraînement, sur le val.

Sortie : data/processed/runs/<nom>/ avec config.json (réglages, puis meilleure époque à la fin), history.csv,
best.pt (poids de la meilleure époque) et val_images.csv (scores image par image à la meilleure époque, pour comparer
les variantes par paires).

Usage (depuis la racine du projet) : PYTHONPATH=src python -m prmi.train --name unet_effb0
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from torch.utils.data import DataLoader

from prmi import dataset
from prmi.dataset import RootCrops, load_lot, pad_image
from prmi.inventory import ROOT
from prmi.model import build_model, count_used_parameters

RUNS_DIR = ROOT / "data" / "processed" / "runs"
LR_MIN = 1e-6
N_TRAIN_EVAL = 44   # autant que d'images avec racine dans le val
MIN_FP = 100        # pixels ; en dessous, pas de fausse détection (la plupart des morceaux de racine annotés dépassent 500 px)
EIGHT = np.ones((3, 3))     # connexité à 8 voisins


def dice_loss(logits, target, smooth=1.0):
    """1 - Dice, carré par carré, moyenné sur les carrés qui contiennent de la racine (0 s'il n'y en a aucun)."""
    p, y = torch.sigmoid(logits).flatten(1), target.flatten(1)
    has_root = y.sum(1) > 0
    if not has_root.any():
        return logits.sum() * 0
    dice = (2 * (p * y).sum(1) + smooth) / (p.sum(1) + y.sum(1) + smooth)
    return 1 - dice[has_root].mean()


def loss_fn(logits, target):
    return 0.5 * dice_loss(logits, target) + 0.5 * F.binary_cross_entropy_with_logits(logits, target)


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


def image_scores(pred, mask):
    """Scores d'une image, à partir de masques booléens. Dice et erreur de longueur : NaN si l'image n'a pas de racine."""
    skeleton = skeletonize(mask) if mask.any() else mask
    roots, found = count_roots(pred, mask, skeleton)
    scores = {"tp": int((pred & mask).sum()), "fp": int((pred & ~mask).sum()), "fn": int((~pred & mask).sum()),
              "roots": roots, "roots_found": found, "ghosts": count_ghosts(pred, mask), "pred_px": int(pred.sum())}
    if not mask.any():
        return {"dice": np.nan, "length_err": np.nan, **scores}
    length = int(skeleton.sum())
    return {"dice": 2 * scores["tp"] / (pred.sum() + mask.sum()),
            "length_err": abs(int(skeletonize(pred).sum()) - length) / length, **scores}


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
    tp, fp, fn = df["tp"].sum(), df["fp"].sum(), df["fn"].sum()
    return {"dice": df["dice"].mean(), "global_dice": 2 * tp / max(2 * tp + fp + fn, 1),
            "roots_found": df["roots_found"].sum() / max(df["roots"].sum(), 1), "ghosts": int(df["ghosts"].sum()),
            "length_err": df["length_err"].median(), "fp_empty": (df.loc[~df["has_root"], "pred_px"] >= MIN_FP).mean(),
            "val_loss": np.mean(losses)}, df


def train_eval_items(items, seed=0):
    """Images du train suivies à chaque époque : N_TRAIN_EVAL images avec racine, tirées une fois pour toutes."""
    rooted = [it for it in items if it["mask"].any()]
    return [rooted[i] for i in sorted(np.random.default_rng(seed).choice(len(rooted), N_TRAIN_EVAL, replace=False))]


def train(train_items, val_items, name="unet_effb0", epochs=100, patience=20, batch=8, lr=1e-3, encoder_lr=1e-4,
          weight_decay=1e-4, seed=0, augment=False):
    """Entraîne le modèle. Renvoie le modèle (poids de la meilleure époque), l'historique et la meilleure époque."""
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = RUNS_DIR / name
    out.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(RootCrops(train_items, seed, augment), batch_size=batch, shuffle=True, drop_last=True)
    train_eval = train_eval_items(train_items, seed)
    model = build_model().to(device)
    n_params = count_used_parameters(model)
    groups = [{"params": model.encoder.parameters(), "lr": encoder_lr},
              {"params": [*model.decoder.parameters(), *model.segmentation_head.parameters()], "lr": lr}]
    opt = torch.optim.AdamW(groups, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=LR_MIN)
    config = {"name": name, "epochs": epochs, "patience": patience, "batch": batch, "lr": lr, "encoder_lr": encoder_lr,
              "lr_min": LR_MIN, "scale": round(train_items[0]["mask_full"].shape[1] / train_items[0]["mask"].shape[1], 3),
              "weight_decay": weight_decay, "seed": seed, "augment": augment, "crop": dataset.CROP, "crops": dataset.CROPS,
              "forced": dataset.FORCED, "encoder": "efficientnet-b0", "encoder_weights": "imagenet", "parameters": n_params,
              "n_train": len(train_items), "n_val": len(val_items), "stop_metric": "global_dice", "best_epoch": None,
              "best_score": None}
    (out / "config.json").write_text(json.dumps(config, indent=1))
    print(f"{device}, {n_params:,} paramètres utilisés, {len(train_items)} images train, {len(val_items)} images val, "
          f"{len(loader)} batchs par époque", flush=True)

    best, best_epoch, wait, history = -1.0, 0, 0, []
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        lr_encoder, lr_decoder = sched.get_last_lr()
        model.train()
        total = 0.0
        for x, y in loader:
            opt.zero_grad()
            loss = loss_fn(model(x.to(device)), y.to(device))
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


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="unet_effb0")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--scale", type=float, default=1)
    args = p.parse_args()
    train(load_lot("train", args.scale), load_lot("val", args.scale), args.name, args.epochs, args.patience)
