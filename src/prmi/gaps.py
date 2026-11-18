"""Comblement des fentes des masques PRMI d'origine (tracé en chaîne de rectangles).

Deux défauts, tous deux comblés par les ponts d'une fermeture morphologique (disque de rayon RADIUS) :
- fente : coupure étroite entre deux morceaux distincts du masque (souvent une racine latérale séparée de sa racine
  principale, ou deux rectangles consécutifs disjoints) ;
- pincement : deux rectangles consécutifs qui ne se touchent que par un ou deux pixels, en laissant un trait vide en
  travers de la racine. Une ouverture de rayon PINCH coupe ces contacts ; on cherche alors les ponts entre les morceaux
  obtenus.

Un pont est une zone ajoutée par la fermeture qui touche au moins deux morceaux. Il n'est gardé que s'il touche
l'**extrémité** d'au moins un des morceaux (extrémité contre extrémité, ou extrémité contre bord) : deux bords longs
proches (racines parallèles) ne sont jamais reliés. Les autres ajouts de la fermeture (angles entre branches, creux des
bords) sont ignorés : la largeur des racines ne change pas.

Extrémité (`is_end`) : autour du point de contact, pixels du morceau à moins de END_WINDOW px et leur axe principal ; le
contact est une extrémité si presque tous ces pixels sont d'un seul côté du contact le long de cet axe.

Enregistrement (`save_filled`) : masques comblés écrits une fois pour toutes dans data/processed/masks_combles/<split>/
(un canal, 0 ou 255, comme les masques corrigés), pour les images avec racine du train et du val (150 DPI, hors lots
QuPath). Le test n'est traité qu'au moment de l'évaluation finale.

Usage (depuis la racine du projet) : PYTHONPATH=src python -m prmi.gaps

Validation (`bridge_table`, `skeleton_table`) : contre les masques corrigés à la main dans QuPath. Une fente comblée est
« confirmée » si les morceaux qu'elle relie forment un seul morceau du masque corrigé.
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from skimage.measure import regionprops
from skimage.morphology import closing, disk, opening, skeletonize

from prmi.dataset import MASKS_DIR, filled_path, population
from prmi.diagnostic import overlay_mask
from prmi.inventory import RAW
from prmi.qupath_io import QUPATH_DIR

RADIUS = 4          # px, résolution d'origine : comble les écarts jusqu'à environ 8 px
PINCH = 1           # px : l'ouverture de rayon 1 coupe les contacts de 1 à 2 px
END_WINDOW = 25     # px : voisinage du contact pour décider extrémité ou bord
EIGHT = np.ones((3, 3), dtype=bool)
SHIFTS = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]
BRIDGE_COLOR = (255, 140, 0)    # orange : pixels ajoutés par le comblement


def _touching(added, pieces):
    """Contacts (8 voisins) entre zones ajoutées et morceaux : lignes (zone, morceau, y, x du pixel du morceau)."""
    h, w = added.shape
    out = []
    for dy, dx in SHIFTS:
        a = added[max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)]
        p = pieces[max(-dy, 0):h + min(-dy, 0), max(-dx, 0):w + min(-dx, 0)]
        both = (a > 0) & (p > 0)
        ys, xs = np.nonzero(both)
        out.append(np.stack([a[both], p[both], ys + max(-dy, 0), xs + max(-dx, 0)], axis=1))
    return np.concatenate(out)


def is_end(pieces, k, cy, cx, window=END_WINDOW, tol=3, frac=0.1):
    """Le contact (cy, cx) est-il à une extrémité du morceau k ? Extrémité si moins de `frac` des pixels du morceau
    proches du contact sont d'un côté du contact (au-delà de `tol` px) le long de leur axe principal."""
    h, w = pieces.shape
    y0, x0 = max(cy - window, 0), max(cx - window, 0)
    ys, xs = np.nonzero(pieces[y0:min(cy + window + 1, h), x0:min(cx + window + 1, w)] == k)
    ys, xs = ys + y0 - cy, xs + x0 - cx
    near = ys ** 2 + xs ** 2 <= window ** 2
    pts = np.stack([ys[near], xs[near]], axis=1).astype(float)
    if len(pts) < 10:       # tout petit morceau : extrémité
        return True
    axis = np.linalg.svd(pts - pts.mean(0), full_matrices=False)[2][0]
    t = pts @ axis
    return min((t < -tol).mean(), (t > tol).mean()) < frac


def find_bridges(mask, pieces, radius=RADIUS):
    """Ponts de la fermeture entre les morceaux `pieces` (image étiquetée). Renvoie l'image des zones ajoutées
    (étiquetées) et un dictionnaire par pont : morceaux reliés, nombre d'extrémités, gardé ou non, taille, position."""
    added = ndi.label(closing(mask, disk(radius)) & ~mask, structure=EIGHT)[0]
    contacts = _touching(added, pieces)
    bridges = []
    if not len(contacts):
        return added, bridges
    props = {r.label: r for r in regionprops(added)}
    for a in np.unique(contacts[:, 0]):
        rows = contacts[contacts[:, 0] == a]
        ks = np.unique(rows[:, 1])
        if len(ks) < 2:
            continue
        ends = [is_end(pieces, k, int(rows[rows[:, 1] == k, 2].mean()), int(rows[rows[:, 1] == k, 3].mean()))
                for k in ks]
        bridges.append({"label": int(a), "pieces": {int(k) for k in ks}, "ends": int(sum(ends)), "kept": any(ends),
                        "px": int(props[a].area), "y": int(props[a].centroid[0]), "x": int(props[a].centroid[1])})
    return added, bridges


def fill_details(mask, radius=RADIUS, pinch=PINCH):
    """Comblement en deux passes (fentes, puis pincements). Renvoie le masque comblé et la liste des ponts examinés,
    avec leur sorte (« fente » ou « pincement »)."""
    out, found = mask.copy(), []
    passes = [("fente", lambda m: ndi.label(m, structure=EIGHT)[0])]
    if pinch:
        passes.append(("pincement", lambda m: ndi.label(opening(m, disk(pinch)), structure=EIGHT)[0]))
    for kind, label in passes:
        added, bridges = find_bridges(out, label(out), radius)
        kept = [b["label"] for b in bridges if b["kept"]]
        if kept:
            out = out | np.isin(added, kept)
        found += [{**b, "kind": kind} for b in bridges]
    return out, found


def fill_gaps(mask, radius=RADIUS, pinch=PINCH):
    """Masque avec ses fentes et ses pincements comblés (ne retire jamais un pixel)."""
    return fill_details(mask, radius, pinch)[0]


def load_pairs(lots=("train", "val")):
    """Masques d'origine et corrigés des lots QuPath, images avec racine dans les deux versions."""
    pairs = []
    for lot in lots:
        for _, r in pd.read_csv(QUPATH_DIR / lot / "lot.csv").iterrows():
            o = np.asarray(Image.open(QUPATH_DIR / lot / r["name"] / "mask.png").convert("L")) > 127
            c = np.asarray(Image.open(MASKS_DIR / lot / f"GT_{r['name']}.png").convert("L")) > 127
            if o.any() and c.any():
                pairs.append({"lot": lot, "name": r["name"], "original": o, "corrected": c})
    return pairs


def _corrected_piece(pieces, corrected_pieces, k):
    """Morceau du masque corrigé qui recouvre le plus le morceau k du masque d'origine (0 : morceau supprimé)."""
    v = corrected_pieces[pieces == k]
    v = v[v > 0]
    return int(np.bincount(v).argmax()) if len(v) else 0


def bridge_table(pairs, radius=RADIUS, pinch=PINCH):
    """Un pont examiné par ligne (fentes et pincements). Pour les fentes : confirmée ou non par la correction manuelle."""
    rows = []
    for pr in pairs:
        _, found = fill_details(pr["original"], radius, pinch)
        pieces = ndi.label(pr["original"], structure=EIGHT)[0]
        corrected_pieces = ndi.label(pr["corrected"], structure=EIGHT)[0]
        for b in found:
            row = {"lot": pr["lot"], "name": pr["name"], **{k: b[k] for k in ("kind", "kept", "ends", "px", "y", "x")}}
            if b["kind"] == "fente":
                ids = {_corrected_piece(pieces, corrected_pieces, k) for k in b["pieces"]}
                row["confirmed"], row["piece_removed"] = len(ids) == 1 and 0 not in ids, 0 in ids
            rows.append(row)
    df = pd.DataFrame(rows)
    return df.astype({"confirmed": "boolean", "piece_removed": "boolean"})   # vides pour les pincements


def manual_merges(pairs):
    """Réunions faites à la main : pour chaque morceau d'origine réuni à un autre dans le masque corrigé, distance (px)
    au plus proche des morceaux avec lesquels il a été réuni."""
    rows = []
    for pr in pairs:
        pieces, n = ndi.label(pr["original"], structure=EIGHT)
        corrected_pieces = ndi.label(pr["corrected"], structure=EIGHT)[0]
        target = pd.Series({k: _corrected_piece(pieces, corrected_pieces, k) for k in range(1, n + 1)})
        target = target[target > 0]
        for _, group in target.groupby(target):
            if len(group) < 2:
                continue
            for k in group.index:
                others = np.isin(pieces, [x for x in group.index if x != k])
                rows.append({"name": pr["name"], "distance": float(ndi.distance_transform_edt(~others)[pieces == k].min())})
    return pd.DataFrame(rows)


def skeleton_ends(skeleton):
    """Nombre de bouts d'un squelette (pixels à un seul voisin)."""
    neighbours = ndi.convolve(skeleton.astype(int), np.ones((3, 3), dtype=int), mode="constant") - 1
    return int((skeleton & (neighbours == 1)).sum())


def skeleton_table(pairs, variants):
    """Longueur du squelette (px) et nombre de bouts, par image, pour chaque variante {nom: fonction(masque d'origine)},
    plus le masque corrigé (référence)."""
    rows = []
    for pr in pairs:
        row = {"name": pr["name"]}
        masks = {**{k: f(pr["original"]) for k, f in variants.items()}, "corrigé": pr["corrected"]}
        for k, m in masks.items():
            sk = skeletonize(m)
            row[f"longueur {k}"], row[f"bouts {k}"] = int(sk.sum()), skeleton_ends(sk)
        rows.append(row)
    return pd.DataFrame(rows)


def compare_settings(pairs, settings):
    """Pour chaque réglage {nom: (rayon, pincement)} : morceaux restants, pixels ajoutés et leur part dans le masque
    corrigé, fentes comblées et confirmées, pincements bouchés."""
    rows = {}
    n_pieces = lambda m: ndi.label(m, structure=EIGHT)[1]
    for name, (radius, pinch) in settings.items():
        filled = [fill_gaps(p["original"], radius, pinch) for p in pairs]
        added = [f & ~p["original"] for f, p in zip(filled, pairs)]
        table = bridge_table(pairs, radius, pinch)
        slits = table[(table["kind"] == "fente") & table["kept"]]
        rows[name] = {"morceaux": sum(n_pieces(f) for f in filled), "px ajoutés": int(sum(a.sum() for a in added)),
                      "part dans le corrigé": sum((a & p["corrected"]).sum() for a, p in zip(added, pairs))
                      / max(sum(a.sum() for a in added), 1),
                      "fentes comblées": len(slits), "dont confirmées": int(slits["confirmed"].sum()),
                      "pincements bouchés": int(((table["kind"] == "pincement") & table["kept"]).sum())}
    out = pd.DataFrame(rows).T
    out.loc["origine", "morceaux"] = sum(n_pieces(p["original"]) for p in pairs)
    out.loc["corrigé", "morceaux"] = sum(n_pieces(p["corrected"]) for p in pairs)
    return out


def bridge_views(pairs, table, half=50, zoom=2):
    """Pour chaque pont du tableau : vignette du masque d'origine (bleu) avec les ajouts du comblement (orange), et
    vignette du masque corrigé, centrées sur le pont et agrandies. Renvoie les images et leurs titres, deux par pont."""
    by_name = {p["name"]: p for p in pairs}
    images, titles = [], []
    big = lambda a: np.kron(a, np.ones((zoom, zoom, 1), dtype=np.uint8))
    for _, b in table.iterrows():
        pr = by_name[b["name"]]
        img = np.asarray(Image.open(QUPATH_DIR / pr["lot"] / b["name"] / "image.jpg").convert("RGB"))
        h, w = pr["original"].shape
        y0, x0 = np.clip(b["y"] - half, 0, h - 2 * half), np.clip(b["x"] - half, 0, w - 2 * half)
        win = (slice(y0, y0 + 2 * half), slice(x0, x0 + 2 * half))
        added = fill_gaps(pr["original"]) & ~pr["original"]
        left = overlay_mask(overlay_mask(img, pr["original"]), added, color=BRIDGE_COLOR, alpha=0.9)[win]
        right = overlay_mask(img, pr["corrected"])[win]
        if b["kind"] == "fente":
            verdict = "confirmée" if b["confirmed"] else ("morceau supprimé" if b["piece_removed"] else "non réunie à la main")
        else:
            verdict = "pincement"
        images += [big(left), big(right)]
        titles += [f"{b['name'][7:27]} | {b['kind']} {b['px']} px {'gardée' if b['kept'] else 'refusée'}",
                   f"corrigé | {verdict}"]
    return images, titles


def count_bridges(rows, raw=RAW, radius=RADIUS, pinch=PINCH):
    """Fentes et pincements comblés par image, pour des lignes de l'inventaire (masques d'origine lus sur la clé, images
    avec racine)."""
    out = []
    for _, r in rows[rows["has_root"]].iterrows():
        mask = np.asarray(Image.open(raw / r["mask_path"]))[..., 0] > 0
        _, found = fill_details(mask, radius, pinch)
        kept = [b for b in found if b["kept"]]
        out.append({"name": r["name"], "fentes": sum(b["kind"] == "fente" for b in kept),
                    "pincements": sum(b["kind"] == "pincement" for b in kept)})
    return pd.DataFrame(out)


def save_filled(rows, raw=RAW, workers=8):
    """Écrit le masque comblé de chaque image avec racine (déjà écrit : ignoré). Renvoie le nombre de masques écrits."""
    todo = [r for _, r in rows[rows["has_root"]].iterrows() if not filled_path(r).exists()]

    def one(r):
        mask = np.asarray(Image.open(raw / r["mask_path"]))[..., 0] > 0
        path = filled_path(r)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(fill_gaps(mask).astype(np.uint8) * 255).save(path, optimize=True)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, todo))
    return len(todo)


if __name__ == "__main__":
    import time
    for split in ("train", "val"):
        t = time.time()
        rows = population(split)
        n = save_filled(rows)
        print(f"{split} : {int(rows['has_root'].sum())} images avec racine, {n} masques écrits en {time.time() - t:.0f} s")
