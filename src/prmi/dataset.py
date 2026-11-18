"""Jeux d'entraînement et de validation : images du lot QuPath et masques corrigés.

Résolution de travail : image et masque réduits d'un facteur `scale` après le prétraitement (1 : pleine résolution). Le
masque d'origine reste disponible (`mask_full`) : les scores se calculent toujours à la résolution d'origine.

Entraînement (`RootCrops`) : CROPS carrés de CROP pixels par image et par époque, à des positions retirées à chaque fois.
- Image sans racine : carré placé n'importe où.
- Image avec racine : une fois sur deux n'importe où, une fois sur deux forcé sur une racine (morceau de racine tiré au
  hasard, chaque morceau à égalité, puis pixel de ce morceau, placé à une position aléatoire dans le carré).
Un carré tiré à cheval sur le bord est repoussé dans l'image (aucun pixel inventé), ce qui fait voir les bords plus souvent.

Deux sources d'images d'entraînement :
- `load_lot` : images du lot QuPath (train ou val), masques corrigés ou d'origine ;
- `load_extra` : images du train officiel (150 DPI) hors lot QuPath (tirées au hasard par `draw_extra`, ou toutes avec
  `population`), lues sur la clé, avec les masques d'origine comblés (`prmi.gaps`, enregistrés une fois pour toutes dans
  FILLED_DIR) ou bruts. Sert aussi au grand val, tiré dans le val officiel. Option `compact` : image en entiers 8 bits et
  pas de masque pleine résolution, pour faire tenir tout le train en mémoire (images d'entraînement seulement). Avec `crops_per_epoch`, chaque carré vient d'une image tirée au hasard parmi toutes.

Validation (`pad_image`) : image entière, complétée en bas pour avoir une hauteur multiple de 16.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import ndimage as ndi
from torch.utils.data import Dataset

from prmi.arrays import pad_image, shrink, to_display, to_tensors  # noqa: F401 (réexportées)
from prmi.image_stats import image_type
from prmi.inventory import RAW, ROOT
from prmi.preprocess import load_flatfields, preprocess_image, preprocess_mask
from prmi.qupath_io import QUPATH_DIR

MASKS_DIR = ROOT / "data" / "processed" / "masks_corriges"
STATS = ROOT / "data" / "interim" / "image_stats.parquet"
EXTRA = ROOT / "data" / "processed" / "train_extra.csv"
VAL_LARGE = ROOT / "data" / "processed" / "val_large.csv"
FILLED_DIR = ROOT / "data" / "processed" / "masks_combles"     # masques d'origine comblés, un dossier par split
CROP = 256
CROPS = 2
FORCED = 0.5




def make_item(name, group, kind, img, mask, flat, scale, masks, compact=False):
    """Prétraite une image et son masque (résolution d'origine, carte de fond `flat` du groupe), puis les réduit.

    compact : image en entiers 8 bits (convertie en flottants au moment du carré), masque compressé (8 pixels par octet,
    décompressé au moment du carré, voir `work_mask`) et pas de masque pleine résolution.
    """
    dpi = int(group.split()[0])
    image, full = preprocess_image(img, flat, dpi), preprocess_mask(mask, dpi)
    small = full
    if scale != 1:
        image = np.stack([shrink(image[..., c], scale) for c in range(3)], axis=2)
        small = shrink(full.astype(np.float32), scale) > 0.5
    if compact:
        return {"name": name, "group": group, "type": kind, "image": np.round(np.clip(image, 0, 1) * 255).astype(np.uint8),
                "mask": None, "mask_bits": np.packbits(small, axis=1), "has_root": bool(small.any()), "mask_full": None,
                "masks": masks}
    return {"name": name, "group": group, "type": kind, "image": image, "mask": small, "mask_full": full, "masks": masks}


def load_lot(lot, scale=1, masks="corriges"):
    """Images prétraitées et masques d'un lot, en mémoire : une liste de dictionnaires, un par image.

    masks : "corriges" (masques corrigés dans QuPath) ou "origine" (masques PRMI d'origine, copiés à l'export QuPath).
    """
    assert masks in ("corriges", "origine"), masks
    flats = load_flatfields()
    items = []
    for _, r in pd.read_csv(QUPATH_DIR / lot / "lot.csv").iterrows():
        img = np.asarray(Image.open(QUPATH_DIR / lot / r["name"] / "image.jpg").convert("RGB"))
        path = MASKS_DIR / lot / f"GT_{r['name']}.png" if masks == "corriges" else QUPATH_DIR / lot / r["name"] / "mask.png"
        mask = np.asarray(Image.open(path).convert("L")) > 127
        items.append(make_item(r["name"], r["group"], r["type"], img, mask, flats[r["group"]], scale, masks))
    return items


def _official(split, dpi):
    """Images d'un split officiel à un DPI donné, hors lot QuPath du même split, avec leur type."""
    stats = pd.read_parquet(STATS)
    stats["type"] = image_type(stats)
    stats["name"] = stats["image_path"].map(lambda p: Path(p).stem)
    taken = set(pd.read_csv(QUPATH_DIR / split / "lot.csv")["name"])
    pop = stats[(stats["split"] == split) & (stats["dpi"] == dpi) & ~stats["name"].isin(taken)]
    return pop[["name", "group", "type", "has_root", "tube_id", "position", "date", "image_path",
                "mask_path"]].reset_index(drop=True)


def population(split="train"):
    """Images d'un split officiel (train ou val, jamais test), 150 DPI, hors lot QuPath du même split, avec leur type."""
    assert split in ("train", "val"), "le test ne sert qu'à l'évaluation finale"
    return _official(split, 150)


def test_rows(dpi=150):
    """Grand test : toutes les images du test officiel à ce DPI (150 : hors des 50 images corrigées). Évaluation finale
    seulement, sur les masques d'origine comblés."""
    return _official("test", dpi)


def val_learning():
    """Val d'apprentissage du classifieur des taches : images du val officiel (150 DPI) hors val corrigé et hors grand
    val (`VAL_LARGE`)."""
    taken = set(pd.read_csv(VAL_LARGE)["name"])
    rows = population("val")
    return rows[~rows["name"].isin(taken)].reset_index(drop=True)


def draw_extra(n, seed=0, path=EXTRA, split="train"):
    """Tirage au hasard de n images de `population(split)` (la part d'images vides suit la part réelle). Le tirage est
    enregistré dans `path` et relu ensuite, jamais refait."""
    if path.exists():
        drawn = pd.read_csv(path)
        assert len(drawn) == n, f"{path.name} contient {len(drawn)} images, pas {n} : le supprimer pour refaire le tirage"
        return drawn
    drawn = population(split).sample(n, random_state=seed)
    drawn.to_csv(path, index=False)
    return drawn


def filled_path(row):
    """Chemin du masque comblé d'une image (dossier du split d'origine)."""
    return FILLED_DIR / Path(row["image_path"]).parts[0] / f"GT_{row['name']}.png"


def _read_extra(row, raw, filled):
    img = np.asarray(Image.open(raw / row["image_path"]).convert("RGB"))
    if not row["has_root"]:
        return img, np.zeros(img.shape[:2], dtype=bool)
    if filled:
        path = filled_path(row)
        if not path.exists():
            raise FileNotFoundError(f"{path} absent : lancer PYTHONPATH=src python -m prmi.gaps depuis la racine du projet")
        return img, np.asarray(Image.open(path).convert("L")) > 127
    return img, np.asarray(Image.open(raw / row["mask_path"]))[..., 0] > 0


def load_extra(rows, scale=1, raw=RAW, filled=True, compact=False, workers=8):
    """Images lues sur la clé, masques d'origine comblés (`filled`) ou bruts, vides si has_root = 0 (sans lire de fichier).
    Lecture et prétraitement en parallèle, ordre des lignes conservé."""
    flats = load_flatfields()

    def one(r):
        img, mask = _read_extra(r, raw, filled)
        return make_item(r["name"], r["group"], r["type"], img, mask, flats[r["group"]], scale,
                         "comblés" if filled else "origine", compact)

    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, (r for _, r in rows.iterrows())))


def work_mask(item, rows=slice(None)):
    """Masque de travail d'une image (lignes `rows`), qu'il soit stocké tel quel ou compressé (images compactes)."""
    if item["mask"] is not None:
        return item["mask"][rows]
    return np.unpackbits(item["mask_bits"][rows], axis=1, count=item["image"].shape[1]).astype(bool)








# Augmentations photométriques (image seule, le masque n'est pas touché) : probabilité et plage de tirage.
GAMMA, GAMMA_RANGE = 0.5, (0.7, 1.4)
OVEREXPOSE, OVEREXPOSE_RANGE = 0.25, (0, 0.25)
NOISE, NOISE_RANGE = 0.25, (0.01, 0.05)
BLUR, BLUR_RANGE = 0.25, (0.5, 1.2)


def orient(image, mask, flip, k):
    """Une des 8 orientations du carré : miroir horizontal (ou non), puis k quarts de tour. Image et masque ensemble."""
    if flip:
        image, mask = image[:, ::-1], mask[:, ::-1]
    return np.rot90(image, k), np.rot90(mask, k)


def gamma(image, g):
    return image ** g


def overexpose(image, shift):
    return image + shift


def add_noise(image, sigma, rng):
    return image + rng.normal(0, sigma, image.shape).astype(np.float32)


def blur(image, sigma):
    return ndi.gaussian_filter(image, (sigma, sigma, 0))


# Occlusion : morceau de sol de la même image (endroit sans racine), collé sur l'axe d'une racine avec un bord adouci. Le
# masque reste entier : la racine continue sous la terre. Chaque demi-axe de l'ellipse est tiré entre OCCLUDE_RANGE fois
# le diamètre local d de la racine (px, résolution de travail), soit une à trois fois sa largeur, avec un plancher
# (OCCLUDE_MIN) pour les racines fines et un plafond (OCCLUDE_MAX) pour ne pas cacher de longs tronçons.
OCCLUDE_RANGE = (0.6, 1.5)
OCCLUDE_MIN = (4, 8)    # px : bornes basse et haute minimales des demi-axes
OCCLUDE_MAX = 30        # px : demi-axe maximal
OCCLUDE_SOFT = 1.5      # px : flou du bord de l'ellipse


def ellipse_alpha(a, b, angle, soft=OCCLUDE_SOFT):
    """Poids d'une ellipse (demi-axes a, b, angle en radians) de bord adouci, sur un carré centré de côté impair."""
    r = int(np.ceil(max(a, b) + 2 * soft))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
    u, v = xx * np.cos(angle) + yy * np.sin(angle), -xx * np.sin(angle) + yy * np.cos(angle)
    inside = ((u / a) ** 2 + (v / b) ** 2 <= 1).astype(np.float32)
    return ndi.gaussian_filter(inside, soft)


def occlusion_axes(d):
    """Bornes des demi-axes de l'ellipse d'occlusion (px) pour une racine de diamètre local d (px)."""
    hi = min(max(OCCLUDE_MIN[1], OCCLUDE_RANGE[1] * d), OCCLUDE_MAX)
    return min(max(OCCLUDE_MIN[0], OCCLUDE_RANGE[0] * d), hi), hi


def occlude(image, mask, source, source_mask, rng, tries=10):
    """Colle sur une racine du carré (image, mask) un morceau de sol pris dans `source` (image entière, flottants ou
    entiers 8 bits) là où `source_mask` est vide. Le centre est tiré sur l'axe des racines (maximum local de la distance
    au bord), avec une chance proportionnelle au diamètre ; la taille suit le diamètre à cet endroit (`occlusion_axes`). Renvoie l'image (le masque ne change pas),
    inchangée si aucun morceau de sol n'est trouvé."""
    dist = ndi.distance_transform_edt(mask)
    ys, xs = np.nonzero(mask & (dist >= ndi.maximum_filter(dist, 3)))
    weight = dist[ys, xs]       # tirage pondéré par le diamètre : chaque racine selon sa surface, comme tout pixel
    y, x = ys[(i := rng.choice(len(ys), p=weight / weight.sum()))], xs[i]
    lo, hi = occlusion_axes(2 * dist[y, x] - 1)     # diamètre : 1 px pour une racine de 1 px de large
    alpha = ellipse_alpha(rng.uniform(lo, hi), rng.uniform(lo, hi), rng.uniform(0, np.pi))
    r = alpha.shape[0] // 2
    H, W = source.shape[:2]
    for _ in range(tries):
        sy, sx = rng.integers(r, H - r), rng.integers(r, W - r)
        if not source_mask[sy - r:sy + r + 1, sx - r:sx + r + 1].any():
            break
    else:
        return image
    patch = source[sy - r:sy + r + 1, sx - r:sx + r + 1]
    patch = patch.astype(np.float32) / 255 if patch.dtype == np.uint8 else patch
    h, w = mask.shape
    y0, y1, x0, x1 = max(y - r, 0), min(y + r + 1, h), max(x - r, 0), min(x + r + 1, w)
    a = alpha[y0 - (y - r):y1 - (y - r), x0 - (x - r):x1 - (x - r), None]
    out = image.copy()
    out[y0:y1, x0:x1] = a * patch[y0 - (y - r):y1 - (y - r), x0 - (x - r):x1 - (x - r)] + (1 - a) * out[y0:y1, x0:x1]
    return out


def augment(image, mask, rng):
    """Augmentation à la volée : les 8 orientations du carré (aucun pixel inventé), gamma, surexposition, bruit et flou.

    Pas de dominante de couleur : la normalisation par canal l'annule.
    """
    image, mask = orient(image, mask, rng.random() < 0.5, rng.integers(4))
    if rng.random() < GAMMA:
        image = gamma(image, rng.uniform(*GAMMA_RANGE))
    if rng.random() < OVEREXPOSE:
        image = overexpose(image, rng.uniform(*OVEREXPOSE_RANGE))
    if rng.random() < NOISE:
        image = add_noise(image, rng.uniform(*NOISE_RANGE), rng)
    if rng.random() < BLUR:
        image = blur(image, rng.uniform(*BLUR_RANGE))
    return np.clip(image, 0, 1).astype(np.float32), mask


class RootCrops(Dataset):
    """Carrés d'entraînement. Chaque accès tire un nouveau carré (à utiliser avec num_workers=0).

    Option `occlusion` (avec `augment`) : probabilité de coller un ou deux morceaux de sol sur les racines d'un carré
    (`occlude`).
    Sans `crops_per_epoch` : CROPS carrés par image et par époque. Avec : `crops_per_epoch` carrés par époque, chacun
    pris dans une image tirée au hasard parmi toutes (pour garder le même temps de calcul quand le nombre d'images grandit).
    """

    def __init__(self, items, seed=0, augment=False, crops_per_epoch=None, occlusion=0.0):
        assert min(items[0]["image"].shape[:2]) >= CROP, "image plus petite qu'un carré"
        self.items = items
        self.augment = augment
        self.crops_per_epoch = crops_per_epoch
        self.occlusion = occlusion      # probabilité d'occlusion d'un carré avec racine (avec augment seulement)
        self.rng = np.random.default_rng(seed)
        self.pieces = [self._pieces(work_mask(it)) if it.get("has_root", True) else [] for it in items]  # pixels par morceau

    @staticmethod
    def _pieces(mask):
        labels, n = ndi.label(mask, structure=np.ones((3, 3)))
        return [np.argwhere(labels == k).astype(np.int32) for k in range(1, n + 1)]

    def __len__(self):
        return self.crops_per_epoch or CROPS * len(self.items)

    def _corner(self, i, h, w):
        """Coin haut gauche du carré, toujours dans l'image : le centre est tiré dans toute l'image, puis le carré est repoussé."""
        r = self.rng
        if self.pieces[i] and r.random() < FORCED:
            piece = self.pieces[i][r.integers(len(self.pieces[i]))]
            y, x = piece[r.integers(len(piece))]
            y0, x0 = y - r.integers(CROP), x - r.integers(CROP)
        else:
            y0, x0 = r.integers(h) - CROP // 2, r.integers(w) - CROP // 2
        return np.clip(y0, 0, h - CROP), np.clip(x0, 0, w - CROP)

    def __getitem__(self, idx):
        i = self.rng.integers(len(self.items)) if self.crops_per_epoch else idx % len(self.items)
        it = self.items[i]
        h, w = it["image"].shape[:2]
        y0, x0 = self._corner(i, h, w)
        image, mask = it["image"][y0:y0 + CROP, x0:x0 + CROP], work_mask(it, slice(y0, y0 + CROP))[:, x0:x0 + CROP]
        if image.dtype == np.uint8:     # image compacte
            image = image.astype(np.float32) / 255
        if self.augment and self.occlusion and mask.any() and self.rng.random() < self.occlusion:
            for _ in range(self.rng.integers(1, 3)):      # un ou deux morceaux de sol
                image = occlude(image, mask, it["image"], work_mask(it), self.rng)
        if self.augment:
            image, mask = augment(image, mask, self.rng)
        return to_tensors(image, mask)
