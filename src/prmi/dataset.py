"""Jeux d'entraînement et de validation : images du lot QuPath et masques corrigés.

Résolution de travail : image et masque réduits d'un facteur `scale` après le prétraitement (1 : pleine résolution). Le
masque d'origine reste disponible (`mask_full`) : les scores se calculent toujours à la résolution d'origine.

Entraînement (`RootCrops`) : CROPS carrés de CROP pixels par image et par époque, à des positions retirées à chaque fois.
- Image sans racine : carré placé n'importe où.
- Image avec racine : une fois sur deux n'importe où, une fois sur deux forcé sur une racine (morceau de racine tiré au
  hasard, chaque morceau à égalité, puis pixel de ce morceau, placé à une position aléatoire dans le carré).
Un carré tiré à cheval sur le bord est repoussé dans l'image (aucun pixel inventé), ce qui fait voir les bords plus souvent.

Validation (`pad_image`) : image entière, complétée en bas pour avoir une hauteur multiple de 16.
"""
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import ndimage as ndi
from torch.utils.data import Dataset

from prmi.inventory import ROOT
from prmi.preprocess import load_flatfields, preprocess_image, preprocess_mask
from prmi.qupath_io import QUPATH_DIR

MASKS_DIR = ROOT / "data" / "processed" / "masks_corriges"
CROP = 256
CROPS = 2
FORCED = 0.5
MULTIPLE = 16
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def shrink(array, scale):
    """Réduit un tableau flottant (H, W) par moyenne de surface."""
    h, w = array.shape
    return np.asarray(Image.fromarray(array).resize((round(w / scale), round(h / scale)), Image.BOX))


def load_lot(lot, scale=1):
    """Images prétraitées et masques corrigés d'un lot, en mémoire : une liste de dictionnaires, un par image."""
    flats = load_flatfields()
    items = []
    for _, r in pd.read_csv(QUPATH_DIR / lot / "lot.csv").iterrows():
        dpi = int(r["group"].split()[0])
        img = np.asarray(Image.open(QUPATH_DIR / lot / r["name"] / "image.jpg").convert("RGB"))
        mask = np.asarray(Image.open(MASKS_DIR / lot / f"GT_{r['name']}.png").convert("L")) > 127
        image, full = preprocess_image(img, flats[r["group"]], dpi), preprocess_mask(mask, dpi)
        small = full
        if scale != 1:
            image = np.stack([shrink(image[..., c], scale) for c in range(3)], axis=2)
            small = shrink(full.astype(np.float32), scale) > 0.5
        items.append({"name": r["name"], "group": r["group"], "type": r["type"], "image": image, "mask": small,
                      "mask_full": full})
    return items


def to_tensors(image, mask):
    """Standardisation ImageNet, puis tenseurs (3, H, W) et (1, H, W)."""
    x = (image - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy()), torch.from_numpy(mask[None].astype(np.float32))


def to_display(x):
    """Inverse de la standardisation : tenseur (3, H, W) vers image RGB uint8, pour l'affichage."""
    img = x.numpy().transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def pad_image(image, mask):
    """Complète l'image en bas et à droite (miroir) pour que ses côtés soient multiples de 16. Le masque reste entier."""
    h, w = image.shape[:2]
    pad = ((0, -h % MULTIPLE), (0, -w % MULTIPLE), (0, 0))
    return to_tensors(np.pad(image, pad, mode="reflect"), mask)


def augment(image, mask, rng):
    """Augmentation à la volée : les 8 orientations du carré (aucun pixel inventé), gamma, surexposition, bruit et flou.

    Pas de dominante de couleur : la normalisation par canal l'annule.
    """
    if rng.random() < 0.5:
        image, mask = image[:, ::-1], mask[:, ::-1]
    k = rng.integers(4)
    image, mask = np.rot90(image, k), np.rot90(mask, k)
    if rng.random() < 0.5:
        image = image ** rng.uniform(0.7, 1.4)
    if rng.random() < 0.25:
        image = image + rng.uniform(0, 0.25)
    if rng.random() < 0.25:
        image = image + rng.normal(0, rng.uniform(0.01, 0.05), image.shape).astype(np.float32)
    if rng.random() < 0.25:
        image = ndi.gaussian_filter(image, (rng.uniform(0.5, 1.2),) * 2 + (0,))
    return np.clip(image, 0, 1).astype(np.float32), mask


class RootCrops(Dataset):
    """Carrés d'entraînement. Chaque accès tire un nouveau carré (à utiliser avec num_workers=0)."""

    def __init__(self, items, seed=0, augment=False):
        assert min(items[0]["mask"].shape) >= CROP, "image plus petite qu'un carré"
        self.items = items
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        self.pieces = [self._pieces(it["mask"]) for it in items]    # coordonnées des pixels de chaque morceau de racine

    @staticmethod
    def _pieces(mask):
        labels, n = ndi.label(mask, structure=np.ones((3, 3)))
        return [np.argwhere(labels == k) for k in range(1, n + 1)]

    def __len__(self):
        return CROPS * len(self.items)

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
        i = idx % len(self.items)
        it = self.items[i]
        h, w = it["mask"].shape
        y0, x0 = self._corner(i, h, w)
        image, mask = it["image"][y0:y0 + CROP, x0:x0 + CROP], it["mask"][y0:y0 + CROP, x0:x0 + CROP]
        if self.augment:
            image, mask = augment(image, mask, self.rng)
        return to_tensors(image, mask)
