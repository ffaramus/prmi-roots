"""Prétraitement des images d'arachide. Ne contient que les étapes validées dans les notebooks ;
le module se remplit au fur et à mesure.

Chaîne (`preprocess_image`) :
1. recadrage du 120 DPI : retrait des bandes de remplissage (mêmes indices pour les masques, `preprocess_mask`) ;
2. correction du fond : division par l'image moyenne lissée du groupe (vignettage, dégradés) ;
3. normalisation par image : étirement par canal entre les centiles 1 et 99, gain plafonné à ×3.

Les cartes de fond se calculent sur le split train seul (`python -m prmi.preprocess`).
"""
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

FLATFIELDS = Path(__file__).resolve().parents[2] / "data" / "processed" / "flatfields.npz"

# 120 DPI (640x480) : colonnes 0 à 15 blanches, 16 à 18 noires ; lignes 0 à 3 blanches, 472 à 479 noires.
# Les colonnes 19 à 23 sont aussi retirées : trop sombres, bruitées et teintées après correction du fond.
CROP_120 = (slice(4, 472), slice(24, 640))
FULL = (slice(None), slice(None))


def crop_for(dpi):
    return CROP_120 if dpi == 120 else FULL


def flatfield(mean_image, crop, sigma=4, max_gain=8):
    """Carte de fond (1 = niveau du centre) : image moyenne du groupe, recadrée et lissée.

    Lissage faible (la moyenne de milliers d'images est déjà lisse) : un lissage large lisserait aussi le
    coude du dégradé et surcorrigerait de 5 à 10 %. Le gain appliqué (1 / carte) est plafonné à max_gain
    pour ne pas gonfler le bruit sur les toutes premières colonnes.
    """
    m = ndi.gaussian_filter(mean_image[crop].astype(float), sigma, mode="nearest")
    h, w = m.shape
    m /= m[h // 4: 3 * h // 4, w // 4: 3 * w // 4].mean()
    return np.maximum(m, 1 / max_gain)


def preprocess(img, flat, crop):
    """Recadre puis divise par la carte de fond. img : (H, W) ou (H, W, 3)."""
    out = img[crop].astype(float)
    out = out / (flat if out.ndim == 2 else flat[..., None])
    return np.clip(out, 0, 255).astype(np.uint8)


def normalize(img, max_gain=3):
    """Étirement par canal entre les centiles 1 et 99 ; max_gain=None : sans plafond.

    Gain = 1 / (centile 99 - centile 1). Si la plage est plus étroite que 1 / max_gain, l'étirement est limité
    à max_gain, centré sur le milieu de la plage. Renvoie l'image (0 à 1) et l'étendue utilisée par canal.
    """
    lo, hi = np.percentile(img, [1, 99], axis=(0, 1))
    span = hi - lo if max_gain is None else np.maximum(hi - lo, 1 / max_gain)
    return np.clip((img - (lo + hi) / 2) / np.maximum(span, 1e-3) + 0.5, 0, 1), span


def preprocess_image(img, flat, dpi):
    """Chaîne complète. img : RGB uint8 brut ; flat : carte de fond du groupe. Renvoie un float32 dans [0, 1]."""
    fond = preprocess(img, flat, crop_for(dpi)).astype(np.float32) / 255
    return normalize(fond)[0].astype(np.float32)


def preprocess_mask(mask, dpi):
    """Même recadrage que l'image."""
    return mask[crop_for(dpi)]


def save_flatfields(path=FLATFIELDS):
    """Cartes de fond par groupe, à partir des images moyennes du train."""
    from prmi.image_stats import load_mean_images      # outils d'exploration : inutiles à l'inférence
    maps = {g: flatfield(m, crop_for(int(g.split()[0]))).astype(np.float32) for g, m in load_mean_images().items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **maps)
    return maps


def load_flatfields(path=FLATFIELDS):
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


if __name__ == "__main__":
    maps = save_flatfields()
    print({g: m.shape for g, m in maps.items()}, "->", FLATFIELDS)
