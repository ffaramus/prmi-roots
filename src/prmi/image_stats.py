"""Caractéristiques des images d'arachide : un seul parcours de la clé, mis en cache.

Par image (zone utile, sans les bandes de remplissage du 120 DPI) : couleur, exposition,
éclairage, netteté, entrelacement ; pour les images avec racine, contraste racine / sol.
Image moyenne en niveaux de gris par groupe (image brute complète, bandes comprises), calculée sur le
split train seul : `run_mean_images`.

Usage (depuis la racine du projet) :
  PYTHONPATH=src python -m prmi.image_stats          # statistiques par image (environ 20 min)
  PYTHONPATH=src python -m prmi.image_stats means    # images moyennes, train seul (environ 12 min)
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from tqdm import tqdm

from prmi.inventory import RAW, ROOT, load_inventory

IMAGE_STATS = ROOT / "data" / "interim" / "image_stats.parquet"
MEAN_IMAGES = ROOT / "data" / "interim" / "image_means.npz"

# Bandes de remplissage du 120 DPI (640x480) : blanche à gauche et en haut, noire en bas.
BANDS_120 = {"left": (slice(None), slice(0, 16)), "top": (slice(0, 4), slice(None)),
             "bottom": (slice(472, 480), slice(None))}
USEFUL_120 = (slice(4, 472), slice(16, 640))


def group_name(row):
    return f"{row['dpi']} {row['site']} {row['date'].year}"


def gray(img):
    """Luminance (pondération ITU-R 601, celle de PIL en mode L)."""
    return img @ np.array([0.299, 0.587, 0.114])


def separability(a, b):
    """Écart des moyennes rapporté à l'écart-type commun (d')."""
    return (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)


def color_separability(root, soil):
    """d' de la meilleure combinaison linéaire des 3 canaux (analyse discriminante)."""
    diff = root.mean(0) - soil.mean(0)
    cov = (np.cov(root.T) + np.cov(soil.T)) / 2
    return float(np.sqrt(diff @ np.linalg.solve(cov + 1e-6 * np.eye(3), diff)))


def image_stats(row, raw=RAW):
    full = np.asarray(Image.open(raw / row["image_path"]).convert("RGB"))
    out = {}
    if row["dpi"] == 120:
        for name, sl in BANDS_120.items():
            band = full[sl].astype(float).mean(-1)
            out[f"band_{name}_mean"], out[f"band_{name}_std"] = band.mean(), band.std()
        img = full[USEFUL_120].astype(float)
    else:
        img = full.astype(float)
    y = gray(img)
    H, W = y.shape

    # Couleur
    out["R"], out["G"], out["B"] = img.reshape(-1, 3).mean(0)
    out["chroma"] = (img.max(-1) - img.min(-1)).mean()
    out["rb_ratio"] = out["R"] / max(out["B"], 1e-6)

    # Exposition et dynamique
    out["lum_mean"], out["lum_std"] = y.mean(), y.std()
    out["lum_p1"], out["lum_p50"], out["lum_p99"] = np.percentile(y, [1, 50, 99])
    out["clip_low"] = (img.max(-1) <= 2).mean()
    out["clip_high"] = (img.max(-1) >= 254).mean()

    # Éclairage : centre / bords, gauche / droite, haut / bas
    c = (slice(H // 4, 3 * H // 4), slice(W // 4, 3 * W // 4))
    border = np.ones_like(y, bool)
    border[c] = False
    out["center_border"] = y[c].mean() / max(y[border].mean(), 1e-6)
    out["left_right"] = y[:, : W // 3].mean() / max(y[:, -W // 3:].mean(), 1e-6)
    out["top_bottom"] = y[: H // 3].mean() / max(y[-H // 3:].mean(), 1e-6)

    # Netteté (variance du laplacien) et entrelacement (lignes voisines vs lignes à 2 d'écart)
    out["sharpness"] = ndi.laplace(y).var()
    out["comb"] = np.abs(np.diff(y, axis=0)).mean() / max(np.abs(y[2:] - y[:-2]).mean(), 1e-6)

    # Contraste racine / sol (images avec masque non vide)
    if row["has_root"] and row["mask_present"] and row["root_frac"] > 0:
        mask = np.asarray(Image.open(raw / row["mask_path"]))[..., 0] > 0
        if row["dpi"] == 120:
            mask = mask[USEFUL_120]
        root = ndi.binary_erosion(mask, iterations=1)                    # cœur de la racine
        soil = ~ndi.binary_dilation(mask, iterations=5)                  # sol à distance des bords
        if root.sum() >= 50:
            out["d_gray"] = separability(y[root], y[soil])
            for k, ch in enumerate("RGB"):
                out[f"d_{ch}"] = separability(img[..., k][root], img[..., k][soil])
            out["d_color"] = color_separability(img[root], img[soil])
            out["rb_root"] = img[..., 0][root].mean() / max(img[..., 2][root].mean(), 1e-6)
            out["rb_soil"] = img[..., 0][soil].mean() / max(img[..., 2][soil].mean(), 1e-6)
    return out


HARD_TYPES = ["orangée", "grise", "saturée", "faible contraste", "sombre"]


def image_type(stats):
    """Type d'image, une seule étiquette (priorité : orangée, grise, saturée, sombre, faible contraste, sinon typique).

    Seuils : R/B > 1,3 ; écart entre canaux < 5 ; plus de 5 % de pixels saturés ; 10 % les plus bas du groupe
    en luminosité ou en écart-type (fixés sur train et val, jamais sur le test).
    """
    ref = stats[stats["split"] != "test"].groupby("group")
    t = pd.Series("typique", index=stats.index)
    t[stats["lum_std"] < stats["group"].map(ref["lum_std"].quantile(0.10))] = "faible contraste"
    t[stats["lum_mean"] < stats["group"].map(ref["lum_mean"].quantile(0.10))] = "sombre"
    t[stats["clip_high"] > 0.05] = "saturée"
    t[stats["chroma"] < 5] = "grise"
    t[stats["rb_ratio"] > 1.3] = "orangée"
    return t


def run_image_stats(inv=None, workers=8, batch=256):
    inv = load_inventory() if inv is None else inv
    sel = inv[inv["species"] == "Peanut"].reset_index(drop=True)
    sel["group"] = sel.apply(group_name, axis=1)
    rows = []
    with ThreadPoolExecutor(workers) as ex, tqdm(total=len(sel), desc="images", unit="img") as bar:
        for start in range(0, len(sel), batch):
            chunk = sel.iloc[start:start + batch]
            rows += list(ex.map(image_stats, (r for _, r in chunk.iterrows())))
            bar.update(len(chunk))
    df = pd.concat([sel, pd.DataFrame(rows)], axis=1)
    df.to_parquet(IMAGE_STATS, index=False)
    return df


def mean_gray(row, raw=RAW):
    """Image brute complète (bandes comprises) en niveaux de gris."""
    return gray(np.asarray(Image.open(raw / row["image_path"]).convert("RGB"), dtype=float)).astype(np.float32)


def run_mean_images(inv=None, split="train", workers=8, batch=256):
    """Image moyenne en gris par groupe, calculée sur un seul split (train : pas de fuite vers val et test)."""
    inv = load_inventory() if inv is None else inv
    sel = inv[(inv["species"] == "Peanut") & (inv["split"] == split)].reset_index(drop=True)
    sel["group"] = sel.apply(group_name, axis=1)
    sums, counts = {}, {}
    # Par lots : chaque tâche renvoie une image, il ne faut pas les accumuler en mémoire.
    with ThreadPoolExecutor(workers) as ex, tqdm(total=len(sel), desc=f"moyennes ({split})", unit="img") as bar:
        for start in range(0, len(sel), batch):
            chunk = sel.iloc[start:start + batch]
            for g, img in zip(chunk["group"], ex.map(mean_gray, (r for _, r in chunk.iterrows()))):
                sums[g] = sums.get(g, 0) + img
                counts[g] = counts.get(g, 0) + 1
            bar.update(len(chunk))
    np.savez_compressed(MEAN_IMAGES, **{k: sums[k] / counts[k] for k in sums})
    return counts


def compare_conversions(stats, n_per_group=200, n_pixels=3000, seed=0, raw=RAW):
    """Séparabilité racine / sol |d'| pour des conversions fixes de l'image.

    Relit n_per_group images avec racine par groupe sur la clé. La combinaison couleur fixe
    (analyse discriminante sur les canaux normalisés par image) est apprise sur la moitié
    des images et évaluée sur l'autre moitié.
    """
    rng = np.random.default_rng(seed)
    sample = stats.dropna(subset=["d_gray"]).groupby("group").sample(n_per_group, random_state=seed)
    rows, pixels = [], []
    for _, r in sample.iterrows():
        img = np.asarray(Image.open(raw / r["image_path"]).convert("RGB"))
        mask = np.asarray(Image.open(raw / r["mask_path"]))[..., 0] > 0
        if r["dpi"] == 120:
            img, mask = img[USEFUL_120], mask[USEFUL_120]
        x = img.reshape(-1, 3).astype(float) / 255
        root = np.flatnonzero(ndi.binary_erosion(mask))
        soil = np.flatnonzero(~ndi.binary_dilation(mask, iterations=5))
        root = rng.choice(root, min(n_pixels, len(root)), replace=False)
        soil = rng.choice(soil, n_pixels, replace=False)
        px, lab = x[np.r_[root, soil]], np.r_[np.ones(len(root)), np.zeros(len(soil))].astype(bool)
        conv = {"gris": gray(px), "max des canaux": px.max(1), "rouge seul": px[:, 0]}
        rows.append({"group": r["group"], **{k: abs(separability(v[lab], v[~lab])) for k, v in conv.items()},
                     "couleur, optimum par image": r["d_color"]})
        pixels.append(((px - x.mean(0)) / (x.std(0) + 1e-6), lab))
    res = pd.DataFrame(rows)
    order = rng.permutation(len(pixels))
    train, test = order[: len(order) // 2], order[len(order) // 2:]
    z = np.concatenate([pixels[i][0] for i in train])
    y = np.concatenate([pixels[i][1] for i in train])
    cov = (np.cov(z[y].T) + np.cov(z[~y].T)) / 2
    w = np.linalg.solve(cov, z[y].mean(0) - z[~y].mean(0))
    w /= np.abs(w).sum()
    res["couleur, combinaison fixe"] = np.nan
    for i in test:
        v, lab = pixels[i][0] @ w, pixels[i][1]
        res.loc[i, "couleur, combinaison fixe"] = abs(separability(v[lab], v[~lab]))
    return res, w


def load_image_stats(path=IMAGE_STATS):
    return pd.read_parquet(path)


def load_mean_images(path=MEAN_IMAGES):
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["means"]:
        print(run_mean_images(), "images par groupe ->", MEAN_IMAGES)
    else:
        df = run_image_stats()
        print(f"{len(df)} images -> {IMAGE_STATS}")
