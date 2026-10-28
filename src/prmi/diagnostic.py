"""Diagnostic des masques PRMI : décalage, coupures entre segments, remplissage.

Indicateurs par image (voir `diagnose`) :
- décalage (dy, dx) : translation du masque qui maximise le contraste local moyen sous
  le masque, avec la saillance du pic (z_peak) et le score sans translation (z0).
  Les racines sont en général plus claires que le sol ; si le masque d'origine est posé
  sur une structure nettement sombre (z0 < -1), la racine est supposée sombre.
  Estimation jugée fiable (`reliable`) si le squelette du masque fait au moins 150 px
  et si le pic n'est pas sur le bord de la fenêtre de recherche ;
- coupures : composantes connexes reliées par une fermeture morphologique de rayon 2 px,
  rapportées à la longueur du squelette ;
- petits trous : pixels non remplis entourés de masque (trous de moins de 50 px) ;
- débord : luminosité juste à l'extérieur du masque recalé, rapportée à celle de
  l'intérieur ; proche de 1 si la racine est plus large que le masque.

Usage (depuis la racine du projet) : PYTHONPATH=src python -m prmi.diagnostic
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from scipy.signal import fftconvolve
from skimage.morphology import closing, disk, skeletonize
from tqdm import tqdm

from prmi.inventory import RAW, ROOT, load_inventory

DIAGNOSTIC = ROOT / "data" / "interim" / "mask_diagnostic.parquet"
RADIUS = 60  # décalage maximal recherché (px)
AMP_MIN, GAIN_NET, GAIN_POSSIBLE = 5, 2.5, 1.0  # seuils de décalage (voir load_diagnostic)


def load_pair(row, raw=RAW):
    """Image RGB (uint8) et masque booléen d'une ligne de l'inventaire."""
    img = np.asarray(Image.open(raw / row["image_path"]).convert("RGB"))
    mask = np.asarray(Image.open(raw / row["mask_path"]))[..., 0] > 0
    return img, mask


def overlay_mask(img, mask, color=(30, 100, 255), alpha=0.45, shift=(0, 0)):
    """Masque (éventuellement translaté de (dy, dx)) en aplat coloré semi-transparent sur l'image."""
    out = img.astype(float)
    m = np.roll(mask, shift, axis=(0, 1))
    out[m] = (1 - alpha) * out[m] + alpha * np.array(color)
    return out.astype(np.uint8)


def local_contrast(img, sigma_bg=15, margin=16):
    """Luminosité moins le fond local, réduite : positive sur les structures claires.

    Une bande de `margin` px est mise à zéro pour neutraliser le vignettage des bords.
    """
    g = ndi.gaussian_filter(img.mean(-1), 1.5)
    h = g - ndi.gaussian_filter(g, sigma_bg)
    h /= h.std()
    h[:margin], h[-margin:], h[:, :margin], h[:, -margin:] = 0, 0, 0, 0
    return h


def shift_scores(h, mask, radius=RADIUS):
    """Score moyen de h sous le masque translaté de (dy, dx), pour |dy|, |dx| <= radius.

    Le résultat est centré sur la médiane et réduit : un pic net est un pic de z élevé.
    """
    H, W = mask.shape
    c = fftconvolve(h, mask[::-1, ::-1].astype(float), mode="full")
    c = c[H - 1 - radius:H + radius, W - 1 - radius:W + radius] / mask.sum()
    return (c - np.median(c)) / c.std()


def estimate_offset(h, mask, radius=RADIUS):
    s = shift_scores(h, mask, radius)
    polarity = -1 if s[radius, radius] < -1 else 1
    s = polarity * s
    iy, ix = np.unravel_index(s.argmax(), s.shape)
    return {
        "dy": int(iy - radius), "dx": int(ix - radius), "polarity": polarity,
        "z_peak": float(s.max()), "z0": float(s[radius, radius]),
        "at_border": bool(min(iy, ix) == 0 or max(iy, ix) == 2 * radius),
    }


def mask_metrics(mask):
    n_cc = ndi.label(mask)[1]
    n_closed = ndi.label(closing(mask, disk(2)))[1]
    skel_len = int(skeletonize(mask).sum())
    holes, n_holes = ndi.label(ndi.binary_fill_holes(mask) & ~mask)
    sizes = np.bincount(holes.ravel())[1:]
    return {
        "n_components": n_cc,
        "gaps": n_cc - n_closed,
        "skeleton_px": skel_len,
        "gaps_per_100px": 100 * (n_cc - n_closed) / max(skel_len, 1),
        "small_holes_px": int(sizes[sizes < 50].sum()),
        "small_holes_frac": float(sizes[sizes < 50].sum() / mask.sum()),
    }


def overflow_ratio(h, mask, shift, ring=3):
    """Contraste moyen dans un anneau de `ring` px autour du masque recalé / à l'intérieur."""
    m = np.roll(mask, shift, axis=(0, 1))
    outer = ndi.binary_dilation(m, iterations=ring) & ~m
    inside = h[m].mean()
    return float(h[outer].mean() / inside) if inside > 0 else np.nan


def diagnose(row, raw=RAW):
    img, mask = load_pair(row, raw)
    h = local_contrast(img)
    off = estimate_offset(h, mask)
    metrics = mask_metrics(mask)
    reliable = metrics["skeleton_px"] >= 150 and not off["at_border"]
    overflow = overflow_ratio(off["polarity"] * h, mask, (off["dy"], off["dx"]))
    return {**off, "reliable": reliable, **metrics, "overflow": overflow}


def sample_sessions(inv, n_sessions=40, per_session=10, seed=0):
    """Tire n_sessions sessions (tube, date) par sous-dossier, puis jusqu'à per_session images
    avec racine dans chacune. Les sessions de moins de 3 images sont écartées."""
    r = inv[inv["has_root"] & inv["mask_present"] & (inv["root_frac"] > 0)]
    keys = ["folder", "tube_id", "date"]
    sizes = r.groupby(keys).size()
    sessions = sizes[sizes >= 3].reset_index()[keys]
    sessions = sessions.sample(frac=1, random_state=seed).groupby("folder").head(n_sessions)
    picked = r.merge(sessions, on=keys).sample(frac=1, random_state=seed)
    return picked.groupby(keys).head(per_session).sort_values(keys).reset_index(drop=True)


def run_diagnostic(inv=None, out=DIAGNOSTIC, workers=8):
    inv = load_inventory() if inv is None else inv
    sample = sample_sessions(inv)
    with ThreadPoolExecutor(workers) as ex:
        res = list(tqdm(ex.map(diagnose, (row for _, row in sample.iterrows())),
                        total=len(sample), desc="diagnostic", unit="img"))
    df = pd.concat([sample.reset_index(drop=True), pd.DataFrame(res)], axis=1)
    df.to_parquet(out, index=False)
    return df


def load_diagnostic(path=DIAGNOSTIC):
    """Charge le cache et ajoute l'amplitude, le gain et les deux niveaux de décalage.

    Seuils calibrés à l'œil sur 18 images (notebook 02) : au-dessus d'un gain de 2,5,
    aucun recalage faux n'a été vu (« net », borne basse) ; entre 1 et 2,5, environ la
    moitié des cas décidables sont de fausses alertes (« possible », borne haute).
    """
    d = pd.read_parquet(path)
    d["amp"] = np.hypot(d["dy"], d["dx"])
    d["gain"] = d["z_peak"] - d["z0"]
    shifted = d["reliable"] & (d["amp"] > AMP_MIN)
    d["offset_net"] = shifted & (d["gain"] >= GAIN_NET)
    d["offset_possible"] = shifted & (d["gain"] >= GAIN_POSSIBLE)
    return d


def summarize(d):
    """Indicateurs par sous-dossier (taux de décalage sur les estimations fiables)."""
    r = d[d["reliable"]]
    aligned = d[d["reliable"] & (d["amp"] <= 3)]
    return pd.DataFrame({
        "images": d.groupby("folder").size(),
        "fiables": d.groupby("folder")["reliable"].mean(),
        "décalage net": r.groupby("folder")["offset_net"].mean(),
        "décalage possible": r.groupby("folder")["offset_possible"].mean(),
        "amplitude médiane (net, px)": r[r["offset_net"]].groupby("folder")["amp"].median(),
        "masques avec coupure": d.groupby("folder")["gaps"].apply(lambda g: (g > 0).mean()),
        "petits trous (% surface)": 100 * d.groupby("folder")["small_holes_frac"].median(),
        "débord (alignés)": aligned.groupby("folder")["overflow"].median(),
    })


if __name__ == "__main__":
    df = run_diagnostic()
    print(f"{len(df)} images diagnostiquées -> {DIAGNOSTIC}")
