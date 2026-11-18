"""Mesure des traits racinaires sur un masque : longueur et diamètre, en pixels puis en millimètres.

Échelle : le DPI annoncé par PRMI (150 ou 120) est un réglage du logiciel d'annotation, pas l'échelle réelle (notebook
A3 : des images prises tous les 13,5 mm le long du tube ne se recouvrent pas, ce qui exclut 5,9 px/mm). Une image
correspond à un cadre de la caméra Bartz d'environ 18 x 13,5 mm (FRAME_MM), soit 40,9 px/mm pour les images de
736 x 552 px et 35,6 px/mm pour celles de 640 x 480 px.

Longueur : squelette du masque, élagué de ses petites branches parasites (coins des rectangles, irrégularités du bord),
éventuellement complété par des ponts entre morceaux d'une même racine prédite en pointillés ; chaque pas diagonal
compte pour racine de 2. Diamètre : deux fois la distance au bord du masque, le long du squelette.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.draw import line
from skimage.morphology import skeletonize

FRAME_MM = (18.0, 13.5)          # largeur et hauteur d'une image, mm (cadre de la caméra Bartz BTC-100X)
WIDTH_PX = {150: 736, 120: 640}  # largeur des images d'origine selon le « DPI » PRMI
EIGHT = np.ones((3, 3), dtype=bool)
# Étalonnage de la longueur : sur le val corrigé, la longueur prédite est 16,7 % trop courte (racines non détectées,
# notebook A3) ; facteur à appliquer aux totaux (par tube, par date), pas à une image seule. Vérifié sur le test (07).
LENGTH_FACTOR = round(1 / (1 - 0.167), 3)


def px_per_mm(dpi):
    """Échelle (pixels par millimètre) des images d'un capteur, d'après la taille du cadre."""
    return WIDTH_PX[dpi] / FRAME_MM[0]


def _neighbors(skel):
    return ndi.convolve(skel.astype(np.uint8), EIGHT.astype(np.uint8), mode="constant") - skel


def skeleton_length(skel):
    """Longueur d'un squelette (px) : pas horizontaux et verticaux comptés 1, pas diagonaux racine de 2 (sauf quand un
    chemin à angle droit relie déjà les deux pixels)."""
    s = skel.astype(bool)
    ortho = (s[:, :-1] & s[:, 1:]).sum() + (s[:-1, :] & s[1:, :]).sum()
    d1 = s[:-1, :-1] & s[1:, 1:] & ~s[:-1, 1:] & ~s[1:, :-1]
    d2 = s[:-1, 1:] & s[1:, :-1] & ~s[:-1, :-1] & ~s[1:, 1:]
    return float(ortho + np.sqrt(2) * (d1.sum() + d2.sum()))


def _trace(skel, start, junction, max_len):
    """Chemin depuis une extrémité jusqu'à un carrefour (exclu), une autre extrémité ou max_len pixels. Renvoie le
    chemin et la raison de l'arrêt ("carrefour", "fin" ou "long")."""
    path, prev, cur = [start], None, start
    h, w = skel.shape
    while len(path) <= max_len:
        y, x = cur
        nxt = [(y + dy, x + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy or dx)
               and 0 <= y + dy < h and 0 <= x + dx < w and skel[y + dy, x + dx] and (y + dy, x + dx) != prev
               and (y + dy, x + dx) not in path]
        if not nxt:
            return path, "fin"
        if any(junction[p] for p in nxt) or len(nxt) > 1:
            return path, "carrefour"
        prev, cur = cur, nxt[0]
        path.append(cur)
    return path, "long"


def prune(skel, min_branch):
    """Retire les branches terminales de moins de `min_branch` px qui partent d'un carrefour (branches parasites). Un
    morceau isolé, sans carrefour, est gardé quelle que soit sa longueur."""
    s = skel.astype(bool).copy()
    if min_branch <= 0 or not s.any():
        return s
    n = _neighbors(s)
    junction = s & (n >= 3)
    for y, x in np.argwhere(s & (n == 1)):
        path, why = _trace(s, (y, x), junction, min_branch)
        if why == "carrefour" and len(path) < min_branch:
            for p in path:
                s[p] = False
    return s


def _end_direction(skel, end, k=8):
    """Direction (vecteur unitaire) dans laquelle pointe une extrémité : de son k-ième voisin le long du squelette vers
    elle."""
    path, _ = _trace(skel, end, np.zeros_like(skel), k)
    v = np.array(end, float) - np.array(path[-1], float)
    return v / np.linalg.norm(v) if np.linalg.norm(v) else v


def reconnect(skel, max_gap, max_angle=35):
    """Relie par un segment les extrémités de deux morceaux différents qui se font face : distance au plus `max_gap`
    px, et chaque extrémité pointe vers l'autre à `max_angle` degrés près. Chaque extrémité sert au plus une fois, les
    paires les plus proches d'abord. Renvoie le squelette complété et le nombre de ponts."""
    s = skel.astype(bool).copy()
    if max_gap <= 0 or not s.any():
        return s, 0
    labels = ndi.label(s, structure=EIGHT)[0]
    ends = [tuple(e) for e in np.argwhere(s & (_neighbors(s) == 1))]
    if len(ends) < 2:
        return s, 0
    dirs = {e: _end_direction(s, e) for e in ends}
    cos = np.cos(np.radians(max_angle))
    pairs = []
    for i, a in enumerate(ends):
        for b in ends[i + 1:]:
            if labels[a] == labels[b]:
                continue
            v = np.array(b, float) - np.array(a, float)
            d = np.linalg.norm(v)
            if 0 < d <= max_gap and dirs[a] @ v / d >= cos and dirs[b] @ -v / d >= cos:
                pairs.append((d, a, b))
    used, n = set(), 0
    for d, a, b in sorted(pairs):
        if a in used or b in used:
            continue
        rr, cc = line(*a, *b)
        s[rr, cc] = True
        used |= {a, b}
        n += 1
    return s, n


def measure(mask, min_branch=0, max_gap=0, width_mask=None):
    """Traits d'un masque (px) : longueur du squelette (élagué, éventuellement reconnecté) et diamètre moyen (deux fois
    la distance au bord le long du squelette d'origine, moyenne pondérée par la longueur). `width_mask` : masque sur
    lequel mesurer la largeur (par défaut `mask`)."""
    if not mask.any():
        return {"longueur": 0.0, "diametre": np.nan, "ponts": 0}
    skel = prune(skeletonize(mask), min_branch)
    width = width_mask if width_mask is not None else mask
    dist = ndi.distance_transform_edt(width)
    on = skel & width
    diam = float(np.mean(2 * dist[on] - 1)) if on.any() else np.nan
    skel, bridges = reconnect(skel, max_gap)
    return {"longueur": skeleton_length(skel), "diametre": diam, "ponts": bridges}


# --- Vérification de l'échelle -----------------------------------------------------------------------------------

def best_shift(a, b):
    """Décalage (dy, dx) qui superpose au mieux le masque b sur le masque a, et part du plus petit masque alors
    superposée."""
    from scipy.signal import fftconvolve
    c = fftconvolve(a.astype(float), b[::-1, ::-1].astype(float), mode="full")
    iy, ix = np.unravel_index(c.argmax(), c.shape)
    return iy - (b.shape[0] - 1), ix - (b.shape[1] - 1), c.max() / min(a.sum(), b.sum())


def overlap_test(stats, read_mask, n=100, seed=0):
    """Meilleur décalage entre masques de positions consécutives d'un même tube à la même date, et entre masques tirés
    au hasard dans des tubes différents (témoin). `stats` : une ligne par image (tube_id, date, position, mask_path),
    images avec racine ; read_mask(chemin) -> masque booléen."""
    import pandas as pd
    rows = []
    for (tube, date), g in stats.groupby(["tube_id", "date"]):
        g = g.sort_values("position").set_index("position")
        for p in g.index:
            if p + 1 in g.index and len(rows) < n:
                rows.append(("consécutives", *best_shift(read_mask(g.loc[p, "mask_path"]), read_mask(g.loc[p + 1, "mask_path"]))))
        if len(rows) >= n:
            break
    rng = np.random.default_rng(seed)
    while len(rows) < 2 * n:
        a, b = stats.iloc[rng.integers(len(stats), size=2)].itertuples()
        if a.tube_id != b.tube_id:
            rows.append(("au hasard", *best_shift(read_mask(a.mask_path), read_mask(b.mask_path))))
    return pd.DataFrame(rows, columns=["paires", "dy", "dx", "recouvrement"])


def compare(preds, refs, ppm, **kw):
    """Traits par image (mm) d'une liste de prédictions et des références correspondantes (masques booléens)."""
    import pandas as pd
    rows = []
    for p, r in zip(preds, refs):
        P, R = measure(p, **kw), measure(r, **{k: v for k, v in kw.items() if k == "min_branch"})
        rows.append({"longueur réf.": R["longueur"] / ppm, "longueur prédite": P["longueur"] / ppm,
                     "diamètre réf.": R["diametre"] / ppm, "diamètre prédit": P["diametre"] / ppm, "ponts": P["ponts"]})
    return pd.DataFrame(rows)


def errors(t):
    """Erreur relative médiane par image et biais d'ensemble (somme prédite / somme de référence - 1 ; diamètre moyen
    pondéré par la longueur) d'un tableau de `compare`."""
    L = (t["longueur prédite"] - t["longueur réf."]).abs() / t["longueur réf."]
    D = (t["diamètre prédit"] - t["diamètre réf."]).abs() / t["diamètre réf."]
    wd = lambda d, l: (t[d] * t[l]).sum() / t[l].sum()
    return {"longueur : erreur médiane": L.median(), "longueur : biais": t["longueur prédite"].sum() / t["longueur réf."].sum() - 1,
            "diamètre : erreur médiane": D.median(),
            "diamètre : biais": wd("diamètre prédit", "longueur prédite") / wd("diamètre réf.", "longueur réf.") - 1}


def skeleton_widths(mask):
    """Diamètres locaux (px) le long du squelette d'un masque : deux fois la distance au bord, moins un."""
    skel = skeletonize(mask)
    return 2 * ndi.distance_transform_edt(mask)[skel] - 1


def missed_extra(pred, ref, tol=3):
    """Longueur de référence sans prédiction à moins de `tol` px (racines manquées) et longueur prédite sans
    référence à moins de `tol` px (en trop), en px."""
    sr, sp = skeletonize(ref), skeletonize(pred)
    near_p, near_r = ndi.binary_dilation(pred, iterations=tol), ndi.binary_dilation(ref, iterations=tol)
    return {"référence": skeleton_length(sr), "manquée": skeleton_length(sr & ~near_p),
            "prédite": skeleton_length(sp), "en trop": skeleton_length(sp & ~near_r)}
