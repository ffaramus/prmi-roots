"""Post-traitement des prédictions : description des taches prédites et filtre des taches parasites.

Chaque tache prédite (composante connexe) est décrite par sa taille, sa probabilité moyenne et maximale (sortie du
modèle après sigmoïde), sa forme (longueur du squelette, largeur moyenne = surface / longueur, allongement = longueur /
largeur : une racine est longue et fine, une tache parasite plutôt courte et trapue) et, si une référence est donnée,
ce qu'elle contient de vrai (pixels de racine, longueur de racine de référence couverte).

Sert au classifieur des taches (`prmi.blob_classifier`, notebook E4) et au seuillage par hystérésis (`hysteresis`,
essai écarté). Réglages sur le val de PRMI, jamais sur le test.
"""
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

EIGHT = np.ones((3, 3), dtype=bool)
MIN_FP = 100        # px : une tache fausse compte comme fantôme à partir de cette taille (comme prmi.train)

def components(pred, prob, ref=None, ring=6):
    """Tableau des taches de `pred` (booléen) avec la carte de probabilité `prob` (même taille) et éventuellement la
    référence `ref`. Renvoie le tableau et l'image des étiquettes.

    Descripteurs de la tache : taille, probabilités, forme. Descripteurs de contexte : traînée (probabilité autour de
    la tache, hors prédiction, dans un anneau de `ring` px : moyenne et centile 90), distance à la tache voisine la plus
    proche et taille de celle-ci, distance au bord de l'image, nombre de taches et surface prédite dans l'image.
    """
    labels, n = ndi.label(pred, structure=EIGHT)
    ref_skel = skeletonize(ref) if ref is not None and ref.any() else None
    H, W = pred.shape
    sizes = np.bincount(labels.ravel())
    rows = []
    for k, sl in enumerate(ndi.find_objects(labels), start=1):
        m = labels[sl] == k
        length = max(int(skeletonize(m).sum()), 1)
        px = int(m.sum())
        ys, xs = np.nonzero(m)
        ys, xs = ys + sl[0].start, xs + sl[1].start
        p = prob[sl][m]
        win = (slice(max(sl[0].start - ring - 1, 0), sl[0].stop + ring + 1),
               slice(max(sl[1].start - ring - 1, 0), sl[1].stop + ring + 1))
        around = ndi.binary_dilation(labels[win] == k, iterations=ring) & ~pred[win]
        pr = prob[win][around]
        if n > 1:
            dist, (iy, ix) = ndi.distance_transform_edt((labels == 0) | (labels == k), return_indices=True)
            j = int(np.argmin(dist[ys, xs]))
            near = labels[iy[ys[j], xs[j]], ix[ys[j], xs[j]]]
            d_near, px_near = float(dist[ys[j], xs[j]]), int(sizes[near])
        else:
            d_near, px_near = float(np.hypot(H, W)), 0
        row = {"label": k, "y": ys.mean(), "x": xs.mean(), "y0": sl[0].start, "y1": sl[0].stop, "x0": sl[1].start,
               "x1": sl[1].stop, "px": px, "p_moy": float(p.mean()), "p_max": float(p.max()), "longueur": length,
               "largeur": px / length, "allongement": length / (px / length),
               "p_trainee": float(pr.mean()) if pr.size else 0.0,
               "p_trainee_90": float(np.quantile(pr, 0.9)) if pr.size else 0.0,
               "dist_voisine": d_near, "px_voisine": px_near,
               "dist_bord": int(min(ys.min(), xs.min(), H - 1 - ys.max(), W - 1 - xs.max())),
               "n_taches": n, "px_image": int(pred.sum())}
        if ref is not None:
            sk = ref_skel[ys, xs] if ref_skel is not None else np.zeros(px, bool)
            row.update(vraie=bool(ref[ys, xs].any()), px_vrais=int(ref[ys, xs].sum()), squelette_ref=int(sk.sum()))
        rows.append(row)
    return pd.DataFrame(rows), labels


def hysteresis(prob, high=0.5, low=0.5):
    """Seuillage par hystérésis : les pixels au-dessus de `high` servent de graines, on y ajoute les pixels au-dessus de
    `low` qui leur sont reliés. high = low = 0,5 : prédiction habituelle. Un `low` plus bas prolonge les racines sûres
    par leurs traînées peu sûres ; un `high` plus haut efface les taches qui n'ont aucun pixel très sûr."""
    from skimage.filters import apply_hysteresis_threshold
    return apply_hysteresis_threshold(prob, low, high) if low < high else prob > high


def variants(highs=(0.5, 0.8, 0.9), lows=(0.2, 0.3, 0.5)):
    """Variantes comparées dans le notebook E4 : {nom : fonction prob -> masque}. Sans post-traitement (seuil 0,5) et
    hystérésis pour chaque couple (graine, prolongement)."""
    from functools import partial
    out = {"sans post-traitement": lambda p: p > 0.5}
    for high in highs:
        for low in lows:
            if (high, low) != (0.5, 0.5) and low <= high:
                out[f"hystérésis {high:.1f} / {low:.1f}".replace(".", ",")] = partial(hysteresis, high=high, low=low)
    return out



def removal_effects(images, blobs, gone):
    """Effet du retrait des taches `gone` (booléen aligné sur `blobs`), calculé sur les tables sans refaire passer le
    modèle : part des taches fausses retirées, fantômes restants, images vides touchées (au moins MIN_FP px prédits),
    Dice global, part de la longueur de racine de référence perdue, nombre de taches vraies retirées."""
    gone = np.asarray(gone, bool)
    true = blobs["vraie"].astype(bool).to_numpy()
    tp = images["tp"].sum() - blobs.loc[gone, "px_vrais"].sum()
    fp = images["fp"].sum() - (blobs.loc[gone, "px"] - blobs.loc[gone, "px_vrais"]).sum()
    fn = images["fn"].sum() + blobs.loc[gone, "px_vrais"].sum()
    empty = images.loc[~images["has_root"], "name"]
    kept = blobs[~gone]
    left = kept[kept["name"].isin(empty)].groupby("name")["px"].sum().reindex(empty, fill_value=0)
    return {"fausses retirées": (gone & ~true).sum() / max((~true).sum(), 1),
            "fantômes": int((~gone & ~true & (blobs["px"] >= MIN_FP).to_numpy()).sum()),
            "images vides touchées": (left >= MIN_FP).mean(),
            "Dice global": 2 * tp / (2 * tp + fp + fn),
            "longueur perdue": blobs.loc[gone & true, "squelette_ref"].sum() / max(images["skel_ref"].sum(), 1),
            "vraies retirées": int((gone & true).sum())}
