"""Root Segmentator : de l'image de minirhizotron aux mesures racinaires en millimètres.

Chaîne pour une image (`RootSegmentator.analyze`) :
1. capteur reconnu à la taille de l'image (736 x 552 px : « 150 DPI », 640 x 480 px : « 120 DPI ») ;
2. prétraitement validé à l'entraînement : recadrage (120 DPI), correction du fond par la carte du capteur,
   normalisation par image, réduction de moitié ;
3. U-Net (carte de probabilité), puis classifieur des taches, qui retire les taches prédites jugées fausses ;
4. mesures (`prmi.traits`) : longueur du squelette, diamètre moyen, surface, intensité racinaire (longueur par
   surface d'image observée), converties en mm à l'échelle vérifiée du capteur (40,9 ou 35,6 px/mm).

Les fichiers du modèle viennent de la release GitHub (`prmi.weights`). Tout tourne sur processeur, sans GPU.
"""
import re
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from prmi import traits, weights
from prmi.arrays import pad_image, shrink_image
from prmi.blob_classifier import classifier_filter
from prmi.model import build_model
from prmi.preprocess import preprocess_image

SENSORS = {(552, 736): 150, (480, 640): 120}          # (hauteur, largeur) des images d'origine -> capteur
SCALE = 2                                             # réduction appliquée à l'entraînement
STEP_MM, TUBE_ANGLE = 13.5, 45                        # pas entre deux positions le long du tube, angle du tube (degrés)
PRMI_NAME = re.compile(r"(?P<espece>[A-Za-z]+)_T(?P<tube>\d+)_L(?P<position>\d+)_(?P<date>\d{4}\.\d{2}\.\d{2})_"
                       r"(?P<heure>\d{6})_(?P<site>[A-Za-z]+)_DPI(?P<dpi>\d+)")


def parse_name(name):
    """Tube, position, date et site lus dans un nom d'image au format PRMI ; profondeur estimée (cm) : position x pas
    de 13,5 mm le long d'un tube incliné à 45 degrés. Champs vides si le nom ne suit pas ce format."""
    m = PRMI_NAME.search(name)
    if not m:
        return {"site": None, "tube": None, "position": None, "date": None, "profondeur_cm": None}
    pos = int(m["position"])
    return {"site": m["site"], "tube": int(m["tube"]), "position": pos, "date": m["date"].replace(".", "-"),
            "profondeur_cm": round(pos * STEP_MM * np.sin(np.radians(TUBE_ANGLE)) / 10, 1)}


@dataclass
class Result:
    """Résultat pour une image : mesures, image prétraitée (affichage), probabilité et masque à la résolution
    d'origine (après recadrage)."""
    mesures: dict
    image: np.ndarray
    proba: np.ndarray
    masque: np.ndarray


class RootSegmentator:
    """Modèle chargé une fois, puis `analyze(image)` pour chaque image."""

    def __init__(self, device=None, use_classifier=True):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = build_model(weights=None).to(self.device)
        self.model.load_state_dict(torch.load(weights.fetch("unet"), map_location=self.device))
        self.model.eval()
        clf = joblib.load(weights.fetch("classifier"))
        self.clf, self.tau = clf["model"], clf["tau"]
        self.use_classifier = use_classifier
        with np.load(weights.fetch("flatfields")) as f:
            self.flats = {int(k): f[k] for k in f.files}

    @torch.no_grad()
    def probability(self, rgb, dpi):
        """Carte de probabilité de racine (résolution d'origine, après recadrage) et image prétraitée."""
        image = preprocess_image(rgb, self.flats[dpi], dpi)
        small = shrink_image(image, SCALE)
        x, _ = pad_image(small, np.zeros(small.shape[:2], bool))
        h, w = small.shape[:2]
        logits = self.model(x[None].to(self.device))[..., :h, :w]
        logits = F.interpolate(logits, size=image.shape[:2], mode="bilinear", align_corners=False)
        return torch.sigmoid(logits)[0, 0].cpu().numpy(), image

    def analyze(self, rgb, name="image", px_per_mm=None):
        """Analyse une image RGB (uint8, taille d'origine du capteur). Renvoie un `Result`."""
        rgb = np.asarray(rgb)[..., :3]
        dpi = SENSORS.get(rgb.shape[:2])
        if dpi is None:
            raise ValueError(f"{name} : taille {rgb.shape[1]} x {rgb.shape[0]} px non reconnue (attendu : 736 x 552 "
                             "ou 640 x 480, images PRMI d'arachide)")
        prob, image = self.probability(rgb, dpi)
        raw = prob > 0.5
        mask = classifier_filter(prob, self.clf, self.tau) if self.use_classifier else raw
        ppm = px_per_mm or traits.px_per_mm(dpi)
        m = traits.measure(mask)
        area_cm2 = mask.size / ppm ** 2 / 100
        length = m["longueur"] / ppm
        mesures = {"image": name, **parse_name(name), "capteur": f"{dpi} DPI", "px_par_mm": round(ppm, 2),
                   "racine": bool(mask.any()), "longueur_mm": round(length, 2),
                   "longueur_etalonnee_mm": round(length * traits.LENGTH_FACTOR, 2),
                   "diametre_moyen_mm": round(m["diametre"] / ppm, 3) if mask.any() else None,
                   "surface_mm2": round(mask.sum() / ppm ** 2, 3), "surface_image_cm2": round(area_cm2, 3),
                   "intensite_mm_cm2": round(length / area_cm2, 2),
                   "taches_retirees": int(_n_blobs(raw) - _n_blobs(mask))}
        return Result(mesures, image, prob, mask)

    def analyze_file(self, path, **kw):
        path = Path(path)
        return self.analyze(np.asarray(Image.open(path).convert("RGB")), name=path.stem, **kw)


def _n_blobs(mask):
    from scipy import ndimage as ndi
    return ndi.label(mask, structure=np.ones((3, 3)))[1]


def overlay(image, mask, color=(30, 100, 255), alpha=0.45):
    """Masque en aplat semi-transparent sur l'image prétraitée (flottants entre 0 et 1), en RGB uint8."""
    out = (np.clip(image, 0, 1) * 255).astype(float)
    out[mask] = (1 - alpha) * out[mask] + alpha * np.array(color)
    return out.astype(np.uint8)


def summarize(table, depth_class_cm=10):
    """Totaux par site, tube, date et classe de profondeur, à partir du tableau par image : nombre d'images, part avec
    racine, longueur totale (brute et étalonnée), intensité racinaire moyenne (mm/cm²), diamètre moyen pondéré par la
    longueur. Les images sans nom PRMI sont regroupées sous des champs vides."""
    import pandas as pd
    t = table.copy()
    t["profondeur_classe_cm"] = (t["profondeur_cm"] // depth_class_cm * depth_class_cm).astype("Int64").astype(str) \
        .str.cat((t["profondeur_cm"] // depth_class_cm * depth_class_cm + depth_class_cm).astype("Int64").astype(str),
                 sep="-")
    t["_ld"] = t["longueur_mm"] * t["diametre_moyen_mm"].fillna(0)
    keys = ["site", "tube", "date", "profondeur_classe_cm"]
    g = t.groupby(keys, dropna=False)
    out = pd.DataFrame({
        "images": g.size(),
        "part_avec_racine": g["racine"].mean().round(3),
        "longueur_mm": g["longueur_mm"].sum().round(1),
        "longueur_etalonnee_mm": g["longueur_etalonnee_mm"].sum().round(1),
        "intensite_mm_cm2": g["intensite_mm_cm2"].mean().round(2),
        "diametre_moyen_mm": (g["_ld"].sum() / g["longueur_mm"].sum()).round(3),
    }).reset_index()
    return out
