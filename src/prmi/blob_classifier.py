"""Classifieur des taches prédites : vraie racine ou faux positif, à partir des descripteurs de `postprocess.components`.

Deuxième étage après le U-Net : chaque tache prédite reçoit une probabilité d'être vraie ; les taches sous un seuil sont
retirées. Le modèle (gradient boosting) est construit et entraîné dans le notebook E4, puis enregistré avec son seuil
(`save`) ; ce module sert à l'appliquer dans la chaîne de prédiction.
"""
import joblib
import numpy as np
import pandas as pd

from prmi.postprocess import components, removal_effects

FEATURES = ["px", "p_moy", "p_max", "longueur", "largeur", "allongement", "p_trainee", "p_trainee_90", "dist_voisine",
            "px_voisine", "dist_bord", "n_taches", "px_image"]
def p_true(model, blobs):
    """Probabilité que chaque tache soit vraie."""
    return model.predict_proba(blobs[FEATURES])[:, 1]


def tradeoff(images, blobs, scores, taus=np.linspace(0, 1, 201)):
    """Effet du retrait des taches de score inférieur au seuil, pour chaque seuil (`removal_effects`)."""
    return pd.DataFrame([{"seuil": t, **removal_effects(images, blobs, scores < t)} for t in taus])


def threshold_for(curve, max_loss):
    """Seuil qui retire le plus de taches fausses avec une perte de longueur d'au plus `max_loss`."""
    ok = curve[curve["longueur perdue"] <= max_loss]
    return float(ok.loc[ok["fausses retirées"].idxmax(), "seuil"])


def classifier_filter(prob, model, tau):
    """Post-traitement : prédiction (probabilité > 0,5) sans les taches que le classifieur juge fausses (score < tau)."""
    pred = prob > 0.5
    comps, labels = components(pred, prob)
    if comps.empty:
        return pred
    return pred & ~np.isin(labels, comps.loc[p_true(model, comps) < tau, "label"])


def save(model, tau, path):
    joblib.dump({"model": model, "tau": tau, "features": FEATURES}, path)


def load(path):
    d = joblib.load(path)
    return d["model"], d["tau"]


def v11_options(model, tau):
    """Les deux chaînes comparées sur le test (notebook 07) : U-Net seul (version 1) et U-Net + classifieur (1.1)."""
    return {"U-Net seul": lambda p: p > 0.5, "U-Net + classifieur": lambda p: classifier_filter(p, model, tau)}
