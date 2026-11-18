"""Description des taches prédites et seuillage par hystérésis."""
import numpy as np

from prmi.postprocess import components, hysteresis


def scene():
    pred, prob = np.zeros((80, 200), bool), np.zeros((80, 200), np.float32)
    pred[40:46, 10:190], prob[40:46, 10:190] = True, 0.95       # racine : longue, fine, sûre
    pred[10:20, 20:30], prob[10:20, 20:30] = True, 0.6          # tache : courte, trapue, peu sûre
    return pred, prob


def test_components_describe_shape_and_confidence():
    pred, prob = scene()
    ref = np.zeros_like(pred)
    ref[40:46, 10:190] = True
    c = components(pred, prob, ref)[0].sort_values("px")
    small, root = c.iloc[0], c.iloc[1]
    assert root["allongement"] > 10 * small["allongement"]
    assert root["p_moy"] > small["p_moy"]
    assert bool(root["vraie"]) and not bool(small["vraie"])
    assert root["n_taches"] == 2


def test_hysteresis_keeps_trails_attached_to_sure_pixels():
    prob = np.zeros((20, 100), np.float32)
    prob[10, 10:50] = 0.9                         # racine sûre
    prob[10, 50:70] = 0.3                         # traînée peu sûre qui la prolonge
    prob[10, 80:95] = 0.3                         # tache peu sûre isolée
    out = hysteresis(prob, high=0.5, low=0.2)
    assert out[10, 50:70].all() and not out[10, 80:95].any()
    assert (hysteresis(prob) == (prob > 0.5)).all()
