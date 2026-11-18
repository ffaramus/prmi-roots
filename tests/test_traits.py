"""Mesure des traits : longueur du squelette, diamètre, échelle, élagage et reconnexion."""
import numpy as np
import pytest

from prmi import traits as tr


def test_scale_from_camera_frame():
    assert tr.px_per_mm(150) == pytest.approx(736 / 18)
    assert tr.px_per_mm(120) == pytest.approx(640 / 18)


def test_skeleton_length_straight_and_diagonal():
    s = np.zeros((20, 120), bool)
    s[10, 10:110] = True
    assert tr.skeleton_length(s) == pytest.approx(99)
    d = np.eye(50, dtype=bool)
    assert tr.skeleton_length(d) == pytest.approx(49 * np.sqrt(2))


def test_measure_bar_width_and_length():
    m = np.zeros((60, 300), bool)
    m[26:35, 20:280] = True                      # racine droite de 9 px de large, 260 px de long
    r = tr.measure(m)
    assert r["diametre"] == pytest.approx(9, abs=1)
    assert r["longueur"] == pytest.approx(260, rel=0.05)


def test_prune_removes_short_spur_only():
    s = np.zeros((40, 120), bool)
    s[20, 10:110] = True
    s[14:20, 60] = True                          # branche parasite de 6 px
    pruned = tr.prune(s, min_branch=10)
    assert not pruned[14:19, 60].any() and pruned[20, 10:110].all()    # au plus 1 px reste au pied de la branche


def test_reconnect_bridges_aligned_pieces_only():
    s = np.zeros((40, 200), bool)
    s[20, 10:90] = True
    s[20, 110:190] = True                        # même racine, coupée sur 20 px
    s[5, 120:180] = True                         # autre racine, parallèle
    joined, n = tr.reconnect(s, max_gap=30)
    assert n == 1 and joined[20, 90:110].all() and not joined[6:20, 120:180].any()


def test_errors_on_perfect_prediction():
    t = tr.compare([np.eye(30, dtype=bool)], [np.eye(30, dtype=bool)], ppm=10)
    e = tr.errors(t)
    assert e["longueur : biais"] == pytest.approx(0) and e["longueur : erreur médiane"] == pytest.approx(0)
