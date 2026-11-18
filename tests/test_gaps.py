"""Comblement des fentes de l'annotation PRMI : relier des morceaux bout à bout, jamais deux racines côte à côte."""
import numpy as np
from scipy import ndimage as ndi

from prmi.gaps import fill_gaps


def n_pieces(mask):
    return ndi.label(mask, structure=np.ones((3, 3)))[1]


def test_gap_between_aligned_rectangles_is_filled():
    m = np.zeros((60, 200), bool)
    m[26:34, 10:95] = True                       # deux rectangles bout à bout, fente de 4 px
    m[26:34, 99:190] = True
    assert n_pieces(m) == 2
    assert n_pieces(fill_gaps(m)) == 1


def test_parallel_roots_are_not_joined():
    m = np.zeros((60, 200), bool)
    m[20:28, 10:190] = True                      # deux racines parallèles, à 4 px l'une de l'autre
    m[32:40, 10:190] = True
    assert n_pieces(fill_gaps(m)) == 2


def test_width_is_unchanged_away_from_gaps():
    m = np.zeros((60, 200), bool)
    m[26:34, 10:95] = True
    m[26:34, 99:190] = True
    filled = fill_gaps(m)
    assert filled[:, 20:85].sum() == m[:, 20:85].sum()     # pas d'élargissement le long des rectangles
