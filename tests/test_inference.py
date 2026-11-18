"""Chaîne complète sur les images d'exemple (modèle téléchargé depuis la release au premier lancement)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prmi import cli
from prmi.inference import RootSegmentator, parse_name, summarize

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "images"


def test_parse_prmi_name():
    r = parse_name("Peanut_T019_L046_2015.06.04_104607_BAZ_DPI150")
    assert (r["site"], r["tube"], r["position"], r["date"]) == ("BAZ", 19, 46, "2015-06-04")
    assert r["profondeur_cm"] == pytest.approx(46 * 13.5 * np.sin(np.pi / 4) / 10, abs=0.1)
    assert parse_name("photo_quelconque")["tube"] is None


def test_summarize_totals_by_tube_and_depth():
    t = pd.DataFrame({"site": ["BAZ"] * 3, "tube": [1] * 3, "date": ["2016-06-13"] * 3, "profondeur_cm": [12.0, 15.0, 31.0],
                      "racine": [True, False, True], "longueur_mm": [10.0, 0.0, 5.0], "longueur_etalonnee_mm": [12.0, 0.0, 6.0],
                      "intensite_mm_cm2": [4.0, 0.0, 2.0], "diametre_moyen_mm": [0.3, None, 0.2]})
    s = summarize(t).set_index("profondeur_classe_cm")
    assert s.loc["10-20", "images"] == 2 and s.loc["10-20", "longueur_mm"] == 10.0
    assert s.loc["10-20", "part_avec_racine"] == 0.5 and s.loc["30-40", "diametre_moyen_mm"] == pytest.approx(0.2)


@pytest.fixture(scope="module")
def segmentator():
    return RootSegmentator(device="cpu")


def test_analyze_finds_roots_and_empty_images(segmentator):
    roots = segmentator.analyze_file(EXAMPLES / "Peanut_T019_L083_2016.06.13_130232_BAZ_DPI150.jpg").mesures
    empty = segmentator.analyze_file(EXAMPLES / "Peanut_T018_L064_2015.06.04_104218_BAZ_DPI150.jpg").mesures
    assert roots["racine"] and roots["longueur_mm"] > 10 and 0.1 < roots["diametre_moyen_mm"] < 1
    assert not empty["racine"] and empty["longueur_mm"] == 0


def test_analyze_rejects_unknown_image_size(segmentator):
    with pytest.raises(ValueError):
        segmentator.analyze(np.zeros((100, 100, 3), np.uint8))


def test_command_line_writes_all_outputs(tmp_path):
    cli.main([str(EXAMPLES), str(tmp_path)])
    images = pd.read_csv(tmp_path / "mesures_images.csv")
    assert len(images) == 5 and images["racine"].sum() == 4
    assert (tmp_path / "mesures_tubes.csv").exists()
    assert len(list((tmp_path / "masques").glob("*.png"))) == 5
    assert len(list((tmp_path / "superpositions").glob("*.jpg"))) == 5
