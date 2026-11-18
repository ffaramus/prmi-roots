"""Démo Streamlit : la page s'exécute sans erreur sur une image d'exemple et affiche les mesures."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "demo" / "app.py"


def test_demo_runs_on_an_example():
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception
    assert [m.label for m in at.metric][:2] == ["Longueur (mm)", "Diamètre moyen (mm)"]
    assert len(at.dataframe) >= 1


def test_demo_without_classifier():
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    at.sidebar.checkbox[0].uncheck().run()
    assert not at.exception
