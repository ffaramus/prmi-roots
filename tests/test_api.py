"""API web : santé, version, analyse d'une image, erreurs explicites."""
import base64
import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from prmi.api import app

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "images"
ROOTS = EXAMPLES / "Peanut_T019_L083_2016.06.13_130232_BAZ_DPI150.jpg"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:                 # « with » : lance le démarrage (chargement du modèle)
        yield c


def test_health_and_version(client):
    assert client.get("/health").json() == {"status": "ok", "modele_charge": True}
    v = client.get("/version").json()
    assert v["modele"] == "v1.1" and v["px_par_mm"]["150 DPI"] == pytest.approx(40.89, abs=0.01)


def test_predict_returns_measures_and_mask(client):
    with open(ROOTS, "rb") as f:
        r = client.post("/predict", files={"image": (ROOTS.name, f, "image/jpeg")}, params={"masque": True})
    assert r.status_code == 200
    m = r.json()["mesures"]
    assert m["racine"] and m["tube"] == 19 and m["longueur_mm"] > 10 and m["capteur"] == "150 DPI"
    mask = Image.open(io.BytesIO(base64.b64decode(r.json()["masque_png_base64"])))
    assert mask.size == (736, 552)


def test_overlay_is_a_jpeg(client):
    with open(ROOTS, "rb") as f:
        r = client.post("/predict/superposition", files={"image": (ROOTS.name, f, "image/jpeg")})
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"


def test_clear_errors(client):
    r = client.post("/predict", files={"image": ("notes.txt", b"pas une image", "text/plain")})
    assert r.status_code == 400
    buf = io.BytesIO()
    Image.new("RGB", (100, 100)).save(buf, format="PNG")
    r = client.post("/predict", files={"image": ("petite.png", buf.getvalue(), "image/png")})
    assert r.status_code == 422 and "non reconnue" in r.json()["detail"]
