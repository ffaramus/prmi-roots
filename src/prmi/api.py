"""API web de Root Segmentator (FastAPI).

Lancement : `rootseg-api` (ou `uvicorn prmi.api:app`), puis documentation interactive sur http://localhost:8000/docs.

Points d'accès :
- GET  /health  : le service répond et le modèle est chargé ;
- GET  /version : version du modèle, échelles des capteurs, facteur d'étalonnage de la longueur ;
- POST /predict : une image de minirhizotron en entrée (formulaire, champ `image`), les mesures en JSON en sortie,
  et sur demande le masque des racines (PNG encodé en base64) ;
- POST /predict/superposition : même analyse, réponse = image JPEG avec les racines en bleu.

Le modèle est chargé une seule fois, au démarrage. Les points d'analyse sont des fonctions ordinaires (pas `async`) :
FastAPI les exécute dans un groupe de fils, ce qui évite de bloquer le serveur pendant le calcul.
"""
import base64
import io
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import RedirectResponse, Response
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from prmi import traits, weights
from prmi.inference import SENSORS, RootSegmentator, overlay

MAX_BYTES = 10 * 1024 * 1024          # taille maximale d'une image envoyée (10 Mo)
try:
    PACKAGE_VERSION = version("rootseg")
except PackageNotFoundError:          # code utilisé sans installation du paquet
    PACKAGE_VERSION = "inconnue"


# --- Schémas des réponses (Pydantic : types vérifiés, décrits dans /docs) ---------------------------------------------

class Health(BaseModel):
    status: str = Field(description="« ok » si le service répond")
    modele_charge: bool = Field(description="le modèle est en mémoire, prêt à analyser")


class VersionInfo(BaseModel):
    paquet: str = Field(description="version du paquet rootseg")
    modele: str = Field(description="release GitHub des fichiers du modèle")
    px_par_mm: dict[str, float] = Field(description="échelle de chaque capteur reconnu")
    facteur_longueur: float = Field(description="facteur d'étalonnage de la longueur, à appliquer aux totaux")
    tailles_acceptees: list[str] = Field(description="tailles d'image reconnues (largeur x hauteur, px)")


class Mesures(BaseModel):
    image: str = Field(description="nom du fichier envoyé, sans extension")
    site: str | None = Field(description="site, lu dans le nom PRMI (vide sinon)")
    tube: int | None = Field(description="numéro de tube, lu dans le nom PRMI")
    position: int | None = Field(description="position le long du tube, lue dans le nom PRMI")
    date: str | None = Field(description="date de prise de vue (AAAA-MM-JJ), lue dans le nom PRMI")
    profondeur_cm: float | None = Field(description="profondeur estimée (tube à 45 degrés, pas de 13,5 mm)")
    capteur: str = Field(description="capteur reconnu à la taille de l'image")
    px_par_mm: float = Field(description="échelle utilisée")
    racine: bool = Field(description="au moins une racine détectée")
    longueur_mm: float = Field(description="longueur totale des racines (squelette)")
    longueur_etalonnee_mm: float = Field(description="longueur corrigée du biais moyen, pour des totaux")
    diametre_moyen_mm: float | None = Field(description="diamètre moyen, pondéré par la longueur")
    surface_mm2: float = Field(description="surface de racine projetée")
    surface_image_cm2: float = Field(description="surface d'image observée")
    intensite_mm_cm2: float = Field(description="intensité racinaire : longueur par surface observée")
    taches_retirees: int = Field(description="taches retirées par le classifieur (fausses détections)")


class Prediction(BaseModel):
    mesures: Mesures
    masque_png_base64: str | None = Field(None, description="masque des racines (PNG, 0 ou 255), si demandé")


# --- Application ------------------------------------------------------------------------------------------------------

models = {}


@asynccontextmanager
async def lifespan(app):
    """Au démarrage : chargement du modèle (téléchargé depuis la release au premier lancement)."""
    models["segmentator"] = RootSegmentator()
    yield
    models.clear()


app = FastAPI(title="Root Segmentator", version=PACKAGE_VERSION, lifespan=lifespan,
              description="Segmentation et mesure des racines sur images de minirhizotron (arachide, PRMI). "
                          "Envoyer une image à /predict pour obtenir longueur, diamètre et intensité racinaire en mm.")


def _read_image(upload):
    """Contenu de l'image envoyée, en RGB ; erreur 400 si ce n'est pas une image lisible ou si elle est trop grosse."""
    data = upload.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"image trop volumineuse (plus de {MAX_BYTES // 2**20} Mo)")
    try:
        return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
    except UnidentifiedImageError:
        raise HTTPException(400, "fichier illisible : envoyer une image JPEG, PNG ou TIFF")


def _analyze(upload, classifier):
    rgb = _read_image(upload)
    name = (upload.filename or "image").rsplit(".", 1)[0]
    try:
        return models["segmentator"].analyze(rgb, name=name, use_classifier=classifier)
    except ValueError as e:                                   # taille d'image non reconnue
        raise HTTPException(422, str(e))


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/health", response_model=Health)
def health():
    return Health(status="ok", modele_charge=bool(models))


@app.get("/version", response_model=VersionInfo)
def version_info():
    return VersionInfo(paquet=PACKAGE_VERSION, modele=weights.VERSION,
                       px_par_mm={f"{d} DPI": round(traits.px_per_mm(d), 2) for d in (150, 120)},
                       facteur_longueur=traits.LENGTH_FACTOR,
                       tailles_acceptees=[f"{w} x {h}" for h, w in SENSORS])


@app.post("/predict", response_model=Prediction)
def predict(image: UploadFile = File(description="image de minirhizotron (736 x 552 ou 640 x 480 px)"),
            masque: bool = Query(False, description="joindre le masque des racines (PNG en base64)"),
            classifieur: bool = Query(True, description="retirer les fausses détections (version 1.1)")):
    res = _analyze(image, classifieur)
    png = None
    if masque:
        buf = io.BytesIO()
        Image.fromarray(res.masque.astype(np.uint8) * 255).save(buf, format="PNG", optimize=True)
        png = base64.b64encode(buf.getvalue()).decode()
    return Prediction(mesures=Mesures(**res.mesures), masque_png_base64=png)


@app.post("/predict/superposition", response_class=Response,
          responses={200: {"content": {"image/jpeg": {}}, "description": "image avec les racines en bleu"}})
def predict_overlay(image: UploadFile = File(description="image de minirhizotron"),
                    classifieur: bool = Query(True, description="retirer les fausses détections")):
    res = _analyze(image, classifieur)
    buf = io.BytesIO()
    Image.fromarray(overlay(res.image, res.masque)).save(buf, format="JPEG", quality=90)
    return Response(buf.getvalue(), media_type="image/jpeg")


def main():
    """Commande `rootseg-api` : lance le serveur (http://0.0.0.0:8000 par défaut)."""
    import argparse

    import uvicorn
    p = argparse.ArgumentParser(prog="rootseg-api", description="Serveur web de Root Segmentator")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
