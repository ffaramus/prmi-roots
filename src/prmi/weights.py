"""Fichiers du modèle publiés dans une release GitHub, téléchargés au premier usage et gardés en cache.

Trois fichiers par version :
- U-Net (poids, `torch.save` du state_dict) ;
- classifieur des taches et son seuil (`joblib`, scikit-learn 1.8) ;
- cartes de fond par capteur (« 150 » : moyenne des trois groupes 150 DPI ; « 120 » : groupe EED 2017), pour corriger
  le vignettage d'images dont on ne connaît pas le groupe.

Cache : variable d'environnement ROOTSEG_CACHE, sinon ~/.cache/rootseg/<version>. Pour travailler sans réseau, mettre
les trois fichiers dans ROOTSEG_WEIGHTS (prioritaire). Préparer les fichiers d'une nouvelle release (depuis la racine
du projet) : `PYTHONPATH=src python -m prmi.weights`.
"""
import hashlib
import os
import urllib.request
from pathlib import Path

VERSION = "v1.1"
REPO = "ffaramus/rootseg"
FILES = {"unet": "rootseg-unet.pt", "classifier": "rootseg-classifier.joblib", "flatfields": "rootseg-flatfields.npz"}
SHA256 = {             # empreintes des fichiers publiés (sortie de `python -m prmi.weights`)
    "unet": "49b4dcb72f9be433ca9b39948ba75ff61ab5030b726191c74878fc9320c0a6f5",
    "classifier": "2e6dc5f3e25172147765f6f16bd9e008c9ba32d415fb0b891800df8d3c331122",
    "flatfields": "9ee4ef728704b6d5884a05ea99bf89ebe15e597465861fa001edcdf318f041bc",
}


def cache_dir():
    if os.environ.get("ROOTSEG_WEIGHTS"):
        return Path(os.environ["ROOTSEG_WEIGHTS"])
    return Path(os.environ.get("ROOTSEG_CACHE", Path.home() / ".cache" / "rootseg")) / VERSION


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(key):
    """Chemin local du fichier `key` (unet, classifier, flatfields), téléchargé depuis la release si absent."""
    path = cache_dir() / FILES[key]
    if not path.exists():
        url = f"https://github.com/{REPO}/releases/download/{VERSION}/{FILES[key]}"
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Téléchargement de {url}")
        tmp = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(path)
    if SHA256.get(key) and _sha256(path) != SHA256[key]:
        raise RuntimeError(f"{path} : empreinte inattendue (fichier corrompu ou d'une autre version) ; le supprimer")
    return path


def build_release(out=None, run="unet_effb0_tout"):
    """Écrit les trois fichiers de la release dans `out` (par défaut data/processed/release/<version>) et renvoie
    leurs empreintes. Depuis les fichiers d'entraînement du projet (data/, non publié)."""
    import shutil

    import numpy as np

    from prmi.preprocess import load_flatfields

    root = Path(__file__).resolve().parents[2] / "data" / "processed"
    out = Path(out) if out else root / "release" / VERSION
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(root / "runs" / run / "best.pt", out / FILES["unet"])
    shutil.copyfile(root / "runs" / run / "classifieur_taches.joblib", out / FILES["classifier"])
    flats = load_flatfields()
    per_sensor = {"150": np.mean([v for k, v in flats.items() if k.startswith("150")], axis=0).astype(np.float32),
                  "120": flats["120 EED 2017"].astype(np.float32)}
    np.savez_compressed(out / FILES["flatfields"], **per_sensor)
    return {k: _sha256(out / f) for k, f in FILES.items()}


if __name__ == "__main__":
    for k, h in build_release().items():
        print(f'"{k}": "{h}",')
