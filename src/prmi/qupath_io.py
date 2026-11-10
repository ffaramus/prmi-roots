"""Échange avec QuPath : export des images du lot et de leurs masques d'origine, à corriger dans QuPath.

Sortie, un dossier par lot (train, val, test) :
    data/interim/qupath/<lot>/<nom de l'image>/image.jpg   copie de l'image d'origine (JPEG, sans recompression)
    data/interim/qupath/<lot>/<nom de l'image>/mask.png    masque d'origine, un canal, 0 ou 255 (vide si sans racine)
    data/interim/qupath/<lot>/lot.csv                      métadonnées (groupe, type, position...) et colonne `minutes`

Un lot déjà exporté n'est pas réécrit : le projet QuPath pointe sur ces fichiers.

Usage (depuis la racine du projet) : PYTHONPATH=src python -m prmi.qupath_io
"""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from prmi.annotation_pool import POOL
from prmi.inventory import RAW, ROOT

QUPATH_DIR = ROOT / "data" / "interim" / "qupath"
META = ["name", "group", "type", "has_root", "root_frac", "tube_id", "date", "position", "split", "image_path"]


def export_lot(pool, lot, out=QUPATH_DIR, raw=RAW):
    """Exporte les images d'un lot et leurs masques d'origine. Renvoie le dossier du lot, ou None s'il existe déjà."""
    lot_dir = out / lot
    if lot_dir.exists():
        print(f"{lot} : déjà exporté ({lot_dir.relative_to(ROOT)}), non réécrit")
        return None
    rows = pool[pool["lot"] == lot].assign(name=lambda d: d["image_path"].map(lambda p: Path(p).stem))
    for _, r in rows.iterrows():
        d = lot_dir / r["name"]
        d.mkdir(parents=True)
        shutil.copyfile(raw / r["image_path"], d / "image.jpg")
        mask = np.asarray(Image.open(raw / r["mask_path"]))[..., 0] > 0
        Image.fromarray(mask.astype(np.uint8) * 255).save(d / "mask.png")
    rows[META].assign(minutes=np.nan).to_csv(lot_dir / "lot.csv", index=False)
    print(f"{lot} : {len(rows)} images -> {lot_dir.relative_to(ROOT)}")
    return lot_dir


def export_pool(pool, lots=("train", "val", "test"), out=QUPATH_DIR):
    return [export_lot(pool, lot, out) for lot in lots]


if __name__ == "__main__":
    export_pool(pd.read_csv(POOL))
