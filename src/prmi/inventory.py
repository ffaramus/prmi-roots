"""Inventaire du jeu PRMI : une ligne par image, en un seul parcours de la clé.

Les métadonnées viennent des JSON `labels_image_gt`. Seuls les masques des images
marquées `has_root=1` sont ouverts pour calculer la part de pixels racine ; pour les
autres, elle vaut 0 (hypothèse vérifiable avec `check_empty_masks`).

Usage : python src/prmi/inventory.py
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
INVENTORY = ROOT / "data" / "interim" / "inventory.parquet"
SPLITS = ("train", "val", "test")


def read_metadata(raw=RAW):
    """Lit les JSON de métadonnées et liste une fois les dossiers de masques."""
    rows = []
    for split in SPLITS:
        for f in sorted((raw / split / "labels_image_gt").glob("*.json")):
            folder = f.stem.removesuffix(f"_{split}")
            mask_dir = raw / split / "masks_pixel_gt" / folder
            masks = set(os.listdir(mask_dir)) if mask_dir.is_dir() else set()
            for r in json.loads(f.read_text()):
                # Nom du masque d'après le README : le champ binary_mask vaut parfois "N/A"
                # alors que le fichier existe (38 tuiles switchgrass du val).
                mask_name = f"GT_{Path(r['image_name']).stem}.png"
                mask_present = mask_name in masks
                rows.append({
                    "split": split,
                    "folder": folder,
                    "species": r["crop"],
                    "dpi": int(r["dpi"]),
                    "site": r["location"],
                    "tube": r["tube_num"],
                    "position": int(r["depth"].lstrip("L")),
                    "date": r["date"],
                    "time": r["image_name"].split("_")[4],
                    "row": r.get("row_idx"),
                    "col": r.get("col_idx"),
                    "has_root": bool(int(r["has_root"])),
                    "mask_present": mask_present,
                    "image_path": f"{split}/images/{folder}/{r['image_name']}",
                    "mask_path": f"{split}/masks_pixel_gt/{folder}/{mask_name}" if mask_present else None,
                })
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], format="%Y.%m.%d")
    df[["row", "col"]] = df[["row", "col"]].astype("Int64")
    # Un numéro de tube n'est unique qu'au sein d'une espèce et d'un site.
    df["tube_id"] = df["species"] + "_" + df["site"] + "_" + df["tube"]
    return df


def root_fraction(mask_path, raw=RAW):
    """Part des pixels racine (masque RGBA, R = G = B dans {0, 255})."""
    return float((np.asarray(Image.open(raw / mask_path))[..., 0] > 0).mean())


def add_root_fraction(df, raw=RAW, workers=8):
    """Ajoute `root_frac` : calculée si has_root, 0 sinon, NaN sans masque."""
    df = df.copy()
    df["root_frac"] = np.where(df["mask_present"], 0.0, np.nan)
    todo = df.index[df["has_root"] & df["mask_present"]]
    with ThreadPoolExecutor(workers) as ex:
        fracs = list(tqdm(ex.map(lambda p: root_fraction(p, raw), df.loc[todo, "mask_path"]),
                          total=len(todo), desc="masques", unit="img"))
    df.loc[todo, "root_frac"] = fracs
    return df


def check_empty_masks(df, n=200, seed=0, raw=RAW):
    """Ouvre n masques has_root=0 tirés au hasard et renvoie leur part de racine."""
    sample = df[~df["has_root"] & df["mask_present"]].sample(n, random_state=seed)
    return sample.assign(root_frac=[root_fraction(p, raw) for p in tqdm(sample["mask_path"], desc="contrôle")])


def build_inventory(raw=RAW, out=INVENTORY):
    if not raw.exists():
        raise FileNotFoundError(f"{raw} introuvable : la clé USB est-elle branchée ?")
    df = add_root_fraction(read_metadata(raw), raw)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return df


def load_inventory(path=INVENTORY):
    return pd.read_parquet(path)


if __name__ == "__main__":
    df = build_inventory()
    print(f"{len(df)} images -> {INVENTORY}")
