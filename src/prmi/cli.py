"""Commande `rootseg` : analyse un dossier d'images de minirhizotron et écrit masques, superpositions et mesures.

    rootseg images/ resultats/

Sorties dans le dossier de résultats :
- masques/<image>.png : masque des racines (0 ou 255) ;
- superpositions/<image>.jpg : image prétraitée avec les racines en bleu ;
- mesures_images.csv : une ligne par image (tube, position, date et profondeur lus dans le nom PRMI, longueur, diamètre,
  surface, intensité racinaire) ;
- mesures_tubes.csv : totaux par site, tube, date et classe de profondeur de 10 cm.
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from PIL import Image

EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def find_images(paths):
    """Images des chemins donnés (fichiers ou dossiers, sans descendre dans les sous-dossiers), triées."""
    found = []
    for p in map(Path, paths):
        if p.is_dir():
            found += sorted(f for f in p.iterdir() if f.suffix.lower() in EXTENSIONS)
        elif p.suffix.lower() in EXTENSIONS:
            found.append(p)
    return found


def main(argv=None):
    parser = argparse.ArgumentParser(prog="rootseg", description="Root Segmentator : segmentation et mesure des "
                                     "racines sur images de minirhizotron (arachide, PRMI).")
    parser.add_argument("entrees", nargs="+", help="images ou dossiers d'images")
    parser.add_argument("sortie", help="dossier de résultats (créé si besoin)")
    parser.add_argument("--sans-classifieur", action="store_true", help="garder toutes les taches prédites par le U-Net")
    parser.add_argument("--sans-images", action="store_true", help="n'écrire que les tableaux de mesures")
    parser.add_argument("--px-par-mm", type=float, help="échelle à imposer (par défaut : celle du capteur)")
    parser.add_argument("--profondeur-classe", type=float, default=10, help="largeur des classes de profondeur (cm)")
    parser.add_argument("--device", help="cpu ou cuda (par défaut : cuda si disponible)")
    args = parser.parse_args(argv)

    images = find_images(args.entrees)
    if not images:
        sys.exit("Aucune image trouvée (formats acceptés : " + ", ".join(sorted(EXTENSIONS)) + ").")
    out = Path(args.sortie)
    if not args.sans_images:
        (out / "masques").mkdir(parents=True, exist_ok=True)
        (out / "superpositions").mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    from prmi.inference import RootSegmentator, overlay, summarize      # import tardif : `rootseg -h` reste rapide
    seg = RootSegmentator(device=args.device, use_classifier=not args.sans_classifieur)
    rows, failed, t0 = [], [], time.perf_counter()
    for k, path in enumerate(images, start=1):
        try:
            res = seg.analyze_file(path, px_per_mm=args.px_par_mm)
        except ValueError as e:
            failed.append(str(e))
            continue
        rows.append(res.mesures)
        if not args.sans_images:
            Image.fromarray(res.masque.astype("uint8") * 255).save(out / "masques" / f"{path.stem}.png", optimize=True)
            Image.fromarray(overlay(res.image, res.masque)).save(out / "superpositions" / f"{path.stem}.jpg", quality=90)
        print(f"\r{k}/{len(images)} images", end="", flush=True)
    print()
    for msg in failed:
        print("Ignorée :", msg)
    if not rows:
        sys.exit("Aucune image analysée.")
    table = pd.DataFrame(rows)
    table.to_csv(out / "mesures_images.csv", index=False)
    summarize(table, args.profondeur_classe).to_csv(out / "mesures_tubes.csv", index=False)
    dt = time.perf_counter() - t0
    print(f"{len(rows)} images analysées en {dt:.0f} s ({60 * len(rows) / dt:.0f} par minute) -> {out}")


if __name__ == "__main__":
    main()
