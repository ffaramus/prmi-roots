"""Démo en ligne de Root Segmentator (Streamlit) : déposer une image de minirhizotron, voir les racines et les mesures.

Lancement local : `streamlit run demo/app.py`. Le modèle est appelé directement (sans l'API), chargé une seule fois.
"""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))           # hébergement sans installation du paquet

from prmi.inference import RootSegmentator, overlay, summarize  # noqa: E402

EXAMPLES = ROOT / "examples" / "images"
EXAMPLE_NAMES = {"Racines denses": "Peanut_T019_L083_2016.06.13_130232_BAZ_DPI150.jpg",
                 "Racines fines": "Peanut_T019_L046_2015.06.04_104607_BAZ_DPI150.jpg",
                 "Image sombre": "Peanut_T019_L077_2016.07.25_170233_BAZ_DPI150.jpg",
                 "Autre caméra (120 DPI)": "Peanut_T026_L049_2017.07.27_123345_EED_DPI120.jpg",
                 "Sans racine": "Peanut_T018_L064_2015.06.04_104218_BAZ_DPI150.jpg"}
REPO = "https://github.com/ffaramus/prmi-roots"
LABELS = {"longueur_mm": "Longueur (mm)", "diametre_moyen_mm": "Diamètre moyen (mm)",
          "intensite_mm_cm2": "Intensité racinaire (mm/cm²)", "taches_retirees": "Fausses détections retirées"}

st.set_page_config(page_title="Root Segmentator", layout="wide")


@st.cache_resource(show_spinner="Chargement du modèle (première visite : environ une minute)...")
def load_model():
    return RootSegmentator(device="cpu")


st.title("Root Segmentator")
st.write("Segmentation et mesure des racines sur images de **minirhizotron** (arachide, jeu de données public PRMI). "
         "Déposez une image : le modèle repère les racines et donne leur longueur, leur diamètre et l'intensité "
         f"racinaire en millimètres. Code, méthode et résultats : [dépôt GitHub]({REPO}).")
st.caption("Démo gratuite : si personne ne l'a utilisée récemment, elle met environ une minute à se réveiller.")

with st.sidebar:
    st.header("Images")
    uploads = st.file_uploader("Vos images (736 x 552 ou 640 x 480 px)", type=["jpg", "jpeg", "png", "tif", "tiff"],
                               accept_multiple_files=True)
    example = st.selectbox("ou une image d'exemple (test PRMI)", ["(aucune)"] + list(EXAMPLE_NAMES), index=1)
    classifier = st.checkbox("Retirer les fausses détections", value=True,
                             help="Classifieur des taches prédites (version 1.1). Décocher : U-Net seul.")

inputs = [(Path(u.name).stem, Image.open(u).convert("RGB")) for u in uploads or []]
if not inputs and example != "(aucune)":
    file = EXAMPLE_NAMES[example]
    inputs = [(Path(file).stem, Image.open(EXAMPLES / file).convert("RGB"))]
if not inputs:
    st.info("Déposez une image ou choisissez un exemple dans le panneau de gauche.")
    st.stop()

model = load_model()
rows = []
for name, img in inputs:
    try:
        res = model.analyze(np.asarray(img), name=name, use_classifier=classifier)
    except ValueError as e:
        st.error(str(e))
        continue
    rows.append(res.mesures)
    m = res.mesures
    st.subheader(name)
    left, right = st.columns(2)
    left.image(res.image, caption="Image (après correction de l'éclairage)", width="stretch")
    right.image(overlay(res.image, res.masque), caption="Racines détectées (en bleu)", width="stretch")
    cols = st.columns(4)
    cols[0].metric(LABELS["longueur_mm"], f"{m['longueur_mm']:.1f}")
    cols[1].metric(LABELS["diametre_moyen_mm"], f"{m['diametre_moyen_mm']:.2f}" if m["diametre_moyen_mm"] else "aucune racine")
    cols[2].metric(LABELS["intensite_mm_cm2"], f"{m['intensite_mm_cm2']:.1f}")
    cols[3].metric(LABELS["taches_retirees"], m["taches_retirees"])
    buf = io.BytesIO()
    Image.fromarray(res.masque.astype(np.uint8) * 255).save(buf, format="PNG")
    st.download_button("Télécharger le masque (PNG)", buf.getvalue(), f"{name}_masque.png", "image/png", key=f"m_{name}")

if rows:
    table = pd.DataFrame(rows)
    st.subheader("Mesures")
    st.dataframe(table, hide_index=True, width="stretch")
    st.download_button("Télécharger les mesures (CSV)", table.to_csv(index=False), "mesures_images.csv", "text/csv")
    if len(rows) > 1:
        st.dataframe(summarize(table), hide_index=True, width="stretch")

with st.expander("Précision et limites"):
    st.markdown("""
- **Évaluée sur des tubes jamais vus** (test PRMI) : Dice médian de 0,74 à 0,79 par image, y compris sur une autre caméra.
- **Fausses détections** : 0 à 20 % des images sans racine touchées, selon le jeu de test.
- **Longueur** : sous-estimée d'environ 20 % (racines fines ou peu visibles non détectées) ; la colonne
  « longueur étalonnée » corrige ce biais moyen pour des totaux par tube ou par date, pas pour une image seule.
- **Diamètre** : juste à quelques % près pour des racines d'au moins 0,25 mm environ, surestimé pour les plus fines.
- **Échelle** : 40,9 px/mm (150 DPI) et 35,6 px/mm (120 DPI), vérifiées sur les données ; le « DPI » des
  métadonnées PRMI n'est pas l'échelle réelle.
- **Domaine** : images d'arachide des capteurs PRMI (736 x 552 ou 640 x 480 px) ; d'autres espèces ou caméras
  demanderaient une vérification.
""")
