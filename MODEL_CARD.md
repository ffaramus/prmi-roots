# Fiche du modèle : Root Segmentator v1.1

## Usage prévu

Repérer les racines sur des images de minirhizotron d'**arachide** et mesurer leur longueur, leur diamètre et
l'intensité racinaire (longueur par surface observée), pour suivre l'enracinement par tube, date et profondeur.
Utilisateurs visés : laboratoires et équipes d'agronomie qui exploitent des séries d'images de minirhizotron.

**Hors périmètre** sans vérification préalable : autres espèces, autres caméras ou résolutions, mesures sur une image
isolée prises pour exactes (la précision est meilleure sur des totaux).

## Modèle

| Élément | Détail |
|---|---|
| Segmentation | U-Net, encodeur EfficientNet-B0 pré-entraîné sur ImageNet (environ 0,9 million de paramètres utilisés), une sortie : probabilité de racine par pixel |
| Entrée | image RGB prétraitée (recadrage, correction du vignettage, normalisation), réduite de moitié |
| Post-traitement | classifieur des taches prédites (gradient boosting, 13 descripteurs de taille, probabilité, forme et contexte) qui retire les fausses détections ; seuil 0,24 |
| Mesures | longueur du squelette, diamètre (deux fois la distance au bord le long du squelette), échelle 40,9 px/mm (736 x 552 px) ou 35,6 px/mm (640 x 480 px) |
| Fichiers | [release v1.1](https://github.com/ffaramus/prmi-roots/releases/tag/v1.1) (U-Net, classifieur, cartes de fond), empreintes vérifiées au téléchargement |

## Données d'entraînement

Jeu public PRMI (arachide, 150 DPI), découpage officiel par tube :

- **U-Net** : 11 485 images du train (149 corrigées à la main, les autres avec leurs annotations d'origine aux fentes
  comblées automatiquement) ; arrêt de l'apprentissage sur 74 images de validation corrigées à la main.
- **Classifieur des taches** : 2 273 images du val officiel (apprentissage), 1 000 autres (arrêt et choix du seuil).
- Aucune image de test n'a servi à entraîner ou à régler quoi que ce soit.

## Performances

Sur des tubes jamais vus (test PRMI), mesuré une seule fois :

| | Test corrigé à la main (50 images) | Grand test 150 DPI (4 743) | Autre caméra, 120 DPI (3 542) |
|---|---|---|---|
| Dice médian par image | 0,79 | 0,74 | 0,79 |
| clDice médian (continuité des racines) | 0,84 | 0,85 | 0,88 |
| Racines trouvées | 80 % | 67 % | 85 % |
| Images sans racine faussement détectées | 0 % | 13 % | 20 % |

Mesures sur le test corrigé (32 images avec racine) :

| | Erreur médiane par image | Biais d'ensemble |
|---|---|---|
| Longueur, brute | 21 % | -23 % |
| Longueur, étalonnée (x 1,2, facteur fixé sur la validation) | 23 % | -7,5 % |
| Diamètre moyen | 15 % | +24 % (racines du test plus fines ; +2 % sur la validation) |

Vitesse : environ 360 images par minute sur processeur de portable (16 fils), lecture et écriture comprises.

## Limites et biais connus

- Les racines fines ou peu visibles sont en partie manquées : la longueur est sous-estimée, et la part manquée varie
  d'un jeu d'images à l'autre.
- Le diamètre des racines de moins de 0,25 mm environ est surestimé (prédiction à demi-résolution).
- Le classifieur des taches a appris sur des annotations d'origine qui oublient des racines faibles : il peut retirer
  quelques fragments de vraies racines, surtout sur des images presque vides.
- Échelle déduite de la taille du cadre de la caméra (incertitude d'environ 4 %), le DPI des métadonnées PRMI n'étant
  pas l'échelle réelle.
- Références des grands jeux de test : annotations d'origine, moins fiables que le test corrigé à la main ; un seul
  annotateur pour les corrections.
