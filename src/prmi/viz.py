"""Style commun des figures du projet."""
from pathlib import Path

import matplotlib.pyplot as plt

FIGURES = Path(__file__).resolve().parents[2] / "reports" / "figures"
ACCENT = "#2a78d6"
INK, INK_2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
# Groupes de l'arachide (capteur, site, campagne), toujours dans cet ordre et avec ces couleurs.
GROUPS = ["150 BAZ 2015", "150 BAZ 2016", "150 CZ 2016", "120 EED 2017"]
GROUP_COLORS = dict(zip(GROUPS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "xtick.color": INK_2, "ytick.color": INK_2, "font.size": 10,
    })


def image_grid(images, titles, ncols, width=12):
    """Affiche des images (même taille) en grille, sans axes."""
    nrows = -(-len(images) // ncols)
    h, w = images[0].shape[:2]
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, width * nrows * h / (ncols * w) + 0.3 * nrows),
                             squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, img, title in zip(axes.flat, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=8, loc="left")
    fig.tight_layout()
    return fig


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, dpi=150, bbox_inches="tight")
