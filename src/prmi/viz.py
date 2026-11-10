"""Style commun des figures du projet."""
from pathlib import Path

import matplotlib.pyplot as plt

from prmi.diagnostic import overlay_mask

FIGURES = Path(__file__).resolve().parents[2] / "reports" / "figures"
ACCENT = "#2a78d6"
VAL_COLOR = "#eb6834"
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


def plot_history(history, best_epoch):
    """Suivi époque par époque : perte, Dice, racines retrouvées, longueur, fausses détections, taux d'apprentissage."""
    train_style, val_style = dict(color=ACCENT, label="train"), dict(color=VAL_COLOR, linestyle="--", label="val")
    fig, axes = plt.subplots(2, 4, figsize=(19, 7))
    panels = [("Perte", ("train_loss", "val_loss")), ("Dice par image", ("train_dice", "dice")),
              ("Dice global (val, images vides comprises)", (None, "global_dice")),
              ("Racines retrouvées (val, part)", (None, "roots_found")),
              ("Erreur relative de longueur (val, médiane)", (None, "length_err")),
              ("Images vides avec fausse détection (val, part)", (None, "fp_empty")),
              ("Taches fantômes (val, nombre)", (None, "ghosts"))]
    x = history["epoch"]
    for ax, (title, (tr, val)) in zip(axes.flat, panels):
        if val not in history:      # colonne absente des essais antérieurs
            ax.axis("off")
            continue
        if tr:
            ax.plot(x, history[tr], **train_style)
        ax.plot(x, history[val], **val_style)
        ax.axvline(best_epoch, color=INK_2, linestyle=":", label=f"meilleure époque : {best_epoch}")
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlabel("Époque")
        ax.legend(fontsize=8)
    ax = axes[1, 3]
    ax.plot(x, history["lr"], color=INK_2, label="décodeur")
    if "lr_encoder" in history:     # absent des entraînements à taux unique
        ax.plot(x, history["lr_encoder"], color=INK_2, linestyle="--", label="encodeur")
        ax.legend(fontsize=8)
    ax.set_yscale("log")
    ax.set_title("Taux d'apprentissage", loc="left", fontsize=10)
    ax.set_xlabel("Époque")
    fig.tight_layout()
    return fig


def plot_dice(per_image):
    """Répartition du Dice par image, sur les images avec racine."""
    dice = per_image["dice"].dropna()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(dice, bins=20, range=(0, 1), color=ACCENT, edgecolor=SURFACE)
    ax.axvline(dice.median(), color=INK_2, linestyle=":", label=f"médiane : {dice.median():.2f}")
    ax.set_title(f"Dice par image ({len(dice)} images avec racine)", loc="left", fontsize=10)
    ax.set_xlabel("Dice")
    ax.set_ylabel("Images")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def browse_predictions(items, preds, per_image):
    """Parcours au curseur : référence à gauche, prédiction à droite, du plus bas Dice au plus haut, images vides à la fin."""
    import ipywidgets as widgets

    by_name = {it["name"]: it for it in items}
    rows = per_image.sort_values(["dice", "pred_px"], ascending=[True, False], na_position="last").reset_index(drop=True)

    def show(rang):
        r = rows.iloc[rang]
        it = by_name[r["name"]]
        img = (it["image"] * 255).astype("uint8")
        score = f"Dice {r['dice']:.2f}" if r["has_root"] else "sans racine"
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
        fig.suptitle(f"{rang + 1}/{len(rows)} | {r['group']} | {r['type']} | {score} | {r['pred_px']} px prédits", fontsize=11)
        for ax, mask, title in zip(axes, (it["mask"], preds[r["name"]]), ("Référence", "Prédiction")):
            ax.imshow(overlay_mask(img, mask))
            ax.set_title(title, loc="left", fontsize=10)
            ax.axis("off")
        fig.tight_layout()
        plt.show()

    widgets.interact(show, rang=widgets.IntSlider(value=0, min=0, max=len(rows) - 1, description="Image",
                                                  continuous_update=False, layout=widgets.Layout(width="80%")))


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, dpi=150, bbox_inches="tight")
