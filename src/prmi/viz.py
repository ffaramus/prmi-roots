"""Style commun des figures du projet."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from prmi.diagnostic import overlay_mask

FIGURES = Path(__file__).resolve().parents[2] / "reports" / "figures"
ACCENT = "#2a78d6"
FALSE_COLOR = "#ff0000"      # fausses détections : rouge vif
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
    fig, axes = plt.subplots(3, 3, figsize=(16, 10))
    panels = [("Perte", ("train_loss", "val_loss")), ("Dice par image", ("train_dice", "dice")),
              ("Dice global (val, images vides comprises)", (None, "global_dice")),
              ("clDice par image (val, lignes centrales)", (None, "cldice")),
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
    ax = axes[2, 2]
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


def show_components(ax, img, labels, comps):
    """Taches prédites sur l'image : vraies détections (touchent la référence) en bleu, fausses en rouge, avec leur
    probabilité moyenne et leur allongement (longueur / largeur)."""
    out = img.copy()
    for _, c in comps.iterrows():
        out = overlay_mask(out, labels == c["label"], color=(30, 100, 255) if c["vraie"] else (255, 0, 0), alpha=0.6)
    ax.imshow(out)
    for _, c in comps.iterrows():
        if c["px"] >= 15:
            ax.annotate(f"{c['p_moy']:.2f} | {c['allongement']:.0f}".replace(".", ","), (c["x"], c["y"]),
                        xytext=(6, -6), textcoords="offset points", fontsize=8, color=INK,
                        bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none", alpha=0.8))


def browse_predictions(items, preds, per_image, raw=False, probs=None):
    """Parcours au curseur : référence à gauche, prédiction à droite, du plus bas Dice au plus haut, images vides à la fin.

    raw : ajoute l'image seule (prétraitée), à gauche. probs : {nom : carte de probabilité} ; ajoute une ligne en
    dessous avec la carte de probabilité (contour du seuil 0,5) et les taches prédites, vraies en bleu, fausses en
    rouge, annotées « probabilité moyenne | allongement ».
    """
    import ipywidgets as widgets
    from prmi.postprocess import components

    by_name = {it["name"]: it for it in items}
    rows = per_image.sort_values(["dice", "pred_px"], ascending=[True, False], na_position="last").reset_index(drop=True)

    def show(rang):
        r = rows.iloc[rang]
        it = by_name[r["name"]]
        img = (it["image"] * 255).astype("uint8")
        score = f"Dice {r['dice']:.2f}" if r["has_root"] else "sans racine"
        panels = [(img, "Image")] if raw else []
        panels += [(overlay_mask(img, it["mask"]), "Référence"), (overlay_mask(img, preds[r["name"]]), "Prédiction")]
        nrows = 2 if probs is not None else 1
        fig, axes = plt.subplots(nrows, len(panels), figsize=(7 * len(panels), 5.6 * nrows), squeeze=False)
        fig.suptitle(f"{rang + 1}/{len(rows)} | {r['group']} | {r['type']} | {score} | {r['pred_px']} px prédits", fontsize=11)
        for ax, (shown, title) in zip(axes[0], panels):
            ax.imshow(shown)
            ax.set_title(title, loc="left", fontsize=10)
        if probs is not None:
            prob = probs[r["name"]]
            from matplotlib.colors import PowerNorm
            im = axes[1, 0].imshow(prob, cmap="Blues", norm=PowerNorm(0.4, vmin=0, vmax=1))   # faibles valeurs visibles
            axes[1, 0].contour(prob, levels=[0.5], colors=[INK], linewidths=0.8)
            fig.colorbar(im, ax=axes[1, 0], fraction=0.035, pad=0.02, ticks=[0, 0.05, 0.2, 0.5, 1])
            axes[1, 0].set_title("Probabilité (échelle accentuée sur les faibles valeurs ; contour : seuil 0,5)",
                                 loc="left", fontsize=10)
            comps, labels = components(preds[r["name"]], prob, it["mask"])
            show_components(axes[1, 1], img, labels, comps)
            n_false = int((~comps["vraie"].astype(bool)).sum()) if len(comps) else 0
            axes[1, 1].set_title(f"Taches : vraies en bleu, fausses en rouge ({n_false}) ; "
                                 "probabilité moyenne | allongement", loc="left", fontsize=10)
            for ax in axes[1, 2:]:
                ax.set_visible(False)
        for ax in axes.flat:
            ax.axis("off")
        fig.tight_layout()
        plt.show()

    widgets.interact(show, rang=widgets.IntSlider(value=0, min=0, max=len(rows) - 1, description="Image",
                                                  continuous_update=False, layout=widgets.Layout(width="80%")))


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, dpi=150, bbox_inches="tight")


# Résultats de segmentation : libellés des métriques (moyennes de `prmi.train.summarize`).
METRIC_LABELS = {"dice": "Dice par image", "global_dice": "Dice global", "cldice": "clDice par image",
                 "global_cldice": "clDice global", "roots_found": "Racines trouvées",
                 "fp_empty": "Images vides touchées", "length_err": "Erreur de longueur (médiane)", "ghosts": "Fantômes",
                 "longueur : biais d'ensemble": "Biais de longueur",
                 "diamètre : biais d'ensemble": "Biais de diamètre"}


def plot_by_amount(tables, colors, metric="dice", bins=10):
    """Score par image selon la quantité de racine de référence (longueur du squelette du masque, déciles) : médiane et
    bande interquartile, une courbe par jeu."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for (name, df), c in zip(tables.items(), colors):
        d = df[df["has_root"] & df[metric].notna()]
        q = pd.qcut(d["skel_ref"], bins, duplicates="drop")
        g = d.groupby(q, observed=True).agg(x=("skel_ref", "median"), med=(metric, "median"),
                                            lo=(metric, lambda v: v.quantile(0.25)), hi=(metric, lambda v: v.quantile(0.75)))
        ax.fill_between(g["x"], g["lo"], g["hi"], color=c, alpha=0.15, lw=0)
        ax.plot(g["x"], g["med"], "-o", color=c, lw=2, ms=6, mec=SURFACE, mew=1.5, label=f"{name} ({len(d)} images)")
    ax.set_xscale("log")
    ticks = [t for t in (50, 100, 200, 500, 1000, 2000, 5000) if ax.get_xlim()[0] <= t <= ax.get_xlim()[1]]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_ylim(0, 1)
    ax.set_xlabel("Longueur de racine de référence dans l'image (px de squelette, échelle log ; déciles)")
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.set_title(f"{METRIC_LABELS[metric]} selon la quantité de racine (médiane et quartiles)", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    return fig


TEST_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]     # trois premiers emplacements de la palette (validés deux à deux)


def plot_test_results(tables, colors=TEST_COLORS):
    """Synthèse des résultats sur des jeux de test : violons des scores par image (images avec racine), puis part des
    images sans racine où le modèle détecte quelque chose (au moins 100 px).

    tables : {nom du jeu : tableau image par image (`prmi.train.predict_items` ou `predict_rows`)}. L'erreur de longueur
    est coupée à 100 % pour l'affichage (la part coupée est indiquée).
    """
    names = list(tables)
    x = np.arange(len(names))
    panels = [("dice", "Dice par image", "recouvrement des surfaces"),
              ("cldice", "clDice par image", "continuité des racines"),
              ("length_err", "Erreur relative de longueur", "squelette brut, non élagué")]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.8), gridspec_kw={"width_ratios": [1, 1, 1, 0.8]})
    for ax, (m, title, sub) in zip(axes, panels):
        data = [t.loc[t["has_root"], m].dropna().clip(upper=1).to_numpy() for t in tables.values()]
        parts = ax.violinplot(data, x, widths=0.8, showextrema=False)
        for body, c in zip(parts["bodies"], colors):
            body.set(facecolor=c, edgecolor=c, alpha=0.35, linewidth=1)
        for xi, d, c in zip(x, data, colors):
            q1, med, q3 = np.percentile(d, [25, 50, 75])
            ax.plot([xi, xi], [q1, q3], color=c, lw=4, solid_capstyle="round")
            ax.plot(xi, med, "o", ms=8, color=SURFACE, mec=INK, mew=1.5)
            ax.annotate(f"{med:.2f}".replace(".", ","), (xi, med), xytext=(9, 0), textcoords="offset points", va="center", fontsize=9,
                        color=INK)
        if m == "length_err":
            for xi, t in zip(x, tables.values()):
                over = (t.loc[t["has_root"], m] > 1).mean()
                if over:
                    ax.annotate(f"{100 * over:.0f} % > 100 %", (xi, 1), xytext=(0, 6), textcoords="offset points",
                                ha="center", fontsize=8, color=INK_2)
        n = [len(d) for d in data]
        ax.set_xticks(x, [f"{name}\n({k} images)" for name, k in zip(names, n)], fontsize=9)
        ax.set_ylim(0, 1.08)
        ax.set_title(f"{title}\n{sub}", loc="left", fontsize=10)
        ax.grid(axis="x", visible=False)
    ax = axes[3]
    empty = [(~t["has_root"]).sum() for t in tables.values()]
    touched = [(t.loc[~t["has_root"], "pred_px"] >= 100).mean() for t in tables.values()]
    ax.bar(x, touched, width=0.6, color=colors)
    for xi, v in zip(x, touched):
        ax.annotate(f"{100 * v:.0f} %", (xi, v), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9, color=INK)
    ax.set_xticks(x, [f"{name}\n({k} images)" for name, k in zip(names, empty)], fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f} %"))
    ax.set_title("Fausses détections\nimages sans racine touchées", loc="left", fontsize=10)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


def _fmt(metric, v):
    """Valeur lisible selon la mesure : nombre entier, pourcentage, biais signé ou score à trois décimales."""
    if metric == "ghosts":
        return f"{v:.0f}"
    if "biais" in metric:
        return f"{100 * v:+.1f} %".replace(".", ",")
    if metric in ("roots_found", "fp_empty"):
        return f"{100 * v:.1f} %".replace(".", ",")
    return f"{v:.3f}".replace(".", ",")


def plot_runs(summary, metrics):
    """Comparaison d'essais : un panneau par métrique, un point par essai (valeur écrite à côté).

    summary : tableau indexé par essai (libellés lisibles), une colonne par métrique (moyennes de `summarize`).
    """
    y = np.arange(len(summary))[::-1]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.2 * len(metrics) + 2.8, 0.5 * len(summary) + 1.3),
                             sharey=True, squeeze=False)
    for ax, m in zip(axes[0], metrics):
        ax.plot(summary[m], y, "o", ms=9, color=ACCENT, mec=SURFACE, mew=1.5)
        for yi, v in zip(y, summary[m]):
            ax.annotate(_fmt(m, v), (v, yi), xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=8, color=INK_2)
        ax.set_title(METRIC_LABELS.get(m, m), loc="left", fontsize=10)
        ax.grid(axis="y", visible=False)
        ax.set_ylim(y.min() - 0.6, y.max() + 0.8)
        ax.margins(x=0.2)
    axes[0, 0].set_yticks(y, summary.index)
    fig.tight_layout()
    return fig


def plot_history_pair(histories, metrics, colors=(ACCENT, VAL_COLOR)):
    """Deux entraînements comparés époque par époque sur le val : un panneau par métrique, une courbe par essai."""
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.5 * len(metrics), 3.6), squeeze=False)
    for ax, m in zip(axes[0], metrics):
        for (name, h), c in zip(histories.items(), colors):
            ax.plot(h["epoch"], h[m], color=c, lw=2, label=name)
        ax.set_title(f"{METRIC_LABELS.get(m, m)} (val)", loc="left", fontsize=10)
        ax.set_xlabel("Époque")
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def show_view(image, mask, pred, title):
    """Image seule, référence et prédiction côte à côte (masques en aplat bleu)."""
    img = (np.clip(image, 0, 1) * 255).astype("uint8") if image.dtype != np.uint8 else image
    fig, axes = plt.subplots(1, 3, figsize=(21, 5.6))
    fig.suptitle(title, fontsize=11)
    for ax, shown, label in zip(axes, (img, overlay_mask(img, mask), overlay_mask(img, pred)),
                                ("Image", "Référence", "Prédiction")):
        ax.imshow(shown)
        ax.set_title(label, loc="left", fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    return fig


def browse_loaded(table, get):
    """Parcours au curseur d'un grand jeu, image chargée à la demande : du plus bas Dice au plus haut, images sans
    racine à la fin (de la plus touchée à la moins touchée).

    table : tableau image par image (`prmi.train.predict_rows`). get(nom) -> (image, référence, prédiction).
    """
    import ipywidgets as widgets

    rows = table.sort_values(["dice", "pred_px"], ascending=[True, False], na_position="last").reset_index(drop=True)

    def show(rang):
        r = rows.iloc[rang]
        score = f"Dice {r['dice']:.2f}" if r["has_root"] else "sans racine"
        where = " | ".join(str(r[c]) for c in ("tube_id", "position", "date") if c in r)
        show_view(*get(r["name"]), f"{rang + 1}/{len(rows)} | {r['group']} | {where} | {r['type']} | {score} | "
                                   f"{r['pred_px']} px prédits")
        plt.show()

    widgets.interact(show, rang=widgets.IntSlider(value=0, min=0, max=len(rows) - 1, description="Image",
                                                  continuous_update=False, layout=widgets.Layout(width="80%")))


def plot_false_detections(blobs, colors=TEST_COLORS):
    """Fausses détections sur les images sans racine : part des images dont la plus grosse tache prédite atteint au
    moins une taille donnée (px), une courbe par jeu. Montre ce qu'un filtre des petites taches retirerait."""
    sizes = np.geomspace(50, 20000, 200)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for (name, b), c in zip(blobs.items(), colors):
        ax.plot(sizes, [(b >= s).mean() for s in sizes], color=c, lw=2, label=f"{name} ({len(b)} images vides)")
    ax.axvline(100, color=INK_2, ls=":", lw=1)
    ax.annotate("seuil de comptage actuel (100 px)", (100, 0.97), xytext=(4, 0), textcoords="offset points",
                fontsize=8, color=INK_2, va="top")
    ax.set_xscale("log")
    ticks = [50, 100, 300, 1000, 3000, 10000]
    ax.set_xticks(ticks, [f"{t:,}".replace(",", " ") for t in ticks])
    ax.minorticks_off()
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f} %"))
    ax.set_xlabel("Taille de la plus grosse tache prédite (px, résolution d'origine, échelle log)")
    ax.set_ylabel("Images sans racine touchées")
    ax.set_title("Fausses détections : images sans racine dont la plus grosse tache dépasse une taille donnée",
                 loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def plot_traits(tables, colors=TEST_COLORS):
    """Longueur et diamètre moyen par image, prédits contre référence (`prmi.train.traits`), échelles log. Le premier
    jeu (petit, référence corrigée) est tracé par-dessus, en gros points."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    panels = [("longueur", "Longueur (px de squelette)"), ("diamètre", "Diamètre moyen (px)")]
    order = list(tables.items())[1:] + list(tables.items())[:1]
    color_of = dict(zip(tables, colors))
    for ax, (k, label) in zip(axes, panels):
        pred_col = f"{k} prédite" if k == "longueur" else f"{k} prédit"
        values = []
        for name, t in order:
            t = t[(t[f"{k} réf."] > 0) & (t[pred_col] > 0)]
            big = name == list(tables)[0]
            ax.plot(t[f"{k} réf."], t[pred_col], "o", ms=7 if big else 2.5, alpha=0.9 if big else 0.25,
                    color=color_of[name], mec=SURFACE if big else "none", mew=1, label=f"{name} ({len(t)} images)")
            values += [t[f"{k} réf."], t[pred_col]]
        lo, hi = np.quantile(pd.concat(values), [0.002, 0.998]) * [0.8, 1.25]    # valeurs extrêmes hors cadre
        ax.plot([lo, hi], [lo, hi], color=INK_2, ls=":", lw=1)
        ax.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi), xlabel=f"{label} : référence",
               ylabel=f"{label} : prédiction")
        ticks = [t for t in (2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000) if lo <= t <= hi]
        ax.set_xticks(ticks, [f"{t:,}".replace(",", " ") for t in ticks])
        ax.set_yticks(ticks, [f"{t:,}".replace(",", " ") for t in ticks])
        ax.minorticks_off()
        ax.set_title(f"{label} par image (sur la diagonale : exact)", loc="left", fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles[-1:] + handles[:-1], labels[-1:] + labels[:-1], frameon=False, fontsize=9,
                   markerscale=1.5, loc="upper left")
    fig.tight_layout()
    return fig


def plot_blobs(blobs):
    """Taches prédites : probabilité moyenne contre surface (échelle log), vraies (touchent la référence) en bleu,
    fausses en rouge."""
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for truth, c, label in ((True, ACCENT, "vraies"), (False, FALSE_COLOR, "fausses")):
        t = blobs[blobs["vraie"].astype(bool) == truth]
        ax.plot(t["p_moy"], t["px"], "o", ms=3.5, color=c, alpha=0.45, mec="none", label=f"{label} ({len(t)})")
    ax.set(yscale="log", xlim=(0.5, 1), xlabel="Probabilité moyenne de la tache (U-Net)",
           ylabel="Surface de la tache (px, échelle log)")
    ax.set_title("Taches prédites : probabilité et taille", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=9, loc="upper left", markerscale=2)
    fig.tight_layout()
    return fig

def blob_gallery(crops, ncols=4):
    """Vignettes de taches : image avec la tache (contour rouge si elle est fausse, noir si elle est vraie) et la
    référence (aplat bleu), puis la carte de probabilité (échelle accentuée sur les faibles valeurs). Deux cases par
    tache."""
    from matplotlib.colors import PowerNorm
    nrows = -(-len(crops) // ncols)
    fig, axes = plt.subplots(nrows, 2 * ncols, figsize=(2.2 * 2 * ncols, 2.5 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for k, c in enumerate(crops):
        a, b = axes[k // ncols, 2 * (k % ncols)], axes[k // ncols, 2 * (k % ncols) + 1]
        a.imshow(overlay_mask(c["image"], c["ref"], alpha=0.35))
        edge = INK if c.get("vraie") else FALSE_COLOR
        a.contour(c["tache"], levels=[0.5], colors=[edge], linewidths=1.5)
        a.set_title(c["titre"], loc="left", fontsize=8)
        b.imshow(c["prob"], cmap="Blues", norm=PowerNorm(0.4, vmin=0, vmax=1))
        b.contour(c["tache"], levels=[0.5], colors=[edge], linewidths=1)
    fig.tight_layout()
    return fig


def plot_tradeoff(curve, points):
    """Compromis d'un filtre des taches selon la longueur de racine de référence perdue : part des taches fausses
    retirées (gauche) et part des images vides encore touchées (droite), en faisant varier le seuil du classifieur
    (courbe) ; d'autres filtres en points (`points` : {libellé : effets}, `postprocess.removal_effects`)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    panels = [("fausses retirées", "Taches fausses retirées (%)"), ("images vides touchées", "Images sans racine touchées (%)")]
    xmax = max(3, 100 * max(r["longueur perdue"] for r in points.values()) * 1.5)
    for ax, (k, label) in zip(axes, panels):
        ax.plot(100 * curve["longueur perdue"], 100 * curve[k], color=ACCENT, lw=2, label="classifieur (seuil variable)")
        for (name, r), c, mk in zip(points.items(), [FALSE_COLOR, INK, "#1baf7a"], ["s", "o", "D"]):
            ax.plot(100 * r["longueur perdue"], 100 * r[k], mk, ms=9, color=c, mec=SURFACE, mew=1.5, label=name)
        ax.set(xlim=(0, xmax), ylim=(0, 100), xlabel="Longueur de racine de référence perdue (%)", ylabel=label)
        ax.set_title(label, loc="left", fontsize=10)
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle("Filtre des taches sur le grand val", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    return fig


def plot_classifier(clf, sets, colors=(INK_2, ACCENT, VAL_COLOR, "#1baf7a")):
    """Classifieur des taches : courbe d'apprentissage (perte logistique à chaque arbre, entraînement et images mises
    de côté pour l'arrêt précoce) et courbes ROC sur chaque ensemble (`sets` : {nom : (étiquettes, scores)})."""
    from sklearn.metrics import roc_auc_score, roc_curve
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axes[0]
    ax.plot(-clf.train_score_, color=INK_2, lw=2, label="entraînement")
    ax.plot(-clf.validation_score_, color=ACCENT, lw=2, label="images mises de côté")
    best = int(np.argmax(clf.validation_score_))
    ax.axvline(best, color=INK_2, ls=":", lw=1)
    ax.annotate(f"meilleur : {best} arbres", (best, ax.get_ylim()[1]), xytext=(4, -12), textcoords="offset points",
                fontsize=8, color=INK_2)
    ax.set(xlabel="Nombre d'arbres", ylabel="Perte logistique")
    ax.set_title("Courbe d'apprentissage", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    ax = axes[1]
    ax.plot([0, 1], [0, 1], color=GRID, lw=1)
    for (name, (y, s)), c in zip(sets.items(), colors):
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, color=c, lw=2, label=f"{name} (AUC {roc_auc_score(y, s):.3f})".replace(".", ","))
    ax.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="Taux de taches fausses gardées", ylabel="Taux de taches vraies gardées")
    ax.set_title("Courbes ROC (taches vraies contre fausses)", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    return fig


def plot_importance(mean, std, title="Importance des descripteurs (perte d'AUC quand on les mélange au hasard)"):
    """Importance par permutation, en barres horizontales triées, avec l'écart-type des répétitions."""
    order = mean.sort_values().index
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(mean) + 1.2))
    ax.barh(order, mean[order], xerr=std[order], color=ACCENT, height=0.6, error_kw=dict(ecolor=INK_2, lw=1))
    for y, (v, e) in enumerate(zip(mean[order], std[order])):
        ax.annotate(f"{v:.3f}".replace(".", ","), (max(v, 0) + e, y), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK_2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Baisse de l'AUC ROC")
    ax.set_title(title, loc="left", fontsize=10)
    fig.tight_layout()
    return fig


def plot_compare(table, metrics, ncols=3, colors=(INK_2, ACCENT)):
    """Comparaison de quelques options (lignes de `table`) : un panneau par mesure, une barre par option, valeur
    écrite au-dessus. Grille de `ncols` panneaux par ligne."""
    nrows = -(-len(metrics) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.6 * nrows), squeeze=False)
    x = np.arange(len(table))
    for ax, m in zip(axes.flat, metrics):
        v = table[m].to_numpy(dtype=float)
        ax.bar(x, v, width=0.55, color=list(colors)[:len(table)])
        ax.axhline(0, color=INK_2, lw=0.8)
        for xi, vi in zip(x, v):
            ax.annotate(_fmt(m, vi), (xi, vi), xytext=(0, 4 if vi >= 0 else -12), textcoords="offset points",
                        ha="center", fontsize=10, color=INK)
        lo, hi = min(v.min(), 0), max(v.max(), 0)
        pad = 0.18 * (hi - lo)
        ax.set_ylim(lo - (pad if lo < 0 else 0), hi + (pad if hi > 0 else 0))
        ax.set_xticks(x, table.index, fontsize=9)
        ax.set_title(METRIC_LABELS.get(m, m), loc="left", fontsize=11)
        ax.grid(axis="x", visible=False)
        if "biais" in m or m in ("roots_found", "fp_empty"):
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda t, _: f"{100 * t:.0f} %"))
    for ax in axes.flat[len(metrics):]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


def plot_compare_sets(results, metrics, ncols=3, colors=(INK_2, ACCENT)):
    """Deux options comparées sur plusieurs jeux : un panneau par mesure, un groupe de barres par jeu (une barre par
    option), valeur écrite au-dessus. results : {jeu : tableau indexé par option, une colonne par mesure}."""
    sets = list(results)
    options = list(results[sets[0]].index)
    nrows = -(-len(metrics) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.8 * nrows), squeeze=False)
    x, w = np.arange(len(sets)), 0.38
    for ax, m in zip(axes.flat, metrics):
        allv = []
        for j, (opt, c) in enumerate(zip(options, colors)):
            v = np.array([results[s].loc[opt, m] for s in sets], dtype=float)
            allv += list(v)
            pos = x + (j - 0.5) * w
            ax.bar(pos, v, width=w, color=c, label=opt)
            for p, vi in zip(pos, v):
                ax.annotate(_fmt(m, vi), (p, vi), xytext=(0, 3 if vi >= 0 else -11), textcoords="offset points",
                            ha="center", fontsize=8, color=INK)
        ax.axhline(0, color=INK_2, lw=0.8)
        lo, hi = min(min(allv), 0), max(max(allv), 0)
        pad = 0.2 * (hi - lo)
        ax.set_ylim(lo - (pad if lo < 0 else 0), hi + (pad if hi > 0 else 0))
        ax.set_xticks(x, [s.replace(" 150 DPI", "\n150 DPI").replace(" 120 DPI", "\n120 DPI") for s in sets], fontsize=9)
        ax.set_title(METRIC_LABELS.get(m, m), loc="left", fontsize=11)
        ax.grid(axis="x", visible=False)
        if "biais" in m or m in ("roots_found", "fp_empty"):
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda t, _: f"{100 * t:.0f} %"))
    for ax in axes.flat[len(metrics):]:
        ax.set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=len(options), frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig
