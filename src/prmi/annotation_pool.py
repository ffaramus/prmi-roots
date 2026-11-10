"""Images à annoter : tirage dans les images déjà pré-annotées (train, val et test, 150 DPI).

Train et val, tirage stratifié (quotas par groupe, voir QUOTAS) :
- des cas difficiles avec racine (dominante orange, grise, saturée, faible contraste, sombre), sur-représentés ;
- des cas courants avec racine, répartis entre profondeur (au prorata) et densité de racine (tiers égaux) ;
- des images sans racine, à la proportion réelle de chaque groupe : une partie tirée avec le reste du lot (QUOTAS),
  une partie ajoutée ensuite (EXTRA_EMPTY), au hasard, donc à la répartition réelle par date, profondeur et tube.
  Les images ajoutées sont tirées en dernier pour ne rien changer au reste du tirage.
Test : 50 images tirées au hasard dans chaque groupe, au prorata de la taille des groupes (représentatif de la
réalité), sans quota de type ni plafond par tube (un plafond fausserait la part des groupes : BAZ 2015 n'a que 6 tubes
dans le test).
Train et val : au plus CAPS images par tube et par campagne (sauf pour les images sans racine ajoutées).

Usage (depuis la racine du projet) : PYTHONPATH=src python -m prmi.annotation_pool
"""
from collections import Counter

import numpy as np
import pandas as pd

from prmi.image_stats import HARD_TYPES, image_type, load_image_stats
from prmi.inventory import ROOT

POOL = ROOT / "data" / "processed" / "annotation_pool.csv"
GROUPS = ["150 BAZ 2015", "150 BAZ 2016", "150 CZ 2016"]
DEPTHS = [0, 35, 70, 200]          # haut, milieu, bas (positions dans le tube)
CAPS = {"train": 4, "val": 6}      # images par tube et par campagne, au plus (le val n'a que 4 tubes par campagne)
N_TEST = 50
QUOTAS = {
    "train": {   # 85 images avec racine
        "150 BAZ 2015": {"vide": 7, "orangée": 5, "saturée": 4, "faible contraste": 2, "sombre": 2, "typique": 25},
        "150 BAZ 2016": {"vide": 6, "grise": 5, "saturée": 2, "faible contraste": 2, "sombre": 3, "typique": 22},
        "150 CZ 2016": {"vide": 2, "faible contraste": 1, "sombre": 2, "typique": 10},
    },
    "val": {     # 42 images avec racine ; servent à toutes les décisions
        "150 BAZ 2015": {"vide": 3, "orangée": 2, "saturée": 2, "faible contraste": 1, "sombre": 1, "typique": 11},
        "150 BAZ 2016": {"vide": 3, "grise": 3, "saturée": 1, "faible contraste": 1, "sombre": 1, "typique": 9},
        "150 CZ 2016": {"vide": 2, "faible contraste": 1, "sombre": 2, "typique": 7},
    },
}
# Images sans racine ajoutées, pour atteindre la proportion réelle par groupe (train : 51, 32, 39 % ; val : 54, 31, 32 %).
# Total sans racine : train 40, 16, 8 (149 images) ; val 20, 7, 5 (74 images).
EXTRA_EMPTY = {
    "train": {"150 BAZ 2015": 33, "150 BAZ 2016": 10, "150 CZ 2016": 6},
    "val": {"150 BAZ 2015": 17, "150 BAZ 2016": 4, "150 CZ 2016": 3},
}


def split_quota(n, weights, floor=0):
    """Répartit n entre des cases, au prorata des poids, avec un minimum par case (méthode du plus fort reste)."""
    w = np.asarray(weights, float) / np.sum(weights)
    q = np.maximum(np.floor(n * w).astype(int), floor)
    while q.sum() > n:
        q[np.argmax(q - n * w)] -= 1
    while q.sum() < n:
        q[np.argmax(n * w - q)] += 1
    return q


def pick(pop, n, rng, taken, cap):
    """n images tirées au hasard dans pop, au plus `cap` par tube et par campagne (assoupli d'un cran si la population
    est trop pauvre). cap=None : sans plafond."""
    chosen = []
    caps = [np.inf] if cap is None else range(cap, cap + 4)
    for c in caps:
        for i in rng.permutation(len(pop)):
            if len(chosen) == n:
                return chosen
            r = pop.iloc[i]
            if r["image_path"] in {x["image_path"] for x in chosen} or taken[(r["group"], r["tube_id"])] >= c:
                continue
            chosen.append(r)
            taken[(r["group"], r["tube_id"])] += 1
    return chosen


def spread(pop, n, rng, taken, cap):
    """n images courantes réparties entre profondeur (au prorata de la population) et densité de racine (tiers égaux)."""
    depth = pd.cut(pop["position"], DEPTHS, labels=False)
    dens = pd.qcut(pop["root_frac"], 3, labels=False)
    cells = [(a, b) for a in range(3) for b in range(3)]
    weights = [(depth == a).mean() / 3 for a, b in cells]
    chosen = []
    for (a, b), q in zip(cells, split_quota(n, weights, floor=1 if n >= 9 else 0)):
        chosen += pick(pop[(depth == a) & (dens == b)], q, rng, taken, cap)
    return chosen


def check(part, n, what):
    if len(part) != n:
        raise ValueError(f"{what} : {len(part)} images tirées pour un quota de {n}")


def build_pool(stats, seed=0):
    """Images à annoter : une ligne par image, colonne `lot` = train, val ou test."""
    rng = np.random.default_rng(seed)
    stats = stats.assign(type=image_type(stats))
    pops = {s: stats[(stats["split"] == s) & stats["group"].isin(GROUPS)] for s in ("train", "val", "test")}
    rows = []
    for split in ("train", "val"):
        taken = Counter()
        for g in GROUPS:
            pop = pops[split][pops[split]["group"] == g]
            rooted = pop[pop["d_gray"].notna()]
            q = QUOTAS[split][g]
            part = pick(pop[~pop["has_root"]], q["vide"], rng, taken, CAPS[split])
            for t in HARD_TYPES:
                part += pick(rooted[rooted["type"] == t], q.get(t, 0), rng, taken, CAPS[split])
            part += spread(rooted[rooted["type"] == "typique"], q["typique"], rng, taken, CAPS[split])
            check(part, sum(q.values()), f"{split}, {g}")
            rows.append(pd.DataFrame(part).assign(lot=split))
    for split in ("train", "val"):                                  # images sans racine ajoutées, sans plafond
        done = set(pd.concat(rows)["image_path"])
        for g, n in EXTRA_EMPTY[split].items():
            pop = pops[split]
            pop = pop[(pop["group"] == g) & ~pop["has_root"] & ~pop["image_path"].isin(done)]
            part = pick(pop, n, rng, Counter(), None)
            check(part, n, f"{split}, {g}, sans racine ajoutées")
            rows.append(pd.DataFrame(part).assign(lot=split))
    test = pops["test"]                                             # test : au hasard, au prorata des groupes
    for g, n in zip(GROUPS, split_quota(N_TEST, [(test["group"] == g).sum() for g in GROUPS])):
        part = pick(test[test["group"] == g], n, rng, Counter(), None)
        check(part, n, f"test, {g}")
        rows.append(pd.DataFrame(part).assign(lot="test"))
    cols = ["lot", "group", "split", "tube_id", "date", "position", "root_frac", "type", "has_root", "image_path", "mask_path"]
    return pd.concat(rows)[cols].reset_index(drop=True)


if __name__ == "__main__":
    pool = build_pool(load_image_stats())
    POOL.parent.mkdir(parents=True, exist_ok=True)
    pool.to_csv(POOL, index=False)
    print(pool.groupby(["lot", "group"]).size().unstack(), "->", POOL)
