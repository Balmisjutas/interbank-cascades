"""Experiment B. Which banks are fragile? A ladder of simple models, an MLP and a GNN.

Protocol fixed in PREDICTION.md. Writes results/scores.csv, results/control.csv,
results/gnn_checks.csv.
"""

import warnings

import numpy as np
import pandas as pd

from cascades.generate import randomise_edges
from cascades.learn import (fit_boosting, fit_neural, make_sample, make_samples, score,
                            spearman)

warnings.filterwarnings("ignore")

TAU = 0.5
TRAIN_SEEDS, TEST_SEEDS, MODEL_SEEDS = range(3000, 3030), range(4000, 4015), range(10)
FRACTIONS, COPIES = [0.0, 0.1, 0.25, 0.5, 1.0], 5
OTHER = {"er": "cp", "cp": "er"}

train = {f: make_samples(f, TRAIN_SEEDS, TAU) for f in ("er", "cp")}
test = {f: make_samples(f, TEST_SEEDS, TAU) for f in ("er", "cp")}

scores, control, checks = [], [], []

for fam in ("er", "cp"):
    for target in ("y", "y_hit"):
        models = {}
        for seed in MODEL_SEEDS:
            for rung in (1, 2, 3, 4, 5):
                models[f"boosting rung {rung}", seed] = fit_boosting(train[fam], rung, seed, target)
            for rung in (3, 4):
                models[f"mlp rung {rung}", seed] = fit_neural(train[fam], "mlp", rung, seed, target)
            models["gnn (rung 3 inputs)", seed] = fit_neural(train[fam], "gnn", 3, seed, target)

        for (name, seed), predict in models.items():
            for test_fam in (fam, OTHER[fam]):
                scores.append({"train": fam, "test": test_fam, "label": target, "model": name,
                               "seed": seed, **score(test[test_fam], predict, target)})

        if target != "y":
            continue

        # GNN checks on the untouched test networks: no edges, edges the wrong way round
        for seed in MODEL_SEEDS:
            g = models["gnn (rung 3 inputs)", seed]
            for mode in ("normal", "none", "reversed"):
                checks.append({"family": fam, "seed": seed, "edges": mode,
                               **score(test[fam], lambda s, m=mode: g(s, m))})

        # control: randomise the links of the test networks, keep the old labels
        for frac in FRACTIONS:
            for copy in range(COPIES):
                rng = np.random.default_rng(10_000 + copy)
                for s in test[fam]:
                    new = make_sample(randomise_edges(s.net, frac, rng))
                    moved = float(((s.net.E > 0) != (new.net.E > 0)).sum() / 2 / (s.net.E > 0).sum())
                    control.append({"family": fam, "fraction": frac, "copy": copy, "model": "simulator (reference)",
                                    "seed": 0, "links_moved": moved,
                                    "vs_old_labels": spearman(s.y, new.y), "vs_new_labels": 1.0})
                    for name in ("boosting rung 3", "boosting rung 4", "boosting rung 5",
                                 "mlp rung 3", "gnn (rung 3 inputs)"):
                        for seed in MODEL_SEEDS:
                            p = models[name, seed](new)
                            control.append({"family": fam, "fraction": frac, "copy": copy, "model": name,
                                            "seed": seed, "links_moved": moved,
                                            "vs_old_labels": spearman(s.y, p),
                                            "vs_new_labels": spearman(new.y, p)})
    print("done", fam, flush=True)

pd.DataFrame(scores).to_csv("results/scores.csv", index=False)
pd.DataFrame(control).to_csv("results/control.csv", index=False)
pd.DataFrame(checks).to_csv("results/gnn_checks.csv", index=False)

sc = pd.DataFrame(scores)
print(sc.groupby(["label", "train", "test", "model"])[["spearman", "precision10"]]
        .agg(["mean", "std"]).round(3).to_string())
print(pd.DataFrame(checks).groupby(["family", "edges"])[["spearman", "precision10"]].mean().round(3).to_string())
ct = pd.DataFrame(control)
print(ct.groupby(["family", "model", "fraction"])[["vs_old_labels", "vs_new_labels", "links_moved"]]
        .mean().round(3).to_string())
