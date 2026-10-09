"""Which banks are fragile? A ladder of simple models, an MLP and a GNN.

The protocol is fixed in PREDICTION.md. Rows marked "post-hoc" were added after the
first run and its review; they are not part of the protocol.
Writes results/scores.csv (one row per model, seed and test network),
results/control.csv, results/gnn_checks.csv and results/depth.csv.
"""

import warnings

import numpy as np
import pandas as pd

from cascades.features import OWN
from cascades.generate import randomise_edges
from cascades.learn import (fit_boosting, fit_neural, make_sample, make_samples, score, score_rows,
                            spearman)
from cascades.sim import all_cascades, fragility

warnings.filterwarnings("ignore", category=FutureWarning)

TAU = 0.5
TRAIN_SEEDS, TEST_SEEDS, MODEL_SEEDS = range(3000, 3030), range(4000, 4015), range(10)
FRACTIONS, COPIES = [0.0, 0.1, 0.25, 0.5, 1.0, 3.0], 5
OTHER = {"er": "cp", "cp": "er"}
GNN, GNN_ES = "gnn (rung 3 inputs)", "post-hoc: gnn, early stopping"

train = {f: make_samples(f, TRAIN_SEEDS, TAU) for f in ("er", "cp")}
test = {f: make_samples(f, TEST_SEEDS, TAU) for f in ("er", "cp")}

scores, control, checks, depth = [], [], [], []


def add(rows, **keys):
    scores.extend({**keys, **r} for r in rows)


for fam in ("er", "cp"):
    # no model at all: one hand count, and the simulator stopped after two rounds
    def killers(s):
        return s.X[3][:, OWN.index("n_single_killers")]

    def two_rounds(s):
        return fragility(all_cascades(s.net.E, s.net.capital, s.net.lgd, max_rounds=2)[0])

    for name, f in (("no model: count of single killers", killers), ("no model: simulator, 2 rounds", two_rounds)):
        add(score_rows(test[fam], f), train=fam, test=fam, label="y", model=name, seed=0)

    for s in test[fam]:
        F, R, _ = all_cascades(s.net.E, s.net.capital, s.net.lgd)
        r = R[F & (R > 0)]
        depth.append({"family": fam, "network": s.net.seed, "failures": len(r), "after_round_1": int((r > 1).sum()),
                      "after_round_2": int((r > 2).sum())})

    for target in ("y", "y_hit"):
        models = {}
        for rung in (1, 2, 3, 4, 5):  # deterministic: one fit is enough
            models[f"boosting rung {rung}", 0] = fit_boosting(train[fam], rung, 0, target)
        for seed in MODEL_SEEDS:
            for rung in (3, 4):
                models[f"mlp rung {rung}", seed] = fit_neural(train[fam], "mlp", rung, seed, target)
            models[GNN, seed] = fit_neural(train[fam], "gnn", 3, seed, target)
            if target == "y":
                models[GNN_ES, seed] = fit_neural(train[fam][:25], "gnn", 3, seed, target,
                                                  validation=train[fam][25:], max_epochs=600)

        for (name, seed), predict in models.items():
            for test_fam in (fam, OTHER[fam]):
                add(score_rows(test[test_fam], predict, target), train=fam, test=test_fam, label=target,
                    model=name, seed=seed)

        if target != "y":
            continue

        # GNN checks on the untouched test networks: no edges, edges the wrong way round
        for seed in MODEL_SEEDS:
            g = models[GNN, seed]
            for mode in ("normal", "none", "reversed"):
                checks.append({"family": fam, "seed": seed, "edges": mode,
                               **score(test[fam], lambda s, m=mode: g(s, m))})

        # control: randomise the links of the test networks, keep the old labels
        for frac in FRACTIONS:
            for copy in range(COPIES):
                rng = np.random.default_rng(10_000 + copy)
                for s in test[fam]:
                    new = make_sample(randomise_edges(s.net, frac, rng))
                    moved = float(((s.net.E > 0) & ~(new.net.E > 0)).sum() / (s.net.E > 0).sum())
                    base = {"family": fam, "network": s.net.seed, "fraction": frac, "copy": copy,
                            "links_moved": moved}
                    control.append({**base, "model": "simulator (reference)", "seed": 0,
                                    "vs_old_labels": spearman(s.y, new.y), "vs_new_labels": 1.0})
                    for (name, seed), predict in models.items():
                        if name in ("boosting rung 3", "boosting rung 4", "mlp rung 3", GNN):
                            p = predict(new)
                            control.append({**base, "model": name, "seed": seed,
                                            "vs_old_labels": spearman(s.y, p), "vs_new_labels": spearman(new.y, p)})
    print("done", fam, flush=True)

pd.DataFrame(scores).to_csv("results/scores.csv", index=False)
pd.DataFrame(control).to_csv("results/control.csv", index=False)
pd.DataFrame(checks).to_csv("results/gnn_checks.csv", index=False)
pd.DataFrame(depth).to_csv("results/depth.csv", index=False)
