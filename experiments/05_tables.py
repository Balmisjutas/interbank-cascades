"""Print every number quoted in the README from the CSV files in results/."""

import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
sc = pd.read_csv("results/scores.csv")
GNN, R4 = "gnn (rung 3 inputs)", "boosting rung 4"

print("== main table (label y, same family), mean over seeds and test networks")
m = sc[(sc.label == "y") & (sc.train == sc.test)]
print(m.groupby(["model", "train"])[["spearman", "concordance", "precision10"]].mean().unstack().round(3).to_string())

print("\n== paired difference per test network: model A minus model B (mean over seeds), 95 % interval, wins of A")
per = m.groupby(["train", "model", "network"])[["spearman", "concordance", "precision10"]].mean()
pairs = [(GNN, R4), (GNN, "mlp rung 3"), (GNN, "boosting rung 3"), ("post-hoc: gnn, early stopping", R4),
         ("boosting rung 5", R4), (GNN, "no model: count of single killers")]
for fam in ("er", "cp"):
    for a, b in pairs:
        for col in ("spearman", "concordance"):
            d = (per.loc[fam, a][col] - per.loc[fam, b][col]).dropna()
            half = 2.145 * d.std(ddof=1) / np.sqrt(len(d))  # t, 14 degrees of freedom
            print(f"{fam} {col:11s} {a} - {b}: {d.mean():+.3f} [{d.mean() - half:+.3f}, {d.mean() + half:+.3f}] wins {int((d > 0).sum())}/{len(d)}")

print("\n== trained on one family, tested on the other (label y)")
x = sc[(sc.label == "y") & (sc.train != sc.test)]
print(x.groupby(["model", "train"])[["spearman", "concordance"]].mean().unstack().round(3).to_string())

print("\n== second label (y_hit), same family")
h = sc[(sc.label == "y_hit") & (sc.train == sc.test)]
print(h.groupby(["model", "train"])[["spearman", "concordance"]].mean().unstack().round(3).to_string())

print("\n== GNN checks"); print(pd.read_csv("results/gnn_checks.csv").groupby(["family", "edges"])[["spearman", "concordance"]].mean().round(3).to_string())

print("\n== control"); c = pd.read_csv("results/control.csv")
print(c.groupby(["family", "model", "fraction"])[["links_moved", "vs_old_labels", "vs_new_labels"]].mean().round(3).to_string())

print("\n== depth of the cascades on the test networks (pooled over failures)"); d = pd.read_csv("results/depth.csv").groupby("family").sum(numeric_only=True)
d["share_after_round_1"] = d.after_round_1 / d.failures; d["share_after_round_2"] = d.after_round_2 / d.failures
print(d[["failures", "share_after_round_1", "share_after_round_2"]].round(3).to_string())

print("\n== label: distinct values and zeros per test network")
from cascades.learn import make_samples
for fam in ("er", "cp"):
    S = make_samples(fam, range(4000, 4015), 0.5)
    print(fam, "distinct", np.mean([len(np.unique(s.y)) for s in S]).round(1), "zeros", np.mean([(s.y == 0).mean() for s in S]).round(3),
          "true top-10 % set size with ties", np.mean([(s.y >= np.sort(s.y)[-10]).sum() for s in S]).round(1))
