"""Print every number quoted in the README from the CSV files in results/."""

import warnings

import numpy as np
import pandas as pd

from cascades.generate import make_network
from cascades.learn import make_samples

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.width", 200)

GNN, GNN_ES, R4 = "gnn (rung 3 inputs)", "post-hoc: gnn, early stopping", "boosting rung 4"
COLS = ["spearman", "concordance", "precision10"]
T14 = 2.145  # 97.5 % quantile of Student's t with 14 degrees of freedom (15 test networks)

sc = pd.read_csv("results/scores.csv")
same = sc[(sc.label == "y") & (sc.train == sc.test)]

print("== Result 1: regimes (mean over 20 networks)")
reg = pd.read_csv("results/regimes.csv").groupby(["family", "tau"]).mean(numeric_only=True)
print(reg[["share_contagion", "mean_cascade", "worst_cascade", "share_zero_labels"]].round(3).to_string())
print(pd.read_csv("results/gai_kapadia.csv").groupby("z")[["frequency", "extent"]].mean().round(3).to_string())

print("\n== main table (label y, same family), mean over seeds and test networks")
print(same.groupby(["model", "train"])[COLS].mean().unstack().round(3).to_string())

print("\n== paired difference per test network: A minus B (mean over seeds), 95 % interval, wins of A")
per = same.groupby(["train", "model", "network"])[COLS].mean()
pairs = [(GNN, R4), (GNN, "mlp rung 3"), (GNN, "boosting rung 3"), (GNN_ES, R4), ("boosting rung 5", R4),
         (GNN, "no model: count of single killers")]
for fam in ("er", "cp"):
    for a, b in pairs:
        for col in COLS:
            d = (per.loc[fam, a][col] - per.loc[fam, b][col]).dropna()
            half = T14 * d.std(ddof=1) / np.sqrt(len(d))
            print(f"{fam} {col:11s} {a} - {b}: {d.mean():+.3f} [{d.mean() - half:+.3f}, {d.mean() + half:+.3f}] "
                  f"wins {int((d > 0).sum())}/{len(d)}")

print("\n== trained on one family, tested on the other (label y)")
other = sc[(sc.label == "y") & (sc.train != sc.test)]
print(other.groupby(["model", "train"])[["spearman", "concordance"]].mean().unstack().round(3).to_string())

print("\n== second label (y_hit), same family")
hit = sc[(sc.label == "y_hit") & (sc.train == sc.test)]
print(hit.groupby(["model", "train"])[["spearman", "concordance"]].mean().unstack().round(3).to_string())

print("\n== GNN checks")
checks = pd.read_csv("results/gnn_checks.csv")
print(checks.groupby(["family", "edges"])[["spearman", "concordance"]].mean().round(3).to_string())

print("\n== control")
control = pd.read_csv("results/control.csv")
print(control.groupby(["family", "model", "fraction"])[["links_moved", "vs_old_labels", "vs_new_labels"]]
      .mean().round(3).to_string())

print("\n== depth of the cascades on the test networks (pooled over failures)")
depth = pd.read_csv("results/depth.csv").groupby("family").sum(numeric_only=True)
depth["share_after_round_1"] = depth.after_round_1 / depth.failures
depth["share_after_round_2"] = depth.after_round_2 / depth.failures
print(depth[["failures", "share_after_round_1", "share_after_round_2"]].round(3).to_string())

print("\n== the label on the test networks, and the ceiling of Spearman when ties are broken at random")
rng = np.random.default_rng(0)
for fam in ("er", "cp"):
    samples = make_samples(fam, range(4000, 4015), 0.5)
    distinct = np.mean([len(np.unique(s.y)) for s in samples])
    zeros = np.mean([(s.y == 0).mean() for s in samples])
    top = np.mean([(s.y >= np.sort(s.y)[-10]).sum() for s in samples])
    ceiling = np.mean([pd.Series(s.y).corr(pd.Series(s.y + 1e-9 * rng.random(len(s.y))), method="spearman")
                       for s in samples for _ in range(20)])
    over = np.mean([((n.E / n.capital[:, None])[n.E > 0] > 0.25).mean()
                    for n in (make_network(fam, seed, tau_mean=0.5) for seed in range(4000, 4015))])
    print(f"{fam}: distinct values {distinct:.1f}, zeros {zeros:.3f}, size of the true top 10 % with ties {top:.1f}, "
          f"Spearman ceiling {ceiling:.3f}, links above 25 % of the lender's capital {over:.2f}")
