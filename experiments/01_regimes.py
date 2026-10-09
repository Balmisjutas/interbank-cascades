"""Experiment A. In which regime does contagion exist at all?

Part 1: sweep the mean threshold tau for both network families.
Part 2: sanity check against a known result (Gai and Kapadia 2010): on a random
        network with equal weights, contagion exists only in a window of the
        average degree, and inside the window it is rare but system-wide
        ("robust yet fragile").
Writes results/regimes.csv and results/gai_kapadia.csv.
"""

import numpy as np
import pandas as pd

from cascades.generate import make_network
from cascades.sim import all_cascades, fragility, summarise

TAUS = [0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0, 1.25]
SEEDS = range(1000, 1020)

rows = []
for family in ("er", "cp"):
    for tau in TAUS:
        for seed in SEEDS:
            net = make_network(family, seed, tau_mean=tau)
            F, R, _ = all_cascades(net.E, net.capital, net.lgd)
            frag = fragility(F)
            contagion = F.copy()
            contagion[np.arange(net.n), np.arange(net.n)] = False
            late = (R[contagion] > 2).mean() if contagion.any() else np.nan
            rows.append(
                {"family": family, "tau": tau, "seed": seed, **summarise(F, R),
                 "share_zero_labels": float((frag == 0).mean()),
                 "label_sd": float(frag.std()),
                 "share_failures_after_round_2": late}
            )
pd.DataFrame(rows).to_csv("results/regimes.csv", index=False)

# Gai-Kapadia: 1,000 banks, equal weights, every bank fails when it loses 20 % of its
# lending (capital 4 % of assets, interbank assets 20 %, no recovery). 200 random triggers.
N, TAU_GK = 1000, 0.2
gk = []
for z in [0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]:
    for seed in range(2000, 2005):
        net = make_network("er", seed, tau_mean=TAU_GK, lgd=1.0, n=N, n_links=int(z * N),
                           tau_sigma=0.0, equal_weights=True)
        triggers = np.random.default_rng(seed).choice(N, size=200, replace=False)
        F, _, _ = all_cascades(net.E, net.capital, net.lgd, triggers)
        size = (F.sum(axis=1) - 1) / N
        big = size > 0.05
        gk.append({"z": z, "seed": seed, "frequency": float(big.mean()),
                   "extent": float(size[big].mean()) if big.any() else np.nan})
pd.DataFrame(gk).to_csv("results/gai_kapadia.csv", index=False)

df = pd.DataFrame(rows).groupby(["family", "tau"]).mean(numeric_only=True).drop(columns="seed")
print(df.round(3).to_string())
print(pd.DataFrame(gk).groupby("z")[["frequency", "extent"]].mean().round(3).to_string())
