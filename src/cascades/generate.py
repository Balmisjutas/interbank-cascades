"""Simulated interbank networks.

E[i, j] = amount that bank i lent to bank j. Two families with the same number
of banks, the same number of links and the same weight distribution:
    "er"  Erdos-Renyi: every ordered pair is equally likely
    "cp"  core-periphery: a small core that is densely linked, a periphery that
          mostly deals with the core
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_BANKS = 100
N_LINKS = 495  # density 5 % of the 9,900 ordered pairs
N_CORE = 10


@dataclass
class Network:
    E: np.ndarray  # exposures, lender in rows
    capital: np.ndarray
    tau: np.ndarray  # own threshold: capital / (lgd * interbank assets)
    lgd: float
    family: str
    seed: int

    @property
    def n(self) -> int:
        return self.E.shape[0]

    @property
    def shares(self) -> np.ndarray:
        """shares[i, j] = part of bank i's interbank lending that goes to j."""
        a = self.E.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(a > 0, self.E / a, 0.0)


def _pair_probabilities(family: str, n: int, n_core: int) -> np.ndarray:
    p = np.ones((n, n))
    if family == "cp":
        core = np.zeros(n, dtype=bool)
        core[:n_core] = True
        cc = np.outer(core, core)
        pp = np.outer(~core, ~core)
        p = np.where(cc, 0.80, np.where(pp, 0.01, 0.19))
    elif family != "er":
        raise ValueError(f"unknown family {family!r}")
    np.fill_diagonal(p, 0.0)
    return p


def adjacency(family: str, n: int, n_links: int, rng, n_core: int = N_CORE) -> np.ndarray:
    """Draw exactly `n_links` directed links, no self-loops, no double links."""
    p = _pair_probabilities(family, n, n_core).ravel()
    chosen = rng.choice(n * n, size=n_links, replace=False, p=p / p.sum())
    A = np.zeros(n * n, dtype=bool)
    A[chosen] = True
    return A.reshape(n, n)


def make_capital(E: np.ndarray, tau_mean: float, lgd: float, rng, tau_sigma: float = 0.3):
    """Capital of each bank, written through its threshold tau.

    tau_i = capital_i / (lgd * interbank assets_i) is the share of its lending
    that bank i can lose before it fails. tau_i is drawn around `tau_mean`
    (lognormal, sd `tau_sigma` on the log scale), so that banks differ.
    A bank that lends to nobody cannot be hit; it gets tau and an arbitrary capital of 1.
    """
    n = E.shape[0]
    assets = E.sum(axis=1)
    tau = tau_mean * np.exp(rng.normal(-0.5 * tau_sigma**2, tau_sigma, size=n))
    capital = np.where(assets > 0, tau * lgd * assets, 1.0)
    return capital, tau


def make_network(
    family: str,
    seed: int,
    tau_mean: float,
    lgd: float = 0.6,
    n: int = N_BANKS,
    n_links: int = N_LINKS,
    tau_sigma: float = 0.3,
    equal_weights: bool = False,
) -> Network:
    rng = np.random.default_rng(seed)
    A = adjacency(family, n, n_links, rng)
    w = np.ones((n, n)) if equal_weights else rng.lognormal(0.0, 1.0, size=(n, n))
    E = np.where(A, w, 0.0)
    capital, tau = make_capital(E, tau_mean, lgd, rng, tau_sigma)
    return Network(E=E, capital=capital, tau=tau, lgd=lgd, family=family, seed=seed)


def common_shock(net: Network, share: float) -> Network:
    """Every bank loses `share` of its capital before any default (as in WP 545).

    With this rule a common shock of size s is the same as multiplying every
    threshold tau by (1 - s). It is kept as a function because this is how the
    central-bank papers describe their stress scenario.
    """
    k = 1.0 - share
    return Network(net.E, net.capital * k, net.tau * k, net.lgd, net.family, net.seed)


def randomise_edges(net: Network, fraction: float, rng) -> Network:
    """Degree-preserving edge randomisation (double edge swaps).

    Two links (i -> j) and (k -> l) become (i -> l) and (k -> j). Each lender
    keeps its number of borrowers AND its own set of amounts, so its interbank
    assets, its capital and its threshold do not change. Each borrower keeps
    its number of lenders. Only WHO is linked to whom changes.
    `fraction` is the share of links that we try to move (1.0 = every link once).
    """
    E = net.E.copy()
    lenders, borrowers = np.nonzero(E)
    m = len(lenders)
    edges = list(zip(lenders.tolist(), borrowers.tolist()))
    present = set(edges)
    target = int(round(fraction * m / 2))
    done, tries = 0, 0
    while done < target and tries < 50 * max(target, 1):
        tries += 1
        a, b = rng.choice(m, size=2, replace=False)
        (i, j), (k, l) = edges[a], edges[b]
        if i == k or j == l or i == l or k == j:
            continue
        if (i, l) in present or (k, j) in present:
            continue
        E[i, l], E[k, j] = E[i, j], E[k, l]
        E[i, j], E[k, l] = 0.0, 0.0
        present -= {(i, j), (k, l)}
        present |= {(i, l), (k, j)}
        edges[a], edges[b] = (i, l), (k, j)
        done += 1
    return Network(E, net.capital, net.tau, net.lgd, net.family, net.seed)
