"""Node features, built as a ladder. Each rung adds information to the one before.

rung 1  own threshold tau
rung 2  + size and number of counterparties
rung 3  + the bank's own row of exposures (how concentrated its lending is)
rung 4  + the same numbers for its borrowers, summed up by hand (sum, mean, max, min, weighted mean)
rung 5  + centrality in the whole network (PageRank, hub, authority)

Rungs 1 to 3 use only what a bank knows about itself. They do not change when
the links are randomised with `generate.randomise_edges`.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from .generate import Network

OWN = ["tau", "log_assets", "n_borrowers", "n_lenders",
       "max_share", "top2_share", "hhi", "n_single_killers", "share_single_killers"]
RUNGS = {1: 1, 2: 4, 3: 9}  # number of own columns used by rungs 1 to 3


def own_features(net: Network) -> np.ndarray:
    E, S = net.E, net.shares
    assets = E.sum(axis=1)
    top = -np.sort(-S, axis=1)
    killers = (S >= net.tau[:, None]) & (S > 0)  # one borrower whose failure alone is enough
    cols = [
        net.tau,
        np.log1p(assets),
        (E > 0).sum(axis=1),
        (E > 0).sum(axis=0),
        top[:, 0],
        top[:, :2].sum(axis=1),
        (S**2).sum(axis=1),
        killers.sum(axis=1),
        (S * killers).sum(axis=1),
    ]
    return np.column_stack(cols).astype(float)


def neighbour_features(net: Network, X: np.ndarray) -> np.ndarray:
    """For each lender: sum, mean, max, min and share-weighted mean of every
    own feature over its borrowers. Zeros for a bank with no borrower."""
    A = net.E > 0
    S = net.shares
    deg = A.sum(axis=1)
    out = []
    for c in range(X.shape[1]):
        x = X[:, c]
        M = np.where(A, x[None, :], np.nan)
        with np.errstate(all="ignore"):
            total = np.nansum(M, axis=1)
            mean = np.where(deg > 0, total / np.maximum(deg, 1), 0.0)
            mx = np.where(deg > 0, np.nanmax(np.where(A, x[None, :], -np.inf), axis=1), 0.0)
            mn = np.where(deg > 0, np.nanmin(np.where(A, x[None, :], np.inf), axis=1), 0.0)
        out += [total, mean, mx, mn, S @ x]
    return np.column_stack(out)


def centrality_features(net: Network) -> np.ndarray:
    """PageRank, hub and authority scores.

    PageRank is computed on the reversed graph (links from borrower to lender,
    the direction of the losses): a bank scores high when it lends to banks
    that score high. Eigenvector centrality is not used because it is not
    defined on these sparse directed networks (they are not strongly connected).
    """
    G = nx.from_numpy_array(net.E, create_using=nx.DiGraph)
    n = net.n
    pr = nx.pagerank(G.reverse(), weight="weight")
    hubs, auth = nx.hits(G, max_iter=1000)
    return np.column_stack([[pr[i] for i in range(n)], [hubs[i] for i in range(n)], [auth[i] for i in range(n)]])


def ladder(net: Network) -> dict[int, np.ndarray]:
    X = own_features(net)
    nb = neighbour_features(net, X)
    cen = centrality_features(net)
    return {
        1: X[:, : RUNGS[1]],
        2: X[:, : RUNGS[2]],
        3: X,
        4: np.hstack([X, nb]),
        5: np.hstack([X, nb, cen]),
    }
