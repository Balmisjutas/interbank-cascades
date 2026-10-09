"""Models and scores for the question "which banks are fragile?".

Three kinds of model, all trained on some networks and scored on OTHER networks:
    boosting  gradient boosting on one rung of the feature ladder
    mlp       a small neural network on the same table (no graph)
    gnn       two GraphConv layers; it gets the own features of each bank
              (rung 3) and must find the neighbour information by itself
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from torch import nn
from torch_geometric.nn import GraphConv

from .features import ladder
from .generate import Network, make_network
from .sim import all_cascades, fragility, hit_share

HIDDEN, EPOCHS, LR = 64, 300, 0.01  # fixed before the runs, never tuned


@dataclass
class Sample:
    net: Network
    y: np.ndarray  # fragility: share of scenarios in which the bank fails
    y_hit: np.ndarray  # share of scenarios in which it loses more than 10 % of capital
    X: dict  # feature ladder


def make_sample(net: Network) -> Sample:
    F, _, L = all_cascades(net.E, net.capital, net.lgd)
    return Sample(net, fragility(F), hit_share(L), ladder(net))


def make_samples(family: str, seeds, tau: float) -> list[Sample]:
    return [make_sample(make_network(family, s, tau_mean=tau)) for s in seeds]


# ---------------------------------------------------------------- scores

def spearman(y, p) -> float:
    if np.std(y) == 0 or np.std(p) == 0:
        return np.nan
    return float(spearmanr(y, p).statistic)


def precision_at_10pct(y, p) -> float:
    """Of the 10 % of banks that the model ranks as most fragile, how many are
    among the truly most fragile 10 %? Ties in the truth are all accepted."""
    k = max(1, len(y) // 10)
    cut = np.sort(y)[-k]
    if cut <= 0:
        return np.nan
    top = np.argsort(-p, kind="stable")[:k]
    return float((y[top] >= cut).mean())


def concordance(y, p) -> float:
    """Among the pairs of banks with DIFFERENT true fragility, the share that the
    model puts in the right order (a tie in the prediction counts one half).
    0.5 is chance, 1 is perfect. Unlike Spearman, it does not punish a model for
    ordering banks that have exactly the same label."""
    dy = y[:, None] - y[None, :]
    dp = p[:, None] - p[None, :]
    pairs = dy > 0
    if not pairs.any():
        return np.nan
    return float(((dp > 0) & pairs).sum() + 0.5 * ((dp == 0) & pairs).sum()) / float(pairs.sum())


def score_rows(samples, predict, target="y") -> list[dict]:
    """One row of scores per test network."""
    rows = []
    for s in samples:
        y, p = getattr(s, target), predict(s)
        rows.append({"network": s.net.seed, "spearman": spearman(y, p),
                     "concordance": concordance(y, p), "precision10": precision_at_10pct(y, p)})
    return rows


def score(samples, predict, target="y") -> dict:
    rows = score_rows(samples, predict, target)
    return {k: float(np.nanmean([r[k] for r in rows])) for k in ("spearman", "concordance", "precision10")}


# ---------------------------------------------------------------- boosting

def fit_boosting(train, rung: int, seed: int = 0, target="y"):
    """Default settings. With no subsampling the fit is deterministic: `seed` changes nothing."""
    X = np.vstack([s.X[rung] for s in train])
    y = np.concatenate([getattr(s, target) for s in train])
    model = HistGradientBoostingRegressor(random_state=seed).fit(X, y)
    return lambda s: model.predict(s.X[rung])


# ---------------------------------------------------------------- neural models

class Scaler:
    def __init__(self, train, rung, target):
        X = np.vstack([s.X[rung] for s in train])
        y = np.concatenate([getattr(s, target) for s in train])
        self.mu, self.sd = X.mean(axis=0), X.std(axis=0) + 1e-9
        self.y_sd = y.std() + 1e-9
        self.rung, self.target = rung, target

    def x(self, s):
        return torch.tensor((s.X[self.rung] - self.mu) / self.sd, dtype=torch.float32)

    def y(self, s):
        return torch.tensor(getattr(s, self.target) / self.y_sd, dtype=torch.float32)


def edges(net: Network, mode: str = "normal"):
    """Edges as PyTorch Geometric wants them: messages go from source to target.
    A loss goes from the BORROWER to the LENDER, so source = borrower j,
    target = lender i, weight = share of i's lending that goes to j."""
    S = net.shares
    lender, borrower = np.nonzero(S)
    w = S[lender, borrower]
    if mode == "none":
        lender, borrower, w = lender[:0], borrower[:0], w[:0]
    src, dst = (lender, borrower) if mode == "reversed" else (borrower, lender)
    return (torch.tensor(np.vstack([src, dst]), dtype=torch.long),
            torch.tensor(w, dtype=torch.float32))


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN),
                                 nn.ReLU(), nn.Linear(HIDDEN, 1))

    def forward(self, x, edge_index=None, edge_weight=None):
        return self.net(x).squeeze(-1)


class GNN(nn.Module):
    """Two rounds of message passing. One GraphConv layer computes
    W1 * x_i + W2 * sum_j share_ij * x_j, which has the shape of one cascade round."""

    def __init__(self, d):
        super().__init__()
        self.c1 = GraphConv(d, HIDDEN, aggr="add")
        self.c2 = GraphConv(HIDDEN, HIDDEN, aggr="add")
        self.out = nn.Linear(HIDDEN, 1)

    def forward(self, x, edge_index, edge_weight):
        h = torch.relu(self.c1(x, edge_index, edge_weight))
        h = torch.relu(self.c2(h, edge_index, edge_weight))
        return self.out(h).squeeze(-1)


def fit_neural(train, kind: str, rung: int, seed: int, target="y", validation=None, max_epochs=EPOCHS):
    """Returns predict(sample, mode="normal"). `mode` changes the edges given to
    the GNN at prediction time: "normal", "none" or "reversed".

    With `validation` (a list of samples that are NOT in `train`), the weights
    kept are those of the epoch with the best rank correlation on the validation
    networks, checked every 10 epochs. This variant was added after the first
    run; the protocol of PREDICTION.md is the call without `validation`."""
    torch.manual_seed(seed)
    sc = Scaler(train, rung, target)
    model = (GNN if kind == "gnn" else MLP)(train[0].X[rung].shape[1])
    data = [(sc.x(s), *edges(s.net), sc.y(s)) for s in train]
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    def predict(s, mode="normal"):
        model.eval()
        with torch.no_grad():
            return model(sc.x(s), *edges(s.net, mode)).numpy()

    best, best_state = -np.inf, None
    for epoch in range(1, max_epochs + 1):
        model.train()
        opt.zero_grad()
        loss = sum(((model(x, ei, ew) - y) ** 2).mean() for x, ei, ew, y in data) / len(data)
        loss.backward()
        opt.step()
        if validation and epoch % 10 == 0:
            val = np.nanmean([spearman(getattr(s, target), predict(s)) for s in validation])
            if val > best:
                best, best_state = val, {k: v.clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return predict
