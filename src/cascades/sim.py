"""Sequential default cascade with a fixed loss given default (Furfine 2003).

Convention used everywhere in this project:
    E[i, j] = amount that bank i LENT to bank j (an asset of i, a liability of j).
A bank fails when its capital is wiped out. When a bank fails, each of its
lenders loses LGD times its exposure to it. Same rule as Banque de France
Working Paper 432, eq. 2.2:

    C_k(i) = max( C_0(i) - LGD * sum_{j failed} E[i, j], 0 )

The loss is always computed over ALL banks failed so far and compared with the
STARTING capital, so no loss is counted twice.
"""

from __future__ import annotations

import numpy as np


def cascade(E: np.ndarray, capital: np.ndarray, trigger: int, lgd: float = 0.6):
    """Run one cascade started by the failure of `trigger`.

    Returns (failed, rounds, loss_share):
        failed      boolean vector, True for every failed bank (trigger included)
        rounds      int vector, round in which each bank failed (0 = trigger, -1 = never)
        loss_share  loss of each bank as a share of its starting capital (capped at 1)
    """
    n = E.shape[0]
    failed = np.zeros(n, dtype=bool)
    rounds = np.full(n, -1, dtype=int)
    failed[trigger] = True
    rounds[trigger] = 0
    k = 0
    while True:
        k += 1
        loss = lgd * E[:, failed].sum(axis=1)
        newly = (~failed) & (loss >= capital)
        if not newly.any():
            break
        failed |= newly
        rounds[newly] = k
    loss = lgd * E[:, failed].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        loss_share = np.where(capital > 0, np.minimum(loss / capital, 1.0), 0.0)
    loss_share[trigger] = 1.0
    return failed, rounds, loss_share


def all_cascades(E: np.ndarray, capital: np.ndarray, lgd: float = 0.6, triggers=None, max_rounds=None):
    """Run one cascade per trigger bank, all at once. `max_rounds` stops the cascade early.

    Returns (F, R, L), each of shape (n_triggers, n):
        F[s, i]  True if bank i failed in the scenario where triggers[s] fails first
        R[s, i]  round of failure (0 = trigger, -1 = never)
        L[s, i]  loss of bank i as a share of its starting capital (capped at 1)
    """
    n = E.shape[0]
    triggers = np.arange(n) if triggers is None else np.asarray(triggers)
    s = len(triggers)
    F = np.zeros((s, n), dtype=bool)
    F[np.arange(s), triggers] = True
    R = np.full((s, n), -1, dtype=int)
    R[np.arange(s), triggers] = 0
    Et = E.T.astype(float)
    k = 0
    while max_rounds is None or k < max_rounds:
        k += 1
        loss = lgd * (F.astype(float) @ Et)  # loss[s, i] = lgd * sum_j F[s, j] E[i, j]
        newly = (~F) & (loss >= capital[None, :])
        if not newly.any():
            break
        F |= newly
        R[newly] = k
    loss = lgd * (F.astype(float) @ Et)
    with np.errstate(divide="ignore", invalid="ignore"):
        L = np.where(capital[None, :] > 0, np.minimum(loss / capital[None, :], 1.0), 0.0)
    L[np.arange(s), triggers] = 1.0
    return F, R, L


def summarise(F: np.ndarray, R: np.ndarray) -> dict:
    """Statistics over the scenarios of one network. The trigger is never counted."""
    n = F.shape[1]
    size = F.sum(axis=1) - 1  # banks failed by contagion
    return {
        "share_contagion": float((size > 0).mean()),
        "mean_cascade": float(size.mean()),
        "worst_cascade": int(size.max()),
        "share_systemic": float((size > 0.05 * n).mean()),  # more than 5 % of banks fail
        "mean_rounds": float(R.max(axis=1).mean()),
    }


def fragility(F: np.ndarray) -> np.ndarray:
    """Share of the scenarios started by ANOTHER bank in which each bank fails."""
    s, n = F.shape
    assert s == n, "fragility needs one scenario per bank, in order"
    return (F.sum(axis=0) - np.diag(F)) / (n - 1)


def hit_share(L: np.ndarray, level: float = 0.10) -> np.ndarray:
    """Share of the scenarios started by another bank in which a bank loses more
    than `level` of its capital (the definition used in ACPR Debats no. 15)."""
    s, n = L.shape
    assert s == n, "hit_share needs one scenario per bank, in order"
    H = L > level
    return (H.sum(axis=0) - np.diag(H)) / (n - 1)


def importance(F: np.ndarray) -> np.ndarray:
    """Number of other banks that fail when each bank is the trigger."""
    return F.sum(axis=1) - 1
