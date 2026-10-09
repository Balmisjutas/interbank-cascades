# interbank-cascades

Default cascades on simulated interbank networks, and a fair test of what a graph neural network (GNN) adds to
simple methods when the task is to find the fragile banks.

This is a small reproduction and an honest comparison. It is not a new method, and it says nothing about real banks:
real bank-to-bank exposures are confidential, so every network here is simulated.

## See it

An interactive page shows the cascades: choose a network, set how strong the banks are, click a bank to make it
fail. Open `docs/index.html` in a browser, or the hosted copy: https://balmisjutas.github.io/interbank-cascades/

[![The interactive page](figures/viewer.png)](https://balmisjutas.github.io/interbank-cascades/)

The page runs the same rule in JavaScript. `node tests/viewer.test.mjs` checks that it gives the same 1,800
cascades as the Python simulator.

## The question

1. When one bank fails, in which conditions do other banks fail after it?
2. Can a model find the banks that are most at risk? Does a GNN do it better than gradient boosting on features
   that anyone can build by hand?

## The cascade rule

`E[i, j]` is what bank `i` lent to bank `j`. A bank fails when its capital is wiped out. When a bank fails, each of
its lenders loses a fixed share (the loss given default, LGD) of what it lent to it:

    C_k(i) = max( C_0(i) - LGD * sum over failed j of E[i, j], 0 )

The rounds repeat until no new bank fails. This is the sequential default algorithm of Furfine (2003), with a fixed
LGD, as written in Banque de France Working Paper 432 (eq. 2.2) and in Upper (BIS Working Paper 234). It is not the
Eisenberg-Noe clearing model. LGD is 60 %, the value used in Working Paper 432.

"Capital" should be read as the buffer above the minimum requirement: a supervisor acts long before capital is zero.

With this rule, a bank fails when the failed share of its lending is at or above its threshold

    tau_i = capital_i / (LGD * interbank assets_i)

So capital, LGD and the size of interbank lending are one parameter, not three, and the rule has no scale
(`tests/test_sim.py` checks both). A common shock that takes the same share `s` of every bank's capital is the same
as multiplying every `tau` by `1 - s`. This is only true for a uniform shock; in the central-bank papers the shock
differs from bank to bank.

## Networks

100 banks and 495 links (density 5 %), lognormal amounts, two families with the same number of banks and links:

- random: every pair is equally likely;
- core-periphery: 10 core banks, densely linked; the other 90 deal mostly with the core.

Each bank draws its own threshold around a mean `tau`, so that banks differ.

## Result 1. Contagion needs weak banks

![regimes](figures/regimes.png)

- Above `tau` = 0.75, one failure almost never brings down more than a few banks.
- At `tau` = 0.4 or below, the worst case takes most of the system (79 to 99 banks of 100, on average over 20
  networks; single networks vary a lot at `tau` = 0.4).
- The core-periphery network has fewer contagious failures than the random one at low `tau`, and more at high `tau`.
- The grey band is a rough range for European banks. It is my own arithmetic on averages from Working Paper 545:
  equity of 4.2 to 4.7 % of assets, loans to banks of 6.8 to 8.3 % of assets, LGD 60 %, so `tau` is between
  4.2 / (0.6 x 8.3) = 0.84 and 4.7 / (0.6 x 6.8) = 1.15. It is a guide, not a measure, and a mean hides what matters
  most: the largest single exposure compared with capital.
- Real banks sit where a single failure does little. The learning task below is therefore run in a stressed regime,
  `tau` = 0.5.
- The EU large-exposure limit makes the same point: one exposure is at most 25 % of Tier 1 capital (CRR article 395,
  with exemptions). With LGD 60 %, a bank at that limit needs 7 failed counterparties to lose all its capital.
  The generator does NOT impose this limit: at `tau` = 0.5 most links are above it.

Sanity check against a known result (Gai and Kapadia 2010; 1,000 banks, equal amounts, `tau` = 0.2, no recovery):

![gai-kapadia](figures/gai_kapadia.png)

Contagion exists only in a window of connectivity (between about 1 and 9 borrowers per bank). Above about 4
borrowers per bank, more links make a crisis rarer, but when it comes it takes nearly every bank: "robust yet
fragile".

## Result 2. With the links, a GNN matches boosting with hand-made neighbour features. It does not beat it

Label: fragility of a bank = share of the 99 scenarios started by another bank in which it fails.
30 training networks, 15 other test networks, 10 seeds for the neural models (boosting with default settings is
deterministic). The protocol and five predictions were committed before the first run (`PREDICTION.md`; a local
commit, with no external timestamp).

Two scores, both computed inside each test network and averaged:

- **order**: among the pairs of banks with different fragility, the share that the model puts in the right order
  (0.5 is chance). This is the main score.
- **Spearman** rank correlation, the score fixed in `PREDICTION.md`. It is kept, but it is sensitive to ties: the
  label takes only 5 to 7 distinct values per network and 34 to 45 % of the banks are at zero, so a model loses
  points for ordering banks that are exactly tied. A perfect score with ties broken at random gets about 0.93.

![ladder](figures/ladder.png)

| Model | What it sees | Order: random | Order: core-periph. | Spearman: random | Spearman: core-periph. |
|---|---|---|---|---|---|
| No model: one count (borrowers whose failure alone is enough) | own exposures | 0.87 | 0.89 | 0.88 | 0.89 |
| Boosting, rung 1 | own threshold | 0.72 | 0.69 | 0.47 | 0.40 |
| Boosting, rung 2 | + size, number of counterparties | 0.77 | 0.83 | 0.57 | 0.68 |
| Boosting, rung 3 | + own exposures | 0.87 | 0.89 | 0.78 | 0.80 |
| MLP | same as rung 3 | 0.86 | 0.89 | 0.74 | 0.79 |
| **GNN**, 2 GraphConv layers (protocol) | same as rung 3, plus the links | **0.90** | **0.97** | 0.79 | 0.89 |
| GNN with early stopping (added after the first run) | same | 0.94 | 0.97 | 0.85 | 0.90 |
| **Boosting, rung 4** | + sum, mean, max, min, weighted mean of the borrowers' features | **0.93** | **0.97** | 0.84 | 0.89 |
| Boosting, rung 5 | + PageRank, hub and authority | 0.93 | 0.97 | 0.84 | 0.89 |
| Reference: the simulator cut after 2 rounds | everything, 2 rounds | 0.95 | 0.97 | 0.95 | 0.96 |

What the numbers say (differences are paired over the 15 test networks, with a 95 % interval):

- **The GNN uses the links.** It beats the MLP that has the same inputs: +0.05 on random networks (0.03 to 0.06)
  and +0.08 on core-periphery networks (0.06 to 0.09). With the links removed at test time it falls to the MLP
  level (0.86 and 0.89). With the links reversed it falls to 0.79 and 0.70.
- **Under the fixed protocol the GNN does not beat boosting with neighbour features.** Core-periphery: level
  (-0.003, from -0.009 to +0.004). Random: behind by 0.03 (0.02 to 0.04).
- **That gap comes from how the GNN was trained, not from the GNN.** The protocol trains for 300 epochs with no
  validation set. With early stopping on 5 held-out training networks, the GNN is level with boosting or slightly
  ahead: +0.011 on random networks (0.002 to 0.021) and +0.006 on core-periphery (0.001 to 0.012). This run was
  added after the first results, and boosting got no tuning either, so I read it as "level", not as a win.
- **One count with no model does as well as boosting on own features.** Most of the signal is the first round.
  The models with links add the later rounds: 36 % (core-periphery) and 66 % (random) of the contagion failures on
  the test networks happen after round 1; 20 % and 48 % after round 2.
- **Centrality adds nothing** once the neighbour features are there (PageRank, hub and authority: +0.002 and 0.000).
- **Another family of networks.** Trained on one family and tested on the other, boosting keeps its score (0.93
  and 0.96). The protocol GNN keeps it one way (0.93) and loses it the other way (0.82, unstable across seeds); the
  MLP with neighbour features fails the same way. The early-stopped GNN keeps it both ways (0.94 and 0.94).
- **Top 10 %.** Precision in the top 10 % shows no measurable difference between the GNN and boosting. Ties make
  this score generous (16 to 18 banks count as "top 10"), so I do not lean on it.
- **Second label** (share of scenarios in which a bank loses more than 10 % of its capital, the definition of ACPR
  Débats économiques et financiers no. 15, which uses a clearing model, not this rule): same order. Rung 3: 0.78
  and 0.89; rung 4: 0.85 and 0.95; GNN: 0.84 and 0.94.

### Control: move the links, keep the old labels

Degree-preserving edge randomisation at test time: pairs of links are swapped, so that every lender keeps its
number of borrowers and the amounts it lent, and every borrower keeps its number of lenders (not the amounts it
owes). Only who is linked to whom changes. The labels stay those of the original network. The reference is the
simulator itself, run on the new links. The x-axis is the share of links that really changed place.

![control](figures/control.png)

- Models that see only the bank itself do not move, as they must.
- Boosting with neighbour features and the GNN both fall as links move. They fall less than the reference
  (0.06 to 0.09 against 0.17 to 0.20 when about 55 to 60 % of the links have moved), because most of their score
  comes from the bank's own features.
- Scored against the labels of the NEW network, both keep their level (0.83 and 0.90 for boosting, 0.80 and 0.90
  for the GNN, Spearman). Both follow the links that they are given.

### Predictions against results

| Prediction (written before the runs) | Result |
|---|---|
| The GNN beats the MLP with the same inputs | Yes |
| The GNN does not beat boosting with neighbour features | Yes under the protocol; level with early stopping |
| Centrality adds little | Yes for PageRank, hub and authority. The first run used an eigenvector score that was silently zero; fixed after review |
| Under randomisation, own-feature models do not move; the others fall about as much as the reference | Partly: they fall, but by less than half of the reference |
| On the 10 % loss label, own exposures are already close to the best score | No: neighbour information adds 0.06 to 0.07 |

## Limits

- Simulated networks. The results describe this generator, not a banking system. The generator ignores the
  large-exposure limit.
- The labels come from a simulator that runs in milliseconds, so a model that imitates it has no practical use
  here. The useful case would be hidden links: supervisors do not see every exposure. Not done.
- One cascade rule: solvency only, fixed LGD. No liquidity hoarding, no fire sales.
- A stressed regime. At realistic capital the label is almost all zeros.
- No tuning of any model beyond the early-stopping run. The first protocol had no validation set; that was a
  weakness of the protocol, found by a review after the first run. The changes made after that review are marked
  in the table and in the code (`post-hoc`).
- The label has few distinct values. The choice of the score changes some conclusions; both scores are shown.

## Prior work

- Furfine (2003), "Interbank exposures: quantifying the risk of contagion"; Upper (2007), BIS Working Paper 234.
- Fourel, Héam, Salakhova, Tavolaro (2013), Banque de France Working Paper 432; Gabrieli, Salakhova, Vuillemey
  (2015), Working Paper 545; Hauton and Héam (2014), ACPR Débats économiques et financiers no. 15.
- Gai and Kapadia (2010), "Contagion in financial networks".
- León, Machado, Sarmiento (2014), Banco de la República, Borradores de Economía 816: hub and authority centrality
  to find "super-spreaders" in the Colombian interbank network (the source of the hub and authority features).
- Balmaseda, Coronado, de Cadenas-Santiago (2023), "Predicting systemic risk in financial systems using Deep Graph
  Learning": GNNs beat classic models, but those models get no network feature and the split is over the nodes of
  one graph.
- Uddin (2026), arXiv 2604.14232: XGBoost is level with the GNN (0.944 against 0.939).
- Hussain et al. (2021) and Fosdick et al. (2018) for the edge-randomisation control.

## Run it

    uv sync
    uv run pytest
    uv run python experiments/01_regimes.py         # about 15 seconds
    uv run python experiments/02_compare.py         # about 10 minutes on a laptop CPU
    uv run python experiments/03_figures.py
    uv run python experiments/05_tables.py          # every number quoted above
    uv run python experiments/04_export_viewer.py   # data of the interactive page
    node tests/viewer.test.mjs

`src/cascades/sim.py` is the cascade rule (about 40 lines for one cascade). `tests/test_sim.py` has four cases
solved by hand and eight properties of the rule.
