# Prediction, written before the learning runs

Date: 2026-10-09. This file is committed before `experiments/02_compare.py` is run for the first time.
The simulator, the regime sweep (`results/regimes.csv`) and the Gai-Kapadia check are done. No model has been trained.

## Fixed in advance

- Threshold for the learning task: the largest tau of the sweep grid for which 20 % to 60 % of the labels are zero in
  both families. The sweep gives tau = 0.5 (40 % zeros on random networks, 32 % on core-periphery networks).
- Label: fragility = share of the 99 scenarios started by another bank in which the bank fails.
  Second label, reported next to it: share of scenarios in which the bank loses more than 10 % of its capital.
- Training networks: seeds 3000 to 3029. Test networks: seeds 4000 to 4014. Model seeds: 0 to 9. No choice is made
  on the test networks.
- Models: gradient boosting with default settings on rungs 1 to 5 of the feature ladder; an MLP on rungs 3 and 4;
  a GNN with two GraphConv layers on the rung 3 features. Hidden size 64, 300 epochs, learning rate 0.01. No tuning.
- Scores: Spearman rank correlation inside each test network, and precision in the top 10 %. Mean over test networks.
- Control: degree-preserving edge randomisation at test time (0, 10, 25, 50, 100 % of the links), labels kept from
  the original network. Reference: the labels that the simulator gives on the randomised network.

## What I expect

1. The GNN beats the MLP on the same rung 3 features. The MLP cannot see the neighbours; the GNN can.
2. The GNN does NOT beat gradient boosting with hand-made neighbour features (rung 4). I expect them to be level,
   within the spread over seeds, or the boosting to be a little ahead.
3. Centrality (rung 5) adds little once the neighbour features are there.
4. Under edge randomisation, the scores of rungs 1 to 3 and of the MLP do not move (their inputs do not change).
   The scores of rung 4 and of the GNN fall, and they fall about as much as the reference.
5. On the 10 % capital-loss label, rung 3 is already close to the best score, because a direct exposure is enough
   to lose 10 % of capital.

Whatever the result is, it goes in the README.
