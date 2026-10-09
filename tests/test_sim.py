"""Small cases solved by hand, then general properties of the rule."""

import numpy as np

from cascades.features import OWN, own_features
from cascades.generate import common_shock, make_network, randomise_edges
from cascades.learn import concordance
from cascades.sim import all_cascades, cascade, fragility, importance

LGD = 0.6


def chain():
    # bank 1 lent 10 to bank 0; bank 2 lent 10 to bank 1
    E = np.zeros((3, 3))
    E[1, 0] = 10
    E[2, 1] = 10
    return E


def test_chain_stops_at_the_strong_bank():
    # bank 0 fails. Bank 1 loses 0.6 * 10 = 6 >= 5: it fails in round 1.
    # Bank 2 then loses 6 < 7: it survives.
    failed, rounds, loss = cascade(chain(), np.array([1.0, 5.0, 7.0]), trigger=0, lgd=LGD)
    assert failed.tolist() == [True, True, False]
    assert rounds.tolist() == [0, 1, -1]
    assert np.isclose(loss[2], 6 / 7)


def test_chain_goes_through_when_capital_equals_the_loss():
    # capital "wiped out" means loss >= capital: 6 >= 6, so bank 2 fails in round 2
    failed, rounds, _ = cascade(chain(), np.array([1.0, 5.0, 6.0]), trigger=0, lgd=LGD)
    assert failed.tolist() == [True, True, True]
    assert rounds.tolist() == [0, 1, 2]


def test_losses_add_up_over_two_failed_borrowers():
    # bank 3 lent 5 to bank 0 and 5 to bank 1, capital 5.
    # After bank 0 fails: loss 3 < 5. Bank 1 fails in round 1 (see above).
    # Then the loss of bank 3 is 0.6 * (5 + 5) = 6 >= 5: it fails in round 2.
    E = np.zeros((4, 4))
    E[1, 0] = 10
    E[3, 0] = 5
    E[3, 1] = 5
    failed, rounds, _ = cascade(E, np.array([1.0, 5.0, 100.0, 5.0]), trigger=0, lgd=LGD)
    assert failed.tolist() == [True, True, False, True]
    assert rounds.tolist() == [0, 1, -1, 2]


def test_a_bank_that_borrowed_from_nobody_hurts_nobody():
    failed, _, _ = cascade(chain(), np.array([1.0, 5.0, 6.0]), trigger=2, lgd=LGD)
    assert failed.tolist() == [False, False, True]


def test_no_contagion_when_every_threshold_is_above_one():
    # a bank cannot lose more than its whole lending book
    net = make_network("er", seed=1, tau_mean=1.3, tau_sigma=0.0)
    F, _, _ = all_cascades(net.E, net.capital, net.lgd)
    assert importance(F).sum() == 0


def test_the_rule_has_no_scale():
    net = make_network("cp", seed=2, tau_mean=0.5)
    F1, _, _ = all_cascades(net.E, net.capital, net.lgd)
    F2, _, _ = all_cascades(1000 * net.E, 1000 * net.capital, net.lgd)
    assert (F1 == F2).all()


def test_lgd_and_capital_are_one_parameter():
    net = make_network("er", seed=3, tau_mean=0.5)
    F1, _, _ = all_cascades(net.E, net.capital, lgd=0.6)
    F2, _, _ = all_cascades(net.E, net.capital / 2, lgd=0.3)
    assert (F1 == F2).all()


def test_more_capital_never_gives_more_failures():
    net = make_network("er", seed=4, tau_mean=0.4)
    F1, _, _ = all_cascades(net.E, net.capital, net.lgd)
    F2, _, _ = all_cascades(net.E, 1.5 * net.capital, net.lgd)
    assert not (F2 & ~F1).any()


def test_batch_version_equals_the_single_version():
    net = make_network("cp", seed=5, tau_mean=0.4)
    F, R, L = all_cascades(net.E, net.capital, net.lgd)
    for t in (0, 7, 55, 99):
        f, r, l = cascade(net.E, net.capital, t, net.lgd)
        assert (F[t] == f).all() and (R[t] == r).all() and np.allclose(L[t], l)


def test_fragility_does_not_count_a_bank_as_its_own_victim():
    F, _, _ = all_cascades(chain(), np.array([1.0, 5.0, 6.0]), LGD)
    # bank 0 never fails by contagion; bank 1 fails in 1 of the 2 scenarios started by
    # another bank (when 0 starts); bank 2 fails in both (when 0 starts and when 1 starts)
    assert np.allclose(fragility(F), [0.0, 0.5, 1.0])


def test_both_families_have_the_same_number_of_links():
    a = make_network("er", seed=6, tau_mean=0.5)
    b = make_network("cp", seed=6, tau_mean=0.5)
    assert (a.E > 0).sum() == (b.E > 0).sum() == 495
    assert np.diag(a.E).sum() == 0 and np.diag(b.E).sum() == 0


def test_edge_randomisation_keeps_what_it_says_it_keeps():
    net = make_network("cp", seed=7, tau_mean=0.5)
    new = randomise_edges(net, 1.0, np.random.default_rng(0))
    assert np.allclose(net.E.sum(axis=1), new.E.sum(axis=1))  # interbank assets of each lender
    assert ((net.E > 0).sum(axis=1) == (new.E > 0).sum(axis=1)).all()  # number of borrowers
    assert ((net.E > 0).sum(axis=0) == (new.E > 0).sum(axis=0)).all()  # number of lenders
    assert np.allclose(np.sort(net.E, axis=1), np.sort(new.E, axis=1))  # each lender's amounts
    assert np.diag(new.E).sum() == 0
    assert ((net.E > 0) != (new.E > 0)).sum() > 300  # and the links really moved


def test_a_common_shock_is_a_lower_threshold():
    net = make_network("er", seed=8, tau_mean=0.8)
    shocked = common_shock(net, 0.25)
    weaker = make_network("er", seed=8, tau_mean=0.6)  # 0.8 * (1 - 0.25)
    lends = net.E.sum(axis=1) > 0  # a bank that lends to nobody cannot be hit; its capital is arbitrary
    assert np.allclose(shocked.tau, 0.75 * net.tau)
    assert np.allclose(shocked.capital[lends], weaker.capital[lends])
    F1, _, _ = all_cascades(shocked.E, shocked.capital, shocked.lgd)
    F2, _, _ = all_cascades(weaker.E, weaker.capital, weaker.lgd)
    assert (F1 == F2).all()


def test_the_hand_count_is_the_first_round():
    # "single killers" of a bank = borrowers whose failure alone wipes out its capital
    net = make_network("cp", seed=9, tau_mean=0.5)
    F, _, _ = all_cascades(net.E, net.capital, net.lgd, max_rounds=1)
    count = own_features(net)[:, OWN.index("n_single_killers")]
    assert np.allclose(count, F.sum(axis=0) - 1)


def test_order_score():
    y = np.array([0.0, 0.0, 1.0, 2.0])
    assert concordance(y, y) == 1.0
    assert concordance(y, -y) == 0.0
    assert concordance(y, np.zeros(4)) == 0.5
    assert concordance(y, np.array([5.0, -5.0, 6.0, 7.0])) == 1.0  # the order inside a tie does not count
