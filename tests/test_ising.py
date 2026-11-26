"""Unit tests for ising.py. Fixed seeds."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
import ising  # noqa: E402

TC = ising.tc_exact()


def test_energy_matches_bruteforce_and_ground_state():
    rng = np.random.default_rng(1)
    s = rng.choice(np.array([-1, 1], dtype=np.int8), size=(6, 6))
    L = 6
    brute = 0.0
    for i in range(L):
        for j in range(L):
            brute += -config.J * s[i, j] * s[i, (j + 1) % L]
            brute += -config.J * s[i, j] * s[(i + 1) % L, j]
    assert ising.energy(s) == pytest.approx(brute)
    up = np.ones((6, 6), dtype=np.int8)
    assert ising.energy(up) == -2 * 36


def test_checkerboard_only_chosen_colour_changes():
    L = 8
    masks = ising.colour_masks(L)
    rng = np.random.default_rng(2)
    s = rng.choice(np.array([-1, 1], dtype=np.int8), size=(L, L))
    for c in (0, 1):
        before = s.copy()
        ising.metropolis_half_step(s, 100.0, masks[c], rng)
        changed = before != s
        assert not np.any(changed & ~masks[c])
        assert np.any(changed & masks[c])


@pytest.mark.parametrize("T", [2.0, TC, 3.0])
def test_metropolis_vs_exact_enumeration_4x4(T):
    exact = ising.exact_enumeration(4, T)
    rng = ising.make_rng(0, 4, 0)
    out = ising.run_metropolis(4, T, 20000, 400000, rng)
    blocks = ising.block_means(out["E"], out["M"])
    N = 16
    est, err = ising.estimators_with_errors(blocks, N, T)
    for k in ["e_mean", "absm_mean", "C", "chi_c", "U"]:
        pull = (est[k] - exact[k]) / err[k]
        assert abs(pull) < 4, (T, k, est[k], exact[k], err[k])


@pytest.mark.parametrize("T", [2.0, TC, 3.0])
def test_wolff_vs_exact_enumeration_4x4(T):
    exact = ising.exact_enumeration(4, T)
    rng = ising.make_rng(1, 4, 0)
    out = ising.run_wolff(4, T, 2000, 200000, rng)
    blocks = ising.block_means(out["E"], out["M"])
    est, err = ising.estimators_with_errors(blocks, 16, T)
    for k in ["e_mean", "absm_mean"]:
        pull = (est[k] - exact[k]) / err[k]
        assert abs(pull) < 4, (T, k, est[k], exact[k], err[k])


def test_wolff_energy_bookkeeping_matches_direct_energy():
    rng = ising.make_rng(1, 6, 3)
    out = ising.run_wolff(6, 2.3, 10, 50, rng)
    s = out["spins"]
    assert out["E"][-1] == ising.energy(s)
    assert out["M"][-1] == ising.magnetization(s)


def test_detailed_balance_sanity():
    L = 8
    masks = ising.colour_masks(L)
    rng = np.random.default_rng(3)
    s = np.ones((L, L), dtype=np.int8)
    flips = 0
    n = 200
    for _ in range(n):
        flips += ising.metropolis_sweep(s, 1e6, rng, masks)
    assert flips / (n * L * L) > 0.999
    s = np.ones((L, L), dtype=np.int8)
    total = 0
    for _ in range(100):
        total += ising.metropolis_sweep(s, 0.05, rng, masks)
    assert total == 0


def test_reproducibility_and_distinct_streams():
    a = ising.run_metropolis(8, 2.3, 20, 50, ising.make_rng(0, 8, 5))
    b = ising.run_metropolis(8, 2.3, 20, 50, ising.make_rng(0, 8, 5))
    assert np.array_equal(a["E"], b["E"]) and np.array_equal(a["M"], b["M"])
    r1 = ising.make_rng(0, 8, 5).random(10)
    r2 = ising.make_rng(0, 8, 6).random(10)
    assert not np.allclose(r1, r2)
    c = ising.run_wolff(8, 2.3, 20, 50, ising.make_rng(1, 8, 5))
    d = ising.run_wolff(8, 2.3, 20, 50, ising.make_rng(1, 8, 5))
    assert np.array_equal(c["E"], d["E"]) and np.array_equal(c["sizes"], d["sizes"])


def test_autocorrelation_time_ar1():
    rng = np.random.default_rng(4)
    n, phi = 400000, 0.9
    eps = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = eps[0]
    for t in range(1, n):
        x[t] = phi * x[t - 1] + eps[t]
    tau, W, ok = ising.integrated_autocorr_time(x)
    assert ok
    assert abs(tau - 9.5) / 9.5 < 0.10


def test_jackknife_normal_data():
    rng = np.random.default_rng(5)
    n = 2000
    x = rng.normal(0.0, 2.0, size=(n, 1))
    se = ising.jackknife_se(x, lambda m: m[..., 0])
    assert abs(se - 2.0 / np.sqrt(n)) / (2.0 / np.sqrt(n)) < 0.20


def test_exact_formulas():
    assert TC == pytest.approx(2.0 / np.log(1.0 + np.sqrt(2.0)))
    assert ising.m0_exact(TC) == 0.0
    assert ising.m0_exact(3.0) == 0.0
    assert 0.99 < ising.m0_exact(1.0) < 1.0
    assert abs(ising.u_exact(TC * (1 - 1e-9)) + np.sqrt(2.0)) < 1e-6
    assert ising.u_exact(TC) == pytest.approx(-np.sqrt(2.0))
    assert abs(ising.u_exact(0.5) + 2.0) < 1e-3


def test_binder_limits_on_synthetic_data():
    m = np.array([1.0, -1.0, 1.0, -1.0])
    means = np.array([[0.0, 0.0, np.mean(np.abs(m)), np.mean(m ** 2), np.mean(m ** 4)]])
    assert ising.estimators(means, 1, 1.0)["U"][0] == pytest.approx(2.0 / 3.0)
    rng = np.random.default_rng(6)
    g = rng.normal(0.0, 0.1, 2000000)
    means = np.array([[0.0, 0.0, np.mean(np.abs(g)), np.mean(g ** 2), np.mean(g ** 4)]])
    assert abs(ising.estimators(means, 1, 1.0)["U"][0]) < 0.02


def test_crossing_rule():
    T = [1.0, 2.0, 3.0, 4.0, 5.0]
    # one sign change between T=2 and T=3: D = -1 -> +1, crossing at 2.5
    assert ising.binder_crossing(T, [0, 0, 1, 1, 1], [1, 1, 0, 0, 0]) == pytest.approx(2.5)
    # two sign changes: D = [-1, 1, 1, -1, 3]; last (<=0 then >0) is between T=4 and T=5
    Ua = [0, 2, 2, 0, 4]
    Ub = [1, 1, 1, 1, 1]
    assert ising.binder_crossing(T, Ua, Ub) == pytest.approx(4.25)  # D goes -1 -> 3 between T=4 and T=5, zero at 4 + 1/4
    # none
    assert np.isnan(ising.binder_crossing(T, [1, 1, 1, 1, 1], [0, 0, 0, 0, 0]))
    assert np.isnan(ising.binder_crossing(T, [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]))


def test_scaling_fit_synthetic():
    L = np.array([8, 16, 32, 64])
    assert ising.loglog_slope(L, 3.0 * L ** 1.75) == pytest.approx(1.75, abs=1e-9)


def test_bootstrap_reproducible_and_runs():
    rng = np.random.default_rng(7)
    blocks = rng.normal(size=(50, 5)) + 10.0
    f = lambda means: float(means[0][2])
    a = ising.bootstrap([blocks], f, n_boot=100)
    b = ising.bootstrap([blocks], f, n_boot=100)
    assert a == b
    assert a["n_nan"] == 0 and a["se"] > 0


# ---- seed replicates (added after the first version, see UPGRADE_PLAN.md) ----

def test_replicate_run_indices_never_collide_with_the_first_version():
    import run_replicates as rr
    first_version = set(range(25)) | {999}
    seen = set()
    for rep in rr.REPLICATES:
        for T in config.T_FINE:
            idx = rr.run_index_for(rep, T)
            assert idx not in first_version
            assert idx not in seen
            seen.add(idx)


def test_replicate_streams_are_reproducible_and_differ_between_replicates():
    import run_replicates as rr
    T = config.T_FINE[3]
    a = ising.make_rng(config.ALGO_METROPOLIS, 16, rr.run_index_for(1, T)).random(5)
    b = ising.make_rng(config.ALGO_METROPOLIS, 16, rr.run_index_for(1, T)).random(5)
    c = ising.make_rng(config.ALGO_METROPOLIS, 16, rr.run_index_for(2, T)).random(5)
    assert np.array_equal(a, b)
    assert a[0] != c[0]


def test_identical_replicates_give_zero_spread_and_no_division_by_zero():
    import analyze_replicates as ar
    sem, dsem, same, verdict = ar.verdict([2.26, 2.26, 2.26, 2.26], 2.26)
    assert sem == 0.0
    assert np.isnan(dsem)
    assert verdict == "consistent with noise"


def test_verdict_rule_on_two_hand_worked_examples():
    import analyze_replicates as ar
    tc = config.TC_EXACT  # 2.26919
    # Case 1: crossings 2.260, 2.261, 2.262, 2.263 (mean 2.2615).
    #   deviations from the mean: -1.5e-3, -0.5e-3, +0.5e-3, +1.5e-3, so the sum of squares is 5.0e-6,
    #   the sample variance is 5.0e-6 / 3 = 1.667e-6 and the SD is 1.291e-3.
    #   sem = SD / sqrt(4) = 6.45e-4, and 3 sem = 1.94e-3.
    #   The pooled crossing 2.2615 is 7.69e-3 below Tc, which is more than 1.94e-3.
    #   All four crossings are below Tc (2.26919). So the verdict is a finite-size effect.
    sem, dsem, same, verdict = ar.verdict([2.260, 2.261, 2.262, 2.263], 2.2615, tc)
    assert same and abs(sem - 6.455e-4) < 1e-6 and verdict == "finite-size effect"
    # Case 2: crossings 2.265, 2.272, 2.268, 2.271 (mean 2.269).
    #   2.265 is below Tc and 2.272 is above Tc, so they are not all on the same side.
    #   The pooled crossing 2.269 is 1.9e-4 from Tc, far less than 3 sem (3 x 1.65e-3 = 4.9e-3).
    #   So the verdict is consistent with noise on both rules.
    sem, dsem, same, verdict = ar.verdict([2.265, 2.272, 2.268, 2.271], 2.269, tc)
    assert not same and verdict == "consistent with noise"
