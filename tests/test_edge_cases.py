"""Edge cases: the binary second site, Sobol input laws, long start-up delays, the capped intake and short derating."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

from resdesal import GCRP600, critical_deficit, meets_critical_demand
from resdesal.architectures import HAZARDS, build, option_sigma
from resdesal.montecarlo import evaluate, max_shortfall, sample_inputs

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

P = GCRP600
DC = P.critical_demand


def test_second_site_is_a_binary_state_not_a_fraction():
    a3, bloom = build("A3", 0.5), HAZARDS["severe_bloom"]
    unaffected = 0.5 * P.capacity * 0.95 / DC
    assert option_sigma(a3, bloom, DC, "low") == 0.0
    assert option_sigma(a3, bloom, DC, "high") == pytest.approx(unaffected)
    assert option_sigma(a3, bloom, DC, "base") == pytest.approx(0.3 * unaffected)  # expected value only
    x = sample_inputs(bloom, 20_000, seed=1, dc_share=(0.4, 0.4))
    sig = evaluate(a3, bloom, x).sigma
    assert set(np.round(np.unique(sig), 6)) == {0.0, round(unaffected, 6)}
    assert np.mean(sig > 0) == pytest.approx(bloom.factors["site_b"].base, abs=0.01)


def test_sobol_inputs_follow_the_monte_carlo_laws():
    pytest.importorskip("SALib")
    from run_studies import sobol_inputs

    _, x = sobol_inputs("A1", "short_bloom", n_base=256)
    d = x["duration"]
    lo, hi = HAZARDS["short_bloom"].duration.low, HAZARDS["short_bloom"].duration.high
    assert d.min() >= lo and d.max() <= hi
    assert np.mean(d) == pytest.approx((lo + hi) / 2, rel=0.05)  # uniform, not log-uniform
    problem, x = sobol_inputs("A1", "severe_bloom", n_base=256)
    s = HAZARDS["severe_bloom"].duration
    assert np.median(x["duration"]) == pytest.approx(math.sqrt(s.low * s.high), rel=0.1)  # log-uniform
    assert "med_process" not in problem["names"] and "site_b" not in problem["names"]  # not on A1's paths


def test_startup_delay_longer_than_storage_is_infeasible_whatever_sigma():
    # The sigma < 1 condition beta + sigma (1 - theta) >= 1 would wrongly accept this case.
    T, sigma, tau, b = 12.0, 2.0, 3.0, 2.5
    assert b / T + sigma * (1 - tau / T) >= 1
    assert critical_deficit(T, sigma, startup_delay=tau) == tau
    assert not meets_critical_demand(b * DC, DC, T, sigma * DC, startup_delay=tau)


def test_capped_subsurface_intake_bridges_only_a_few_days():
    sigma = option_sigma(build("A2", 0.25), HAZARDS["severe_bloom"], DC, "base")  # capped at 12.5% (E14)
    limit = P.storage / DC / (1 - sigma)
    assert limit == pytest.approx(2.5 / (1 - 75_000 * 0.95 / DC), abs=1e-3)  # about 3.6 days
    assert critical_deficit(limit - 0.01, sigma) <= P.storage / DC < critical_deficit(limit + 0.01, sigma)


def test_derate_shorter_than_ramp_has_no_plateau():
    args = dict(duration=10.0, sigmas=[[0.2]], recovery_time=5.0, restart_ratio=2.0, restart_fraction=0.3)
    short = max_shortfall(derate_time=2.0, **args)
    at_ramp = max_shortfall(derate_time=5.0, **args)
    assert short == pytest.approx(at_ramp)
