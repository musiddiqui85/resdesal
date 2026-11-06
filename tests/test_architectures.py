import math

import numpy as np
import pytest

from resdesal import GCRP600, Outage, critical_deficit, simulate
from resdesal import economics as ec
from resdesal.architectures import MAX_SHARE, HAZARDS, build, combine, option_sigma
from resdesal.montecarlo import evaluate, max_shortfall, run, sample_inputs

P = GCRP600
DC = P.critical_demand


@pytest.mark.parametrize("T, sigma, tau, rec, rho", [
    (12, 0.0, 0, 0, math.inf), (12, 0.6, 2, 5, 1.125), (55, 0.9, 1, 10, 2.5), (12, 1.3, 2, 5, 1.0)])
def test_max_shortfall_matches_single_asset_form(T, sigma, tau, rec, rho):
    got = max_shortfall(T, [sigma], [tau], rec, rho)
    want = critical_deficit(T, sigma, startup_delay=tau, recovery_time=rec, restart_ratio=rho)
    assert float(got) == pytest.approx(want)


def test_max_shortfall_two_assets_against_simulator():
    # 0.5 D_c from day 0, 0.7 D_c from day 3; restart 1.5 D_c over 4 d.
    T, dt = 12.0, 0.01
    t = (np.arange(round((T + 6) / dt)) + 0.5) * dt
    supply = np.where(t < T, 0.5 + 0.7 * (t >= 3), 1.2 + 1.5 * np.clip((t - T) / 4, 0, 1)) * DC
    s0 = 2.0 * DC
    res = simulate(supply, DC, dt=dt, s0=s0, s_max=s0)
    peak = float(max_shortfall(T, [0.5, 0.7], [0.0, 3.0], 4.0, 1.5))
    assert peak == pytest.approx(1.5)  # 3 d at rate 0.5, then surplus
    assert res.min_storage == pytest.approx(s0 - peak * DC, abs=1.0)


def test_max_shortfall_with_derated_restart_against_simulator():
    # sigma 0.2 for 10 d; lost output (2.0 D_c) ramps over 3 d to 30%, held to day 20 after the event.
    T, dt = 10.0, 0.01
    t = (np.arange(round((T + 30) / dt)) + 0.5) * dt
    u = t - T
    supply = np.where(t < T, 0.2, np.where(u < 20, 0.2 + 0.6 * np.clip(u / 3, 0, 1), 2.2)) * DC
    s0 = 15.0 * DC
    res = simulate(supply, DC, dt=dt, s0=s0, s_max=s0)
    peak = float(max_shortfall(T, [0.2], None, 3.0, 2.0, 0.3, 20.0))
    assert peak == pytest.approx(8.0 + 1.5 + 3.4)
    assert res.min_storage == pytest.approx(s0 - peak * DC, abs=1.0)
    # A zero derated period means full output at the end of the ramp: 8 + 1.5 days.
    assert float(max_shortfall(T, [0.2], None, 3.0, 2.0, 0.3, 0.0)) == pytest.approx(9.5)
    # Full output that never reaches critical demand never ends the shortfall.
    assert float(max_shortfall(T, [0.2], None, 3.0, 0.5, 0.3, 0.0)) == math.inf


def test_option_sigmas():
    bloom, oil = HAZARDS["severe_bloom"], HAZARDS["oil"]
    assert option_sigma(build("A0"), bloom, DC) == 0.0
    assert option_sigma(build("A2", 0.4), bloom, DC) == pytest.approx(MAX_SHARE["A2"] * P.capacity * 0.95 / DC)  # capped
    assert option_sigma(build("A5"), bloom, DC) == pytest.approx(0.4 * P.capacity * 0.92 / DC)
    assert option_sigma(build("A5"), oil, DC) == 0.0  # shares the open intake
    assert option_sigma(build("A4"), HAZARDS["ccgt_trip"], DC) == pytest.approx(0.6 * P.capacity * 0.95 / DC)
    for key in ("A0", "A1", "A2", "A3", "A4", "A5"):
        assert option_sigma(build(key), HAZARDS["grid_blackout"], DC) == 0.0


def test_combined_option_keeps_capacity_and_adds_cost():
    combo = combine(("A1", 0.4), ("A2", 0.25))
    assert sum(a.capacity for a in combo.assets) == pytest.approx(P.capacity)
    assert combo.extra_capex == pytest.approx(build("A1", 0.4).extra_capex + build("A2", 0.25).extra_capex)
    oil = option_sigma(combo, HAZARDS["oil"], DC)
    assert oil == pytest.approx(option_sigma(build("A2"), HAZARDS["oil"], DC))


def test_monte_carlo_agrees_with_simulator_on_one_draw():
    hazard, option = HAZARDS["short_bloom"], build("A1")
    x = sample_inputs(hazard, 5, seed=3)
    x["restart_fraction"][:], x["derate"][:] = 1.0, 0.0  # plain ramp; the plateau has its own test
    d = evaluate(option, hazard, x)
    i = 0
    rho = (sum(a.capacity * a.availability for a in option.assets) - d.sigma[i] * d.critical_demand[i]) / d.critical_demand[i]
    ev = Outage(duration=round(d.duration[i], 2), surviving=d.sigma[i] * d.critical_demand[i], capacity=P.capacity,
                critical_demand=d.critical_demand[i], normal_demand=P.normal_demand,
                recovery_time=round(x["recovery"][i], 2), restart_capacity=rho * d.critical_demand[i], tail=1)
    sim = ev.simulate(0.01, s0=d.storage[i], s_max=P.storage)
    x1 = {k: v[i:i + 1] for k, v in x.items()}
    x1["duration"], x1["recovery"] = np.array([ev.duration]), np.array([ev.recovery_time])
    d1 = evaluate(option, hazard, x1)
    assert sim.unserved(d.critical_demand[i]) == pytest.approx(float(d1.unserved[0]), abs=5.0)


def test_common_random_numbers_and_ordering():
    bloom = HAZARDS["severe_bloom"]
    a0, a5 = run(build("A0"), bloom, 4000), run(build("A5"), bloom, 4000)
    assert np.array_equal(a0.duration, a5.duration)
    assert np.all(a5.unserved <= a0.unserved + 1e-6)


def test_economics():
    assert ec.annuity_factor() == pytest.approx(11.654, abs=1e-3)
    assert ec.boiler_fuel_cost() == pytest.approx(0.593, abs=1e-3)
    m = ec.Measure("x", 1000.0, 0.5)
    assert m.capex_per_surviving() == 2000.0 == m.lifecycle_per_surviving()
