import math

import numpy as np
import pytest

from resdesal import GCRP600, Outage, critical_deficit, refill_time, required_sigma, simulate

P = GCRP600
DT = 0.01


def outage(**kw):
    base = dict(
        capacity=P.capacity,
        critical_demand=P.critical_demand,
        normal_demand=P.normal_demand,
    )
    return Outage(**{**base, **kw})


CASES = [
    # duration, sigma, startup_delay, recovery_time, restart_ratio
    (12.0, 0.0, 0.0, 0.0, math.inf),
    (12.0, 0.5, 0.0, 0.0, math.inf),
    (12.0, 0.5, 2.0, 0.0, math.inf),
    (12.0, 0.3, 1.0, 5.0, 1.125),
    (55.0, 0.8, 2.0, 10.0, 2.5),
    (55.0, 0.95, 0.0, 4.0, 0.5),
]


@pytest.mark.parametrize("duration, sigma, delay, rec, rho", CASES)
@pytest.mark.parametrize("phi", [1.0, 0.5])
def test_unserved_matches_closed_form(duration, sigma, delay, rec, rho, phi):
    s0 = phi * P.storage
    q = sigma * P.critical_demand
    restart = None if math.isinf(rho) else rho * P.critical_demand
    ev = outage(
        duration=duration, surviving=q, startup_delay=delay,
        recovery_time=rec, restart_capacity=restart, tail=5.0,
    )
    res = ev.simulate(DT, s0=s0, s_max=P.storage)
    deficit = critical_deficit(duration, sigma, startup_delay=delay, recovery_time=rec, restart_ratio=rho)
    expected = max(0.0, deficit * P.critical_demand - s0)
    assert res.unserved(P.critical_demand) == pytest.approx(expected, abs=1.0)


@pytest.mark.parametrize("duration, delay, rec, rho", [(12.0, 0.0, 0.0, math.inf), (12.0, 2.0, 0.0, math.inf),
                                                      (12.0, 1.0, 5.0, 1.125), (55.0, 0.0, 10.0, 0.5)])
def test_required_sigma_just_avoids_shortfall(duration, delay, rec, rho):
    sigma = required_sigma(duration, P.storage, P.critical_demand,
                           startup_delay=delay, recovery_time=rec, restart_ratio=rho)
    restart = None if math.isinf(rho) else rho * P.critical_demand
    kw = dict(duration=duration, startup_delay=delay, recovery_time=rec, restart_capacity=restart, tail=2.0)
    ok = outage(surviving=sigma * P.critical_demand, **kw).simulate(DT, P.storage, P.storage)
    short = outage(surviving=0.99 * sigma * P.critical_demand, **kw).simulate(DT, P.storage, P.storage)
    assert ok.unserved(P.critical_demand) < 1.0
    assert ok.min_storage == pytest.approx(0.0, abs=1.0)
    assert short.unserved(P.critical_demand) > 1_000.0


def test_mass_balance_and_bounds():
    rng = np.random.default_rng(1)
    supply = rng.uniform(0, 800_000, 2_000)
    demand = rng.uniform(100_000, 600_000, 2_000)
    res = simulate(supply, demand, dt=0.05, s0=300_000, s_max=600_000)
    assert res.storage.min() >= 0.0 and res.storage.max() <= 600_000
    assert np.all(res.delivered <= res.demand + 1e-9)
    inflow = (res.supply - res.curtailed - res.delivered).sum() * res.dt
    assert res.storage[-1] - res.storage[0] == pytest.approx(inflow, abs=1e-3)


def test_loss_is_drawn_from_storage():
    res = simulate(np.full(100, 500_000.0), 500_000.0, dt=0.1, s0=600_000, s_max=600_000, loss=10_000)
    assert res.storage[-1] == pytest.approx(600_000 - 10 * 10_000)


def test_refill_after_recovery():
    ev = outage(duration=12.0, surviving=0.5 * P.critical_demand, tail=30.0)
    res = ev.simulate(DT, s0=P.storage, s_max=P.storage)
    k_end = round(ev.end_of_recovery / DT)
    full = np.flatnonzero(res.storage[k_end:] >= P.storage - 1e-6)[0] * DT
    expected = refill_time(P.storage - res.storage[k_end], P.capacity, P.normal_demand)
    assert full == pytest.approx(expected, abs=DT)
