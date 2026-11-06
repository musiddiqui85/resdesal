import math

import pytest

from resdesal import (
    COSTS,
    GCRP600,
    Asset,
    breakeven_duration,
    bridge_days,
    cover_cost,
    critical_deficit,
    effective_sigma,
    recovery_deficit,
    refill_time,
    required_sigma,
)

S0, DC = GCRP600.storage, GCRP600.critical_demand


def test_storage_bridges_two_and_a_half_days():
    assert bridge_days(S0, DC) == 2.5


@pytest.mark.parametrize(
    "duration, q_needed",
    [(12.0, 190_000.0), (55.0, 229_090.9), (120.0, 235_000.0)],
)
def test_required_sigma_base_case(duration, q_needed):
    # Worked numbers for GCRP-600
    assert required_sigma(duration, S0, DC) * DC == pytest.approx(q_needed, abs=0.1)


def test_startup_delay_raises_sigma_for_short_event():
    assert required_sigma(12.0, S0, DC) == pytest.approx(0.7917, abs=1e-4)
    assert required_sigma(12.0, S0, DC, startup_delay=2.0) == pytest.approx(0.95)


def test_delay_longer_than_storage_cannot_be_covered():
    assert required_sigma(12.0, S0, DC, startup_delay=3.0) == math.inf


@pytest.mark.parametrize("delay", [0.0, 1.0, 2.0])
@pytest.mark.parametrize("rec, rho", [(0.0, math.inf), (5.0, 1.125), (10.0, 2.5), (6.0, 0.4)])
def test_required_sigma_is_on_the_boundary(delay, rec, rho):
    outage = dict(startup_delay=delay, recovery_time=rec, restart_ratio=rho)
    sigma = required_sigma(12.0, S0, DC, **outage)
    assert critical_deficit(12.0, sigma, **outage) <= S0 / DC + 1e-9
    assert critical_deficit(12.0, sigma - 1e-6, **outage) > S0 / DC


def test_recovery_deficit():
    assert recovery_deficit(0.5, 4.0, 1.0) == pytest.approx(0.5)
    assert recovery_deficit(1.2, 4.0, 1.0) == 0.0
    assert recovery_deficit(0.2, 4.0, 0.5) == math.inf


def test_breakeven_range_matches_outline():
    assert breakeven_duration(1_100, 200) == pytest.approx(5.5)
    assert breakeven_duration(2_300, 160) == pytest.approx(14.375)
    assert breakeven_duration(COSTS.med_capex, COSTS.storage_capex) == pytest.approx(7.78, abs=0.01)


@pytest.mark.parametrize(
    "duration, storage_usd, capacity_usd",
    [(55.0, 2.268e9, 0.3207e9), (12.0, 0.4104e9, 0.266e9)],
)
def test_cover_costs_match_outline(duration, storage_usd, capacity_usd):
    c = cover_cost(duration, S0, DC, COSTS.med_capex, COSTS.storage_capex)
    assert c.storage_cost == pytest.approx(storage_usd, rel=1e-3)
    assert c.capacity_cost == pytest.approx(capacity_usd, rel=1e-3)


@pytest.mark.parametrize("delay", [0.0, 2.0])
def test_costs_equal_at_breakeven(delay):
    t_star = breakeven_duration(COSTS.med_capex, COSTS.storage_capex, delay)
    c = cover_cost(t_star, S0, DC, COSTS.med_capex, COSTS.storage_capex, startup_delay=delay)
    assert c.storage_cost == pytest.approx(c.capacity_cost)


def test_common_mode_removes_credit():
    ro = Asset("RO", 600_000, path=frozenset({"open_intake", "dmf", "grid"}))
    med = Asset("MED on CCGT steam", 120_000, path=frozenset({"open_intake", "ccgt_steam"}))
    med_boiler = Asset("MED, own boiler", 120_000, path=frozenset({"open_intake", "boiler"}))
    plant = [ro, med, med_boiler]
    assert effective_sigma(plant, {"dmf"}, DC) == pytest.approx(1.0)  # bloom
    assert effective_sigma(plant, {"open_intake"}, DC) == 0.0  # oil at the intake
    assert effective_sigma(plant, {"grid", "ccgt_steam"}, DC) == pytest.approx(0.5)  # power trip


def test_refill_time():
    assert refill_time(600_000, 600_000, 540_000) == pytest.approx(10.0)
    assert refill_time(600_000, 500_000, 540_000) == math.inf
