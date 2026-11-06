"""MSF options A6/A7 and the linked MED–MSF draws ."""

from dataclasses import replace

import numpy as np
import pytest

from resdesal import economics as ec
from resdesal.architectures import A_MSF, HAZARDS, LINKED, Spread, asset_output, base_factors, build, combine
from resdesal.montecarlo import draw, sample_inputs
from resdesal.params import COSTS, GCRP600


def test_msf_options_replace_ro():
    for key, extra in (("A6", "ccgt_steam"), ("A7", "boiler")):
        opt = build(key, 0.3)
        ro, msf = opt.assets
        assert ro.capacity + msf.capacity == pytest.approx(GCRP600.capacity)
        assert msf.name == "MSF" and msf.availability == A_MSF
        assert msf.path == frozenset({"open_intake", "msf_process", extra, "grid"})
    k = 0.3 * GCRP600.capacity
    assert build("A6", 0.3).extra_capex == pytest.approx((COSTS.msf_capex - COSTS.swro_capex) * k)
    assert build("A7", 0.3).extra_capex == pytest.approx(build("A6", 0.3).extra_capex + ec.boiler_capex("msf") * k)


def test_msf_boiler_and_energy():
    # 0.111 t/m³ × 2.30 GJ/t / 0.90 = 3.29 kW of fuel per m³/d, at 120 USD/kW
    assert ec.boiler_capex("msf") == pytest.approx(394.4, abs=0.5)
    assert ec.boiler_capex() == pytest.approx(404.4, abs=0.5)  # MED unchanged (X2)
    assert ec.med_energy_premium("ccgt", tech="msf") > ec.med_energy_premium("ccgt")


def test_msf_lost_with_open_intake_and_shares_ccgt_steam():
    a6 = build("A6", 0.3)
    msf = a6.assets[1]
    assert asset_output(msf, base_factors(HAZARDS["oil"])) == 0.0
    assert asset_output(msf, base_factors(HAZARDS["ccgt_trip"])) == 0.0
    both = combine(("A4", 0.2), ("A6", 0.3))
    assert {a.name for a in both.assets} == {"RO", "MED", "MSF"}
    assert all(asset_output(a, base_factors(HAZARDS["ccgt_trip"])) == 0.0 for a in both.assets if a.name != "RO")


def test_draw_matches_numpy_samplers():
    n = 10_000
    s = Spread(0.1, 0.4, 0.8)
    assert np.array_equal(draw(s, n, np.random.default_rng(1)), np.random.default_rng(1).uniform(0.1, 0.8, n))
    assert np.array_equal(draw(s, n, np.random.default_rng(1), "triangular"),
                          np.random.default_rng(1).triangular(0.1, 0.4, 0.8, n))
    d = Spread(7.0, 55.0, 120.0, "loguniform")
    assert np.array_equal(draw(d, n, np.random.default_rng(1)),
                          np.exp(np.random.default_rng(1).uniform(np.log(7.0), np.log(120.0), n)))


def test_linked_draws_share_one_variate():
    h = HAZARDS["severe_bloom"]
    x = sample_inputs(h, 5_000, seed=3)
    assert np.array_equal(x["med_process"], x["msf_process"])
    xi = sample_inputs(h, 5_000, seed=3, linked=False)
    assert not np.array_equal(xi["med_process"], xi["msf_process"])
    # Different ranges stay comonotonic: the same rank order.
    h2 = replace(h, factors={**h.factors, "msf_process": Spread(0.5, 0.9, 1.0)})
    x2 = sample_inputs(h2, 5_000, seed=3)
    assert np.array_equal(np.argsort(x2["med_process"]), np.argsort(x2["msf_process"]))


def test_design_families_and_msf_costs():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from design_study import costs, family, option_for

    assert [family(0, 0), family(3, 0), family(0, 1), family(2, 1)] == ["ro", "med", "msf", "med_msf"]
    d_c = 0.4 * GCRP600.capacity
    k = GCRP600.msf_unit_capacity
    for variant, boiler in (("A4", 0.0), ("A5", ec.boiler_capex("msf"))):
        cap0, _ = costs(4, 2, 3, variant, 4.5, d_c)
        cap1, _ = costs(4, 2, 3, variant, 4.5, d_c, n=1)
        assert cap1 - cap0 == pytest.approx((COSTS.msf_capex - COSTS.swro_capex + boiler) * k)
    opt = option_for(4, 2, 3, "A4", n=1)
    msf = [a for a in opt.assets if a.name == "MSF"]
    assert len(msf) == 1 and msf[0].capacity == pytest.approx(k) and "ccgt_steam" in msf[0].path
    assert "boiler" in [a for a in option_for(0, 0, 0, "A5", n=2).assets if a.name == "MSF"][0].path


def test_msf_node_does_not_shift_other_draws():
    """Adding the linked MSF node leaves every other draw as it was (A0–A5 results unchanged)."""
    h = HAZARDS["short_bloom"]
    without = replace(h, factors={k: v for k, v in h.factors.items() if k != "msf_process"})
    a, b = sample_inputs(h, 5_000, seed=9), sample_inputs(without, 5_000, seed=9)
    for k in b:
        assert np.array_equal(a[k], b[k]), k
    assert set(LINKED) <= set(h.factors)
