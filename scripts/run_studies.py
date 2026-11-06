"""Run the option studies of the paper and write their tables to results/.

    PYTHONPATH=src python scripts/run_studies.py

Writes Table 3 (option sigma by hazard), the cost per surviving capacity and
break-even table, the Monte Carlo summary, a storage sweep and Sobol indices.
"""

from __future__ import annotations

import csv
import math
from dataclasses import replace
from pathlib import Path

import numpy as np

from resdesal import COSTS, GCRP600, required_sigma
from resdesal import economics as ec
from resdesal.architectures import HAZARDS, LABELS, MAX_SHARE, Spread, build, combine, option_sigma
from resdesal.montecarlo import evaluate, run, sample_inputs

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
N_DRAWS = 50_000
SEED = 20261102
DC_BASE = (0.4, 0.4)  # P12 base. D_c is a policy choice, swept on its own

P = GCRP600


def share_needed(key: str, hazard_key: str, duration: float, which: str = "base") -> float:
    """Smallest share of P2 in ``key`` that gives the sigma a ``duration``-day outage needs."""
    target = required_sigma(duration, P.storage, P.critical_demand)
    hazard = HAZARDS[hazard_key]
    cap = MAX_SHARE.get(key, 1.0)
    if option_sigma(build(key, cap), hazard, P.critical_demand, which) < target:
        return math.inf
    lo, hi = 0.0, cap
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if option_sigma(build(key, mid), hazard, P.critical_demand, which) >= target:
            hi = mid
        else:
            lo = mid
    return hi


def sized_share(key: str) -> float:
    """Share of P2 for a 55-day severe bloom at base values, rounded up to whole units.

    RO trains are 1/16 of P2 (P5), MED units 29,630 m³/d (M1) and MSF units 92,600 m³/d (F1).
    A2 is capped at 12.5% (E14); A3 cannot be sized this way and puts half the plant at a second site.
    """
    s = share_needed(key, "severe_bloom", 55.0)
    if key == "A3" or math.isinf(s):
        return {"A2": MAX_SHARE["A2"], "A3": 0.5}[key]
    unit = unit_share(key)
    return math.ceil(s / unit - 1e-9) * unit


def unit_share(key: str) -> float:
    """Share of P2 in one unit of the option's protected or diverse part."""
    if key in ("A4", "A5"):
        return P.med_unit_capacity / P.capacity
    if key in ("A6", "A7"):
        return P.msf_unit_capacity / P.capacity
    return P.train_capacity / P.capacity


SHARE = {k: sized_share(k) for k in ("A1", "A2", "A3", "A4", "A5", "A6", "A7")}
OPTIONS = {
    "A0": build("A0"),
    **{k: build(k, v) for k, v in SHARE.items()},
    "A1+A2": combine(("A1", SHARE["A1"]), ("A2", SHARE["A2"])),
    "A2+A5": combine(("A2", SHARE["A2"]), ("A5", SHARE["A5"])),
    "A2+A7": combine(("A2", SHARE["A2"]), ("A7", SHARE["A7"])),
}


def table3() -> list[dict]:
    """sigma by option and hazard at the low, base and high node factors.

    Where a binary (Bernoulli) node is on an option's path, low and high are the two physical
    states and base is the expected value; the column {hazard}_binary flags this.
    """
    rows = []
    for key, option in OPTIONS.items():
        row = {"option": key, "label": option.label, "share": option.share, "extra_capex_musd": option.extra_capex / 1e6}
        on_path = set().union(*(a.path for a in option.assets))
        for h, hazard in HAZARDS.items():
            for which in ("low", "base", "high"):
                row[f"{h}_{which}"] = option_sigma(option, hazard, P.critical_demand, which)
            row[f"{h}_binary"] = any(s.kind == "bernoulli" for n, s in hazard.factors.items() if n in on_path)
        if "+" not in key and key != "A0":
            for T in (12.0, 55.0):
                row[f"share_bloom_{T:.0f}d"] = share_needed(key, "severe_bloom", T)
        rows.append(row)
    return rows


def cost_table() -> list[dict]:
    cs_cap, cs_lc = COSTS.storage_capex, ec.storage_lifecycle()
    rows = []
    for m in ec.measures():
        rows.append({
            "measure": m.label,
            "basis": m.basis,
            "capex_per_installed": m.capex,
            "survival": m.survival,
            "capex_per_surviving": m.capex_per_surviving(),
            "tstar_capex_d": m.capex_per_surviving() / cs_cap,
            "lifecycle_per_surviving": m.lifecycle_per_surviving(),
            "tstar_lifecycle_d": m.lifecycle_per_surviving() / cs_lc,
        })
    return rows


def gas_sensitivity() -> list[dict]:
    """A5 and A7 at base load: lifecycle T* across the E9 gas price range."""
    rows = []
    for key, tech in (("A5", "med"), ("A7", "msf")):
        measure = next(m for m in ec.measures() if m.label.startswith(f"{key} Hybrid"))
        for gas in (1.25, 2.15, 3.0):
            m = replace(measure, energy=ec.med_energy_premium("boiler", gas_price=gas, tech=tech))
            rows.append({"option": key, "gas_usd_mmbtu": gas, "energy_premium_usd_m3": m.energy,
                         "tstar_lifecycle_d": m.lifecycle_per_surviving() / ec.storage_lifecycle()})
    return rows


def monte_carlo(n: int = N_DRAWS, dc_share: tuple[float, float] = DC_BASE) -> list[dict]:
    """Scenario coverage (share of draws meeting D_c) and unserved water (m³) by option and hazard."""
    rows = []
    for h, hazard in HAZARDS.items():
        x = sample_inputs(hazard, n, SEED, dc_share=dc_share)
        for key, option in OPTIONS.items():
            d = evaluate(option, hazard, x)
            u = d.unserved
            rows.append({"hazard": h, "option": key, "p_met": d.met.mean(),
                         "mean_unserved_m3": u.mean(), "p95_unserved_m3": np.quantile(u, 0.95)})
    return rows


def storage_sweep(days=np.round(np.geomspace(0.5, 40, 25), 3), n: int = 20_000) -> list[dict]:
    """P(critical demand met) against tank size, in days of base D_c."""
    rows = []
    for h in ("severe_bloom", "short_bloom", "oil", "oil_fouled"):
        hazard = HAZARDS[h]
        x = sample_inputs(hazard, n, SEED, dc_share=DC_BASE)
        for b in days:
            plant = replace(P, storage=b * P.critical_demand)
            for key, option in OPTIONS.items():
                d = evaluate(option, hazard, x, plant)
                rows.append({"hazard": h, "option": key, "storage_days": b, "p_met": d.met.mean()})
    return rows


def dc_sweep(shares=np.round(np.arange(0.2, 0.61, 0.05), 2), n: int = 20_000) -> list[dict]:
    """P(critical demand met) against the critical demand share of P2."""
    rows = []
    for h in ("severe_bloom", "short_bloom"):
        hazard = HAZARDS[h]
        for share in shares:
            x = sample_inputs(hazard, n, SEED, dc_share=(share, share))
            for key, option in OPTIONS.items():
                rows.append({"hazard": h, "option": key, "dc_share": share,
                             "p_met": evaluate(option, hazard, x).met.mean()})
    return rows


def pretreatment_threshold(durations=(12.0, 55.0, 120.0), shares=np.round(np.arange(0.2, 0.61, 0.05), 2)) -> list[dict]:
    """Smallest bloom survival factor f of DAF+UF at which retrofitting every train meets D_c.

    With every train retrofitted, sigma = P2 a_RO f / D_c, so f* = sigma_req D_c / (P2 a_RO).
    Above f* strengthening RO is enough on its own; below it, diversity is needed.
    """
    from resdesal.architectures import A_RO

    rows = []
    for T in durations:
        for share in shares:
            d_c = share * P.capacity
            f = required_sigma(T, P.storage, d_c) * d_c / (P.capacity * A_RO)
            rows.append({"duration_d": T, "dc_share": share, "f_star": f})
    return rows


def sobol_inputs(option_key: str, hazard_key: str, n_base: int = 4096) -> tuple[dict, dict]:
    """Saltelli design whose inputs follow the same laws as the Monte Carlo (sample_inputs).

    Each design column is a uniform variate on [0, 1) mapped through the input's own law:
    log-uniform durations on a log scale, uniform inputs linearly, and a Bernoulli node to
    its binary state with the base probability. Only nodes on the option's paths enter.
    """
    from SALib.sample import sobol as sample

    hazard, option = HAZARDS[hazard_key], OPTIONS[option_key]
    on_path = set().union(*(a.path for a in option.assets))
    spreads = {"duration": hazard.duration, "recovery": hazard.recovery, "fill": Spread(0.5, 0.75, 1.0)}
    spreads |= {n: s for n, s in hazard.factors.items() if n in on_path}
    spreads |= {"restart_fraction": hazard.restart_fraction, "derate": hazard.derate}
    names = [n for n, s in spreads.items() if s.low != s.high]
    problem = {"num_vars": len(names), "names": names, "bounds": [[0.0, 1.0]] * len(names)}
    u = sample.sample(problem, n_base, calc_second_order=False, seed=SEED)
    x = {}
    for i, name in enumerate(names):
        s = spreads[name]
        if s.kind == "loguniform":
            x[name] = np.exp(np.log(s.low) + u[:, i] * (np.log(s.high) - np.log(s.low)))
        elif s.kind == "bernoulli":
            x[name] = (u[:, i] < s.base).astype(float)
        else:
            x[name] = s.low + u[:, i] * (s.high - s.low)
    for name, s in spreads.items():
        x.setdefault(name, np.full(len(u), s.base))
    for node, s in hazard.factors.items():
        x.setdefault(node, np.full(len(u), s.base))
    x["dc_share"] = np.full(len(u), DC_BASE[0])
    return problem, x


def sobol(option_key: str, hazard_key: str, n_base: int = 4096) -> list[dict]:
    """Sobol indices of unserved critical water U (m³), with the Monte Carlo's input laws."""
    from SALib.analyze import sobol as analyze

    problem, x = sobol_inputs(option_key, hazard_key, n_base)
    y = evaluate(OPTIONS[option_key], HAZARDS[hazard_key], x).unserved
    res = analyze.analyze(problem, y, calc_second_order=False, seed=SEED)
    return [{"input": nm, "S1": s1, "S1_conf": c1, "ST": st, "ST_conf": ct, "share_nonzero": float(np.mean(y > 0))}
            for nm, s1, c1, st, ct in zip(problem["names"], res["S1"], res["S1_conf"], res["ST"], res["ST_conf"])]


def write_csv(name: str, rows: list[dict]) -> None:
    OUT.mkdir(exist_ok=True)
    with open(OUT / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def robust_sizing(n: int = N_DRAWS) -> list[dict]:
    """Size A1, A4, A5, A6 and A7 so that sigma is met at the pessimistic end of the survival range.

    For each bloom type and design outage (55 d severe, 12 d short), the share is
    rounded up to whole units. The Monte Carlo then tests the sized option.
    """
    rows = []
    for h, T in (("severe_bloom", 55.0), ("short_bloom", 12.0)):
        hazard = HAZARDS[h]
        x = sample_inputs(hazard, n, SEED, dc_share=DC_BASE)
        for key in ("A1", "A4", "A5", "A6", "A7"):
            s = share_needed(key, h, T, which="low")
            row = {"hazard": h, "design_days": T, "option": key, "share": s}
            if not math.isinf(s):
                unit = unit_share(key)
                s = math.ceil(s / unit - 1e-9) * unit
                option = build(key, s)
                row.update(share=s, units=round(s / unit), extra_capex_musd=option.extra_capex / 1e6,
                           p_met=evaluate(option, hazard, x).met.mean())
            rows.append(row)
    return rows


def a3_conditional(n: int = N_DRAWS) -> list[dict]:
    """A3 coverage conditional on site B escaping or not, so the escape probability p stays separate.

    Overall coverage is p * P(met | escapes) + (1 - p) * P(met | hit); the rows give it at the
    low, base and high p of each hazard.
    """
    rows, option = [], OPTIONS["A3"]
    for h in ("severe_bloom", "short_bloom", "oil", "oil_fouled"):
        hazard = HAZARDS[h]
        x = sample_inputs(hazard, n, SEED, dc_share=DC_BASE)
        met = {}
        for state in (0.0, 1.0):
            xs = {**x, "site_b": np.full(n, state)}
            met[state] = evaluate(option, hazard, xs).met.mean()
        s = hazard.factors["site_b"]
        row = {"hazard": h, "p_met_if_hit": met[0.0], "p_met_if_escapes": met[1.0]}
        for which in ("low", "base", "high"):
            p = getattr(s, which)
            row[f"p_escape_{which}"] = p
            row[f"p_met_at_p_{which}"] = p * met[1.0] + (1 - p) * met[0.0]
        rows.append(row)
    return rows


def oil_thermal_sensitivity(n: int = N_DRAWS) -> list[dict]:
    """Oil hazards if MED or MSF keeps running with oil in the shared intake (Umm Al Nar 1988, O5).

    The thermal asset's open-intake node is replaced by a node that keeps 50–100% of output
    under oil; everything else, including the draws, is as in the base case.
    """
    rows = []
    for h in ("oil", "oil_fouled"):
        hazard = HAZARDS[h]
        hz = replace(hazard, factors={**hazard.factors, "thermal_oil": Spread(0.5, 1.0, 1.0)})
        x = sample_inputs(hz, n, SEED, dc_share=DC_BASE)
        for key in ("A0", "A4", "A5", "A2+A5", "A6", "A7", "A2+A7"):
            option = OPTIONS[key]
            assets = tuple(replace(a, path=(a.path - {"open_intake"}) | {"thermal_oil"}) if a.name in ("MED", "MSF") else a
                           for a in option.assets)
            rows.append({"hazard": h, "option": key, "p_met_base": evaluate(option, hz, x).met.mean(),
                         "p_met_thermal_runs": evaluate(replace(option, assets=assets), hz, x).met.mean()})
    return rows


def matched_thermal(n: int = N_DRAWS) -> list[dict]:
    """MED and MSF at the same thermal share (1-3 MSF units' worth), so that coverage differences come only
    from response and timing and cost differences only from cost."""
    rows = []
    for units in (1, 2, 3):
        share = units * P.msf_unit_capacity / P.capacity
        for h in ("severe_bloom", "short_bloom", "oil"):
            x = sample_inputs(HAZARDS[h], n, SEED, dc_share=DC_BASE)
            for key in ("A4", "A5", "A6", "A7"):
                option = build(key, share)
                rows.append({"msf_units_equivalent": units, "share": share, "hazard": h, "option": key,
                             "p_met": evaluate(option, HAZARDS[h], x).met.mean(),
                             "extra_capex_musd": option.extra_capex / 1e6})
    return rows


def main() -> None:
    write_csv("table3_sigma.csv", table3())
    write_csv("robust_sizing.csv", robust_sizing())
    write_csv("costs_breakeven.csv", cost_table())
    write_csv("gas_sensitivity.csv", gas_sensitivity())
    write_csv("mc_summary.csv", monte_carlo())
    write_csv("mc_summary_dc_range.csv", monte_carlo(dc_share=(0.2, 0.6)))
    write_csv("storage_sweep.csv", storage_sweep())
    write_csv("dc_sweep.csv", dc_sweep())
    write_csv("pretreatment_threshold.csv", pretreatment_threshold())
    write_csv("oil_thermal_sensitivity.csv", oil_thermal_sensitivity())
    write_csv("a3_conditional.csv", a3_conditional())
    write_csv("matched_thermal.csv", matched_thermal())
    for key in ("A1", "A5", "A7"):
        for h in ("severe_bloom", "short_bloom"):
            write_csv(f"sobol_{key}_{h}.csv", sobol(key, h))
    print(f"Wrote results to {OUT}")


if __name__ == "__main__":
    main()
