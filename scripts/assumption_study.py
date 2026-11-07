"""Tests of the assumptions that decide the design choice (Sections 3.4 and 3.5 of the paper).

    PYTHONPATH=src python scripts/assumption_study.py [sensitivity] [stress] [map] [grid]

Builds on the equal-service enumeration of scripts/design_study.py and finds, for each
case, the cheapest design in each family (no thermal, MED, MSF, MED and MSF) that meets
critical demand in at least 99% of sampled scenarios of each of severe blooms, short
blooms and precautionary oil closures. Writes:

- results/survival_map.csv: joint map over the output kept by DAF + UF and by the thermal
  units in a short (picoplankton-rich) bloom, both fixed instead of sampled, at D_c 40%, 50%
  and 60%. MSF keeps the same share as MED (the linked draw).
- results/response_grid.csv: the output kept by DAF + UF, MED and MSF each fixed at 0, 0.25,
  0.5, 0.75 or 1 in one bloom type (short or severe), the others sampled as usual.
- results/assumption_sensitivity.csv: tank fill, DAF + UF start-up delay, intake cap,
  an optional second site, two alternative weightings of the input ranges, independent
  MED and MSF draws, and the design space without MSF.
- results/stress_cases.csv: Table 5 designs under a few coherent adverse cases.

The minimum storage that meets a target is found from order statistics: a design meets
coverage X in a hazard with b days of storage when at least X of its draws have
peak shortfall / fill ≤ b. This gives the same answer as design_study.py's loop over
storage levels.
"""

from __future__ import annotations

import csv
import dataclasses
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from design_study import FAMILY, MED_MAX, MSF_MAX, MSF_UNIT, STORAGE_DAYS, TARGET_HAZARDS, TRAIN, costs, family, option_for  # noqa: E402
from run_studies import N_DRAWS, OUT, P, SEED  # noqa: E402

from resdesal import economics as ec  # noqa: E402
from resdesal.architectures import HAZARDS, MAX_SHARE, PREMIUM, combine  # noqa: E402
from resdesal.montecarlo import evaluate, sample_inputs  # noqa: E402
from resdesal.params import COSTS  # noqa: E402

TARGET = 0.99
MED_UNIT = P.med_unit_capacity / P.capacity
BASE_SUB_CAP = MAX_SHARE["A2"]  # E14: 12.5% of capacity
BASE_SUB_TRAINS = round(BASE_SUB_CAP / TRAIN)
MAP_DC = (0.4, 0.5, 0.6)
MAP_N = 20_000
GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)
RESPONSE_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)  # the report's sensitivity levels: settings, not estimates
LABEL = {"ro": "No thermal", "med": "With MED", "msf": "With MSF", "med_msf": "With MED and MSF"}
BASES = ("capital_musd", "lifecycle_musd")


# --- Designs ---------------------------------------------------------------

def design_list(sub_max: int = BASE_SUB_TRAINS, site_b: tuple[int, ...] = (0,), msf_max: int = MSF_MAX,
                med_max: int = MED_MAX):
    """(DAF+UF trains, subsurface trains, MED units, MSF units, second-site trains) with total share ≤ 1."""
    for k1 in range(17):
        for k2 in range(sub_max + 1):
            for k3 in site_b:
                for m in range(med_max + 1):
                    for n in range(msf_max + 1):
                        if (k1 + k2 + k3) * TRAIN + m * MED_UNIT + n * MSF_UNIT <= 1.0 + 1e-9:
                            yield k1, k2, m, n, k3


def option(k1: int, k2: int, m: int, n: int = 0, k3: int = 0, daf_delay: float = 0.0, sub_cap: float = BASE_SUB_CAP):
    """The design as an Option. ``sub_cap`` is the subsurface-intake cap as a share of capacity; the
    modelled subsurface capacity must equal the k2 trains that design_cost charges for."""
    caps = {"A2": sub_cap}
    if k3 == 0:
        opt = option_for(k1, k2, m, max_share=caps, n=n)
    else:
        parts = [(key, s) for key, s in (("A1", k1 * TRAIN), ("A2", k2 * TRAIN), ("A3", k3 * TRAIN),
                                         ("A5", m * MED_UNIT), ("A7", n * MSF_UNIT)) if s > 0]
        opt = combine(*parts, max_share=caps)
    modelled = sum(a.capacity for a in opt.assets if a.name == "RO subsurface")
    if abs(modelled - k2 * P.train_capacity) > 1e-6:
        raise ValueError(f"{k2} subsurface trains charged but {modelled:.0f} m3/d modelled (cap {sub_cap:.3f})")
    if daf_delay > 0 and k1 > 0:
        opt = dataclasses.replace(opt, delays={"RO DAF+UF": daf_delay})
    return opt


def design_cost(k1: int, k2: int, m: int, n: int, k3: int, storage_days: float, d_c: float,
                variant: str = "A4") -> tuple[float, float]:
    """Extra capital and lifecycle cost, USD, with every thermal unit on CCGT steam ("A4") or its own boiler ("A5")."""
    capital, lifecycle = costs(k1, k2, m, variant, storage_days, d_c, n)
    if k3:
        capex_b = PREMIUM["site_b"] * COSTS.swro_capex * k3 * P.train_capacity
        capital += capex_b
        lifecycle += capex_b * (1.0 + ec.FIXED_OM * ec.annuity_factor())
    return capital, lifecycle


# --- Coverage --------------------------------------------------------------

def need_days(peak: np.ndarray, fill: np.ndarray, target: float = TARGET) -> float:
    """Smallest storage, days of D_c, that meets ``target`` of draws."""
    r = peak / fill
    k = math.ceil(target * r.size - 1e-9)
    return float(np.partition(r, k - 1)[k - 1])


def storage_levels(d_c: float) -> list[float]:
    return sorted(b for b in set(STORAGE_DAYS) | {P.storage / d_c} if b >= P.storage / d_c - 1e-12)


def cheapest(need: dict[tuple, float], d_c: float, sub_cap: float = BASE_SUB_CAP) -> dict[tuple[str, str], dict]:
    """Cheapest design meeting the target, per (family, cost basis), from the storage each needs.

    Thermal designs are costed on CCGT steam and on their own boilers, and the cheaper of the two is
    kept for each basis (for MED, CCGT steam is cheaper on both; for MSF, not always).
    """
    levels = storage_levels(d_c)
    best: dict[tuple[str, str], dict] = {}
    for (k1, k2, m, n, k3), b_need in need.items():
        if sum(a.capacity * a.availability for a in option(k1, k2, m, n, k3, sub_cap=sub_cap).assets) < P.normal_demand:
            continue
        b = next((lv for lv in levels if lv >= b_need - 1e-12), None)
        if b is None:
            continue
        fam = family(m, n)
        for variant in (("A4", "A5") if m or n else ("A4",)):
            capital, lifecycle = design_cost(k1, k2, m, n, k3, b, d_c, variant)
            row = dict(daf_uf_trains=k1, subsurface_trains=k2, med_units=m, msf_units=n, site_b_trains=k3,
                       steam={"A4": "CCGT", "A5": "boiler"}[variant] if m or n else "-", storage_days=b,
                       capital_musd=capital / 1e6, lifecycle_musd=lifecycle / 1e6)
            for basis in BASES:
                key = (fam, basis)
                if key not in best or (row[basis], row["capital_musd"]) < (best[key][basis], best[key]["capital_musd"]):
                    best[key] = row
    return best


def needs(dc_share: float, draws: dict, designs, daf_delay: float = 0.0, hazards=TARGET_HAZARDS,
          sub_cap: float = BASE_SUB_CAP, shape: str = "plateau") -> dict:
    """Storage each design needs to meet the target in every hazard, days of D_c."""
    out = {}
    for d in designs:
        opt = option(*d, daf_delay=daf_delay, sub_cap=sub_cap)
        out[d] = max(need_days(evaluate(opt, HAZARDS[h], draws[h], shape=shape).shortfall, draws[h]["fill"])
                     for h in hazards)
    return out


def draws_for(dc_share: float, n: int = N_DRAWS, fill=(0.5, 1.0), law: str = "base", site_b_p: float | None = None,
              linked: bool = True, sub_oil: float | None = None):
    """Draws for the target hazards. ``sub_oil`` fixes the protected intake's output in precautionary oil closures
    (its bloom response is unchanged), separating its oil response from its size."""
    out = {}
    for h in TARGET_HAZARDS:
        x = sample_inputs(HAZARDS[h], n, SEED, dc_share=(dc_share, dc_share), fill=fill, law=law, linked=linked)
        if site_b_p is not None:
            x["site_b"] = (np.random.default_rng(SEED + 7).random(n) < site_b_p).astype(float)
        if sub_oil is not None and h == "oil":
            x["subsurface_intake"] = np.full(n, sub_oil)
        out[h] = x
    return out


def design_code(b: dict) -> str:
    """DAF+UF trains / subsurface trains / MED units / MSF units / storage days."""
    return f"{b['daf_uf_trains']}/{b['subsurface_trains']}/{b['med_units']}/{b['msf_units']}/{b['storage_days']:g}"


def family_columns(best: dict) -> dict:
    """Cost and design of the cheapest design in each family, plus the cheapest family, per basis."""
    row = {}
    for fam in FAMILY:
        for basis in BASES:
            row[f"{fam}_{basis}"] = best[fam, basis][basis] if (fam, basis) in best else float("nan")
        if (fam, "lifecycle_musd") in best:
            row[f"{fam}_design"] = design_code(best[fam, "lifecycle_musd"])
    for basis in BASES:
        found = [(best[f, basis][basis], f) for f in FAMILY if (f, basis) in best]
        row[f"cheapest_{basis}"] = min(found)[1] if found else ""
    return row


# --- Joint survival map ----------------------------------------------------

def _map_point(args):
    dc_share, f_daf, f_med, fixed_need = args
    x = sample_inputs(HAZARDS["short_bloom"], MAP_N, SEED, dc_share=(dc_share, dc_share))
    x["daf_uf"] = np.full(MAP_N, f_daf)
    x["med_process"] = np.full(MAP_N, f_med)
    x["msf_process"] = np.full(MAP_N, f_med)  # linked draw: MSF keeps what MED keeps
    need = {}
    for d, other in fixed_need.items():
        short = need_days(evaluate(option(*d), HAZARDS["short_bloom"], x).shortfall, x["fill"])
        need[d] = max(other, short)
    return {"dc_share": dc_share, "f_daf_uf": f_daf, "f_med": f_med,
            **family_columns(cheapest(need, dc_share * P.capacity))}


def survival_map() -> list[dict]:
    designs = list(design_list())
    jobs = []
    for dc in MAP_DC:
        draws = draws_for(dc, MAP_N)
        fixed_need = needs(dc, draws, designs, hazards=("severe_bloom", "oil"))
        jobs += [(dc, fd, fm, fixed_need) for fd in GRID for fm in GRID]
    with ProcessPoolExecutor() as pool:
        return list(pool.map(_map_point, jobs, chunksize=8))


# --- Response grid over DAF + UF, MED and MSF ------------------------------

def _grid_point(args):
    """One (D_c, bloom, DAF + UF, MED) point, over every MSF level. Designs without MSF are evaluated once."""
    dc_share, h, f_daf, f_med, fixed_need = args
    x = sample_inputs(HAZARDS[h], MAP_N, SEED, dc_share=(dc_share, dc_share))
    x["daf_uf"] = np.full(MAP_N, f_daf)
    x["med_process"] = np.full(MAP_N, f_med)
    rows, no_msf = [], {}
    for f_msf in RESPONSE_GRID:
        x["msf_process"] = np.full(MAP_N, f_msf)
        need = {}
        for d, other in fixed_need.items():
            if d[3] == 0 and d in no_msf:
                need[d] = no_msf[d]
                continue
            need[d] = max(other, need_days(evaluate(option(*d), HAZARDS[h], x).shortfall, x["fill"]))
            if d[3] == 0:
                no_msf[d] = need[d]
        rows.append({"dc_share": dc_share, "hazard": h, "f_daf_uf": f_daf, "f_med": f_med, "f_msf": f_msf,
                     **family_columns(cheapest(need, dc_share * P.capacity))})
    return rows


def response_grid() -> list[dict]:
    designs = list(design_list())
    jobs = []
    for dc in MAP_DC:
        draws = draws_for(dc, MAP_N)
        for h in ("short_bloom", "severe_bloom"):
            fixed_need = needs(dc, draws, designs, hazards=tuple(k for k in TARGET_HAZARDS if k != h))
            jobs += [(dc, h, fd, fm, fixed_need) for fd in RESPONSE_GRID for fm in RESPONSE_GRID]
    with ProcessPoolExecutor() as pool:
        return [r for rows in pool.map(_grid_point, jobs) for r in rows]


# --- Sensitivity cases -----------------------------------------------------

CASES = [
    ("Base case", {}),
    ("Tanks full at onset", {"fill": (1.0, 1.0)}),
    ("Tanks half full at onset", {"fill": (0.5, 0.5)}),
    ("DAF + UF online 1 d after onset", {"daf_delay": 1.0}),
    ("DAF + UF online 2 d after onset", {"daf_delay": 2.0}),
    ("No subsurface intake", {"sub_max": 0}),
    ("Subsurface intake cap 6.25% (one train)", {"sub_max": 1}),
    ("Subsurface intake cap 25% (beyond demonstrated)", {"sub_max": 4}),
    ("Second site allowed, base escape probabilities", {"site_b": (0, 2, 4)}),
    ("Second site allowed, escape probability 0.8", {"site_b": (0, 2, 4), "site_b_p": 0.8}),
    ("Triangular weighting, peak at base values", {"law": "triangular"}),
    ("Triangular weighting, peak at adverse ends", {"law": "adverse"}),
    ("Severe-bloom durations uniform (more long events)", {"law": "uniform_duration"}),
    ("MED and MSF responses drawn independently", {"linked": False}),
    ("No MSF in the design space (before 3 Nov 2026)", {"msf_max": 0}),
    ("MED capped at 12 units (draft v0.5)", {"med_max": 12}),
    ("Protected intake keeps no output in oil closures", {"sub_oil": 0.0}),
    ("Protected intake keeps half its output in oil closures", {"sub_oil": 0.5}),
    ("Rising recovery to the same full-return date", {"shape": "rising"}),
]


def _case(args):
    name, kw, dc = args
    draws = draws_for(dc, N_DRAWS, kw.get("fill", (0.5, 1.0)), kw.get("law", "base"), kw.get("site_b_p"),
                      kw.get("linked", True), kw.get("sub_oil"))
    sub_max = kw.get("sub_max", BASE_SUB_TRAINS)
    sub_cap = max(sub_max, 1) * TRAIN  # the cap equals the largest subsurface size enumerated
    designs = list(design_list(sub_max, kw.get("site_b", (0,)), kw.get("msf_max", MSF_MAX), kw.get("med_max", MED_MAX)))
    best = cheapest(needs(dc, draws, designs, kw.get("daf_delay", 0.0), sub_cap=sub_cap,
                          shape=kw.get("shape", "plateau")), dc * P.capacity, sub_cap)
    return [{"case": name, "dc_share": dc, "family": LABEL[fam], "basis": basis, **b}
            for (fam, basis), b in best.items()]


def sensitivity() -> list[dict]:
    jobs = [(name, kw, dc) for name, kw in CASES for dc in (0.4, 0.5)]
    with ProcessPoolExecutor() as pool:
        return [r for rows in pool.map(_case, jobs) for r in rows]


# --- Coherent adverse (stress) cases ---------------------------------------

STRESS = [
    ("Long severe bloom, weak response", "severe_bloom",
     dict(duration=120.0, recovery=10.0, dmf=0.0, daf_uf=0.5, med_process=0.8, msf_process=0.8, subsurface_intake=0.9,
          site_b=0.0, restart_fraction=0.5, derate=60.0, fill=0.5)),
    ("Mixed bloom, pretreatment fails, slow return", "short_bloom",
     dict(duration=12.0, recovery=7.0, dmf=0.0, daf_uf=0.1, med_process=0.8, msf_process=0.8, subsurface_intake=0.9,
          site_b=0.0, restart_fraction=0.4, derate=40.0, fill=0.5)),
    ("As above, MED and MSF also lose half their output", "short_bloom",
     dict(duration=12.0, recovery=7.0, dmf=0.0, daf_uf=0.1, med_process=0.5, msf_process=0.5, subsurface_intake=0.9,
          site_b=0.0, restart_fraction=0.4, derate=40.0, fill=0.5)),
    ("As above, MED loses half but MSF keeps 80%", "short_bloom",
     dict(duration=12.0, recovery=7.0, dmf=0.0, daf_uf=0.1, med_process=0.5, msf_process=0.8, subsurface_intake=0.9,
          site_b=0.0, restart_fraction=0.4, derate=40.0, fill=0.5)),
    ("Long precautionary closure, tanks half full", "oil",
     dict(duration=3.0, recovery=1.0, open_intake=0.0, subsurface_intake=0.7, site_b=0.0, fill=0.5)),
]


def stress_cases() -> list[dict]:
    with open(OUT / "design_best.csv", encoding="utf-8") as f:
        table5 = [r for r in csv.DictReader(f) if r.get("capital_musd") and float(r["target"]) == TARGET]
    seen, rows = set(), []
    for r in table5:
        d = (int(r["daf_uf_trains"]), int(r["subsurface_trains"]), int(r["med_units"]), int(r["msf_units"]),
             float(r["storage_days"]), float(r["dc_share"]))
        if d in seen:
            continue
        seen.add(d)
        k1, k2, m, n, b, dc = d
        opt = option(k1, k2, m, n)
        for name, h, v in STRESS:
            x = {k: np.array([val]) for k, val in v.items()}
            x["dc_share"] = np.array([dc])
            peak = evaluate(opt, HAZARDS[h], x).shortfall[0]
            rows.append({"dc_share": dc, "design": f"{k1}/{k2}/{m}/{n}/{b:g}", "family": LABEL[family(m, n)],
                         "case": name, "peak_shortfall_days": peak, "storage_available_days": v["fill"] * b,
                         "unserved_days": max(peak - v["fill"] * b, 0.0)})
    return rows


def write(name: str, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(OUT / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main(parts: list[str]) -> None:
    """Run all studies, or only those named: sensitivity, stress, map, grid."""
    parts = parts or ["sensitivity", "stress", "map", "grid"]
    if "sensitivity" in parts:
        write("assumption_sensitivity.csv", sensitivity())
    if "stress" in parts:
        write("stress_cases.csv", stress_cases())
    if "map" in parts:
        write("survival_map.csv", survival_map())
    if "grid" in parts:
        write("response_grid.csv", response_grid())
    print(f"Wrote assumption studies to {OUT}")


if __name__ == "__main__":
    main(sys.argv[1:])
