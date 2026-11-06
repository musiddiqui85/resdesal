"""Equal-service comparison of complete designs (Section 3.3 of the paper).

    PYTHONPATH=src python scripts/design_study.py

Enumerates designs for the greenfield-replacement context: DAF + UF on k of 16 RO trains,
a subsurface intake on 0–2 trains' worth of capacity (cap 12.5%, E14), 0–20 MED units and 0–6 MSF
units, all thermal units on CCGT steam or all with their own boiler, and 2.5–12 days of
critical demand in storage. Every
design keeps a nominal capacity of 600,000 m³/d and must deliver normal demand at
long-run availability. For each design it records scenario coverage (the share of draws
in which critical demand is met) in each hazard and the full extra capital and lifecycle
cost over the baseline plant, storage included. Writes results/design_enumeration.csv
and results/design_best.csv.

The second-site option (A3) is left out: it is a binary state whose coverage equals its
assumed escape probability (results/a3_conditional.csv), so it cannot reach a high
coverage target by sizing.
"""

from __future__ import annotations

import csv
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_studies import DC_BASE, N_DRAWS, OUT, P, SEED  # noqa: E402

from resdesal import economics as ec  # noqa: E402
from resdesal.architectures import A_MED, A_MSF, A_RO, HAZARDS, MAX_SHARE, build, combine  # noqa: E402
from resdesal.montecarlo import evaluate, sample_inputs  # noqa: E402
from resdesal.params import COSTS  # noqa: E402

TRAIN = P.train_capacity / P.capacity  # 1/16
MED_UNIT = P.med_unit_capacity / P.capacity
MSF_UNIT = P.msf_unit_capacity / P.capacity
MSF_MAX = 6  # 0-6 units of 92,600 m³/d
MED_MAX = 20  # up to the plant's capacity (Marafiq has 27 such units)
SUB_TRAINS = round(MAX_SHARE["A2"] / TRAIN)  # subsurface trains allowed by the cap
STORAGE_DAYS = (2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 10.0, 10.5, 11.0, 12.0)
TARGET_HAZARDS = ("severe_bloom", "short_bloom", "oil")
OTHER_HAZARDS = ("oil_fouled", "grid_blackout")
TARGETS = (0.95, 0.99)
FRONT_TARGETS = (0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 0.999)
# Design families by thermal technology (the report's cases B-E; A is the baseline plant).
FAMILY = {"ro": "Strengthened RO and storage (no thermal)", "med": "With MED", "msf": "With MSF",
          "med_msf": "With MED and MSF"}


def family(m: int, n: int) -> str:
    """Family key of a design with m MED and n MSF units."""
    return {(False, False): "ro", (True, False): "med", (False, True): "msf", (True, True): "med_msf"}[m > 0, n > 0]


def _fam(key: str, steam: str | None = None):
    return lambda r: family(r["med_units"], r["msf_units"]) == key and (steam is None or r["steam"] == steam)


FAMILIES = (("Strengthened RO and storage", _fam("ro")),
            ("With MED on CCGT steam", _fam("med", "CCGT")), ("With MED and own boiler", _fam("med", "boiler")),
            ("With MSF on CCGT steam", _fam("msf", "CCGT")), ("With MSF and own boiler", _fam("msf", "boiler")),
            ("With MED and MSF on CCGT steam", _fam("med_msf", "CCGT")),
            ("With MED and MSF and own boiler", _fam("med_msf", "boiler")))


def designs():
    """(DAF+UF trains, subsurface trains, MED units, MSF units) with total share at most 1."""
    for k1, k2, m, n in itertools.product(range(17), range(SUB_TRAINS + 1), range(MED_MAX + 1), range(MSF_MAX + 1)):
        share = (k1 + k2) * TRAIN + m * MED_UNIT + n * MSF_UNIT
        if share <= 1.0 + 1e-9:
            yield k1, k2, m, n


def option_for(k1: int, k2: int, m: int, med_variant: str = "A5", max_share: dict[str, float] | None = None,
               n: int = 0):
    """The design as an Option. MSF units take the same steam supply as the MED units (A6 with A4, A7 with A5)."""
    msf_variant = {"A4": "A6", "A5": "A7"}[med_variant]
    parts = [(key, s) for key, s in (("A1", k1 * TRAIN), ("A2", k2 * TRAIN), (med_variant, m * MED_UNIT),
                                     (msf_variant, n * MSF_UNIT)) if s > 0]
    if not parts:
        return build("A0")
    if len(parts) == 1:
        return build(*parts[0], max_share=max_share)
    return combine(*parts, max_share=max_share)


def costs(k1: int, k2: int, m: int, variant: str, storage_days: float, d_c: float, n: int = 0) -> tuple[float, float]:
    """Extra capital and lifecycle cost over the baseline plant, USD. ``variant`` "A5" puts every
    thermal unit (MED and MSF) on its own boiler; anything else on CCGT steam."""
    c_ro, c_med, af = COSTS.swro_capex, COSTS.med_capex, ec.annuity_factor()
    k_daf, k_sub, k_med = k1 * P.train_capacity, k2 * P.train_capacity, m * P.med_unit_capacity
    boiler = variant == "A5"
    capex_daf = 0.08 * c_ro * k_daf
    capex_sub = 0.13 * c_ro * k_sub
    med_unit_capex = c_med - c_ro + (ec.boiler_capex() if boiler else 0.0)
    capex_med = med_unit_capex * k_med
    extra_storage = max(storage_days * d_c - P.storage, 0.0)
    capex_sto = COSTS.storage_capex * extra_storage
    capital = capex_daf + capex_sub + capex_med + capex_sto
    source = "boiler" if boiler else "ccgt"
    energy = ec.med_energy_premium(source)
    annual = (ec.FIXED_OM * (capex_daf + capex_sub + capex_med) + ec.STORAGE_OM * capex_sto
              + ec.DAF_UF_OPEX * 365 * A_RO * k_daf + energy * 365 * A_MED * k_med)
    if n:  # MSF terms added separately so designs without MSF cost exactly what they did before
        k_msf = n * P.msf_unit_capacity
        capex_msf = (COSTS.msf_capex - c_ro + (ec.boiler_capex("msf") if boiler else 0.0)) * k_msf
        capital += capex_msf
        annual += ec.FIXED_OM * capex_msf + ec.med_energy_premium(source, tech="msf") * 365 * A_MSF * k_msf
    return capital, capital + af * annual


def enumerate_designs(dc_share: float, n: int = N_DRAWS) -> list[dict]:
    d_c = dc_share * P.capacity
    draws = {h: sample_inputs(HAZARDS[h], n, SEED, dc_share=(dc_share, dc_share))
             for h in TARGET_HAZARDS + OTHER_HAZARDS}
    rows = []
    for k1, k2, m, n in designs():
        option = option_for(k1, k2, m, n=n)
        normal = sum(a.capacity * a.availability for a in option.assets)
        # Peak cumulative shortfall in days of D_c, and fill fraction, per hazard; storage is compared after.
        peaks = {}
        for h, x in draws.items():
            d = evaluate(option, HAZARDS[h], x)
            peaks[h] = (d.shortfall, x["fill"])
        for b in sorted(set(STORAGE_DAYS) | {P.storage / d_c}):  # include the existing tanks
            if b < P.storage / d_c:
                continue  # the existing tanks are kept
            s_max_days = b
            cov = {h: float(np.mean(pk <= fill * s_max_days)) for h, (pk, fill) in peaks.items()}
            for variant in (("A4", "A5") if m or n else ("-",)):
                capital, lifecycle = costs(k1, k2, m, variant, b, d_c, n)
                rows.append({
                    "dc_share": dc_share, "daf_uf_trains": k1, "subsurface_trains": k2, "med_units": m,
                    "msf_units": n, "steam": {"A4": "CCGT", "A5": "boiler", "-": "-"}[variant], "storage_days": b,
                    "normal_output_m3d": normal, "meets_normal_demand": normal >= P.normal_demand,
                    **{f"cov_{h}": cov[h] for h in cov},
                    "cov_min_target": min(cov[h] for h in TARGET_HAZARDS),
                    "capital_musd": capital / 1e6, "lifecycle_musd": lifecycle / 1e6,
                })
    return rows


def best(rows: list[dict]) -> list[dict]:
    """Cheapest feasible design per target, family and cost basis."""
    out = []
    for dc in sorted({r["dc_share"] for r in rows}):
        for target in TARGETS:
            for key, label in FAMILY.items():
                ok = [r for r in rows if r["dc_share"] == dc and r["meets_normal_demand"]
                      and family(r["med_units"], r["msf_units"]) == key and r["cov_min_target"] >= target]
                for basis in ("capital_musd", "lifecycle_musd"):
                    if not ok:
                        out.append({"dc_share": dc, "target": target, "family": label, "basis": basis})
                        continue
                    r = min(ok, key=lambda r: (r[basis], r["capital_musd"]))
                    out.append({"dc_share": dc, "target": target, "family": label, "basis": basis, **r})
    return out


def fronts(rows: list[dict]) -> list[dict]:
    """Minimum capital and lifecycle cost per family against the coverage target."""
    out = []
    for dc in sorted({r["dc_share"] for r in rows}):
        for target in FRONT_TARGETS:
            for family, pred in FAMILIES:
                ok = [r for r in rows if r["dc_share"] == dc and r["meets_normal_demand"] and pred(r)
                      and r["cov_min_target"] >= target]
                row = {"dc_share": dc, "target": target, "family": family}
                if ok:
                    row["min_capital_musd"] = min(r["capital_musd"] for r in ok)
                    row["min_lifecycle_musd"] = min(r["lifecycle_musd"] for r in ok)
                out.append(row)
    return out


def refill(rows: list[dict]) -> list[dict]:
    """Days to refill the storage of each selected 99% design at full production after an event, from empty and
    from half full (Section S8 applied to the selected tanks and their production headroom)."""
    from resdesal.screening import refill_time

    out, seen = [], set()
    for r in rows:
        if r.get("capital_musd") is None or r["target"] != 0.99:
            continue
        key = (r["dc_share"], r["daf_uf_trains"], r["subsurface_trains"], r["med_units"], r["msf_units"],
               r["storage_days"])
        if key in seen:
            continue
        seen.add(key)
        volume = r["storage_days"] * r["dc_share"] * P.capacity
        empty = refill_time(volume, r["normal_output_m3d"], P.normal_demand)
        out.append({"dc_share": r["dc_share"], "family": r["family"], "basis": r["basis"],
                    "design": f"{r['daf_uf_trains']}/{r['subsurface_trains']}/{r['med_units']}/{r['msf_units']}",
                    "storage_days": r["storage_days"], "storage_m3": volume,
                    "headroom_m3d": r["normal_output_m3d"] - P.normal_demand,
                    "refill_empty_d": empty, "refill_half_d": empty / 2})
    return out


def write(name: str, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(OUT / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    rows = enumerate_designs(DC_BASE[0]) + enumerate_designs(0.5)
    write("design_enumeration.csv", rows)
    best_rows = best(rows)
    write("design_best.csv", best_rows)
    write("refill_selected.csv", refill(best_rows))
    write("design_fronts.csv", fronts(rows))
    print(f"Wrote {len(rows)} designs to {OUT}")


if __name__ == "__main__":
    main()
