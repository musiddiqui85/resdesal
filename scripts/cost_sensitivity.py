"""Cost sensitivity of the complete-design comparison (Table 6 and Table S14 of the paper).

    PYTHONPATH=src python scripts/cost_sensitivity.py

Coverage does not depend on cost, so every feasible design in results/design_enumeration.csv is
repriced and the cheapest design in each family (no thermal, MED, MSF, MED and MSF) is reselected
for each cost assumption, changed one at a time over its data-sheet range. Writes
results/cost_sensitivity.csv and results/cost_switching.csv (the MED capital cost at which designs
with MED and without thermal cost the same, and the MSF capital cost at which the cheapest design
with MSF matches the cheapest without it).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from design_study import family  # noqa: E402
from run_studies import OUT, P  # noqa: E402

from resdesal import economics as ec  # noqa: E402
from resdesal.architectures import A_MED, A_MSF, A_RO  # noqa: E402
from resdesal.params import COSTS  # noqa: E402

TARGET = 0.99
BASE = dict(swro_capex=COSTS.swro_capex, med_capex=COSTS.med_capex, daf_premium=0.08, sub_premium=0.13,
            storage_capex=COSTS.storage_capex, steam_penalty=ec.CCGT_STEAM_PENALTY,
            power_price=ec.ELECTRICITY_PRICE, rate=ec.DISCOUNT_RATE, msf_capex=COSTS.msf_capex,
            msf_steam_penalty=ec.CCGT_STEAM_PENALTY_MSF)
LABEL = {"ro": "No thermal", "med": "With MED", "msf": "With MSF", "med_msf": "With MED and MSF"}
# One at a time over the data-sheet ranges. The MED range starts at the SWRO cost (no premium): the
# printed low value of 900 USD per m³/d (Az-Zour North 1) covers the desalination island only.
CASES = [
    ("MED capital, USD per m³/d", "med_capex", (1_100.0, 2_300.0)),            # E4: SWRO cost to World Bank top
    ("DAF + UF premium, % of SWRO capital", "daf_premium", (0.05, 0.18)),      # E12; 18%: Missimer et al. 2018
    ("Subsurface intake premium, % of SWRO capital", "sub_premium", (0.045, 0.20)),  # E13: Missimer et al. 2018
    ("Storage capital, USD per m³", "storage_capex", (160.0, 200.0)),          # E15
    ("CCGT steam penalty, kWh per m³", "steam_penalty", (7.0, 18.0)),          # X3
    ("Electricity price, USD per kWh", "power_price", (0.032, 0.064)),         # E10
    ("Discount rate", "rate", (0.04, 0.12)),                                    # E17
    ("MSF capital, USD per m³/d", "msf_capex", COSTS.msf_capex_range),         # data-sheet row F7
    ("MSF CCGT steam penalty, kWh per m³", "msf_steam_penalty", (8.0, 19.0)),   # data-sheet row F9
]


def design_costs(r: dict, p: dict) -> tuple[float, float]:
    """Extra capital and lifecycle cost, USD million, as in design_study.costs but with cost inputs p."""
    k1, k2, m, n = int(r["daf_uf_trains"]), int(r["subsurface_trains"]), int(r["med_units"]), int(r["msf_units"])
    boiler = r["steam"] == "boiler"
    d_c = float(r["dc_share"]) * P.capacity
    c_ro = p["swro_capex"]
    k_daf, k_sub, k_med = k1 * P.train_capacity, k2 * P.train_capacity, m * P.med_unit_capacity
    capex_daf = p["daf_premium"] * c_ro * k_daf
    capex_sub = p["sub_premium"] * c_ro * k_sub
    capex_med = (p["med_capex"] - c_ro + (ec.boiler_capex() if boiler else 0.0)) * k_med
    capex_sto = p["storage_capex"] * max(float(r["storage_days"]) * d_c - P.storage, 0.0)
    electric = (ec.MED_ELECTRICITY - ec.RO_ENERGY) * p["power_price"]
    energy = electric + (ec.boiler_fuel_cost() if boiler else p["steam_penalty"] * p["power_price"])
    annual = (ec.FIXED_OM * (capex_daf + capex_sub + capex_med) + ec.STORAGE_OM * capex_sto
              + ec.DAF_UF_OPEX * 365 * A_RO * k_daf + energy * 365 * A_MED * k_med)
    capital = capex_daf + capex_sub + capex_med + capex_sto
    if n:  # as in design_study.costs
        k_msf = n * P.msf_unit_capacity
        capex_msf = (p["msf_capex"] - c_ro + (ec.boiler_capex("msf") if boiler else 0.0)) * k_msf
        e_msf = (ec.MSF_ELECTRICITY - ec.RO_ENERGY) * p["power_price"] + (
            ec.boiler_fuel_cost(tech="msf") if boiler else p["msf_steam_penalty"] * p["power_price"])
        capital += capex_msf
        annual += ec.FIXED_OM * capex_msf + e_msf * 365 * A_MSF * k_msf
    return capital / 1e6, (capital + ec.annuity_factor(p["rate"]) * annual) / 1e6


def feasible() -> list[dict]:
    with open(OUT / "design_enumeration.csv", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f)
                if r["meets_normal_demand"] == "True" and float(r["cov_min_target"]) >= TARGET]


def best(rows: list[dict], p: dict, dc: str) -> dict[tuple[str, str], dict]:
    """Cheapest design per family (no thermal, MED, MSF, MED and MSF) and basis (capital / lifecycle)."""
    out: dict[tuple[str, str], dict] = {}
    for r in rows:
        if r["dc_share"] != dc:
            continue
        cap, life = design_costs(r, p)
        fam = LABEL[family(int(r["med_units"]), int(r["msf_units"]))]
        for basis, v in (("capital", cap), ("lifecycle", life)):
            if (fam, basis) not in out or v < out[fam, basis]["value"]:
                out[fam, basis] = {"value": v, "capital_musd": cap, "lifecycle_musd": life,
                                   "design": f"{r['daf_uf_trains']}/{r['subsurface_trains']}/{r['med_units']}/"
                                             f"{r['msf_units']}/{float(r['storage_days']):g}", "steam": r["steam"]}
    return out


def check_base(rows: list[dict]) -> None:
    worst = max(max(abs(design_costs(r, BASE)[0] - float(r["capital_musd"])),
                    abs(design_costs(r, BASE)[1] - float(r["lifecycle_musd"]))) for r in rows)
    if worst > 1e-6:
        raise AssertionError(f"repricing does not reproduce design_study costs (max difference {worst} USD M)")


def switching_med_capex(rows: list[dict], dc: str, basis: str) -> tuple[float | None, str]:
    """MED capital cost (USD per m³/d) at which the cheapest designs with and without MED cost the same."""
    def gap(c):
        b = best(rows, {**BASE, "med_capex": c}, dc)
        return b["With MED", basis]["value"] - b["No thermal", basis]["value"]
    lo, hi = COSTS.swro_capex, 4_000.0
    if ("No thermal", basis) not in best(rows, BASE, dc):
        return None, "only designs with MED meet the target"
    if gap(lo) >= 0:
        return None, "designs without MED are cheaper even with no MED premium over SWRO"
    if gap(hi) < 0:
        return None, "designs with MED are cheaper up to 4,000 USD per m3/d"
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if gap(mid) < 0 else (lo, mid)
    return 0.5 * (lo + hi), "designs with MED are cheaper below this MED capital cost"


def switching_msf_capex(rows: list[dict], dc: str, basis: str) -> tuple[float | None, str]:
    """MSF capital cost (USD per m³/d) at which the cheapest design with MSF costs the same as the cheapest without."""
    def gap(c):
        b = best(rows, {**BASE, "msf_capex": c}, dc)
        with_msf = min(b[f, basis]["value"] for f in ("With MSF", "With MED and MSF") if (f, basis) in b)
        without = min(b[f, basis]["value"] for f in ("No thermal", "With MED") if (f, basis) in b)
        return with_msf - without
    lo, hi = COSTS.swro_capex, 4_000.0
    if gap(lo) >= 0:
        return None, "designs without MSF are cheaper even with no MSF premium over SWRO"
    if gap(hi) < 0:
        return None, "designs with MSF are cheaper up to 4,000 USD per m3/d"
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if gap(mid) < 0 else (lo, mid)
    return 0.5 * (lo + hi), "designs with MSF are cheaper below this MSF capital cost"


JOINT_MED = (1_100.0, 1_400.0, 1_500.0, 1_700.0, 2_000.0, 2_300.0)
JOINT_PENALTY = (7.0, 10.0, 12.0, 14.0, 16.0, 18.0)


def joint(rows: list[dict]) -> list[dict]:
    """MED capital and CCGT steam penalty changed together: cheapest family and the cost of the
    cheapest design without thermal units and with MED, per basis. A scenario map, not a probability."""
    out = []
    for dc in ("0.4", "0.5"):
        for med in JOINT_MED:
            for pen in JOINT_PENALTY:
                b = best(rows, {**BASE, "med_capex": med, "steam_penalty": pen}, dc)
                for basis in ("capital", "lifecycle"):
                    fams = {f: v["value"] for (f, k), v in b.items() if k == basis}
                    out.append({"dc_share": dc, "med_capex": med, "steam_penalty": pen, "basis": basis,
                                "cheapest_family": min(fams, key=fams.get),
                                "no_thermal_musd": fams.get("No thermal"), "with_med_musd": fams.get("With MED")})
    return out


def write(name: str, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(OUT / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    rows = feasible()
    check_base(rows)
    out = []
    for label, key, values in [("Base costs", "med_capex", (BASE["med_capex"],))] + CASES:
        for v in values:
            p = {**BASE, key: v}
            for dc in ("0.4", "0.5"):
                for (fam, basis), b in best(rows, p, dc).items():
                    out.append({"parameter": label, "value": v, "dc_share": dc, "family": fam, "basis": basis,
                                "design": b["design"], "steam": b["steam"], "capital_musd": b["capital_musd"],
                                "lifecycle_musd": b["lifecycle_musd"]})
    write("cost_sensitivity.csv", out)
    sw = []
    for dc in ("0.4", "0.5"):
        for basis in ("capital", "lifecycle"):
            value, note = switching_med_capex(rows, dc, basis)
            msf_value, msf_note = switching_msf_capex(rows, dc, basis)
            sw.append({"dc_share": dc, "basis": basis, "switching_med_capex": value, "note": note,
                       "switching_msf_capex": msf_value, "msf_note": msf_note})
    write("cost_switching.csv", sw)
    write("cost_joint.csv", joint(rows))
    print(f"Wrote cost_sensitivity.csv ({len(out)} rows) and cost_switching.csv: "
          + "; ".join(f"{s['dc_share']} {s['basis']}: MED {s['switching_med_capex']}, MSF {s['switching_msf_capex']}"
                      for s in sw))


if __name__ == "__main__":
    main()
