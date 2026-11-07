"""Independent-seed check of the selected 99% designs.

    PYTHONPATH=src python scripts/holdout_check.py

The designs of results/design_best.csv were selected on one set of 50,000 draws per hazard, so
their coverage is at least 99% in that sample by construction. This re-evaluates each one, with
its selected storage, on 200,000 fresh draws per hazard from another seed and reports the
coverage with its binomial standard error. Writes results/holdout_check.csv.
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from design_study import TARGET_HAZARDS, option_for  # noqa: E402
from run_studies import OUT, P, SEED  # noqa: E402

from resdesal.architectures import HAZARDS  # noqa: E402
from resdesal.montecarlo import evaluate, sample_inputs  # noqa: E402

N = 200_000
HOLDOUT_SEED = SEED + 1


def main() -> None:
    with open(OUT / "design_best.csv", encoding="utf-8") as f:
        best = [r for r in csv.DictReader(f) if r.get("capital_musd") and float(r["target"]) == 0.99]
    seen, rows = set(), []
    draws = {}
    for r in best:
        k1, k2, m, n = (int(r[k]) for k in ("daf_uf_trains", "subsurface_trains", "med_units", "msf_units"))
        dc, b = float(r["dc_share"]), float(r["storage_days"])
        if (dc, k1, k2, m, n, b) in seen:
            continue
        seen.add((dc, k1, k2, m, n, b))
        opt = option_for(k1, k2, m, n=n)
        row = {"dc_share": dc, "family": r["family"], "design": f"{k1}/{k2}/{m}/{n}/{b:g}"}
        for h in TARGET_HAZARDS:
            if (dc, h) not in draws:
                draws[dc, h] = sample_inputs(HAZARDS[h], N, HOLDOUT_SEED, dc_share=(dc, dc))
            x = draws[dc, h]
            met = evaluate(opt, HAZARDS[h], x).shortfall <= b * x["fill"]
            p = float(np.mean(met))
            row[f"cov_{h}"] = p
            row[f"se_{h}"] = math.sqrt(p * (1 - p) / N)
        row["cov_min"] = min(row[f"cov_{h}"] for h in TARGET_HAZARDS)
        rows.append(row)
    with open(OUT / "holdout_check.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lo = min(r["cov_min"] for r in rows)
    print(f"Wrote {len(rows)} designs to {OUT / 'holdout_check.csv'}; lowest holdout coverage {100 * lo:.2f}%")


if __name__ == "__main__":
    main()
