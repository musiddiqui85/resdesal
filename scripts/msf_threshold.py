"""MSF entry threshold (Fig. 9 and Section 3.7 of the paper).

    PYTHONPATH=src python scripts/msf_threshold.py

For each output share kept by MED in a bloom (fixed at 0, 0.1, ..., 1), finds by bisection the
least share MSF must keep, also fixed, for the cheapest design with MSF (MSF alone or with MED) to
cost no more than the cheapest design without it, at the 99% coverage target. The bloom is the
short (picoplankton-rich) or the severe one; DAF + UF and every other input are sampled as in the
Monte Carlo, and the other target hazards keep their sampled draws. Coverage rises with the MSF
share, so the threshold is well defined. Writes results/msf_threshold.csv.
"""

from __future__ import annotations

import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from assumption_study import (BASES, MAP_N, TARGET_HAZARDS, cheapest, design_code, design_list, draws_for,  # noqa: E402
                              need_days, needs, option)
from run_studies import OUT, P, SEED  # noqa: E402

from resdesal.architectures import HAZARDS  # noqa: E402
from resdesal.montecarlo import evaluate, sample_inputs  # noqa: E402

MED_LEVELS = np.round(np.arange(0.0, 1.0001, 0.1), 2)
DC = (0.4, 0.5)
STEPS = 7  # bisection steps: the threshold to within 0.01


def _point(args):
    dc, h, f_med, fixed_need = args
    x = sample_inputs(HAZARDS[h], MAP_N, SEED, dc_share=(dc, dc))
    x["med_process"] = np.full(MAP_N, f_med)
    no_msf = {}  # designs without MSF do not depend on the MSF share

    def best_at(f_msf):
        x["msf_process"] = np.full(MAP_N, f_msf)
        need = {}
        for d, other in fixed_need.items():
            if d[3] == 0 and d in no_msf:
                need[d] = no_msf[d]
                continue
            need[d] = max(other, need_days(evaluate(option(*d), HAZARDS[h], x).shortfall, x["fill"]))
            if d[3] == 0:
                no_msf[d] = need[d]
        return cheapest(need, dc * P.capacity)

    def gap(best, basis):
        with_msf = [best[f, basis][basis] for f in ("msf", "med_msf") if (f, basis) in best]
        without = [best[f, basis][basis] for f in ("ro", "med") if (f, basis) in best]
        if not with_msf:
            return np.inf
        return min(with_msf) - (min(without) if without else np.inf)

    rows = []
    top = best_at(1.0)
    for basis in BASES:
        row = {"dc_share": dc, "hazard": h, "f_med": f_med, "basis": basis}
        if gap(top, basis) > 0:
            row["note"] = "designs without MSF are cheaper even if MSF keeps all its output"
        elif gap(best_at(0.0), basis) <= 0:
            row.update(f_msf_threshold=0.0, note="designs with MSF are cheaper at any MSF share")
        else:
            lo, hi = 0.0, 1.0
            for _ in range(STEPS):
                mid = 0.5 * (lo + hi)
                lo, hi = (lo, mid) if gap(best_at(mid), basis) <= 0 else (mid, hi)
            b = best_at(hi)
            win = min((b[f, basis][basis], f) for f in ("msf", "med_msf") if (f, basis) in b)[1]
            row.update(f_msf_threshold=hi, note="least MSF share at which a design with MSF is cheapest",
                       design=design_code(b[win, basis]), steam=b[win, basis]["steam"], cost_musd=b[win, basis][basis])
        rows.append(row)
    return rows


def main() -> None:
    designs = list(design_list())
    jobs = []
    for dc in DC:
        draws = draws_for(dc, MAP_N)
        for h in ("short_bloom", "severe_bloom"):
            fixed_need = needs(dc, draws, designs, hazards=tuple(k for k in TARGET_HAZARDS if k != h))
            jobs += [(dc, h, f, fixed_need) for f in MED_LEVELS]
    with ProcessPoolExecutor() as pool:
        rows = [r for rs in pool.map(_point, jobs) for r in rs]
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(OUT / "msf_threshold.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUT / 'msf_threshold.csv'}")


if __name__ == "__main__":
    main()
