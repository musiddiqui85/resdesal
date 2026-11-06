"""Coverage and cost of a complete design that you choose (Section 3.3 of the paper).

    python examples/04_custom_design.py                       # the cheapest design with MED at 99%, Table 5
    python examples/04_custom_design.py --daf 14 --sub 2 --storage 8      # the cheapest without thermal units
    python examples/04_custom_design.py --daf 4 --sub 2 --med 4 --storage 4.5 --steam boiler --dc 0.5

A design replaces part of the 16 baseline RO trains with protected or diverse capacity at the same nominal
capacity: DAF + UF ahead of some trains, a subsurface intake for up to 2 trains (the assumed 12.5% cap), MED
units of 29,630 m³/d and MSF units of 92,600 m³/d, and storage in days of critical demand. It is scored
exactly as in scripts/design_study.py: coverage in each target hazard from 50,000 draws with the paper's seed,
and the extra capital and lifecycle cost over the baseline plant (25 years at 7%), storage included.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from design_study import TARGET_HAZARDS, costs, option_for  # noqa: E402
from run_studies import N_DRAWS, SEED  # noqa: E402

from resdesal import GCRP600  # noqa: E402
from resdesal.architectures import HAZARDS  # noqa: E402
from resdesal.montecarlo import evaluate, sample_inputs  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--daf", type=int, default=4, help="RO trains with DAF + UF (0-16)")
ap.add_argument("--sub", type=int, default=2, help="trains' worth of capacity on a subsurface intake (0-2)")
ap.add_argument("--med", type=int, default=4, help="MED units")
ap.add_argument("--msf", type=int, default=0, help="MSF units")
ap.add_argument("--storage", type=float, default=4.5, help="storage in days of critical demand (at least 2.5)")
ap.add_argument("--steam", choices=("ccgt", "boiler"), default="ccgt", help="steam for the thermal units")
ap.add_argument("--dc", type=float, default=0.4, help="critical demand as a share of capacity")
ap.add_argument("--draws", type=int, default=N_DRAWS)
args = ap.parse_args()

P = GCRP600
d_c = args.dc * P.capacity
variant = "A5" if args.steam == "boiler" else "A4"
option = option_for(args.daf, args.sub, args.med, variant, n=args.msf)
normal = sum(a.capacity * a.availability for a in option.assets)

print(f"Design: DAF + UF on {args.daf} of 16 trains, subsurface intake on {args.sub}, {args.med} MED and "
      f"{args.msf} MSF units ({args.steam} steam), {args.storage:g} days of storage, critical demand "
      f"{args.dc:.0%} of capacity")
print(f"Normal output at long-run availability: {normal:,.0f} m³/d "
      f"({'meets' if normal >= P.normal_demand else 'does NOT meet'} normal demand of {P.normal_demand:,.0f})\n")
for h in TARGET_HAZARDS:
    x = sample_inputs(HAZARDS[h], args.draws, SEED, dc_share=(args.dc, args.dc))
    d = evaluate(option, HAZARDS[h], x)
    met = d.shortfall <= x["fill"] * args.storage  # tanks 50-100% full at onset
    print(f"  {HAZARDS[h].label:<40} coverage {np.mean(met):.2%}")
capital, lifecycle = costs(args.daf, args.sub, args.med, variant, args.storage, d_c, args.msf)
print(f"\nExtra cost over the baseline plant: USD {capital / 1e6:.0f} million of capital, "
      f"USD {lifecycle / 1e6:.0f} million over the plant's life")

# Expected for the default design (Table 5): coverage 99.8% / 99.4% / 99.9%, USD 146 million of capital and
# USD 440 million over the plant's life.
