"""Supply and cumulative shortfall over one short bloom (Fig. 3 of the paper).

    python examples/02_one_event.py

A 12-day short bloom in which DAF and UF keep 10% of their output, the low end of their range, with the other
inputs at their base values and critical demand at 40% of capacity. Three cases: the baseline plant with its
existing 2.5 days of storage, and the two cheapest designs to build at the 99% coverage target (Table 5).

Each asset's output is its normal output times the survival factors on its path (equation 1). The peak of
the cumulative shortfall N is found in closed form by montecarlo.max_shortfall (equations 5-11). Critical
demand is met throughout if that peak does not exceed the storage at onset, and the gap is the unserved water.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from design_study import option_for  # noqa: E402  (a design: DAF + UF trains, subsurface trains, MED, MSF units)

from resdesal import GCRP600  # noqa: E402
from resdesal.architectures import HAZARDS, asset_output, build  # noqa: E402
from resdesal.montecarlo import max_shortfall  # noqa: E402

P = GCRP600
hz = HAZARDS["short_bloom"]
T, rec, gamma, L = 12.0, hz.recovery.base, hz.restart_fraction.base, hz.derate.base
factors = {node: spread.base for node, spread in hz.factors.items()}
factors["daf_uf"] = 0.1  # the low end of the assumed 10-80%

cases = [
    ("Baseline plant (A0)", build("A0"), P.storage / P.critical_demand),
    ("Strengthened RO: 14 DAF + UF trains, 2 subsurface trains", option_for(14, 2, 0), 8.0),
    ("With MED: 4 DAF + UF trains, 2 subsurface trains, 4 MED units", option_for(4, 2, 4, "A4"), 4.5),
]
print(f"Short bloom of {T:g} days, recovery over {rec:g} days to {gamma:.0%} of the lost output, derated until "
      f"{L:g} days after the event.\n")
for name, option, storage_days in cases:
    out = np.array([asset_output(a, factors) for a in option.assets])  # m³/d that survives, per asset
    sigma = out / P.critical_demand
    lost = (sum(a.capacity * a.availability for a in option.assets) - out.sum()) / P.critical_demand
    delays = [option.delays.get(a.name, 0.0) for a in option.assets]
    peak = float(max_shortfall(T, sigma, delays, rec, lost, gamma, L))
    unserved = max(0.0, peak - storage_days)
    print(f"{name}")
    print(f"  surviving production sigma = {sigma.sum():.2f}, peak shortfall {peak:.2f} days of D_c, "
          f"storage {storage_days:g} days")
    print("  critical demand met throughout" if unserved == 0 else
          f"  unserved water {unserved:.1f} days of D_c = {unserved * P.critical_demand / 1e6:.2f} million m³")

# Expected (Fig. 3): the baseline plant leaves 10.8 days of critical demand unserved, and both designs keep
# their peak shortfall below their storage.
