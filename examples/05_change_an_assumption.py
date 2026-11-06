"""Change an assumption and see what it does to a design (Section 3.4 of the paper).

    python examples/05_change_an_assumption.py

The output that DAF + UF keeps in a short, picoplankton-rich bloom is the least documented input of the study
(assumed 10-80%, base 40%). Here it is fixed at several values instead of sampled, and the short-bloom
coverage of the two cheapest designs of Table 5 is recomputed. Hazards are frozen dataclasses, so a changed
assumption is a copy made with dataclasses.replace. The same pattern changes any range in
resdesal.architectures.HAZARDS or any plant value in resdesal.params.
"""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from design_study import option_for  # noqa: E402
from run_studies import N_DRAWS, SEED  # noqa: E402

from resdesal.architectures import HAZARDS, fixed  # noqa: E402
from resdesal.montecarlo import evaluate, sample_inputs  # noqa: E402

designs = {
    "Strengthened RO, 8 d storage": (option_for(14, 2, 0), 8.0),
    "With MED, 4.5 d storage": (option_for(4, 2, 4, "A4"), 4.5),
}
base = HAZARDS["short_bloom"]
print("Short-bloom coverage at 40% critical demand, with the output kept by DAF + UF fixed\n")
print(f"{'DAF + UF keeps':>15}" + "".join(f"{name:>32}" for name in designs))
for f in (0.1, 0.2, 0.3, 0.4, 0.6, 0.8):
    hazard = replace(base, factors={**base.factors, "daf_uf": fixed(f)})
    x = sample_inputs(hazard, N_DRAWS, SEED, dc_share=(0.4, 0.4))
    cells = []
    for option, storage_days in designs.values():
        d = evaluate(option, hazard, x)
        cells.append(f"{np.mean(d.shortfall <= x['fill'] * storage_days):.1%}")
    print(f"{f:>15.0%}" + "".join(f"{c:>32}" for c in cells))

# Expected: both designs reach full short-bloom coverage once DAF + UF keeps about 30% or more. At 10% the RO design
# falls to about 84% and the MED design to about 93%. A 99% target is set by the worst draws, which is why the low
# end of the assumed range decides which design is cheapest (Fig. 8 of the paper).
