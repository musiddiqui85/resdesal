"""Monte Carlo coverage of the fixed-size design options A0-A7 (Section 3.2 and Fig. 6a of the paper).

    python examples/03_option_coverage.py [draws]

Coverage is the share of sampled events in which critical demand is met through the event and recovery. Each
hazard's inputs (outage length, recovery, tank level, node survival factors, restart) are drawn once and used
for every option (common random numbers). The default 50,000 draws with the paper's seed give the values of the
paper, and fewer draws give a quick approximation.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_studies import DC_BASE, N_DRAWS, OPTIONS, SEED  # noqa: E402  (the paper's option sizes and settings)

from resdesal.architectures import HAZARDS  # noqa: E402
from resdesal.montecarlo import evaluate, sample_inputs  # noqa: E402

n = int(sys.argv[1]) if len(sys.argv) > 1 else N_DRAWS
hazards = ("severe_bloom", "short_bloom", "oil", "oil_fouled", "grid_blackout")
coverage = {}
for h in hazards:
    x = sample_inputs(HAZARDS[h], n, SEED, dc_share=DC_BASE)  # critical demand fixed at 40% of capacity
    for key, option in OPTIONS.items():
        coverage[key, h] = evaluate(option, HAZARDS[h], x).met.mean()

print(f"Share of {n:,} draws in which critical demand is met\n")
print(f"{'Option':<8}" + "".join(f"{h:>15}" for h in hazards))
for key in OPTIONS:
    print(f"{key:<8}" + "".join(f"{coverage[key, h]:>15.1%}" for h in hazards))
