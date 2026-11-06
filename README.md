# resdesal: resilience screening of large seawater desalination plants

resdesal is the Python package behind the paper

> M. Usama Siddiqui and Syed M. Zubair, *Resilience of large seawater desalination plants to algal blooms and
> oil spills: a least-cost comparison of storage, strengthened reverse osmosis and thermal diversity*
> (manuscript submitted to *Desalination*).

It asks a practical question for a large seawater reverse osmosis (SWRO) plant: which combination of product
storage, strengthened RO (dissolved air flotation with ultrafiltration, a subsurface intake, a second site) and
thermal units (MED, MSF) keeps critical demand supplied through an algal bloom, an oil spill or a power failure at
least cost. It does this in four steps:

1. **Dependency paths.** Every production asset depends on a chain of nodes (intake, pretreatment, power, steam).
   A hazard gives each node a survival factor, and an asset delivers its normal output times the product of the
   factors on its path.
2. **Water balance.** A closed-form storage–yield balance of the plant's product storage over one event, with
   start-up delay, a recovery ramp and a derated restart, gives the peak cumulative shortfall and the unserved
   water. A time-stepping simulator checks it.
3. **Coverage.** Monte Carlo sampling of each hazard gives the share of events in which critical demand is met.
4. **Least cost at equal service.** Every complete design that delivers the same normal demand is priced
   (capital and lifecycle), and the cheapest that reaches a coverage target is selected.

The default plant is the synthetic GCRP-600 reference plant (600,000 m³/d, Saudi Gulf coast) described in the
paper. Its values are in `src/resdesal/params.py` and `src/resdesal/architectures.py`, and any of them can be
changed.

## Install

Python 3.10 or later.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-lock.txt
pip install -e . --no-deps
python -m pytest                   # runs the tests
```

Only `numpy` is needed for the package itself. `SALib` is needed for the sensitivity indices in
`scripts/run_studies.py`. `requirements-lock.txt` pins the exact versions used for the paper.

## Worked examples

| Example | What it shows | Run time |
|---|---|---|
| [`01_storage_and_surviving_production.py`](examples/01_storage_and_surviving_production.py) | One day of storage bridges 2.5 days of critical demand, and the production that must survive outages of 3–120 days | instant |
| [`02_one_event.py`](examples/02_one_event.py) | Supply, cumulative shortfall and unserved water over one short bloom | instant |
| [`03_option_coverage.py`](examples/03_option_coverage.py) | Monte Carlo coverage of the fixed-size options A0–A7 in five hazards | about 10 s |
| [`04_custom_design.py`](examples/04_custom_design.py) | Coverage and cost of a complete design you choose on the command line | about 2 s |
| [`05_change_an_assumption.py`](examples/05_change_an_assumption.py) | How the output kept by DAF + UF in a picoplankton bloom changes coverage | about 5 s |

```bash
python examples/01_storage_and_surviving_production.py
python examples/04_custom_design.py --daf 14 --sub 2 --med 0 --storage 8
```

## Using the package

```python
from resdesal import GCRP600, bridge_days, required_sigma
from resdesal.architectures import HAZARDS, build
from resdesal.montecarlo import evaluate, sample_inputs

P = GCRP600
print(bridge_days(P.storage, P.critical_demand))            # 2.5 days of critical demand
print(required_sigma(55, P.storage, P.critical_demand))     # 0.95 of critical demand must survive a 55-day outage

option = build("A4", 0.45)                                  # 45% of capacity as MED units on turbine steam
x = sample_inputs(HAZARDS["short_bloom"], 20_000, seed=1, dc_share=(0.4, 0.4))
print(evaluate(option, HAZARDS["short_bloom"], x).met.mean())   # share of sampled short blooms covered
```

[docs/model.md](docs/model.md) explains each part of the model and names the function that implements it.

## Changing the inputs

- **Plant and costs:** `resdesal/params.py` (`ReferencePlant`, `Costs`) and the constants at the top of
  `resdesal/economics.py`. Both are frozen dataclasses or module constants, so a case is a copy made with
  `dataclasses.replace`.
- **Hazards and survival factors:** `resdesal/architectures.py`, `HAZARDS`. Each range is a `Spread(low, base,
  high)`. Example 05 shows how to fix one.
- **Designs:** `scripts/design_study.py`, `option_for(daf_trains, subsurface_trains, med_units, steam, n=msf_units)`
  and `costs(...)`. Example 04 uses them.

`scripts/run_studies.py` and `scripts/design_study.py` also run the option studies and the full design
comparison of the paper when called directly. They write their tables to `results/`.

## Repository layout

```
src/resdesal/        the package
  params.py            reference plant and costs (GCRP-600)
  screening.py         closed-form relations: storage, surviving production needed, break-even event length
  architectures.py     design options A0-A7, dependency paths and the six hazards
  montecarlo.py        sampling of the hazards and the exact peak shortfall of many events at once
  waterbalance.py      time-stepping water-balance simulator used for verification
  economics.py         cost of surviving capacity, energy premiums and lifecycle cost
scripts/             the option and design definitions used by the examples
examples/            five worked examples
tests/               unit tests (pytest)
docs/                the model and where the code implements it
```

## Citation

If you use this code, please cite the paper:

> Siddiqui, M. U., & Zubair, S. M. (2026). Resilience of large seawater desalination plants to algal blooms and
> oil spills: a least-cost comparison of storage, strengthened reverse osmosis and thermal diversity. Manuscript
> submitted to *Desalination*.

```bibtex
@unpublished{siddiqui2026resilience,
  author = {Siddiqui, M. Usama and Zubair, Syed M.},
  title  = {Resilience of large seawater desalination plants to algal blooms and oil spills: a least-cost
            comparison of storage, strengthened reverse osmosis and thermal diversity},
  note   = {Manuscript submitted to Desalination},
  year   = {2026}
}
```

The same reference is in [CITATION.cff](CITATION.cff), which GitHub shows as "Cite this repository".

## Licence

[MIT licence](LICENSE). You may use, modify and redistribute the code, including commercially, provided the
copyright notice is kept.

Copyright © 2026 M. Usama Siddiqui and Syed M. Zubair.

## Contact

Syed M. Zubair (corresponding author), King Fahd University of Petroleum & Minerals, smzubair@kfupm.edu.sa.
M. Usama Siddiqui, DHA Suffa University.
