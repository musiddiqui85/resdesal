# resdesal: resilience screening of large seawater desalination plants

resdesal is the Python package behind the paper

> M. Usama Siddiqui and Syed M. Zubair, *Resilience of large seawater desalination plants to algal blooms and
> oil spills: a least-cost comparison of storage, strengthened reverse osmosis and thermal diversity*
> (manuscript submitted to *Desalination*).

The repository also holds the data and results of the paper: the compiled outage events, the 2026 Israeli plant
reports, every model output behind the tables and figures, and the scripts that produce them.

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
`scripts/run_studies.py`, and `matplotlib` and `pillow` for the figures. `requirements-lock.txt` pins the exact
versions used for the paper.

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

## Data

`data/` holds the inputs compiled from published sources.

- `events.csv`: documented outages and closures of seawater desalination plants in blooms and oil spills,
  1997–2026, with the reported duration, the evidence level and the source of each (Table 2 and Fig. 2 of the
  paper).
- `israel2026_plants.csv` and `israel2026_output.csv`: shutdown, restart and reported output of the Israeli plants
  in the 2026 bloom, from news reports (Section 2.7 and Section S16 of the paper).

The data sheet of the GCRP-600 reference plant, with a source and status for every value, is Supplementary
Material S2 of the paper. Its values are the ones in `src/resdesal/params.py` and `src/resdesal/architectures.py`.

## Reproducing the results

`results/` holds every model output of the paper. The scripts below rewrite it, and the figure scripts write
`figures/`. Run them from the repository root, in this order, because later steps read the files of earlier
ones. On Windows, set `PYTHONPATH` with `set PYTHONPATH=src` first and leave it out of each line.

```bash
PYTHONPATH=src python scripts/run_studies.py
PYTHONPATH=src python scripts/design_study.py
PYTHONPATH=src python scripts/holdout_check.py
PYTHONPATH=src python scripts/assumption_study.py
PYTHONPATH=src python scripts/cost_sensitivity.py
PYTHONPATH=src python scripts/msf_threshold.py
PYTHONPATH=src python scripts/israel2026_hindcast.py --compute-only
PYTHONPATH=src python scripts/make_figures.py
PYTHONPATH=src python scripts/make_si_figures.py
```

| Script | Writes to `results/` | Used in the paper | Time |
|---|---|---|---|
| `run_studies.py` | `table3_sigma`, `costs_breakeven`, `gas_sensitivity`, `mc_summary`, `mc_summary_dc_range`, `storage_sweep`, `dc_sweep`, `pretreatment_threshold`, `robust_sizing`, `a3_conditional`, `matched_thermal`, `oil_thermal_sensitivity`, `sobol_*` | Table 4, Figs. 4–6, Sections 3.1, 3.2 and 3.6, Sections S17–S19 | 6 s |
| `design_study.py` | `design_enumeration`, `design_best`, `design_fronts`, `refill_selected` | Table 5, Fig. 7, Section 3.3, Table S16 | 1.5 min |
| `holdout_check.py` | `holdout_check` | Section S12 (check on fresh draws) | 2 s |
| `assumption_study.py` | `survival_map`, `response_grid`, `assumption_sensitivity`, `stress_cases` | Fig. 8, Table 6, Sections 3.4 and 3.5, Fig. S2, Tables S7 and S14 | 50 min |
| `cost_sensitivity.py` | `cost_sensitivity`, `cost_switching`, `cost_joint` | Table 6, Section 3.5, Tables S14 and S15 | 30 s |
| `msf_threshold.py` | `msf_threshold` | Fig. 9, Section 3.7 | 3.5 min |
| `israel2026_hindcast.py --compute-only` | `israel2026_hindcast`, `israel2026_fits`, `israel2026_shortfall` | Section 2.7, Section S16 | 2 s |
| `make_figures.py` | `figures/fig1_*` to `figures/fig9_*`, `figures/figS2_*` | Figs. 1–9 and S2 | 30 s |
| `make_si_figures.py` | `figures/figS1_*` | Fig. S1 | 3 s |

Times are from a run on 12 cores. The random seed (20261102) is fixed in `scripts/run_studies.py`. With the
versions in `requirements-lock.txt` (Python 3.14, Windows 11), every file the scripts write to `results/` is
identical, byte for byte, to the one in this repository, so `git status results` after a run shows no change.
On other platforms or package versions, floating-point details may change the last digits of some values.

`assumption_study.py` takes the parts to run as arguments (`sensitivity`, `stress`, `map`, `grid`), and any
script can be run on its own once the files it reads exist.

`results/israel2026_forecast.json` is not an output. It is a forecast of when two Israeli plants would return to
full output, frozen on 3 November 2026 before later plant reports were read. With `--compute-only` the hindcast
never rewrites it. In the hindcast files, "P1" in a label or column name (for example `p1_low`) means the model
of the paper.

The labels of Fig. 2 carry the reference numbers of the paper only when the paper is built with it, so the
figure drawn here has the same events without the numbers.

## Repository layout

```
src/resdesal/        the package
  params.py            reference plant and costs (GCRP-600)
  screening.py         closed-form relations: storage, surviving production needed, break-even event length
  architectures.py     design options A0-A7, dependency paths and the six hazards
  montecarlo.py        sampling of the hazards and the exact peak shortfall of many events at once
  waterbalance.py      time-stepping water-balance simulator used for verification
  economics.py         cost of surviving capacity, energy premiums and lifecycle cost
scripts/             the studies, the design comparison and the figures of the paper
data/                documented outage events and the 2026 Israeli plant reports
results/             every model output of the paper
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

- Code (`src/`, `scripts/`, `examples/`, `tests/`): [MIT licence](LICENSE). You may use, modify and
  redistribute it, including commercially, provided the copyright notice is kept.
- Data and results (`data/`, `results/`): [Creative Commons Attribution 4.0 International (CC BY 4.0)](LICENSE-DATA.md).
  You may share and adapt them for any purpose, provided you give credit and cite the paper.

The reports, papers and news articles compiled in `data/` remain under their own terms.

Copyright © 2026 M. Usama Siddiqui and Syed M. Zubair.

## Contact

Syed M. Zubair (corresponding author), King Fahd University of Petroleum & Minerals, smzubair@kfupm.edu.sa.
M. Usama Siddiqui, DHA Suffa University.
