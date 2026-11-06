# The model and where the code implements it

Equation numbers are those of the paper. Time is in days, flows in m³/d and volumes in m³. Inside the model,
volumes are divided by the critical demand *D*<sub>c</sub> and so become days of critical demand.

## 1. Dependency paths and surviving production

Every production asset *i* (a group of RO trains, MED units or MSF units) has a rated capacity *K*<sub>i</sub>, a
normal availability *a*<sub>i</sub> and a path *P*<sub>i</sub> of shared nodes: open or subsurface intake, media
filtration or DAF + UF, grid, CCGT steam or boiler, MED or MSF process. A hazard *h* gives each node a survival
factor *f*<sub>n,h</sub> between 0 (failed) and 1 (unaffected), and the surviving production relative to
critical demand is

σ = (1/*D*<sub>c</sub>) Σ<sub>i</sub> *a*<sub>i</sub> *K*<sub>i</sub> Π<sub>n∈P<sub>i</sub></sub> *f*<sub>n,h</sub>   (1)

The product of the factors is a compliant-production ratio relative to normally available output: output that is
both produced and acceptable as drinking water.

| Item | Code |
|---|---|
| Assets, paths and options A0–A7 | `architectures.build`, `architectures.combine`, `architectures.Option` |
| Output of one asset in a hazard | `architectures.asset_output` |
| σ of an option at base, low or high factors | `architectures.option_sigma` |
| The six hazards and their ranges | `architectures.HAZARDS` (`Hazard`, `Spread`) |
| MED and MSF drawn from one shared variate | `architectures.LINKED`, `montecarlo.sample_inputs(linked=True)` |

## 2. Water balance of one event

With storage *S*<sub>0</sub> at onset, outage length *T* and storage capacity *S*<sub>max</sub>:

β = *S*<sub>0</sub>/(*D*<sub>c</sub>*T*), σ = *Q*<sub>s</sub>/*D*<sub>c</sub>, φ = *S*<sub>0</sub>/*S*<sub>max</sub>   (2)

Without delay or recovery, critical demand is met if β + σ ≥ 1 (3), so with *b* = *S*<sub>0</sub>/*D*<sub>c</sub>
the surviving production needed is σ<sub>req</sub> = max(0, 1 − *b*/*T*) (4).

In the full model, asset *j* supplies σ<sub>j</sub> from its start time τ<sub>j</sub> (5). After the event the
lost output ρ<sub>r</sub>*D*<sub>c</sub> ramps linearly over τ<sub>rec</sub> days to a fraction γ, stays there until
*L* days after the event and then returns in full (6). The cumulative shortfall is
*N*(*t*) = ∫<sub>0</sub><sup>t</sup> [1 − *s*(*t*′)] d*t*′ (7), and the unserved critical water is
*U* = *D*<sub>c</sub> max(0, max *N* − *b*) (8). Because supply is piecewise linear, the peak of *N* is found
exactly from the shortfall at the end of the event (9) and the area added during recovery, a triangle (10) or a
ramp with a derated plateau (11).

| Item | Code |
|---|---|
| β, σ, bridge days | `screening.storage_ratio`, `screening.survival_ratio`, `screening.bridge_days` |
| σ<sub>req</sub>, with delay and recovery in closed form | `screening.required_sigma` |
| Single-asset condition (12) and its shortfall | `screening.critical_deficit`, `screening.meets_critical_demand` |
| Peak of *N* for many assets and many events at once, equations (5)–(11) | `montecarlo.max_shortfall` |
| Refill time (13) | `screening.refill_time` |
| Time-stepping simulator, used to verify the closed form | `waterbalance.simulate`, `waterbalance.Outage` |

`max_shortfall` also has a `shape="rising"` restart, in which output climbs linearly from the derated level to
full output. It is a sensitivity case.

## 3. Cost of surviving capacity

A capital premium Δ*c* per m³/d installed, on capacity with availability *a* and survival factor *f*, buys
surviving capacity at *c*<sub>b</sub> = Δ*c*/(*a f*). Against storage at *c*<sub>s</sub> per m³, surviving
capacity is cheaper for events longer than *T*\* = τ + *c*<sub>b</sub>/*c*<sub>s</sub> (14). The lifecycle
version (15) adds fixed O&M and the energy premium of the thermal units over 25 years at 7%.

| Item | Code |
|---|---|
| *T*\* | `screening.breakeven_duration`, `screening.cover_cost` |
| Cost per surviving capacity, capital and lifecycle | `economics.Measure`, `economics.measures` |
| Annuity factor, steam and fuel premiums, boiler capital | `economics.annuity_factor`, `economics.med_energy_premium`, `economics.boiler_capex` |
| Extra capital and lifecycle cost of a complete design | `scripts/design_study.py`, `costs` |

## 4. Coverage and the equal-service comparison

For each hazard, *M* = 50,000 draws sample the outage length, recovery time, tank level at onset (50–100%), node
survival factors and restart. The same draws are used for every design. Coverage is the share of draws in which
the peak shortfall does not exceed the storage at onset (16). A design meets a target if its coverage reaches it
in each target hazard: severe blooms, short blooms and precautionary oil closures.

| Item | Code |
|---|---|
| Draws, with the paper's seed | `montecarlo.sample_inputs` (seed 20261102 in `scripts/run_studies.py`) |
| Peak shortfall, coverage and unserved water of each draw | `montecarlo.evaluate` returning `montecarlo.Draws` (`met`, `unserved`) |
| Every design (0–16 DAF + UF trains, 0–2 subsurface trains, 0–20 MED, 0–6 MSF units, 2.5–12 days of storage) | `scripts/design_study.py`, `designs`, `option_for`, `enumerate_designs` |
| Cheapest design per family and target | `scripts/design_study.py`, `best` |
| Total-order Sobol indices (17) | `scripts/run_studies.py`, `sobol` (SALib) |
| Smallest DAF + UF output for which strengthened RO alone suffices, *f*\* (18) | `scripts/run_studies.py`, `pretreatment_threshold` |
