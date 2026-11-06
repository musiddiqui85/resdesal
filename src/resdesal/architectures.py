"""Architectures A0-A7 for GCRP-600 and the hazards they face.

Each architecture is a set of assets whose ``path`` names the shared nodes it
depends on (Section 2.1 of the paper). A hazard gives each node a survival
factor: 0 failed, 1 unaffected. During an event an asset delivers
a_normal × K × (product of its node factors). Nodes a hazard does not name
keep factor 1.

Hazard values are mostly assumptions (A) that the Monte Carlo sweeps; the
comments give the row of the data sheet (Supplementary Material S2
of the paper) each one rests on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .economics import boiler_capex
from .params import COSTS, GCRP600, Costs, ReferencePlant
from .screening import Asset

A_RO = 0.95  # P13, A. Normal availability of SWRO trains
A_MED = 0.92  # P13, A. Normal availability of MED units
A_MSF = 0.92  # P13/F10, A. Normal availability of MSF units


@dataclass(frozen=True)
class Spread:
    """A value with a low-base-high range.

    ``kind`` says how the Monte Carlo samples it: "uniform", "loguniform"
    (durations), or "bernoulli": a binary state. The node is unaffected
    (factor 1) with probability ``base`` and affected (factor 0) otherwise;
    low-high is the range of that probability for sensitivity runs, not a
    second random draw.
    """

    low: float
    base: float
    high: float
    kind: str = "uniform"


def fixed(value: float) -> Spread:
    return Spread(value, value, value)


@dataclass(frozen=True)
class Hazard:
    """An event type.

    After the event the lost output ramps back over ``recovery`` days, first
    to ``restart_fraction`` of itself, where it stays until ``derate`` days
    after the event ends, and then returns in full.
    """

    label: str
    duration: Spread  # days the affected production is off
    recovery: Spread  # days for restarted production to ramp back
    factors: dict[str, Spread]
    restart_fraction: Spread = fixed(1.0)
    derate: Spread = fixed(0.0)  # days after the event at restart_fraction


# Z4c: MED and MSF share the intake, cooling-water system and steam supply, so in a bloom their
# factors come from one shared draw (each through its own range). Independent draws are a sensitivity.
LINKED = ("med_process", "msf_process")

HAZARDS: dict[str, Hazard] = {
    "severe_bloom": Hazard(
        "Severe bloom (2008-09 class)",
        duration=Spread(7.0, 55.0, 120.0, "loguniform"),  # H4: >7 d to ~4 months
        recovery=Spread(3.0, 5.0, 10.0),  # R6, A
        factors={
            "dmf": fixed(0.0),  # H4: conventional DMF plants shut
            "daf_uf": Spread(0.5, 0.9, 1.0),  # Z2: Fujairah 2 and Shuwaikh DAF kept output
            "med_process": Spread(0.8, 1.0, 1.0),  # Z4a, H5: Fujairah 2 MED unaffected, Kish MED ran with scaling control
            "msf_process": Spread(0.8, 1.0, 1.0),  # Z4b, H5: Fujairah 1 MSF ran while SWRO stopped
            "subsurface_intake": Spread(0.9, 1.0, 1.0),  # A
            "site_b": Spread(0.0, 0.3, 0.6, "bernoulli"),  # H3 regional extent, A
        },
        restart_fraction=Spread(0.5, 0.75, 1.0),  # Z11: Majis <=88%, Barka 1 partial, A
        derate=Spread(0.0, 30.0, 60.0),  # Z11, A
    ),
    "short_bloom": Hazard(
        "Short bloom (Israel 2026 class)",
        duration=Spread(2.0, 12.0, 12.0),  # H6: 2-12 d full outage by plant
        recovery=Spread(1.0, 3.0, 7.0),  # H6, A
        factors={
            "dmf": fixed(0.0),  # H6
            "daf_uf": Spread(0.1, 0.4, 0.8),  # Z3, A: Ashdod (UF) shut too; DAF weaker on ~2 um cells
            "med_process": Spread(0.8, 1.0, 1.0),  # Z4a, A
            "msf_process": Spread(0.8, 1.0, 1.0),  # Z4b, A
            "subsurface_intake": Spread(0.9, 1.0, 1.0),  # A
            "site_b": Spread(0.0, 0.2, 0.5, "bernoulli"),  # H6: 5 of 6 plants hit
        },
        restart_fraction=Spread(0.4, 0.5, 0.85),  # H6: 40-85% for weeks
        derate=Spread(20.0, 30.0, 40.0),  # H6: derated to day 33-45+ after onset
    ),
    "oil": Hazard(
        "Oil: precautionary intake closure",
        duration=Spread(0.125, 2.0, 3.0),  # O4: 3 h to 3 d documented
        recovery=Spread(0.1, 0.5, 1.0),  # A
        factors={
            "open_intake": fixed(0.0),  # O2/O4: precautionary closure
            "subsurface_intake": Spread(0.7, 1.0, 1.0),  # A
            "site_b": Spread(0.3, 0.5, 1.0, "bernoulli"),  # A. Slicks are local
        },
    ),
    "oil_fouled": Hazard(
        "Oil: intake fouled (1991 class)",
        duration=Spread(14.0, 30.0, 90.0, "loguniform"),  # O5, A. Khafji 1991, duration unknown
        recovery=Spread(2.0, 5.0, 10.0),  # A
        factors={
            "open_intake": fixed(0.0),
            "subsurface_intake": Spread(0.7, 1.0, 1.0),  # A
            "site_b": Spread(0.3, 0.5, 1.0, "bernoulli"),  # A
        },
    ),
    "ccgt_trip": Hazard(
        "CCGT trip (grid stays up)",
        duration=Spread(0.25, 1.0, 3.0),  # E21, A
        recovery=Spread(0.1, 0.25, 0.5),  # M8, A
        factors={"ccgt_steam": Spread(0.0, 0.0, 0.67)},  # M11: one GT of three, up to the whole block
    ),
    "grid_blackout": Hazard(
        "Grid blackout",
        duration=Spread(0.1, 0.5, 2.0),  # E21, A
        recovery=Spread(0.25, 0.5, 1.0),  # A
        factors={"grid": fixed(0.0)},
    ),
}


# --- Architectures ---------------------------------------------------------

_RO_MAIN = frozenset({"open_intake", "dmf", "grid"})

LABELS = {
    "A0": "Baseline SWRO",
    "A1": "Bloom-robust pretreatment (DAF+UF)",
    "A2": "Subsurface intake",
    "A3": "Second RO site",
    "A4": "RO + MED on CCGT steam",
    "A5": "RO + MED, own boiler",
    "A6": "RO + MSF on CCGT steam",
    "A7": "RO + MSF, own boiler",
}

MAX_SHARE = {"A2": 0.125}  # E14: 2 trains, 75,000 m³/d, about the largest subsurface intake demonstrated


@dataclass(frozen=True)
class Option:
    """An architecture: its assets and its extra capital cost over A0."""

    key: str
    share: float  # fraction of P2 given to the protected or diverse part
    assets: tuple[Asset, ...]
    extra_capex: float  # USD over A0
    delays: dict[str, float] = field(default_factory=dict)  # startup delay per asset, d

    @property
    def label(self) -> str:
        return " + ".join(LABELS[k] for k in self.key.split("+"))


# Capital premiums as fractions of SWRO capex on the protected share.
PREMIUM = {
    "daf_uf": 0.08,  # E12, S/D. Range 5-12%
    "subsurface": 0.13,  # E13, S. Range 6-20%
    "site_b": 0.15,  # X1, A. Separate site, intake and outfall
}


def build(key: str, share: float = 0.4, plant: ReferencePlant = GCRP600, costs: Costs = COSTS,
          max_share: dict[str, float] | None = None) -> Option:
    """GCRP-600 in architecture ``key`` with ``share`` of P2 protected or diverse.

    Total net capacity stays at P2: the protected or diverse part replaces
    baseline SWRO rather than adding to it. ``share`` is capped at
    MAX_SHARE; ``max_share`` overrides those caps for a sensitivity case, so
    a caller that costs a larger share must also raise the cap.
    """
    caps = MAX_SHARE if max_share is None else {**MAX_SHARE, **max_share}
    share = min(share, caps.get(key, 1.0))
    k = share * plant.capacity
    rest = plant.capacity - k
    c_ro = costs.swro_capex

    def ro(name: str, cap: float, path: frozenset[str]) -> Asset:
        return Asset(name, cap, A_RO, path)

    def med(path_extra: str) -> Asset:
        return Asset("MED", k, A_MED, frozenset({"open_intake", "med_process", path_extra, "grid"}))

    def msf(path_extra: str) -> Asset:
        return Asset("MSF", k, A_MSF, frozenset({"open_intake", "msf_process", path_extra, "grid"}))

    if key == "A0":
        return Option(key, 0.0, (ro("RO", plant.capacity, _RO_MAIN),), 0.0)
    if key == "A1":
        assets = (ro("RO", rest, _RO_MAIN), ro("RO DAF+UF", k, frozenset({"open_intake", "daf_uf", "grid"})))
        return Option(key, share, assets, PREMIUM["daf_uf"] * c_ro * k)
    if key == "A2":
        assets = (ro("RO", rest, _RO_MAIN), ro("RO subsurface", k, frozenset({"subsurface_intake", "grid"})))
        return Option(key, share, assets, PREMIUM["subsurface"] * c_ro * k)
    if key == "A3":
        assets = (ro("RO", rest, _RO_MAIN), ro("RO site B", k, frozenset({"site_b", "grid"})))
        return Option(key, share, assets, PREMIUM["site_b"] * c_ro * k)
    if key == "A4":
        assets = (ro("RO", rest, _RO_MAIN), med("ccgt_steam"))
        return Option(key, share, assets, (costs.med_capex - c_ro) * k)
    if key == "A5":
        assets = (ro("RO", rest, _RO_MAIN), med("boiler"))
        boiler = boiler_capex() * k  # E4a
        return Option(key, share, assets, (costs.med_capex - c_ro) * k + boiler)
    if key == "A6":
        assets = (ro("RO", rest, _RO_MAIN), msf("ccgt_steam"))
        return Option(key, share, assets, (costs.msf_capex - c_ro) * k)
    if key == "A7":
        assets = (ro("RO", rest, _RO_MAIN), msf("boiler"))
        return Option(key, share, assets, (costs.msf_capex - c_ro) * k + boiler_capex("msf") * k)
    raise KeyError(key)


def combine(*parts: tuple[str, float], plant: ReferencePlant = GCRP600, costs: Costs = COSTS,
            max_share: dict[str, float] | None = None) -> Option:
    """Several protections on one plant, e.g. ``combine(("A1", 0.4), ("A2", 0.125))``."""
    extra, total, assets = 0.0, 0.0, []
    for key, share in parts:
        option = build(key, share, plant, costs, max_share)
        assets += [a for a in option.assets if a.path != _RO_MAIN]
        extra += option.extra_capex
        total += option.share
    if total > 1.0:
        raise ValueError("shares add up to more than the plant")
    main = Asset("RO", (1.0 - total) * plant.capacity, A_RO, _RO_MAIN)
    return Option("+".join(k for k, _ in parts), total, (main, *assets), extra)


def asset_output(asset: Asset, factors: dict[str, float]) -> float:
    """Output during the event, m³/d."""
    return asset.capacity * asset.availability * math.prod(factors.get(n, 1.0) for n in asset.path)


def base_factors(hazard: Hazard, which: str = "base") -> dict[str, float]:
    """Node factors at the low, base or high end of each range.

    A Bernoulli node is a binary state, so "low" is its affected state (0),
    "high" its unaffected state (1) and "base" its expected value, the
    probability of escaping. Only "low" and "high" are physical outputs.
    """
    def pick(s: Spread) -> float:
        if s.kind == "bernoulli":
            return {"low": 0.0, "high": 1.0, "base": s.base}[which]
        return getattr(s, which)
    return {n: pick(s) for n, s in hazard.factors.items()}


def option_sigma(option: Option, hazard: Hazard, d_c: float, which: str = "base") -> float:
    """sigma_eff of an option under a hazard, with node factors at one end of their range."""
    f = base_factors(hazard, which)
    return sum(asset_output(a, f) for a in option.assets) / d_c


def restart_ratio(option: Option, factors: dict[str, float], d_c: float) -> float:
    """rho_r: output lost in the event that comes back after it, over D_c."""
    lost = sum(a.capacity * a.availability - asset_output(a, factors) for a in option.assets)
    return lost / d_c
