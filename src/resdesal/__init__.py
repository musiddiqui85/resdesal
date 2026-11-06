"""Resilience screening for large seawater desalination plants ."""

from .params import COSTS, EVENT_BANDS, GCRP600, Costs, ReferencePlant
from .screening import (
    Asset,
    CoverCost,
    breakeven_duration,
    bridge_days,
    cover_cost,
    critical_deficit,
    effective_sigma,
    meets_critical_demand,
    recovery_deficit,
    refill_time,
    required_sigma,
    storage_ratio,
    surviving_production,
    survival_ratio,
)
from .waterbalance import Outage, Result, simulate

__version__ = "1.0.0"
