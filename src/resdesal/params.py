"""GCRP-600 reference plant values used by the screening code.

Each value carries its ID in the data sheet of the reference plant (Supplementary
Material S2 of the paper) and its status there: S sourced, D derived, A assumption.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReferencePlant:
    capacity: float = 600_000.0  # P2, S. Net product, m³/d
    storage: float = 600_000.0  # P11, S. On-site product storage, m³ (24 h)
    critical_demand: float = 240_000.0  # P12, A. 40% of P2, swept 20-60%
    normal_demand: float = 540_000.0  # P14, A. 90% of P2, swept 85-95%
    train_capacity: float = 37_500.0  # P5, A. First-pass train, m³/d
    med_unit_capacity: float = 29_630.0  # M1, S. MED-TVC unit, m³/d
    msf_unit_capacity: float = 92_600.0  # F1, S. MSF unit, m³/d (Ras Al Khair: 92,582 at design rating)


@dataclass(frozen=True)
class Costs:
    swro_capex: float = 1_100.0  # E1, D. USD per m³/d, range 1,083-1,216
    med_capex: float = 1_400.0  # E4, S/D. USD per m³/d; World Bank MED-TVC average, Gulf contracts 900-1,820
    med_capex_range: tuple[float, float] = (900.0, 2_300.0)  # E4
    msf_capex: float = 2_100.0  # F7, S. USD per m³/d; World Bank 2019 average 2.1 M USD per MLD (1.7-3.1), 2016 USD
    msf_capex_range: tuple[float, float] = (1_700.0, 2_800.0)  # F7
    storage_capex: float = 180.0  # E15, D. USD per m³ stored, range 160-200
    storage_capex_range: tuple[float, float] = (160.0, 200.0)  # E15, D


GCRP600 = ReferencePlant()
COSTS = Costs()

# Outage durations of the affected production, days.
EVENT_BANDS = {
    "short": 12.0,  # H6, S. Israel 2026, 9-14 d to partial restart
    "long": 55.0,  # H4, S. Barka 2008
    "extreme": 120.0,  # H4 upper bound, unverified. Sensitivity only
}
