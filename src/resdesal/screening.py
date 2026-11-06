"""Closed-form screening relations for keeping critical supply through an outage.

Rates are in m³/d and times in days. The relations use three dimensionless
groups (Section 2.4 of the paper, derivations in Sections S2–S8 of its
Supplementary Material S1):

    beta  = S0 / (D_c T)   storage relative to critical demand over the event
    sigma = Q_s / D_c      production that survives, relative to critical demand
    theta = tau / T        startup delay of that production, relative to the event

The canonical outage has three phases, with demand held at D_c throughout:

1. From 0 to tau only stored water reaches the network.
2. From tau to T the surviving production delivers sigma D_c.
3. After T the shut-down production restarts. Its output rises linearly from
   zero to rho_r D_c over tau_rec, on top of the surviving production.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass


def storage_ratio(s0: float, d_c: float, duration: float) -> float:
    """beta = S0 / (D_c T)."""
    return s0 / (d_c * duration)


def survival_ratio(q_survive: float, d_c: float) -> float:
    """sigma = Q_s / D_c."""
    return q_survive / d_c


def bridge_days(s0: float, d_c: float) -> float:
    """Days that storage alone can supply critical demand."""
    return s0 / d_c


def _check_delay(startup_delay: float, duration: float) -> None:
    if not 0.0 <= startup_delay <= duration:
        raise ValueError("startup_delay must lie between 0 and the event duration")


def recovery_deficit(sigma: float, recovery_time: float, restart_ratio: float) -> float:
    """Shortfall added by the recovery ramp, in days of critical demand.

    The shortfall rate falls linearly from (1 - sigma) D_c to zero over
    tau_rec (1 - sigma) / rho_r days, which adds ½ tau_rec (1 - sigma)² / rho_r.
    Returns math.inf if sigma + rho_r < 1, since supply never reaches D_c.
    """
    if restart_ratio <= 0.0:
        raise ValueError("restart_ratio must be positive")
    if sigma >= 1.0:
        return 0.0
    if sigma + restart_ratio < 1.0:
        return math.inf
    return 0.5 * recovery_time * (1.0 - sigma) ** 2 / restart_ratio


def critical_deficit(
    duration: float,
    sigma: float,
    *,
    startup_delay: float = 0.0,
    recovery_time: float = 0.0,
    restart_ratio: float = math.inf,
) -> float:
    """Largest cumulative shortfall over the outage, in days of critical demand.

    Critical demand is met throughout if and only if S0 / D_c is at least this.
    The shortfall rate never rises over the canonical outage, so the largest
    cumulative shortfall is reached where the rate first drops to zero.
    """
    _check_delay(startup_delay, duration)
    if sigma >= 1.0:
        return startup_delay
    return (
        startup_delay
        + (1.0 - sigma) * (duration - startup_delay)
        + recovery_deficit(sigma, recovery_time, restart_ratio)
    )


def meets_critical_demand(
    s0: float, d_c: float, duration: float, q_survive: float, **outage: float
) -> bool:
    """True if storage plus surviving production meet D_c through the outage."""
    return s0 / d_c >= critical_deficit(duration, q_survive / d_c, **outage)


def required_sigma(
    duration: float,
    s0: float,
    d_c: float,
    *,
    startup_delay: float = 0.0,
    recovery_time: float = 0.0,
    restart_ratio: float = math.inf,
) -> float:
    """Smallest sigma that keeps D_c met; math.inf if no sigma can.

    With b = S0 / D_c, x = 1 - sigma and a = tau_rec / (2 rho_r), the
    condition b >= tau + x (T - tau) + a x² is a quadratic in x.
    """
    _check_delay(startup_delay, duration)
    if restart_ratio <= 0.0:
        raise ValueError("restart_ratio must be positive")
    b = s0 / d_c
    tau, span = startup_delay, duration - startup_delay
    if b < tau:
        return math.inf
    a = recovery_time / (2.0 * restart_ratio)
    if a == 0.0:
        x = math.inf if span == 0.0 else (b - tau) / span
    else:
        x = (-span + math.sqrt(span**2 + 4.0 * a * (b - tau))) / (2.0 * a)
    return max(0.0, 1.0 - x, 1.0 - restart_ratio)


def refill_time(volume: float, capacity: float, normal_demand: float) -> float:
    """Days to put ``volume`` back into storage at full production."""
    spare = capacity - normal_demand
    return volume / spare if spare > 0.0 else math.inf


# --- Shared dependencies --------------------------------------------------


@dataclass(frozen=True)
class Asset:
    """A production asset and the shared nodes its output depends on.

    ``path`` names nodes such as "open_intake", "dmf", "grid" or "ccgt_steam".
    """

    name: str
    capacity: float  # K_i, m³/d
    availability: float = 1.0  # a_i during the event, 0-1
    path: frozenset[str] = frozenset()


def surviving_production(assets: Iterable[Asset], failed: Iterable[str]) -> float:
    """Sum of a_i K_i delta_i, where delta_i = 0 if any node on asset i's path fails."""
    failed = set(failed)
    return sum(a.availability * a.capacity for a in assets if not set(a.path) & failed)


def effective_sigma(assets: Iterable[Asset], failed: Iterable[str], d_c: float) -> float:
    """sigma_eff = sum(a_i K_i delta_i) / D_c."""
    return surviving_production(assets, failed) / d_c


# --- Capital-cost break-even ----------------------------------------------


@dataclass(frozen=True)
class CoverCost:
    """Capital cost of covering an outage with one measure alone."""

    extra_storage: float  # m³ beyond S0, storage-only cover
    storage_cost: float  # USD
    surviving_capacity: float  # m³/d, capacity-only cover
    capacity_cost: float  # USD


def breakeven_duration(c_capacity: float, c_storage: float, startup_delay: float = 0.0) -> float:
    """Outage length above which surviving capacity costs less than extra storage.

    Storage-only cover costs c_s (D_c T - S0). Capacity that comes online after
    tau costs c_b (D_c T - S0) / (T - tau). They are equal at
    T* = tau + c_b / c_s, whatever S0 and D_c. Capital cost only.
    """
    return startup_delay + c_capacity / c_storage


def cover_cost(
    duration: float,
    s0: float,
    d_c: float,
    c_capacity: float,
    c_storage: float,
    *,
    startup_delay: float = 0.0,
) -> CoverCost:
    """Cost of storage-only versus capacity-only cover, with no recovery ramp."""
    extra = max(0.0, d_c * duration - s0)
    q = required_sigma(duration, s0, d_c, startup_delay=startup_delay) * d_c
    return CoverCost(extra, c_storage * extra, q, c_capacity * q)
