"""Time-stepped water balance for a plant and its product storage.

    S[k+1] = S[k] + dt (Q_supply[k] - Q_delivered[k] - Q_curtailed[k] - Q_loss[k]),
    0 <= S <= S_max

Inputs are step-average rates in m³/d over steps of ``dt`` days. Water is
delivered up to demand while storage lasts. Production that storage cannot
hold is curtailed. Imports can be added to ``supply``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Result:
    dt: float
    storage: np.ndarray  # m³ at the n + 1 step boundaries
    supply: np.ndarray  # m³/d for each of the n steps
    demand: np.ndarray
    delivered: np.ndarray
    curtailed: np.ndarray

    @property
    def t(self) -> np.ndarray:
        """Step boundaries, days."""
        return np.arange(self.storage.size) * self.dt

    @property
    def min_storage(self) -> float:
        return float(self.storage.min())

    def shortfall(self, level: float | None = None) -> np.ndarray:
        """Shortfall rate per step, m³/d. With ``level`` (e.g. D_c), only below that level."""
        target = self.demand if level is None else np.minimum(self.demand, level)
        return np.maximum(target - self.delivered, 0.0)

    def unserved(self, level: float | None = None) -> float:
        """Unserved volume, m³. ``unserved(D_c)`` is U, equation (8) of the paper."""
        return float(self.shortfall(level).sum() * self.dt)

    def shortfall_days(self, level: float | None = None, tol: float = 1e-9) -> float:
        """Days with any shortfall."""
        return float(np.count_nonzero(self.shortfall(level) > tol) * self.dt)


def simulate(
    supply: np.ndarray,
    demand: np.ndarray | float,
    *,
    dt: float,
    s0: float,
    s_max: float,
    loss: np.ndarray | float = 0.0,
) -> Result:
    """Run the storage balance over the given supply and demand series."""
    if not 0.0 <= s0 <= s_max:
        raise ValueError("s0 must lie between 0 and s_max")
    supply = np.asarray(supply, dtype=float)
    demand = np.broadcast_to(np.asarray(demand, dtype=float), supply.shape)
    loss = np.broadcast_to(np.asarray(loss, dtype=float), supply.shape)

    n = supply.size
    storage = np.empty(n + 1)
    delivered = np.empty(n)
    curtailed = np.empty(n)
    storage[0] = s = s0
    for k in range(n):
        available = max(s + dt * (supply[k] - loss[k]), 0.0)
        delivered[k] = min(demand[k], available / dt)
        s = available - dt * delivered[k]
        curtailed[k] = max(s - s_max, 0.0) / dt
        s = min(s, s_max)
        storage[k + 1] = s
    return Result(dt, storage, supply, demand.copy(), delivered, curtailed)


@dataclass(frozen=True)
class Outage:
    """The canonical outage of screening.py as supply and demand series.

    The affected production stops for ``duration`` days. ``surviving`` m³/d
    of production comes online ``startup_delay`` days into the event. After
    the event the affected production restarts and rises linearly to
    ``restart_capacity`` (default: ``capacity - surviving``) over
    ``recovery_time`` days. Demand is ``critical_demand`` from the start of
    the event to the end of recovery and ``normal_demand`` otherwise.
    """

    duration: float
    surviving: float
    capacity: float
    critical_demand: float
    normal_demand: float
    startup_delay: float = 0.0
    recovery_time: float = 0.0
    restart_capacity: float | None = None
    lead: float = 0.0  # days of normal operation before the event
    tail: float = 0.0  # days simulated after recovery ends

    @property
    def end_of_recovery(self) -> float:
        return self.lead + self.duration + self.recovery_time

    def profiles(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Step-average supply and demand, m³/d.

        Rates are taken at step midpoints, which is exact for the linear ramp
        when event times fall on step boundaries.
        """
        n = round((self.end_of_recovery + self.tail) / dt)
        u = (np.arange(n) + 0.5) * dt - self.lead  # time since event start
        restart = (
            self.capacity - self.surviving if self.restart_capacity is None else self.restart_capacity
        )

        supply = np.full(n, float(self.capacity))
        during = (u >= 0.0) & (u < self.duration)
        supply[during] = np.where(u[during] >= self.startup_delay, self.surviving, 0.0)
        after = u >= self.duration
        if self.recovery_time > 0.0:
            ramp = np.clip((u[after] - self.duration) / self.recovery_time, 0.0, 1.0)
        else:
            ramp = 1.0
        supply[after] = self.surviving + restart * ramp

        critical = (u >= 0.0) & (u < self.duration + self.recovery_time)
        demand = np.where(critical, self.critical_demand, self.normal_demand)
        return supply, demand

    def simulate(
        self, dt: float, s0: float, s_max: float, loss: np.ndarray | float = 0.0
    ) -> Result:
        supply, demand = self.profiles(dt)
        return simulate(supply, demand, dt=dt, s0=s0, s_max=s_max, loss=loss)
