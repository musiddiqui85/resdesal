"""Monte Carlo screening of architectures across hazards (Section 2.6 of the paper).

Each draw samples the outage length, recovery time, critical demand share,
how full the tanks are, and every node factor of the hazard. The shortfall
is then evaluated in closed form, so 10⁵ draws take well under a second.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .architectures import LINKED, Hazard, Option, Spread, asset_output
from .params import GCRP600, ReferencePlant


def max_shortfall(duration, sigmas, delays=None, recovery_time=0.0, restart_ratio=np.inf,
                  restart_fraction=1.0, derate_time=0.0, shape="plateau"):
    """Largest cumulative shortfall, in days of D_c, with several surviving assets.

    ``sigmas`` has the assets on its last axis; asset j comes online
    ``delays[j]`` days into the event (an asset not online by T counts from T).
    After the event the lost output ramps linearly over ``recovery_time``
    days to ``restart_fraction × restart_ratio`` D_c. With ``shape`` "plateau"
    it stays there until ``derate_time`` days after the event and then returns
    in full; with "rising" it climbs linearly from there to full output by the
    same day (a sensitivity on the shape, Section S5). If
    ``derate_time`` is no longer than the ramp, including zero, the plateau has
    zero length and full output returns at the end of the ramp. With one asset
    and no derating this equals screening.critical_deficit.
    """
    sig = np.asarray(sigmas, dtype=float)
    tau = np.zeros(sig.shape[-1]) if delays is None else np.asarray(delays, dtype=float)
    T = np.asarray(duration, dtype=float)
    rate = np.ones(np.broadcast_shapes(T.shape, sig.shape[:-1]))
    t = np.zeros_like(rate)
    n = np.zeros_like(rate)
    peak = np.zeros_like(rate)
    for j in np.argsort(tau, kind="stable"):
        d = np.minimum(tau[j], T)
        n = n + rate * (d - t)
        t = d
        peak = np.maximum(peak, n)
        rate = rate - sig[..., j]
    n = n + rate * (T - t)
    peak = np.maximum(peak, n)

    rho = np.asarray(restart_ratio, dtype=float)
    rho_p = np.asarray(restart_fraction, dtype=float) * rho
    rec = np.asarray(recovery_time, dtype=float)
    derate = np.asarray(derate_time, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        # Shortfall ends during the ramp: a triangle.
        triangle = 0.5 * rec * rate**2 / rho_p
        span = np.maximum(derate, rec) - rec  # length of the derated period after the ramp
        if shape == "plateau":
            # The whole ramp, then the derated plateau until full output returns.
            after = (rate - rho_p) * span
        elif shape == "rising":
            # The whole ramp, then a linear climb from the derated level to full output over `span`:
            # the shortfall rate falls from a to b, and the area is cut where it reaches zero.
            a, b = rate - rho_p, rate - rho
            after = np.where(b >= 0.0, 0.5 * span * (a + b), 0.5 * span * a * a / (a - b))
            after = np.where(span > 0.0, after, 0.0)
        else:
            raise ValueError(shape)
        plateau = rec * (rate - 0.5 * rho_p) + after
    within = rate <= rho_p
    extra = np.where(within, triangle, plateau)
    # Full output (rho) that still falls short of critical demand never ends the shortfall.
    extra = np.where(~within & (rate > rho), np.inf, extra)
    extra = np.where(rate > 0.0, extra, 0.0)
    return np.maximum(peak, n + extra)


LAWS = ("base", "triangular", "adverse", "uniform_duration")


def from_uniform(spread: Spread, u: np.ndarray, law: str = "base", worse: str = "low") -> np.ndarray:
    """Map uniform variates on [0, 1) through the law of ``spread`` (see draw)."""
    if spread.low == spread.high:
        return np.full(u.shape, spread.base)
    if spread.kind == "bernoulli":
        return (u < spread.base).astype(float)
    log = spread.kind == "loguniform" and law != "uniform_duration"
    lo, hi = (np.log(spread.low), np.log(spread.high)) if log else (spread.low, spread.high)
    if law in ("triangular", "adverse"):
        mode = spread.base if law == "triangular" else (spread.low if worse == "low" else spread.high)
        mode = np.log(mode) if log else mode
        # Inverse CDF, as numpy's triangular sampler computes it.
        base, left = hi - lo, mode - lo
        x = np.where(u <= left / base, lo + np.sqrt(u * (left * base)), hi - np.sqrt((1.0 - u) * ((hi - mode) * base)))
    else:
        x = lo + (hi - lo) * u
    return np.exp(x) if log else x


def draw(spread: Spread, n: int, rng: np.random.Generator, law: str = "base", worse: str = "low") -> np.ndarray:
    """Sample a Spread. ``law`` changes the weighting for sensitivity runs: "triangular" peaks
    each range at its base value (on a log scale for durations); "adverse" peaks it at its
    worse end, ``worse`` ("low" or "high"); "uniform_duration" samples log-uniform ranges
    uniformly, which puts more weight on long events. Uses one uniform variate per sample
    (none for a fixed value), so the draws equal numpy's own samplers bit for bit."""
    if spread.low == spread.high:
        return np.full(n, spread.base)
    return from_uniform(spread, rng.random(n), law, worse)


@dataclass(frozen=True)
class Draws:
    duration: np.ndarray  # d
    critical_demand: np.ndarray  # m³/d
    storage: np.ndarray  # m³ at onset
    sigma: np.ndarray  # total surviving output / D_c
    shortfall: np.ndarray  # largest cumulative shortfall, days of D_c

    @property
    def met(self) -> np.ndarray:
        """Draws where critical demand is met throughout."""
        return self.shortfall <= self.storage / self.critical_demand

    @property
    def unserved(self) -> np.ndarray:
        """Unserved critical water, m³."""
        return np.maximum(self.shortfall * self.critical_demand - self.storage, 0.0)

    @property
    def unserved_fraction(self) -> np.ndarray:
        """Legacy U / (D_c T), kept for backward compatibility. Its numerator includes the
        recovery period but its denominator does not, so it is not a share of demand."""
        return self.unserved / (self.critical_demand * self.duration)

    @property
    def unserved_days(self) -> np.ndarray:
        """Unserved critical water in days of critical demand, U / D_c."""
        return self.unserved / self.critical_demand


def sample_inputs(
    hazard: Hazard,
    n: int,
    seed: int = 0,
    dc_share: tuple[float, float] = (0.2, 0.6),  # P12
    fill: tuple[float, float] = (0.5, 1.0),  # phi, A
    law: str = "base",
    linked: bool = True,
) -> dict[str, np.ndarray]:
    """Draw the uncertain inputs. The same seed gives the same draws for every option.

    With ``linked`` the nodes in LINKED (MED and MSF) take one shared uniform variate,
    drawn where the first of them comes; otherwise each node is drawn on its own.
    """
    rng = np.random.default_rng(seed)
    x = {
        "duration": draw(hazard.duration, n, rng, law, "high"),
        "recovery": draw(hazard.recovery, n, rng, law, "high"),
        "dc_share": rng.uniform(*dc_share, n),
        "fill": rng.uniform(*fill, n),
    }
    shared = None
    for node, spread in hazard.factors.items():
        if linked and node in LINKED and spread.low != spread.high:
            if shared is None:
                shared = rng.random(n)
            x[node] = from_uniform(spread, shared, law)
        else:
            x[node] = draw(spread, n, rng, law)
    x["restart_fraction"] = draw(hazard.restart_fraction, n, rng, law)
    x["derate"] = draw(hazard.derate, n, rng, law, "high")
    return x


def evaluate(option: Option, hazard: Hazard, x: dict[str, np.ndarray], plant: ReferencePlant = GCRP600,
             shape: str = "plateau") -> Draws:
    """Shortfall of ``option`` for given input draws (from sample_inputs or a Sobol design). ``shape`` is the
    restart shape after the ramp, "plateau" (the model) or "rising" (a sensitivity; see max_shortfall)."""
    d_c = x["dc_share"] * plant.capacity
    factors = {node: x[node] for node in hazard.factors}
    out = np.stack(
        [np.broadcast_to(asset_output(a, factors), d_c.shape) for a in option.assets], axis=-1
    )
    lost = sum(a.capacity * a.availability for a in option.assets) - out.sum(axis=-1)
    sig = out / d_c[:, None]
    delays = [option.delays.get(a.name, 0.0) for a in option.assets]
    peak = max_shortfall(x["duration"], sig, delays, x["recovery"], lost / d_c,
                         x.get("restart_fraction", hazard.restart_fraction.base),
                         x.get("derate", hazard.derate.base), shape)
    return Draws(x["duration"], d_c, x["fill"] * plant.storage, sig.sum(axis=-1), peak)


def run(option: Option, hazard: Hazard, n: int = 20_000, seed: int = 0, **ranges) -> Draws:
    return evaluate(option, hazard, sample_inputs(hazard, n, seed, **ranges))
