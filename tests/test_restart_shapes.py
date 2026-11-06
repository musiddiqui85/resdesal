"""Derated restart: a zero-length plateau, full output short of demand, and the rising-shape sensitivity."""

import numpy as np
import pytest

from resdesal.montecarlo import max_shortfall


def test_zero_length_derate_returns_full_output_at_ramp_end():
    """A derated period of exactly zero behaves like a short one: full output at the end of the ramp."""
    kw = dict(duration=10, sigmas=[[0.2]], recovery_time=5, restart_ratio=2, restart_fraction=0.3)
    zero = max_shortfall(derate_time=0, **kw)[0]
    tiny = max_shortfall(derate_time=1e-9, **kw)[0]
    assert zero == pytest.approx(10.5)  # 8 days in the event plus 2.5 in the ramp
    assert zero == pytest.approx(tiny)


def test_full_output_short_of_demand_never_ends_the_shortfall():
    assert np.isinf(max_shortfall(duration=10, sigmas=[[0.2]], recovery_time=5, restart_ratio=0.5,
                                  restart_fraction=0.5, derate_time=0)[0])


def _quadrature(T, s, rec, rho, g, L, dt=2e-4):
    """Peak cumulative shortfall of the rising shape on a fine time grid (days of D_c)."""
    span = max(L, rec) - rec
    t = np.arange(0, max(L, rec) + 50, dt) + dt / 2
    rising = s + g * rho + (1 - g) * rho * (t - rec) / span if span > 0 else np.full_like(t, s + rho)
    sup = np.where(t < rec, s + g * rho * t / rec, np.where(t < rec + span, rising, s + rho))
    return (1 - s) * T + max(np.cumsum((1 - sup) * dt).max(), 0.0)


def test_rising_shape_matches_quadrature():
    rng = np.random.default_rng(7)
    for _ in range(40):
        T, s, rec = rng.uniform(2, 50), rng.uniform(0, 0.9), rng.uniform(0.5, 10)
        rho, g, L = rng.uniform(1.0 - s, 1.5), rng.uniform(0.2, 1), rng.uniform(0, 60)
        cf = max_shortfall(T, [[s]], recovery_time=rec, restart_ratio=rho, restart_fraction=g, derate_time=L,
                           shape="rising")[0]
        assert cf == pytest.approx(_quadrature(T, s, rec, rho, g, L), abs=1e-3)


def test_rising_shape_never_exceeds_the_plateau():
    rng = np.random.default_rng(3)
    T, s = rng.uniform(2, 50, 500), rng.uniform(0, 0.9, 500)
    kw = dict(recovery_time=rng.uniform(0.5, 10, 500), restart_ratio=rng.uniform(0.5, 1.5, 500) + 1 - s,
              restart_fraction=rng.uniform(0.2, 1, 500), derate_time=rng.uniform(0, 60, 500))
    plateau = max_shortfall(T, s[:, None], **kw)
    rising = max_shortfall(T, s[:, None], shape="rising", **kw)
    assert np.all(rising <= plateau + 1e-12)
