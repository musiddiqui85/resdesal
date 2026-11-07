"""Draw the Supplementary Material S1 figures into figures/ (same journal format as make_figures.py).

    PYTHONPATH=src python scripts/make_si_figures.py

Fig. S1 checks the closed-form shortfall against the time-stepping
water-balance simulator: one worked event (a) and 300 random events (b).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_figures import BLUE_STEPS, DC, INK, INK2, MM, MUTED, PALE, SIGMA, WIDTH, finish, panel, save  # noqa: E402

from resdesal import GCRP600 as P  # noqa: E402
from resdesal import Outage, critical_deficit  # noqa: E402

DT = 0.01
SEED = 7


def outage(**kw) -> Outage:
    return Outage(capacity=P.capacity, critical_demand=P.critical_demand, normal_demand=P.normal_demand, **kw)


def random_cases(n: int = 300) -> list[dict]:
    """Random events on the simulator's time grid, with sigma + rho_r >= 1 so the closed form is finite."""
    rng = np.random.default_rng(SEED)
    cases = []
    while len(cases) < n:
        T = float(rng.choice([7.0, 12.0, 30.0, 55.0]))
        sigma = round(float(rng.uniform(0.0, 0.95)), 3)
        tau = round(float(rng.uniform(0.0, min(2.5, T))), 2)
        rec = round(float(rng.uniform(0.0, 10.0)), 2)
        rho = float(rng.choice([0.5, 1.125, 1.5, 2.5]))
        phi = round(float(rng.uniform(0.5, 1.0)), 3)
        if sigma + rho >= 1.0:
            cases.append(dict(T=T, sigma=sigma, tau=tau, rec=rec, rho=rho, phi=phi))
    return cases


def unserved_pair(c: dict) -> tuple[float, float]:
    s0 = c["phi"] * P.storage
    ev = outage(duration=c["T"], surviving=c["sigma"] * P.critical_demand, startup_delay=c["tau"],
                recovery_time=c["rec"], restart_capacity=c["rho"] * P.critical_demand, tail=2.0)
    sim = ev.simulate(DT, s0=s0, s_max=P.storage).unserved(P.critical_demand)
    deficit = critical_deficit(c["T"], c["sigma"], startup_delay=c["tau"], recovery_time=c["rec"],
                               restart_ratio=c["rho"])
    return max(0.0, deficit * P.critical_demand - s0), sim


def fig_s1() -> None:
    fig = plt.figure(figsize=(WIDTH, 82 * MM), layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], height_ratios=[1, 1])
    a1, a2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
    b = fig.add_subplot(gs[:, 1])

    lead = 2.0
    ev = outage(duration=12.0, surviving=0.5 * P.critical_demand, startup_delay=1.0, recovery_time=5.0,
                lead=lead, tail=14.0)
    res = ev.simulate(DT, s0=P.storage, s_max=P.storage)
    t = res.t - lead
    tm = t[:-1] + DT / 2
    k = 1e-3
    a1.axvspan(0, 12, color=PALE, lw=0)
    a1.plot(tm, res.supply * k, color=BLUE_STEPS[1], label="Supply")
    a1.plot(tm, res.demand * k, color=INK, lw=0.9, ls=(0, (4, 2)), label="Demand")
    a1.plot(tm, res.delivered * k, color="#eb6834", lw=0.9, label="Delivered")
    a1.set_ylabel("Flow (thousand m³/d)")
    a1.set_ylim(0, 650)
    a1.set_xlim(t[0], t[-1])
    a1.tick_params(labelbottom=False)
    a1.legend(loc="lower right", ncol=3, borderaxespad=0.2)
    a1.text(6, 615, "event", ha="center", va="top", fontsize=6, color=INK2)
    a1.text(14.5, 615, "recovery", ha="center", va="top", fontsize=6, color=INK2)
    panel(a1, "a")

    a2.axvspan(0, 12, color=PALE, lw=0)
    a2.plot(t, res.storage * k, color=BLUE_STEPS[2])
    a2.set_ylabel("Storage (thousand m³)")
    a2.set_xlabel("Time from event onset (days)")
    a2.set_xlim(t[0], t[-1])
    a2.set_ylim(-20, 650)
    short = res.shortfall(P.critical_demand) > 1e-6
    if short.any():
        a2.axvspan(tm[short][0], tm[short][-1], color="#eb6834", alpha=0.15, lw=0)
        a2.text(tm[short].mean(), 330, f"unserved\n{res.unserved(P.critical_demand) / 1e6:.2f} million m³",
                ha="center", fontsize=6, color=INK)
    cf = max(0.0, critical_deficit(12.0, 0.5, startup_delay=1.0, recovery_time=5.0,
                                   restart_ratio=(P.capacity - 0.5 * P.critical_demand) / P.critical_demand)
             * P.critical_demand - P.storage)
    a2.text(t[-1] - 0.3, 330, f"closed form: {cf / 1e6:.2f} million m³", ha="right", va="top", fontsize=6,
            color=INK)

    pairs = np.array([unserved_pair(c) for c in random_cases()]) / 1e6
    lim = pairs.max() * 1.05
    b.plot([0, lim], [0, lim], color=MUTED, lw=0.8)
    b.plot(pairs[:, 0], pairs[:, 1], "o", ms=2.8, color=BLUE_STEPS[1], mec="white", mew=0.3, alpha=0.9)
    err = np.abs(pairs[:, 1] - pairs[:, 0]).max() * 1e6
    b.text(0.04 * lim, 0.96 * lim, f"300 random events\nlargest difference {err:.0f} m³", va="top", fontsize=6,
           color=INK)
    b.set_xlim(0, lim)
    b.set_ylim(0, lim)
    b.set_aspect("equal")
    b.set_xlabel("Unserved water, closed form (million m³)")
    b.set_ylabel("Unserved water, simulator (million m³)")
    panel(b, "b")
    finish(fig, (a1, b))
    save(fig, "figS1_verification")
    print(f"largest difference over 300 events: {err:.3f} m3")


if __name__ == "__main__":
    fig_s1()
    print("Wrote figures/figS1_verification")
