"""Israel 2026 bloom: hindcast of plant output and a frozen forecast (Section 2.7 of the paper and Section S16
of its Supplementary Material S1).

    PYTHONPATH=src python scripts/israel2026_hindcast.py
    PYTHONPATH=src python scripts/israel2026_hindcast.py --compute-only   # never touch the forecast
    PYTHONPATH=src python scripts/israel2026_hindcast.py --refreeze   # replace the frozen forecast

The short-bloom ranges (data-sheet rows H6 and Z11) were set from these same news reports, so the hindcast is not
out-of-sample. What it tests is the shape the model gives a restarted plant (architectures.Hazard): a
linear ramp to a restart fraction r, a flat derated plateau, then full output. A plant that shut
completely after T days delivers, as a fraction of its normal output,

    0 until T, r (t - T) / tau_rec over the ramp, r until T + max(D, tau_rec), then 1,

which is the trajectory montecarlo.max_shortfall integrates. Steps:

1. Envelope: the range of that output on each reported date over the published ranges.
2. Fit: the smallest misfit one trajectory achieves for each plant, with the published ranges and
   with free ranges, against a rising alternative in which output climbs linearly from r to full.
3. Shortfall: the production deficit against critical demand up to the last report, with no storage
   or imports, under the model's ranges and under the reports. The reports are joined by straight lines
   and bounded by the earliest and latest steps they allow, which assumes output stays between
   adjacent reports; the reports themselves do not rule out a dip or overshoot between them. The
   check records whether the two ranges overlap and whether the reports' range lies inside the
   model's; overlap is a weaker statement than containment.
4. Forecast: when Ashdod and Palmachim reach full output under each structure. It is written once,
   before any report after the evidence cut-off is read, and is not rewritten without --refreeze.
   --compute-only never writes it and stops if it is missing: the frozen record is committed
   evidence, not a regenerable output.

Each report's limit_cause is bloom, other (e.g. Sorek B's pump fault), unstated or none (full
output). Fits, envelope and shortfall use every report except "other", so they are all-cause
consistency checks; only "bloom" reports can score the bloom-specific forecast.

Inputs: data/israel2026_plants.csv and data/israel2026_output.csv. Writes
results/israel2026_hindcast.csv, israel2026_fits.csv, israel2026_shortfall.csv and
israel2026_forecast.json.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from resdesal.architectures import HAZARDS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results"
FORECAST = OUT / "israel2026_forecast.json"

H = HAZARDS["short_bloom"]
PUBLISHED = dict(restart=(H.restart_fraction.low, H.restart_fraction.high),
                 recovery=(H.recovery.low, H.recovery.high), hold=(H.derate.low, H.derate.high))
BASE = dict(restart=H.restart_fraction.base, recovery=H.recovery.base, hold=H.derate.base)
# Wide enough that a misfit left over comes from the shape, not from the range.
FREE = dict(restart=(0.0, 1.0), recovery=(0.25, 30.0), hold=(0.0, 120.0))
FREE_RISE = dict(restart=(0.0, 1.0), recovery=(0.25, 30.0), hold=(0.5, 200.0))

GRID = 61  # points per parameter in the fits
GRID_SHORTFALL = 15  # points per parameter in the shortfall envelope
STEP = 0.05  # days, shortfall integration
TOL = 0.02  # fits within this of the best misfit count as equally good for the forecast
DC_SHARES = (0.2, 0.4, 0.6)  # critical demand as a share of the plant's normal output (P12 range)


# --- Trajectories ------------------------------------------------------------


def plateau(t, T, r, rec, hold):
    """Model: output of a fully shut plant, fraction of normal, t days after shutdown.

    ``hold`` is the derate D: days after the event at r before full output returns.
    """
    u = t - T
    ramp = r * np.clip(u / rec, 0.0, 1.0)
    return np.where(u < 0.0, 0.0, np.where(u < np.maximum(hold, rec), ramp, 1.0))


def rising(t, T, r, rec, hold):
    """Alternative: the same ramp to r, then a linear climb to full output over ``hold`` days."""
    u = t - T
    ramp = r * np.clip(u / rec, 0.0, 1.0)
    climb = r + (1.0 - r) * np.clip((u - rec) / hold, 0.0, 1.0)
    return np.where(u < 0.0, 0.0, np.where(u < rec, ramp, climb))


def full_day(f, T, r, rec, hold):
    """Days after shutdown at which output first reaches full."""
    return T + np.maximum(hold, rec) if f is plateau else T + rec + hold


STRUCTURES = {
    "P1 plateau, published ranges": (plateau, PUBLISHED),
    "P1 plateau, free ranges": (plateau, FREE),
    "Rising, free ranges": (rising, FREE_RISE),
}


# --- Data --------------------------------------------------------------------


def load() -> dict[str, dict]:
    plants = {}
    with open(DATA / "israel2026_plants.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            day0 = dt.date.fromisoformat(r["shutdown"])
            T = sorted({(dt.date.fromisoformat(r[k]) - day0).days for k in ("restart_early", "restart_late")})
            plants[r["plant"]] = dict(day0=day0, T=T, obs=[])
    with open(DATA / "israel2026_output.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            p = plants[r["plant"]]
            date = dt.date.fromisoformat(r["date"])
            if r["limit_cause"] not in ("bloom", "other", "unstated", "none"):
                raise ValueError(f"unknown limit_cause {r['limit_cause']!r}")
            p["obs"].append(dict(date=date, day=(date - p["day0"]).days, low=float(r["low"]),
                                 high=float(r["high"]), cause=r["limit_cause"],
                                 limited=r["limit_cause"] != "other", note=r["note"]))
    for p in plants.values():
        p["obs"].sort(key=lambda o: o["date"])
        p["limited"] = [o for o in p["obs"] if o["limited"]]
    return plants


def mid_T(p: dict) -> float:
    return float(np.mean(p["T"]))


# --- Grids and fits ----------------------------------------------------------


def params(T: list[int], ranges: dict, n: int) -> list[np.ndarray]:
    """T, r, tau_rec, hold over a full grid, as flat arrays."""
    axes = [np.asarray(T, float)] + [np.linspace(*ranges[k], n) for k in ("restart", "recovery", "hold")]
    return [m.ravel() for m in np.meshgrid(*axes, indexing="ij")]


def misfit(y: np.ndarray, obs: list[dict]) -> np.ndarray:
    """Largest distance from each trajectory to the reported intervals (0 if inside all of them)."""
    low = np.array([o["low"] for o in obs])
    high = np.array([o["high"] for o in obs])
    return np.maximum(np.maximum(low - y, y - high), 0.0).max(axis=-1)


def fit(p: dict, f, ranges: dict, n: int = GRID):
    """Every grid trajectory and its misfit to the plant's bloom-limited reports."""
    t = np.array([o["day"] for o in p["limited"]], float)
    g = params(p["T"], ranges, n)
    y = f(t, *(x[:, None] for x in g))
    return g, misfit(y, p["limited"])


def envelope(p: dict, day: float) -> tuple[float, float, float]:
    """Model output on ``day``: lowest and highest over the published ranges, and at base values."""
    g = params(p["T"], PUBLISHED, GRID)
    y = plateau(float(day), *g)
    base = plateau(float(day), mid_T(p), BASE["restart"], BASE["recovery"], BASE["hold"])
    return float(y.min()), float(y.max()), float(base)


# --- Shortfall ---------------------------------------------------------------


def unserved(y: np.ndarray, dc: float) -> np.ndarray:
    """Production deficit against critical demand, days of D_c, with no storage or imports, for output
    sampled every STEP days. Not the water customers went without, which tanks and imports reduce."""
    return np.maximum(dc - y, 0.0).sum(axis=-1) * STEP / dc


def observed_path(p: dict, t: np.ndarray) -> np.ndarray:
    """The reports joined by straight lines: 0 to the restart, then each report's midpoint."""
    xs = [0.0, mid_T(p)] + [float(o["day"]) for o in p["limited"]]
    ys = [0.0, 0.0] + [0.5 * (o["low"] + o["high"]) for o in p["limited"]]
    if any(b <= a for a, b in zip(xs, xs[1:])):
        raise ValueError("reports must come after the restart and in date order")
    return np.interp(t, xs, ys)


def observed_bounds(p: dict, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lowest and highest output the reports allow between report dates.

    The low path restarts at the late restart date and holds the lower of each pair of adjacent
    report lows until the later report; the high path restarts early at the first report's high
    and holds the higher of each pair of highs. These bound the path only on the assumption that
    output stays between adjacent reports; the reports alone do not exclude a dip between them.
    """
    days = [float(o["day"]) for o in p["limited"]]
    lows = [o["low"] for o in p["limited"]]
    highs = [o["high"] for o in p["limited"]]
    lower, upper = np.zeros_like(t), np.zeros_like(t)
    for out, edges, vals, pick in ((lower, [max(p["T"]), *days], [0.0, *lows], min),
                                   (upper, [min(p["T"]), *days], [highs[0], *highs], max)):
        for k in range(len(days)):
            out[(t > edges[k]) & (t <= edges[k + 1])] = pick(vals[k], vals[k + 1])
    return lower, upper


def shortfall_rows(plants: dict) -> list[dict]:
    rows = []
    for name, p in plants.items():
        end = p["limited"][-1]["day"]
        t = (np.arange(round(end / STEP)) + 0.5) * STEP
        lower, upper = observed_bounds(p, t)
        line = observed_path(p, t)
        g = params(p["T"], PUBLISHED, GRID_SHORTFALL)
        model = plateau(t[None, :], *(x[:, None] for x in g))
        base = plateau(t, mid_T(p), BASE["restart"], BASE["recovery"], BASE["hold"])
        for share in DC_SHARES:
            u = unserved(model, share)
            lo, hi = float(unserved(upper, share)), float(unserved(lower, share))
            mid = float(unserved(line, share))
            m_lo, m_hi = float(u.min()), float(u.max())
            rows.append(dict(plant=name, window_days=end, dc_share=share, reports_low=lo,
                             reports_line=mid, reports_high=hi,
                             p1_base=float(unserved(base, share)), p1_low=m_lo, p1_high=m_hi,
                             overlap=bool(lo <= m_hi + 1e-9 and hi >= m_lo - 1e-9),
                             contained=bool(lo >= m_lo - 1e-9 and hi <= m_hi + 1e-9),
                             line_within=bool(m_lo - 1e-9 <= mid <= m_hi + 1e-9)))
    return rows


# --- Forecast ----------------------------------------------------------------


def date_range(p: dict, days: np.ndarray, at_limit: bool | None = None) -> dict:
    lo, hi = float(days.min()), float(days.max())
    out = dict(earliest=str(p["day0"] + dt.timedelta(days=round(lo))),
               latest=str(p["day0"] + dt.timedelta(days=round(hi))),
               days_after_shutdown=[round(lo, 1), round(hi, 1)])
    if at_limit is not None:
        out["latest_at_grid_limit"] = at_limit
    return out


def forecast(plants: dict) -> dict:
    """Return-to-full dates for plants whose last report is bloom-limited and below full."""
    pending = {n: p for n, p in plants.items() if p["obs"][-1]["limited"] and p["obs"][-1]["high"] < 1.0}
    out = {}
    for name, p in pending.items():
        entry = {}
        g = params(p["T"], PUBLISHED, GRID)
        entry["P1 plateau, published ranges, any values"] = date_range(p, full_day(plateau, *g))
        entry["P1 plateau, published ranges, at base values"] = date_range(
            p, np.array([mid_T(p) + max(BASE["hold"], BASE["recovery"])]))
        for label, (f, ranges) in STRUCTURES.items():
            g, m = fit(p, f, ranges)
            keep = m <= m.min() + TOL
            T, r, rec, hold = (x[keep] for x in g)
            # A kept fit at the top of the recovery or hold range means the latest date is cut off by the grid.
            at_limit = bool(np.any(np.isclose(hold, ranges["hold"][1]) | np.isclose(rec, ranges["recovery"][1])))
            e = dict(best_misfit=round(float(m.min()), 3), fits_kept=int(keep.sum()),
                     full_output=date_range(p, full_day(f, T, r, rec, hold), at_limit))
            if f is rising:
                frac = np.where(r < 0.95, (0.95 - r) / np.where(r < 1.0, 1.0 - r, 1.0), 0.0)
                e["output_95pct"] = date_range(p, T + rec + hold * frac, at_limit)
            entry[f"{label}, best fits"] = e
        out[name] = entry
    return out


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def freeze(plants: dict, refreeze: bool) -> bool:
    """Write the forecast unless one is already frozen. Returns True if it was written."""
    if FORECAST.exists() and not refreeze:
        return False
    cutoff = max(o["date"] for p in plants.values() for o in p["obs"])
    record = dict(
        frozen_on=dt.date.today().isoformat(),
        evidence_cutoff=str(cutoff),
        statement=(
            "Written before any report on Israeli plant output after the evidence cut-off was read. "
            "If a plant below is reported under 95% of capacity because of the bloom after the latest "
            "'P1 plateau, published ranges, any values' date, the short-bloom derate range (Z11) is too "
            "short. If it returns to full output near the 'Rising' dates and not before, the flat plateau "
            "should give way to a climb. Reports blaming another cause (as for Sorek B) do not count."),
        inputs={str(q.relative_to(ROOT)).replace("\\", "/"): sha256(q)
                for q in (DATA / "israel2026_plants.csv", DATA / "israel2026_output.csv", Path(__file__))},
        git_head=git("rev-parse", "HEAD"),
        git_dirty=bool(git("status", "--porcelain")),
        forecast=forecast(plants),
    )
    FORECAST.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return True


# --- Main --------------------------------------------------------------------


def write(name: str, rows: list[dict]) -> None:
    with open(OUT / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--refreeze", action="store_true", help="replace an existing frozen forecast")
    mode.add_argument("--compute-only", action="store_true",
                      help="write the hindcast tables only; never write the forecast (reproduction)")
    args = ap.parse_args()
    if args.compute_only and not FORECAST.exists():
        raise SystemExit(f"{FORECAST} is missing. It is committed evidence, frozen before later reports "
                         "were read; restore it from git rather than regenerating it.")
    plants = load()

    hind = []
    for name, p in plants.items():
        for o in p["obs"]:
            lo, hi, base = envelope(p, o["day"])
            hind.append(dict(plant=name, date=str(o["date"]), day=o["day"], reported_low=o["low"],
                             reported_high=o["high"], limit_cause=o["cause"], p1_low=lo, p1_high=hi,
                             p1_base=base, overlap=lo <= o["high"] and hi >= o["low"], note=o["note"]))
    write("israel2026_hindcast.csv", hind)

    fits = []
    for name, p in plants.items():
        for label, (f, ranges) in STRUCTURES.items():
            (T, r, rec, hold), m = fit(p, f, ranges)
            i = int(m.argmin())
            fits.append(dict(plant=name, structure=label, reports=len(p["limited"]), misfit=float(m[i]),
                             T=float(T[i]), restart=float(r[i]), recovery=float(rec[i]), hold=float(hold[i])))
    write("israel2026_fits.csv", fits)

    short = shortfall_rows(plants)
    write("israel2026_shortfall.csv", short)
    written = False if args.compute_only else freeze(plants, args.refreeze)

    print("Reported output against the P1 envelope (all reports except other causes):")
    for h in hind:
        if h["limit_cause"] != "other":
            flag = "overlap" if h["overlap"] else "OUTSIDE"
            print(f"  {flag} {h['plant']:<9} {h['date']} [{h['limit_cause']}]  reported "
                  f"{h['reported_low']:.2f}-{h['reported_high']:.2f}"
                  f"  P1 {h['p1_low']:.2f}-{h['p1_high']:.2f} (base {h['p1_base']:.2f})")
    print("\nSmallest misfit of a single trajectory, fraction of output:")
    for name in plants:
        row = {f["structure"]: f["misfit"] for f in fits if f["plant"] == name}
        print(f"  {name:<9} " + "  ".join(f"{k}: {v:.3f}" for k, v in row.items()))
    print("\nProduction deficit to the last report, no storage, days of D_c (reports vs P1 range):")
    for s in short:
        flag = "contained" if s["contained"] else "overlap  " if s["overlap"] else "disjoint "
        print(f"  {flag} {s['plant']:<9} D_c {s['dc_share']:.1f}  reports {s['reports_low']:.2f}-{s['reports_high']:.2f}"
              f" (line {s['reports_line']:.2f})"
              f"  P1 {s['p1_low']:.2f}-{s['p1_high']:.2f} (base {s['p1_base']:.2f})")
    frozen = json.loads(FORECAST.read_text(encoding="utf-8"))
    print(f"\nForecast {'written' if written else 'already frozen'} on {frozen['frozen_on']} "
          f"(evidence to {frozen['evidence_cutoff']}): {FORECAST}")


if __name__ == "__main__":
    main()
