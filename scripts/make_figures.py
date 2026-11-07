"""Draw the figures of the paper into figures/.

    PYTHONPATH=src python scripts/run_studies.py   # first, writes results/
    PYTHONPATH=src python scripts/make_figures.py

Text is Arial, panel letters are bold lowercase, and there are no titles inside
a figure (titles and notes are in the captions of the paper). Each figure is written as vector PDF with
embedded fonts, and as 600 dpi TIFF (LZW) and PNG.
"""

from __future__ import annotations

import csv
import os
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_studies import OPTIONS, P, SHARE  # noqa: E402

from resdesal import COSTS, required_sigma  # noqa: E402
from resdesal import economics as ec  # noqa: E402
from resdesal.architectures import HAZARDS, MAX_SHARE, build, option_sigma  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
# RESDESAL_FIG_DIR redirects the output, e.g. to try new versions without touching the manuscript figures.
FIG = Path(os.environ.get("RESDESAL_FIG_DIR", ROOT / "figures"))

MM = 1 / 25.4
WIDTH = 160 * MM  # the Word text width. Elsevier prints full width at 190 mm. Text is at least 7.5 pt here
COLUMN = 90 * MM  # Elsevier single column, printed at this size
DPI = 600

# Text is black; colour is kept for data. Colour follows the option (dataviz skill palette).
INK, INK2, MUTED = "#000000", "#404040", "#898781"
LIGHT, PALE = "#e6e6e6", "#f0efec"
COLOR = {
    "A0": MUTED,
    "A1": "#2a78d6",
    "A2": "#eb6834",
    "A3": "#1baf7a",
    "A4": "#eda100",
    "A5": "#e87ba4",
    "A1+A2": "#008300",
    "A2+A5": "#6250d6",
    # Slot 8 (red) for MSF. A6 and A7 differ only in steam supply and cost, like A4 and A5, and are
    # told apart by their row labels; A2+A7 shares the hue.
    "A6": "#e34948",
    "A7": "#e34948",
    "A2+A7": "#e34948",
}
BLUE_STEPS = ["#86b6ef", "#2a78d6", "#104281"]  # ordinal ramp, steps 250/450/650
RED_STEPS = ["#ee8786", "#e34948", "#b02524", "#6e1312"]  # ordinal ramp on the MSF hue (validated --ordinal)
SHORT = {
    "A0": "A0 Baseline SWRO",
    "A1": f"A1 DAF+UF, {round(SHARE['A1'] * 16)}/16 trains",
    "A2": f"A2 Subsurface intake, {MAX_SHARE['A2'] * 100:g}%",
    "A3": "A3 Second RO site, 50%",
    "A4": f"A4 {round(SHARE['A4'] * 600_000 / 29_630)} MED units, CCGT steam",
    "A5": f"A5 {round(SHARE['A5'] * 600_000 / 29_630)} MED units, own boiler",
    "A6": f"A6 {round(SHARE['A6'] * 600_000 / 92_600)} MSF units, CCGT steam",
    "A7": f"A7 {round(SHARE['A7'] * 600_000 / 92_600)} MSF units, own boiler",
    "A1+A2": "A1 + A2",
    "A2+A5": "A2 + A5",
    "A2+A7": "A2 + A7",
}
DC = r"$D_\mathrm{c}$"
SIGMA = r"$\sigma$"

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 8,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "axes.titlesize": 8,
    "axes.titlelocation": "left",
    "axes.titlepad": 5,
    "axes.labelsize": 8,
    "axes.labelcolor": INK,
    "axes.edgecolor": INK,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "grid.color": LIGHT,
    "grid.linewidth": 0.4,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "legend.frameon": False,
    "legend.fontsize": 7.5,
    "legend.handlelength": 1.6,
    "lines.linewidth": 1.1,
    "lines.markersize": 4,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "figure.constrained_layout.h_pad": 0.04,
    "figure.constrained_layout.w_pad": 0.04,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


def read(name: str) -> list[dict]:
    with open(RES / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def typeset(s: str) -> str:
    """Typographic fixes for text that comes from data files."""
    s = s.replace(">=", "≥").replace("<=", "≤").replace("~", "≈")
    return re.sub(r"(\d)-(\d)", "\\1–\\2", s)


def panel(ax, letter: str, header: str = "") -> None:
    """Reserve room above the axes for the panel letter and optional header (placed by finish)."""
    ax._panel = (letter, header)
    # Placeholder of the same height but not the same width: the layout engine fits the axes under their
    # title, so a long heading would make the panel narrow. finish() draws the real heading.
    ax.set_title(letter + (f" {DC}" if "$" in header else ""), fontsize=8.5, pad=9)


def finish(fig, axes) -> None:
    """Fix the layout, then put each panel letter at the left edge of its tick labels.

    Letters of panels in the same row share one baseline.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    spots = []
    for ax in axes:
        ax._left_title.set_visible(False)
        spots.append((ax, ax.get_tightbbox(r).transformed(inv)))
    fig.set_layout_engine("none")
    rows: dict[float, float] = {}
    for ax, bb in spots:
        key = round(ax.get_position().y1, 2)
        rows[key] = max(rows.get(key, 0.0), bb.y1)
    for ax, bb in spots:
        letter, header = ax._panel
        y = rows[round(ax.get_position().y1, 2)] + 0.01
        fig.text(bb.x0, y, letter, fontsize=9, fontweight="bold", va="bottom")
        if header:
            fig.text(bb.x0 + 3.0 * MM / fig.get_figwidth(), y, header, fontsize=8, va="bottom")


def save(fig, name: str) -> None:
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", metadata={"CreationDate": None, "Creator": None, "Producer": None})
    fig.savefig(FIG / f"{name}.png", dpi=DPI)
    fig.savefig(FIG / f"{name}.tiff", dpi=DPI, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def log_days_axis(ax, lo, hi, ticks=(0.5, 1, 2, 3, 7, 14, 30, 60, 120)):
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    t = [x for x in ticks if lo <= x <= hi]
    ax.set_xticks(t)
    ax.set_xticklabels([f"{x:g}" for x in t])
    ax.minorticks_off()


def range_bar(ax, lo, hi, y, color, lw=1.6):
    ax.plot([lo, hi], [y, y], color=color, lw=lw, solid_capstyle="butt")


# --- Figure 1: framework and dependency graph --------------------------------

def fig_framework() -> None:
    """Framework (a), dependency graph (b) and analysis workflow (c). Arrows run only horizontally or vertically."""
    fig = plt.figure(figsize=(WIDTH, 150 * MM))
    a = fig.add_axes([0.0, 0.30, 0.25, 0.66])
    b = fig.add_axes([0.28, 0.30, 0.72, 0.66])
    c = fig.add_axes([0.0, 0.0, 1.0, 0.22])
    for ax in (a, b, c):
        ax.set_axis_off()
        ax.set_xlim(-0.012, 1.012)  # room for box outlines at the edges
        ax.set_ylim(-0.01, 1.0)
    for x, y, letter in ((0.003, 0.985, "a"), (0.283, 0.985, "b"), (0.003, 0.245, "c")):
        fig.text(x, y, letter, fontsize=9, fontweight="bold", va="top")

    def box(ax, x, y, w, h, text, edge=MUTED, lw=0.6, fill="white", bold=False, size=7.5):
        # No padding, so the outline is exactly (x, y, w, h) and arrows meet it.
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.01",
                                    fc=fill, ec=edge, lw=lw, zorder=1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK,
                linespacing=1.15, fontweight="bold" if bold else "normal")

    def arrow(ax, p0, p1):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>,head_length=0.45,head_width=0.22", mutation_scale=10,
                                     color=INK2, lw=0.75, shrinkA=0, shrinkB=0, joinstyle="miter", zorder=3))

    def line(ax, *pts):
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=INK2, lw=0.75, solid_capstyle="projecting", solid_joinstyle="miter", zorder=3)

    # (a) Three levels of protection, from the weakest (bottom) to the target (top).
    levels = [
        ("Service continuity", f"{DC} met from storage and\nsurviving production", 0.68),
        ("Plant continuity", "Some safe production\ncontinues", 0.37),
        ("Equipment protection", "Shut down to protect\nthe membranes", 0.06),
    ]
    for name, text, y in levels:
        a.add_patch(FancyBboxPatch((0.04, y), 0.92, 0.26, boxstyle="round,pad=0,rounding_size=0.02",
                                   fc=PALE, ec=MUTED, lw=0.6, zorder=1))
        a.text(0.09, y + 0.185, name, fontweight="bold", color=INK, va="center", fontsize=8)
        a.text(0.09, y + 0.075, text, color=INK, va="center", fontsize=7.5, linespacing=1.2)
    for y0, y1 in ((0.32, 0.37), (0.63, 0.68)):
        arrow(a, (0.5, y0), (0.5, y1))

    # (b) Dependency graph in five columns: intake, pretreatment, production, storage, demand.
    w, h = 0.175, 0.115
    col = {"intake": 0.0, "pre": 0.2075, "prod": 0.415, "store": 0.66, "demand": 0.85}
    wc = {"intake": w, "pre": w, "prod": 0.19, "store": 0.15, "demand": 0.15}
    row = {"dmf": 0.82, "daf": 0.66, "thermal": 0.50, "steam": 0.335, "sub": 0.17, "siteb": 0.0}
    cy = {k: v + h / 2 for k, v in row.items()}
    # Open intake serves the first three rows; the protected intakes their own row.
    box(b, col["intake"], row["thermal"], w, row["dmf"] + h - row["thermal"], "Open intake")
    box(b, col["intake"], row["sub"], w, h, "Subsurface\nintake (A2)", COLOR["A2"], 1.3)
    box(b, col["intake"], row["siteb"], w, h, "Site B intake\n+ DMF (A3)", COLOR["A3"], 1.3)
    box(b, col["pre"], row["dmf"], w, h, "DMF")
    box(b, col["pre"], row["daf"], w, h, "DAF + UF (A1)", COLOR["A1"], 1.3)
    box(b, col["pre"], row["thermal"], w, h, "Thermal feed")
    box(b, col["pre"], row["steam"], w, h, "CCGT steam or\nboiler (A4–A7)", COLOR["A4"], 1.3)
    box(b, col["prod"], row["daf"], wc["prod"], row["dmf"] + h - row["daf"], "RO trains")
    box(b, col["prod"], row["thermal"], wc["prod"], h, "MED (A4, A5)\nor MSF (A6, A7)", COLOR["A5"], 1.3)
    box(b, col["prod"], row["sub"], wc["prod"], h, "RO trains (A2)")
    box(b, col["prod"], row["siteb"], wc["prod"], h, "RO trains (A3)")
    box(b, col["store"], row["daf"], wc["store"], row["dmf"] + h - row["daf"], "Product\nstorage")
    box(b, col["demand"], row["daf"], wc["demand"], row["dmf"] + h - row["daf"], f"Critical\ndemand {DC}")
    box(b, col["store"], row["sub"], 1.0 - col["store"], h,
        "Grid power for every\nRO and thermal pump")

    right = {k: v + wc[k] for k, v in col.items()}
    for r in ("dmf", "daf", "thermal"):  # open intake to pretreatment
        arrow(b, (right["intake"], cy[r]), (col["pre"], cy[r]))
    for r in ("dmf", "daf"):  # pretreatment to the RO trains
        arrow(b, (right["pre"], cy[r]), (col["prod"], cy[r]))
    arrow(b, (right["pre"], cy["thermal"]), (col["prod"], cy["thermal"]))
    # Steam: right, then up into the thermal units.
    x_up = col["prod"] + wc["prod"] / 2
    line(b, (right["pre"], cy["steam"]), (x_up, cy["steam"]))
    arrow(b, (x_up, cy["steam"]), (x_up, row["thermal"]))
    for r in ("sub", "siteb"):  # protected intakes straight to their RO trains
        arrow(b, (right["intake"], cy[r]), (col["prod"], cy[r]))
    # Every production row joins a collector, which feeds the storage.
    x_col = right["prod"] + 0.027
    rows_in = ["daf", "thermal", "sub", "siteb"]
    y_ro = (row["daf"] + row["dmf"] + h) / 2
    for y in [y_ro] + [cy[r] for r in ("thermal", "sub", "siteb")]:
        line(b, (right["prod"], y), (x_col, y))
    line(b, (x_col, cy["siteb"]), (x_col, y_ro))
    arrow(b, (x_col, y_ro), (col["store"], y_ro))
    arrow(b, (right["store"], y_ro), (col["demand"], y_ro))

    def hazard(x, y, label):
        b.text(x, y, f"× {label}", fontsize=7.5, color="#c62828", fontweight="bold", va="bottom")

    hazard(col["pre"] + 0.004, row["dmf"] + h + 0.012, "Bloom")
    hazard(col["intake"] + 0.004, row["dmf"] + h + 0.012, "Oil")
    hazard(col["pre"] + 0.004, row["steam"] + h + 0.008, "CCGT trip")
    hazard(col["store"] + 0.004, row["sub"] + h + 0.012, "Blackout")

    # (c) Analysis workflow, with the section that describes each step.
    steps = [
        ("Framework", "service and\nplant continuity", "§2.1"),
        ("Plant, hazards", "reference plant,\nresponse factors", "§2.2–2.3"),
        ("Water balance", "shortfall,\nunserved water", "§2.4"),
        ("Options, costs", "surviving capacity,\nbreak-even length", "§2.5"),
        ("Design choice", "Monte Carlo\ncoverage, cost", "§2.6"),
        ("Tests", "assumptions,\nMSF, verification", "§2.6–2.7"),
    ]
    n, gap = len(steps), 0.025
    bw = (1.0 - 0.01 - gap * (n - 1)) / n
    for i, (head, text, sec) in enumerate(steps):
        x = 0.005 + i * (bw + gap)
        c.add_patch(FancyBboxPatch((x, 0.18), bw, 0.62, boxstyle="round,pad=0,rounding_size=0.01",
                                   fc=PALE, ec=MUTED, lw=0.6, zorder=1))
        c.text(x + bw / 2, 0.64, head, ha="center", va="center", fontsize=7.5, fontweight="bold", color=INK)
        c.text(x + bw / 2, 0.40, text, ha="center", va="center", fontsize=7.5, color=INK, linespacing=1.15)
        c.text(x + bw / 2, 0.04, sec, ha="center", va="center", fontsize=7.5, color=INK2)
        if i < n - 1:
            arrow(c, (x + bw, 0.49), (x + bw + gap, 0.49))
    save(fig, "fig1_framework")


# --- Figure 2: documented outage durations -----------------------------------

def cite_text(keys: list[str], numbers: dict[str, int]) -> str:
    """Reference numbers as the manuscript prints them (wordkit.Doc.cite): ' [3,7–9]'."""
    missing = [k for k in keys if k not in numbers]
    if missing:
        raise KeyError(f"Fig. 2 references not cited in the manuscript: {missing}")
    nums = sorted({numbers[k] for k in keys})
    groups, start = [], nums[0]
    for a, b in zip(nums, nums[1:] + [None]):
        if b != a + 1:
            groups.append(str(start) if start == a else (f"{start},{a}" if a == start + 1 else f"{start}–{a}"))
            start = b
    return " [" + ",".join(groups) + "]"


def fig_events() -> None:
    path = ROOT / "data" / "events.csv"
    if not path.exists():
        print("Fig. 2 skipped: data/events.csv not found")
        return
    with open(path, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("plot", "1") == "1"]
    # Reference numbers come from the build of the paper, which is not part of this repository, so here the
    # labels are drawn without them.
    numbers_path = ROOT / "manuscript" / "ref_numbers.json"
    numbers = json.loads(numbers_path.read_text(encoding="utf-8")) if numbers_path.exists() else None
    if numbers is None:
        print("Fig. 2: manuscript/ref_numbers.json not found, labels drawn without reference numbers")
    kinds = {"bloom": "#2a78d6", "derate": "#1baf7a", "oil": "#eb6834", "presence": MUTED}
    kind_label = {"bloom": "Bloom: full outage", "derate": "Bloom: derated operation", "oil": "Oil: intake closure",
                  "presence": "Bloom present in region"}
    fig, ax = plt.subplots(figsize=(WIDTH, (5.6 * len(rows) + 34) * MM), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.25)  # the notes past the last bar need a margin at the right edge
    for i, r in enumerate(rows):
        lo, hi = float(r["days_low"]), float(r["days_high"])
        c = kinds.get(r["kind"], MUTED)
        face = c if r.get("level", "") == "read" else "white"
        if hi > lo:
            range_bar(ax, lo, hi, i, c, lw=1.4)
        for x in ({lo, hi}):
            ax.plot([x], [i], "o", ms=3.8, color=face, mec=c, mew=0.9, zorder=3)
        ax.text(hi * 1.18, i, typeset(r["note"]), va="center", fontsize=7.5, color=INK2, zorder=4,
                bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([typeset(r["label"]) + (cite_text(r["refs"].split(";"), numbers) if numbers else "")
                        for r in rows])
    ax.tick_params(axis="y", length=0)
    ax.invert_yaxis()
    log_days_axis(ax, 0.1, 3000, ticks=(0.1, 0.25, 1, 3, 7, 14, 30, 120))
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    b0 = P.storage / P.critical_demand
    ax.axvline(b0, color=INK, lw=0.8, ls=(0, (4, 2)))
    ax.set_ylim(len(rows) - 0.5, -1.3)
    ax.text(b0 * 1.06, -0.8, f"GCRP-600 storage, {b0:g} days of {DC}", fontsize=7.5, color=INK, va="center")
    ax.set_xlabel("Outage or closure duration (days, log scale)")
    handles = [plt.Line2D([], [], color=c, marker="o", ms=3.8, lw=1.4, label=kind_label[k])
               for k, c in kinds.items() if any(r["kind"] == k for r in rows)]
    handles.append(plt.Line2D([], [], color=INK2, marker="o", ms=3.8, mfc="white", mew=0.9, lw=0,
                              label="Open marker: second-hand or estimated"))
    fig.legend(handles=handles, loc="outside lower center", ncol=3, columnspacing=1.5)
    save(fig, "fig2_events")


# --- Figure 4: surviving production needed and what each option supplies -----

def fig_design_map() -> None:
    t3 = {r["option"]: r for r in read("table3_sigma.csv")}
    b0 = P.storage / P.critical_demand
    fig = plt.figure(figsize=(WIDTH, 150 * MM), layout="constrained")
    gs = fig.add_gridspec(2, 1, height_ratios=[0.9, 1.0])
    low = gs[1].subgridspec(1, 2, width_ratios=[1.0, 1.0])  # own grid: margins independent of row a
    a, pb, pc = fig.add_subplot(gs[0]), fig.add_subplot(low[0]), fig.add_subplot(low[1])

    b = np.geomspace(0.3, 60, 300)
    a.fill_between(b, np.clip(1 - b / 7, 0, None), np.clip(1 - b / 120, 0, None), color="#cde2fb", lw=0,
                   label="Severe bloom, 7–120 d")
    a.fill_between(b, np.clip(1 - b / 2, 0, None), np.clip(1 - b / 12, 0, None), color="#86b6ef", lw=0,
                   label="Short bloom, 2–12 d")
    for T, y_lab in ((1, 0.3), (3, 0.45), (7, 0.3), (12, 0.45), (55, 0.6), (120, 0.75)):
        a.plot(b, np.clip(1 - b / T, 0, None), color=INK2, lw=0.7)
        a.text(T * (1 - y_lab), y_lab, f"{T} d", fontsize=7.5, color=INK, ha="center", va="center",
               bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
    a.axvline(b0, color=INK, lw=0.8, ls=(0, (4, 2)))
    a.text(b0 * 1.06, 1.055, f"GCRP-600 storage, {b0:g} d", fontsize=7.5, color=INK, va="center")
    log_days_axis(a, 0.3, 60, ticks=(0.5, 1, 2.5, 7, 14, 30, 60))
    a.set_ylim(0, 1.1)
    a.set_yticks(np.arange(0, 1.01, 0.2))
    a.set_xlabel(f"Storage at onset (days of {DC})")
    a.set_ylabel(f"Surviving production needed, {SIGMA} = $Q$/{DC}")
    bands = a.get_legend_handles_labels()[0]  # the two bloom bands, shown in the key below
    panel(a, "a")

    keys = list(OPTIONS)
    for ax, hz, letter, header, needs in [
        (pb, "severe_bloom", "b", "Severe bloom", [(12, "12 d", "right"), (55, "55 d", "left")]),
        (pc, "oil", "c", "Oil at intake", [(3, "3 d", "left"), (30, "30 d", "left")]),
    ]:
        for i, k in enumerate(keys):
            r = t3[k]
            lo, base, hi = (float(r[f"{hz}_{w}"]) for w in ("low", "base", "high"))
            if r.get(f"{hz}_binary") == "True":
                # Second site: two physical states (hit, escaped), not a range; base is only an expectation.
                ax.plot([lo, hi], [i, i], color=COLOR[k], lw=0.6, ls=(0, (1, 1.5)))
                ax.plot([lo, hi], [i, i], "D", ms=3.8, mfc="white", mec=COLOR[k], mew=1.0, zorder=3)
                continue
            range_bar(ax, lo, hi, i, COLOR[k])
            ax.plot([base], [i], "o", ms=4.2, color=COLOR[k], mec="white", mew=0.6, zorder=3)
        for T, lab, ha in needs:
            s = required_sigma(T, P.storage, P.critical_demand)
            ax.axvline(s, color=INK, lw=0.6, ls=(0, (4, 2)))
            ax.text(s + (0.05 if ha == "left" else -0.05), -1.0, lab, fontsize=7.5, color=INK, ha=ha, va="center")
        ax.set_yticks(range(len(keys)))
        ax.set_yticklabels([SHORT[k] for k in keys] if ax is pb else [])
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(len(keys) - 0.4, -1.6)
        ax.set_xlim(0, 2.2)
        ax.set_xticks([0, 1, 2])
        ax.set_xlabel(f"{SIGMA} of option")
        panel(ax, letter, header)
    # Key for the markers, so the caption need not explain them.
    fig.legend(handles=[
        *bands,
        plt.Line2D([], [], color=INK2, marker="o", ms=4.2, mec="white", lw=0, label="Base value"),
        plt.Line2D([], [], color=INK2, lw=1.6, label="Sampled range"),
        plt.Line2D([], [], color=INK2, marker="D", ms=3.8, mfc="white", lw=0.6, ls=(0, (1, 1.5)),
                   label="A3: site reached or missed"),
        plt.Line2D([], [], color=INK, lw=0.6, ls=(0, (4, 2)), label=f"{SIGMA} needed, 2.5 d storage"),
    ], loc="outside lower center", ncol=3, fontsize=7.5)
    panel(a, "a", "Surviving production needed")
    finish(fig, (a, pb, pc))
    save(fig, "fig4_design_map")


# --- Figure 5: cost break-even -----------------------------------------------

def fig_breakeven() -> None:
    rows = read("costs_breakeven.csv")
    measure_color = [r["measure"][:2] if r["measure"][:2] in COLOR else "A5" for r in rows]  # standby MED: A5
    fig = plt.figure(figsize=(COLUMN, 150 * MM), layout="constrained")  # sized for one column
    gs = fig.add_gridspec(2, 1, height_ratios=[1.35, 1.0])
    a, b = fig.add_subplot(gs[0].subgridspec(1, 1)[0]), fig.add_subplot(gs[1].subgridspec(1, 1)[0])

    bands = [("Oil, precautionary closure", 0.125, 3), ("Oil, fouled intake", 14, 90), ("Short bloom", 2, 12),
             ("Severe bloom", 7, 120)]
    band_y = [-1.2 - 0.6 * i for i in range(len(bands))]
    for (lab, lo, hi), y in zip(bands, band_y):
        range_bar(a, lo, hi, y, MUTED, lw=2.4)
    for i, (r, k) in enumerate(zip(rows, measure_color)):
        cap, lc = float(r["tstar_capex_d"]), float(r["tstar_lifecycle_d"])
        a.plot([cap, lc], [i, i], color=COLOR[k], lw=0.9)
        a.plot([cap], [i], "o", ms=4.2, mfc="white", mec=COLOR[k], mew=1.0, zorder=3)
        a.plot([lc], [i], "o", ms=4.2, color=COLOR[k], mec=COLOR[k], mew=1.0, zorder=3)
        if r["measure"].startswith("A2"):
            # Capped (E14), the subsurface intake alone cannot cover long outages.
            sig = option_sigma(build("A2", MAX_SHARE["A2"]), HAZARDS["severe_bloom"], P.critical_demand)
            a.text(max(cap, lc) * 1.6, i, f"alone covers ≤{P.storage / P.critical_demand / (1 - sig):.1f} d",
                   fontsize=7.5, color=INK2, va="center")
    # The hazard bands are named in the row labels, so nothing needs to sit to the right of the bars.
    a.set_yticks(band_y + list(range(len(rows))))
    a.set_yticklabels([lab for lab, *_ in bands] +
                      [r["measure"].replace(" (added)", ", added") + (" (expected)" if r.get("basis") == "expected" else "")
                       for r in rows])
    for t in a.get_yticklabels()[:len(bands)]:
        t.set_color(INK2)
    a.tick_params(axis="y", length=0)
    a.set_ylim(len(rows) - 0.5, -3.4)
    log_days_axis(a, 0.1, 150, ticks=(0.1, 0.25, 1, 2, 4, 7, 14, 30, 60, 120))
    a.grid(axis="x")
    a.set_axisbelow(True)
    a.set_xlabel("Break-even length $T$* (days, log scale)")
    a.plot([], [], "o", ms=4.2, mfc="white", mec=INK2, mew=1.0, label="Capital cost")
    a.plot([], [], "o", ms=4.2, color=INK2, label="Lifecycle cost")
    a.legend(loc="center right", bbox_to_anchor=(1.0, -0.6), bbox_transform=a.get_yaxis_transform(),
             handletextpad=0.3, borderaxespad=0.2)  # in the gap between the hazard bands and A1
    panel(a, "a", "Break-even event length of each measure")

    T = np.geomspace(1, 120, 200)
    s0, dc = P.storage, P.critical_demand
    storage = COSTS.storage_capex * np.maximum(dc * T - s0, 0) / 1e9
    sig = np.array([required_sigma(t, s0, dc) for t in T])
    measures = {m.label: m for m in ec.measures()}
    b.plot(T, storage, color=MUTED, label="More storage only")
    for label, k in [("A1 DAF+UF retrofit", "A1"), ("A5 Hybrid MED, own boiler", "A5"),
                     ("A7 Hybrid MSF, own boiler", "A7")]:
        cost = sig * dc * measures[label].capex_per_surviving() / 1e9
        b.plot(T, cost, color=COLOR[k], label=label)
    b.set_xscale("log")
    b.set_yscale("log")
    b.set_xlim(1, 120)
    b.set_ylim(0.01, 40)  # headroom holds the legend clear of the curves
    b.set_xticks([1, 3, 7, 14, 30, 60, 120])
    b.set_xticklabels(["1", "3", "7", "14", "30", "60", "120"])
    b.set_yticks([0.01, 0.1, 1, 10])
    b.set_yticklabels(["0.01", "0.1", "1", "10"])
    b.minorticks_off()
    b.set_xlabel("Outage length $T$ (days, log scale)")
    b.set_ylabel("Capital cost (USD billion)")
    b.legend(loc="upper left", borderaxespad=0.2)
    panel(b, "b", "Capital cost of covering an outage")
    finish(fig, (a, b))
    save(fig, "fig5_breakeven")


# --- Figure 6: Monte Carlo ---------------------------------------------------

def fig_monte_carlo() -> None:
    mc = read("mc_summary.csv")
    hazards = ["severe_bloom", "short_bloom", "oil", "oil_fouled", "ccgt_trip", "grid_blackout"]
    hz_label = ["Severe\nbloom", "Short\nbloom", "Oil,\nprecaut.", "Oil,\nfouled", "CCGT\ntrip", "Grid\nblackout"]
    keys = list(OPTIONS)
    grid = np.array([[float(next(r for r in mc if r["hazard"] == h and r["option"] == k)["p_met"])
                      for h in hazards] for k in keys])

    fig = plt.figure(figsize=(WIDTH, 205 * MM), layout="constrained")
    gs = fig.add_gridspec(3, 1, height_ratios=[1.05, 1.15, 0.62])
    mid = gs[1].subgridspec(1, 2)  # own grid: margins independent of rows a and d
    a = fig.add_subplot(gs[0])
    cmap = LinearSegmentedColormap.from_list("blue", [PALE, "#cde2fb", "#86b6ef", "#2a78d6", "#104281"])
    im = a.imshow(grid, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(len(keys)):
        for j in range(len(hazards)):
            v = grid[i, j]
            a.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5, color="white" if v > 0.6 else INK)
    a.set_xticks(range(len(hazards)))
    a.set_xticklabels(hz_label)
    a.xaxis.tick_top()
    a.set_yticks(range(len(keys)))
    a.set_yticklabels([SHORT[k] for k in keys])
    a.tick_params(length=0)
    for s in a.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=a, fraction=0.05, pad=0.02, aspect=25)
    cb.set_label(f"Coverage of {DC} (share of scenarios)")
    cb.outline.set_linewidth(0.6)
    cb.ax.tick_params(width=0.6, length=2.5)
    panel(a, "a", "Coverage by option and hazard")

    b = fig.add_subplot(mid[0])
    sweep = read("storage_sweep.csv")
    for k in ["A0", "A1", "A2", "A3", "A5", "A7", "A1+A2"]:
        pts = [(float(r["storage_days"]), float(r["p_met"])) for r in sweep
               if r["hazard"] == "severe_bloom" and r["option"] == k]
        x, y = zip(*pts)
        b.plot(x, y, color=COLOR[k], label=k)
    b.plot([P.storage / P.critical_demand] * 2, [-0.02, 1.02], color=INK, lw=0.6, ls=(0, (4, 2)))
    log_days_axis(b, 0.5, 40, ticks=(0.5, 1, 2.5, 7, 14, 40))
    b.set_ylim(-0.02, 1.3)  # headroom above 1 holds the legend clear of the curves
    b.set_yticks(np.arange(0, 1.01, 0.2))
    b.set_xlabel(f"Storage (days of {DC}, log scale)")
    b.set_ylabel(f"Coverage of {DC}, severe bloom")
    b.legend(loc="upper center", ncol=4, handlelength=1.2, columnspacing=1.0, borderaxespad=0.2)
    panel(b, "b", "Severe bloom: coverage against storage")

    c = fig.add_subplot(mid[1])
    thr = read("pretreatment_threshold.csv")
    z2 = HAZARDS["severe_bloom"].factors["daf_uf"]
    c.axhspan(z2.low, z2.high, color=PALE, lw=0)
    c.text(0.205, 0.975, f"Sampled DAF+UF output kept,\nsevere bloom ({z2.low:g}–{z2.high:g})", fontsize=7.5,
           color=INK, va="top")
    for T, col in zip(("12.0", "55.0", "120.0"), BLUE_STEPS):
        pts = [(float(r["dc_share"]), float(r["f_star"])) for r in thr if r["duration_d"] == T]
        x, y = zip(*pts)
        c.plot(x, y, color=col, label=f"{float(T):g} d")
    z3 = HAZARDS["short_bloom"].factors["daf_uf"]
    c.errorbar([0.4], [z3.base], yerr=[[z3.base - z3.low], [z3.high - z3.base]], fmt="o", color=INK, ms=3.5,
               capsize=2.5, lw=0.8, capthick=0.8, zorder=4)
    c.text(0.41, z3.high, f"Picoplankton bloom,\nassumed {z3.low:g}–{z3.high:g},\nbase {z3.base:g}", fontsize=7.5,
           color=INK, va="top")
    c.set_xlim(0.2, 0.6)
    c.set_xticks(np.arange(0.2, 0.61, 0.1))
    c.set_ylim(0, 1)
    c.set_xlabel(f"Critical demand {DC} (share of plant capacity)")
    c.set_ylabel("Minimum DAF+UF output kept, $f$*")
    c.legend(loc="lower right", handlelength=1.0, borderaxespad=0.2, title="Outage",
             title_fontsize=7.5)  # below the lines, right of the error bar
    panel(c, "c", "Least DAF + UF output, strengthened RO alone")

    d = fig.add_subplot(gs[2])
    names_map = {"duration": "Outage length", "recovery": "Recovery time", "fill": "Tank level at onset",
                 "daf_uf": "DAF+UF output kept", "med_process": "MED output kept", "msf_process": "MSF output kept"}
    st = {k: {r["input"]: float(r["ST"]) for r in read(f"sobol_{k}_severe_bloom.csv")} for k in ("A1", "A5", "A7")}
    names = [n for n in names_map if max(s.get(n, 0) for s in st.values()) > 0.01]
    y = np.arange(len(names))
    for off, k in zip((-0.27, 0.0, 0.27), st):
        d.barh(y + off, [max(st[k].get(n, 0), 0) for n in names], height=0.25, color=COLOR[k], label=k)
    d.set_yticks(y)
    d.set_yticklabels([names_map[n] for n in names])
    d.tick_params(axis="y", length=0)
    d.invert_yaxis()
    d.set_xlim(0, 1)
    d.set_xlabel("Total-order Sobol index of unserved water")
    d.legend(loc="center right", borderaxespad=0.3)
    panel(d, "d", "Sensitivity of unserved water, severe bloom")
    finish(fig, (a, b, c, d))
    save(fig, "fig6_monte_carlo")


# --- Figure 7: equal-service design fronts -----------------------------------

def fig_design_fronts() -> None:
    """Minimum extra cost of a complete design that reaches each coverage target (scripts/design_study.py)."""
    rows = read("design_fronts.csv")
    # The MSF line is the cheapest design containing MSF at each target: MSF alone or with MED, on CCGT steam or
    # own boilers. Each point is one design, so its marker shows which kind it is (open: MSF alone, filled: with MED).
    msf_alone = ("With MSF on CCGT steam", "With MSF and own boiler")
    families = [("Strengthened RO and storage", ("Strengthened RO and storage",), COLOR["A1"], "-", "o"),
                ("With MED on CCGT steam", ("With MED on CCGT steam",), COLOR["A4"], "-", "o"),
                ("With MED and own boiler", ("With MED and own boiler",), COLOR["A5"], "-", "o"),
                ("Cheapest design with MSF",
                 msf_alone + ("With MED and MSF on CCGT steam", "With MED and MSF and own boiler"), COLOR["A6"], "-",
                 "s")]
    targets = sorted({float(r["target"]) for r in rows})
    xs = np.arange(len(targets))
    fig = plt.figure(figsize=(WIDTH, 140 * MM), layout="constrained")
    gs = fig.add_gridspec(2, 2)
    axes = []
    for i, (dc, dc_label) in enumerate(((0.4, "40%"), (0.5, "50%"))):
        for j, (col, label) in enumerate((("min_capital_musd", "Extra capital cost (USD million)"),
                                          ("min_lifecycle_musd", "Extra lifecycle cost (USD million)"))):
            ax = fig.add_subplot(gs[i, j])
            for fam, members, color, ls, marker in families:
                best: dict[int, tuple[float, str]] = {}
                for r in rows:
                    if float(r["dc_share"]) == dc and r["family"] in members and r.get(col):
                        x = targets.index(float(r["target"]))
                        if float(r[col]) < best.get(x, (np.inf, ""))[0]:
                            best[x] = (float(r[col]), r["family"])
                if not best:
                    continue
                x, y = zip(*((k, v[0]) for k, v in sorted(best.items())))
                if marker == "o":
                    ax.plot(x, y, color=color, ls=ls, marker=marker, ms=3.2, mew=0.9, label=fam)
                    continue
                ax.plot(x, y, color=color, ls=ls, label=fam)
                for k, (v, f) in best.items():
                    ax.plot([k], [v], marker=marker, ms=3.4, color=color, mew=0.9,
                            mfc="white" if f in msf_alone else color, ls="none", zorder=3)
            ax.set_xticks(xs)
            ax.set_xticklabels([f"{100 * t:g}" for t in targets])
            ax.set_xlabel("Coverage target (% of scenarios)")
            ax.set_ylabel(label)
            ax.set_ylim(bottom=0)
            ax.grid(axis="y")
            ax.set_axisbelow(True)
            if i == 0 and j == 0:
                handles, labels = ax.get_legend_handles_labels()
                for fill, text in (("white", "MSF alone"), (COLOR["A6"], "MSF with MED")):
                    handles.append(plt.Line2D([], [], color=COLOR["A6"], ls="none", marker="s", ms=3.4, mew=0.9,
                                              mfc=fill))
                    labels.append(f"   {text}")
                ax.legend(handles, labels, loc="upper left", borderaxespad=0.2)
            panel(ax, "abcd"[2 * i + j], f"{DC} = {dc_label} of capacity")
            axes.append(ax)
    finish(fig, axes)
    save(fig, "fig7_design_fronts")


# --- Figure 8: joint survival map ------------------------------------------

def fig_survival_map() -> None:
    """Joint survival map: cost of the cheapest design with MED minus without (scripts/assumption_study.py)."""
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch, Rectangle

    rows = read("survival_map.csv")
    grid = sorted({float(r["f_daf_uf"]) for r in rows})
    step = grid[1] - grid[0]
    edges = np.r_[np.array(grid) - step / 2, grid[-1] + step / 2]
    levels = [-300, -100, -50, -20, 0, 20, 50, 100, 300]
    # Diverging: orange where designs with MED are cheaper, blue where designs without MED are cheaper.
    cmap = ListedColormap(["#8a5a00", "#c98500", "#eda100", "#f7d58a", "#bcd6f5", "#86b6ef", "#2a78d6", "#104281"])
    cmap.set_over("#104281")
    cmap.set_under("#8a5a00")
    norm = BoundaryNorm(levels, cmap.N)
    only_med, only_ro, none = "#595959", "#c3c2b7", "#ffffff"
    fig = plt.figure(figsize=(WIDTH, 126 * MM), layout="constrained")
    gs = fig.add_gridspec(2, 3)
    axes = []
    for i, (basis, name) in enumerate((("capital", "capital"), ("lifecycle", "lifecycle"))):
        for j, dc in enumerate((0.4, 0.5, 0.6)):
            ax = fig.add_subplot(gs[i, j])
            z = np.full((len(grid), len(grid)), np.nan)
            only = np.zeros_like(z, dtype=bool)
            only_no_thermal = np.zeros_like(z, dtype=bool)  # not the same as "no design"
            msf_pts = []
            for r in rows:
                if float(r["dc_share"]) != dc:
                    continue
                a, b = grid.index(float(r["f_med"])), grid.index(float(r["f_daf_uf"]))

                def cost(fam):
                    v = r.get(f"{fam}_{basis}_musd", "")
                    return float(v) if v not in ("", "nan") else np.nan

                ro = cost("ro")
                # Thermal: the cheapest of MED, MSF and both; MSF keeps what MED keeps (linked draw).
                thermal = {f: cost(f) for f in ("med", "msf", "med_msf")}
                best = min((v, f) for f, v in thermal.items() if not np.isnan(v)) if any(
                    not np.isnan(v) for v in thermal.values()) else (np.nan, "")
                med = best[0]
                if not np.isnan(med) and best[1] != "med" and (np.isnan(ro) or med < ro):
                    msf_pts.append((float(r["f_daf_uf"]), float(r["f_med"])))
                if np.isnan(ro) and not np.isnan(med):
                    only[a, b] = True
                elif np.isnan(med) and not np.isnan(ro):
                    only_no_thermal[a, b] = True
                else:
                    z[a, b] = med - ro
            ax.pcolormesh(edges, edges, z, cmap=cmap, norm=norm, shading="flat", rasterized=True)
            ax.pcolormesh(edges, edges, np.where(only, 1.0, np.nan), cmap=ListedColormap([only_med]),
                          shading="flat", rasterized=True)
            ax.pcolormesh(edges, edges, np.where(only_no_thermal, 1.0, np.nan), cmap=ListedColormap([only_ro]),
                          shading="flat", rasterized=True)
            ax.set_facecolor(none)
            if np.isfinite(z).sum() > 3:
                ax.contour(grid, grid, np.ma.masked_invalid(z), levels=[0], colors=INK, linewidths=0.9)
            ax.add_patch(Rectangle((0.1, 0.8), 0.7, 0.2, fill=False, ec="white", lw=1.6, zorder=3))
            ax.add_patch(Rectangle((0.1, 0.8), 0.7, 0.2, fill=False, ec=INK, lw=0.8, ls=(0, (3, 2)), zorder=4))
            ax.plot(0.4, 1.0, "o", ms=4, mfc="white", mec=INK, mew=0.8, zorder=5, clip_on=False)
            if msf_pts:
                ax.plot(*zip(*msf_pts), "o", ms=1.6, color="white", mec=INK, mew=0.4, zorder=5)
            ax.set_xlim(edges[0], edges[-1])
            ax.set_ylim(edges[0], edges[-1])
            ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_aspect("equal")
            for s in ("top", "right"):
                ax.spines[s].set_visible(True)
            if i == 1:
                ax.set_xlabel("Output kept by DAF + UF")
            if j == 0:
                ax.set_ylabel("Output kept by MED and MSF")
            panel(ax, "abcdef"[3 * i + j], f"{DC} = {round(dc * 100)}%, {name} cost")
            axes.append(ax)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=axes, location="right", shrink=0.8, aspect=30, extend="both",
                      spacing="uniform", ticks=levels)
    cb.set_label("Cheapest design with thermal units minus cheapest without (USD million)")
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(labelsize=7.5)
    handles = [Patch(fc=only_med, label="Only designs with thermal units meet the target"),
               Patch(fc=only_ro, label="Only designs without thermal units meet the target"),
               Patch(fc=none, ec=MUTED, lw=0.5, label="No design meets the target"),
               plt.Line2D([], [], color=INK, lw=0.9, label="Equal cost"),
               Patch(fc="none", ec=INK, ls=(0, (3, 2)), lw=0.8, label="Sampled range"),
               plt.Line2D([], [], marker="o", ls="", ms=4, mfc="white", mec=INK, label="Base values"),
               plt.Line2D([], [], marker="o", ls="", ms=2.2, mfc="white", mec=INK, mew=0.5,
                          label="Cheapest design includes MSF")]
    fig.legend(handles=handles, loc="outside lower center", ncol=3)
    finish(fig, axes)
    save(fig, "fig8_survival_map")


# --- Figure 9: when MSF enters the cheapest design ---------------------------

def fig_msf_threshold() -> None:
    """MSF entry threshold with DAF + UF sampled (scripts/msf_threshold.py). The fixed-response grid is Fig. S2 (figS2)."""
    from matplotlib.patches import Rectangle

    thr = read("msf_threshold.csv")
    fig = plt.figure(figsize=(WIDTH, 84 * MM), layout="constrained")
    gs = fig.add_gridspec(1, 4)
    axes = []

    blooms = (("short_bloom", "Short bloom", "#2a78d6", "o"), ("severe_bloom", "Severe bloom", "#104281", "s"))
    for j, dc in enumerate((0.4, 0.5)):
        ax = fig.add_subplot(gs[0, 2 * j:2 * j + 2])
        ax.plot([0, 1], [0, 1], color="black", lw=0.8, ls=(0, (1, 1.5)))
        ax.text(0.47, 0.42, "MSF keeps as much as MED", fontsize=7.5, color=INK2, ha="left", va="top")
        ax.add_patch(Rectangle((0.8, 0.8), 0.2, 0.2, fill=False, ec=INK, lw=0.8, ls=(0, (3, 2)), zorder=4))
        ax.text(0.81, 0.785, "Sampled\nranges", fontsize=7.5, color=INK, ha="left", va="top")
        for h, label, color, marker in blooms:
            pts = sorted((float(r["f_med"]), float(r["f_msf_threshold"])) for r in thr
                         if float(r["dc_share"]) == dc and r["hazard"] == h and r["basis"] == "capital_musd"
                         and r.get("f_msf_threshold"))
            if pts:
                ax.plot(*zip(*pts), color=color, marker=marker, ms=3.4, mfc="white", mew=0.9, label=label)
            else:
                ax.plot([], [], color=color, marker=marker, ms=3.4, mfc="white", mew=0.9,
                        label=f"{label}: never cheapest")
        never_life = all(not r.get("f_msf_threshold") for r in thr
                         if float(r["dc_share"]) == dc and r["basis"] == "lifecycle_musd")
        if never_life:
            ax.text(0.97, 0.03, "On lifecycle cost, no MSF share\nin this experiment makes a\ndesign with MSF cheapest",
                    fontsize=7.5, color=INK, ha="right", va="bottom")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks(np.arange(0, 1.01, 0.2))
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.grid(True)
        ax.set_axisbelow(True)
        ax.set_xlabel("Output kept by MED in the bloom")
        ax.set_ylabel("Least output MSF must keep")
        ax.legend(loc="upper left", borderaxespad=0.3)
        panel(ax, "ab"[j], f"{DC} = {round(dc * 100)}%, capital cost, DAF + UF sampled")
        axes.append(ax)
    finish(fig, axes)
    save(fig, "fig9_msf_threshold")


def figS2() -> None:
    """Fig. S2: fixed-response grid over DAF + UF, MED and MSF (scripts/assumption_study.py)."""
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    grid_rows = read("response_grid.csv")
    fig = plt.figure(figsize=(WIDTH, 175 * MM), layout="constrained")
    gs = fig.add_gridspec(3, 4)
    axes = []
    levels = [0.0, 0.25, 0.5, 0.75]
    cmap = ListedColormap(RED_STEPS)
    norm = BoundaryNorm([-0.125, 0.125, 0.375, 0.625, 0.875], cmap.N)
    lv = sorted({float(r["f_med"]) for r in grid_rows})
    step = lv[1] - lv[0]
    edges = np.r_[np.array(lv) - step / 2, lv[-1] + step / 2]
    cols = (("short_bloom", "capital"), ("short_bloom", "lifecycle"), ("severe_bloom", "capital"),
            ("severe_bloom", "lifecycle"))
    letters = iter("abcdefghijkl")
    used = set()
    for i, dc in enumerate((0.4, 0.5, 0.6)):
        for j, (h, basis) in enumerate(cols):
            ax = fig.add_subplot(gs[i, j])
            z = np.full((len(lv), len(lv)), np.nan)
            gaps = []  # cells where MSF wins at some DAF + UF levels but not at every lower one
            for a, f_msf in enumerate(lv):
                for b, f_med in enumerate(lv):
                    wins = sorted(float(r["f_daf_uf"]) for r in grid_rows
                                  if float(r["dc_share"]) == dc and r["hazard"] == h and float(r["f_med"]) == f_med
                                  and float(r["f_msf"]) == f_msf and r[f"cheapest_{basis}_musd"] in ("msf", "med_msf"))
                    if wins:
                        if wins != levels[:len(wins)]:
                            gaps.append((f_med, f_msf))
                        z[a, b] = wins[-1]
                        used.add(wins[-1])
            ax.pcolormesh(edges, edges, z, cmap=cmap, norm=norm, shading="flat", ec="white", lw=0.8,
                          rasterized=True)
            if gaps:
                ax.plot(*zip(*gaps), "x", ms=3.5, color="white", mew=1.0, zorder=4)
            ax.set_facecolor(PALE)
            ax.set_xlim(edges[0], edges[-1])
            ax.set_ylim(edges[0], edges[-1])
            ax.set_xticks([0, 0.5, 1])
            ax.set_yticks([0, 0.5, 1])
            ax.set_aspect("equal")
            ax.tick_params(length=0)
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 2:
                ax.set_xlabel("MED output kept")
            if j == 0:
                ax.set_ylabel("MSF output kept")
            name = {"short_bloom": "short", "severe_bloom": "severe"}[h]
            panel(ax, next(letters), f"{round(dc * 100)}%, {name}, {basis}")
            axes.append(ax)
    handles = [Patch(fc=c, label=f"≤ {v:g}") for c, v in zip(RED_STEPS, levels) if v in used]
    handles.append(Patch(fc=PALE, label="Never"))
    handles.append(plt.Line2D([], [], marker="x", ls="", ms=3.5, color=INK, mew=1.0,
                              label="Not at every lower DAF + UF level"))
    fig.legend(handles=handles, loc="outside lower center", ncol=6,
               title="A design with MSF is cheapest while DAF + UF keeps at most:", title_fontsize=7.5)
    finish(fig, axes)
    save(fig, "figS2_response_grid")


# --- Figure 3: one event step by step ----------------------------------------

def fig_event() -> None:
    """Supply (a) and cumulative shortfall (b) over one short bloom for the baseline plant and the two cheapest
    designs at the 99% target and 40% critical demand (Table 5, lowest capital), in a 12-day draw in which DAF + UF
    keeps 10% of its output. Other inputs at their base values. The peak of each curve is checked against
    montecarlo.max_shortfall, so the figure shows the model's own result."""
    from design_study import option_for

    from resdesal.architectures import asset_output, build as build_option
    from resdesal.montecarlo import max_shortfall

    hz = HAZARDS["short_bloom"]
    T, rec, gamma, L = 12.0, hz.recovery.base, hz.restart_fraction.base, hz.derate.base
    factors = {node: s.base for node, s in hz.factors.items()}
    factors["daf_uf"] = hz.factors["daf_uf"].low
    d_c = P.critical_demand

    best = {r["family"]: r for r in read("design_best.csv")
            if r["dc_share"] == "0.4" and r["target"] == "0.99" and r["basis"] == "capital_musd"}
    ro, med = best["Strengthened RO and storage (no thermal)"], best["With MED"]

    cases = [
        ("Baseline plant (A0)", build_option("A0"), P.storage / d_c, MUTED),
        ("Strengthened RO and storage", option_for(int(ro["daf_uf_trains"]), int(ro["subsurface_trains"]), 0),
         float(ro["storage_days"]), COLOR["A1"]),
        ("With MED", option_for(int(med["daf_uf_trains"]), int(med["subsurface_trains"]), int(med["med_units"]),
                                "A4"), float(med["storage_days"]), COLOR["A4"]),
    ]

    t = np.linspace(0.0, 45.0, 45_001)
    dt = t[1] - t[0]
    fig = plt.figure(figsize=(WIDTH, 84 * MM), layout="constrained")
    gs = fig.add_gridspec(1, 2)
    a, b = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    for ax in (a, b):
        ax.axvspan(0, T, color=PALE, lw=0, zorder=0)
    a.axhline(1.0, color=INK, lw=0.7, ls=(0, (4, 2)))
    a.text(44.5, 1.04, f"Critical demand, {DC}", fontsize=7.5, ha="right", va="bottom")
    a.text(0.6, 2.62, f"Outage, $T$ = {T:g} d", fontsize=7.5, ha="left", va="top")
    b.text(0.6, 14.6, f"Outage, $T$ = {T:g} d", fontsize=7.5, ha="left", va="top")

    for name, option, storage_days, color in cases:
        out = np.array([asset_output(x, factors) for x in option.assets])
        delays = np.array([option.delays.get(x.name, 0.0) for x in option.assets])
        sig_j = out / d_c
        sigma = sig_j.sum()
        rho = (sum(x.capacity * x.availability for x in option.assets) - out.sum()) / d_c
        u = t - T
        during = np.array([sig_j[delays <= ti].sum() for ti in np.minimum(t, T)])
        after = np.where(u < max(L, rec), sigma + gamma * rho * np.minimum(1.0, np.maximum(u, 0) / rec),
                         sigma + rho)
        s = np.where(t < T, during, after)
        n = np.concatenate([[0.0], np.cumsum((1.0 - s[:-1]) * dt)])
        peak = max_shortfall(T, sig_j, delays, rec, rho, gamma, L)
        assert abs(n.max() - float(peak)) < 0.01, (name, n.max(), peak)

        a.plot(t, s, color=color, lw=1.2, label=name)
        shown = n >= 0
        b.plot(t[shown], n[shown], color=color, lw=1.2)
        b.axhline(storage_days, color=color, lw=0.7, ls=(0, (4, 2)))
        b.text(36.0, storage_days + 0.15, f"Storage {storage_days:g} d", fontsize=7.5, color=INK, ha="right",
               va="bottom")
        k = int(np.argmax(n))
        b.plot([t[k]], [n[k]], "o", ms=3.2, color=color, mec="white", mew=0.5, zorder=3)
        if n[k] > storage_days:
            # Unserved water is the peak minus the storage at onset: a bracket at the peak, not an area.
            b.annotate("", xy=(t[k], storage_days), xytext=(t[k], n[k]),
                       arrowprops=dict(arrowstyle="|-|,widthA=0.25,widthB=0.25", color=INK, lw=0.6,
                                       shrinkA=0, shrinkB=0))
            b.text(t[k] + 0.8, 0.5 * (n[k] + storage_days) - 1.0, f"Unserved, $U$ =\n{n[k] - storage_days:.1f} d "
                   f"of {DC}", fontsize=7.5, ha="left", va="center")

    a.annotate("Ramp", xy=(T + rec / 2, 0.75), xytext=(T + 4.5, 0.35), fontsize=7.5,
               arrowprops=dict(arrowstyle="-", color=INK, lw=0.5))
    a.text(T + (rec + L) / 2 - 1, 1.32, "Derated restart", fontsize=7.5, ha="center", va="bottom")
    a.text(T + L - 0.8, 2.36, "Full output", fontsize=7.5, ha="right", va="center")
    a.set_xlim(0, 45)
    a.set_ylim(0, 2.65)
    a.set_xlabel("Days from onset")
    a.set_ylabel(f"Supply relative to {DC}, $s$")
    a.legend(loc="upper left", bbox_to_anchor=(0.0, 0.93), borderaxespad=0.3, fontsize=7.5)
    panel(a, "a", f"Supply, 12-day short bloom, {DC} = 40% of capacity")
    b.set_xlim(0, 45)
    b.set_ylim(0, 14.8)
    b.set_xlabel("Days from onset")
    b.set_ylabel(f"Cumulative shortfall, $N$ (days of {DC})")
    panel(b, "b", "Cumulative shortfall against storage")
    finish(fig, (a, b))
    save(fig, "fig3_event")


if __name__ == "__main__":
    for f in (fig_framework, fig_events, fig_event, fig_design_map, fig_breakeven, fig_monte_carlo,
              fig_design_fronts, fig_survival_map, fig_msf_threshold, figS2):
        f()
    print(f"Wrote figures to {FIG}")
