#!/usr/bin/env python
"""Compose figure S4, the timing of the four windows relative to trough passage.

A drawing of the design and no more: the hours are the constants of the pipeline, read
from the modules that use them, so the figure cannot state another window than the one
the counts were made over. Time runs along the horizontal axis in hours relative to the
passage of the trough at the sampling meridian. Four bars show the response window, the
antecedent window, the reading of the fixed box and the tracing of the inflow.

Writes fig_timeline.png (300 dpi).
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aew.observed import window_offsets

RESP_H = 24.0            # the response window, hours after passage
ANTE_FROM, ANTE_TO = -48.0, -24.0
BOX_H = -24.0            # the fixed box is read, and the parcels are seeded, at this hour
TRACE_H = 48.0           # the inflow is traced this many hours back from the seed
C_RESP, C_ANTE, C_ENV = "#b2182b", "#8c510a", "#54278f"


def windows():
    """The four windows as (label, start hour, end hour, color, row), checked against the pipeline."""
    offs = window_offsets(RESP_H, antecedent=True)
    resp = [o for o in offs if o >= 0]
    ante = [o for o in offs if o < 0]
    if (min(resp), max(resp) + 3.0) != (0.0, RESP_H) or (min(ante), max(ante) + 3.0) != (ANTE_FROM, ANTE_TO):
        raise SystemExit("the windows drawn are not those the counts are made over")
    return [("response count, within 8 degrees of the passage longitude", 0.0, RESP_H, C_RESP, 3),
            ("antecedent count, within 10 degrees of the passage longitude", ANTE_FROM, ANTE_TO, C_ANTE, 2),
            ("inflow traced back from the seed", BOX_H - TRACE_H, BOX_H, C_ENV, 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="fig_timeline.png")
    a = ap.parse_args()
    plt.rcParams.update({"font.size": 12, "axes.unicode_minus": False})
    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    for label, t0, t1, color, row in windows():
        ax.barh(row, t1 - t0, left=t0, height=0.46, color=color, alpha=0.85, zorder=3)
        # the response label ends at its bar, so it stays inside the figure
        x, ha = (t1, "right") if t0 >= 0 else (t0 + 0.5 * (t1 - t0), "center")
        ax.text(x, row + 0.36, label, ha=ha, va="bottom", fontsize=11, color=color)
    ax.plot([BOX_H], [0], marker="D", ms=9, color=C_ENV, zorder=4)
    ax.text(BOX_H, 0.3, "fixed box read, parcels seeded", ha="center", va="bottom", fontsize=11, color=C_ENV)
    ax.plot([BOX_H - TRACE_H], [1], marker="o", ms=8, color=C_ENV, mec="white", zorder=5)
    ax.text(BOX_H - TRACE_H, 0.55, "earliest\nhumidity sample", ha="center", va="top", fontsize=10, color=C_ENV)
    ax.axvline(0.0, color="k", lw=1.4, zorder=2)
    ax.text(0.6, -0.45, "trough passage", ha="left", va="center", fontsize=11)
    ax.set_xlim(-78, 30)
    ax.set_ylim(-0.8, 4.0)
    ax.set_xticks(range(-72, 25, 12))
    ax.set_yticks([])
    ax.set_xlabel("time relative to trough passage (h)")
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    fig.savefig(a.out, dpi=300)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
