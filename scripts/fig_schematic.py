#!/usr/bin/env python
"""Compose figure F9, the two-layer gating schematic (rebuild of the approved design).

Design grammar, per the review rulings: transport differences are drawn ONLY in the
700 hPa arrows (widths and routes), state differences ONLY in the 850 hPa fill; the
850 hPa inflow arrows are identical southwest monsoon arrows in both panels (the
measured ground-relative origin distribution, west and south dominated); all flow
arrows are ground-relative, stated in the legend line, with the caption noting the
850 hPa arrows reverse in the wave-relative frame; the wake-recovery link is labeled
interpretation; the theta-e glyph reads "vapor outweighs cooling"; in-panel labels are
print-safe charcoal, colored labels use the darkest stops of their ramps, and only the
W/E axis ticks stay muted.

Purely a drawing; no data dependencies. Writes fig_schematic.png (300 dpi).
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

CHAR = "#333333"          # charcoal labels
MUTED = "#999999"         # muted axis ticks
SAHARA = "#8c510a"        # dark tan (Saharan-side arrows and dry labels)
MONSOON = "#01665e"       # dark teal-green (monsoon-side arrows and moist labels)
AEJ = "#4d4d4d"           # jet-level carrier arrow
FILL700 = "#eef2f5"       # neutral jet-level band, identical in both panels
FILL850_ACTIVE = "#c7e9c0"   # moister, cooler monsoon layer
FILL850_QUIET = "#f6e8c3"    # drier, warmer monsoon layer


def arrow(ax, xy0, xy1, color, lw, label=None, label_xy=None, ls="-", fs=8.0,
          label_color=None, ha="left", style="-|>", mutation=16):
    ax.add_patch(FancyArrowPatch(xy0, xy1, arrowstyle=style, color=color, lw=lw,
                                 linestyle=ls, mutation_scale=mutation, zorder=6))
    if label:
        ax.text(*label_xy, label, fontsize=fs, color=label_color or color,
                ha=ha, va="center", zorder=7)


def trough_axis(ax, x, y0, y1):
    yy = np.linspace(y0, y1, 100)
    xx = x + 0.12 * np.sin(2.5 * np.pi * (yy - y0) / (y1 - y0))
    ax.plot(xx, yy, color=CHAR, lw=1.6, ls=(0, (5, 3)), zorder=5)


def panel(ax, active):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10.6)
    ax.axis("off")
    title = "MCS-ACTIVE trough" if active else "MCS-QUIET trough"
    ax.text(5.0, 10.25, title, fontsize=11.5, fontweight="bold", ha="center",
            color=CHAR)
    from aew.plotting import panel_label
    panel_label(ax, "a" if active else "b", 15)

    # the two layers: state differences appear only in the 850 hPa fill
    ax.add_patch(Rectangle((0.3, 5.6), 9.4, 3.6, fc=FILL700, ec="#c5ccd3", lw=0.8))
    ax.add_patch(Rectangle((0.3, 0.9), 9.4, 3.6,
                           fc=FILL850_ACTIVE if active else FILL850_QUIET,
                           ec="#c5ccd3", lw=0.8))
    ax.text(0.55, 8.85, "700 hPa  (jet level)", fontsize=8.5, color=CHAR, va="top")
    ax.text(0.55, 4.15, "850 hPa  (monsoon layer)", fontsize=8.5, color=CHAR,
            va="top")

    # trough axis through both layers
    trough_axis(ax, 6.6, 0.9, 9.2)
    ax.text(6.6, 9.55, "trough axis", fontsize=8, color=CHAR, ha="center")
    # muted W/E orientation ticks
    ax.text(0.35, 0.35, "W", fontsize=9, color=MUTED)
    ax.text(9.5, 0.35, "E", fontsize=9, color=MUTED, ha="right")

    # ---- 700 hPa: transport differences drawn only in the arrows ----
    # the jet-level carrier inflow from the east, same in both panels
    arrow(ax, (9.4, 7.4), (7.1, 7.4), AEJ, 2.2, "AEJ inflow", (9.35, 7.85),
          label_color=CHAR, ha="right")
    if active:
        # reduced Saharan-side import (thin), enhanced monsoon-side origins (thicker)
        arrow(ax, (8.6, 9.0), (7.2, 7.9), SAHARA, 1.1,
              "reduced Saharan-side import", (8.15, 9.25), fs=7.8, ha="center")
        arrow(ax, (8.6, 5.9), (7.2, 6.9), MONSOON, 2.6,
              "more monsoon-side origins", (8.2, 5.75), fs=7.8, ha="center")
    else:
        arrow(ax, (8.6, 9.0), (7.2, 7.9), SAHARA, 3.0,
              "drier, warmer inflow", (8.15, 9.25), fs=7.8, ha="center")
        arrow(ax, (8.6, 5.9), (7.2, 6.9), MONSOON, 1.1,
              "fewer monsoon-side origins", (8.2, 5.75), fs=7.8, ha="center")

    # pre-trough convective response region, west of and at the axis
    if active:
        cx = 5.6
        for dx, dy, r in ((0, 0, 0.62), (0.55, 0.18, 0.44), (-0.55, 0.14, 0.44)):
            ax.add_patch(plt.Circle((cx + dx, 7.35 + dy), r, fc="white",
                                    ec=CHAR, lw=1.0, zorder=6))
        ax.text(cx, 6.35, "deep convection\ndevelops", fontsize=7.8, color=CHAR,
                ha="center", va="top", zorder=7)
    else:
        ax.text(5.6, 7.3, "convection\nstays shallow", fontsize=7.8, color=CHAR,
                ha="center", style="italic", zorder=7)

    # ---- 850 hPa: identical southwest monsoon arrows in both panels ----
    for x0 in (1.3, 3.1):
        arrow(ax, (x0, 1.35), (x0 + 1.5, 2.75), MONSOON, 1.8)
    ax.text(1.25, 0.6, "shared SW monsoon inflow", fontsize=7.8, color=CHAR)
    if active:
        ax.text(5.35, 3.0, "moister, cooler,\nhigher theta-e\n(state contrast)",
                fontsize=8.2, color="#00441b", ha="center", va="center", zorder=7)
        # wake recovery, labeled interpretation
        for dx in (-0.3, 0.15, 0.6):
            ax.add_patch(plt.Circle((8.35 + dx, 2.95), 0.3, fc="white", ec=CHAR,
                                    lw=0.9, zorder=6))
        for dx in (-0.25, 0.15, 0.55):
            ax.plot([8.4 + dx, 8.25 + dx], [2.6, 2.15], color=MONSOON, lw=0.9,
                    zorder=6)
        ax.text(8.35, 1.55, "prior-day convection;\nwake recovery (interpretation)",
                fontsize=7.2, color=CHAR, ha="center", va="top", style="italic",
                zorder=7)
    else:
        ax.text(5.35, 2.9, "drier, warmer\nmonsoon layer", fontsize=8.2,
                color="#7f4909", ha="center", va="center", zorder=7)

    # theta-e glyph between the layers, the round-14 wording
    if active:
        ax.add_patch(FancyBboxPatch((1.15, 4.65), 4.3, 0.75,
                                    boxstyle="round,pad=0.12", fc="white", ec=CHAR,
                                    lw=0.9, zorder=8))
        ax.text(3.3, 5.02, "theta-e > 0: vapor outweighs cooling", fontsize=8.2,
                color=CHAR, ha="center", va="center", zorder=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="fig_schematic.png")
    a = ap.parse_args()
    fig, (axa, axq) = plt.subplots(1, 2, figsize=(12.0, 6.0))
    panel(axa, active=True)
    panel(axq, active=False)
    fig.suptitle("Two-layer thermodynamic gating of convective development",
                 fontsize=12.5, color=CHAR)
    fig.text(0.5, 0.015,
             "All flow arrows are ground-relative (the frame of the trajectory "
             "integration); in the wave-relative frame the 850 hPa arrows reverse.",
             fontsize=8.5, color=CHAR, ha="center")
    fig.tight_layout(rect=(0, 0.035, 1, 0.95))
    fig.savefig(a.out, dpi=300)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
