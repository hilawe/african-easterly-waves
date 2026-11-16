#!/usr/bin/env python
"""Draw the supplement figure of the cross-reanalysis alignment check on both grids.

The composite of the ERA5 700 hPa curvature vorticity against longitude offset from the
trough center is made on the 1.5 degree global band and on the 0.5 degree regional fields
by scripts/check_track_alignment_region.py, which saves one profile for each grid and one
row of summaries for each. This script draws the two saved profiles on one axis and
writes their summaries beside them: the offset of the peak, the centroid of the offsets
whose mean exceeds half the peak, and the median over troughs of the offset of each
trough's own maximum, with the number of troughs that median is taken over.

A pure consumer. It computes no composite. Before it draws, it recomputes the peak and the
centroid from each saved profile and stops unless they equal the saved summary, so the
curves drawn are the ones the stated numbers were taken from.

Writes fig_track_alignment.png (300 dpi).
"""

import argparse
import os

import numpy as np
import pandas as pd

GRIDS = (("global band", "alignment_band_profile.csv", "1.5 degree band", "#5e3c99", "o", 4.5),
         ("regional", "alignment_region_profile.csv", "0.5 degree regional fields", "#d95f02", "s", 2.6))
TOLERANCE = 5e-6          # the summaries are saved to six decimals


def refuse(msg):
    raise SystemExit(f"REFUSED: {msg}")


def peak_and_centroid(prof):
    """The offset of the peak and the centroid of the offsets whose mean exceeds half the peak."""
    off = np.asarray(prof.offset_deg, float)
    val = np.asarray(prof.mean_curvature_vorticity, float)
    ipk = int(np.argmax(val))
    lobe = val > 0.5 * val[ipk]
    return float(off[ipk]), float(np.sum(off[lobe] * val[lobe]) / np.sum(val[lobe]))


def load(dep):
    """The summary row and the profile of each grid, each profile checked against its row."""
    summary = pd.read_csv(os.path.join(dep, "alignment_region.csv"))
    if sorted(summary.grid) != sorted(g[0] for g in GRIDS):
        refuse(f"alignment_region.csv holds the grids {sorted(summary.grid)}")
    out = []
    for grid, name, label, color, marker, size in GRIDS:
        row = summary[summary.grid == grid].iloc[0]
        prof = pd.read_csv(os.path.join(dep, name))
        step = np.diff(np.asarray(prof.offset_deg, float))
        if not np.allclose(step, float(row.grid_step_deg)) or abs(float(prof.offset_deg.iloc[0]) + float(prof.offset_deg.iloc[-1])) > 1e-9:
            refuse(f"{name} is not a window about zero on a grid of {row.grid_step_deg} degrees")
        peak, centroid = peak_and_centroid(prof)
        if abs(peak - float(row.peak_deg)) > TOLERANCE or abs(centroid - float(row.centroid_deg)) > TOLERANCE:
            refuse(f"{name} gives peak {peak} and centroid {centroid:.6f}, and its summary states "
                   f"{row.peak_deg} and {row.centroid_deg}")
        out.append(dict(grid=grid, label=label, color=color, marker=marker, size=size, prof=prof, row=row))
    return out


def summary_line(d):
    r = d["row"]
    return (f"{float(r.grid_step_deg):.1f} degree: peak {float(r.peak_deg):+.1f}, centroid {float(r.centroid_deg):+.2f}, "
            f"median of maxima {float(r.median_argmax_deg):+.1f} (n = {int(r.n_argmax):,})")


def draw(data, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.unicode_minus": False})
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    ax.axvline(0, color="k", lw=0.8, ls="--", zorder=0.5)
    for d in data:
        ax.plot(d["prof"].offset_deg, d["prof"].mean_curvature_vorticity * 1e6, "-", color=d["color"], lw=1.3,
                marker=d["marker"], ms=d["size"], label=d["label"], zorder=3)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.30 * (hi - lo))           # room above the curves for the summaries
    for d in data:                                   # the centroid of each grid, marked on the lower axis
        ax.plot([float(d["row"].centroid_deg)], [ax.get_ylim()[0]], marker="^", ms=7, color=d["color"],
                clip_on=False, zorder=5)
    n = int(data[0]["row"].n_matched)
    lines = [f"{n:,} trough observations on each grid, offsets in degrees east"] + [summary_line(d) for d in data]
    ax.text(0.015, 0.985, "\n".join(lines), transform=ax.transAxes, fontsize=7.6, va="top", color="#222222", linespacing=1.35,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.5), zorder=6)
    ax.set_xlabel("longitude offset from the AEWC trough center (degrees, east positive)")
    ax.set_ylabel(r"mean ERA5 700 hPa curvature vorticity ($10^{-6}$ s$^{-1}$)")
    ax.set_title("Cross-reanalysis alignment on two grids", fontsize=10)
    ax.legend(loc="center right", frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deposit", default="deposit")
    ap.add_argument("--out", default="fig_track_alignment.png")
    a = ap.parse_args()
    data = load(a.deposit)
    if len({int(d["row"].n_matched) for d in data}) != 1:
        refuse("the two grids do not hold the same number of trough observations")
    for line in draw(data, a.out):
        print(line)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
