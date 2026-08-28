#!/usr/bin/env python
"""Compose figure F6, the Lagrangian supply contrast, from the deposit tables.

Panel (a): parcel origins 72 h before trough passage in trough-relative coordinates,
MCS-active and MCS-quiet, with the -24 h sampling box and the group-mean inflow paths.
Panel (b): the hero curve, the MCS-active minus MCS-quiet relative humidity along the
tracked inflow against passage-relative time, with the wave-cluster bootstrap band and
mean parcel separations annotated, and the fixed-box Eulerian estimates overplotted in
grey on the same axes (meridian and displaced boxes at -24 h, the longer-lead box at
-36 h, the source-offset controls at -72 h). All values come from the deposit driver's
tables (canonical_numbers.csv, parcels_*.csv, cases_*.csv, paths_*.csv); this script
draws and annotates only.

Run scripts/build_deposit.py first. Writes fig_supply_contrast.png (300 dpi).
"""

import argparse
import os

import numpy as np
import pandas as pd

from aew.plotting import panel_label

C_QUIET = "tab:blue"
C_ACTIVE = "tab:red"
C_LAGR = "#5e3c99"
C_EUL = "#666666"


def get(df, tier, level, stat, trel=None):
    m = df[(df.tier == tier) & (df.level == level) & (df.statistic == stat)]
    if trel is not None:
        m = m[m.time_rel_h == trel]
    if len(m) != 1:
        raise ValueError(f"expected one row for {stat} t{trel}, got {len(m)}")
    return m.iloc[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deposit", default="deposit")
    ap.add_argument("--tier", default="pooled")
    ap.add_argument("--level", type=int, default=700)
    ap.add_argument("--out", default="fig_supply_contrast.png")
    a = ap.parse_args()
    df = pd.read_csv(os.path.join(a.deposit, "canonical_numbers.csv"))
    parcels = pd.read_csv(os.path.join(a.deposit, f"parcels_{a.tier}_{a.level}.csv"))
    cases = pd.read_csv(os.path.join(a.deposit, f"cases_{a.tier}_{a.level}.csv"),
                        usecols=["case", "lon", "label"])
    paths = pd.read_csv(os.path.join(a.deposit, f"paths_{a.tier}_{a.level}.csv"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    plt.rcParams.update({
        "axes.unicode_minus": False,
        "font.size": 14.5,
        "axes.titlesize": 16,
        "axes.labelsize": 14.5,
        "xtick.labelsize": 14.5,
        "ytick.labelsize": 14.5,
        "legend.fontsize": 14.5,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 6.2),
                                   gridspec_kw={"width_ratios": [1.05, 1.0]})

    # ---- panel a: origins in trough-relative coordinates ----
    from matplotlib import patheffects as pe
    p = parcels.merge(cases, on="case")
    p["rel_lon"] = p.origin_lon - p.lon
    p = p[np.isfinite(p.rel_lon) & np.isfinite(p.origin_lat)]
    # Class-specific density contours survive reduction better than tens of thousands
    # of sub-point, low-alpha markers. The contours are display-only and do not alter
    # any statistic in the canonical table.
    def smooth2d(z, passes=2):
        z = np.asarray(z, dtype=float)
        for _ in range(passes):
            q = np.pad(z, 1, mode="edge")
            z = (q[:-2, :-2] + 2 * q[:-2, 1:-1] + q[:-2, 2:]
                 + 2 * q[1:-1, :-2] + 4 * q[1:-1, 1:-1] + 2 * q[1:-1, 2:]
                 + q[2:, :-2] + 2 * q[2:, 1:-1] + q[2:, 2:]) / 16.0
        return z

    def mass_threshold(z, fraction):
        ordered = np.sort(z.ravel())[::-1]
        csum = np.cumsum(ordered)
        return ordered[np.searchsorted(csum, fraction * csum[-1])]

    xedges = np.linspace(-14, 26, 49)
    yedges = np.linspace(-3, 29, 41)
    for lab, col in (("MCS-quiet", C_QUIET), ("MCS-active", C_ACTIVE)):
        g = p[p.label == lab]
        hist, _, _ = np.histogram2d(g.rel_lon, g.origin_lat, bins=(xedges, yedges))
        hist = smooth2d(hist)
        levels = sorted({mass_threshold(hist, 0.80), mass_threshold(hist, 0.50)})
        if len(levels) == 2 and levels[0] < levels[1]:
            xc = 0.5 * (xedges[:-1] + xedges[1:])
            yc = 0.5 * (yedges[:-1] + yedges[1:])
            ax1.contour(xc, yc, hist.T, levels=levels, colors=col,
                        linewidths=(1.6, 2.4), alpha=0.75, zorder=2)
    for lab, col, name, ls in (("MCS-quiet", C_QUIET, "MCS-quiet", (0, (4, 2))),
                               ("MCS-active", C_ACTIVE, "MCS-active", "-")):
        pt = paths[paths.group == lab].sort_values("elapsed_h")
        ax1.plot(pt.mean_rel_lon, pt.mean_lat, color=col, lw=2.8, ls=ls,
                 solid_capstyle="round", zorder=5, label=f"{name} mean inflow",
                 path_effects=[pe.Stroke(linewidth=4.4, foreground="white"),
                               pe.Normal()])
        end = pt.iloc[-1]
        ax1.plot(end.mean_rel_lon, end.mean_lat, "o", color=col, ms=5.5, zorder=6,
                 mec="white", mew=0.8)
        # flow direction: air moves from the -72 h origin toward the sampling box
        ax1.annotate("", xy=(pt.iloc[0].mean_rel_lon, pt.iloc[0].mean_lat),
                     xytext=(pt.iloc[4].mean_rel_lon, pt.iloc[4].mean_lat),
                     arrowprops=dict(arrowstyle="-|>", color=col, lw=2.2), zorder=6)
    box = Rectangle((-5, 5), 10, 10, fill=False, ls="--", lw=1.6, ec="k", zorder=4)
    ax1.add_patch(box)
    ax1.annotate("sampling box (-24 h)", (-5, 15.3), fontsize=14.5, color="k")
    ax1.annotate("origins at -72 h", (0.03, 0.03), xycoords="axes fraction",
                 fontsize=14.5, color="#333333")
    leg = ax1.legend(fontsize=14.5, loc="upper right")
    for lh in leg.get_lines():
        lh.set_alpha(1)
    ax1.set_xlim(-14, 26)
    ax1.set_ylim(-3, 29)
    ax1.set_xlabel("longitude relative to the trough (deg)")
    ax1.set_ylabel("latitude (deg N)")
    ax1.set_title("Parcel origins and mean inflow paths")
    panel_label(ax1, "a", 15)
    ax1.grid(alpha=0.25)

    # ---- panel b: the supply contrast against the fixed frame ----
    lag = df[(df.tier == a.tier) & (df.level == a.level)
             & (df.statistic == "lagrangian_rh")].sort_values("time_rel_h")
    t = lag.time_rel_h.values
    ax2.fill_between(t, lag.ci_lo, lag.ci_hi, color=C_LAGR, alpha=0.18,
                     label="wave-cluster bootstrap 95% interval")
    eul = [(get(df, a.tier, a.level, "eulerian_box", -24), -24, 0.0),
           (get(df, a.tier, a.level, "eulerian_box_L-5", -24), -24, -1.2),
           (get(df, a.tier, a.level, "eulerian_box_L-8", -24), -24, 1.2),
           (get(df, a.tier, a.level, "eulerian_box_-36h", -36), -36, 0.0),
           (get(df, a.tier, a.level, "eulerian_control_L+8", -72), -72, -1.2),
           (get(df, a.tier, a.level, "eulerian_control_L+12", -72), -72, 1.2)]
    for i, (r, trel, jit) in enumerate(eul):
        ax2.errorbar(trel + jit, r["diff"], yerr=[[r["diff"] - r.ci_lo],
                                                  [r.ci_hi - r["diff"]]],
                     fmt="s", color=C_EUL, mfc="white", mec=C_EUL, mew=1.2,
                     ms=6, elinewidth=1.6, capsize=3,
                     label="Fixed boxes (Eulerian)" if i == 0 else None, zorder=2)

    ax2.plot(t, lag["diff"], "o-", color=C_LAGR, lw=2.4, ms=6, zorder=4,
             label="Tracked inflow (Lagrangian)")

    # Two ratios, each drawn where its own comparison lives. The seed-box ratio
    # (attenuation_factor) compares the -72 h tracked value against the -24 h box, so
    # its arrow belongs at -24 h; drawing it at the -72 h end implied a gap against
    # the -72 h fixed boxes that is about a third smaller (found 2026-07-25). The matched-lead ratio is drawn at -72 h against the mean of the
    # two -72 h control boxes.
    att = get(df, a.tier, a.level, "attenuation_factor")["diff"]
    matched = get(df, a.tier, a.level, "attenuation_factor_matched")["diff"]
    r72 = lag[lag.time_rel_h == -72].iloc[0]
    ebox = get(df, a.tier, a.level, "eulerian_box", -24)
    r24 = lag[lag.time_rel_h == -24].iloc[0]
    e72 = 0.5 * (get(df, a.tier, a.level, "eulerian_control_L+8", -72)["diff"]
                 + get(df, a.tier, a.level, "eulerian_control_L+12", -72)["diff"])

    ax2.annotate("", xy=(-23.0, ebox["diff"]), xytext=(-23.0, r72["diff"]),
                 arrowprops=dict(arrowstyle="->", color="#333333", lw=1.6))
    ax2.annotate(f"{att:.1f}x vs\nseed box", (-27.5, 0.5 * (ebox["diff"]
                 + r72["diff"])), fontsize=14.5, color="#333333", va="center",
                 ha="right")
    ax2.annotate("", xy=(-69.5, e72), xytext=(-69.5, r72["diff"]),
                 arrowprops=dict(arrowstyle="->", color="#333333", lw=1.6))
    ax2.annotate(f"{matched:.1f}x at -72 h",
                 (-68.8, 0.5 * (e72 + r72["diff"])), fontsize=14.5,
                 color="#333333", va="center")
    print(f"F6 ratios: seed-box {att:.2f}x (Lagrangian -72 h {r72['diff']:+.3f} over "
          f"-24 h box {ebox['diff']:+.3f}); matched-lead {matched:.2f}x (over the "
          f"-72 h control-box mean {e72:+.3f}); Lagrangian at -24 h {r24['diff']:+.3f}")

    ax2.axhline(0, color="k", lw=1.2)
    ax2.set_xlim(-76, -20)
    ax2.set_xlabel("time relative to trough passage (h)")
    ax2.set_ylabel(f"MCS-active minus MCS-quiet {a.level} hPa RH (%)")
    ax2.set_title("The supply contrast in two frames")
    panel_label(ax2, "b", 15)
    handles, labels = ax2.get_legend_handles_labels()
    short = {
        "wave-cluster bootstrap 95% interval": "95% bootstrap interval",
        "Tracked inflow (Lagrangian)": "Tracked inflow",
        "Fixed boxes (Eulerian)": "Fixed boxes",
    }
    ax2.legend(handles, [short.get(x, x) for x in labels], fontsize=14.5,
               loc="upper right")
    ax2.grid(alpha=0.25)

    tier_years = {"dev": "2000-2004", "heldout": "1983-1999 and 2005-2007",
                  "pooled": "1983-2007"}
    fig.tight_layout()
    fig.savefig(a.out, dpi=300)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
