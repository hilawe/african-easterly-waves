#!/usr/bin/env python
"""Compose figure F8, the organization-axis control, from the deposit driver's table.

The point of F8 is a shape contrast under identical machinery: the 600-925 hPa shear
deficit shrinks as the sampling box moves upstream and is indistinguishable from zero
at the wave level, while the 700 hPa moisture contrast grows along the tracked inflow
and survives collapse to the wave level. Panel (a) draws the shear estimands (meridian
box, displaced boxes, wave level); panel (b) draws the moisture estimands (meridian
box, the tracked inflow at -48 and -72 h, wave level). Every value is read from
deposit/canonical_numbers.csv; this script draws and annotates only.

Run scripts/build_deposit.py first. Writes fig_organization_axis.png (300 dpi).
"""

import argparse

import numpy as np
import pandas as pd

C_SHEAR = "#b35806"
C_MOIST = "#5e3c99"


def get(df, tier, level, stat, trel=None):
    m = df[(df.tier == tier) & (df.statistic == stat)]
    m = m[m.level == level] if level is not None else m[m.level.isna()]
    if trel is not None:
        m = m[m.time_rel_h == trel]
    if len(m) != 1:
        raise ValueError(f"expected one row for {stat} t{trel}, got {len(m)}")
    return m.iloc[0]


def panel(ax, rows, labels, color, unit, title):
    x = np.arange(len(rows))
    d = [r["diff"] for r in rows]
    lo = [r["diff"] - r.ci_lo for r in rows]
    hi = [r.ci_hi - r["diff"] for r in rows]
    ax.axhline(0, color="k", lw=0.8)
    ax.errorbar(x, d, yerr=[lo, hi], fmt="o-", color=color, ms=6, lw=1.6,
                elinewidth=1.1, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel(f"MCS-active minus MCS-quiet ({unit})")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25, axis="y")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="deposit/canonical_numbers.csv")
    ap.add_argument("--tier", default="pooled")
    ap.add_argument("--out", default="fig_organization_axis.png")
    a = ap.parse_args()
    df = pd.read_csv(a.table)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    shear = [get(df, a.tier, None, "shear_box"),
             get(df, a.tier, None, "shear_box_L-5"),
             get(df, a.tier, None, "shear_box_L-8"),
             get(df, a.tier, None, "shear_wave_level")]
    panel(ax1, shear, ["meridian\nbox", "box at\nL-5", "box at\nL-8", "wave\nlevel"],
          C_SHEAR, "m/s", "(a) 600-925 hPa shear: shrinks upstream, zero at wave level")
    moist = [get(df, a.tier, 700, "eulerian_box", -24),
             get(df, a.tier, 700, "lagrangian_rh", -48),
             get(df, a.tier, 700, "lagrangian_rh", -72),
             get(df, a.tier, 700, "lagrangian_rh_wave_level", -72)]
    panel(ax2, moist, ["meridian\nbox (-24 h)", "inflow\nat -48 h", "inflow\nat -72 h",
                       "wave\nlevel"],
          C_MOIST, "%", "(b) 700 hPa moisture: grows along the inflow, survives")
    fig.suptitle("The organization axis against the moisture axis, 1983-2007",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(a.out, dpi=300)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
