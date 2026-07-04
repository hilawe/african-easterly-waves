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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0),
                                   gridspec_kw={"width_ratios": [1.05, 1.0]})

    # ---- panel a: origins in trough-relative coordinates ----
    from matplotlib import patheffects as pe
    p = parcels.merge(cases, on="case")
    p["rel_lon"] = p.origin_lon - p.lon
    p = p[np.isfinite(p.rel_lon) & np.isfinite(p.origin_lat)]
    # display-only subsample, interleaved so neither class overpaints the other
    # (fixed seed; every statistic comes from the canonical table, not this scatter)
    rng = np.random.default_rng(0)
    per_class = 20000
    show = pd.concat([
        p[p.label == lab].sample(n=min(per_class, (p.label == lab).sum()),
                                 random_state=0)
        for lab in ("MCS-quiet", "MCS-active")])
    show = show.iloc[rng.permutation(len(show))]
    ax1.scatter(show.rel_lon, show.origin_lat, s=1.2,
                c=np.where(show.label == "MCS-active", C_ACTIVE, C_QUIET),
                alpha=0.10, rasterized=True)
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
    box = Rectangle((-5, 5), 10, 10, fill=False, ls="--", lw=1.2, ec="k", zorder=4)
    ax1.add_patch(box)
    ax1.annotate("sampling box (-24 h)", (-5, 15.3), fontsize=8, color="k")
    ax1.annotate("origins at -72 h", (0.03, 0.03), xycoords="axes fraction",
                 fontsize=8, color="#333333")
    leg = ax1.legend(fontsize=8, loc="upper right")
    for lh in leg.get_lines():
        lh.set_alpha(1)
    ax1.set_xlim(-14, 26)
    ax1.set_ylim(-3, 29)
    ax1.set_xlabel("longitude relative to the trough (deg)")
    ax1.set_ylabel("latitude (deg N)")
    ax1.set_title("(a) Parcel origins and mean inflow paths")
    ax1.grid(alpha=0.25)

    # ---- panel b: the supply contrast against the fixed frame ----
    lag = df[(df.tier == a.tier) & (df.level == a.level)
             & (df.statistic == "lagrangian_rh")].sort_values("time_rel_h")
    t = lag.time_rel_h.values
    ax2.fill_between(t, lag.ci_lo, lag.ci_hi, color=C_LAGR, alpha=0.18,
                     label="wave-cluster bootstrap 95% interval")
    ax2.plot(t, lag["diff"], "o-", color=C_LAGR, lw=2.0, ms=5,
             label="tracked inflow (Lagrangian)")
    for _, r in lag.iterrows():
        sep = r["note"].split("separation")[-1].strip()
        ax2.annotate(sep, (r.time_rel_h, r.ci_hi), textcoords="offset points",
                     xytext=(0, 5), ha="center", fontsize=7, color="#555555")

    eul = [(get(df, a.tier, a.level, "eulerian_box", -24), -24, 0.0),
           (get(df, a.tier, a.level, "eulerian_box_L-5", -24), -24, -0.7),
           (get(df, a.tier, a.level, "eulerian_box_L-8", -24), -24, 0.7),
           (get(df, a.tier, a.level, "eulerian_box_-36h", -36), -36, 0.0),
           (get(df, a.tier, a.level, "eulerian_control_L+8", -72), -72, -0.7),
           (get(df, a.tier, a.level, "eulerian_control_L+12", -72), -72, 0.7)]
    for i, (r, trel, jit) in enumerate(eul):
        ax2.errorbar(trel + jit, r["diff"], yerr=[[r["diff"] - r.ci_lo],
                                                  [r.ci_hi - r["diff"]]],
                     fmt="s", color=C_EUL, ms=4.5, elinewidth=0.9, capsize=2,
                     label="fixed boxes (Eulerian)" if i == 0 else None)

    att = get(df, a.tier, a.level, "attenuation_factor")["diff"]
    r72 = lag[lag.time_rel_h == -72].iloc[0]
    e72 = get(df, a.tier, a.level, "eulerian_box", -24)
    ax2.annotate("", xy=(-69.5, e72["diff"]), xytext=(-69.5, r72["diff"]),
                 arrowprops=dict(arrowstyle="->", color="#333333", lw=1.0))
    ax2.annotate(f"{att:.1f}x attenuation\nin the fixed frame",
                 (-68.8, 0.5 * (e72["diff"] + r72["diff"])), fontsize=8,
                 color="#333333", va="center")

    ax2.axhline(0, color="k", lw=0.7)
    ax2.set_xlim(-76, -20)
    ax2.set_xlabel("time relative to trough passage (h)")
    ax2.set_ylabel(f"MCS-active minus MCS-quiet {a.level} hPa RH (%)")
    ax2.set_title("(b) The supply contrast in two frames")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(alpha=0.25)

    tier_years = {"dev": "2000-2004", "heldout": "1983-1999 and 2005-2007",
                  "pooled": "1983-2007"}
    fig.suptitle(f"The Lagrangian supply contrast, "
                 f"{tier_years.get(a.tier, a.tier)} ({a.level} hPa)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(a.out, dpi=300)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
