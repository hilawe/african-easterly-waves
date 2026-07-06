#!/usr/bin/env python
"""Compose figure F7, the airmass fingerprint, from the deposit driver's table.

Panel (a): the vapor/temperature decomposition of the along-inflow contrast
(MCS-active minus MCS-quiet) at -24, -48, and -72 h relative to trough passage, both
levels, with wave-cluster bootstrap intervals. Panel (b): the theta-e budget at -72 h,
the measured vapor and cooling terms, their predicted net, and the observed theta-e
contrast, at both levels. Every number is read from deposit/canonical_numbers.csv
(the one-number-one-source table); this script draws and annotates only.

Run scripts/build_deposit.py first. Writes fig_fingerprint.png (300 dpi).
"""

import argparse

import numpy as np
import pandas as pd

from aew.plotting import panel_label


def rows(df, tier, level, stat):
    m = df[(df.tier == tier) & (df.level == level) & (df.statistic == stat)]
    return m.sort_values("time_rel_h", ascending=False)


def one(df, tier, level, stat, trel=None):
    m = df[(df.tier == tier) & (df.level == level) & (df.statistic == stat)]
    if trel is not None:
        m = m[m.time_rel_h == trel]
    if len(m) != 1:
        raise ValueError(f"expected one row for {stat} L{level} t{trel}, got {len(m)}")
    return m.iloc[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="deposit/canonical_numbers.csv")
    ap.add_argument("--tier", default="pooled")
    ap.add_argument("--out", default="fig_fingerprint.png")
    a = ap.parse_args()
    df = pd.read_csv(a.table)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C_MIX = "#1b7837"   # vapor green
    C_T = "#762a83"     # temperature purple
    C_OBS = "#2166ac"   # observed theta-e blue
    C_PRED = "#999999"  # predicted net grey

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

    # ---- panel a: decomposition along the inflow ----
    times = (-24.0, -48.0, -72.0)
    width = 0.16
    xbase = np.arange(len(times), dtype=float)
    for k, level in enumerate((700, 850)):
        mix = [one(df, a.tier, level, "decomp_mix", t) for t in times]
        tt = [one(df, a.tier, level, "decomp_T", t) for t in times]
        hatch = None if level == 700 else "//"
        for j, (vals, col, lab) in enumerate(
                ((mix, C_MIX, f"mixing ratio, {level} hPa (g/kg)"),
                 (tt, C_T, f"temperature, {level} hPa (K)"))):
            x = xbase + (2 * k + j - 1.5) * width
            d = [r["diff"] for r in vals]
            lo = [r["diff"] - r["ci_lo"] for r in vals]
            hi = [r["ci_hi"] - r["diff"] for r in vals]
            ax1.bar(x, d, width * 0.9, color=col, alpha=0.85 if level == 700 else 0.55,
                    hatch=hatch, label=lab, edgecolor="white", linewidth=0.5)
            ax1.errorbar(x, d, yerr=[lo, hi], fmt="none", ecolor="#333333",
                         elinewidth=1.0, capsize=2)
    ax1.axhline(0, color="k", lw=0.8)
    ax1.set_xticks(xbase)
    ax1.set_xticklabels([f"{int(t)} h" for t in times])
    ax1.set_xlabel("time relative to trough passage")
    ax1.set_ylabel("MCS-active minus MCS-quiet")
    ax1.set_title("Vapor and temperature along the inflow")
    panel_label(ax1, "a", 20)
    ax1.legend(fontsize=7.5, loc="lower left", frameon=False)
    ax1.grid(alpha=0.25, axis="y")

    # ---- panel b: the theta-e budget at -72 h ----
    labels = ["vapor\nterm", "cooling\nterm", "predicted\nnet", "observed\ntheta-e"]
    xb = np.arange(len(labels), dtype=float)
    for k, level in enumerate((700, 850)):
        vap = one(df, a.tier, level, "thetae_budget_vapor_term")["diff"]
        cool = one(df, a.tier, level, "thetae_budget_cooling_term")["diff"]
        pred = one(df, a.tier, level, "thetae_budget_predicted")["diff"]
        obs = one(df, a.tier, level, "thetae", -72.0)
        d = [vap, cool, pred, obs["diff"]]
        cols = [C_MIX, C_T, C_PRED, C_OBS]
        x = xb + (k - 0.5) * 0.32
        ax2.bar(x, d, 0.30, color=cols,
                alpha=0.85 if level == 700 else 0.55,
                hatch=None if level == 700 else "//",
                edgecolor="white", linewidth=0.5)
        ax2.errorbar(x[-1], obs["diff"],
                     yerr=[[obs["diff"] - obs["ci_lo"]], [obs["ci_hi"] - obs["diff"]]],
                     fmt="none", ecolor="#333333", elinewidth=1.0, capsize=2)
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_xticks(xb)
    ax2.set_xticklabels(labels, fontsize=8.5)
    ax2.set_ylabel("theta-e contribution (K)")
    ax2.set_title("The theta-e budget at -72 h")
    panel_label(ax2, "b", 20)
    ax2.grid(alpha=0.25, axis="y")
    ax2.set_ylim(bottom=ax2.get_ylim()[0] - 0.14)
    n7 = one(df, a.tier, 700, "thetae_north_origin")
    n8 = one(df, a.tier, 850, "thetae_north_origin")
    ax2.text(0.98, 0.03,
             "solid 700 hPa, hatched 850 hPa\n"
             "north-origin arm (conditional): "
             f"{n7['diff']:+.2f} K at 700 hPa, {n8['diff']:+.2f} K at 850 hPa",
             transform=ax2.transAxes, fontsize=7.5, va="bottom", ha="right",
             color="#333333")

    fig.suptitle("The airmass fingerprint, 1983-2007", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(a.out, dpi=300)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
