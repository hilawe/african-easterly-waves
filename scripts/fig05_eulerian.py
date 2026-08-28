#!/usr/bin/env python
"""Figure 5, the Eulerian (fixed-box) 700 hPa moisture composite, from canonical tables.

A pure consumer of the canonical deposit: the per-selected-trough fixed-box relative
humidity 24 h before passage (``box_rh_m24`` in cases_<tier>_<level>.csv) and the
canonical ``eulerian_box`` contrast (canonical_numbers.csv / canonical_registry.csv).
It recomputes nothing, so the figure cannot drift from the numbers the manuscript cites.
The legacy scripts/fig_developing.py (which recomputed from raw fields with the
pre-repair pipeline) is kept only for exploratory --env modes and is no longer the
figure source.
"""

import argparse
import os

import numpy as np
import pandas as pd

EDGES = np.arange(-30, 41, 10.0)
CENTERS = 0.5 * (EDGES[:-1] + EDGES[1:])
Q, A = "MCS-quiet", "MCS-active"
C_QUIET, C_ACTIVE = "tab:blue", "tab:red"


def canonical_row(table, tier, level, stat, trel):
    d = pd.read_csv(table)
    m = ((d.tier == tier) & (d.statistic == stat)
         & (np.isclose(d.level.fillna(-1), level))
         & (np.isclose(d.time_rel_h.fillna(-999), trel)))
    if not m.any():
        raise SystemExit(f"{stat} not found in {table}")
    return d[m].iloc[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deposit", default="deposit")
    ap.add_argument("--tier", default="pooled")
    ap.add_argument("--level", type=int, default=700)
    ap.add_argument("--out", default="fig05_eulerian.png")
    a = ap.parse_args()

    cases = pd.read_csv(f"{a.deposit}/cases_{a.tier}_{a.level}.csv")
    cases = cases[np.isfinite(cases["box_rh_m24"])]
    q = cases[cases.label == Q]["box_rh_m24"].values
    ac = cases[cases.label == A]["box_rh_m24"].values

    table = (f"{a.deposit}/canonical_registry.csv"
             if os.path.exists(f"{a.deposit}/canonical_registry.csv")
             else f"{a.deposit}/canonical_numbers.csv")
    r = canonical_row(table, a.tier, float(a.level), "eulerian_box", -24.0)
    diff, lo, hi = r["diff"], r["ci_lo"], r["ci_hi"]
    sig = "significant" if not (lo <= 0 <= hi) else "not significant"

    # per-longitude-bin class means (display only; the contrast above is the estimand)
    lonbin = pd.cut(cases.lon, EDGES, right=False, labels=False)
    qmean, amean, reliable = [], [], []
    for b in range(len(CENTERS)):
        qb = cases[(lonbin == b) & (cases.label == Q)]["box_rh_m24"]
        ab = cases[(lonbin == b) & (cases.label == A)]["box_rh_m24"]
        qmean.append(qb.mean()); amean.append(ab.mean())
        reliable.append(len(qb) >= 15 and len(ab) >= 15)
    qmean, amean, reliable = np.array(qmean), np.array(amean), np.array(reliable)

    print(f"eulerian_box {a.level} hPa: {A} {ac.mean():.1f} vs {Q} {q.mean():.1f}, "
          f"diff {diff:+.2f} [{lo:+.2f}, {hi:+.2f}] ({sig}); "
          f"n {len(q)}/{len(ac)}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from aew.plotting import panel_label
    plt.rcParams.update({
        "axes.unicode_minus": False,
        "font.size": 15,
        "axes.titlesize": 16,
        "axes.labelsize": 15,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 15,
    })
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13, 5.8), gridspec_kw={"width_ratios": [0.75, 1.25]})

    parts = ax1.violinplot([q, ac], positions=[0, 1], showmeans=True, showextrema=False)
    for pc, col in zip(parts["bodies"], (C_QUIET, C_ACTIVE)):
        pc.set_facecolor(col); pc.set_alpha(0.5)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels([f"{Q}\n(n={len(q):,})", f"{A}\n(n={len(ac):,})"])
    ax1.set_ylabel(f"Pre-trough ERA5 {a.level} hPa RH (%)")
    ax1.set_title("Pre-trough relative humidity by MCS class")
    ax1.grid(alpha=0.3, axis="y")

    ax2.plot(CENTERS[reliable], qmean[reliable], "s--", color=C_QUIET, lw=2.2,
             ms=7, mfc="white", mec=C_QUIET, mew=1.5, label=Q)
    ax2.plot(CENTERS[reliable], amean[reliable], "o-", color=C_ACTIVE, lw=2.2,
             ms=7, label=A)
    if (~reliable).any():
        ax2.plot(CENTERS[~reliable], qmean[~reliable], "x", color=C_QUIET, ms=7,
                 mew=1.5, alpha=0.55)
        ax2.plot(CENTERS[~reliable], amean[~reliable], "x", color=C_ACTIVE, ms=7,
                 mew=1.5, alpha=0.55)
        ax2.plot([], [], "x", color="grey", ms=7, mew=1.5,
                 label="Under-sampled (<15/group)")
    ax2.set_xlabel("longitude bin (deg E)")
    ax2.set_ylabel(f"Pre-trough ERA5 {a.level} hPa RH (%)")
    ax2.set_title(f"{A} vs {Q} by longitude")
    panel_label(ax1, "a", 16)
    panel_label(ax2, "b", 16)
    ax2.legend(fontsize=15)
    ax2.grid(alpha=0.3)

    fig.tight_layout(); fig.savefig(a.out, dpi=150)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
