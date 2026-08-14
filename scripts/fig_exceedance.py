#!/usr/bin/env python
"""Supplementary exceedance figure, from canonical tables.

The magnitude paragraph argues that a small mean shift in inflow moisture converts into a
substantial change in the probability of crossing the moist threshold, because deep
convection responds nonlinearly to free-tropospheric humidity. This figure shows that
conversion directly: the cumulative distributions of the -72 h along-inflow 700 hPa
relative humidity for the two classes, with the exceedance probabilities at the pooled
70th percentile annotated.

A pure consumer of the canonical deposit: the per-selected-trough along-inflow humidity
(``rh_m72`` in cases_<tier>_<level>.csv) and the canonical exceedance statistics
(``exceedance_threshold``, ``exceedance_p70``, ``exceedance_ratio``). It recomputes
nothing, so it cannot drift from the manuscript's numbers.
"""

import argparse
import os

import numpy as np
import pandas as pd

Q, A = "MCS-quiet", "MCS-active"


def canon(table, tier, level, stat, trel):
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
    ap.add_argument("--out", default="fig_exceedance.png")
    a = ap.parse_args()

    cases = pd.read_csv(f"{a.deposit}/cases_{a.tier}_{a.level}.csv")
    cases = cases[np.isfinite(cases["rh_m72"])]
    nd = np.sort(cases[cases.label == Q]["rh_m72"].values)
    dv = np.sort(cases[cases.label == A]["rh_m72"].values)

    table = (f"{a.deposit}/canonical_registry.csv"
             if os.path.exists(f"{a.deposit}/canonical_registry.csv")
             else f"{a.deposit}/canonical_numbers.csv")
    thr = float(canon(table, a.tier, float(a.level), "exceedance_threshold", -72.0)["diff"])
    exc = canon(table, a.tier, float(a.level), "exceedance_p70", -72.0)
    p_q, p_a = exc["mean_quiet"], exc["mean_active"]
    diff, lo, hi = exc["diff"], exc["ci_lo"], exc["ci_hi"]
    ratio = float(canon(table, a.tier, float(a.level), "exceedance_ratio", -72.0)["diff"])

    print(f"threshold (pooled 70th pctl): {thr:.1f}%; P(exceed) {A} {p_a:.1f}% vs "
          f"{Q} {p_q:.1f}% (ratio {ratio:.2f}); diff {diff:+.1f} [{lo:+.1f}, {hi:+.1f}]")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(nd, np.linspace(0, 1, nd.size), color="tab:blue", label=f"{Q} (n={nd.size})")
    ax.plot(dv, np.linspace(0, 1, dv.size), color="tab:red", label=f"{A} (n={dv.size})")
    ax.axvline(thr, color="k", lw=1, ls="--")
    # Low on the axis, in the clear space below both curves. At the threshold the two
    # cumulative curves sit near 0.65 and 0.75 by construction (they are 1 minus the
    # exceedance probabilities printed above), so a label starting at 0.42 ran straight
    # through them.
    ax.text(thr - 0.7, 0.04, f"pooled 70th pctl ({thr:.1f}%)", fontsize=8,
            rotation=90, ha="right", va="bottom")
    ax.axhline(1 - p_q / 100.0, color="tab:blue", lw=0.8, ls=":")
    ax.axhline(1 - p_a / 100.0, color="tab:red", lw=0.8, ls=":")
    ax.text(0.02, 0.92,
            f"P(exceed): {p_a:.1f}% vs {p_q:.1f}%  (ratio {ratio:.2f})\n"
            f"difference {diff:+.1f} points [{lo:+.1f}, {hi:+.1f}]",
            transform=ax.transAxes, fontsize=9, va="top")
    ax.set_xlabel("700 hPa RH along the inflow, 72 h prior to trough passage (%)")
    ax.set_ylabel("cumulative fraction of troughs")
    ax.set_title("Inflow-moisture distributions, 1983-2007")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
