#!/usr/bin/env python
"""Compose the control-model supplement figure from the deposit tables.

Panel (a): the along-inflow moisture effect (incidence-rate ratio per standard
deviation) as environmental controls are added, showing it fall toward one once
antecedent convection and column water vapour enter. Panel (b): the full standardized
panel, showing antecedent convection as the dominant term and the moisture term
indistinguishable from one. All values from control_model_ladder.csv and
control_model_panel.csv (run scripts/control_model.py first). Draws only.

Writes fig_control_model.png (300 dpi).
"""

import argparse
import os

import numpy as np
import pandas as pd

from aew.plotting import panel_label

LABELS = {"inflow_rh": "inflow moisture", "antecedent": "antecedent convection",
          "amplitude": "wave amplitude", "shear": "shear", "tcwv": "column vapour"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deposit", default="deposit")
    ap.add_argument("--out", default="fig_control_model.png")
    a = ap.parse_args()
    lad = pd.read_csv(os.path.join(a.deposit, "control_model_ladder.csv"))
    pan = pd.read_csv(os.path.join(a.deposit, "control_model_panel.csv"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({
        "axes.unicode_minus": False,
        "font.size": 13.5,
        "axes.titlesize": 13.5,
        "axes.labelsize": 13.5,
        "xtick.labelsize": 13.5,
        "ytick.labelsize": 13.5,
        "legend.fontsize": 13.5,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5.4),
                                   gridspec_kw={"width_ratios": [1.05, 1.0]})

    # panel a: the control ladder for the inflow-moisture IRR
    y = np.arange(len(lad))[::-1]
    ax1.axvline(1.0, color="k", lw=1.2, ls="--")
    for yi, (_, r) in zip(y, lad.iterrows()):
        sig = r.pvalue < 0.05
        col = "#5e3c99" if sig else "#999999"
        ax1.plot([r.irr_lo, r.irr_hi], [yi, yi], color=col, lw=2.2,
                 solid_capstyle="round")
        ax1.plot(r.irr, yi, "o", color=col, ms=7)
    ax1.set_yticks(y)
    ax1.set_yticklabels(lad.step, fontsize=13.5)
    ax1.set_title("Moisture effect by adjustment", fontsize=13.5)
    ax1.text(0.97, 0.96, "(a)", transform=ax1.transAxes, fontsize=15,
             fontweight="bold", va="top", ha="right",
             bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                       alpha=0.75, edgecolor="none"))
    ax1.set_xlabel("inflow-moisture incidence-rate ratio\nper standard deviation",
                   fontsize=13.5)
    ax1.set_ylim(-0.6, len(lad) - 1 + 0.85)
    ax1.grid(alpha=0.25, axis="x")

    # panel b: the full standardized panel
    order = ["antecedent", "amplitude", "tcwv", "inflow_rh", "shear"]
    pan = pan.set_index("predictor").loc[order].reset_index()
    yb = np.arange(len(pan))[::-1]
    ax2.axvline(1.0, color="k", lw=1.2, ls="--")
    for yi, (_, r) in zip(yb, pan.iterrows()):
        sig = r.pvalue < 0.05
        col = "#1b7837" if sig else "#999999"
        ax2.plot([r.irr_lo, r.irr_hi], [yi, yi], color=col, lw=2.2,
                 solid_capstyle="round")
        ax2.plot(r.irr, yi, "o", color=col, ms=7)
    ax2.set_yticks(yb)
    ax2.set_yticklabels([LABELS[p] for p in pan.predictor], fontsize=13.5)
    ax2.set_title("Pooled predictor effects", fontsize=13.5)
    ax2.text(0.97, 0.96, "(b)", transform=ax2.transAxes, fontsize=15,
             fontweight="bold", va="top", ha="right",
             bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                       alpha=0.75, edgecolor="none"))
    ax2.set_xlabel("incidence-rate ratio\nper standard deviation", fontsize=13.5)
    ax2.set_ylim(-0.6, len(pan) - 1 + 0.85)
    ax2.grid(alpha=0.25, axis="x")

    # Every colour the figure uses is named. Panel (a) marks a resolved moisture
    # effect in purple and panel (b) a resolved predictor in green, so a two-entry
    # "coloured vs gray" key left the green unexplained.
    key = [
        Line2D([0], [0], color="#5e3c99", marker="o", lw=2.2,
               label="(a) moisture effect, p < 0.05"),
        Line2D([0], [0], color="#1b7837", marker="o", lw=2.2,
               label="(b) predictor effect, p < 0.05"),
        Line2D([0], [0], color="#999999", marker="o", lw=2.2,
               label="p >= 0.05 (either panel)"),
    ]
    fig.legend(handles=key, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.005), fontsize=13.5)
    fig.tight_layout(rect=(0, 0.12, 1, 0.98))
    fig.savefig(a.out, dpi=300)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
