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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6),
                                   gridspec_kw={"width_ratios": [1.05, 1.0]})

    # panel a: the control ladder for the inflow-moisture IRR
    y = np.arange(len(lad))[::-1]
    ax1.axvline(1.0, color="k", lw=0.8, ls="--")
    for yi, (_, r) in zip(y, lad.iterrows()):
        sig = r.pvalue < 0.05
        col = "#5e3c99" if sig else "#999999"
        ax1.plot([r.irr_lo, r.irr_hi], [yi, yi], color=col, lw=2.2,
                 solid_capstyle="round")
        ax1.plot(r.irr, yi, "o", color=col, ms=7)
    ax1.set_yticks(y)
    ax1.set_yticklabels(lad.step, fontsize=9)
    ax1.set_xlabel("inflow-moisture incidence-rate ratio per standard deviation")
    ax1.set_title("(a) Moisture effect as controls are added", fontsize=10)
    ax1.grid(alpha=0.25, axis="x")

    # panel b: the full standardized panel
    order = ["antecedent", "amplitude", "tcwv", "inflow_rh", "shear"]
    pan = pan.set_index("predictor").loc[order].reset_index()
    yb = np.arange(len(pan))[::-1]
    ax2.axvline(1.0, color="k", lw=0.8, ls="--")
    for yi, (_, r) in zip(yb, pan.iterrows()):
        sig = r.pvalue < 0.05
        col = "#1b7837" if sig else "#999999"
        ax2.plot([r.irr_lo, r.irr_hi], [yi, yi], color=col, lw=2.2,
                 solid_capstyle="round")
        ax2.plot(r.irr, yi, "o", color=col, ms=7)
    ax2.set_yticks(yb)
    ax2.set_yticklabels([LABELS[p] for p in pan.predictor], fontsize=9)
    ax2.set_xlabel("incidence-rate ratio per standard deviation")
    ax2.set_title("(b) Full standardized panel (pooled)", fontsize=10)
    ax2.grid(alpha=0.25, axis="x")

    fig.suptitle("Regime control model for the moisture-gating claim, 1983-2007",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(a.out, dpi=300)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
