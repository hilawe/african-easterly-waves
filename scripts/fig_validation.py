#!/usr/bin/env python
"""Compose figure F1, the pipeline-validation triptych, from cached panel data.

Panel (a): the published wave-relative composite reproduced from the original inputs
(272 base dates from the 3.26136 m/s threshold at 10 N, 0 E), drawn with the same
Hovmoller renderer as the standalone reproduction. Panel (b): the ERA5 wave series
against the original ERA-Interim series at the basepoint (r = 0.92, 2000-2004).
Panel (c): the open GridSat-B1 tracker against the legacy ISCCP record across twelve
JAS months (full-grid and occupied-cell pattern correlations, with Huang et al. 2018
as the published comparison).

Inputs are the caches written by the three validation scripts:
  scripts/fig2_real.py           --save-npz figdata/f1a_composite.npz
  scripts/wk_independence.py     --save-npz figdata/f1b_series.npz
  scripts/validate_multimonth.py --out figdata/f1c_multimonth.png  (its CSV sibling)

Writes fig_validation.png (300 dpi).
"""

import argparse
import os

import numpy as np
import pandas as pd

from aew.plotting import hovmoller, panel_label

CHAR = "#333333"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figdata", default="figdata")
    ap.add_argument("--out", default="fig_validation.png")
    a = ap.parse_args()
    pa = np.load(os.path.join(a.figdata, "f1a_composite.npz"))
    pb = np.load(os.path.join(a.figdata, "f1b_series.npz"))
    pc = pd.read_csv(os.path.join(a.figdata, "f1c_multimonth.csv"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12.5, 6.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.25], hspace=0.42, wspace=0.22)
    axa = fig.add_subplot(gs[:, 0])
    axb = fig.add_subplot(gs[0, 1])
    axc = fig.add_subplot(gs[1, 1])

    # ---- (a) the published composite, reproduced ----
    hovmoller(pa["shaded"], pa["lon_centers"], pa["lag"], contour=pa["contour"],
              contour_lon=pa["contour_lon"], contour_lag=pa["contour_lag"],
              base_lon=float(pa["base_lon"]), lon_range=(-40, 80),
              title=None, shaded_label="MCS count anomaly", ax=axa)
    axa.set_title("Published composite reproduced\n"
                  f"{int(pa['n_dates'])} base dates, threshold "
                  f"{float(pa['thr']):.5f} m/s", fontsize=10)
    panel_label(axa, "a", 20)

    # ---- (b) the wave series rebuilt from public ERA5 ----
    # display window: the first JAS season (the r value is over the full record)
    t = pd.DatetimeIndex(pb["common"])
    seg = (t >= "2000-07-01") & (t < "2000-10-01")
    axb.plot(t[seg], pb["td"][seg], color="tab:blue", lw=0.9, alpha=0.8,
             label="original ERA-Interim series")
    axb.plot(t[seg], pb["era5"][seg], color="tab:red", lw=0.9,
             label="rebuilt from public ERA5")
    axb.set_ylabel("filtered v700 (m/s)")
    axb.legend(fontsize=7.5, loc="upper right")
    axb.grid(alpha=0.3)
    import matplotlib.dates as mdates
    axb.xaxis.set_major_locator(mdates.MonthLocator())
    axb.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axb.tick_params(axis="x", labelsize=7.5)
    axb.set_title(f"Wave series from public data, full-record "
                  f"r = {float(pb['r']):.2f} (n = {int(pb['n'])}; JAS 2000 shown)",
                  fontsize=10)
    panel_label(axb, "b", 20)

    # ---- (c) the open tracker against the legacy record, 12 JAS months ----
    x = np.arange(len(pc))
    rg, og = pc["r_gs"].to_numpy(), pc["ro_gs"].to_numpy()
    rh = pc["r_hu"].to_numpy()
    axc.plot(x, rg, "o-", color="tab:red",
             label=f"open tracker, full grid (mean {np.nanmean(rg):.2f})")
    axc.plot(x, og, "o--", color="tab:red", alpha=0.55,
             label=f"open tracker, occupied cells (mean {np.nanmean(og):.2f})")
    axc.plot(x, rh, "s:", color="tab:blue", alpha=0.8,
             label=f"Huang et al. (2018), full grid (mean {np.nanmean(rh):.2f})")
    axc.set_xticks(x)
    axc.set_xticklabels(pc["ym"], rotation=55, fontsize=7)
    axc.set_ylabel("pattern r vs legacy record")
    axc.set_ylim(0.55, 1.0)
    axc.legend(fontsize=7.5, loc="lower left")
    axc.grid(alpha=0.3)
    axc.set_title("Open GridSat-B1 tracker vs legacy ISCCP record", fontsize=10)
    panel_label(axc, "c", 20)

    fig.suptitle("Pipeline validation", fontsize=12, color=CHAR)
    fig.savefig(a.out, dpi=300, bbox_inches="tight")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
