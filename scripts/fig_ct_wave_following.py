#!/usr/bin/env python
"""First cold-cloud detections relative to the moving AEW trough (CT family record).

Two DESCRIPTIVE curves: all first detections, and families ALREADY COLDER THAN 200 K AT
FIRST DETECTION (REPAIR_SPEC.md R5 relabel: both curves plot first-detection positions,
so the second is "born already deep", not "deepens here"; the lifecycle transition
estimand is a separate analysis).

No randomization inference is reported for this figure (REPAIR_SPEC.md, 2026-07-24
note). The R1 support assertion is unsatisfiable for the regional CT catalog: its
longitude domain ([-55, +54.7]) cannot contain the +/-30-degree relative bins for
troughs spanning [-40, +40], for the observed configuration as much as for any placed
one, and within the +/-10-degree search interval the anchor-permutation null loses a
mean 2.4 percent (max 3.2) of placements to the domain edge while the observed
configuration loses none, which biases observed-minus-null upward. Per the R5
precedent the claim is dropped rather than approximated; Fig. 2 (global storm catalog,
full support everywhere) carries the collocation inference.
"""

import argparse

import numpy as np
import pandas as pd

from aew.composites import wave_relative_counts
from aew.data.aewc import load_aewc_troughs
from aew.data.ct import from_ct_genesis

REL_C = np.arange(-30.0, 30.1, 2.0)
LAT_C = np.arange(0.0, 25.1, 2.0)
BAND = (LAT_C >= 5) & (LAT_C <= 15)
SEARCH = (-10.0, 10.0)


def band_prof(tr_time, tr_lon, ev):
    c, _ = wave_relative_counts(tr_time, tr_lon, ev.time, ev.lon, ev.lat,
                                REL_C, LAT_C, time_tol_hours=3.0)
    return c[BAND].mean(axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aewc-glob", default="data/aewc/ERA-Int_ew_700hPa_*_AFR.nc")
    ap.add_argument("--ct", default="data/original/CT/ct_genesis_time.nc")
    ap.add_argument("--out", default="fig_ct_wave_following.png")
    a = ap.parse_args()

    tr = (load_aewc_troughs(a.aewc_glob)
          .filter_region(min_lat=5, max_lat=20, min_lon=-40, max_lon=40)
          .filter_months([7, 8, 9]))
    allg = from_ct_genesis(a.ct)
    deep = from_ct_genesis(a.ct, deep_core_K=200.0)
    # the full overlap of the CT record (1984-2007) with the AEWC troughs
    ylo, yhi = 1984, 2007

    def win(ev):
        yr = pd.DatetimeIndex(ev.time).year
        return ev.filter((yr >= ylo) & (yr <= yhi))

    allg, deep = win(allg), win(deep)
    events = {"all": allg, "deep_at_detection": deep}
    print(f"AEWC troughs {len(tr)}; CT first detections all {len(allg)}, "
          f"already deep at first detection {len(deep)}")

    obs = {k: band_prof(tr.time, tr.lon, ev) for k, ev in events.items()}

    # descriptive peaks over the same search interval the R1 figures use, ties west
    sel = (REL_C >= SEARCH[0]) & (REL_C <= SEARCH[1])
    rows = []
    for k, prof in obs.items():
        i = np.flatnonzero(sel)[int(np.argmax(prof[sel]))]
        rows.append(dict(figure="F3", subset=k, n_events=len(events[k]),
                         peak_count=prof[i], peak_rel_lon=REL_C[i]))
        print(f"F3 {k:18s}: peak {prof[i]:.0f} at {REL_C[i]:+.0f} deg "
              f"(descriptive; n={len(events[k])})")
    pd.DataFrame(rows).to_csv("deposit/fig3_descriptive_stats.csv",
                              index=False, float_format="%.6f")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(REL_C, obs["all"], color="tab:red",
            label=f"all first cold-cloud detections (n={len(allg)})")
    ax.plot(REL_C, obs["deep_at_detection"], color="tab:purple",
            label=f"already deep (<200 K) at first detection (n={len(deep)})")
    ax.axvline(0, color="green", lw=2)
    ax.set_xlabel("Longitude relative to trough (deg; east positive)")
    ax.set_ylabel("first-detection count, 5-15N mean")
    ax.set_title("First cold-cloud detection relative to the moving trough "
                 "(JAS; descriptive)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
