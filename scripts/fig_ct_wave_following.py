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
    """Return (band-mean profile, total matches in the band).

    The profile is a MEAN over the band's latitude rows, so its sum is the match total
    divided by the number of rows. Both are returned because the normalized shares use
    the profile while the honest count is the total (round 7: the deposited
    n_matches was the mean-scaled value and so was not a match count).
    """
    c, _ = wave_relative_counts(tr_time, tr_lon, ev.time, ev.lon, ev.lat,
                                REL_C, LAT_C, time_tol_hours=3.0)
    return c[BAND].mean(axis=0), float(c[BAND].sum())


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

    prof_tot = {k: band_prof(tr.time, tr.lon, ev) for k, ev in events.items()}
    obs = {k: v[0] for k, v in prof_tot.items()}
    tot = {k: v[1] for k, v in prof_tot.items()}

    # descriptive peaks over the same search interval the R1 figures use, ties west.
    # The born-deep set is a SUBSET of all first detections, so raw counts scale with
    # sample size and cannot be compared for concentration. Each profile is therefore
    # also normalized by its own total, and the shape statistics (peak fraction, the
    # share inside the core interval, the share west of the axis, and the centroid)
    # are what the caption may compare.
    #
    # DENOMINATOR, stated exactly because the first version of this fix got it wrong.
    # prof is a 5-15 N mean of wave_relative_counts, which counts a system once per
    # trough whose window contains it, over the plotted +/-30 degree support. So
    # prof.sum() is a count of binned event-trough matches inside that support, NOT the
    # n_events first detections. The shares below are shares of those matches, and no
    # label may call them shares of first detections.
    sel = (REL_C >= SEARCH[0]) & (REL_C <= SEARCH[1])
    core = (REL_C >= -4.0) & (REL_C <= 2.0)
    west = REL_C < 0.0
    rows = []
    for k, prof in obs.items():
        i = np.flatnonzero(sel)[int(np.argmax(prof[sel]))]
        frac = prof / prof.sum()
        centroid = float((REL_C * frac).sum())
        rows.append(dict(figure="F3", subset=k, n_events=len(events[k]),
                         n_matches=tot[k],
                         bandmean_profile_total=float(prof.sum()),
                         peak_count=prof[i], peak_rel_lon=REL_C[i],
                         peak_frac_pct=100.0 * float(frac[i]),
                         share_core_pct=100.0 * float(frac[core].sum()),
                         share_west_pct=100.0 * float(frac[west].sum()),
                         centroid_rel_lon=centroid))
        print(f"F3 {k:18s}: peak {prof[i]:.0f} at {REL_C[i]:+.0f} deg "
              f"(n={len(events[k])}); normalized peak {100 * frac[i]:.2f}%, "
              f"core[-4,+2] {100 * frac[core].sum():.2f}%, "
              f"west {100 * frac[west].sum():.2f}%, centroid {centroid:+.2f} deg")
    pd.DataFrame(rows).to_csv("deposit/fig3_descriptive_stats.csv",
                              index=False, float_format="%.6f")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from aew.plotting import panel_label

    lab_all = f"all first cold-cloud detections (n={len(allg)})"
    lab_deep = f"already deep (<200 K) at first detection (n={len(deep)})"
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(12, 5))

    # (a) where first detections fall, in counts. The two curves are NOT comparable
    # for shape here, only for how many detections each set contributes.
    axa.plot(REL_C, obs["all"], color="tab:red", label=lab_all)
    axa.plot(REL_C, obs["deep_at_detection"], color="tab:purple", label=lab_deep)
    axa.axvline(0, color="green", lw=2)
    axa.set_xlabel("Longitude relative to trough (deg; east positive)")
    axa.set_ylabel("first-detection count, 5-15N mean")
    axa.set_title("Where first detections fall (counts)")
    axa.legend(fontsize=8)
    axa.grid(alpha=0.3)
    panel_label(axa, "a", 17)

    # (b) the shape comparison, each curve normalized by its own total so the
    # born-deep subset's smaller sample does not read as a narrower distribution
    for k, c, lab in (("all", "tab:red", lab_all),
                      ("deep_at_detection", "tab:purple", lab_deep)):
        f = obs[k] / obs[k].sum()
        axb.plot(REL_C, 100.0 * f, color=c, label=lab)
    axb.axvline(0, color="green", lw=2)
    axb.set_xlabel("Longitude relative to trough (deg; east positive)")
    axb.set_ylabel("share of the set's binned trough matches (% per 2-deg bin)")
    axb.set_title("Each curve normalized by its own binned total")
    axb.legend(fontsize=8)
    axb.grid(alpha=0.3)
    panel_label(axb, "b", 17)

    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
