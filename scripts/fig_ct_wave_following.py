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

One interval is given, added for the revision at a reviewer's request: the difference
between the centroids of the two curves, with whole seasons resampled. The profiles are
sums over seasons, so a replicate is a sum of the seasons it drew, the same seasons for
both curves. The interval describes how far the difference moves from one draw of
seasons to another. It is not a test against a null of the catalog's geometry, which
the paragraph above explains cannot be built. Written to
deposit/fig3_centroid_interval.csv.
"""

import argparse

import numpy as np
import pandas as pd

from aew.composites import wave_relative_counts
from aew.data.aewc import load_aewc_troughs
from aew.data.ct import from_ct_genesis, repeated_rows
import cloud_inputs

REL_C = np.arange(-30.0, 30.1, 2.0)
LAT_C = np.arange(0.0, 25.1, 2.0)
BAND = (LAT_C >= 5) & (LAT_C <= 15)
SEARCH = (-10.0, 10.0)
N_BOOT = 20_000


def centroid(prof):
    """The centroid in relative longitude of a profile, each bin weighted by its share."""
    prof = np.asarray(prof, float)
    return float((REL_C * prof).sum() / prof.sum())


def centroid_difference_interval(prof_all, prof_deep, n_boot=N_BOOT, seed=0):
    """The deep-at-detection centroid less the all-detection centroid, with seasons resampled.

    ``prof_all`` and ``prof_deep`` hold one profile for each season, in the same order. A
    replicate draws seasons with replacement and sums the profiles of both curves over
    the same draw, so the two centroids of a replicate come from one set of seasons.
    """
    a, d = np.asarray(prof_all, float), np.asarray(prof_deep, float)
    if a.shape != d.shape or a.ndim != 2 or a.shape[1] != REL_C.size or a.shape[0] < 2:
        raise SystemExit("REFUSED: the season profiles of the two curves do not share one shape")
    n = a.shape[0]
    point_all, point_deep = centroid(a.sum(axis=0)), centroid(d.sum(axis=0))
    rng = np.random.default_rng([int(seed), 3, 5])
    w = rng.multinomial(n, np.full(n, 1.0 / n), size=int(n_boot)).astype(float)
    ra, rd = w @ a, w @ d
    reps = (rd * REL_C).sum(axis=1) / rd.sum(axis=1) - (ra * REL_C).sum(axis=1) / ra.sum(axis=1)
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return dict(centroid_all=point_all, centroid_deep=point_deep, diff=point_deep - point_all, ci_lo=float(lo),
                ci_hi=float(hi), n_seasons=int(n), n_boot=int(n_boot), significant=bool(not (lo <= 0.0 <= hi)))


def band_prof(tr_time, tr_lon, ev):
    """Return (band-mean profile, total matches in the band).

    The profile is a MEAN over the band's latitude rows, so its sum is the match total
    divided by the number of rows. Both are returned because the normalized shares use
    the profile while the true count is the total (a defect found in checking: the
    deposited n_matches was the mean-scaled value and so was not a match count).
    """
    c, _ = wave_relative_counts(tr_time, tr_lon, ev.time, ev.lon, ev.lat,
                                REL_C, LAT_C, time_tol_hours=3.0)
    return c[BAND].mean(axis=0), float(c[BAND].sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aewc-glob", default="data/aewc/ERA-Int_ew_700hPa_*_AFR.nc")
    ap.add_argument("--ct", default=cloud_inputs.TRACKING_FILE)
    ap.add_argument("--out", default="fig_ct_wave_following.png")
    a = ap.parse_args()
    cloud_inputs.checked(a.ct)
    # families the file lists more than once are counted once (aew.data.ct.repeated_rows)
    n_repeats = int(repeated_rows(a.ct).sum())

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
          f"already deep at first detection {len(deep)}; repeated family rows dropped "
          f"{n_repeats}")

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
                         n_repeats_dropped=n_repeats, n_matches=tot[k],
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

    # the centroid difference with whole seasons resampled. The season profiles must add
    # to the profiles above, which holds because no trough of one season lies within the
    # matching tolerance of a detection of another.
    tr_year = pd.DatetimeIndex(tr.time).year
    by_season = {k: [] for k in events}
    for y in range(ylo, yhi + 1):
        keep = tr_year == y
        for k, ev in events.items():
            one = ev.filter(pd.DatetimeIndex(ev.time).year == y)
            by_season[k].append(band_prof(tr.time[keep], tr.lon[keep], one)[0])
    for k in events:
        if not np.allclose(np.sum(by_season[k], axis=0), obs[k], rtol=0.0, atol=1e-9):
            raise SystemExit(f"REFUSED: the season profiles of {k} do not add to its profile")
    ci = centroid_difference_interval(by_season["all"], by_season["deep_at_detection"])
    for k, name in (("all", "centroid_all"), ("deep_at_detection", "centroid_deep")):
        stated = next(r["centroid_rel_lon"] for r in rows if r["subset"] == k)
        if abs(ci[name] - stated) > 1e-9:
            raise SystemExit(f"REFUSED: the centroid of {k} differs from the one stated above")
    pd.DataFrame([ci]).to_csv("deposit/fig3_centroid_interval.csv", index=False, float_format="%.6f")
    print(f"F3 centroid difference {ci['diff']:+.2f} deg ({ci['ci_lo']:+.2f} to {ci['ci_hi']:+.2f}), "
          f"{ci['n_seasons']} seasons resampled, {ci['n_boot']} replicates")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from aew.plotting import panel_label

    plt.rcParams.update({
        "axes.unicode_minus": False,
        "font.size": 14,
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    })

    lab_all = f"All detections (n={len(allg):,})"
    lab_deep = f"Already deep (<200 K) at detection (n={len(deep):,})"
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(12, 5.8))

    # (a) where first detections fall, in counts. The two curves are NOT comparable
    # for shape here, only for how many detections each set contributes.
    axa.axvline(0, color="green", lw=1.5, zorder=0.5)
    axa.plot(REL_C, obs["all"], color="tab:red", lw=2.0, zorder=3, label=lab_all)
    axa.plot(REL_C, obs["deep_at_detection"], color="tab:purple", lw=2.0,
             ls="--", zorder=3, label=lab_deep)
    axa.set_xlabel("Longitude relative to trough (deg; east positive)")
    axa.set_ylabel("first-detection count, 5-15N mean")
    axa.set_title("Where first detections fall (counts)")
    axa.grid(alpha=0.3)
    panel_label(axa, "a", 15)

    # (b) the shape comparison, each curve normalized by its own total so the
    # born-deep subset's smaller sample does not read as a narrower distribution
    for k, c, lab in (("all", "tab:red", lab_all),
                      ("deep_at_detection", "tab:purple", lab_deep)):
        f = obs[k] / obs[k].sum()
        axb.plot(REL_C, 100.0 * f, color=c, lw=2.0,
                 ls="--" if k == "deep_at_detection" else "-", zorder=3, label=lab)
    axb.axvline(0, color="green", lw=1.5, zorder=0.5)
    axb.set_xlabel("Longitude relative to trough (deg; east positive)")
    axb.set_ylabel("Share of binned matches (% per 2-deg bin)")
    axb.set_title("Each curve normalized by its own binned total")
    axb.grid(alpha=0.3)
    panel_label(axb, "b", 15)

    handles, labels = axb.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               fontsize=14, bbox_to_anchor=(0.5, 0.01))
    fig.tight_layout(rect=(0, 0.10, 1, 1), w_pad=2.0)
    fig.savefig(a.out, dpi=150)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
