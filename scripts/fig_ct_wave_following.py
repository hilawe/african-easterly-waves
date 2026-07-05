#!/usr/bin/env python
"""Wave-following composite of CT family GENESIS relative to the moving AEW trough.

Extends the wave-following composite from total MCS presence to convective-system
INITIATION: where ISCCP CT families first appear (genesis) relative to the trough. Uses
the genesis slice with reliable ctime. Compares all genesis vs deep (<200 K core) genesis,
each against a shifted-trough null.
"""

import argparse

import numpy as np

from aew.composites import wave_relative_counts
from aew.data.aewc import load_aewc_troughs
from aew.data.ct import from_ct_genesis

REL_C = np.arange(-30.0, 30.1, 2.0)
LAT_C = np.arange(0.0, 25.1, 2.0)
BAND = (LAT_C >= 5) & (LAT_C <= 15)


def excess(tr, ev, rng, n_null=15):
    counts, n = wave_relative_counts(tr.time, tr.lon, ev.time, ev.lon, ev.lat,
                                     REL_C, LAT_C, time_tol_hours=3.0)
    null = np.empty((n_null,) + counts.shape)
    for i in range(n_null):
        sh = (tr.lon + rng.uniform(-180, 180, tr.lon.size) + 180) % 360 - 180
        null[i], _ = wave_relative_counts(tr.time, sh, ev.time, ev.lon, ev.lat,
                                          REL_C, LAT_C, time_tol_hours=3.0)
    return (counts - null.mean(0))[BAND].mean(0), 2 * null.std(0)[BAND].mean(0), int(counts.sum())


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
    # the full overlap of the CT record (1984-2007) with the AEWC troughs; an earlier
    # 1984-2001 cutoff was a stale development-era restriction the full-record review
    # caught (its counts, 59,929/17,966, were what the draft mistakenly dated 1984-2007)
    ylo, yhi = 1984, 2007
    def win(ev):
        import pandas as pd
        yr = pd.DatetimeIndex(ev.time).year
        return ev.filter((yr >= ylo) & (yr <= yhi))
    allg, deep = win(allg), win(deep)
    print(f"AEWC troughs {len(tr)}; CT genesis all {len(allg)}, deep {len(deep)}")

    rng = np.random.default_rng(0)
    ea, na, ca = excess(tr, allg, rng)
    ed, nd, cd_ = excess(tr, deep, rng)
    print(f"peak genesis excess (5-15N): all {np.max(ea):.0f}, deep {np.max(ed):.0f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.fill_between(REL_C, -na, na, color="grey", alpha=0.2, label="shifted-trough null +/-2 sigma")
    ax.plot(REL_C, ea, color="tab:red",
            label=f"all first cold-cloud detections (n={len(allg)})")
    ax.plot(REL_C, ed, color="tab:purple",
            label=f"deep systems <200 K (n={len(deep)})")
    ax.axvline(0, color="green", lw=2); ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("Longitude relative to trough (deg; east positive)")
    ax.set_ylabel("first-detection excess over null, 5-15N mean")
    ax.set_title("First cold-cloud detection relative to the moving trough (JAS)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(a.out, dpi=150)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
