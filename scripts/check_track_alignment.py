#!/usr/bin/env python
"""Cross-reanalysis mapping check: AEWC (ERA-Interim) trough centers vs ERA5 vorticity.

The wave tracks come from the ERA-Interim-based AEW climatology while the environment and
lead-lag fields are ERA5, so a referee can ask whether the ERA-Interim track centers land
where ERA5 actually puts the trough. This composites the ERA5 700 hPa curvature vorticity
as a function of longitude offset from each AEWC trough center (at the trough's own
latitude row and time). If the two reanalyses agree, the composite peaks at zero offset;
a systematic displacement would bias every ERA5 box mean sampled at AEWC centers.

Inputs: AEWC trajectories (data/aewc), ERA5 u/v 700 6-hourly global (data/era5/global6h).
Prints the offset profile, the peak offset, and the profile centroid near the peak, and
writes the supplement figure (the alignment profile with the centroid annotated). The
default globs take every year on disk, which is the pooled record.
"""

import argparse
import glob

import numpy as np
import pandas as pd
import xarray as xr

from aew.data.aewc import load_aewc_troughs
from aew.waves import curvature_vorticity

OFF_STEPS = 8          # +/- 8 grid steps of 1.5 deg = +/- 12 deg


def load_era5_curv(u_glob, v_glob):
    def _concat(paths, var):
        ts, lat, lon, blocks = [], None, None, []
        for p in sorted(glob.glob(paths)):
            ds = xr.open_dataset(p)
            ts.append(pd.DatetimeIndex(ds["valid_time"].values))
            lat = np.asarray(ds["latitude"].values, float)
            lon = np.asarray(ds["longitude"].values, float)
            blocks.append(np.asarray(ds[var].squeeze().values, float))
            ds.close()
        t = pd.DatetimeIndex(np.concatenate([x.values for x in ts]))
        arr = np.concatenate(blocks, axis=0)
        o = np.argsort(t.values)
        return t[o], lat, lon, arr[o]

    times, lat, lon, u = _concat(u_glob, "u")
    tv, latv, lonv, v = _concat(v_glob, "v")
    if not (times.equals(tv) and np.array_equal(lat, latv) and np.array_equal(lon, lonv)):
        raise ValueError("ERA5 u and v files do not share the same time/lat/lon grid")
    if lat[0] > lat[-1]:
        lat = lat[::-1]; u = u[:, ::-1, :]; v = v[:, ::-1, :]
    return times, lat, lon, curvature_vorticity(u, v, lat, lon)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aewc-glob", default="data/aewc/ERA-Int_ew_700hPa_*_AFR.nc")
    ap.add_argument("--u-glob", default="data/era5/global6h/era5_u700_*_6h_global.nc")
    ap.add_argument("--v-glob", default="data/era5/global6h/era5_v700_*_6h_global.nc")
    ap.add_argument("--out", default="fig_track_alignment.png")
    a = ap.parse_args()

    tr = (load_aewc_troughs(a.aewc_glob)
          .filter_region(min_lat=5, max_lat=20, min_lon=-30, max_lon=40)
          .filter_months([7, 8, 9]))
    print(f"AEWC JAS corridor trough observations: {len(tr)}")

    times, lat, lon, crv = load_era5_curv(a.u_glob, a.v_glob)
    tpos = {t: i for i, t in enumerate(times.values.astype("datetime64[ns]"))}
    dlon = float(np.median(np.diff(lon)))

    offsets = np.arange(-OFF_STEPS, OFF_STEPS + 1)
    prof_sum = np.zeros(offsets.size)
    prof_n = np.zeros(offsets.size)
    argmax_off = []
    n_matched = 0
    for t, la, lo_ in zip(tr.time.astype("datetime64[ns]"), tr.lat, tr.lon):
        i = tpos.get(t)
        if i is None:
            continue
        jlat = int(np.argmin(np.abs(lat - la)))
        jlon = int(np.argmin(np.abs(lon - lo_)))
        cols = (jlon + offsets) % lon.size
        row = crv[i, jlat, cols]
        good = np.isfinite(row)
        prof_sum[good] += row[good]
        prof_n[good] += 1
        if good.all():
            argmax_off.append(offsets[int(np.argmax(row))] * dlon)
        n_matched += 1
    print(f"matched to an ERA5 time step: {n_matched}")

    prof = prof_sum / np.maximum(prof_n, 1)
    off_deg = offsets * dlon
    print("\noffset(deg)  mean ERA5 curvature vorticity (1e-6/s)")
    for o, p in zip(off_deg, prof * 1e6):
        marker = "  <-- AEWC center" if o == 0 else ""
        print(f"{o:+8.1f}   {p:10.2f}{marker}")

    ipk = int(np.argmax(prof))
    # centroid of the positive lobe around the peak (sub-grid estimate of the mean offset)
    lobe = prof > 0.5 * prof[ipk]
    centroid = float(np.sum(off_deg[lobe] * prof[lobe]) / np.sum(prof[lobe]))
    med_argmax = float(np.median(argmax_off)) if argmax_off else np.nan
    print(f"\ncomposite peak at {off_deg[ipk]:+.1f} deg; positive-lobe centroid "
          f"{centroid:+.2f} deg; median per-trough argmax offset {med_argmax:+.1f} deg "
          f"(grid step {dlon:.1f} deg)")
    print("A composite peak and centroid within one grid step of zero means the "
          "ERA-Interim-based AEWC centers coincide with the ERA5 curvature-vorticity trough, "
          "so ERA5 box means sampled at AEWC centers carry no systematic cross-reanalysis "
          "displacement at this resolution.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.plot(off_deg, prof * 1e6, "o-", color="#5e3c99")
    ax.axvline(centroid, color="#e08214", lw=1.2)
    ax.text(0.02, 0.97,
            f"peak {off_deg[ipk]:+.1f} deg\n"
            f"positive-lobe centroid {centroid:+.2f} deg\n"
            f"median per-trough argmax {med_argmax:+.1f} deg",
            transform=ax.transAxes, fontsize=8.5, va="top", color="#333333")
    ax.set_xlabel("longitude offset from the AEWC trough center (deg)")
    ax.set_ylabel("mean ERA5 700 hPa curvature vorticity (1e-6 s$^{-1}$)")
    ax.set_title(f"Cross-reanalysis alignment ({n_matched} trough observations)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(a.out, dpi=300)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
