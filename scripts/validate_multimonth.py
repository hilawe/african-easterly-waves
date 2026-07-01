#!/usr/bin/env python
"""Multi-month validation of the in-house GridSat tracker vs original ISCCP CS.

For each JAS month in a set of years: track the in-house GridSat systems (cached), bin
in-house / ISCCP CS / Huang onto a common Africa-box grid at a 90 km cut, and compute the
spatial pattern correlation of the in-house and Huang count maps against ISCCP CS. Reports
the distribution of r across months (the robust version of the single July-2000 number).
"""

import argparse
import glob
import os
import time

import numpy as np
import pandas as pd
import xarray as xr

from aew.binning import bin_sum
from aew.data.gridsat_track import regrid_tb, track_systems, to_pyflextrkr_netcdf
from aew.data.huang import load_huang_month
from aew.data.pyflextrkr import from_pyflextrkr
from aew.tracks import Tracks

WEST, EAST, SOUTH, NORTH = -40.0, 40.0, 0.0, 25.0
LON_C = np.arange(WEST, EAST + 1e-6, 2.0)
LAT_C = np.arange(SOUTH, NORTH + 1e-6, 2.0)


def cmap_counts(lon, lat):
    g, _ = bin_sum(LON_C, LAT_C, lon, lat, z=np.ones(lon.size), variant="fixed")
    return g


TRACK_CFG = dict(shield=245.0, core=220.0, factor=4, overlap_thresh=0.1, min_radius=90.0)
CFG_TAG = (f"s{TRACK_CFG['shield']:.0f}c{TRACK_CFG['core']:.0f}f{TRACK_CFG['factor']}"
           f"o{TRACK_CFG['overlap_thresh']}r{TRACK_CFG['min_radius']:.0f}")


def track_month(year, month):
    # config-tagged cache so a file from different tracker settings is never reused blindly
    out = f"data/gridsat/tracks_{year}_{month:02d}_{CFG_TAG}.nc"
    if os.path.exists(out):
        return out
    files = sorted(glob.glob(f"data/gridsat_jas/gridsat_{year}{month:02d}*.nc"))
    if not files:
        return None
    da = xr.concat([xr.open_dataset(f)["irwin_cdr"].load() for f in files],
                   dim="time").sortby("time")
    da = regrid_tb(da, factor=TRACK_CFG["factor"])
    times, tracks = track_systems(da, shield=TRACK_CFG["shield"], core=TRACK_CFG["core"],
                                  min_radius_km=TRACK_CFG["min_radius"],
                                  overlap_thresh=TRACK_CFG["overlap_thresh"])
    to_pyflextrkr_netcdf(times, tracks, out)
    return out


def load_isccp_month(year, month):
    ds = xr.open_dataset("data/original/csct/csct_africa_cs245_csize-ge-90.nc")
    t = pd.DatetimeIndex(ds["time"].values)
    sel = (t.year == year) & (t.month == month)
    lat = np.asarray(ds["lat"].values, float)[sel]
    lon = np.asarray(ds["lon"].values, float)[sel]
    return Tracks(time=t[sel].values, lat=lat, lon=lon, variables={})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="1985,1988,1991,1994")
    ap.add_argument("--out", default="validate_multimonth.png")
    a = ap.parse_args()
    years = [int(y) for y in a.years.split(",")]
    rows = []
    for y in years:
        for m in (7, 8, 9):
            t0 = time.time()
            tp = track_month(y, m)
            if tp is None:
                print(f"{y}-{m:02d}: no GridSat data, skip")
                continue
            gs = from_pyflextrkr(tp, min_radius_km=TRACK_CFG["min_radius"]).filter_region(SOUTH, NORTH, WEST, EAST)
            isccp = load_isccp_month(y, m).filter_region(SOUTH, NORTH, WEST, EAST)
            hu_path = f"data/huang/MCS_record_1985_02_2008-11/MCS_record_{y}-{m:02d}.txt"
            hu = (load_huang_month(hu_path, min_radius_km=90.0)
                  .filter_region(SOUTH, NORTH, WEST, EAST) if os.path.exists(hu_path) else None)
            ref_map = cmap_counts(isccp.lon, isccp.lat)
            ref = ref_map.ravel()
            occ = ref > 0  # cells where ISCCP actually has systems (harder, honest metric)

            def corrs(lon, lat):
                m = cmap_counts(lon, lat).ravel()
                r_all = np.corrcoef(ref, m)[0, 1]
                r_occ = (np.corrcoef(ref[occ], m[occ])[0, 1] if occ.sum() > 2 else np.nan)
                return r_all, r_occ

            r_gs, ro_gs = corrs(gs.lon, gs.lat)
            r_hu, ro_hu = corrs(hu.lon, hu.lat) if hu is not None else (np.nan, np.nan)
            rows.append(dict(ym=f"{y}-{m:02d}", n_isccp=len(isccp), n_gs=len(gs),
                             n_hu=(len(hu) if hu else 0), r_gs=r_gs, ro_gs=ro_gs,
                             r_hu=r_hu, ro_hu=ro_hu))
            print(f"{y}-{m:02d}: ISCCP {len(isccp):5d} | GridSat {len(gs):5d} "
                  f"(r={r_gs:.2f}, occ={ro_gs:.2f}) | Huang {(len(hu) if hu else 0):5d} "
                  f"(r={r_hu:.2f}, occ={ro_hu:.2f})   [{time.time()-t0:.0f}s]")

    df = pd.DataFrame(rows)
    rg, rh = df["r_gs"].to_numpy(), df["r_hu"].to_numpy()
    og, oh = df["ro_gs"].to_numpy(), df["ro_hu"].to_numpy()
    print(f"\n(climatological pattern agreement, JAS Africa, >=90 km; full-grid r and "
          f"occupied-cell r)")
    print(f"In-house GridSat vs ISCCP CS:  r = {np.nanmean(rg):.2f}+/-{np.nanstd(rg):.2f}  "
          f"occ = {np.nanmean(og):.2f}+/-{np.nanstd(og):.2f}  (n={len(rg)})")
    print(f"Huang        vs ISCCP CS:  r = {np.nanmean(rh):.2f}+/-{np.nanstd(rh):.2f}  "
          f"occ = {np.nanmean(oh):.2f}+/-{np.nanstd(oh):.2f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(df))
    ax.plot(x, df["r_gs"], "o-", color="tab:red", label=f"in-house GridSat (mean {np.nanmean(rg):.2f})")
    ax.plot(x, df["r_hu"], "s--", color="tab:blue", label=f"Huang 2018 (mean {np.nanmean(rh):.2f})")
    ax.set_xticks(x); ax.set_xticklabels(df["ym"], rotation=60, fontsize=7)
    ax.set_ylabel("spatial pattern r vs ISCCP CS"); ax.set_ylim(0, 1)
    ax.axhline(np.nanmean(rg), color="tab:red", lw=0.6, alpha=0.5)
    ax.grid(alpha=0.3); ax.legend()
    ax.set_title(f"In-house tracker validation vs original ISCCP CS, JAS, Africa, >=90 km "
                 f"({len(df)} months)")
    fig.tight_layout(); fig.savefig(a.out, dpi=150)
    print("wrote", a.out)
    df.to_csv(a.out.replace(".png", ".csv"), index=False)


if __name__ == "__main__":
    main()
