#!/usr/bin/env python
"""The cross-reanalysis alignment check repeated on the 0.5 degree regional fields.

scripts/check_track_alignment.py composites the ERA5 700 hPa curvature vorticity against
longitude offset from each trough center of the wave climatology, on the 1.5 degree
global band, and finds its peak at zero offset. The environment boxes and the
trajectories of the paper use the 0.5 degree regional fields, and a reviewer asked on
which grid the check was made. This script makes the same composite on the regional
fields, with the same three summaries: the offset of the composite's peak, the centroid
of its positive lobe (the offsets whose mean exceeds half the peak), and the median over
troughs of the offset of each trough's own maximum.

The composite is accumulated a season at a time, since the regional fields of the whole
record do not fit in memory at once. A trough takes part only when the whole window of
offsets lies inside the regional grid, which does not wrap in longitude.

Before it writes, the script runs the same function on the 1.5 degree global band and
requires the peak, the centroid and the median it obtains there to equal the constants
that the paper already states, so the function is known to be the one of the earlier
check and not a second definition.

A pure consumer of the archived trajectories and the reanalysis winds. Writes
deposit/alignment_region.csv, with one row for each grid, and the composite of each grid
as deposit/alignment_band_profile.csv and deposit/alignment_region_profile.csv, from which
scripts/fig_track_alignment.py draws the supplement figure.
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import xarray as xr

from aew.data.aewc import load_aewc_troughs
from aew.waves import curvature_vorticity

HALF_DEG = 12.0          # the window of offsets, as in the earlier check (8 steps of 1.5 degrees)
CORRIDOR = dict(min_lat=5, max_lat=20, min_lon=-30, max_lon=40)
MONTHS = (7, 8, 9)


def refuse(msg):
    raise SystemExit(f"REFUSED: {msg}")


def load_curv(u_path, v_path):
    """Times, ascending latitudes, longitudes and the curvature vorticity of one pair of files."""
    out = []
    for path, var in ((u_path, "u"), (v_path, "v")):
        ds = xr.open_dataset(path)
        out.append((pd.DatetimeIndex(ds["valid_time"].values), np.asarray(ds["latitude"].values, float),
                    np.asarray(ds["longitude"].values, float), np.asarray(ds[var].squeeze().values, float)))
        ds.close()
    (t, lat, lon, u), (tv, latv, lonv, v) = out
    if not (t.equals(tv) and np.array_equal(lat, latv) and np.array_equal(lon, lonv)):
        refuse(f"{u_path} and {v_path} do not share one grid and one set of times")
    order = np.argsort(t.values)
    t, u, v = t[order], u[order], v[order]
    if lat[0] > lat[-1]:
        lat, u, v = lat[::-1], u[:, ::-1, :], v[:, ::-1, :]
    return t, lat, lon, curvature_vorticity(u, v, lat, lon)


class Composite:
    """The running sums of the offset composite over any number of seasons."""

    def __init__(self, dlon, wrap):
        self.dlon, self.wrap = float(dlon), bool(wrap)
        self.steps = int(round(HALF_DEG / self.dlon))
        if abs(self.steps * self.dlon - HALF_DEG) > 1e-9:
            refuse(f"a grid step of {self.dlon} degrees does not divide the window of {HALF_DEG} degrees")
        self.offsets = np.arange(-self.steps, self.steps + 1)
        self.total = np.zeros(self.offsets.size)
        self.count = np.zeros(self.offsets.size)
        self.argmax, self.matched, self.outside = [], 0, 0

    def add(self, times, lat, lon, crv, tr_time, tr_lat, tr_lon):
        if abs(float(np.median(np.diff(lon))) - self.dlon) > 1e-9:
            refuse("a file's grid step differs from the composite's")
        pos = {t: i for i, t in enumerate(times.values.astype("datetime64[ns]"))}
        for t, la, lo in zip(np.asarray(tr_time).astype("datetime64[ns]"), tr_lat, tr_lon):
            i = pos.get(t)
            if i is None:
                continue
            jlat = int(np.argmin(np.abs(lat - la)))
            jlon = int(np.argmin(np.abs(lon - lo)))
            cols = jlon + self.offsets
            if self.wrap:
                cols = cols % lon.size
            elif cols[0] < 0 or cols[-1] >= lon.size:
                self.outside += 1                # the window leaves a grid that does not wrap
                continue
            row = crv[i, jlat, cols]
            good = np.isfinite(row)
            self.total[good] += row[good]
            self.count[good] += 1
            if good.all():
                self.argmax.append(self.offsets[int(np.argmax(row))] * self.dlon)
            self.matched += 1

    def summary(self):
        if self.matched == 0 or (self.count == 0).any():
            refuse("the composite holds an offset with no trough")
        prof = self.total / self.count
        off = self.offsets * self.dlon
        ipk = int(np.argmax(prof))
        lobe = prof > 0.5 * prof[ipk]
        return dict(grid_step_deg=self.dlon, n_matched=int(self.matched), n_window_outside_grid=int(self.outside),
                    peak_deg=float(off[ipk]), centroid_deg=float(np.sum(off[lobe] * prof[lobe]) / np.sum(prof[lobe])),
                    median_argmax_deg=float(np.median(self.argmax)) if self.argmax else np.nan,
                    n_argmax=int(len(self.argmax))), pd.DataFrame(dict(offset_deg=off, mean_curvature_vorticity=prof, n=self.count))


def composite_of(u_glob, v_glob, tr, wrap):
    us, vs = sorted(glob.glob(u_glob)), sorted(glob.glob(v_glob))
    year = lambda p: re.search(r"_(\d{4})_", os.path.basename(p)).group(1)
    if not us or [year(p) for p in us] != [year(p) for p in vs]:
        refuse(f"{u_glob} and {v_glob} do not name the same seasons")
    comp = None
    for up, vp in zip(us, vs):
        times, lat, lon, crv = load_curv(up, vp)
        if comp is None:
            comp = Composite(float(np.median(np.diff(lon))), wrap)
        comp.add(times, lat, lon, crv, tr.time, tr.lat, tr.lon)
    return comp, len(us)


def constants(path):
    c = pd.read_csv(path)
    m = (c.tier == "validation") & (c.statistic == "val_align_centroid_deg")
    if int(m.sum()) != 1:
        refuse(f"{path} states the alignment centroid {int(m.sum())} times")
    return float(c["diff"][m].iloc[0])


def require_earlier_check(b, want):
    """The summary on the global band must be the one the paper states. Stops on a mismatch."""
    if b["peak_deg"] != 0.0 or b["median_argmax_deg"] != 0.0 or round(b["centroid_deg"], 2) != round(want, 2):
        refuse(f"on the 1.5 degree band the function gives peak {b['peak_deg']}, centroid {b['centroid_deg']:.4f} and median "
               f"{b['median_argmax_deg']}, and the paper states 0.0, {want} and 0.0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aewc-glob", default="data/aewc/ERA-Int_ew_700hPa_*_AFR.nc")
    ap.add_argument("--u-glob", default="data/era5/region6h/era5_u700_*_6h_region.nc")
    ap.add_argument("--v-glob", default="data/era5/region6h/era5_v700_*_6h_region.nc")
    ap.add_argument("--global-u-glob", default="data/era5/global6h/era5_u700_*_6h_global.nc")
    ap.add_argument("--global-v-glob", default="data/era5/global6h/era5_v700_*_6h_global.nc")
    ap.add_argument("--constants", default="scripts/validation_constants.csv")
    ap.add_argument("--deposit", default="deposit")
    a = ap.parse_args()
    tr = load_aewc_troughs(a.aewc_glob).filter_region(**CORRIDOR).filter_months(MONTHS)

    # the function is first bound to the earlier check on the grid that check used
    band, n_band = composite_of(a.global_u_glob, a.global_v_glob, tr, wrap=True)
    b, band_prof = band.summary()
    want = constants(a.constants)
    require_earlier_check(b, want)
    print(f"the function reproduces the earlier check on the {b['grid_step_deg']} degree band: peak {b['peak_deg']:+.1f}, "
          f"centroid {b['centroid_deg']:+.2f}, median {b['median_argmax_deg']:+.1f}, {b['n_matched']:,} troughs, {n_band} seasons")

    region, n_region = composite_of(a.u_glob, a.v_glob, tr, wrap=False)
    r, prof = region.summary()
    if n_region != n_band:
        refuse(f"the regional fields cover {n_region} seasons and the band {n_band}")
    rows = [dict(grid="global band", seasons=n_band, **b), dict(grid="regional", seasons=n_region, **r)]
    pd.DataFrame(rows).to_csv(os.path.join(a.deposit, "alignment_region.csv"), index=False, float_format="%.6f")
    prof.to_csv(os.path.join(a.deposit, "alignment_region_profile.csv"), index=False, float_format="%.9g")
    band_prof.to_csv(os.path.join(a.deposit, "alignment_band_profile.csv"), index=False, float_format="%.9g")
    print(f"on the {r['grid_step_deg']} degree regional fields: peak {r['peak_deg']:+.1f}, centroid {r['centroid_deg']:+.2f}, "
          f"median {r['median_argmax_deg']:+.1f}, {r['n_matched']:,} troughs, {r['n_window_outside_grid']} with a window outside the grid")
    print(f"wrote {a.deposit}/alignment_region.csv and the profile of each grid")


if __name__ == "__main__":
    main()
