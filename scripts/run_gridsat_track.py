#!/usr/bin/env python
"""Run the minimal ISCCP-style tracker on a month of downloaded GridSat Tb."""
import argparse, glob, time
import numpy as np, xarray as xr
from aew.data.gridsat_track import regrid_tb, track_systems, to_pyflextrkr_netcdf

ap = argparse.ArgumentParser()
ap.add_argument("--glob", default="data/gridsat/gridsat_*.nc")
ap.add_argument("--factor", type=int, default=4)
ap.add_argument("--min-radius", type=float, default=90.0)
ap.add_argument("--out", default="data/gridsat/tracks_2000_07.nc")
a = ap.parse_args()

t0 = time.time()
files = sorted(glob.glob(a.glob))
print(f"opening {len(files)} frames")
pieces = [xr.open_dataset(f)["irwin_cdr"].load() for f in files]
da = xr.concat(pieces, dim="time").sortby("time")
print(f"raw grid: {dict(da.sizes)}")
da = regrid_tb(da, factor=a.factor)
print(f"regridded: {dict(da.sizes)}  ({float(da.lon[1]-da.lon[0]):.3f} deg)")
print(f"Tb range: {float(da.min()):.1f}..{float(da.max()):.1f} K")

times, tracks = track_systems(da, shield=245.0, core=220.0,
                              min_radius_km=a.min_radius, overlap_thresh=0.1)
durs = np.array([len(p) for p in tracks.values()])
print(f"\nsystems tracked: {len(tracks)} tracks")
print(f"track duration (frames): mean {durs.mean():.1f}, max {durs.max()}, "
      f">=8 frames (24h): {(durs>=8).sum()}")
npts = int(durs.sum())
print(f"total system-time points: {npts}")
to_pyflextrkr_netcdf(times, tracks, a.out)
print(f"wrote {a.out}  in {time.time()-t0:.0f}s")
