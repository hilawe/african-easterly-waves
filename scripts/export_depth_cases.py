#!/usr/bin/env python3
"""Export real region-growth cases so version 1's own walk can be depth-counted.

WHY. `scripts/measure_region_depth.py` reports how deep `isolate_region_f`'s recursion
goes, and it does so with a Python reimplementation of the walk rather than by running the
MATLAB. Two reviews pointed out, correctly, that the write-ups treated that as a
MEASUREMENT of the original's recursion when it is a simulation of it. The simulation is
only as good as three assumptions:

  1. the eight neighbours are visited in the source's own order
  2. neighbours are TESTED against the caller's `Z` but MARKED in `Zn`, and the child is
     handed `Zn`, so a cell marked by an earlier sibling is invisible to a later one
  3. all eight are appended before any recursion, and the recursion then runs over them
     in that order

Rather than state those as assumptions, this exports the real masks and seeds so the
archived function can be run on them under Octave with its actual call depth recorded, and
the two numbers compared. An assumption that can be tested cheaply should be tested.

WHAT IS EXPORTED, per case: the binary field exactly as `merge_contours_f` would hand it
to `isolate_region_f` (ones above threshold, zeros elsewhere), the seed as a one-based
[row, col], and this simulator's own predicted depth so the comparison needs nothing
recomputed on the Octave side.
"""
import argparse
import os
import sys

import numpy as np
from scipy.io import savemat

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aew.v1port import climatology as clim  # noqa: E402
from aew.v1port import load as L  # noqa: E402
from aew.v1port import pipeline as P  # noqa: E402
from aew.v1port.climo_cache import load_or_build  # noqa: E402
from aew.v1port.detection import MAX_ZONAL_WIND, _prepare, trough_axes  # noqa: E402
from measure_region_depth import walk_depth  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--year", type=int, default=1990)
    ap.add_argument("--start", type=int, default=600)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--cases", type=int, default=12,
                    help="how many cases to export, deepest first plus a spread")
    ap.add_argument("--directory", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "eraint", "v1port_buffered"))
    args = ap.parse_args(argv)
    out_dir = os.environ["AEW_ORACLE_DIR"]

    climo = load_or_build(L.available_years(args.directory, "eraint"),
                          args.directory, "eraint", os.environ["AEW_CLIMO_CACHE"],
                          verbose=False)
    times, latgrid, longrid, u, v = L.load_year(args.year, args.directory, "eraint")
    curvature = P.curvature_from_winds(latgrid, longrid, u, v)
    anomaly = P.anomaly_from_climatology(curvature, times, climo)
    del curvature
    advection = P._advection(latgrid, longrid, u, v, anomaly)
    native = abs(float(latgrid[1, 0] - latgrid[0, 0]))
    coarse = {n: clim.gaussian_decimate(f, native, P.COARSE_RESOLUTION_DEG)
              for n, f in (("u", u), ("anomaly", anomaly), ("advection", advection))}
    del advection
    lat_c, lon_c = P.coarse_grid(latgrid, longrid, native, P.COARSE_RESOLUTION_DEG)
    rows_c, cols_c = P._subset(lat_c, lon_c, P.DOMAIN_LAT, P.DOMAIN_LON)
    rows_f, cols_f = P._subset(latgrid, longrid, P.DOMAIN_LAT, P.DOMAIN_LON)
    LG, LO = lat_c[np.ix_(rows_c, cols_c)], lon_c[np.ix_(rows_c, cols_c)]
    FG, FO = latgrid[np.ix_(rows_f, cols_f)], longrid[np.ix_(rows_f, cols_f)]
    ct, ft = P.thresholds_for("ERA-Int", 700)

    found = []
    for step in range(args.start, args.start + args.steps):
        wind = clim.smooth9(coarse["u"][step][np.ix_(rows_c, cols_c)])
        field = _prepare(coarse["anomaly"][step][np.ix_(rows_c, cols_c)], LG[:, 0])
        adv = clim.smooth9(coarse["advection"][step][np.ix_(rows_c, cols_c)])
        westerly = wind > MAX_ZONAL_WIND
        adv = np.where(westerly, np.nan, adv)
        field = np.where(westerly, np.nan, field)
        adv = np.where(field < ct, np.nan, adv)
        fine = _prepare(clim.smooth9(anomaly[step][np.ix_(rows_f, cols_f)]), FG[:, 0])
        fine_mask = np.nan_to_num(fine, nan=-np.inf) >= ft
        for axis_lat, axis_lon in trough_axes(LG, LO, adv):
            rf = int(np.argmin(np.abs(FG[:, 0] - float(np.mean(axis_lat)))))
            cf = int(np.argmin(np.abs(FO[0, :] - float(np.mean(axis_lon)))))
            found.append((walk_depth(fine_mask, (rf, cf)), step, rf, cf, fine_mask))
        print(f"  step {step}: {len(found)} cases so far", flush=True)

    found.sort(key=lambda x: -x[0])
    # the deepest, plus a spread across the range so agreement is not tested only at the
    # extreme, where a simulator that is wrong in general could still be right by accident
    picked = found[:args.cases // 2]
    rest = found[args.cases // 2:]
    picked += [rest[i] for i in np.linspace(0, len(rest) - 1,
                                            args.cases - len(picked)).astype(int)]

    payload = {"n": float(len(picked))}
    for i, (depth, step, r, c, mask) in enumerate(picked):
        payload[f"mask{i}"] = mask.astype(float)
        # ONE-BASED [row, col], which is what isolate_region_f indexes Z with
        payload[f"seed{i}"] = np.array([r + 1, c + 1], dtype=float)
        payload[f"predicted{i}"] = float(depth)
        payload[f"step{i}"] = float(step)
    path = os.path.join(out_dir, "depth_cases.mat")
    savemat(path, payload, do_compression=True)
    print(f"\nwrote {len(picked)} cases to {path}")
    print(f"  predicted depths: "
          f"{', '.join(str(int(p[0])) for p in picked)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
