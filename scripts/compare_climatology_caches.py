#!/usr/bin/env python3
"""Two climatology caches compared over a predeclared longitude range, values and masks.

The eastern pilot builds the reference-period climatology again on the wider grid. The
raw winds are the same on the shared columns, but the curvature's derivative stencil is
masked at the old grid's boundary columns and computed there on the wider grid, so equality
over every shared column is the wrong gate. Instead the two caches are compared over a
STENCIL-SUPPORTED range declared before the comparison (155 W to 53 E for the pilot): on
every cell in that range the keys and counts must be identical, the mask (where the mean
is missing) must be identical, and the values must be identical. The shared columns
outside the range are reported separately as the expected boundary difference, with
their counts of differing cells, and never folded into the gate. No cell is excluded after
the result is seen.

The caches carry no coordinates, so each is paired with the input directory it was built
from, whose first file gives the longitude of every column.

    python3 scripts/compare_climatology_caches.py --a data/climo/climo_era5_1979_2010.npz --a-inputs data/era5/v1port_buffered \\
        --b data/climo/east75E/climo_era5_1979_2010.npz --b-inputs data/era5/pilot_east_75E --lon-range -155 53 [--out <json>]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import exact_tracks as X  # noqa: E402


def longitudes(inputs_dir):
    import netCDF4
    files = sorted(glob.glob(os.path.join(inputs_dir, "*_6h_region.nc")))
    if not files:
        raise SystemExit(f"REFUSED: no input file in {inputs_dir} to read the longitudes from")
    with netCDF4.Dataset(files[0]) as ds:
        lon = ds.variables["longitude"][:] if "longitude" in ds.variables else ds.variables["lon"][:]
        return np.asarray(lon, float)


def compare(a_path, a_lon, b_path, b_lon, lon_range):
    a, b = np.load(a_path), np.load(b_path)
    out = {"a": {"path": a_path, "sha256": X.digest(a_path), "columns": int(a_lon.size)},
           "b": {"path": b_path, "sha256": X.digest(b_path), "columns": int(b_lon.size)},
           "lon_range": list(lon_range), "problems": [], "boundary": {}}
    for k in ("years", "keys", "counts"):
        if not np.array_equal(a[k], b[k]):
            out["problems"].append(f"{k} differ")
    ma, mb = a["mean"], b["mean"]
    if ma.shape[:-1] != mb.shape[:-1] or ma.shape[-1] != a_lon.size or mb.shape[-1] != b_lon.size:
        out["problems"].append(f"shapes {ma.shape} and {mb.shape} do not match the caches' input grids ({a_lon.size} and {b_lon.size} columns)")
        return out
    shared = np.isin(a_lon, b_lon)
    if not shared.all():
        out["problems"].append("the second grid does not hold every column of the first")
        return out
    cols_b = np.array([int(np.where(b_lon == x)[0][0]) for x in a_lon])
    inside = (a_lon >= lon_range[0]) & (a_lon <= lon_range[1])
    out["cells_compared"] = int(inside.sum()) * int(np.prod(ma.shape[:-1]))
    xa, xb = ma[..., inside], mb[..., cols_b[inside]]
    mask_a, mask_b = np.isnan(xa), np.isnan(xb)
    if not np.array_equal(mask_a, mask_b):
        out["problems"].append(f"masks differ inside the range on {int(np.count_nonzero(mask_a != mask_b))} cells")
    both = ~mask_a & ~mask_b
    if not np.array_equal(xa[both], xb[both]):
        d = np.abs(xa[both] - xb[both])
        out["problems"].append(f"values differ inside the range on {int(np.count_nonzero(d))} cells, largest {float(d.max()):.3e}")
    # the shared columns outside the range, reported and never gated
    for j in np.where(~inside)[0]:
        ya, yb = ma[..., j], mb[..., cols_b[j]]
        out["boundary"][f"{a_lon[j]:g}"] = {"mask_cells_differing": int(np.count_nonzero(np.isnan(ya) != np.isnan(yb))),
                                           "value_cells_differing": int(np.count_nonzero(~np.isnan(ya) & ~np.isnan(yb) & (ya != yb)))}
    out["passed"] = not out["problems"]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--a", required=True)
    ap.add_argument("--a-inputs", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--b-inputs", required=True)
    ap.add_argument("--lon-range", nargs=2, type=float, required=True, help="the predeclared stencil-supported range, west then east")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    r = compare(args.a, longitudes(args.a_inputs), args.b, longitudes(args.b_inputs), tuple(args.lon_range))
    for p in r["problems"]:
        print("PROBLEM:", p)
    for lon, entry in r["boundary"].items():
        print(f"boundary column {lon} E, outside the range: {entry}")
    print("PASSED" if r.get("passed") else "NOT PASSED")
    if args.out:
        try:
            X.publish_json(args.out, r, exclusive=True)
        except FileExistsError:
            raise SystemExit(f"REFUSED: {args.out} exists and reports are never overwritten")
    return 0 if r.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
