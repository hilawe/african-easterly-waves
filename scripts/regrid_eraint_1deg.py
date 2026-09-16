#!/usr/bin/env python3
"""Regrid the retrieved 0.75 degree ERA-Interim winds onto the 1.0 degree mesh version 1 used.

WHY. Version 1's published wave edges land exactly on whole degrees, 66 distinct values
across the 8,109 observations of one Africa file with a smallest distinct spacing of
1.0, while a mean position in the same file and the same float32 type keeps full
precision. So version 1 ran on a ONE DEGREE grid and this project retrieved ERA-Interim
at its native 0.75. That is not a cosmetic difference: `decimate_f.m` takes the floor of
the resolution ratio, so a one degree input is filtered over two cells and subsampled by
two, giving a 2.0 degree tracking grid, while 0.75 gives width three, stride three and a
2.25 degree grid. Every contour and every merge decision happens on a different mesh.

WHAT THIS IS AND IS NOT. This interpolates the data already on disk. It is NOT the same
as asking the archive for a one degree product, which would interpolate from the
spectral representation instead, and the difference matters most for derivatives, which
is exactly what the tracker computes. So a result here says whether the MESH changes the
outcome, and a confirmed result deserves a real one degree retrieval before it is
published. Stated here so a later reader does not mistake one for the other.

Bilinear and separable, which is what a regular target grid on a regular source grid
admits. The target grid stays strictly inside the source so nothing is extrapolated.

    .venv/bin/python scripts/regrid_eraint_1deg.py
    .venv/bin/python scripts/regrid_eraint_1deg.py --years 1990 --out data/eraint/v1port_1deg
"""

import argparse
import glob
import os
import sys

import numpy as np

TIME_CHUNK = 200          # keeps the interpolation's working array to a few hundred MB


def weights(source, target):
    """Bilinear weights mapping `source` coordinates onto `target`, as a dense matrix.

    Refuses to extrapolate rather than silently clamping, because a clamped edge would
    duplicate the boundary row and the vorticity stencil reads two rows in from it.
    """
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    ascending = source[1] > source[0]
    ordered = source if ascending else source[::-1]
    if target.min() < ordered.min() - 1e-9 or target.max() > ordered.max() + 1e-9:
        raise ValueError(
            f"the target range [{target.min()}, {target.max()}] leaves the source range "
            f"[{ordered.min()}, {ordered.max()}], which would extrapolate")
    right = np.clip(np.searchsorted(ordered, target), 1, ordered.size - 1)
    left = right - 1
    span = ordered[right] - ordered[left]
    frac = np.where(span == 0, 0.0, (target - ordered[left]) / np.where(span == 0, 1, span))
    matrix = np.zeros((target.size, source.size))
    rows = np.arange(target.size)
    if ascending:
        matrix[rows, left] = 1.0 - frac
        matrix[rows, right] += frac
    else:
        matrix[rows, source.size - 1 - left] = 1.0 - frac
        matrix[rows, source.size - 1 - right] += frac
    return matrix


def regrid_file(path, out_path, target_lat, target_lon, variable):
    import netCDF4 as nc

    with nc.Dataset(path) as ds:
        lat = np.asarray(ds.variables["latitude"][:], dtype=float)
        lon = np.asarray(ds.variables["longitude"][:], dtype=float)
        w_lat = weights(lat, target_lat)
        w_lon = weights(lon, target_lon)
        var = ds.variables[variable]
        n_time = ds.dimensions["valid_time"].size
        out = np.empty((n_time, target_lat.size, target_lon.size), dtype=np.float32)
        for start in range(0, n_time, TIME_CHUNK):
            stop = min(start + TIME_CHUNK, n_time)
            block = np.asarray(var[start:stop], dtype=float)
            while block.ndim > 3:
                axis = next(i for i, n in enumerate(block.shape[1:], start=1) if n == 1)
                block = np.squeeze(block, axis=axis)
            block = np.einsum("ij,tjk->tik", w_lat, block)
            out[start:stop] = np.einsum("tik,lk->til", block, w_lon).astype(np.float32)
        time_values = np.asarray(ds.variables["valid_time"][:])
        time_units = ds.variables["valid_time"].units
        var_units = getattr(var, "units", "m s**-1")

    with nc.Dataset(out_path, "w") as out_ds:
        out_ds.createDimension("valid_time", time_values.size)
        out_ds.createDimension("latitude", target_lat.size)
        out_ds.createDimension("longitude", target_lon.size)
        t = out_ds.createVariable("valid_time", "i8", ("valid_time",))
        t.units = time_units
        t[:] = time_values
        la = out_ds.createVariable("latitude", "f8", ("latitude",))
        la.units = "degrees_north"
        la[:] = target_lat
        lo = out_ds.createVariable("longitude", "f8", ("longitude",))
        lo.units = "degrees_east"
        lo[:] = target_lon
        v = out_ds.createVariable(variable, "f4",
                                  ("valid_time", "latitude", "longitude"), zlib=True)
        v.units = var_units
        v[:] = out
        out_ds.source_note = ("bilinear regrid of the 0.75 degree retrieval onto the "
                              "1.0 degree mesh version 1 used; NOT an archive-side "
                              "1.0 degree product")
    return out.shape


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--directory", default="data/eraint/v1port")
    ap.add_argument("--out", default="data/eraint/v1port_1deg")
    ap.add_argument("--prefix", default="eraint")
    ap.add_argument("--years", nargs="+", type=int, default=None)
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    years = args.years
    if years is None:
        years = sorted({int(os.path.basename(p).split("_")[2]) for p in
                        glob.glob(os.path.join(args.directory,
                                               f"{args.prefix}_u700_*_6h_region.nc"))})
    # The target mesh: whole degrees, strictly inside the 0.75 degree retrieval's span
    # of -35 to 34.75 in latitude and -140 to 40 in longitude.
    target_lat = np.arange(-35.0, 34.5, 1.0)
    target_lon = np.arange(-140.0, 40.5, 1.0)
    print(f"target grid {target_lat.size} by {target_lon.size}, "
          f"latitude {target_lat[0]} to {target_lat[-1]}, "
          f"longitude {target_lon[0]} to {target_lon[-1]}")
    print(f"{len(years)} years to regrid", flush=True)

    for n, year in enumerate(years, start=1):
        for var in ("u700", "v700"):
            name = f"{args.prefix}_{var}_{year}_6h_region.nc"
            source = os.path.join(args.directory, name)
            target = os.path.join(args.out, name)
            if not os.path.exists(source):
                raise SystemExit(f"{source} is missing")
            if os.path.exists(target):
                continue
            shape = regrid_file(source, target, target_lat, target_lon, var[0])
            print(f"  {year} {var}: {shape}", flush=True)
        print(f"  ({n}/{len(years)})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
