#!/usr/bin/env python3
"""Does striding a half-degree retrieval give what a direct one-degree request gives?

WHY THIS EXISTS. The project puts ERA5 on version 1's whole degrees by taking every second
point of a half-degree retrieval, and its own text asserted that this "selects the
coordinates a one degree request returns rather than interpolating". A review pointed out
that the equivalence had never been checked, and it was right. What ECMWF actually
documents is that a NetCDF request is interpolated from the archived grids to whatever
regular grid is asked for, bilinearly for continuous parameters, and that coarser output
needs care about aliasing. It documents no filtering.

WHAT THE DOCUMENTATION MAKES LIKELY, AND WHY THAT IS NOT ENOUGH. Bilinear interpolation is
local and pointwise, so a target coordinate present in both requests draws on the same
native neighbours either way, and the two routes should agree to rounding. That is a
reasoned expectation from a documented method, and that grade of evidence has failed twice
here already. The decimation window and the convex-hull starting vertex were both obviously
fine from the documentation and both wrong in practice. So this runs the comparison instead
of arguing it.

WHAT IT CANNOT SETTLE. Neither route filters, so both carry the roughly 31 km ERA5 field
aliased onto whole degrees. Version 1's ERA-Interim came from a T255 model at roughly 80 km,
so ERA5 on the same nominal grid holds variance at scales ERA-Interim never resolved. If the
two routes agree perfectly, that difference is untouched, and it stays a live candidate for
why the two reanalyses give different thresholds. Agreement here is a narrow result about
retrieval bookkeeping, not a licence to treat ERA5 at one degree as ERA-Interim at one
degree.

WHAT IT NEEDS, STATED PRECISELY. A DIRECT one-degree ERA5 retrieval, meaning one the
Climate Data Store returned on that grid. One-degree files do exist here and none of them
qualifies: `data/aewc_v2_pilot/qtrack_run/era5_uv700_2004_1deg.nc` was produced by local
linear interpolation from 1.5 degrees in `scripts/aewc_v2_qtrack_probe.py`, and
`data/eraint/v1port_1deg` is ERA-Interim rather than ERA5. An earlier version of this
docstring said nothing on disk is one degree, which was simply false. Interpolating locally
cannot settle the question, because what is being tested is what ECMWF's own interpolation
does.

WHAT IT CHECKS, AND WHAT IT DOES NOT. It compares winds and curvature on shared coordinates
and shared timestamps, and it REFUSES a pair whose grids do not cover the same ground or
carry the same timestamps. It does NOT compare the climatological anomaly or the resulting
detections. PROJECT_PLAN.md at one point described a check that included both; this is the
narrower thing that actually exists, and the plan should say so.

    .venv/bin/python scripts/compare_grid_routes.py \
        --strided-dir data/era5/v1port --direct-dir data/era5/v1port_1deg --year 1981
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from aew.v1port import load as L  # noqa: E402
from aew.v1port.pipeline import curvature_from_winds  # noqa: E402

# What counts as agreement. Bilinear interpolation of the same native field to the same
# coordinate should reproduce to storage precision, and these files are float32, so a
# relative difference at the 1e-6 level is rounding and anything larger is a real
# difference in what was retrieved.
ROUNDING = 1e-6


def _shared_grid(lat_a, lon_a, lat_b, lon_b):
    """Indices of the coordinates the two grids have in common, on each side.

    Matching on VALUES rather than assuming a stride lines up. A half-degree grid strided by
    two and a one-degree grid can differ in phase (whether they start on a whole or a half
    degree) and in extent, and a phase difference is one of the things worth catching.
    """
    def index(values, targets):
        table = {round(float(v), 4): i for i, v in enumerate(values)}
        pairs = [(table[round(float(t), 4)], j)
                 for j, t in enumerate(targets) if round(float(t), 4) in table]
        return np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])

    rows_a, rows_b = index(lat_a, lat_b)
    cols_a, cols_b = index(lon_a, lon_b)
    return rows_a, cols_a, rows_b, cols_b


def coverage_mismatches(shape_s, shape_d, shared_rows, shared_cols,
                        n_times_s, n_times_d, n_shared_times):
    """Every way the two retrievals fail to cover the same ground and the same times.

    COVERAGE IS PART OF THE ANSWER, NOT A FOOTNOTE, and this function exists because two
    earlier versions got that wrong in the same way. The first printed a note when the grids
    differed in extent and then compared the overlap and reported agreement, so a direct tree
    missing a row and a column passed. The second fixed the grid but tested timestamps as
    `n_shared != min(n_times_s, n_times_d)`, which is satisfied by a STRICT SUBSET, so a
    direct tree missing its last timestep passed too. Both were found by review, both by the
    same shape of counterexample.

    The rule that covers both is EQUALITY ON BOTH SIDES rather than a comparison against the
    smaller one. A retrieval is a substitute for the other only if it lands on the same
    coordinates, over the same extent, at the same times, so anything either one holds alone
    is a mismatch.
    """
    out = []
    rows_d, cols_d = shape_d
    rows_s, cols_s = shape_s
    if shared_rows != rows_d or shared_cols != cols_d:
        out.append(f"the direct grid has {rows_d} rows and {cols_d} columns but shares "
                   f"only {shared_rows} x {shared_cols} with the strided one")
    if shared_rows != rows_s or shared_cols != cols_s:
        out.append(f"the strided grid has {rows_s} rows and {cols_s} columns but shares "
                   f"only {shared_rows} x {shared_cols} with the direct one")
    # BOTH equalities, deliberately. Against min() a strict subset passes, which is the
    # defect this replaced; against the strided count alone, extra timestamps on the direct
    # side pass instead. Each clause catches what the other misses.
    if n_times_s != n_times_d or n_shared_times != n_times_s:
        out.append(f"the two retrievals carry {n_times_s} and {n_times_d} timestamps and "
                   f"share {n_shared_times}; a substitute must carry the same ones")
    return out


def _report(name, a, b):
    """One field compared, returning True when the two routes agree to rounding."""
    finite = np.isfinite(a) & np.isfinite(b)
    if not finite.any():
        print(f"  {name:14} no finite cells in common")
        return False
    diff = np.abs(a[finite] - b[finite])
    scale = float(np.sqrt(np.mean(np.square(b[finite]))))
    worst = float(diff.max())
    normalized = float(np.sqrt(np.mean(np.square(diff))) / scale) if scale else float("nan")
    unequal = int((diff > 0).sum())
    agrees = scale > 0 and normalized <= ROUNDING
    print(f"  {name:14} cells {finite.sum():8d}  differing {unequal:8d}  "
          f"max {worst:11.4e}  rms/scale {normalized:9.2e}  "
          f"{'agree' if agrees else 'DIFFER'}")
    return agrees


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--strided-dir", default="data/era5/v1port")
    ap.add_argument("--direct-dir", required=True,
                    help="a directory holding a DIRECT one-degree retrieval")
    ap.add_argument("--prefix", default="era5")
    ap.add_argument("--year", type=int, default=1981)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--subsample", type=int, default=2)
    args = ap.parse_args(argv)

    times_s, lat_s, lon_s, u_s, v_s = L.load_year(args.year, args.strided_dir, args.prefix)
    times_d, lat_d, lon_d, u_d, v_d = L.load_year(args.year, args.direct_dir, args.prefix)

    native_s = abs(float(lat_s[1, 0] - lat_s[0, 0]))
    native_d = abs(float(lat_d[1, 0] - lat_d[0, 0]))
    print(f"strided source {native_s} deg -> stride {args.subsample}, "
          f"direct source {native_d} deg")
    if abs(native_d - native_s * args.subsample) > 1e-6:
        raise SystemExit(
            f"the direct tree is {native_d} degrees, but striding the {native_s} degree "
            f"tree by {args.subsample} gives {native_s * args.subsample}. These are not "
            f"the same target grid, so comparing them would measure the wrong thing.")

    lat_s, lon_s, u_s, v_s = L._stride(args.subsample, lat_s, lon_s, u_s, v_s)
    curv_s = curvature_from_winds(lat_s, lon_s, u_s, v_s)
    curv_d = curvature_from_winds(lat_d, lon_d, u_d, v_d)

    # TIME IS MATCHED BY VALUE, not by position, because two retrievals of the same year can
    # differ in how many steps they carry and where they start.
    all_shared = np.intersect1d(np.round(times_s, 6), np.round(times_d, 6))
    shared_times = all_shared[:args.steps]
    if shared_times.size == 0:
        raise SystemExit("the two retrievals share no timestamps")
    ts = [int(np.argmin(np.abs(times_s - t))) for t in shared_times]
    td = [int(np.argmin(np.abs(times_d - t))) for t in shared_times]

    rows_s, cols_s, rows_d, cols_d = _shared_grid(
        lat_s[:, 0], lon_s[0, :], lat_d[:, 0], lon_d[0, :])
    print(f"grid: strided {lat_s.shape}, direct {lat_d.shape}, "
          f"shared {rows_s.size} x {cols_s.size} coordinates")
    if rows_s.size == 0 or cols_s.size == 0:
        raise SystemExit(
            "the two grids share no coordinates, which is itself the finding. The strided "
            "grid is out of phase with the direct one and they are not interchangeable")

    mismatch = coverage_mismatches(lat_s.shape, lat_d.shape, rows_s.size, cols_s.size,
                                   times_s.size, times_d.size, all_shared.size)

    def take(field, steps, rows, cols):
        return field[np.ix_(steps, rows, cols)]

    print(f"\nover {len(ts)} shared timesteps:")
    outcomes = [
        _report("u wind", take(u_s, ts, rows_s, cols_s), take(u_d, td, rows_d, cols_d)),
        _report("v wind", take(v_s, ts, rows_s, cols_s), take(v_d, td, rows_d, cols_d)),
        _report("curvature", take(curv_s, ts, rows_s, cols_s),
                take(curv_d, td, rows_d, cols_d)),
    ]

    print()
    if mismatch:
        print("THE TWO GRIDS DO NOT COVER THE SAME GROUND, so whether their shared cells "
              "agree is beside the point. A direct retrieval is a substitute for the "
              "strided one only if it lands on the same coordinates over the same extent "
              "at the same times.")
        for m in mismatch:
            print(f"  - {m}")
        return 1

    if all(outcomes):
        print("THE TWO ROUTES AGREE to storage rounding on this sample, which is what "
              "pointwise bilinear interpolation predicts. This is a result about retrieval "
              "bookkeeping only. Both routes still alias the 31 km field onto whole "
              "degrees, and ERA-Interim at the same nominal grid came from a much coarser "
              "model, so neither the aliasing question nor the reanalysis-difference "
              "question is touched by it.")
        return 0
    print("THE TWO ROUTES DIFFER. The stride is not a substitute for a direct one-degree "
          "request, the project text must not claim it is, and the production input should "
          "be the direct retrieval.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
