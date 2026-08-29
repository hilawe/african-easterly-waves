#!/usr/bin/env python3
"""Compute version 1's two detection thresholds for a reanalysis, from its own data.

WHY THIS IS NEEDED AT ALL. find_ews_f.m picks its thresholds by reanalysis NAME from a
hardcoded table, and it has no ERA5 entry. The numbers are percentiles of each
reanalysis's own curvature-vorticity anomaly distribution, so they cannot be borrowed
from a neighbour: doing that would produce a record that looks like version 1's and was
detected at a different sensitivity.

THE RECIPE IS KNOWN AND THE SAMPLE IS NOT, and the difference matters. find_ews_f.m's own
comments state what the numbers are:

    curv_thr_c ... 55th Percentile (90% for smoothing)
    curv_thr_f ... 66th Percentile

and the masks show which field each gates: the coarse threshold gates the DECIMATED
anomaly and the fine one gates the anomaly on the input grid, after the nine-point smoother
and the southern-hemisphere sign flip. That much is recoverable from the archived source.
What is NOT in the archive is the script that computed the percentiles, so the exact sample
Belanger drew them over, which years, which region, whether every timestep, is unknown.

SO THE RECIPE HAS TO BE VALIDATED RATHER THAN TRUSTED, and there is a way to do it. Running
this on ERA-INTERIM should reproduce version 1's own published pair for that reanalysis,
7.16e-7 and 2.80e-6 at 700 hPa. If it does, both the recipe and the sample are confirmed
and the ERA5 numbers can be used with the same confidence. If it does not, the difference
is the thing to explain before any ERA5 threshold means anything. Use --expect to make that
comparison explicit:

    .venv/bin/python scripts/compute_thresholds.py --prefix eraint \\
        --directory data/eraint/v1port --expect 7.16e-7 2.80e-6

A PERCENTILE OVER A SAMPLE, because the population is about eighteen gigabytes at ERA5's
resolution. A random sample estimates the right quantity, unlike per-year percentiles
averaged afterwards, and the sampling noise is measured and reported rather than assumed
small. If the reported spread is not small against the estimate, the answer is to raise
--per-step, not to use the number anyway.

    .venv/bin/python scripts/compute_thresholds.py            # ERA5, the full window
    .venv/bin/python scripts/compute_thresholds.py --years 1981 1982 1983
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from aew.v1port import climatology as clim  # noqa: E402
from aew.v1port import load as L  # noqa: E402
from aew.v1port import pipeline as P  # noqa: E402

COARSE_PERCENTILE = clim.COARSE_PERCENTILE      # 55
FINE_PERCENTILE = clim.FINE_PERCENTILE          # 66


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--directory", default="data/era5/v1port")
    ap.add_argument("--prefix", default="era5")
    ap.add_argument("--years", nargs="+", type=int, default=None)
    ap.add_argument("--coarse-resolution", type=float, default=2.5,
                    help="version 1's tracking grid, from p2_track_*.m")
    ap.add_argument("--per-step", type=int, default=400,
                    help="anomaly values sampled per timestep per grid")
    ap.add_argument("--expect", nargs=2, type=float, default=None,
                    metavar=("COARSE", "FINE"),
                    help="version 1's own pair, to validate the recipe against")
    ap.add_argument("--out", default=None, help="write the result as JSON")
    args = ap.parse_args(argv)

    years = args.years or L.available_years(args.directory, args.prefix)
    if not years:
        print(f"no complete year-pairs under {args.directory}", file=sys.stderr)
        return 2
    print(f"thresholds for {args.prefix} over {len(years)} years "
          f"({min(years)}-{max(years)})", flush=True)

    started = time.time()
    print("  pass 1 of 2: the climatological mean", flush=True)
    climatology = L.build_climatology(
        years, args.directory, args.prefix,
        progress=lambda y, n, t: print(f"     {y} ({n}/{t})", flush=True))
    print(f"     {len(climatology['keys'])} calendar steps, "
          f"{time.time()-started:.0f}s", flush=True)
    if climatology["short_steps"]:
        print(f"     NOTE {len(climatology['short_steps'])} calendar steps have fewer "
              f"samples than years, which is expected only for 29 February", flush=True)

    print("  pass 2 of 2: sampling the anomaly", flush=True)
    rng = np.random.default_rng(0)
    coarse_parts, fine_parts = [], []
    native = None
    for n, year in enumerate(years, start=1):
        times, latgrid, longrid, curvature = L.curvature_for_year(
            year, args.directory, args.prefix)
        if native is None:
            native = abs(float(latgrid[1, 0] - latgrid[0, 0]))
        coarse, fine = L.sample_anomaly_values(
            curvature, times, climatology, latgrid[:, 0],
            native_resolution=native, coarse_resolution=args.coarse_resolution,
            per_step=args.per_step, rng=rng)
        coarse_parts.append(coarse)
        fine_parts.append(fine)
        del curvature
        print(f"     {year} ({n}/{len(years)})", flush=True)

    coarse_sample = np.concatenate(coarse_parts)
    fine_sample = np.concatenate(fine_parts)
    coarse_value, coarse_spread = L.threshold_sampling_error(
        coarse_sample, COARSE_PERCENTILE)
    fine_value, fine_spread = L.threshold_sampling_error(fine_sample, FINE_PERCENTILE)

    print(f"\n  native grid {native} deg, coarse {args.coarse_resolution} deg")
    print(f"  coarse ({COARSE_PERCENTILE:g}th percentile, decimated grid): "
          f"{coarse_value:.3e}   sampling spread {coarse_spread:.1%} "
          f"over {coarse_sample.size:,} values")
    print(f"  fine   ({FINE_PERCENTILE:g}th percentile, input grid):      "
          f"{fine_value:.3e}   sampling spread {fine_spread:.1%} "
          f"over {fine_sample.size:,} values")

    result = {"prefix": args.prefix, "years": [min(years), max(years)],
              "n_years": len(years), "native_resolution": native,
              "coarse_resolution": args.coarse_resolution,
              "coarse_threshold": coarse_value, "fine_threshold": fine_value,
              "coarse_sampling_spread": coarse_spread,
              "fine_sampling_spread": fine_spread,
              "coarse_samples": int(coarse_sample.size),
              "fine_samples": int(fine_sample.size)}

    status = 0
    if args.expect:
        want_coarse, want_fine = args.expect
        rc = abs(coarse_value - want_coarse) / want_coarse
        rf = abs(fine_value - want_fine) / want_fine
        result["expected"] = {"coarse": want_coarse, "fine": want_fine,
                              "coarse_relative_difference": rc,
                              "fine_relative_difference": rf}
        print(f"\n  AGAINST VERSION 1'S OWN PAIR for this reanalysis:")
        print(f"    coarse {coarse_value:.3e} vs {want_coarse:.3e}  ({rc:+.1%})")
        print(f"    fine   {fine_value:.3e} vs {want_fine:.3e}  ({rf:+.1%})")
        if max(rc, rf) <= 0.10:
            print("    THE RECIPE REPRODUCES VERSION 1 within ten percent, so the sample "
                  "it was drawn over is close enough to matter little.")
        else:
            print("    THE RECIPE DOES NOT REPRODUCE VERSION 1. The percentile and the "
                  "field are taken from the archived source, so the difference is in the "
                  "SAMPLE, which the archive does not record. Explain it before using a "
                  "threshold computed this way for any other reanalysis.")
            status = 1

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        print(f"\n  written to {args.out}")
    print(f"  total {time.time()-started:.0f}s")
    return status


if __name__ == "__main__":
    sys.exit(main())
