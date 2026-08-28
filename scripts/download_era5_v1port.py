#!/usr/bin/env python3
"""Retrieve the ERA5 winds the version 1 port needs, one year per request, resumably.

WHY THIS EXISTS. Version 1 selects its detection thresholds by reanalysis name and has no
ERA5 branch, and those thresholds are the 55th and 66th percentiles of each reanalysis's
own curvature-vorticity distribution, so they cannot be carried across. ERA5 needs its
own climatology computed before anything downstream can run. This is the retrieval that
makes that possible, and it is the long pole of the project.

EVERY CHOICE BELOW IS PINNED DELIBERATELY, because a Copernicus request is queued with no
throughput guarantee and re-running one on a changed domain costs days rather than
minutes.

VARIABLES: u and v at 700 hPa, and nothing else. The version 1 data stage reads only the
two wind components at a single level; the vorticity decomposition derives everything
else. The optional environment layers in version 1 are each guarded by a file-exists test
and can be absent.

LEVEL: 700 hPa. Version 1 shipped 600, 700 and 850, and 700 is the level the coauthor's
own work and QTrack's strength variable both use, so it is the one that makes the
intercomparison comparable. The other two can follow if wanted.

AREA, as (north, west, south, east) = (45, -120, -20, 70). Wider than the eventual
published domain on every side, on purpose:

  east   70 covers a published eastern boundary near 60 E plus the computational halo
         that version 1's own code adds. The exact boundary is still to be confirmed with
         the coauthor, and retrieving generously now avoids re-retrieving later.
  west  -120 carries a wave across the whole Atlantic and into the Caribbean. The focused
         product excludes the Pacific, but a track has to be followable to its end.
  north   45 and south -20 bracket both preferred wave latitude bands with room for the
         two-row edge the vorticity calculation cannot compute.

GRID: 0.5 degrees. Version 1 ran on a 1-degree grid, and 0.5 coarsens to exactly 1 for a
faithful reproduction while leaving the option of a finer version 2. Requesting ERA5's
native 0.25 would be four times the volume for a resolution the method was never tuned
at.

MONTHS: all twelve. The climatology is a mean per calendar timestep across the full annual
cycle, so a June-to-October subset cannot form it, even though the waves themselves are a
summer phenomenon.

YEARS: 1981 to 2010, matching version 1's own climatology window. The first task is to
compare version 1's method on ERA5 against version 1 on ERA-Interim, and that comparison
is cleanest when the climatology period is identical. A modern normal can be computed
later for the production record.

RESUMABLE. Each year and variable is one file and one request, and an existing file is
skipped, so this can be stopped and restarted freely.

    .venv/bin/python scripts/download_era5_v1port.py            # the full 1981-2010 set
    .venv/bin/python scripts/download_era5_v1port.py --years 1981 1982
    .venv/bin/python scripts/download_era5_v1port.py --dry-run  # show the plan, request nothing
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from aew.data.era5 import download_year_6hourly_region  # noqa: E402

# (north, west, south, east), and the reasons are in the module docstring above
AREA = [45.0, -120.0, -20.0, 70.0]
GRID = (0.5, 0.5)
MONTHS = tuple(range(1, 13))
VARIABLES = ("u700", "v700")
CLIMATOLOGY_YEARS = tuple(range(1981, 2011))
OUT_DIR = "data/era5/v1port"


def plan(years, variables):
    todo, done = [], []
    for year in years:
        for var in variables:
            path = os.path.join(OUT_DIR, f"era5_{var}_{year}_6h_region.nc")
            (done if os.path.exists(path) else todo).append((year, var, path))
    return todo, done


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--years", nargs="+", type=int, default=list(CLIMATOLOGY_YEARS))
    ap.add_argument("--variables", nargs="+", default=list(VARIABLES))
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and request nothing")
    args = ap.parse_args(argv)

    todo, done = plan(args.years, args.variables)
    print("ERA5 retrieval for the version 1 port")
    print("  area   (N,W,S,E) %s" % AREA)
    print("  grid   %s degrees" % (GRID,))
    print("  months %d through %d" % (MONTHS[0], MONTHS[-1]))
    print("  years  %d through %d" % (min(args.years), max(args.years)))
    print("  files  %d already present, %d to retrieve" % (len(done), len(todo)))
    if args.dry_run:
        for year, var, path in todo[:8]:
            print("    would request %s %s" % (var, year))
        if len(todo) > 8:
            print("    ... and %d more" % (len(todo) - 8))
        return 0
    if not todo:
        print("nothing to do")
        return 0

    os.makedirs(OUT_DIR, exist_ok=True)
    started = time.time()
    for n, (year, var, _) in enumerate(todo, start=1):
        t0 = time.time()
        try:
            path = download_year_6hourly_region(
                year, var, months=MONTHS, area=AREA, grid=GRID, out_dir=OUT_DIR)
        except Exception as exc:                      # noqa: BLE001
            # One failed year must not lose the ones already retrieved. Report and carry
            # on: the next run skips whatever landed and retries the rest.
            print("  FAILED %s %d after %.0f s: %s: %s"
                  % (var, year, time.time() - t0, exc.__class__.__name__, exc),
                  flush=True)
            continue
        size = os.path.getsize(path) / 1e6 if os.path.exists(path) else 0.0
        print("  [%d/%d] %s %d  %.0f MB  %.0f s elapsed, %.1f h total"
              % (n, len(todo), var, year, size, time.time() - t0,
                 (time.time() - started) / 3600.0), flush=True)

    remaining, _ = plan(args.years, args.variables)
    print("done; %d file(s) still missing" % len(remaining))
    return 0 if not remaining else 1


if __name__ == "__main__":
    sys.exit(main())
