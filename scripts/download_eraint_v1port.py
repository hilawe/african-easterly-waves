#!/usr/bin/env python3
"""Retrieve the ERA-Interim winds the version 1 VALIDATION run needs, resumably.

WHY THIS IS A DIFFERENT RETRIEVAL FROM THE ERA5 ONE, and why it is worth its cost.

The ERA5 retrieval exists to run version 1's method on a modern reanalysis, which is the
science. This one exists to find out whether the port is version 1 at all.

MATLAB cannot run here, so the original cannot be executed on a test case, and five review
rounds found twenty-three places the port's reading of it was wrong. Reading harder is not
the instrument. The instrument is version 1's own published output, NCEI C00784, which is
in `data/aewc` for 1983 to 2007. Running the port on the reanalysis version 1 used, over a
year the record covers, and comparing the tracks is the only check available that does not
depend on anyone's reading of the MATLAB.

It is also the only way to settle the convex hull's starting vertex, recorded in
`association._hull_polygon` as a divergence that is NOT ESTABLISHED. If MATLAB's Qhull
wrapper and scipy's start their cycles at different vertices, the search polygons are
systematically offset by up to about a degree, and tracks drift apart gradually rather than
breaking at one step. No review can see that. A track comparison can.

WHAT IS PINNED, and it is pinned to version 1's own driver rather than chosen.
p2_track_eraint_700hPa.m sets every one of these:

  LEVEL   700 hPa, the level whose thresholds are hardcoded in find_ews_f.m and whose
          published files are the ones held here.
  DOMAIN  latitude -35 to 35, longitude -140 to 40. Version 1 loads a wider grid and
          subsets to exactly this, and the published files carry it as their
          geospatial bounds.
  GRID    0.75 degrees, ERA-Interim's native resolution. Version 1 decimates from native
          to 2.5 for the coarse tracking grid and keeps native for the fine one, so
          requesting anything coarser would change both.
  HOURS   00, 06, 12, 18. The record is six-hourly and its `time_coverage_resolution` says
          so.
  YEARS   1981 to 2010, version 1's own climatology window. The anomaly every downstream
          stage works on is the departure from a mean field computed over exactly these
          years, so a shorter window is a different anomaly and not a validation.

THRESHOLDS ARE NOT NEEDED FROM THIS DATA, which is the one way this retrieval is cheaper
than the ERA5 one. Version 1 hardcodes its ERA-Interim 700 hPa thresholds, 7.16e-7 coarse
and 2.80e-6 fine, so only the climatological MEAN FIELD has to be computed here, not the
percentiles.

SIZE. About 94 by 241 points at 0.75 degrees, four times a day, so roughly 130 MB per
variable-year and of order 8 GB for the pair across thirty years. That is a fraction of the
ERA5 request because ERA-Interim is nine times coarser in area.

BEFORE THIS RUNS, THE ERA-INTERIM LICENCE HAS TO BE ACCEPTED on the Copernicus account.
The dataset and the request shape below are confirmed correct: a probe reached the licence
check, which is past request validation. The acceptance is a page visit and cannot be done
from here:

    https://cds.climate.copernicus.eu/datasets/reanalysis-era-interim?tab=download#manage-licences

RESUMABLE. One file per year and variable, existing files skipped, so it can be stopped and
restarted freely.

    .venv/bin/python scripts/download_eraint_v1port.py --dry-run
    .venv/bin/python scripts/download_eraint_v1port.py --years 2005    # one validation year
    .venv/bin/python scripts/download_eraint_v1port.py                 # the full window
"""

import argparse
import os
import sys
import time

CDS_DATASET = "reanalysis-era-interim"

# Version 1's own domain, from p2_track_eraint_700hPa.m: lat1 -35, lat2 35, lon1 -140,
# lon2 40. CDS wants north/west/south/east.
AREA = "35/-140/-35/40"
GRID = "0.75/0.75"
LEVEL = "700"
HOURS = "00:00:00/06:00:00/12:00:00/18:00:00"
# ECMWF parameter identifiers: 131 is u, 132 is v, both on table 128.
PARAMS = {"u700": "131.128", "v700": "132.128"}
CLIMATOLOGY_YEARS = tuple(range(1981, 2011))
OUT_DIR = "data/eraint/v1port"

LICENCE_URL = ("https://cds.climate.copernicus.eu/datasets/reanalysis-era-interim"
               "?tab=download#manage-licences")


def request_for(year, var_key):
    """The CDS request for one variable-year. Pure, so the shape is testable offline."""
    if var_key not in PARAMS:
        raise ValueError(f"unknown variable {var_key!r}, expected one of {sorted(PARAMS)}")
    return {
        "class": "ei",
        "dataset": "interim",
        "stream": "oper",
        "type": "an",
        "expver": "1",
        "levtype": "pl",
        "levelist": LEVEL,
        "param": PARAMS[var_key],
        "date": f"{year}-01-01/to/{year}-12-31",
        "time": HOURS,
        "step": "0",
        "grid": GRID,
        "area": AREA,
        "format": "netcdf",
    }


def output_path(year, var_key, out_dir=OUT_DIR):
    return os.path.join(out_dir, f"eraint_{var_key}_{year}_6h_region.nc")


def plan(years, variables, out_dir=OUT_DIR):
    todo, done = [], []
    for year in years:
        for var in variables:
            path = output_path(year, var, out_dir)
            (done if os.path.exists(path) else todo).append((year, var, path))
    return todo, done


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--years", nargs="+", type=int, default=list(CLIMATOLOGY_YEARS))
    ap.add_argument("--variables", nargs="+", default=sorted(PARAMS))
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and request nothing")
    args = ap.parse_args(argv)

    todo, done = plan(args.years, args.variables, args.out_dir)
    print("ERA-Interim retrieval for the version 1 validation run")
    print("  dataset %s" % CDS_DATASET)
    print("  area    %s (north/west/south/east), grid %s" % (AREA, GRID))
    print("  level   %s hPa at %s" % (LEVEL, HOURS))
    print("  years   %d through %d" % (min(args.years), max(args.years)))
    print("  files   %d already present, %d to retrieve" % (len(done), len(todo)))
    if args.dry_run:
        for year, var, _ in todo[:6]:
            print("    would request %s %d" % (var, year))
        if len(todo) > 6:
            print("    ... and %d more" % (len(todo) - 6))
        return 0
    if not todo:
        print("nothing to do")
        return 0

    import cdsapi

    os.makedirs(args.out_dir, exist_ok=True)
    client = cdsapi.Client()
    started, failures = time.time(), 0
    for n, (year, var, path) in enumerate(todo, start=1):
        t0 = time.time()
        try:
            client.retrieve(CDS_DATASET, request_for(year, var), path)
            print("  %3d/%d  %s %d  %.0f s  %.0f MB"
                  % (n, len(todo), var, year, time.time() - t0,
                     os.path.getsize(path) / 1e6), flush=True)
        except Exception as exc:                      # noqa: BLE001
            failures += 1
            message = str(exc)
            if "licence" in message.lower() or "403" in message:
                # The one failure worth stopping for: every later request fails the same
                # way, so continuing would just print the same line thirty more times.
                print("\nSTOPPED: the ERA-Interim licence has not been accepted on this "
                      "Copernicus account.\nAccept it here, then re-run:\n  %s"
                      % LICENCE_URL, flush=True)
                return 2
            print("  FAILED %s %d after %.0f s: %s: %s"
                  % (var, year, time.time() - t0, exc.__class__.__name__, message[:200]),
                  flush=True)
    print("done in %.0f min, %d failed" % ((time.time() - started) / 60.0, failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
