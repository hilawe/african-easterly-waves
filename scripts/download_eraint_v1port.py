#!/usr/bin/env python3
"""Retrieve the ERA-Interim winds the version 1 VALIDATION run needs, resumably.

WHY THIS IS A DIFFERENT RETRIEVAL FROM THE ERA5 ONE, and why it is worth its cost.

The ERA5 retrieval exists to run version 1's method on a modern reanalysis, which is the
science. This one exists to find out whether the port is version 1 at all.

The original is now executable under Octave, but was not when this was written, and five review
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
  DOMAIN  THE TRACKED domain is latitude -35 to 35, longitude -140 to 40, and the
          published files carry it as their geospatial bounds. THE RETRIEVED domain has
          to be WIDER, and an earlier version of this script got that wrong by asking
          for exactly the tracked one.
          The vorticity stencil cannot compute the outer two rows and columns of
          whatever grid it is given, and decimation and the nine-point smoother widen
          that dead margin. Retrieving exactly the tracked domain therefore left the
          usable field running only to 36 E while Africa runs to 40 E, and measured
          against version 1's own 1990 record it put 716 of 8,109 African observations,
          8.8 percent, outside the field entirely, 598 of them on that eastern edge.
          VERSION 1 RETRIEVED A BUFFER AND SAID SO. p1_data_eraint.m loads latitude -40
          to 40 and longitude -145 to 45, already five degrees wider than the tracked
          domain on every side, and then takes TEN FURTHER GRID POINTS beyond that, with
          the comment that they are a buffer for the vorticity calculation. Its data
          domain is therefore -50 to 50 and -155 to 55.
          It also fixes the coarse grid's PHASE. p2_track computes the advection and
          decimates on the full buffered grid and only then subsets, so starting the
          decimation at -50 with a stride of two puts the tracking grid on EVEN degrees,
          where starting at -35 puts it on odd ones. Retrieving the buffer reproduces
          version 1's mesh and not merely its resolution.
          Use --area to request either. The default stays the tracked domain so an
          existing tree rebuilds unchanged.
  GRID    ONE DEGREE, and this was got wrong the first time. An earlier version of this
          note reasoned that version 1 must have used ERA-Interim's native 0.75, since
          the tracker decimates from native to the coarse grid and keeps native for the
          fine one. The published record says otherwise. Every `maxlat` and `minlat` in
          the 1990 Africa file is a whole number, 66 distinct values across 8,109
          observations with a smallest distinct spacing of exactly 1.0, while `lat` in
          the same file and the same float32 type keeps 1,297 distinct values including
          -32.8333, so a write-time rounding would have quantised both and did not.
          VERSION 1 RAN ON A ONE DEGREE GRID.
          It is not a cosmetic difference. `decimate_f.m` takes the FLOOR of the
          resolution ratio, so one degree gives a filter width of two, a stride of two
          and a 2.0 degree tracking grid, while 0.75 gives width three, stride three and
          2.25. Neither is the 2.5 the parameter is named for. The 0.75 retrieval was
          therefore detecting on a different mesh from version 1 throughout, and
          regridding it to one degree recovered a fifth of the waves the port had been
          missing. Use --grid to request either.
  HOURS   00, 06, 12, 18. The record is six-hourly and its `time_coverage_resolution` says
          so.
  YEARS   1981 to 2010, version 1's own climatology window. The anomaly every downstream
          stage works on is the departure from a mean field computed over exactly these
          years, so a shorter window is a different anomaly and not a validation.

THRESHOLDS ARE NOT NEEDED FROM THIS DATA, which is the one way this retrieval is cheaper
than the ERA5 one. Version 1 hardcodes its ERA-Interim 700 hPa thresholds, 7.16e-7 coarse
and 2.80e-6 fine, so only the climatological MEAN FIELD has to be computed here, not the
percentiles.

SIZE. About 71 by 181 points at one degree and 94 by 241 at 0.75, four times a day, so
roughly 75 and 130 MB per variable-year respectively, and of order 4.5 and 8 GB for the
pair across thirty years. That is a fraction of the ERA5 request because ERA-Interim is
nine times coarser in area.

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
# The TRACKED domain, and the DEFAULT, so an existing tree rebuilds the way it was built.
# Version 1's own retrieved domain is the buffered one; see DOMAIN in the module docstring.
AREA = "35/-140/-35/40"
BUFFERED_AREA = "50/-155/-50/55"     # version 1's data domain, from p1_data_eraint.m
# The DEFAULT stays 0.75 so an existing tree keeps rebuilding the way it was built. The
# grid version 1 actually used is one degree; see GRID in the module docstring.
GRID = "0.75/0.75"
LEVEL = "700"
HOURS = "00:00:00/06:00:00/12:00:00/18:00:00"
# ECMWF parameter identifiers: 131 is u, 132 is v, both on table 128.
PARAMS = {"u700": "131.128", "v700": "132.128"}
CLIMATOLOGY_YEARS = tuple(range(1981, 2011))
OUT_DIR = "data/eraint/v1port"

LICENCE_URL = ("https://cds.climate.copernicus.eu/datasets/reanalysis-era-interim"
               "?tab=download#manage-licences")


def request_for(year, var_key, grid=GRID, area=AREA):
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
        "grid": grid,
        "area": area,
        "format": "netcdf",
    }


def output_path(year, var_key, out_dir=OUT_DIR):
    return os.path.join(out_dir, f"eraint_{var_key}_{year}_6h_region.nc")


LEVEL_NAMES = ("pressure_level", "level", "plev", "isobaricInhPa")


def _level_hpa(ds):
    """The file's singleton pressure level in hPa, or a refusal reason.

    The loader squeezes any singleton level axis WITHOUT checking its coordinate, so an
    850 hPa field in a file named u700 would run silently at the wrong level all the way
    into a named threshold case. Genuine retrievals carry pressure_level = [700.0] with
    units hPa, so the coordinate is required, not optional.
    """
    import numpy as np
    for name in LEVEL_NAMES:
        if name in ds.variables:
            values = np.asarray(ds[name][:], dtype=float).ravel()
            if values.size != 1:
                return None, f"{name} holds {values.size} levels, expected exactly one"
            units = str(getattr(ds[name], "units", "")).lower()
            value = float(values[0])
            if units in ("hpa", "millibars", "millibar", "mb"):
                return value, None
            if units in ("pa", "pascal", "pascals"):
                return value / 100.0, None
            return None, f"{name} has unrecognised units {units!r}"
    return None, "no pressure-level coordinate; the level cannot be established"


def expected_coordinates(area, grid):
    """The lat and lon vectors a request over `area` at `grid` returns."""
    import numpy as np
    north, west, south, east = (float(x) for x in area.split("/"))
    dlat, dlon = (float(x) for x in grid.split("/"))
    lat = np.arange(north, south - dlat / 2, -dlat)
    lon = np.arange(west, east + dlon / 2, dlon)
    return lat, lon


def validate_file(path, year, var_key, area, grid):
    """Whether a retrieved file is the complete year it claims, or why not.

    Returns None when the file is sound, otherwise a one-line reason. AN EXISTING PATH
    IS NOT EVIDENCE OF A COMPLETE RETRIEVAL: an interrupted request used to leave a
    partial file at the final path, and every later run skipped it as done, so existing
    files are validated with the same checks a fresh download gets.
    """
    import calendar
    import datetime

    import netCDF4 as nc
    import numpy as np

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "src"))
    from aew.v1port import load as L

    short = {"u700": "u", "v700": "v"}[var_key]
    try:
        with nc.Dataset(path) as ds:
            lat = np.asarray(ds["latitude"][:], dtype=float)
            lon = np.asarray(ds["longitude"][:], dtype=float)
            # THE LOADER'S OWN TIME RULES, not a raw read. A first version counted raw
            # timestamp values, so 1,460 zeros passed, and so did a file whose units
            # were hours since 1800, because nothing ever decoded them. _time_days
            # accepts exactly what the loader accepts and refuses the rest.
            times = L._time_days(ds)
            level, level_reason = _level_hpa(ds)
            var_name = short if short in ds.variables else var_key
            if var_name not in ds.variables:
                return f"no {short} variable (has {sorted(ds.variables)[:6]})"
            shape = ds[var_name].shape
    except Exception as exc:                          # noqa: BLE001
        return f"unreadable ({exc.__class__.__name__}: {str(exc)[:120]})"
    if level_reason is not None:
        return level_reason
    if level != float(LEVEL):
        return f"pressure level is {level} hPa where the request was {LEVEL}"
    want_lat, want_lon = expected_coordinates(area, grid)
    # EXACT equality: allclose's default relative tolerance would bless a grid a
    # millionth of a degree off, which is not the grid that was requested
    if not (np.array_equal(lat, want_lat) and np.array_equal(lon, want_lon)):
        return (f"grid is {lat.size}x{lon.size} from {lat[0] if lat.size else '?'}/"
                f"{lon[0] if lon.size else '?'}, expected {want_lat.size}x"
                f"{want_lon.size} from {want_lat[0]}/{want_lon[0]}, compared exactly")
    days = 366 if calendar.isleap(year) else 365
    expected_steps = days * 4
    jan1 = float((datetime.date(year, 1, 1) - datetime.date(1900, 1, 1)).days)
    one_second = 1.0 / 86400.0
    if times.size != expected_steps:
        return f"{times.size} timesteps where a complete {year} is {expected_steps}"
    if abs(times[0] - jan1) > one_second:
        return (f"first timestamp is day {times[0]:.3f} since 1900 where January 1 of "
                f"{year} is {jan1:.1f}; the data is not the requested year")
    if np.any(np.abs(np.diff(times) - 0.25) > one_second):
        return "timestamps are not strictly six-hourly"
    if shape[0] != times.size or shape[-2:] != (lat.size, lon.size):
        return f"variable shape {shape} disagrees with its own axes"
    return None


def plan(years, variables, out_dir=OUT_DIR, area=BUFFERED_AREA, grid="1.0/1.0",
         repair=True):
    """What to retrieve, with EXISTING FILES VALIDATED rather than trusted.

    A final path that fails validation is moved aside to .invalid and re-queued, so a
    truncated earlier run cannot silently satisfy this one.
    """
    todo, done = [], []
    for year in years:
        for var in variables:
            path = output_path(year, var, out_dir)
            if not os.path.exists(path):
                todo.append((year, var, path))
                continue
            reason = validate_file(path, year, var, area, grid)
            if reason is None:
                done.append((year, var, path))
            elif repair:
                os.replace(path, path + ".invalid")
                print(f"  EXISTING FILE FAILED VALIDATION and was moved aside: "
                      f"{os.path.basename(path)} ({reason})", flush=True)
                todo.append((year, var, path))
            else:
                # A DRY RUN IS READ-ONLY. A first version moved invalid files aside even
                # under --dry-run, which is a write a "print the plan" flag must not do.
                print(f"  EXISTING FILE FAILS VALIDATION and would be moved aside: "
                      f"{os.path.basename(path)} ({reason})", flush=True)
                todo.append((year, var, path))
    return todo, done


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--years", nargs="+", type=int, default=list(CLIMATOLOGY_YEARS))
    ap.add_argument("--variables", nargs="+", default=sorted(PARAMS))
    ap.add_argument("--out-dir", default=OUT_DIR)
    # VERSION 1'S OWN RETRIEVAL IS THE DEFAULT. The tracked domain and 0.75 degrees are
    # what this tree was FIRST built at, before the buffer and the one-degree grid were
    # established as version 1's, and leaving them as defaults meant an unattended rerun
    # would rebuild the superseded tree without saying so.
    ap.add_argument("--area", default=BUFFERED_AREA,
                    help="CDS area as 'north/west/south/east'. The default is version 1's "
                         f"own buffered domain; the tracked-domain {AREA} is what this "
                         "tree was first, and wrongly, built at.")
    ap.add_argument("--grid", default="1.0/1.0",
                    help="CDS grid as 'dlat/dlon'. The default is version 1's own 1.0; "
                         f"the {GRID} this tree was first built at gives a 2.25 degree "
                         "tracking mesh where version 1 had 2.0.")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and request nothing")
    args = ap.parse_args(argv)

    todo, done = plan(args.years, args.variables, args.out_dir,
                      area=args.area, grid=args.grid, repair=not args.dry_run)
    print("ERA-Interim retrieval for the version 1 validation run")
    print("  dataset %s" % CDS_DATASET)
    print("  area    %s (north/west/south/east), grid %s" % (args.area, args.grid))
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
        # DOWNLOAD TO A PARTIAL PATH, validate, then rename atomically. An interrupted
        # request must never leave a file at the final path for the next run to skip.
        partial = path + ".partial"
        try:
            client.retrieve(CDS_DATASET,
                            request_for(year, var, args.grid, args.area), partial)
            reason = validate_file(partial, year, var, args.area, args.grid)
            if reason is not None:
                failures += 1
                os.replace(partial, path + ".invalid")
                print("  FAILED VALIDATION %s %d: %s (kept as .invalid)"
                      % (var, year, reason), flush=True)
                continue
            os.replace(partial, path)
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
