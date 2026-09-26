#!/usr/bin/env python3
"""Retrieve the ERA5 winds for the reanalysis comparison protocol, one year and one
variable per request, on version 1's BUFFERED ONE-DEGREE GRID, validated file by file.

THE REQUEST, corrected 2026-09-25 on Hilawe's approval. The earlier request pinned a
half-degree grid over 45 N to 20 S and 120 W to 70 E, which is not the grid the
protocol runs on, and its files stay untouched in `data/era5/v1port` as the record of
that request. This one asks for what `REANALYSIS_COMPARISON_PROTOCOL.md` section 1
fixes: u and v at 700 hPa, six-hourly at the four synoptic hours, all twelve months,
1979 to 2010, over version 1's own buffered data domain 50 N to 50 S and 155 W to 55 E
at one degree, which is 101 by 211 cells, the same grid the ERA-Interim tree holds.

FILE EXISTENCE IS NOT COMPLETION. Every file, existing or freshly returned, is validated
against the requested coordinates (exact equality), the year's full six-hourly calendar,
the variable, the pressure level and the units, with the ERA-Interim tree's own
validator, and a file that fails is moved aside as `.invalid` and re-queued. A download
lands at a `.partial` path and is renamed only after it validates. Every completed file
gets a completion record beside it with its digest, size, elapsed time and the request.

ONE DURABLE PROCESS per run: this is meant to be started once under nohup with its log,
and its completion records are what say what happened.

    .venv/bin/python3 scripts/download_era5_v1port.py --years 1990            # the approved year
    .venv/bin/python3 scripts/download_era5_v1port.py --dry-run               # plan only
"""

import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aew.data.era5 import SYNOPTIC_HOURS, cds_request  # noqa: E402
from download_eraint_v1port import LEVEL, expected_coordinates, validate_file  # noqa: E402

# version 1's buffered data domain, from p1_data_eraint.m, as the ERA-Interim tree was
# retrieved: (north, west, south, east) and one degree
AREA = [50.0, -155.0, -50.0, 55.0]
GRID = (1.0, 1.0)
AREA_TEXT = "/".join(f"{x:g}" for x in AREA)
GRID_TEXT = "/".join(f"{x:g}" for x in GRID)
MONTHS = tuple(range(1, 13))
VARIABLES = ("u700", "v700")
PROTOCOL_YEARS = tuple(range(1979, 2011))
OUT_DIR = "data/era5/v1port_buffered"
EXPECTED_SHAPE = (101, 211)


def output_path(year, var_key, out_dir=OUT_DIR):
    return os.path.join(out_dir, f"era5_{var_key}_{year}_6h_region.nc")


def request_for(year, var_key):
    """The exact (dataset, request) pair that is sent, built by the pure builder."""
    return cds_request(var_key, year, months=MONTHS, hours=SYNOPTIC_HOURS, area=AREA, grid=GRID)


def plan(years, variables, out_dir=OUT_DIR, repair=True):
    """What to retrieve, with EXISTING FILES VALIDATED rather than trusted."""
    todo, done = [], []
    for year in years:
        for var in variables:
            path = output_path(year, var, out_dir)
            if not os.path.exists(path):
                todo.append((year, var, path))
                continue
            reason = validate_file(path, year, var, AREA_TEXT, GRID_TEXT)
            if reason is None:
                # A VALID FILE WITHOUT ITS RECORD IS NOT DONE. The record names the bytes,
                # and a restart after a stop between the rename and the record must write
                # it rather than count the file silently.
                if not record_matches(path):
                    if repair:
                        dataset, request = request_for(year, var)
                        completion_record(path, year, var, None, None, dataset, request, reconciled=True)
                        print(f"  RECONCILED a completion record for {os.path.basename(path)}", flush=True)
                    else:
                        print(f"  {os.path.basename(path)} is valid but has no completion record", flush=True)
                done.append((year, var, path))
                continue
            if repair:
                os.replace(path, path + ".invalid")
                print(f"  EXISTING FILE FAILED VALIDATION and was moved aside: "
                      f"{os.path.basename(path)} ({reason})", flush=True)
            else:
                print(f"  EXISTING FILE FAILS VALIDATION and would be moved aside: "
                      f"{os.path.basename(path)} ({reason})", flush=True)
            todo.append((year, var, path))
    return todo, done


def completion_record(path, year, var, started, elapsed, dataset, request, reconciled=False):
    """The record beside a validated file, written through a temporary file and an atomic
    rename. `reconciled` marks a record written at restart for a valid file that had none,
    whose elapsed time is therefore unknown."""
    with open(path, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    record = {"file": os.path.basename(path), "year": year, "variable": var,
              "sha256": digest, "bytes": os.path.getsize(path),
              "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)) if started else None,
              "elapsed_seconds": round(elapsed, 1) if elapsed is not None else None,
              "dataset": dataset, "request": request, "reconciled_at_restart": bool(reconciled),
              "validated": "passed the shared validator: exact grid, full six-hourly calendar, "
                           "epoch, variable, dimensions, units and a finite non-constant payload"}
    final = path + ".completion.json"
    tmp = final + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(record, fh, indent=1, sort_keys=True)
    os.replace(tmp, final)
    return record


def record_matches(path):
    """Whether the completion record beside `path` exists and names these bytes."""
    final = path + ".completion.json"
    if not os.path.exists(final):
        return False
    try:
        with open(final) as fh:
            record = json.load(fh)
        with open(path, "rb") as fh:
            return record.get("sha256") == hashlib.sha256(fh.read()).hexdigest()
    except (OSError, ValueError):
        return False


def main(argv=None, client=None):
    """`client` is injectable for tests: anything with retrieve(dataset, request, target)."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--years", nargs="+", type=int, default=list(PROTOCOL_YEARS))
    ap.add_argument("--variables", nargs="+", default=list(VARIABLES))
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--dry-run", action="store_true", help="print the plan and request nothing")
    args = ap.parse_args(argv)
    want_lat, want_lon = expected_coordinates(AREA_TEXT, GRID_TEXT)
    if (want_lat.size, want_lon.size) != EXPECTED_SHAPE:
        raise SystemExit(f"REFUSED: the request would return {want_lat.size}x{want_lon.size}, "
                         f"not the protocol's {EXPECTED_SHAPE}")
    todo, done = plan(args.years, args.variables, args.out_dir, repair=not args.dry_run)
    print("ERA5 retrieval on version 1's buffered one-degree grid")
    print("  area   (N,W,S,E) %s, grid %s degrees, %dx%d cells" % (AREA, GRID, *EXPECTED_SHAPE))
    print("  level  %s hPa at %s, months %d through %d" % (LEVEL, SYNOPTIC_HOURS, MONTHS[0], MONTHS[-1]))
    print("  years  %d through %d" % (min(args.years), max(args.years)))
    print("  files  %d present and valid, %d to retrieve" % (len(done), len(todo)))
    if args.dry_run:
        for year, var, _ in todo[:8]:
            dataset, request = request_for(year, var)
            print("    would request %s %d from %s: area %s grid %s level %s"
                  % (var, year, dataset, request["area"], request["grid"], request["pressure_level"]))
        return 0
    if not todo:
        print("nothing to do")
        return 0
    os.makedirs(args.out_dir, exist_ok=True)
    if client is None:
        import cdsapi
        client = cdsapi.Client()
    failures = 0
    for n, (year, var, path) in enumerate(todo, start=1):
        started = time.time()
        partial = path + ".partial"
        dataset, request = request_for(year, var)
        try:
            client.retrieve(dataset, request, partial)
        except Exception as exc:                      # noqa: BLE001
            failures += 1
            if os.path.exists(partial):
                os.replace(partial, path + ".interrupted")
            print("  FAILED %s %d after %.0f s: %s: %s" % (var, year, time.time() - started,
                                                           exc.__class__.__name__, exc), flush=True)
            continue
        reason = validate_file(partial, year, var, AREA_TEXT, GRID_TEXT)
        if reason is not None:
            failures += 1
            os.replace(partial, path + ".invalid")
            print("  RETURNED FILE FAILED VALIDATION and was moved aside: %s %d (%s)"
                  % (var, year, reason), flush=True)
            continue
        os.replace(partial, path)
        record = completion_record(path, year, var, started, time.time() - started, dataset, request)
        print("  [%d/%d] %s %d  %.1f MB  %.0f s" % (n, len(todo), var, year,
                                                    record["bytes"] / 1e6, record["elapsed_seconds"]), flush=True)
    remaining, _ = plan(args.years, args.variables, args.out_dir, repair=False)
    print("done; %d failure(s), %d file(s) still missing or invalid" % (failures, len(remaining)))
    return 0 if not remaining else 1


if __name__ == "__main__":
    sys.exit(main())
