#!/usr/bin/env python3
"""Every year's wider retrieval against the production file on the columns they share.

The eastern pilot retrieves the reference years again on a grid that reaches 75 E. The
1990 probe was bit-identical to the production file on the 211 shared columns, and that
does not certify the other years, so each year is checked the same way: identical
latitudes, identical timestamps, identical longitude values on the shared columns, every
wind value on those columns equal bit for bit, and the same ERA5 experiment versions. A
year with any difference is named with what differs. Exit 0 only when every requested year
and variable passes.

    python3 scripts/check_shared_columns.py --production data/era5/v1port_buffered \\
        --wider data/era5/pilot_east_75E --years 1979 2010 [--out <json>]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import exact_tracks as X  # noqa: E402

VARIABLES = ("u700", "v700")


def _coords(ds):
    lat = ds.variables["latitude"][:] if "latitude" in ds.variables else ds.variables["lat"][:]
    lon = ds.variables["longitude"][:] if "longitude" in ds.variables else ds.variables["lon"][:]
    time = ds.variables["valid_time"][:] if "valid_time" in ds.variables else ds.variables["time"][:]
    return np.asarray(lat), np.asarray(lon), np.asarray(time)


def _field(ds):
    skip = {"latitude", "lat", "longitude", "lon", "valid_time", "time", "number", "pressure_level", "expver"}
    names = [v for v in ds.variables if v not in skip]
    if len(names) != 1:
        raise SystemExit(f"REFUSED: expected one field variable, found {names}")
    return names[0]


def compare_year(production_path, wider_path):
    """The differences between the two files on the shared columns, as a list of
    strings, empty when they agree, plus a small summary."""
    import netCDF4
    problems = []
    with netCDF4.Dataset(production_path) as a, netCDF4.Dataset(wider_path) as b:
        lat_a, lon_a, t_a = _coords(a)
        lat_b, lon_b, t_b = _coords(b)
        if not np.array_equal(lat_a, lat_b):
            problems.append("latitudes differ")
        if not np.array_equal(t_a, t_b):
            problems.append(f"timestamps differ ({t_a.size} against {t_b.size})")
        cols = np.where(np.isin(lon_b, lon_a))[0]
        if cols.size != lon_a.size or not np.array_equal(lon_b[cols], lon_a):
            problems.append(f"the wider grid does not hold every production longitude ({cols.size} of {lon_a.size})")
        summary = {"shared_columns": int(cols.size), "new_columns": int(lon_b.size - cols.size),
                   "east_edge_production": float(lon_a.max()), "east_edge_wider": float(lon_b.max())}
        if problems:
            return problems, summary
        name_a, name_b = _field(a), _field(b)
        if name_a != name_b:
            problems.append(f"field names differ ({name_a} against {name_b})")
            return problems, summary
        xa = np.asarray(a.variables[name_a][:])
        xb = np.asarray(b.variables[name_b][:])[..., cols]
        if xa.shape != xb.shape:
            problems.append(f"shapes differ on the shared columns ({xa.shape} against {xb.shape})")
        elif not np.array_equal(xa, xb):
            d = np.abs(xa.astype(float) - xb.astype(float))
            problems.append(f"values differ on the shared columns, {int(np.count_nonzero(d))} cells, largest {float(np.nanmax(d)):.3e}")
        for ds, label in ((a, "production"), (b, "wider")):
            if "expver" in ds.variables:
                summary[f"expver_{label}"] = sorted(set(np.asarray(ds.variables["expver"][:]).ravel().tolist()))
        if summary.get("expver_production") != summary.get("expver_wider"):
            problems.append(f"experiment versions differ ({summary.get('expver_production')} against {summary.get('expver_wider')})")
    return problems, summary


def check(production, wider, years):
    report, passed = {}, True
    for year in range(years[0], years[1] + 1):
        for var in VARIABLES:
            name = f"era5_{var}_{year}_6h_region.nc"
            pa, pb = os.path.join(production, name), os.path.join(wider, name)
            if not os.path.exists(pa) or not os.path.exists(pb):
                report[name] = {"problems": [f"missing: {'production' if not os.path.exists(pa) else 'wider'} file absent"]}
                passed = False
                continue
            problems, summary = compare_year(pa, pb)
            report[name] = {"problems": problems, **summary, "wider_sha256": X.digest(pb), "production_sha256": X.digest(pa)}
            passed &= not problems
    return {"generated_by": "scripts/check_shared_columns.py", "production": production, "wider": wider,
            "years": list(years), "files": report, "passed": passed}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--production", required=True)
    ap.add_argument("--wider", required=True)
    ap.add_argument("--years", nargs=2, type=int, required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    r = check(args.production, args.wider, tuple(args.years))
    for name, entry in r["files"].items():
        print(f"{name}: {'OK' if not entry['problems'] else '; '.join(entry['problems'])}")
    print("PASSED" if r["passed"] else "NOT PASSED")
    if args.out:
        try:
            X.publish_json(args.out, r, exclusive=True)
        except FileExistsError:
            raise SystemExit(f"REFUSED: {args.out} exists and reports are never overwritten")
    return 0 if r["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
