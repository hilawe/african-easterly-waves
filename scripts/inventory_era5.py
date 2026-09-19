#!/usr/bin/env python3
"""Inventory every ERA5 file on disk: time coverage, grid, step, and missing values.

Read-only. It downloads nothing and runs no tracker. Its purpose is the exact
missing-data list a calibration proposal needs: which years, which domain, which
variables are present, and whether any file carries masked or NaN cells. Every array is
read in full for the missing-value count, so the run takes a few minutes.

    .venv/bin/python scripts/inventory_era5.py --root data/era5 --out <artifact.json>
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

try:
    import netCDF4 as nc
except ImportError:                                            # pragma: no cover
    sys.exit("netCDF4 is required")

SKIP_VARS = ("time", "valid_time", "latitude", "longitude", "lat", "lon", "expver", "number",
             "pressure_level")

# Version 1's buffered one-degree grid and the two candidate population periods, which
# the descriptive summary is measured AGAINST. What the summary states is metadata
# read from the files: the extent envelope over a tree (the union of every file's
# axes, named as an envelope because two components can each cover half of it), the
# spacing, the field-and-level combinations, the years in which each needed field has
# a file, whether each such year is a complete six-hourly year, and whether the two
# wind components share one grid in each year, judged on the full coordinate arrays. IT DOES NOT ASSESS CALIBRATION
# READINESS. A first version returned a "usable" verdict from the envelope and the
# years alone, and a review produced a tree with two observations per year and a tree
# whose u covered the northern half and v the southern half, both reported usable.
# Readiness is a calibration-gate question that belongs with the proposal, not here.
TARGET_LAT = (-50.0, 50.0)
TARGET_LON = (-155.0, 55.0)
TARGET_SPACING_DEG = 1.0
TARGET_PERIODS = {"1979-2010": (1979, 2010), "1981-2010": (1981, 2010)}
NEEDED_FIELDS = (("u", 700.0), ("v", 700.0))


def axis(ds, names):
    """One coordinate axis: its endpoints, count and first spacing for the reader, and a
    FINGERPRINT of the whole array for comparison, since a review showed two grids
    with equal endpoints and different resolution passing as one grid when only the
    endpoints were compared. The fingerprint is the sha256 of the coordinates rounded
    to six decimals and sorted ascending, so a reversed axis is the same grid."""
    import hashlib
    for n in names:
        if n in ds.variables:
            a = np.asarray(ds[n][:], dtype=float)
            step = float(np.unique(np.round(np.diff(a), 6))[0]) if a.size > 1 else None
            canonical = np.round(np.sort(a), 6).tobytes()
            return {"name": n, "n": int(a.size), "first": float(a[0]), "last": float(a[-1]),
                    "step": step, "fingerprint": hashlib.sha256(canonical).hexdigest()}
    return None


def describe(path):
    ds = nc.Dataset(path)
    try:
        info = {"size": os.path.getsize(path),
                "vars": [v for v in ds.variables if v not in ds.dimensions],
                "dims": {k: len(v) for k, v in ds.dimensions.items()}}
        for tn in ("time", "valid_time"):
            if tn in ds.variables:
                t = ds[tn]
                vals = np.asarray(t[:], dtype=float)
                cal = getattr(t, "calendar", "standard")
                dt = nc.num2date(vals[[0, -1]], t.units, cal)
                steps = np.unique(np.diff(vals))
                info["time"] = {"n": int(vals.size), "first": str(dt[0]), "last": str(dt[1]),
                                "units": t.units, "calendar": cal,
                                "unique_steps": [float(x) for x in steps[:5]]}
                break
        info["lat"] = axis(ds, ("latitude", "lat"))
        info["lon"] = axis(ds, ("longitude", "lon"))
        info["pressure_level"] = ([float(x) for x in np.asarray(ds["pressure_level"][:]).ravel()]
                                  if "pressure_level" in ds.variables else None)
        miss = {}
        for v in info["vars"]:
            if v in SKIP_VARS:
                continue
            var = ds[v]
            if var.ndim >= 2:
                a = var[:]
                masked = int(np.ma.count_masked(a)) if np.ma.isMaskedArray(a) else 0
                nan = int(np.isnan(np.ma.filled(a, 0)).sum())
                miss[v] = {"masked": masked, "nan": nan, "dtype": str(var.dtype),
                           "shape": list(a.shape)}
        info["missing"] = miss
        return info
    finally:
        ds.close()


def full_six_hourly_year(info):
    """Whether a file's time axis is a complete six-hourly calendar year: it starts at
    00Z on January 1, ends at 18Z on December 31, has one uniform six-hour step, and
    holds four steps per calendar day of that year."""
    t = info.get("time")
    if not t or len(t["unique_steps"]) != 1:
        return False
    year = int(t["first"][:4])
    days = 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365
    return (t["first"] == f"{year}-01-01 00:00:00" and t["last"] == f"{year}-12-31 18:00:00"
            and t["n"] == 4 * days)


def summarize_tree(rec):
    """Descriptive coverage of one tree, from the recorded coordinates, levels and
    time axes. No readiness verdict; see the module header."""
    files = [v for v in rec["files"].values() if "error" not in v]
    if not files:
        return {"verdict": "no readable file"}
    lat_lo = min(min(f["lat"]["first"], f["lat"]["last"]) for f in files if f.get("lat"))
    lat_hi = max(max(f["lat"]["first"], f["lat"]["last"]) for f in files if f.get("lat"))
    lon_lo = min(min(f["lon"]["first"], f["lon"]["last"]) for f in files if f.get("lon"))
    lon_hi = max(max(f["lon"]["first"], f["lon"]["last"]) for f in files if f.get("lon"))
    spacing = sorted({abs(f["lat"]["step"]) for f in files if f.get("lat") and f["lat"]["step"]}
                     | {abs(f["lon"]["step"]) for f in files if f.get("lon") and f["lon"]["step"]})
    combos = {}
    per_year = {}          # (combo, year) -> {"complete": bool, "axes": (lat, lon)}
    for f in files:
        levels = f.get("pressure_level") or [None]
        for var in f.get("missing", {}):
            for lev in levels:
                key = f"{var}@{int(lev) if lev is not None else 'surface'}"
                combos[key] = combos.get(key, 0) + 1
                if f.get("time"):
                    year = int(f["time"]["first"][:4])
                    axes = (f["lat"]["fingerprint"] if f.get("lat") else None,
                            f["lon"]["fingerprint"] if f.get("lon") else None)
                    per_year[(key, year)] = {"complete": full_six_hourly_year(f), "axes": axes}
    needed = [f"{v}@{int(l)}" for v, l in NEEDED_FIELDS]
    have_needed = all(k in combos for k in needed)
    years_needed = sorted(set.intersection(*({y for (k, y) in per_year if k == n}
                                             for n in needed))) if have_needed else []
    complete_years = [y for y in years_needed
                      if all(per_year[(n, y)]["complete"] for n in needed)]
    shared_grid_years = [y for y in years_needed
                         if len({per_year[(n, y)]["axes"] for n in needed}) == 1]
    missing_years = {name: sorted(set(range(a, b + 1)) - set(years_needed))
                     for name, (a, b) in TARGET_PERIODS.items()}
    envelope_reaches = (lat_lo <= TARGET_LAT[0] and lat_hi >= TARGET_LAT[1]
                        and lon_lo <= TARGET_LON[0] and lon_hi >= TARGET_LON[1])
    fine_enough = bool(spacing) and max(spacing) <= TARGET_SPACING_DEG
    notes = []
    if not have_needed:
        notes.append(f"lacks {[k for k in needed if k not in combos]}")
    if not envelope_reaches:
        notes.append(f"extent envelope latitude {lat_lo} to {lat_hi} and longitude {lon_lo} "
                     f"to {lon_hi} does not reach {TARGET_LAT} by {TARGET_LON}")
    if not fine_enough:
        notes.append(f"spacing {spacing} coarser than {TARGET_SPACING_DEG} degree")
    if have_needed and missing_years["1981-2010"]:
        notes.append(f"years without both needed fields vs 1981-2010: "
                     f"{len(missing_years['1981-2010'])}")
    if have_needed and len(complete_years) != len(years_needed):
        notes.append(f"years with both fields but not a complete six-hourly year: "
                     f"{len(years_needed) - len(complete_years)}")
    if have_needed and len(shared_grid_years) != len(years_needed):
        notes.append(f"years in which u and v do not share one grid: "
                     f"{len(years_needed) - len(shared_grid_years)}")
    return {"lat_extent_envelope": [lat_lo, lat_hi], "lon_extent_envelope": [lon_lo, lon_hi],
            "spacing_deg": spacing, "field_level_combinations": combos,
            "n_field_level_combinations": len(combos),
            "years_with_needed_fields": years_needed,
            "years_with_needed_fields_complete_six_hourly": complete_years,
            "years_with_needed_fields_on_one_grid": shared_grid_years,
            "missing_years": missing_years, "has_needed_fields": have_needed,
            "envelope_reaches_target_domain": envelope_reaches, "fine_enough": fine_enough,
            "calibration_readiness": "not assessed by this inventory",
            "notes": notes}


def render_report(out):
    lines = ["# ERA5 on disk", "",
             "Generated by scripts/inventory_era5.py from every file's own coordinates, "
             "levels and time axis. The target is version 1's buffered one-degree grid, "
             f"{TARGET_LAT[0]:.0f} to {TARGET_LAT[1]:.0f} latitude and {TARGET_LON[0]:.0f} to "
             f"{TARGET_LON[1]:.0f} longitude, with 700 hPa u and v six-hourly over 1979 to "
             "2010 (source-faithful population period) or 1981 to 2010. Nothing here "
             "downloads anything.", "",
             "This is a descriptive inventory. Calibration readiness is NOT assessed here. "
             "The latitude and longitude columns are the extent ENVELOPE over a tree, the "
             "union of every file's axes, which two components can each cover half of; the "
             "last two columns say in how many of the years holding both fields each file "
             "is a complete six-hourly calendar year and the two components share one grid.",
             "",
             "| Tree | Files | Unreadable | Masked or NaN cells | Spacing | Latitude envelope | "
             "Longitude envelope | Field and level combinations | Years with 700 hPa u and v | "
             "Missing vs 1979 to 2010 | Missing vs 1981 to 2010 | Complete years | "
             "One-grid years |",
             "|---|---:|---:|---:|---|---|---|---:|---|---:|---:|---:|---:|"]
    for name, rec in out["dirs"].items():
        sm = rec["summary"]
        if "lat_extent_envelope" not in sm:
            lines.append(f"| {name} | {rec['n_files']} | {rec['n_errors']} | | | | | | | | | | "
                         f"{sm['verdict']} |")
            continue
        yrs = sm["years_with_needed_fields"]
        span = f"{yrs[0]} to {yrs[-1]}" if yrs else "none"
        lines.append(
            f"| {name} | {rec['n_files']} | {rec['n_errors']} | {rec['masked_or_nan_cells']} | "
            f"{', '.join(str(x) for x in sm['spacing_deg'])} | "
            f"{sm['lat_extent_envelope'][0]} to {sm['lat_extent_envelope'][1]} | "
            f"{sm['lon_extent_envelope'][0]} to {sm['lon_extent_envelope'][1]} | "
            f"{sm['n_field_level_combinations']} | {span} | "
            f"{len(sm['missing_years']['1979-2010'])} | {len(sm['missing_years']['1981-2010'])} | "
            f"{len(sm['years_with_needed_fields_complete_six_hourly'])} of {len(yrs)} | "
            f"{len(sm['years_with_needed_fields_on_one_grid'])} of {len(yrs)} |")
    lines.append("")
    for name, rec in out["dirs"].items():
        sm = rec["summary"]
        if sm.get("notes"):
            lines.append(f"- {name}: " + ". ".join(sm["notes"]) + ".")
        combos = sm.get("field_level_combinations")
        if combos:
            lines.append(f"  Fields: " + ", ".join(f"{k} ({n} files)" for k, n in sorted(combos.items())) + ".")
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--root", default="data/era5")
    ap.add_argument("--out", default=None)
    ap.add_argument("--report", default=None, help="write the generated markdown summary")
    args = ap.parse_args(argv)
    out = {"generated_by": "scripts/inventory_era5.py", "root": args.root, "dirs": {}}
    for d in sorted(glob.glob(os.path.join(args.root, "*"))):
        if not os.path.isdir(d):
            continue
        rec = {"files": {}}
        for f in sorted(glob.glob(os.path.join(d, "*.nc"))):
            try:
                rec["files"][os.path.basename(f)] = describe(f)
            except Exception as e:                                # noqa: BLE001
                rec["files"][os.path.basename(f)] = {"error": repr(e),
                                                     "size": os.path.getsize(f)}
        rec["n_files"] = len(rec["files"])
        rec["n_errors"] = sum(1 for v in rec["files"].values() if "error" in v)
        rec["masked_or_nan_cells"] = sum(m["masked"] + m["nan"]
                                         for v in rec["files"].values()
                                         for m in v.get("missing", {}).values())
        rec["summary"] = summarize_tree(rec)
        out["dirs"][os.path.basename(d)] = rec
        print(f"{os.path.basename(d)}: {rec['n_files']} files, {rec['n_errors']} unreadable, "
              f"{rec['masked_or_nan_cells']} masked or NaN cells; "
              f"{rec['summary'].get('notes', rec['summary'].get('verdict'))}", flush=True)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=1, sort_keys=True)
        print(f"written to {args.out}", flush=True)
    if args.report:
        with open(args.report, "w") as fh:
            fh.write(render_report(out))
        print(f"report written to {args.report}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
