#!/usr/bin/env python3
"""Row-wise measurements for a field-first reference entry of one eastern interval, from
the retained ERA5 level files and a declared annotation
(`EASTERN_DIRECTION_RECOMMENDATION_8_2026-10-15.md`, section 8).

Per six-hourly step and per integer latitude row inside a declared box it lists every local
minimum of the meridional wind along longitude and every local maximum of the relative
vorticity, under the field renderer's validity (a wind cell valid where the surface
pressure is above the level strictly and the winds are finite; a vorticity cell valid only
where it and its four stencil neighbors are valid wind cells; box edges blank). A declared
annotation names, per step, the longitude window and rows of the structure sought (I1), of
the second core (E) and of the named vorticity features (ZA, ZS). Extrema are labeled by
window and none is discarded. Four measurements are kept separate: the I1 wind minimum on
the row (the deepest inside the window, others recorded as alternatives), the band's
western edge inside the window under a declared threshold with one sensitivity value, the
eastern reversal (the first crossing from northerly to southerly east of the wind
minimum, interpolated between integer points, interrupted where an invalid cell comes
first), and the vorticity maximum of each named feature. Rows outside the annotation at a
step are abstained. Nothing here is a slope, a track or an identity.

    python3 scripts/pilot_i1_entry_listing.py --year 1990 --start 1990-08-08T12 --end 1990-08-15T12 \
        --annotation <json> --out <json> --table <md>
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pilot_field_sequence as F  # noqa: E402

DATA_DIR = "data/era5/region6h"
TABLE_ROWS = (14, 18, 22, 26)


def sha256_of(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def local_extrema(values, valid, kind):
    """Indices of interior local minima (kind 'min') or maxima ('max') among valid cells,
    where both neighbors are valid and strictly worse, plus a flat run's first cell."""
    out = []
    for j in range(1, len(values) - 1):
        if not (valid[j] and valid[j - 1] and valid[j + 1]):
            continue
        a, b, c = values[j - 1], values[j], values[j + 1]
        if kind == "min" and b < a and b <= c and (b < c or not out or out[-1] != j - 1):
            out.append(j)
        if kind == "max" and b > a and b >= c and (b > c or not out or out[-1] != j - 1):
            out.append(j)
    return out


def listing_grid(path, year, grid="integer"):
    """The listing's grid from a retained level file. `integer` is the frozen behavior of the
    entries (the file's integer-degree latitudes and longitudes). `native-lon` keeps every
    longitude column of the file at its own spacing with integer latitude rows, added for
    the second development package (2026-10-17) because the files are on a half-degree grid
    and the integer sampling was the listing's own choice. The reader's adjacency rule is
    in rows of one degree, so latitude stays at integer rows."""
    g = F.grid_from_level_file(path, year)
    if grid == "integer":
        return {**g, "grid": "integer", "lon_spacing_deg": float(np.median(np.abs(np.diff(g["longrid"][0, :]))))}
    if grid == "native-lon":
        import netCDF4
        d = netCDF4.Dataset(path)
        try:
            lon = np.asarray(d.variables["longitude"][:], float)
        finally:
            d.close()
        la = g["latgrid"][:, 0]
        latgrid, longrid = np.meshgrid(la, lon, indexing="ij")
        return {**g, "latgrid": latgrid, "longrid": longrid, "grid": "native-lon", "lon_spacing_deg": float(np.median(np.abs(np.diff(lon))))}
    raise SystemExit(f"REFUSED: unknown grid {grid!r}")


def step_annotation(annotation, stamp):
    for s in annotation["steps"]:
        if s["from"] <= stamp <= s["to"]:
            return s
    return None


def label(lon, row, ann, key):
    w = ann.get(key)
    return bool(w) and w["lon"][0] <= lon <= w["lon"][1] and w["rows"][0] <= row <= w["rows"][1]


def row_measurements(lons, v, vvalid, zeta, zvalid, ann, row, thresholds):
    """One row at one step. `ann` is the step's annotation or None (abstained)."""
    lons = np.asarray(lons, float); v = np.asarray(v, float); zeta = np.asarray(zeta, float)
    vvalid = np.asarray(vvalid, bool); zvalid = np.asarray(zvalid, bool)
    out = {"row_lat": float(row), "wind_valid_cells": int(vvalid.sum()), "zeta_valid_cells": int(zvalid.sum()),
           "wind_minima": [], "zeta_maxima": [], "i1": None, "i1_alternatives": [], "edge": {}, "reversal": None, "za": None, "zs": None}
    if ann is None:
        out["status"] = "abstained"
        return out
    mins = [{"lon": float(lons[j]), "v": float(v[j]), "label": ("I1" if label(lons[j], row, ann, "i1") else "E" if label(lons[j], row, ann, "e")
                                                                  else "W" if lons[j] < ann["i1"]["lon"][0] else "other")} for j in local_extrema(v, vvalid, "min")]
    out["wind_minima"] = mins
    i1 = sorted([m for m in mins if m["label"] == "I1"], key=lambda m: m["v"])
    in_rows = ann["i1"]["rows"][0] <= row <= ann["i1"]["rows"][1]
    if not in_rows:
        out["status"] = "abstained"
    elif not i1:
        out["status"] = "no I1 minimum on the row"
    else:
        out["status"] = "assigned"
        out["i1"] = i1[0]
        out["i1_alternatives"] = i1[1:]
        lo, hi = ann["i1"]["lon"]
        for name, thr in thresholds.items():
            cells = [float(lons[j]) for j in range(len(lons)) if lo <= lons[j] <= hi and vvalid[j] and v[j] < thr]
            out["edge"][name] = {"threshold_ms": thr, "lon": (min(cells) if cells else None)}
        j0 = int(np.argmin(np.abs(lons - i1[0]["lon"])))
        rev = {"lon": None, "status": "none within the box"}
        for j in range(j0, len(lons) - 1):
            if not (vvalid[j] and vvalid[j + 1]):
                rev = {"lon": None, "status": f"interrupted by an invalid cell at {lons[j + 1] if vvalid[j] else lons[j]:g} E"}
                break
            if v[j] <= 0.0 < v[j + 1]:
                frac = (0.0 - v[j]) / (v[j + 1] - v[j])
                rev = {"lon": float(lons[j] + frac * (lons[j + 1] - lons[j])), "status": "available"}
                break
        out["reversal"] = rev
    maxs = [{"lon": float(lons[j]), "zeta_1e5": float(zeta[j] * 1e5), "label": ("ZA" if label(lons[j], row, ann, "za") else "ZS" if label(lons[j], row, ann, "zs") else "other")}
            for j in local_extrema(zeta, zvalid, "max")]
    out["zeta_maxima"] = maxs
    for key in ("za", "zs"):
        feats = sorted([m for m in maxs if m["label"] == key.upper()], key=lambda m: -m["zeta_1e5"])
        out[key] = {"lon": feats[0]["lon"], "zeta_1e5": feats[0]["zeta_1e5"], "alternatives": feats[1:]} if feats else None
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("--start", "--end", "--annotation", "--out", "--table"):
        ap.add_argument(name, required=True)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--every-hours", type=int, default=6)
    ap.add_argument("--grid", choices=("integer", "native-lon"), default="integer", help="integer (the entries' frozen sampling, default) or native-lon (every longitude column of the file, integer rows)")
    a = ap.parse_args(argv)
    for p in (a.out, a.table):
        if os.path.exists(p):
            raise SystemExit(f"REFUSED: {p} exists and a record is never overwritten")
    ann = json.load(open(a.annotation))
    level = float(ann["level_hpa"])
    files = {n: os.path.join(DATA_DIR, f"era5_{n}_{a.year}_6h_region.nc") for n in (f"u{level:g}", f"v{level:g}", "sp")}
    grid = listing_grid(files[f"u{level:g}"], a.year, a.grid)
    lat = grid["latgrid"][:, 0]; lon = grid["longrid"][0, :]
    (la0, la1), (lo0, lo1) = ann["box"]["lat"], ann["box"]["lon"]
    rows = np.flatnonzero((lat >= la0 - 1e-6) & (lat <= la1 + 1e-6)); cols = np.flatnonzero((lon >= lo0 - 1e-6) & (lon <= lo1 + 1e-6))
    steps = F.sequence_steps(grid["time"], F.to_day(a.start), F.to_day(a.end), a.every_hours)
    if not steps:
        raise SystemExit("REFUSED: no steps in the window")
    step_days = [float(grid["time"][k]) for k in steps]
    lat_vals, lon_vals = lat[rows], lon[cols]
    u = F.field_on(files[f"u{level:g}"], "u", "m s**-1", lat_vals, lon_vals, step_days, level)
    v = F.field_on(files[f"v{level:g}"], "v", "m s**-1", lat_vals, lon_vals, step_days, level)
    sp = F.surface_pressure_on(files["sp"], lat_vals, lon_vals, step_days)
    thresholds = ann["edge_thresholds_ms"]
    record_steps = []
    for k, day in enumerate(step_days):
        stamp = (F.EPOCH + dt.timedelta(days=day)).strftime("%Y-%m-%dT%H")
        wcls = F.terrain_classes(sp[k], np.where(np.isfinite(u[k]) & np.isfinite(v[k]), v[k], np.nan), level)
        zeta, zcls = F.relative_vorticity(u[k], v[k], lat_vals, lon_vals, wcls)
        sa = step_annotation(ann, stamp)
        per_row = []
        for r in range(len(lat_vals)):
            if abs(lat_vals[r] - round(lat_vals[r])) > 1e-6:
                continue
            per_row.append(row_measurements(lon_vals, v[k][r], wcls[r] == "valid", zeta[r], zcls[r] == "valid", sa, int(round(lat_vals[r])), thresholds))
        record_steps.append({"time": stamp, "annotation": None if sa is None else {kk: sa[kk] for kk in sa if kk != "note"}, "rows": per_row,
                             "classes": {"wind": {n: int((wcls == n).sum()) for n in ("valid", "terrain", "sp_missing", "wind_missing")},
                                         "zeta": {n: int((zcls == n).sum()) for n in F.ZETA_CLASSES}}})
    here = os.path.dirname(os.path.abspath(__file__))
    out = {"generated_by": "scripts/pilot_i1_entry_listing.py", "script_sha256": sha256_of(os.path.abspath(__file__)),
           "reuses": {"grid_from_level_file, field_on, surface_pressure_on, terrain_classes, relative_vorticity, sequence_steps": "scripts/pilot_field_sequence.py, unchanged",
                      "renderer_sha256": sha256_of(os.path.join(here, "pilot_field_sequence.py"))},
           "year": a.year, "level_hpa": level, "sources": {n: {"path": p, "sha256": sha256_of(p)} for n, p in files.items()},
           "annotation": {"path": a.annotation, "sha256": sha256_of(a.annotation), "content": ann},
           "declared": {"window": [a.start, a.end], "every_hours": a.every_hours, "box": ann["box"], "rows": [int(round(x)) for x in lat_vals if abs(x - round(x)) < 1e-6],
                        "grid": grid["grid"], "lon_spacing_deg": grid["lon_spacing_deg"],
                        "lons": lon_vals.tolist(), "validity": "the renderer's: wind valid where surface pressure above the level strictly and winds finite; vorticity valid only with four valid stencil neighbors; box edges blank",
                        "measurements": ["I1 wind minimum (deepest local minimum inside the window, alternatives kept)", "western edge inside the window under each threshold",
                                         "eastern reversal east of the I1 minimum, interpolated, interrupted or none", "ZA and ZS vorticity maxima (largest local maximum inside each window, alternatives kept)"],
                        "label": "positions by time and row, availability apart from geometry, no slope, no track, no identity"},
           "steps": record_steps}
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    lines = [f"# I1 entry listing, {a.year}, {level:g} hPa, rows {', '.join(str(r) + ' N' for r in TABLE_ROWS)}; per cell: I1 wind-min lon (value) | edge at {thresholds['primary']:g} / {thresholds['sensitivity']:g} | reversal | ZA | ZS",
             "", "| step | " + " | ".join(f"{r} N" for r in TABLE_ROWS) + " |", "| --- |" + " --- |" * len(TABLE_ROWS)]
    for s in record_steps:
        cells = []
        for r in TABLE_ROWS:
            m = next((x for x in s["rows"] if int(x["row_lat"]) == r), None)
            if m is None or m["status"] != "assigned":
                cells.append(m["status"] if m else "no row")
                continue
            e = m["edge"]; rv = m["reversal"]
            def f(x): return "-" if x is None else f"{x:g}"
            cells.append(f"{m['i1']['lon']:g} ({m['i1']['v']:.1f}) | {f(e['primary']['lon'])} / {f(e['sensitivity']['lon'])} | {f(rv['lon']) if rv['lon'] is not None else rv['status'].split(' ')[0]} | "
                         f"{f(m['za']['lon']) if m['za'] else '-'} | {f(m['zs']['lon']) if m['zs'] else '-'}")
        lines.append(f"| {s['time']} | " + " | ".join(cells) + " |")
    with open(a.table, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    n_assigned = sum(1 for s in record_steps for m in s["rows"] if m["status"] == "assigned")
    n_rev = sum(1 for s in record_steps for m in s["rows"] if m["status"] == "assigned" and m["reversal"]["lon"] is not None)
    print(f"{len(record_steps)} steps, {len(record_steps[0]['rows'])} rows; {n_assigned} row-steps assigned to I1, {n_rev} with a reversal available. Record {a.out}, table {a.table}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
