#!/usr/bin/env python3
"""The across-month tables of the eastern regime characterization
(`EASTERN_REGIME_CHARACTERIZATION_2026-10-18.md`, section 5), derived from the ten month
records written by `eastern_regime_views.py`: the ten-month mean of the 700 hPa northerly
persistence fraction by band and five-degree longitude bin, the cells whose ten-month
mean fraction reaches one half, and per band the count of months whose 700 hPa band-mean
ridge reads westward, standing or eastward under the stated reading rule. Numbers for the
record, not a detector.

    python3 scripts/eastern_regime_composite.py --out-dir DIR --draw DRAW.json --table OUT.md --out OUT.json
"""
import argparse
import json
import os

import numpy as np

STANDING_SLOPE = 0.1          # day per degree; a ridge slope at or under this magnitude reads as standing
LON_BINS = [(20.0, 24.5), (25.0, 29.5), (30.0, 34.5), (35.0, 39.5), (40.0, 44.5), (45.0, 49.5), (50.0, 54.5), (55.0, 59.5), (60.0, 64.5), (65.0, 69.5), (70.0, 75.0)]


def ridge_reading(r):
    """westward, eastward, standing or none, from a month record's ridge for one band."""
    if r["slope_days_per_degree"] is None:
        return "none"
    if abs(r["slope_days_per_degree"]) <= STANDING_SLOPE:
        return "standing"
    return "westward" if r["slope_days_per_degree"] < 0 else "eastward"


def composite(records):
    lat = np.array(records[0]["persistence"]["lat"]); lon = np.array(records[0]["lons"])
    for rec in records:
        if rec["persistence"]["lat"] != records[0]["persistence"]["lat"] or rec["lons"] != records[0]["lons"]:
            raise SystemExit("REFUSED: the month records are not on one grid")
    stack = np.array([rec["persistence"]["fraction"] for rec in records], float)
    mean = np.nanmean(stack, axis=0)
    bands = records[0]["bands"]
    table = {}
    for b, (lo, hi) in bands.items():
        rows = (lat >= lo - 1e-6) & (lat <= hi + 1e-6)
        table[b] = {}
        for (x0, x1) in LON_BINS:
            cols = (lon >= x0 - 1e-6) & (lon <= x1 + 1e-6)
            block = mean[np.ix_(rows, cols)]
            table[b][f"{x0:g} to {x1:g} E"] = None if np.all(np.isnan(block)) else float(np.nanmean(block))
    half = np.isfinite(mean) & (mean >= 0.5)
    cells = [{"lat": float(lat[j]), "lon": float(lon[i]), "mean_fraction": round(float(mean[j, i]), 3)} for j, i in zip(*np.nonzero(half))]
    ridges = {b: {k: 0 for k in ("westward", "standing", "eastward", "none")} for b in bands}
    speeds = {b: [] for b in bands}
    for rec in records:
        for b in bands:
            r = rec["lag_correlation"]["ridges"][b]
            kind = ridge_reading(r); ridges[b][kind] += 1
            if kind == "westward":
                speeds[b].append(r["speed_deg_per_day"])
    return {"months": [rec["label"] for rec in records], "standing_slope_rule": STANDING_SLOPE, "mean_fraction_by_band_and_bin": table,
            "cells_with_mean_fraction_at_least_half": cells, "cells_at_least_half": len(cells), "cells_valid": int(np.isfinite(mean).sum()),
            "ridge_readings_by_band": ridges, "westward_speeds_by_band": speeds}


def table_text(c):
    f = lambda x: "" if x is None else f"{x:.2f}"
    L = ["# the ten months together, derived from the month records (section 5)",
         f"the ten-month mean of the 700 hPa northerly persistence fraction (steps at or below -4 m/s over valid steps) by band and five-degree longitude bin; a ridge reads as standing when its fitted slope magnitude is at most {STANDING_SLOPE:g} day per degree, westward or eastward by the slope's sign, none when fewer than three offsets reached a correlation of 0.3", "",
         "| band | " + " | ".join(next(iter(c["mean_fraction_by_band_and_bin"].values())).keys()) + " |", "| --- |" + " --- |" * len(LON_BINS)]
    for b, row in c["mean_fraction_by_band_and_bin"].items():
        L.append(f"| {b} | " + " | ".join(f(v) for v in row.values()) + " |")
    L.append("")
    L.append(f"cells whose ten-month mean fraction is at least one half: {c['cells_at_least_half']} of {c['cells_valid']} valid cells")
    L.append("")
    L.append("| band | westward months | standing months | eastward months | no ridge | westward speeds (degrees per day) |")
    L.append("| --- |" + " --- |" * 5)
    for b, r in c["ridge_readings_by_band"].items():
        sp = ", ".join(f"{s:.1f}" for s in c["westward_speeds_by_band"][b])
        L.append(f"| {b} | {r['westward']} | {r['standing']} | {r['eastward']} | {r['none']} | {sp} |")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", required=True); ap.add_argument("--draw", required=True); ap.add_argument("--table", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    for p in (a.table, a.out):
        if os.path.exists(p):
            raise SystemExit(f"REFUSED: {p} exists and a record is never overwritten")
    draw = json.load(open(a.draw))
    months = [tuple(x) for x in draw["development_months"]] + [tuple(x) for x in draw["draw"]]
    records = [json.load(open(os.path.join(a.out_dir, f"eval60_regime_{y}_{m:02d}.json"))) for y, m in months]
    c = composite(records)
    c["generated_by"] = {"script": "scripts/eastern_regime_composite.py", "month_records": [rec["generated_by"]["sha256"][:16] for rec in records]}
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(c, fh, indent=1)
    t = table_text(c)
    with open(a.table, "w", encoding="utf-8") as fh:
        fh.write(t)
    print(t)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
