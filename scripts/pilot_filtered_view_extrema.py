#!/usr/bin/env python3
"""Listings from a filtered seasonal view record (`pilot_season_filtered_view.py`), for the
candidate-finding reading of `EASTERN_DIRECTION_RECOMMENDATION_7_2026-10-15.md`.

`rank` lists, per record and band, the largest filtered band-mean magnitudes inside a
longitude range, each separated by at least a declared number of days from a larger one.
`track` lists, per step inside a declared window, the longitude and value of the band
mean's maximum and minimum inside a longitude range. Both are listings of positions and
values from the retained record. Neither is a slope: slopes are read by eye, as apparent
endpoint rates, and never across an unresolved join.

    python3 scripts/pilot_filtered_view_extrema.py rank RECORD... --lon 40 60 --separation-days 3 --top 6
    python3 scripts/pilot_filtered_view_extrema.py track RECORD --band northern --start 1990-08-10T00 \
        --end 1990-08-11T12 --lon 44 64 --every-hours 6
"""
import argparse
import json

import numpy as np


def load(path):
    r = json.load(open(path))
    lons = np.array(r["lons"], float)
    series = {b: np.array([[np.nan if v is None else v for v in row] for row in r["bands"][b]["series"]], float) for b in r["bands"]}
    return r, lons, series


def rank(path, lon, separation_days, top):
    r, lons, series = load(path)
    out = []
    cols = (lons >= lon[0]) & (lons <= lon[1])
    steps = int(round(separation_days * 24 / r["declared"]["time"]["hours"]))
    for band, s in series.items():
        a = np.abs(np.where(cols[None, :], s, np.nan))
        events = []
        for _ in range(top):
            if not np.isfinite(a).any():
                break
            i, j = np.unravel_index(np.nanargmax(a), a.shape)
            events.append({"time": r["times"][i], "lon": float(lons[j]), "value": float(s[i, j])})
            a[max(0, i - steps):i + steps + 1, :] = np.nan
        out.append((band, events))
    return r, out


def track(path, band, start, end, lon, every_hours):
    r, lons, series = load(path)
    s = series[band]
    cols = (lons >= lon[0]) & (lons <= lon[1])
    every = max(1, int(round(every_hours / r["declared"]["time"]["hours"])))
    rows = []
    for i, t in enumerate(r["times"]):
        if t < start or t > end or i % every:
            continue
        row = np.where(cols, s[i], np.nan)
        if not np.isfinite(row).any():
            rows.append({"time": t, "unsupported": True})
            continue
        j, k = int(np.nanargmax(row)), int(np.nanargmin(row))
        rows.append({"time": t, "max": float(row[j]), "max_lon": float(lons[j]), "min": float(row[k]), "min_lon": float(lons[k])})
    return r, rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    pr = sub.add_parser("rank"); pr.add_argument("records", nargs="+"); pr.add_argument("--lon", nargs=2, type=float, required=True)
    pr.add_argument("--separation-days", type=float, default=3.0); pr.add_argument("--top", type=int, default=6)
    pt = sub.add_parser("track"); pt.add_argument("record"); pt.add_argument("--band", required=True)
    pt.add_argument("--start", required=True); pt.add_argument("--end", required=True); pt.add_argument("--lon", nargs=2, type=float, required=True)
    pt.add_argument("--every-hours", type=int, default=6)
    a = ap.parse_args(argv)
    if a.mode == "rank":
        print(f"# rank: largest filtered band-mean magnitudes in {a.lon[0]:g} to {a.lon[1]:g} E, each at least {a.separation_days:g} days from a larger one, top {a.top}")
        for path in a.records:
            r, out = rank(path, a.lon, a.separation_days, a.top)
            for band, events in out:
                print(f"{r['year']} {r['level_hpa']} {r['role']:11s} {band:8s}: " + "; ".join(f"{e['time']} {e['lon']:.1f}E {e['value']:+.2f}" for e in events))
    else:
        r, rows = track(a.record, a.band, a.start, a.end, a.lon, a.every_hours)
        print(f"# track: {r['year']} {r['level_hpa']} hPa ({r['role']}) {a.band}, {a.start} to {a.end}, {a.lon[0]:g} to {a.lon[1]:g} E, every {a.every_hours} h (max@lon, min@lon, m/s)")
        for row in rows:
            if row.get("unsupported"):
                print(f"{row['time']}  unsupported")
            else:
                print(f"{row['time']}  {row['max']:+.2f}@{row['max_lon']:.1f}E  {row['min']:+.2f}@{row['min_lon']:.1f}E")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
