#!/usr/bin/env python3
"""A case-sized GridSat-B1 brightness-temperature sequence along one track, for the
eastern pilot's visual audit, bound to the exact track product and observation.

For one track, identified by the link tuple (product, version, track file and its digest,
track index), every observation gets the infrared brightness temperature (irwin_cdr) in a
box around its position at the nearest 3-hourly GridSat-B1 time, from the retained cropped
imagery where it exists, with the file's own time, the box, and summary readings: the
minimum, the share of the box colder than a stated threshold with the count of those
pixels (a rounded share of 0.000 is not an absence), and the fraction of the returned
pixels that is missing. A box that reaches beyond the retained crop's extent is
reported as PARTIAL spatial coverage, with the clipped width on each side and the covered
fraction of the requested box, separately from missing values inside the returned pixels
(a box wholly outside the crop is NONE, with no pixels and no readings, never an error),
and a file whose internal timestamp is not the requested timestep yields no reading at all.
Where no retained file exists for a timestep, the observation is recorded as NOT COVERED
unless a bounded fetch is allowed, and nothing is ever filled. 1979 has no
GridSat-B1 at all, and that stays a gap.

Every reading is tagged against the storm events of any storm the caller names, as before
or after each of the three events, so convection before and after cyclone formation stays
distinguishable. The readings are context for a human reading of the wind fields. They
are neither OLR nor rainfall, and visible convection is not a requirement for accepting a
wave: a sequence with no cold cloud is a sequence with no cold cloud, not a rejection.

    python3 scripts/gridsat_case_sequence.py --product <product dir> --year 1990 --track-index 12 \\
        --imagery data/gridsat_jas --out <json> [--fetch-missing 8] [--storm-events <json> --storm <SID>]
"""
import argparse
import datetime as dt
import glob
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import exact_tracks as X  # noqa: E402
import season_metrics as S  # noqa: E402

EPOCH = dt.datetime(1900, 1, 1)
COLD_K = 240.0
HALF_WIDTH_DEG = 5.0
EVENTS = ("first_archived", "first_depression_or_stronger", "first_tropical_storm")


def nearest_gridsat_time(days):
    """The 3-hourly GridSat time nearest an observation time in days since 1900."""
    t = EPOCH + dt.timedelta(days=float(days))
    hour = int(round(t.hour / 3.0)) * 3
    base = t.replace(minute=0, second=0, microsecond=0, hour=0) + dt.timedelta(hours=hour)
    return base


def retained_file(imagery, when):
    path = os.path.join(imagery, f"gridsat_{when:%Y%m%d%H}.nc")
    return path if os.path.exists(path) else None


def read_box(path, lat, lon, half=HALF_WIDTH_DEG):
    """The brightness temperature in the requested box as far as the file covers it, the
    returned coordinates, the file's own timestamp, and how much of the requested box the
    file's extent covers (clipped width per side and the covered fraction)."""
    import xarray as xr
    with xr.open_dataset(path) as ds:
        tb = ds["irwin_cdr"].isel(time=0)
        lats, lons = np.asarray(tb["lat"].values, float), np.asarray(tb["lon"].values, float)
        rows = np.where((lats >= lat - half) & (lats <= lat + half))[0]
        cols = np.where((lons >= lon - half) & (lons <= lon + half))[0]
        if rows.size and cols.size:
            values = np.asarray(tb.isel(lat=rows, lon=cols).values, float)
        else:
            # A BOX WHOLLY OUTSIDE THE CROP ON EITHER AXIS: no pixels, reported as such below,
            # never an error (a track south of 25 S or east of 75 E has no retained imagery)
            values = np.zeros((rows.size, cols.size), float)
        when = dt.datetime.fromtimestamp(int(ds["time"].values[0].astype("datetime64[s]").astype(int)), dt.timezone.utc).replace(tzinfo=None)
        extent = {"south": float(lats.min()), "north": float(lats.max()), "west": float(lons.min()), "east": float(lons.max())}
    clipped = {"north": round(max(0.0, (lat + half) - extent["north"]), 3), "south": round(max(0.0, extent["south"] - (lat - half)), 3),
               "east": round(max(0.0, (lon + half) - extent["east"]), 3), "west": round(max(0.0, extent["west"] - (lon - half)), 3)}
    covered_lat = max(0.0, 2 * half - clipped["north"] - clipped["south"])
    covered_lon = max(0.0, 2 * half - clipped["east"] - clipped["west"])
    fraction = round(covered_lat * covered_lon / (2 * half) ** 2, 4)
    coverage = {"spatial_coverage": "complete" if not any(clipped.values()) else "partial" if fraction > 0 else "none",
                "clipped_deg": clipped, "covered_fraction_of_requested_box": fraction}
    return values, lats[rows], lons[cols], when, coverage


def readings(tb):
    if tb.size == 0:
        return {"min_k": None, "cold_fraction": None, "cold_cells": 0, "missing_fraction": None, "cells": 0}
    finite = np.isfinite(tb)
    if not finite.any():
        return {"min_k": None, "cold_fraction": None, "cold_cells": 0, "missing_fraction": 1.0, "cells": int(tb.size)}
    cold = tb[finite] < COLD_K
    # THE COUNT IS KEPT BESIDE THE ROUNDED SHARE: a few cold pixels in a 143 by 143 box round
    # to a share of 0.000, and a rounded zero is not an absence
    return {"min_k": round(float(np.nanmin(tb)), 1), "cold_fraction": round(float(np.mean(cold)), 3), "cold_cells": int(cold.sum()),
            "missing_fraction": round(float(1.0 - finite.mean()), 3), "cells": int(tb.size)}


def event_tags(days, storm):
    """For each storm event the storm has, whether this observation is before or after it."""
    if storm is None:
        return None
    out = {}
    for ev in EVENTS:
        e = storm.get(ev)
        if e is None:
            out[ev] = "the storm has no such event"
            continue
        t_ev = (dt.datetime.strptime(e["time"], "%Y-%m-%dT%H:%M:%SZ") - EPOCH).total_seconds() / 86400.0
        out[ev] = "before" if days < t_ev else "at or after"
    return out


def fetch_missing(when, box, out_dir):
    """One bounded fetch of a GridSat-B1 timestep from the public bucket, cropped to the
    box, written beside the retained imagery under the retained naming. Returns the path."""
    from aew.data.gridsat import S3, _fetch_crop
    url = S3.format(year=when.year, mm=when.month, dd=when.day, hh=when.hour)
    path = os.path.join(out_dir, f"gridsat_{when:%Y%m%d%H}.nc")
    _fetch_crop(url, path, box)
    return path


def sequence(product, year, track_index, imagery, storm=None, fetch_limit=0, fetch_box=(35.0, -45.0, -25.0, 75.0), fetch_dir=None):
    manifest = json.load(open(os.path.join(product, "MANIFEST.json")))
    rel = f"tracks/era5_{year}_tracks.mat"
    with open(os.path.join(product, rel), "rb") as fh:
        blob = fh.read()
    digest = hashlib.sha256(blob).hexdigest()
    if digest != manifest["files"][rel]["sha256"]:
        raise SystemExit(f"REFUSED: {rel} does not have the digest the product manifest lists")
    tracks, _case = S.read_mat_tracks(blob)
    if not 0 <= track_index < len(tracks):
        raise SystemExit(f"REFUSED: track {track_index} is not in {rel}, which holds {len(tracks)}")
    tr = tracks[track_index]
    link = {"product": manifest["product"], "version": manifest["version"], "track_file": rel,
            "track_file_sha256": digest, "track_index": int(track_index)}
    obs, fetched = [], 0
    for k in range(tr["time"].size):
        days, lat, lon = float(tr["time"][k]), float(tr["lat"][k]), float(tr["lon"][k])
        when = nearest_gridsat_time(days)
        path = retained_file(imagery, when)
        if path is None and fetched < fetch_limit and when.year >= 1980:
            path = fetch_missing(when, fetch_box, fetch_dir or imagery)
            fetched += 1
        entry = {"observation_time_days": days, "lat": lat, "lon": lon, "gridsat_time": when.strftime("%Y-%m-%dT%H:00Z"),
                 "storm_events": event_tags(days, storm)}
        if path is None:
            entry["coverage"] = "not covered" if when.year >= 1980 else "no GridSat-B1 before 1980"
        else:
            tb, lats, lons, file_time, spatial = read_box(path, lat, lon)
            entry.update({"file": os.path.relpath(path), "file_sha256": X.digest(path), "file_time": file_time.strftime("%Y-%m-%dT%H:%M:%SZ")})
            if file_time != when:
                # THE FILE'S OWN TIMESTAMP MUST BE THE REQUESTED TIMESTEP: a file named for one
                # time and holding another yields no reading, never a reading filed under the name
                entry["coverage"] = "file timestamp does not match the requested timestep"
            else:
                entry.update({"coverage": "retained imagery", "box_deg": HALF_WIDTH_DEG,
                              "box_lat": [float(lats.min()), float(lats.max())] if lats.size else None,
                              "box_lon": [float(lons.min()), float(lons.max())] if lons.size else None, **spatial, **readings(tb)})
        obs.append(entry)
    covered = sum(1 for o in obs if o["coverage"] == "retained imagery")
    return {"generated_by": "scripts/gridsat_case_sequence.py", "script_sha256": X.digest(os.path.abspath(__file__)),
            "link": link, "storm": None if storm is None else storm.get("sid"),
            "cold_threshold_k": COLD_K, "half_width_deg": HALF_WIDTH_DEG,
            "readings_are": "infrared brightness temperature, neither OLR nor rainfall, context for a wind-field reading and never a requirement",
            "observations": len(obs), "covered": covered, "fetched": fetched, "sequence": obs}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--product", required=True)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--track-index", type=int, required=True)
    ap.add_argument("--imagery", default="data/gridsat_jas")
    ap.add_argument("--storm-events", default=None, help="the pinned storm-event table")
    ap.add_argument("--storm", default=None, help="a storm identifier in that table, for the before-and-after tags")
    ap.add_argument("--fetch-missing", type=int, default=0, help="at most this many timesteps fetched from the public bucket")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    storm = None
    if args.storm:
        table = json.load(open(args.storm_events))["table"]
        storm = dict(table[args.storm], sid=args.storm)
    payload = sequence(args.product, args.year, args.track_index, args.imagery, storm, args.fetch_missing)
    try:
        X.publish_json(args.out, payload, exclusive=True)
    except FileExistsError:
        raise SystemExit(f"REFUSED: {args.out} exists and artifacts are never overwritten")
    print(f"{payload['covered']} of {payload['observations']} observations covered, {payload['fetched']} fetched; wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
