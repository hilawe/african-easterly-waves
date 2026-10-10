#!/usr/bin/env python3
"""Candidate waves for each storm event of selected seasons, every candidate kept, no wave
assigned.

For each storm in the pinned storm-event table whose season is requested, and for each of
its three events separately (first archived, first depression or stronger, first tropical
storm), every track of ONE track product version with an observation inside the time
window and the radius of the event point is a CANDIDATE, recorded with its distance, its
lag, the observation used, and how many other tracks compete. Two (radius, window)
settings are run side by side, the prototype's placeholder and a wider one, so the case
set shows how the candidate count depends on them. Nothing is adjudicated here: a storm is
STRAIGHTFORWARD when every event has exactly one candidate under both settings and it is
the same track, AMBIGUOUS when any event has two or more candidates under either setting,
and UNMATCHED when an event has none under both. These labels describe the candidate set,
not the physics, and are the cases a later association specification is tested on.

THE LINK TO THE TRACK PRODUCT is exact: the product identifier and version, the track
file's path inside the product and its SHA-256 as the product manifest lists it, the
track's index in that file, and the observation's time in days since 1900-01-01 00Z.
Everything downstream that annotates a wave observation, a visual reading or a convection
measurement, carries that same tuple, and a convection measurement at an observation
time is tagged with the storm events whose times it precedes or follows, so measurements
before and after cyclone formation stay distinguishable.

    python3 scripts/storm_case_candidates.py --product data/products/aewc-v2-era5-1979-2025_1.0.0-rc3 \\
        --storm-events docs/aewc_v2/artifacts/storm_events_na_v04r01_1979_2025.json --seasons 1990 2006 \\
        --out docs/aewc_v2/artifacts/storm_candidates_<product version>_<seasons>.json

The artifact's name carries no "_case_", which the residue accounting reserves for its own
case artifacts in the same directory.
"""
import argparse
import datetime as dt
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import exact_tracks as X  # noqa: E402
import season_metrics as S  # noqa: E402

SETTINGS = {"prototype": {"radius_km": 500.0, "window_hours": 6.0},
            "wider": {"radius_km": 750.0, "window_hours": 24.0}}
EVENTS = ("first_archived", "first_depression_or_stronger", "first_tropical_storm")
EPOCH = dt.datetime(1900, 1, 1, tzinfo=dt.timezone.utc)


def days_since_1900(iso):
    t = dt.datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
    return (t - EPOCH).total_seconds() / 86400.0


def great_circle_km(lat1, lon1, lat2, lon2):
    a, b = np.radians(lat1), np.radians(lat2)
    d = np.radians(lon2 - lon1)
    x = np.sin(a) * np.sin(b) + np.cos(a) * np.cos(b) * np.cos(d)
    return 6371.0 * np.arccos(np.clip(x, -1.0, 1.0))


def product_tracks(product, year):
    """The year's tracks from the product, with the file's digest as the manifest lists it,
    verified against the bytes read."""
    manifest = json.load(open(os.path.join(product, "MANIFEST.json")))
    rel = f"tracks/era5_{year}_tracks.mat"
    listed = manifest["files"][rel]["sha256"]
    with open(os.path.join(product, rel), "rb") as fh:
        blob = fh.read()
    import hashlib
    if hashlib.sha256(blob).hexdigest() != listed:
        raise SystemExit(f"REFUSED: {rel} does not have the digest the product manifest lists")
    tracks, _case = S.read_mat_tracks(blob)
    link = {"product": manifest["product"], "version": manifest["version"], "track_file": rel, "track_file_sha256": listed}
    return tracks, link


def candidates(tracks, event, radius_km, window_hours):
    """Every track with an observation within the window and radius of the event, with the
    nearest such observation's distance, lag and time."""
    t0 = days_since_1900(event["time"])
    out = []
    for i, tr in enumerate(tracks):
        near = np.abs(tr["time"] - t0) <= window_hours / 24.0 + 1e-9
        if not near.any():
            continue
        d = great_circle_km(event["lat"], event["lon"], tr["lat"][near], tr["lon"][near])
        if d.min() <= radius_km:
            k = int(np.argmin(d))
            idx = np.where(near)[0][k]
            out.append({"track_index": i, "distance_km": round(float(d[k]), 1),
                        "lag_hours": round(float((tr["time"][idx] - t0) * 24.0), 1),
                        "observation_time_days": float(tr["time"][idx]),
                        "observation_lat": float(tr["lat"][idx]), "observation_lon": float(tr["lon"][idx]),
                        "track_first_lon": float(tr["lon"][0]), "track_observations": int(tr["time"].size)})
    out.sort(key=lambda c: c["distance_km"])
    for c in out:
        c["competing"] = len(out) - 1
    return out


def classify(per_event):
    """From {event: {setting: [candidates]}} over the events the storm has."""
    counts = [len(per_event[e][s]) for e in per_event for s in SETTINGS]
    if counts and all(n == 1 for n in counts):
        chosen = {per_event[e][s][0]["track_index"] for e in per_event for s in SETTINGS}
        if len(chosen) == 1:
            return "straightforward"
    if any(n >= 2 for n in counts):
        return "ambiguous"
    if counts and all(n == 0 for n in counts):
        return "unmatched"
    return "mixed"


def build(product, storm_events_path, seasons):
    table = json.load(open(storm_events_path))
    out, by_year = {}, {}
    for sid, e in table["table"].items():
        if e["season"] not in seasons:
            continue
        year = e["season"]
        if year not in by_year:
            by_year[year] = product_tracks(product, year)
        tracks, link = by_year[year]
        per_event = {}
        for ev in EVENTS:
            if e[ev] is None:
                continue
            per_event[ev] = {name: candidates(tracks, e[ev], **cfg) for name, cfg in SETTINGS.items()}
        out[sid] = {"name": e["name"], "season": year, "class": e["class"], "events": {ev: e[ev] for ev in EVENTS},
                    "candidates": per_event, "candidate_set": classify(per_event), "track_product": link}
    summary = {}
    for v in out.values():
        summary[v["candidate_set"]] = summary.get(v["candidate_set"], 0) + 1
    return {"generated_by": "scripts/storm_case_candidates.py", "script_sha256": X.digest(os.path.abspath(__file__)),
            "storm_events": {"path": storm_events_path, "sha256": X.digest(storm_events_path)},
            "track_product": {y: link for y, (_, link) in by_year.items()},
            "settings": SETTINGS, "seasons": sorted(seasons),
            "link_schema": {"product": "the product identifier", "version": "the product version",
                            "track_file": "the track file's path inside the product", "track_file_sha256": "its digest as the product manifest lists it",
                            "track_index": "the track's index in that file", "observation_time_days": "the observation's time, days since 1900-01-01 00Z"},
            "assigns": "no wave, no flag, no probability: candidate sets only",
            "summary": summary, "storms": out}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--product", required=True)
    ap.add_argument("--storm-events", required=True)
    ap.add_argument("--seasons", nargs="+", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    payload = build(args.product, args.storm_events, set(args.seasons))
    try:
        X.publish_json(args.out, payload, exclusive=True)
    except FileExistsError:
        raise SystemExit(f"REFUSED: {args.out} exists and artifacts are never overwritten")
    print(f"{len(payload['storms'])} storms over {payload['seasons']}: {payload['summary']}; wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
