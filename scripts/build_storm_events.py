#!/usr/bin/env python3
"""The pinned storm-event table for wave-to-storm association, and nothing more.

For every North Atlantic storm in IBTrACS v04r01 over the requested seasons, three events
are kept SEPARATE, because a best track begins at whatever stage the agency first analyzed
and not at a wave precursor: the FIRST ARCHIVED point, the first point at TROPICAL
DEPRESSION strength or stronger, and the first TROPICAL STORM point, each with its time,
position and status, all by the responsible agency's status series (USA_STATUS). Storms
whose archive begins above depression strength, and storms whose first archived nature is
subtropical, are classed as such rather than folded in. Provisional rows (TRACK_TYPE other
than main) are excluded and counted.

THE TABLE ASSIGNS NO WAVE, NO FLAG AND NO PROBABILITY. It is the storm side of an
association, pinned to the IBTrACS file's digest, so that any later association names the
exact storm events it used. A wave association is a separate table bound to one track
product version (scripts/storm_case_candidates.py prepares candidates for the pilot).

IBTrACS reads: the file marks the North Atlantic basin as "NA", which a default CSV read
turns into a missing value, so nothing here uses default missing-value parsing.

    python3 scripts/build_storm_events.py --ibtracs data/aewc_v2_pilot/ibtracs.NA.list.v04r01.csv \\
        --seasons 1979 2025 --out docs/aewc_v2/artifacts/storm_events_na_v04r01_1979_2025.json
"""
import argparse
import hashlib
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import exact_tracks as X  # noqa: E402

DEPRESSION_OR_STRONGER = ("TD", "TS", "HU")
TROPICAL_STORM_OR_STRONGER = ("TS", "HU")
SUBTROPICAL = ("SS", "SD")
COLUMNS = ["SID", "SEASON", "NAME", "ISO_TIME", "NATURE", "LAT", "LON", "USA_STATUS", "USA_AGENCY", "TRACK_TYPE", "BASIN"]


def read_ibtracs(path):
    """The file's rows with the status columns as strings and no default missing-value
    parsing, plus the file's digest."""
    with open(path, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    df = pd.read_csv(path, skiprows=[1], usecols=COLUMNS, dtype=str, keep_default_na=False, low_memory=False)
    df["SEASON"] = df["SEASON"].astype(int)
    df["LAT"], df["LON"] = df["LAT"].astype(float), df["LON"].astype(float)
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], utc=True)
    return df, digest


def _point(row):
    return {"time": row.ISO_TIME.strftime("%Y-%m-%dT%H:%M:%SZ"), "lat": float(row.LAT), "lon": float(row.LON),
            "usa_status": row.USA_STATUS, "nature": row.NATURE}


def storm_events(df, seasons):
    """One entry per storm: the three events, each None when the storm never reaches that
    stage, and the storm's class."""
    lo, hi = seasons
    main = df[(df.SEASON >= lo) & (df.SEASON <= hi) & (df.TRACK_TYPE == "main")]
    excluded = int(((df.SEASON >= lo) & (df.SEASON <= hi) & (df.TRACK_TYPE != "main")).sum())
    out = {}
    for sid, g in main.sort_values(["SID", "ISO_TIME"]).groupby("SID", sort=True):
        first = g.iloc[0]
        td = g[g.USA_STATUS.isin(DEPRESSION_OR_STRONGER)]
        ts = g[g.USA_STATUS.isin(TROPICAL_STORM_OR_STRONGER)]
        if first.USA_STATUS in TROPICAL_STORM_OR_STRONGER:
            klass = "archive begins at tropical-storm strength or stronger"
        elif first.USA_STATUS in SUBTROPICAL or first.NATURE == "SS":
            klass = "subtropical at first archived point"
        elif td.empty:
            klass = "never reaches depression strength in the agency series"
        elif first.USA_STATUS == "TD":
            klass = "archive begins at depression strength"
        else:
            klass = "archive begins below depression strength, then reaches it"
        out[sid] = {"name": first.NAME, "season": int(first.SEASON), "basin": first.BASIN,
                    "n_points": int(len(g)), "agencies": sorted({a for a in g.USA_AGENCY if a.strip()}),
                    "first_archived": _point(first),
                    "first_depression_or_stronger": _point(td.iloc[0]) if not td.empty else None,
                    "first_tropical_storm": _point(ts.iloc[0]) if not ts.empty else None,
                    "class": klass}
    return out, excluded


def build(path, seasons):
    df, digest = read_ibtracs(path)
    events, excluded = storm_events(df, seasons)
    classes = {}
    for e in events.values():
        classes[e["class"]] = classes.get(e["class"], 0) + 1
    return {"generated_by": "scripts/build_storm_events.py", "script_sha256": X.digest(os.path.abspath(__file__)),
            "ibtracs": {"path": path, "sha256": digest, "version": "v04r01", "basin_file": "NA"},
            "seasons": list(seasons), "status_series": "USA_STATUS",
            "events": {"first_archived": "the storm's first point in the archive, whatever its stage",
                       "first_depression_or_stronger": f"the first point with USA_STATUS in {list(DEPRESSION_OR_STRONGER)}",
                       "first_tropical_storm": f"the first point with USA_STATUS in {list(TROPICAL_STORM_OR_STRONGER)}"},
            "provisional_rows_excluded": excluded, "storms": len(events), "classes": classes,
            "assigns": "no wave, no flag, no probability",
            "table": events}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ibtracs", default="data/aewc_v2_pilot/ibtracs.NA.list.v04r01.csv")
    ap.add_argument("--seasons", nargs=2, type=int, default=(1979, 2025))
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    payload = build(args.ibtracs, tuple(args.seasons))
    try:
        X.publish_json(args.out, payload, exclusive=True)
    except FileExistsError:
        raise SystemExit(f"REFUSED: {args.out} exists and artifacts are never overwritten")
    print(f"{payload['storms']} storms, {payload['provisional_rows_excluded']} provisional rows excluded, classes {payload['classes']}; wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
