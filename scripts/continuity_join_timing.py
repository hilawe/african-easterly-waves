#!/usr/bin/env python3
"""A description read after section 15's rule was applied (section 16 of
`CONTINUITY_ACROSS_AFRICA_2026-11-09.md`), computed from the retained tracks of the false-join test.
For every switch and relay it gives the hours since the AEWC v2 track's previous accompanied
observation, and for every switch the distance to the nearest QTrack system it left behind. It
changes no outcome.

    python3 scripts/continuity_join_timing.py --out-dir DIR
"""

import argparse
import gzip
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
import continuity_census as C  # noqa: E402
import continuity_join_test as J  # noqa: E402
import continuity_stage_sample as S  # noqa: E402

with open(os.path.abspath(__file__), "rb") as _fh:
    SCRIPT_SHA256_AT_LOAD = hashlib.sha256(_fh.read()).hexdigest()


def event_timing(row, track, q_index, q_by_id):
    """(kind, hours since the previous accompanied observation, km to the nearest system left
    behind or None) for each switch and relay of one classed track."""
    acc = [float(d) for la, lo, d in zip(track["lat"], track["lon"], track["time"]) if C.companions(la, lo, d, q_index)]
    out = []
    for e in row["events"]:
        if e["kind"] not in ("switch", "relay"):
            continue
        k = acc.index(e["time"])
        if k == 0:
            raise SystemExit(f"REFUSED: a transition of track {row['id']} has no previous accompanied observation")
        km = None
        if e["kind"] == "switch":
            j = track["time"].index(e["time"]); la, lo = track["lat"][j], track["lon"][j]
            dist = []
            for a in e["lost"]:
                qa = q_by_id[a]; hit = np.flatnonzero(np.abs(np.asarray(qa["time"]) - e["time"]) < 1e-6)
                if hit.size:
                    dist.append(float(C.km(la, lo, qa["lat"][hit[0]], qa["lon"][hit[0]])))
            if not dist:
                raise SystemExit(f"REFUSED: a switch of track {row['id']} has no system left behind with a position")
            km = min(dist)
        out.append((e["kind"], round(24.0 * (e["time"] - acc[k - 1])), km))
    return out


def summary(timings):
    out = {}
    for kind in ("switch", "relay"):
        g = [h for k, h, _ in timings if k == kind]
        d = [x for k, _, x in timings if k == kind and x is not None]
        if g:
            out[kind] = {"n": len(g), "gap_hours_min": int(min(g)), "gap_hours_median": float(np.median(g)), "at_6_hours": sum(1 for h in g if h == 6),
                         "over_24_hours": sum(1 for h in g if h > 24)}
            if d:
                out[kind]["left_behind_km_median"] = round(float(np.median(d)), 1)
    return out


def season(year, out_dir):
    import qtrack_pairing_pilot as M
    path = os.path.join(out_dir, f"continuity_join_test_{year}.json.gz")
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        rec = json.load(fh)
    if rec["generated_by"]["sha256"] != J.SCRIPT_SHA256_AT_LOAD:
        raise SystemExit(f"REFUSED: the {year} record was not written by the current false-join test")
    art = json.load(open(rec["artifact"]["path"], encoding="utf-8"))
    if C.sha256_of(art["inputs"]["qtrack"]["path"]) != art["inputs"]["qtrack"]["sha256"]:
        raise SystemExit(f"REFUSED: the {year} QTrack file is not the one the artifact names")
    qt = M.load_qtrack(art["inputs"]["qtrack"]["path"])[0]
    q_index, q_by_id = C.index_by_time(qt), {t["id"]: t for t in qt}
    out = {}
    for name, cfg in rec["configs"].items():
        tracks = {t["id"]: t for t in cfg["tracks"]}
        timings = [x for row in cfg["multi_system"] for x in event_timing(row, tracks[row["id"]], q_index, q_by_id)]
        lost = {}
        for row in cfg["multi_system"]:
            for e in row["events"]:
                if e["kind"] == "switch":
                    for a in e["lost"]:
                        lost[a] = lost.get(a, 0) + 1
        out[name] = {**summary(timings), "systems_left_by_switches": len(lost), "most_left": max(lost.values()) if lost else 0}
    return {"year": year, "record": {"path": path, "sha256": C.sha256_of(path)}, "configs": out}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    p = os.path.join(a.out_dir, "continuity_join_timing.json")
    if os.path.exists(p):
        raise SystemExit(f"REFUSED: {p} exists and records are never overwritten")
    rec = {"generated_by": {"script": "scripts/continuity_join_timing.py", "sha256": SCRIPT_SHA256_AT_LOAD,
                            "status": "descriptive, read after section 15's rule was applied, changes no outcome"},
           "seasons": [season(y, a.out_dir) for y in S.YEARS]}
    with open(p, "x", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    print(json.dumps(rec["seasons"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
