#!/usr/bin/env python3
"""The nearest AEWC v2 position to each QTrack position, an addition to the census of
`CONTINUITY_ACROSS_AFRICA_2026-11-09.md` made after the census tables were read (its section 7).

For every observation of the QTrack cohort tracks that start between 10 and 30 E and reach the
Atlantic side, up to and including the first Atlantic-side observation, it finds the nearest
AEWC v2 observation at the same six-hourly time inside the African and Atlantic sector (40 W to
40 E, 0 to 35 N) and records the distance and the latitude offset. Proximity between stored
positions says nothing about identity, and the nearest position may belong to another
disturbance. Seasons are the census seasons, so reserved seasons and 2005 are not read.

    python3 scripts/continuity_nearest_offsets.py --out-dir DIR
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
import continuity_census as C  # noqa: E402

with open(os.path.abspath(__file__), "rb") as _fh:
    SCRIPT_SHA256_AT_LOAD = hashlib.sha256(_fh.read()).hexdigest()

SECTOR = (-40.0, 40.0, 0.0, 35.0)                    # lon_min, lon_max, lat_min, lat_max


def nearest_in_sector(lat, lon, day, index, sector=SECTOR):
    """(distance km, latitude offset other minus this) to the nearest observation of the other
    record at this step inside the sector, or None when the sector holds none at that step."""
    hit = index.get(C.time_key(day))
    if hit is None:
        return None
    la, lo, _ = hit
    m = (lo >= sector[0]) & (lo <= sector[1]) & (la >= sector[2]) & (la <= sector[3])
    if not m.any():
        return None
    d = np.atleast_1d(C.km(lat, lon, la[m], lo[m]))
    j = int(np.argmin(d))
    return float(d[j]), float(la[m][j] - lat)


def season_offsets(year, artifact_dir=C.ARTIFACT_DIR, loaders=None):
    if year in C.RESERVED or year in C.HELD_OUT:
        raise SystemExit(f"REFUSED: {year} is reserved or held out and is not read at track level")
    import qtrack_pairing_pilot as M
    load_ours, load_q = loaders or (M.load_ours, M.load_qtrack)
    path = os.path.join(artifact_dir, C.ARTIFACT.format(year=year))
    art = json.load(open(path, encoding="utf-8"))
    for key in ("this_record", "qtrack"):
        if C.sha256_of(art["inputs"][key]["path"]) != art["inputs"][key]["sha256"]:
            raise SystemExit(f"REFUSED: {art['inputs'][key]['path']} is not the file the {year} artifact names")
    ours = load_ours(art["inputs"]["this_record"]["path"])[0]
    q = {t["id"]: t for t in load_q(art["inputs"]["qtrack"]["path"])[0]}
    idx = C.index_by_time(ours)
    out = {"none_in_sector": 0, "distance_km": [], "lat_offset": [], "qtrack_lat": []}
    for r in art["sides"]["qtrack"]["tracks"]:
        if r["outcome"] != "atlantic_side" or not (C.START_LONS[0] <= r["first"]["lon"] < C.START_LONS[1]):
            continue
        t = q[r["id"]]; C.check_track(r, t, "qtrack", year)
        for k in range(r["first_atlantic_side"]["index"] + 1):
            n = nearest_in_sector(t["lat"][k], t["lon"][k], t["time"][k], idx)
            if n is None:
                out["none_in_sector"] += 1
                continue
            out["distance_km"].append(n[0]); out["lat_offset"].append(n[1]); out["qtrack_lat"].append(float(t["lat"][k]))
    return out


def summary(parts):
    d = np.array([x for p in parts for x in p["distance_km"]]); o = np.array([x for p in parts for x in p["lat_offset"]])
    ql = np.array([x for p in parts for x in p["qtrack_lat"]]); none = sum(p["none_in_sector"] for p in parts)
    far = d > C.RADIUS_KM
    return {"observations": int(d.size + none), "none_in_sector": int(none), "with_nearest": int(d.size),
            "distance_quartiles_km": [round(float(x), 1) for x in np.percentile(d, [25, 50, 75])],
            "share_within_radius": round(float(np.mean(~far)), 4),
            "beyond_radius": int(far.sum()),
            "beyond_lat_offset_quartiles": [round(float(x), 2) for x in np.percentile(o[far], [25, 50, 75])],
            "beyond_share_north": round(float(np.mean(o[far] > 0)), 4),
            "qtrack_lat_median": round(float(np.median(ql)), 2),
            "beyond_nearest_lat_median": round(float(np.median(ql[far] + o[far])), 2)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    pj = os.path.join(a.out_dir, "continuity_nearest_offsets.json")
    if os.path.exists(pj):
        raise SystemExit(f"REFUSED: {pj} exists and records are never overwritten")
    S = summary([season_offsets(y) for y in C.SEASONS])
    rec = {"generated_by": {"script": "scripts/continuity_nearest_offsets.py", "sha256": SCRIPT_SHA256_AT_LOAD,
                            "census_script_sha256": C.SCRIPT_SHA256_AT_LOAD, "sector": list(SECTOR), "radius_km": C.RADIUS_KM,
                            "seasons": list(C.SEASONS), "added_after": "the census tables were read (record section 7)"},
           "summary": S}
    with open(pj, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    print(json.dumps(S, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
