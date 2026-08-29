"""Multi-season east-of-40E check for the AEWC v2 tracker decision.

Question, asked 2026-08-26 after the one-season pilot: is AEWDAT's empty
delivery east of 40 E (2005) a quiet season or a structural property of its
published tracks, and does QTrack's small eastern delivery recur? The
diagnostic that localizes the gap: AEWDAT's files carry the DETECTED TROUGH
OBJECTS per timestep independently of the TRACKS built from them, so the
detection layer and the track layer can be measured separately.

Seasons: 1988, 1995, 2000, 2005, 2006, 2010, 2015, 2020, 2022 (nine, spanning
the archives' overlap; 2006 included as a season with a known Ethiopian
Highlands origin case in the literature). Both archives' files hold exactly
the June-October window (612 six-hourly steps), which is asserted per file.
All 18 input files are pinned by SHA-256. Every value this check publishes,
per-season table rows included, is
asserted against the EXPECTED_ROWS block, and the 2005 TRACK metrics are
additionally bound to the reviewed pilot values (the object-layer metrics are
new here and have no pilot counterpart).

AEWDAT centers use the pilot's definition: graph nodes deduplicated by
(time, object id), then the unweighted mean of polyline vertices, averaged
across a track's objects at one time. The one-object-per-track-time invariant
the pilot reported for 2005 is asserted for every season here, which is what
makes that averaging equivalent to the pilot's.

Run from the repo root: .venv/bin/python scripts/aewc_v2_east_check.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "data" / "aewc_v2_pilot"
RESULTS = PILOT / "results"

YEARS = [1988, 1995, 2000, 2005, 2006, 2010, 2015, 2020, 2022]
EAST = 40.0
N_STEPS = 612

INPUT_SHA256 = {
    "wts1988.json": "727564ac1621ca574bcf09477db1e710ca25f08661c73c666f31222c4ee48429",
    "wts1995.json": "011b2c6a84d72d6fc9883b0872ccd985cce86dd558b08d9674fabcff54157e4c",
    "wts2000.json": "b11dc41285ba2ea37e6299bde62e2a2429a608da53056eddc536ea5bb7340c05",
    "wts2005.json": "8f5eb49ee1b076d740c4941db9ce5987ffda63a6e32ed29ba8e1a35fa253e272",
    "wts2006.json": "402fe4774cda792c3afe0af7cde42105ade99350278deb1b602634aaffe5e498",
    "wts2010.json": "2078b0b420e5eb18386f2578455b9ea89f78d370f48ccfa6d59909b652402753",
    "wts2015.json": "953cb7ef3d1bacb7a0ac82066cface58126a79366e1374cf6203293ef63352a8",
    "wts2020.json": "98e6c108f3e6175681589470c4ef2665974e422d03ee075a9b9109a341a554b5",
    "wts2022.json": "a494cbd360774e42b70647f73e00aaf8990e4c877f179286988732f40e97b163",
    "AEW_tracks_post_processed_year_1988.nc":
        "c829fad67aca6d2c2843f42967a304518e556837cd991d488e5b5fc2132b48e9",
    "AEW_tracks_post_processed_year_1995.nc":
        "7f20db51164297690a8de86ee64cd34b26b5d20e3523ab0840a1c6964ef27c5a",
    "AEW_tracks_post_processed_year_2000.nc":
        "0e31b3df32eba598cebee699e65ee309c3ac29cd18de575a8303e34a6316faf7",
    "AEW_tracks_post_processed_year_2005.nc":
        "11a51cc4a989e45a2c287aac483183276a0718e040fa2f6f7cebe8570668f9e7",
    "AEW_tracks_post_processed_year_2006.nc":
        "920e6f9475076ef858174b70b3ff5ca814696c99aa4a4edd4023c56756952ba0",
    "AEW_tracks_post_processed_year_2010.nc":
        "31917d324714c689256c683e56a7be5259ffeb756b99e860cd0966f7e1edd29a",
    "AEW_tracks_post_processed_year_2015.nc":
        "f9142c9f6f25bd55b46ecdf2599e58490cdd2073534c5f929b6ab92ab52e0f21",
    "AEW_tracks_post_processed_year_2020.nc":
        "d17ad8c9787f275b490bc61c27ccce6c990375b3aedc019210951a24ea7d5a0a",
    "AEW_tracks_post_processed_year_2022.nc":
        "796f3f6fd10950f2c12abee71b620c25a532cde96c7260d8b62214fe0fb1a43a",
}

# Every published table value, measured 2026-08-26 and pinned. Columns:
# (aewdat obj east any-vertex, aewdat obj east centroid, aewdat longest
#  consecutive eastern-object run in steps, aewdat max track centroid lon,
#  qtrack tracks any-east, qtrack tracks first-east, qtrack max lon)
EXPECTED_ROWS = {
    1988: (16, 13, 3, 33.92, 2, 2, 42.43),
    1995: (20, 15, 2, 35.37, 2, 2, 41.50),
    2000: (18, 15, 4, 31.86, 1, 1, 45.91),
    2005: (21, 10, 1, 36.44, 4, 4, 41.90),
    2006: (19, 12, 2, 35.30, 1, 1, 44.97),
    2010: (16, 12, 2, 34.44, 0, 0, 39.58),
    2015: (22, 17, 3, 32.32, 1, 1, 44.44),
    2020: (17, 9, 2, 35.00, 1, 1, 41.26),
    2022: (27, 22, 2, 32.29, 1, 1, 42.64),
}

# reviewed pilot values for the 2005 TRACK metrics (ARCHIVE_PILOT_2005.md)
PILOT_2005 = {"aewdat_tracks_any": 0, "aewdat_track_max_lon": 36.440,
              "qtrack_any": 4, "qtrack_first": 4, "qtrack_max_lon": 41.903}

def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _assert_window(times, year, what, complete):
    """The June-October 6-hourly grid. QTrack files carry the complete grid
    (complete=True). AEWDAT's timestep list omits steps with no detected
    objects, so it is asserted to be a nonempty subset of the same grid
    (complete=False); in 2005 it held 601 of the 612 steps."""
    t = pd.DatetimeIndex(times).sort_values()
    grid = pd.date_range(f"{year}-06-01T00:00:00", f"{year}-10-31T18:00:00",
                         freq="6h")
    assert len(grid) == N_STEPS
    if complete:
        assert len(t) == N_STEPS and (t == grid).all(), \
            f"{what} {year}: time axis is not the full June-October 6 h grid"
    else:
        assert 0 < len(t) <= N_STEPS and t.isin(grid).all(), \
            f"{what} {year}: timesteps leave the June-October 6 h grid"


def aewdat_year(year):
    d = json.loads((PILOT / f"wts{year}.json").read_text())
    s = d["sets"][0]
    assert float(s["level"]) == 700.0, f"{year}: unexpected level"
    _assert_window([ts["validTime"] for ts in s["timesteps"]], year,
                   "aewdat", complete=False)

    # detection layer: every object at every timestep, tracked or not
    obj_east_centroid = obj_east_any_vertex = n_obj = 0
    max_vertex_lon = -180.0
    east_at_45 = False
    for ts in s["timesteps"]:
        for o in ts.get("objects", []):
            n_obj += 1
            lons = [p["lon"] for p in o["properties"]["linePts"]]
            mx = float(max(lons))
            max_vertex_lon = max(max_vertex_lon, mx)
            if mx == 45.0:            # unrounded boundary contact
                east_at_45 = True
            if float(np.mean(lons)) > EAST:
                obj_east_centroid += 1
            if mx > EAST:
                obj_east_any_vertex += 1

    # track layer: nodes deduplicated by (time, object id), the pilot's rule
    tr_any = tr_first = 0
    max_track_lon = -180.0
    track_points_at = {}
    for trk in s["tracks"]:
        seen = {}
        for e in trk["edges"]:
            for n in [e["parent"]] + list(e.get("children", [])):
                t = n["time"]
                oid = n["object"]["id"]
                if (t, oid) in seen:
                    continue
                lps = n["object"]["properties"]["linePts"]
                seen[(t, oid)] = (float(np.mean([p["lat"] for p in lps])),
                                  float(np.mean([p["lon"] for p in lps])))
        by_time = {}
        for (t, _oid), c in seen.items():
            by_time.setdefault(t, []).append(c)
        # the invariant that makes this equivalent to averaging raw nodes
        assert all(len(v) == 1 for v in by_time.values()), \
            f"{year}: a published track carries more than one object at one time"
        if not by_time:
            continue
        series = sorted((t, v[0]) for t, v in by_time.items())
        for t, (la, lo) in series:
            track_points_at.setdefault(t, []).append((la, lo))
        lons = [lo for _, (_, lo) in series]
        max_track_lon = max(max_track_lon, max(lons))
        if max(lons) > EAST:
            tr_any += 1
        if series[0][1][1] > EAST:
            tr_first += 1

    # persistence and disconnection of the eastern detections
    east_step_flags = []
    near_track = 0
    for ts in s["timesteps"]:
        has = False
        for o in ts.get("objects", []):
            lps = o["properties"]["linePts"]
            lons = [p["lon"] for p in lps]
            if max(lons) > EAST:
                has = True
                la = float(np.mean([p["lat"] for p in lps]))
                lo = float(np.mean(lons))
                for tla, tlo in track_points_at.get(ts["validTime"], []):
                    if abs(tla - la) < 3.0 and abs(tlo - lo) < 3.0:
                        near_track += 1
                        break
        east_step_flags.append(has)
    longest = run = 0
    for h in east_step_flags:
        run = run + 1 if h else 0
        longest = max(longest, run)

    return {"n_tracks": len(s["tracks"]), "n_objects": n_obj,
            "obj_east_centroid": obj_east_centroid,
            "obj_east_any_vertex": obj_east_any_vertex,
            "max_object_vertex_lon": round(max_vertex_lon, 2),
            "object_touches_45_exactly": bool(east_at_45),
            "east_obj_longest_run_steps": int(longest),
            "east_obj_near_any_track": int(near_track),
            "tracks_any_east": tr_any, "tracks_first_east": tr_first,
            "max_track_centroid_lon": round(max_track_lon, 2)}


def qtrack_year(year):
    ds = xr.open_dataset(PILOT / f"AEW_tracks_post_processed_year_{year}.nc")
    _assert_window(ds["time"].values, year, "qtrack", complete=True)
    lon = ds["AEW_lon"].values
    lat = ds["AEW_lat"].values
    any_e = first_e = 0
    mx = -180.0
    for s in range(lon.shape[0]):
        good = np.isfinite(lon[s]) & np.isfinite(lat[s])
        if not good.any():
            continue
        ls = lon[s][good]
        mx = max(mx, float(ls.max()))
        if (ls > EAST).any():
            any_e += 1
        if float(ls[0]) > EAST:
            first_e += 1
    n = int(lon.shape[0])
    ds.close()
    return {"n_tracks": n, "tracks_any_east": any_e,
            "tracks_first_east": first_e, "max_lon": round(mx, 2)}


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)

    for name, want in INPUT_SHA256.items():
        got = _sha256(PILOT / name)
        assert got == want, f"{name}: sha256 {got[:12]}... != pinned {want[:12]}..."

    rows = {}
    for y in YEARS:
        rows[y] = {"aewdat": aewdat_year(y), "qtrack": qtrack_year(y)}
        a, q = rows[y]["aewdat"], rows[y]["qtrack"]
        assert a["n_tracks"] > 0 and q["n_tracks"] > 0, f"{y}: empty archive year"
        # every published table value, against the pinned row
        exp = EXPECTED_ROWS[y]
        got = (a["obj_east_any_vertex"], a["obj_east_centroid"],
               a["east_obj_longest_run_steps"], a["max_track_centroid_lon"],
               q["tracks_any_east"], q["tracks_first_east"], q["max_lon"])
        assert got == exp, f"{y}: measured row {got} != pinned {exp}"
        # the statements made for every season
        assert a["tracks_any_east"] == 0 and a["tracks_first_east"] == 0, \
            f"{y}: aewdat published a track east of {EAST}"
        assert a["max_object_vertex_lon"] == 45.0 and a["object_touches_45_exactly"], \
            f"{y}: aewdat detection no longer touches 45.0 exactly"
        assert a["east_obj_near_any_track"] == 0, \
            f"{y}: an eastern object sits within 3 degrees of a track point"

    # binding of the 2005 TRACK metrics to the reviewed pilot
    a05, q05 = rows[2005]["aewdat"], rows[2005]["qtrack"]
    assert a05["tracks_any_east"] == PILOT_2005["aewdat_tracks_any"]
    assert abs(a05["max_track_centroid_lon"]
               - PILOT_2005["aewdat_track_max_lon"]) < 5e-2
    assert q05["tracks_any_east"] == PILOT_2005["qtrack_any"]
    assert q05["tracks_first_east"] == PILOT_2005["qtrack_first"]
    assert abs(q05["max_lon"] - PILOT_2005["qtrack_max_lon"]) < 5e-2

    # the aggregate statements the note makes
    aew_track_years_east = sum(rows[y]["aewdat"]["tracks_any_east"] > 0
                               for y in YEARS)
    aew_obj_years_east = sum(rows[y]["aewdat"]["obj_east_any_vertex"] > 0
                             for y in YEARS)
    aew_obj_years_east_ctr = sum(rows[y]["aewdat"]["obj_east_centroid"] > 0
                                 for y in YEARS)
    qt_years_east = sum(rows[y]["qtrack"]["tracks_any_east"] > 0 for y in YEARS)
    max_run = max(rows[y]["aewdat"]["east_obj_longest_run_steps"] for y in YEARS)
    assert aew_track_years_east == 0, "0/9 aggregate changed"
    assert aew_obj_years_east == 9 and aew_obj_years_east_ctr == 9, \
        "9/9 aggregate changed (vertex or centroid basis)"
    assert qt_years_east == 8, "8/9 aggregate changed"
    assert 1 <= max_run <= 4, \
        f"eastern-object run range changed: max {max_run} steps"
    for y in YEARS:
        qe = rows[y]["qtrack"]["tracks_any_east"]
        assert (qe == 0) if y == 2010 else (1 <= qe <= 4), \
            f"{y}: qtrack eastern count {qe} outside the reported range"

    summary = {
        "years": YEARS,
        "aewdat_years_with_east_tracks": int(aew_track_years_east),
        "aewdat_years_with_east_objects_vertex": int(aew_obj_years_east),
        "aewdat_years_with_east_objects_centroid": int(aew_obj_years_east_ctr),
        "qtrack_years_with_east_tracks": int(qt_years_east),
        "aewdat_max_east_obj_run_steps": int(max_run),
        "rows": rows,
    }
    out = RESULTS / "east_check_multiseason.json"
    out.write_text(json.dumps(summary, indent=2))

    print(f"{'year':>5} | {'AEWDAT trk E':>12} {'first E':>8} {'max trk':>8} "
          f"| {'obj E (vtx)':>11} {'obj E (ctr)':>11} {'run':>4} {'max vtx':>8} "
          f"| {'QT trk E':>8} {'first E':>8} {'max':>7}")
    for y in YEARS:
        a, q = rows[y]["aewdat"], rows[y]["qtrack"]
        print(f"{y:>5} | {a['tracks_any_east']:>12} {a['tracks_first_east']:>8} "
              f"{a['max_track_centroid_lon']:>8} | "
              f"{a['obj_east_any_vertex']:>11} {a['obj_east_centroid']:>11} "
              f"{a['east_obj_longest_run_steps']:>4} "
              f"{a['max_object_vertex_lon']:>8} | "
              f"{q['tracks_any_east']:>8} {q['tracks_first_east']:>8} "
              f"{q['max_lon']:>7}")
    print(f"\nAEWDAT: {aew_track_years_east}/9 seasons with any track east of "
          f"{EAST}, 9/9 with any detected object east of {EAST} (on both the "
          f"vertex and centroid definitions)")
    print(f"QTrack: {qt_years_east}/9 seasons with any track east of {EAST}")
    print(f"results written to {out}")
    print("ALL ASSERTIONS PASSED")


if __name__ == "__main__":
    main()
