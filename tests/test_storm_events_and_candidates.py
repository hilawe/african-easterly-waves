"""The pinned storm-event table and the candidate sets: three events kept separate, the
North Atlantic basin code read as text, provisional rows excluded and counted, and every
candidate wave retained with its link to one track product version."""
import hashlib
import importlib.util
import json
import os
import textwrap

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def _load(name):
    spec = importlib.util.spec_from_file_location(name + "_under_test", os.path.join(ROOT, "scripts", name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


HEADER = "SID,SEASON,NAME,ISO_TIME,NATURE,LAT,LON,USA_STATUS,USA_AGENCY,TRACK_TYPE,BASIN\n"
UNITS = ",,,,,degrees_north,degrees_east,,,,\n"


def _csv(tmp_path, rows):
    p = tmp_path / "ibtracs.csv"
    p.write_text(HEADER + UNITS + "".join(rows))
    return str(p)


def test_three_events_are_kept_separate_and_the_classes_follow_the_first_status(tmp_path):
    B = _load("build_storm_events")
    rows = [
        # an ordinary storm: a low, then a depression, then a storm
        "A1,1990,ALPHA,1990-08-01 00:00:00,DS,10.0,-20.0,LO,hurdat_atl,main,NA\n",
        "A1,1990,ALPHA,1990-08-01 06:00:00,TS,10.5,-21.0,TD,hurdat_atl,main,NA\n",
        "A1,1990,ALPHA,1990-08-02 00:00:00,TS,11.0,-23.0,TS,hurdat_atl,main,NA\n",
        # begins at depression strength and never becomes a storm
        "B2,1990,BETA,1990-08-10 00:00:00,TS,12.0,-30.0,TD,hurdat_atl,main,NA\n",
        "B2,1990,BETA,1990-08-10 06:00:00,TS,12.0,-31.0,TD,hurdat_atl,main,NA\n",
        # begins as a tropical storm
        "C3,1990,GAMMA,1990-09-01 00:00:00,TS,20.0,-50.0,TS,hurdat_atl,main,NA\n",
        # subtropical at first point, later tropical
        "D4,1990,DELTA,1990-09-05 00:00:00,SS,30.0,-60.0,SD,hurdat_atl,main,NA\n",
        "D4,1990,DELTA,1990-09-06 00:00:00,TS,31.0,-61.0,TS,hurdat_atl,main,NA\n",
        # a provisional row of the same season, and a storm outside the seasons
        "E5,1990,EPS,1990-09-20 00:00:00,TS,15.0,-40.0,TS,hurdat_atl,PROVISIONAL,NA\n",
        "F6,1978,ZETA,1978-09-20 00:00:00,TS,15.0,-40.0,TS,hurdat_atl,main,NA\n",
    ]
    path = _csv(tmp_path, rows)
    table = B.build(path, (1979, 2025))
    assert table["storms"] == 4 and table["provisional_rows_excluded"] == 1
    assert table["ibtracs"]["sha256"] == hashlib.sha256(open(path, "rb").read()).hexdigest()
    a = table["table"]["A1"]
    assert a["basin"] == "NA"                                            # read as text, not as a missing value
    assert a["first_archived"]["time"] == "1990-08-01T00:00:00Z" and a["first_archived"]["usa_status"] == "LO"
    assert a["first_depression_or_stronger"]["time"] == "1990-08-01T06:00:00Z"
    assert a["first_tropical_storm"]["time"] == "1990-08-02T00:00:00Z"
    assert a["class"] == "archive begins below depression strength, then reaches it"
    b = table["table"]["B2"]
    assert b["first_tropical_storm"] is None and b["class"] == "archive begins at depression strength"
    assert table["table"]["C3"]["class"] == "archive begins at tropical-storm strength or stronger"
    assert table["table"]["D4"]["class"] == "subtropical at first archived point"
    assert table["classes"] == {"archive begins below depression strength, then reaches it": 1, "archive begins at depression strength": 1,
                                "archive begins at tropical-storm strength or stronger": 1, "subtropical at first archived point": 1}
    assert table["assigns"] == "no wave, no flag, no probability"


def test_the_committed_table_is_pinned_to_the_ibtracs_file_on_disk():
    path = os.path.join(ROOT, "data", "aewc_v2_pilot", "ibtracs.NA.list.v04r01.csv")
    committed = os.path.join(ROOT, "docs", "aewc_v2", "artifacts", "storm_events_na_v04r01_1979_2025.json")
    if not (os.path.exists(path) and os.path.exists(committed)):
        pytest.skip("the IBTrACS file or the committed table is not in this tree")
    t = json.load(open(committed))
    assert t["ibtracs"]["sha256"] == hashlib.sha256(open(path, "rb").read()).hexdigest()
    assert t["seasons"] == [1979, 2025] and t["storms"] == len(t["table"]) and t["provisional_rows_excluded"] == 0
    assert all(e["first_archived"] is not None for e in t["table"].values())


def _product(tmp_path, tracks_by_year):
    """A minimal track product: one tracks file per year and a manifest listing its digest."""
    from scipy.io import savemat
    root = tmp_path / "product"
    (root / "tracks").mkdir(parents=True)
    files = {}
    for year, tracks in tracks_by_year.items():
        mat = {"n": float(len(tracks)), "case_id": "c" * 32}
        for i, t in enumerate(tracks):
            mat[f"lat{i}"], mat[f"lon{i}"], mat[f"time{i}"] = t["lat"], t["lon"], t["time"]
        rel = f"tracks/era5_{year}_tracks.mat"
        savemat(str(root / rel), mat)
        files[rel] = {"sha256": hashlib.sha256((root / rel).read_bytes()).hexdigest()}
    (root / "MANIFEST.json").write_text(json.dumps({"product": "test-product", "version": "0.0.1", "files": files}))
    return str(root)


def _events(tmp_path, storms):
    tmp_path.mkdir(parents=True, exist_ok=True)
    p = tmp_path / "storm_events.json"
    p.write_text(json.dumps({"table": storms}))
    return str(p)


def _track(t0, lat, lon, n=8, step_lon=-0.5):
    return {"time": t0 + 0.25 * np.arange(n), "lat": np.full(n, lat), "lon": lon + step_lon * np.arange(n)}


def test_candidate_sets_keep_every_candidate_and_classify_the_set_not_the_physics(tmp_path):
    C = _load("storm_case_candidates")
    t0 = C.days_since_1900("1990-08-01T00:00:00Z")
    tracks = [_track(t0 - 1.0, 12.0, -20.0),                           # passes 12 N 24 W at t0: one candidate
              _track(t0 - 1.0, 12.5, -20.5),                           # a near copy, version 1's duplication: a competitor
              _track(t0 - 1.0, 30.0, -60.0)]                           # far away
    product = _product(tmp_path, {1990: tracks})
    storms = {"S1": {"name": "ONE", "season": 1990, "class": "x",
                     "first_archived": {"time": "1990-08-01T00:00:00Z", "lat": 12.0, "lon": -24.0},
                     "first_depression_or_stronger": {"time": "1990-08-01T00:00:00Z", "lat": 12.0, "lon": -24.0},
                     "first_tropical_storm": None},
              "S2": {"name": "TWO", "season": 1990, "class": "x",
                     "first_archived": {"time": "1990-09-15T00:00:00Z", "lat": 12.0, "lon": -24.0},
                     "first_depression_or_stronger": None, "first_tropical_storm": None},
              "S3": {"name": "OTHER YEAR", "season": 1991, "class": "x", "first_archived": {"time": "1991-08-01T00:00:00Z", "lat": 0, "lon": 0},
                     "first_depression_or_stronger": None, "first_tropical_storm": None}}
    out = C.build(product, _events(tmp_path, storms), {1990})
    assert sorted(out["storms"]) == ["S1", "S2"] and out["summary"] == {"ambiguous": 1, "unmatched": 1}
    s1 = out["storms"]["S1"]
    for ev in ("first_archived", "first_depression_or_stronger"):
        for setting in ("prototype", "wider"):
            cands = s1["candidates"][ev][setting]
            assert sorted(c["track_index"] for c in cands) == [0, 1] and all(c["competing"] == 1 for c in cands)
            assert cands[0]["distance_km"] < cands[1]["distance_km"]                                   # nearest first
            assert all(abs(c["lag_hours"]) <= C.SETTINGS[setting]["window_hours"] for c in cands)
    assert "first_tropical_storm" not in s1["candidates"]              # the storm has no such event
    assert s1["track_product"] == {"product": "test-product", "version": "0.0.1", "track_file": "tracks/era5_1990_tracks.mat",
                                   "track_file_sha256": json.load(open(os.path.join(product, "MANIFEST.json")))["files"]["tracks/era5_1990_tracks.mat"]["sha256"]}
    assert set(out["link_schema"]) == {"product", "version", "track_file", "track_file_sha256", "track_index", "observation_time_days"}
    # one candidate under both settings at every event is straightforward
    out2 = C.build(_product(tmp_path / "p2", {1990: tracks[:1] + tracks[2:]}), _events(tmp_path / "e2", {"S1": storms["S1"]}), {1990})
    assert out2["summary"] == {"straightforward": 1}


def test_a_product_file_whose_digest_is_not_the_manifests_is_refused(tmp_path):
    C = _load("storm_case_candidates")
    t0 = C.days_since_1900("1990-08-01T00:00:00Z")
    product = _product(tmp_path, {1990: [_track(t0, 12.0, -24.0)]})
    with open(os.path.join(product, "tracks", "era5_1990_tracks.mat"), "ab") as fh:
        fh.write(b"\0")
    storms = {"S1": {"name": "ONE", "season": 1990, "class": "x", "first_archived": {"time": "1990-08-01T00:00:00Z", "lat": 12.0, "lon": -24.0},
                     "first_depression_or_stronger": None, "first_tropical_storm": None}}
    with pytest.raises(SystemExit, match="does not have the digest the product manifest lists"):
        C.build(product, _events(tmp_path, storms), {1990})
