"""The pilot's assessment instrument on synthetic runs: identity passes only on the same
set of track arrays, and the crosswalk keeps every pairing, classes prefixes as candidates,
flags boundary starts as potentially censored, classifies origin by the retained
classifier, and reports signed start shifts and cohort transitions."""
import hashlib
import importlib.util
import json
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def _load():
    spec = importlib.util.spec_from_file_location("pilot_track_crosswalk_under_test", os.path.join(ROOT, "scripts", "pilot_track_crosswalk.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _regions():
    from matplotlib.path import Path
    return [("NAL", Path([(-60, 0), (-18, 0), (-18, 30), (-60, 30)])), ("AFR", Path([(-18, 0), (52, 0), (52, 30), (-18, 30)]))]


def _day(y, m, d):
    import datetime as dt
    return float((dt.date(y, m, d) - dt.date(1900, 1, 1)).days)


def _track(t0, lat, lon, n, step=-0.5):
    return {"time": t0 + 0.25 * np.arange(n), "lat": np.full(n, float(lat)), "lon": lon + step * np.arange(n)}


def _run(tmp_path, name, tracks, domain_lon, variant=None):
    from scipy.io import savemat
    d = tmp_path / name
    d.mkdir(parents=True)
    mat = {"n": float(len(tracks)), "case_id": name.ljust(32, "x")}
    for i, t in enumerate(tracks):
        mat[f"lat{i}"], mat[f"lon{i}"], mat[f"time{i}"] = t["lat"], t["lon"], t["time"]
    savemat(str(d / "tracker_port.mat"), mat)
    record = {"dataset_specific": {"dataset": "era5", "year": 1990, "tracks_sha256": hashlib.sha256((d / "tracker_port.mat").read_bytes()).hexdigest()},
              "protocol_settings": {"manifest_sha256": "m" * 64, "domain": {"lat": [-35.0, 35.0], "lon": [-140.0, domain_lon]},
                                    **({"domain_variant": variant} if variant else {})}}
    (d / "tracking_era5_1990.json").write_text(json.dumps(record))
    return str(d)


def test_identity_passes_only_on_the_same_set_of_arrays(tmp_path):
    P = _load()
    t0 = _day(1990, 8, 1)
    tracks = [_track(t0, 12.0, 20.0, 10), _track(t0 + 2, 14.0, 30.0, 8)]
    a = P.load_run(_run(tmp_path, "A", tracks, 40.0), year=1990)
    b = P.load_run(_run(tmp_path, "B", list(reversed(tracks)), 40.0), year=1990)        # order does not matter
    r = P.identity(a, b)
    assert r["passed"] and r["shared"] == 2 and r["only_in_a"] == 0 == r["only_in_b"]
    moved = [dict(tracks[0], lat=tracks[0]["lat"] + 1e-9), tracks[1]]
    c = P.load_run(_run(tmp_path, "C", moved, 40.0), year=1990)
    r = P.identity(a, c)
    assert not r["passed"] and r["only_in_a"] == 1 and r["only_in_b"] == 1 and r["shared"] == 1
    fewer = P.load_run(_run(tmp_path, "D", tracks[:1], 40.0), year=1990)
    assert not P.identity(a, fewer)["passed"]
    # version 1's retained duplication: the same track twice on both sides is identical, once on one side is not
    twice = P.load_run(_run(tmp_path, "E", tracks + tracks[:1], 40.0), year=1990)
    twice_too = P.load_run(_run(tmp_path, "F", tracks[:1] + tracks, 40.0), year=1990)
    r = P.identity(twice, twice_too)
    assert r["passed"] and r["tracks_a"] == 3 == r["tracks_b"] and r["shared"] == 3
    r = P.identity(twice, a)
    assert not r["passed"] and r["only_in_a"] == 1 and r["only_in_b"] == 0 and r["shared"] == 2


def test_a_run_whose_tracks_are_not_the_records_is_refused(tmp_path):
    P = _load()
    d = _run(tmp_path, "A", [_track(_day(1990, 8, 1), 12.0, 20.0, 10)], 40.0)
    with open(os.path.join(d, "tracker_port.mat"), "ab") as fh:
        fh.write(b"\0")
    with pytest.raises(SystemExit, match="do not have the digest the record names"):
        P.load_run(d, year=1990)
    with pytest.raises(SystemExit, match="is for 1990, not 1991"):
        P.load_run(_run(tmp_path, "B", [_track(_day(1990, 8, 1), 12.0, 20.0, 10)], 40.0), year=1991)


def test_the_crosswalk_keeps_every_pairing_and_classes_prefixes_as_candidates(tmp_path, monkeypatch):
    P = _load()
    monkeypatch.setattr(P.S, "load_regions", lambda d: (_regions(), {}))
    regions = _regions()
    t0 = _day(1990, 6, 1)                                                   # a June 1 start
    west = _track(t0, 12.0, 35.0, 12)                                        # starts at 35 E in the control
    prefixed = {"time": np.concatenate([t0 - 0.25 * np.arange(4, 0, -1), west["time"]]),
                "lat": np.concatenate([np.full(4, 12.0), west["lat"]]),
                "lon": np.concatenate([35.0 + 0.5 * np.arange(4, 0, -1), west["lon"]])}   # four earlier steps, from 37 E, into May
    untouched = _track(t0 + 10, 10.0, -5.0, 10)
    crosser = _track(t0 + 15, 10.0, -10.0, 20)                                # reaches 19.5 W, the Atlantic side by the synthetic polygons
    endpoint = _track(t0 + 20, 8.0, 0.0, 10)
    endpoint_longer = _track(t0 + 20, 8.0, 0.0, 12)
    eastern = _track(t0 + 30, 11.0, 59.5, 20)                                # a new track at the 60 E edge, reaching 50 E
    dup_a, dup_b = _track(t0 + 40, 9.0, 10.0, 8), _track(t0 + 40, 9.2, 10.1, 8)   # version 1's duplication: two near copies
    a = P.load_run(_run(tmp_path, "B", [west, untouched, endpoint, dup_a, dup_b, crosser], 40.0), year=1990)
    b_dir = _run(tmp_path, "C", [prefixed, untouched, endpoint_longer, eastern, dup_a, dup_b, crosser], 60.0, variant={"note": "pilot"})
    b = P.load_run(b_dir, year=1990)
    r = P.crosswalk(a, b, 1990, regions)
    by = {(p["a"], p["b"]): p for p in r["pairs"]}
    assert by[(0, 0)]["class"] == "prefix changed" and by[(0, 0)]["candidate_prefix_steps"] == 4
    assert by[(0, 0)]["start_shift_days"] == -1.0 and by[(0, 0)]["candidate_prefix_degrees"] == 2.0   # eastward extent
    assert by[(0, 0)]["season_a"] and not by[(0, 0)]["season_b"]            # June 1 moved into May
    assert by[(1, 1)]["class"] == "identical" and by[(2, 2)]["class"] == "endpoint changed"
    assert r["unpaired_b"] == [3] and r["unpaired_a"] == []                  # the eastern track has no control partner
    assert r["multiply_paired_a"] == [3, 4] and r["multiply_paired_b"] == [4, 5]   # the duplicates pair across
    assert r["season_transitions_of_pairs"]["in season to out of season"] == 1 and r["season_transitions_of_pairs"]["out of season to in season"] == 0
    assert r["pair_classes"]["identical"] >= 3 and r["pair_classes"]["otherwise changed"] == 0   # every class present, empty ones as zero
    assert set(r["start_bands"]["b"]) == {f"{lo:+d}..{lo + 10:+d}" for lo in P.S.BAND_EDGES[:-1]} | {"outside"}
    assert r["boundary_counts"]["b"]["start_within_boundary_of_east_edge"] == {"count": 1, "share": round(1 / 7, 4)} and r["boundary_counts"]["b"]["tracks"] == 7
    assert set(r["first_detection_regions"]["a"]) == {c for c, _ in P.S.REGION_FILES} | {"OTH"}
    # the common-domain block excludes the wholly eastern track and reports crossing with both denominators
    cd = r["common_domain"]
    assert cd["common_edge_lon"] == 40.0 and cd["tracks_with_an_observation_west_of_the_edge"] == {"a": 6, "b": 6}
    assert cd["pair_classes"]["prefix changed"] == 1 and cd["start_bands_of_starts_west_of_the_edge"]["b"]["+30..+40"] == 1
    assert cd["start_bands_of_starts_west_of_the_edge"]["b"]["+50..+60"] == 0
    cc = cd["b"]["coast_crossing"]["stored_tracks"]
    assert cc["atlantic_side"] == 1 and cc["denominators"]["cohort"] == cc["cohort"] >= 1 and "fraction_atlantic_side_over_complete_follow_up" in cc
    assert cd["b"]["lifetime_observations_percentiles"] is not None and cd["b"]["coast_crossing"]["instrument"].endswith("coast_crossing_measurement.py")
    ab = r["attributes_b"][3]
    assert ab["potentially_censored_start"] and ab["start_within_boundary_of_east_edge"] and not r["attributes_a"][0]["potentially_censored_start"]
    assert r["boundary_counts"]["b"]["start_within_boundary_of_east_edge"]["count"] == 1 and r["boundary_counts"]["b"]["domain_lon"] == [-140.0, 60.0]
    assert ab["first_detection_region"] == "OTH" and ab["first_entry_into_africa_index"] is not None   # by the classifier, 59.5 E is outside the Africa polygon, 51.5 E inside
    assert r["starts_outside_africa_with_later_entry"]["b"] == 1
    assert r["attributes_a"][0]["first_detection_region"] == "AFR" and r["attributes_a"][0]["first_entry_into_africa_index"] == 0
    assert ab["first_entry_into_africa_lat"] == 11.0 and ab["first_entry_into_africa_time"] > ab["start_time"]
    assert r["attributes_b"][0]["start_month"] == 5 and r["attributes_a"][0]["start_month"] == 6
    assert r["cohort"]["a"]["africa_origin_season"] >= 1 and "not a count of physical waves" in r["cohort"]["a"]["note"]
    assert b["record_file"] == "tracking_era5_1990.json" and b["record_sha256"] == hashlib.sha256(open(os.path.join(b_dir, "tracking_era5_1990.json"), "rb").read()).hexdigest()
