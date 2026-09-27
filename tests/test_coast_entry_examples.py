"""The boundary probe: the cohort is the archive's rule plus the exploratory latitude
range, the first North Atlantic observation is found where it is, returns and record
edges are flagged, the segment labels are where they say, and nothing is a fraction."""
import importlib.util
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, HERE)


def _load():
    spec = importlib.util.spec_from_file_location("coast_entry_examples", os.path.join(ROOT, "scripts", "coast_entry_examples.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _regions(directory):
    """Africa is the box 0 to 30 E, 0 to 30 N; the North Atlantic the box 40 W to 0, 0 to 30 N,
    sharing the meridian 0, which the priority order (North Atlantic before Africa) gives to
    the North Atlantic; every other polygon is a small box far away."""
    import season_metrics as S
    from scipy.io import savemat
    os.makedirs(directory)
    for code, name in S.REGION_FILES:
        if code == "AFR":
            box = np.array([[0.0, 30.0, 30.0, 0.0], [0.0, 0.0, 30.0, 30.0]])
        elif code == "NAL":
            box = np.array([[-40.0, 0.0, 0.0, -40.0], [0.0, 0.0, 30.0, 30.0]])
        else:
            box = np.array([[100.0, 101.0, 101.0, 100.0], [-60.0, -60.0, -59.0, -59.0]])
        savemat(os.path.join(directory, f"{name}.mat"), {name: box})


def _world(tmp_path):
    import test_qtrack_pairing_pilot as T
    import qtrack_pairing_pilot as M
    JUNE1 = T.JUNE1
    ev = tmp_path / "evidence" / "era5_1990"
    ev.mkdir(parents=True)
    # this record, whole year: 0 crosses west at lat 12 (10 E to 20 W) and stays out; 1 stays in Africa;
    # 2 crosses and comes back (5 E, 1 W, 3 E); 3 starts in May (not in season); 4 starts at 28 N (outside the cohort);
    # 5 starts in September, crosses, and its last observation is the record's last step (right-censored follow-up
    # would apply had it not crossed); 6 crosses with a two-step gap before the entry
    from scipy.io import savemat
    dec31 = M._day(1990, 12, 31) + 0.75
    tracks = {
        0: (JUNE1 + 10.0, [12.0] * 7, [10.0, 5.0, 1.0, -3.0, -8.0, -14.0, -20.0]),
        1: (JUNE1 + 12.0, [12.0] * 4, [20.0, 18.0, 16.0, 14.0]),
        2: (JUNE1 + 14.0, [10.0] * 3, [5.0, -1.0, 3.0]),
        3: (JUNE1 - 10.0, [12.0] * 4, [10.0, 5.0, -1.0, -5.0]),
        4: (JUNE1 + 16.0, [28.0] * 4, [10.0, 5.0, -1.0, -5.0]),
        5: (M._day(1990, 9, 28), None, None),                                # filled below: September start, alive to the record's last step
        6: (JUNE1 + 20.0, [15.0] * 3, [4.0, 2.0, -9.0]),
        7: (JUNE1 + 22.0, [25.0] * 3, [12.0, 10.0, 8.0]),                    # exactly on the cohort's upper edge, inclusive
        8: (M._day(1990, 9, 20), [10.0] * 4, [20.0, 18.0, 16.0, 14.0]),     # filled below: ends at the record's last step WITHOUT an entry
    }
    payload = {"n": float(len(tracks)), "case_id": np.array(["c" * 32]), "producer_json": np.array(["{}"])}
    n5 = int(round((dec31 - tracks[5][0]) / 0.25)) + 1
    tracks[5] = (tracks[5][0], [8.0] * n5, list(np.linspace(3.0, -30.0, n5)))
    n8 = int(round((dec31 - tracks[8][0]) / 0.25)) + 1
    tracks[8] = (tracks[8][0], [10.0] * n8, list(np.linspace(20.0, 12.0, n8)))   # stays in Africa to the record's end
    for i, (t0, lats, lons) in tracks.items():
        times = t0 + 0.25 * np.arange(len(lons))
        if i == 6:
            times = np.array([t0, t0 + 0.25, t0 + 0.75])                    # a gap of two steps before the entry
        payload[f"time{i}"], payload[f"lat{i}"], payload[f"lon{i}"] = times, np.array(lats, float), np.array(lons, float)
    savemat(str(ev / "tracker_port.mat"), payload)
    (ev / "tracking_era5_1990.json").write_text(json.dumps({"dataset_specific": {"dataset": "era5", "year": 1990, "tracks_sha256": T._sha(ev / "tracker_port.mat")}}))
    qdir = tmp_path / "with_epac"
    qdir.mkdir()
    # QTrack: system 1 crosses at lat 5 (into the "gulf" label by the segment rule, since its lon is -2); system 2 stays in Africa;
    # system 3 starts in Africa at the file's first time step, so it is left-censored and excluded
    T._qtrack_file(str(qdir / "ERA5_AEW_tracks_with_basins_1990.nc"), [(40, 5.0, 3.0, 4), (60, 12.0, 20.0, 4), (0, 12.0, 15.0, 4)])
    import netCDF4 as nc
    d = nc.Dataset(str(qdir / "ERA5_AEW_tracks_with_basins_1990.nc"), "a")
    d.variables["AEW_lon"][0, 40:44] = [3.0, 1.0, -2.0, -5.0]
    d.close()
    _regions(str(tmp_path / "regions"))
    return str(tmp_path / "evidence"), str(qdir), str(tmp_path / "regions")


def test_entries_returns_edges_and_cohort_are_reported_and_no_fraction_is(tmp_path):
    C = _load()
    evidence, qdir, regions = _world(tmp_path)
    out = tmp_path / "coast.json"
    assert C.main(["--year", "1990", "--campaign-evidence", evidence, "--qtrack-dir", qdir, "--regions-dir", regions, "--out", str(out)]) == 0
    art = json.load(open(out))
    ours = art["seasons"]["this_record"]
    ids = {t["id"]: t for t in ours["tracks"]}
    assert sorted(ids) == [0, 1, 2, 5, 6, 7, 8]                             # 3 starts in May, 4 starts at 28 N, 7 sits on 25 N
    assert ids[8]["ends_at_record_end"] and ids[8]["north_atlantic_entry"] is None and ids[8]["last"]["region"] == "AFR"
    assert ids[0]["north_atlantic_entry"]["lon"] == -3.0 and ids[0]["north_atlantic_entry"]["previous_region"] == "AFR"
    assert ids[0]["north_atlantic_entry"]["later_africa_observations"] == 0 and ids[0]["last"]["region"] == "NAL"
    assert ids[1]["north_atlantic_entry"] is None and ids[1]["regions_visited"] == ["AFR"]
    assert ids[2]["north_atlantic_entry"]["later_africa_observations"] == 1 and ids[2]["last"]["region"] == "AFR"
    assert ids[5]["ends_at_record_end"] and ids[5]["north_atlantic_entry"]["entry_is_last_observation"] is False
    assert ids[6]["north_atlantic_entry"]["steps_from_previous"] == 2
    s = ours["summary"]
    assert s["cohort"] == 7 and s["with_north_atlantic_entry"] == 4 and s["entrants_returning_to_africa"] == 1
    assert s["cohort_ending_at_record_end"] == 2 and s["cohort_ending_at_record_end_without_entry"] == 1 and s["entrants_with_a_gap_before_entry"] == 1
    assert ours["excluded_left_censored"] == [] and s["cohort_starting_at_record_start"] == 0
    assert not any("fraction" in k or "probability" in k for k in s)       # nothing here is a rate
    q = art["seasons"]["qtrack"]
    assert [t["id"] for t in q["tracks"]] == [1, 2] and q["excluded_left_censored"] == [3]   # the record-edge start is out, and named
    assert q["tracks"][0]["north_atlantic_entry"]["segment"] == "gulf_of_guinea" and q["tracks"][1]["north_atlantic_entry"] is None
    assert art["inputs"]["this_record"]["sha256"] and art["inputs"]["region_polygons"]["sha256"]["africa"]
    with pytest.raises(SystemExit):                                          # never overwritten
        C.main(["--year", "1990", "--campaign-evidence", evidence, "--qtrack-dir", qdir, "--regions-dir", regions, "--out", str(out)])


def test_segment_labels_and_cohort_edges_are_where_they_say():
    C = _load()
    assert C.coast_segment(-17.5, 14.7) == "west_coast_4_to_22N"            # off Dakar
    assert C.coast_segment(-7.5, 4.0) == "west_coast_4_to_22N"               # Cape Palmas, both edges inclusive
    assert C.coast_segment(-7.4, 4.0) == "gulf_of_guinea"                    # just east of it
    assert C.coast_segment(0.0, 3.0) == "gulf_of_guinea"
    assert C.coast_segment(-13.0, 24.0) == "north_of_22N"
    assert C.coast_segment(-17.0, 22.0) == "west_coast_4_to_22N" and C.coast_segment(-17.0, 22.01) == "north_of_22N"
    assert C.coast_segment(-5.0, 10.0) == "other"
    assert C.COHORT_LAT == (0.0, 25.0)
