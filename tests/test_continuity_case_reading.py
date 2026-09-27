"""The case reading: the window is 0 to 48 hours inclusive, candidates keep identifiers
and per-step distances, successors are within 0 to 12 hours and 500 km, outcomes outside
the cohort are read from positions and marked, the view is drawn from the case's own
fields, digests are verified, and nothing is overwritten."""
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


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _world(tmp_path):
    """A season with: track 0 (the subject, in Africa 12 N, 20 E to 14 E over 7 steps, ending in
    Africa); QTrack system 1 within 500 km at elapsed 0 to 48 hours; system 2 within 500 km only
    at elapsed 54 hours (outside the window); system 3 far away; track 1 a successor starting 12
    hours after track 0's end about 170 km away and reaching the Atlantic side (outside the cohort,
    since it starts in October); track 2 starting 18 hours after (not a successor); a tracker case
    with synthetic fields."""
    import test_qtrack_pairing_pilot as T
    import test_coast_entry_examples as T0
    import qtrack_pairing_pilot as M
    from scipy.io import savemat
    JUNE1 = T.JUNE1
    ev = tmp_path / "evidence" / "era5_1990"
    ev.mkdir(parents=True)
    t0 = M._day(1990, 9, 29)                                                          # a September start, so the successor 12 hours after its end starts in October, outside the cohort
    lons0 = [20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.5, 13.0, 12.5]          # ten steps, 54 hours
    tracks = {0: (t0, [12.0] * 10, lons0),
              1: (t0 + 9 * 0.25 + 0.5, [12.5] * 4, [11.0, 5.0, -5.0, -12.0]),        # successor, 12 hours after the end, Atlantic side later
              2: (t0 + 9 * 0.25 + 0.75, [12.0] * 3, [12.0, 11.0, 10.0]),             # 18 hours after: not a successor
              3: (JUNE1 - 5.0, [20.0] * 3, [25.0, 24.0, 23.0]),                        # a May start, outside the cohort
              4: (t0 - 0.75, [12.0] * 3, [21.0, 20.5, 20.2]),                            # a predecessor: ends 6 hours before track 0 starts, 20 km away, in the cohort
              5: (t0, [12.2] * 10, lons0)}                                              # a duplicate of track 0 (same times, 0.2 degree north): one group
    payload = {"n": float(len(tracks)), "case_id": np.array(["c" * 32]), "producer_json": np.array(["{}"])}
    for i, (ts, lats, lons) in tracks.items():
        payload[f"time{i}"] = ts + 0.25 * np.arange(len(lons))
        payload[f"lat{i}"], payload[f"lon{i}"] = np.array(lats, float), np.array(lons, float)
    savemat(str(ev / "tracker_port.mat"), payload)
    (ev / "tracking_era5_1990.json").write_text(json.dumps({"dataset_specific": {"dataset": "era5", "year": 1990, "tracks_sha256": T._sha(ev / "tracker_port.mat")}}))
    qdir = tmp_path / "with_epac"
    qdir.mkdir()
    start = int(round((t0 - JUNE1) / 0.25))
    # system 4: within 500 km at 12 to 24 hours only, then 9 degrees away (an excursion inside the window);
    # system 5: starts 6 hours after track 0's end, 150 km from its last position, in October (outside the cohort)
    systems = [(start, 12.0, 20.0, 10), (start, 12.0, 30.0, 10), (start, 12.0, 35.0, 10), (start, 12.0, 20.0, 10), (start + 10, 12.0, 12.0, 4)]
    T._qtrack_file(str(qdir / "ERA5_AEW_tracks_with_basins_1990.nc"), systems)
    import netCDF4 as nc
    d = nc.Dataset(str(qdir / "ERA5_AEW_tracks_with_basins_1990.nc"), "a")
    d.variables["AEW_lon"][0, start:start + 10] = [20.5, 19.5, 18.5, 17.5, 16.5, 15.5, 14.5, 14.0, 13.5, 13.0]   # within 500 km throughout
    d.variables["AEW_lon"][1, start:start + 10] = [30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 14.0]    # within 500 km only at 54 hours
    d.variables["AEW_lon"][3, start:start + 10] = [30.0, 30.0, 18.5, 17.5, 16.5, 25.0, 25.0, 25.0, 25.0, 25.0]    # within 500 km at 12, 18, 24 hours
    d.variables["AEW_lon"][4, start + 10:start + 14] = [11.5, 5.0, -5.0, -12.0]
    d.close()
    T0._regions(str(tmp_path / "regions"))
    # the tracker case: a small one-degree grid covering the tracks, the whole year, with a synthetic anomaly
    cases = tmp_path / "cases" / "era5_1990"
    cases.mkdir(parents=True)
    lat = np.arange(-5.0, 31.0, 1.0)
    lon = np.arange(-40.0, 21.0, 1.0)                                                # the grid ends at 20 E: the requested view east of 12.5 E is clipped
    times = M._day(1990, 1, 1) + 0.25 * np.arange(1460)
    LON, LAT = np.meshgrid(lon, lat)
    anom = np.zeros((times.size, lat.size, lon.size))
    k_end = int(round((tracks[0][0] + 9 * 0.25 - times[0]) / 0.25))
    for k in range(k_end, k_end + 9):
        anom[k] = np.exp(-(((LON - (12.5 - 1.5 * (k - k_end))) ** 2) + (LAT - 12.0) ** 2) / 8.0)   # a blob moving west after the end
    savemat(str(cases / "tracker_case.mat"), {"latgrid": LAT, "longrid": LON, "time": times, "currv_anom": anom, "u": -3.0 + 0 * anom, "v": 0 * anom,
                                              "rean": np.array(["ERA5"]), "level": np.array([[700.0]])})
    return str(tmp_path / "evidence"), str(tmp_path / "cases"), str(qdir), str(tmp_path / "regions")


def _season_artifact(tmp_path, evidence, qdir, regions):
    C = _load("coast_crossing_measurement")
    art_dir = tmp_path / "artifacts"
    art_dir.mkdir()
    out = art_dir / "coast_crossing_1990_2026-09-27.json"
    assert C.main(["--year", "1990", "--campaign-evidence", evidence, "--qtrack-dir", qdir, "--regions-dir", regions, "--out", str(out)]) == 0
    return art_dir


def test_window_candidates_successors_outside_cohort_reading_and_view(tmp_path):
    R = _load("continuity_case_reading")
    evidence, cases, qdir, regions = _world(tmp_path)
    art_dir = _season_artifact(tmp_path, evidence, qdir, regions)
    out, png = tmp_path / "reading.json", tmp_path / "view.png"
    assert R.main(["--year", "1990", "--track", "0", "--role", "case", "--campaign-evidence", evidence, "--tracker-cases", cases,
                   "--qtrack-dir", qdir, "--regions-dir", regions, "--measurement-artifacts", str(art_dir), "--out", str(out), "--figure", str(png)]) == 0
    r = json.load(open(out))
    assert r["track"]["outcome"] == "none" and r["track"]["n"] == 10 and r["decision_history"].startswith("unavailable")
    c = r["candidate_correspondences"]
    assert c["window_observations"] == 9 and c["per_step"][-1]["elapsed_hours"] == 48.0            # 0 to 48 hours inclusive
    ids = [s["system"] for s in c["candidate_systems"]]
    assert ids == [1, 4]                                                                           # system 2 is within 500 km only at 54 hours
    s1 = c["candidate_systems"][0]
    assert s1["first_elapsed_hour_within_500_km"] == 0.0 and s1["window_steps_within_500_km"] == 9 and len(s1["distances_km_by_elapsed_hour"]) == 9
    assert s1["beyond_window"]["shared_steps"] == 1 and s1["beyond_window"]["last_shared_elapsed_hours"] == 54.0
    s4 = c["candidate_systems"][1]
    assert s4["first_elapsed_hour_within_500_km"] == 12.0 and s4["window_steps_within_500_km"] == 3 and len(s4["distances_km_by_elapsed_hour"]) == 9   # distance kept at every shared window step
    assert s4["distances_km_by_elapsed_hour"]["0.0"] > 900 and s4["distances_km_by_elapsed_hour"]["30.0"] > 900 and s4["window_steps_beyond_500_km"] == 6   # the excursion is visible
    assert [step["n_candidates"] for step in c["per_step"]] == [1, 1, 2, 2, 2, 1, 1, 1, 1]        # ambiguity per step preserved
    assert r["successors"] and r["successors"][0]["id"] == 1 and r["successors"][0]["hours_apart"] == 12.0
    assert r["successors"][0]["read_from_positions_outside_cohort"] and r["successors"][0]["outcome"] == "atlantic_side"
    assert len(r["successors"][0]["positions"]) == 4                                               # positions retained for the overlay at panel times
    assert all(s["id"] != 2 for s in r["successors"])                                             # 18 hours after is outside the adjacent window
    assert [p["id"] for p in r["predecessors"]] == [4] and r["predecessors"][0]["hours_apart"] == 6.0 and not r["predecessors"][0]["read_from_positions_outside_cohort"]
    assert [q["system"] for q in r["qtrack_systems_starting_near_end"]] == [5] and r["qtrack_systems_starting_near_end"][0]["read_from_positions_outside_cohort"]
    assert r["qtrack_systems_starting_near_end"][0]["outcome"] == "atlantic_side"                 # read from its positions
    assert r["track"]["group_members"] == [0, 5] and [m["id"] for m in r["track"]["group_member_spans"]] == [5] and r["track"]["group_member_spans"][0]["n"] == 10
    inv = r["qtrack_inventory_during_the_track"]
    assert inv["systems_alive_during_the_track"] == 4 and sorted(s["system"] for s in inv["with_any_position_east_of_17_W_during_it"]) == [1, 2, 3, 4]
    v = r["field_view"]
    assert v["reanalysis"] == "ERA5" and len(v["times"]) == 3 and all(t["available"] for t in v["times"])
    assert v["window_clipped_by_grid"]["east"] and not v["window_clipped_by_grid"]["west"] and all(t["window_clipped"] for t in v["times"])
    assert v["window_requested"]["lon"][1] == 22.5 and v["window"]["lon"][1] == 20.0
    assert v["times"][1]["window_max_at"]["lon"] < v["times"][0]["window_max_at"]["lon"]         # the synthetic blob moves west
    assert png.exists() and png.stat().st_size > 1000
    with pytest.raises(SystemExit):                                                                # never overwritten
        R.main(["--year", "1990", "--track", "0", "--role", "case", "--campaign-evidence", evidence, "--tracker-cases", cases,
                "--qtrack-dir", qdir, "--regions-dir", regions, "--measurement-artifacts", str(art_dir), "--out", str(out), "--figure", str(tmp_path / "v2.png")])


def test_refusals(tmp_path):
    R = _load("continuity_case_reading")
    evidence, cases, qdir, regions = _world(tmp_path)
    art_dir = _season_artifact(tmp_path, evidence, qdir, regions)
    args = ["--year", "1990", "--role", "case", "--campaign-evidence", evidence, "--tracker-cases", cases, "--qtrack-dir", qdir,
            "--regions-dir", regions, "--measurement-artifacts", str(art_dir)]
    with pytest.raises(SystemExit, match="not in the 1990 cohort"):                                  # track 3 starts in May
        R.main(["--track", "3", "--out", str(tmp_path / "a.json"), "--figure", str(tmp_path / "a.png"), *args])
    (tmp_path / "exists.png").write_bytes(b"x")
    with pytest.raises(SystemExit, match="never overwritten"):                                       # an existing figure
        R.main(["--track", "0", "--out", str(tmp_path / "c.json"), "--figure", str(tmp_path / "exists.png"), *args])
    with open(os.path.join(regions, "africa.mat"), "ab") as fh:
        fh.write(b"\0")
    with pytest.raises(SystemExit, match="region polygons"):
        R.main(["--track", "0", "--out", str(tmp_path / "d.json"), "--figure", str(tmp_path / "d.png"), *args])
    with open(os.path.join(regions, "africa.mat"), "rb+") as fh:
        fh.seek(0, 2); fh.truncate(fh.tell() - 1)                                                    # restored
    with open(os.path.join(qdir, "ERA5_AEW_tracks_with_basins_1990.nc"), "ab") as fh:
        fh.write(b"\0")
    with pytest.raises(SystemExit, match="QTrack's file is not the one"):
        R.main(["--track", "0", "--out", str(tmp_path / "e.json"), "--figure", str(tmp_path / "e.png"), *args])
    with open(os.path.join(qdir, "ERA5_AEW_tracks_with_basins_1990.nc"), "rb+") as fh:
        fh.seek(0, 2); fh.truncate(fh.tell() - 1)
    with open(os.path.join(evidence, "era5_1990", "tracker_port.mat"), "ab") as fh:
        fh.write(b"\0")
    with pytest.raises(SystemExit, match="not the one the season artifact names"):
        R.main(["--track", "0", "--out", str(tmp_path / "b.json"), "--figure", str(tmp_path / "b.png"), *args])


def test_membership_is_decided_on_unrounded_distances():
    """A system 500.04 km away would round to 500.0 and pass a rounded test; it must not be a candidate."""
    R = _load("continuity_case_reading")
    import numpy as np
    t = np.array([0.0, 0.25, 0.5])
    subject = {"id": 0, "time": t, "lat": np.array([12.0] * 3), "lon": np.array([20.0] * 3)}
    lo, hi = 20.0, 30.0                                                      # find the longitude at 12 N that is 500.04 km east, by bisection
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if R.km(12.0, 20.0, 12.0, mid) < 500.04 else (lo, mid)
    far = {"id": 1, "time": t, "lat": np.array([12.0] * 3), "lon": np.array([hi] * 3)}
    assert 500.0 < R.km(12.0, 20.0, 12.0, hi) < 500.1
    out = R.candidates(subject, [far], {}, set(), None, 1e9)
    assert out["n_candidate_systems"] == 0 and all(s["n_candidates"] == 0 for s in out["per_step"])
    near = {"id": 2, "time": t, "lat": np.array([12.0] * 3), "lon": np.array([hi - 0.02] * 3)}
    out = R.candidates(subject, [far, near], {2: {"outcome": "none", "follow_up_incomplete": False}}, set(), None, 1e9)
    assert [s["system"] for s in out["candidate_systems"]] == [2] and out["candidate_systems"][0]["window_steps_within_500_km"] == 3


def test_conventions():
    R = _load("continuity_case_reading")
    assert R.WINDOW_HOURS == 48.0 and R.ADJACENT_HOURS == 12.0 and R.RADIUS_KM == 500.0
    assert R.hours(10.0, 8.0) == 48.0 and R.hours(10.25, 8.0) == 54.0
