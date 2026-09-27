"""The pilot script on a synthetic season: strict filename patterns, grid and window
refusals, the record binding of this record's tracks, both populations with their
exclusions, the two labeled subsets, duplicate involvement, both tolerances, and
exclusive publication. The pairing module has its own tests."""
import hashlib
import importlib.util
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src"))


def _load():
    path = os.environ.get("PILOT_SCRIPT", os.path.join(ROOT, "scripts", "qtrack_pairing_pilot.py"))
    spec = importlib.util.spec_from_file_location("qtrack_pairing_pilot", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


D = 32872.0
JUNE1 = D + 151.0                      # June 1 1990 in days since 1900 (1990 is not a leap year)


def _qtrack_file(path, systems, hours_units=True, steps=612, first_day=JUNE1):
    """A QTrack-shaped file: time in hours since 1900 (or seconds since 1970), per-system
    raw positions, a smoothed pair that differs, a system coordinate."""
    import netCDF4 as nc
    d = nc.Dataset(path, "w")
    d.createDimension("time", steps)
    d.createDimension("system", len(systems))
    t = d.createVariable("time", "f8", ("time",))
    days = first_day + 0.25 * np.arange(steps)
    if hours_units:
        t.units = "hours since 1900-01-01"
        t[:] = days * 24.0
    else:
        t.units = "seconds since 1970-01-01"
        t[:] = (days - 25567.0) * 86400.0
    s = d.createVariable("system", "f8", ("system",))
    s[:] = np.arange(1, len(systems) + 1, dtype=float)
    lon = d.createVariable("AEW_lon", "f8", ("system", "time"), fill_value=np.nan)
    lat = d.createVariable("AEW_lat", "f8", ("system", "time"), fill_value=np.nan)
    lons = d.createVariable("AEW_lon_smooth", "f8", ("system", "time"), fill_value=np.nan)
    lats = d.createVariable("AEW_lat_smooth", "f8", ("system", "time"), fill_value=np.nan)
    basin = d.createVariable("basin_des", "f8", ("system", "time"), fill_value=np.nan)
    name = d.createVariable("TC_name", str, ("system",))
    gen = d.createVariable("TC_gen_time", "f8", ("system",), fill_value=np.nan)
    for k, (start, la, lo, n) in enumerate(systems):
        lon[k, start:start + n] = lo + np.zeros(n)
        lat[k, start:start + n] = la
        lons[k, start:start + n] = lo + 1.0
        lats[k, start:start + n] = la
        basin[k, start:start + n] = 2.0
        name[k] = "N/A"
    d.close()


def _ours_file(path, tracks):
    from scipy.io import savemat
    payload = {"n": float(len(tracks)), "case_id": np.array(["c" * 32]), "producer_json": np.array(["{}"])}
    for i, (t0, la, lo, n) in enumerate(tracks):
        payload[f"time{i}"] = t0 + 0.25 * np.arange(n)
        payload[f"lat{i}"] = np.full(n, float(la))
        payload[f"lon{i}"] = lo + np.zeros(n)
    savemat(path, payload)


def _regions(directory):
    import season_metrics as S
    from scipy.io import savemat
    os.makedirs(directory)
    for code, name in S.REGION_FILES:
        # every polygon a small box far from the tracks except AFR, which covers the test's African positions
        box = np.array([[-10.0, 30.0, 30.0, -10.0], [0.0, 0.0, 30.0, 30.0]]) if code == "AFR" else np.array([[100.0, 101.0, 101.0, 100.0], [-60.0, -60.0, -59.0, -59.0]])
        savemat(os.path.join(directory, f"{name}.mat"), {name: box})


def _world(tmp_path, *, qtrack_hours=True, qtrack_steps=612, our_grid_ok=True, east_system=False, shared_tail=False, unused_step_off_grid=False):
    qdir, adir = tmp_path / "with_epac", tmp_path / "atlantic"
    qdir.mkdir(parents=True); adir.mkdir(parents=True)
    # QTrack: system 1 pairs with our track 0 (same times, 0.5 degree east), system 2 is far, system 3 has two coverage points, system 4 is outside the region
    systems = [(40, 12.0, 10.5, 8), (40, 12.0, 30.0, 8), (100, 12.0, 10.0, 2), (40, 50.0, 10.0, 8)]
    if east_system:
        systems.append((200, 12.0, 45.0, 8))                   # starts east of 40 E, outside the region, and enters it later
    if shared_tail:
        systems.append((40, 12.0, 30.0, 8))                    # identical positions to system 2 at the same steps: a shared tail
    _qtrack_file(str(qdir / "ERA5_AEW_tracks_with_basins_1990.nc"), systems, hours_units=qtrack_hours, steps=qtrack_steps)
    if east_system:                                            # the eastern system drifts west into the common region for its last five steps
        import netCDF4 as nc
        d = nc.Dataset(str(qdir / "ERA5_AEW_tracks_with_basins_1990.nc"), "a")
        d.variables["AEW_lon"][4, 200:208] = [45.0, 43.0, 41.0, 39.0, 37.0, 35.0, 33.0, 31.0]
        d.close()
    if unused_step_off_grid:                                   # a time step no system uses, moved one hour off the grid
        import netCDF4 as nc
        d = nc.Dataset(str(qdir / "ERA5_AEW_tracks_with_basins_1990.nc"), "a")
        t = d.variables["time"]; t[611] = t[611] + (1.0 if qtrack_hours else 3600.0)
        d.close()
    (qdir / "ERA5_AEW_tracks_with_basins_1990_extra.nc").write_bytes(b"companion, never an input")
    import netCDF4 as nc
    d = nc.Dataset(str(adir / "ERA5_AEW_tracks_atlantic_1990.nc"), "w")
    d.createDimension("system", 2)
    s = d.createVariable("system", "f8", ("system",)); s[:] = [1.0, 2.0]
    d.close()
    (adir / "ERA5_AEW_tracks_atlantic_1990_owned.nc").write_bytes(b"companion")
    ev = tmp_path / "evidence" / "era5_1990"
    ev.mkdir(parents=True)
    # ours: track 0 in coverage pairing with system 1, track 1 in January (no coverage), track 2 in coverage far from all, track 3 a duplicate of track 0,
    # tracks 4 and 5 duplicates of each other ONLY outside coverage (their shared steps sit north of the region), a whole-record fact
    ours = [(JUNE1 + 10.0, 12.0, 10.0, 8), (D + 5.0, 12.0, 10.0, 8), (JUNE1 + 10.0, -10.0, -100.0, 8), (JUNE1 + 10.0, 12.2, 10.1, 8),
            (JUNE1 + 20.0, 38.0, 0.0, 6), (JUNE1 + 20.0, 38.1, 0.05, 6)]
    if east_system:
        ours.append((JUNE1 + 50.0 + 0.75, 12.0, 39.0, 5))         # sits where the eastern system arrives, for its last five steps
    ours.append((JUNE1 + 30.0, 12.0, -50.0, 3))                    # exactly three observations in coverage: eligible, no counterpart
    ours.append((JUNE1 + 10.0, 12.0, 30.0 - 3.7, 8))               # about 400 km from system 2 at its times: a pair at 500 km only, the sensitivity difference
    if not our_grid_ok:
        ours[2] = (JUNE1 + 10.1, -10.0, -100.0, 8)
    _ours_file(str(ev / "tracker_port.mat"), ours)
    (ev / "tracking_era5_1990.json").write_text(json.dumps({"dataset_specific": {"dataset": "era5", "year": 1990, "tracks_sha256": _sha(ev / "tracker_port.mat")}}))
    _regions(str(tmp_path / "regions"))
    return str(tmp_path / "evidence"), str(qdir), str(adir), str(tmp_path / "regions")


def test_a_season_pairs_and_reports_every_population_subset_and_exclusion(tmp_path):
    M = _load()
    evidence, qdir, adir, regions = _world(tmp_path)
    out = tmp_path / "pilot_1990.json"
    assert M.run(1990, evidence, qdir, adir, regions, str(out)) == 0
    a = json.load(open(out))
    assert a["what_this_is"].startswith("stored-track correspondences")
    pa, pb = a["populations"]["a"], a["populations"]["b"]
    assert pa["total"] == 8 and pa["eligible"] == 5 and pa["no_coverage"] == [1, 4, 5] and pa["under_three"] == []
    assert 6 in [t for t in range(8)] and any(u["id"] == 6 for u in a["results"]["500"]["a"]["no_candidate"])    # exactly three in coverage: eligible
    assert any(p["a"] == 7 and p["b"] == 2 for p in a["results"]["500"]["pairs"])                                # about 400 km: paired at 500
    assert not any(p["a"] == 7 for p in a["results"]["350"]["pairs"]) and a["results"]["350"]["a"]["assigned"] == 1   # and not at 350
    assert pa["duplicate_group_members_whole_record"] == 4 and pa["duplicate_group_members_outside_eligible"] == 2   # tracks 4 and 5, duplicates outside coverage
    assert a["inputs"]["campaign_record"]["sha256"] == _sha(os.path.join(evidence, "era5_1990", "tracking_era5_1990.json"))
    assert set(a["inputs"]["region_polygons"]["sha256"]) == {name for _, name in __import__("season_metrics").REGION_FILES}
    assert pb["total"] == 4 and pb["eligible"] == 2 and pb["no_coverage"] == [4] and pb["under_three"] == [{"id": 3, "in_coverage": 2}]
    assert pa["africa_origin_in_season_ids"] == [0, 3, 7] and pb["atlantic_filter_ids"] == [1, 2]
    assert pa["duplicate_group_members"] == [0, 3]
    r = a["results"]["500"]
    assert r["a"]["n"] == 5 and r["b"]["n"] == 2 and r["a"]["assigned"] == 2 and r["b"]["assigned"] == 2
    assert {(p["a"], p["b"]) for p in r["pairs"]} in ({(0, 1), (7, 2)}, {(3, 1), (7, 2)})   # the duplicate pair shares one QTrack counterpart
    assert len(r["a"]["lost_assignment"]) == 1 and r["a"]["lost_assignment"][0]["candidates"] == 1     # the other duplicate lost the one-to-one
    assert [u["id"] for u in r["a"]["no_candidate"]] == [2, 6] and r["a"]["no_candidate"][0]["nearest_median_km"] > 5000   # our far track coexists in time with systems an ocean away
    assert r["b"]["no_candidate"] == []
    assert r["fractions"]["a_assigned_over_population_a"] == pytest.approx(2 / 5) and r["fractions"]["b_assigned_over_population_b"] == 1.0
    assert r["fractions"]["africa_origin_subset"] == {"eligible": 3, "assigned": 2, "lost_assignment": 1, "no_candidate": 0, "fraction": pytest.approx(2 / 3)}
    assert r["fractions"]["atlantic_filter_subset"] == {"eligible": 2, "assigned": 2, "lost_assignment": 0, "no_candidate": 0, "fraction": 1.0}
    assert r["pairs"][0]["a_first"]["lat"] in (12.0, 12.2) and r["pairs"][0]["b_first"]["lon"] == 10.5 and r["pairs"][0]["b_atlantic_filter"] is True
    assert r["duplicate_involvement"]["a_duplicate_groups"] == {"members": 2, "assigned": 1, "unassigned": 1}
    assert r["distributions"]["shared_observations"]["50"] == 8 and r["distributions"]["separation_km"]["10"] < 100 and r["distributions"]["separation_km"]["90"] > 300
    assert "350" in a["results"]
    assert a["inputs"]["qtrack"]["sha256"] == _sha(os.path.join(qdir, "ERA5_AEW_tracks_with_basins_1990.nc"))
    assert a["rules"]["window_days_since_1900"] == [JUNE1, JUNE1 + 152.75]
    with pytest.raises(SystemExit):                                                  # never overwritten
        M.run(1990, evidence, qdir, adir, regions, str(out))


def test_strict_patterns_refuse_ambiguity_and_companions_are_never_inputs(tmp_path):
    M = _load()
    evidence, qdir, adir, regions = _world(tmp_path)
    assert M.strict_path(qdir, M.QTRACK_FILE, 1990).endswith("ERA5_AEW_tracks_with_basins_1990.nc")
    assert M.strict_path(adir, M.ATLANTIC_FILE, 1990).endswith("ERA5_AEW_tracks_atlantic_1990.nc")
    with pytest.raises(SystemExit):
        M.strict_path(qdir, M.QTRACK_FILE, 1991)
    (tmp_path / "with_epac" / "ERA5_AEW_tracks_with_basins_1990.nc.bak").write_bytes(b"x")
    assert M.strict_path(qdir, M.QTRACK_FILE, 1990).endswith("1990.nc")


def test_off_grid_times_a_wrong_window_and_a_changed_tracks_file_are_refused(tmp_path):
    M = _load()
    evidence, qdir, adir, regions = _world(tmp_path, our_grid_ok=False)
    with pytest.raises(SystemExit) as exc:
        M.run(1990, evidence, qdir, adir, regions, str(tmp_path / "o1.json"))
    assert "six-hour grid" in str(exc.value)
    evidence, qdir, adir, regions = _world(tmp_path / "b", qtrack_steps=600)
    with pytest.raises(SystemExit) as exc:
        M.run(1990, evidence, qdir, adir, regions, str(tmp_path / "o2.json"))
    assert "window" in str(exc.value)
    evidence, qdir, adir, regions = _world(tmp_path / "c")
    with open(os.path.join(evidence, "era5_1990", "tracker_port.mat"), "ab") as fh:
        fh.write(b"\0")
    with pytest.raises(SystemExit) as exc:
        M.run(1990, evidence, qdir, adir, regions, str(tmp_path / "o3.json"))
    assert "not the one its record names" in str(exc.value)


def test_qtrack_time_is_decoded_through_the_files_own_units(tmp_path):
    M = _load()
    evidence, qdir, adir, regions = _world(tmp_path, qtrack_hours=False)
    out = tmp_path / "o.json"
    assert M.run(1990, evidence, qdir, adir, regions, str(out)) == 0
    a = json.load(open(out))
    assert a["inputs"]["qtrack"]["time_units"].startswith("seconds since 1970") and a["results"]["500"]["a"]["assigned"] == 2


def test_an_eastern_system_entering_the_region_is_eligible_and_can_pair(tmp_path):
    M = _load()
    evidence, qdir, adir, regions = _world(tmp_path, east_system=True)
    out = tmp_path / "o.json"
    assert M.run(1990, evidence, qdir, adir, regions, str(out)) == 0
    a = json.load(open(out))
    assert 5 in [t for t in range(1, 6)] and a["populations"]["b"]["eligible"] == 3           # the eastern system counts, with five steps in coverage
    assert any(p["b"] == 5 and p["a"] == 6 for p in a["results"]["500"]["pairs"])          # and pairs with the track waiting for it


def test_shared_tail_members_are_reported_from_qtracks_own_rule(tmp_path):
    M = _load()
    evidence, qdir, adir, regions = _world(tmp_path, shared_tail=True)
    out = tmp_path / "o.json"
    assert M.run(1990, evidence, qdir, adir, regions, str(out)) == 0
    a = json.load(open(out))
    assert a["populations"]["b"]["shared_tail_members"] == [2, 5] and a["populations"]["b"]["shared_tail_pairs"] == 1
    assert a["results"]["500"]["duplicate_involvement"]["b_shared_tail_members"] == {"members": 2, "assigned": 1, "unassigned": 1}


def test_an_off_grid_step_anywhere_on_qtracks_axis_is_refused_even_if_no_system_uses_it(tmp_path):
    M = _load()
    evidence, qdir, adir, regions = _world(tmp_path, unused_step_off_grid=True)
    with pytest.raises(SystemExit) as exc:
        M.run(1990, evidence, qdir, adir, regions, str(tmp_path / "o.json"))
    assert "six-hour grid" in str(exc.value)
