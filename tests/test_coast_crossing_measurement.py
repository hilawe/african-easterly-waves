"""The coast-crossing measurement's edge cases, bound with small tests: any qualifying
later observation is the event and the cutoff meridian counts as west, observations after
the common endpoint take no part, a track alive at the endpoint without an event is
incomplete, a group's outcome comes from cohort members only, single-observation entrants
and the left-censor exclusion are reported, and every denominator adds up."""
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
    spec = importlib.util.spec_from_file_location("coast_crossing_measurement", os.path.join(ROOT, "scripts", "coast_crossing_measurement.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _world(tmp_path):
    """Africa is the box 0 to 30 E, 0 to 30 N; the North Atlantic 40 W to 0, 0 to 30 N (the
    meridian 0 belongs to the North Atlantic by priority). The cutoff 7.5 W lies inside the
    North Atlantic box, so North Atlantic observations east of it are Gulf sector visits."""
    import test_coast_entry_examples as T0
    import test_qtrack_pairing_pilot as T
    import qtrack_pairing_pilot as M
    from scipy.io import savemat
    JUNE1 = T.JUNE1
    ev = tmp_path / "evidence" / "era5_1990"
    ev.mkdir(parents=True)
    endpoint = M._day(1990, 10, 31) + 0.75
    dec31 = M._day(1990, 12, 31) + 0.75
    tracks = {
        # 0: Gulf first, then exactly on the cutoff meridian, then beyond: Atlantic side, both sectors, two qualifying observations
        0: (JUNE1 + 10.0, [12.0] * 5, [10.0, 5.0, -3.0, -7.5, -12.0]),
        # 1: stays in Africa and ends in July: none, follow-up complete
        1: (JUNE1 + 12.0, [12.0] * 4, [20.0, 18.0, 16.0, 14.0]),
        # 2: Gulf sector only, ends in August: gulf only, complete
        2: (JUNE1 + 14.0, [10.0] * 3, [5.0, -2.0, -5.0]),
        # 3: starts September 30, stays in Africa through the endpoint, then reaches the Atlantic side in November:
        #    none and INCOMPLETE, and the November event must not count
        3: (M._day(1990, 9, 30), None, None),
        # 4: starts in May: not in the cohort at all
        4: (JUNE1 - 10.0, [12.0] * 4, [10.0, 5.0, -1.0, -9.0]),
        # 5: starts at 28 N: archive-rule member outside the exploratory cohort
        5: (JUNE1 + 16.0, [28.0] * 4, [10.0, 5.0, -1.0, -9.0]),
        # 6: starts in April, far from every other track: outside the season, so not in the cohort and in no group
        6: (JUNE1 - 50.0, [3.0] * 3, [28.0, 27.0, 26.0]),
        # 7: one qualifying observation then it ends: Atlantic side, a single-observation entrant
        7: (JUNE1 + 18.0, [8.0] * 3, [3.0, 1.0, -9.0]),
        # 8: a duplicate of 0 (same times, 0.2 degree apart): same group, both Atlantic side
        8: (JUNE1 + 10.0, [12.2] * 5, [10.0, 5.0, -3.0, -7.5, -12.0]),
        # 9: a duplicate of 1 that starts in May (excluded) and reaches the Atlantic side: it must give the group no outcome
        9: (M._day(1990, 5, 30), None, None),
        # 10: exactly on the cohort's upper edge, 25 N: in the cohort, band 20 to 25
        10: (JUNE1 + 20.0, [25.0] * 3, [12.0, 10.0, 8.0]),
        # 11: exactly 5 N: band 5 to 10
        11: (JUNE1 + 21.0, [5.0] * 3, [12.0, 10.0, 8.0]),
        # 12 and 13: duplicates of each other that both start in May: a group with NO cohort member, which must not be listed
        12: (JUNE1 - 8.0, [6.0] * 4, [25.0, 24.0, 23.0, 22.0]),
        13: (JUNE1 - 8.0, [6.2] * 4, [25.0, 24.0, 23.0, 22.0]),
        # 14: its only North Atlantic observation has raw longitude 7.4 W, which the quarter-degree rounding would place on
        #     the 7.5 W meridian: the raw longitude decides, so it is a Gulf visit, not Atlantic side
        14: (JUNE1 + 24.0, [9.0] * 3, [3.0, 1.0, -7.4]),
        # 15: reaches the Atlantic side exactly AT the endpoint, October 31 18Z: the observation at the endpoint counts
        15: (M._day(1990, 9, 29), None, None),
        # 16 and 17: duplicates, 16 Gulf-only and complete, 17 none and alive at the endpoint: the group's sector is Gulf-only
        #            and its follow-up is INCOMPLETE (section 0a)
        16: (JUNE1 + 26.0, [7.0] * 6, [5.0, 4.0, 3.0, 2.0, 1.0, -2.0]),
        17: (JUNE1 + 26.0, None, None),
        # 18 and 19: duplicates, both none, 19 alive at the endpoint: the group is INCOMPLETE
        18: (JUNE1 + 28.0, [21.0] * 3, [25.0, 24.0, 23.0]),
        19: (JUNE1 + 28.0, None, None),
        # 20: a duplicate of 3 that starts in May (excluded): the group {3, 20} has ONE cohort member and must keep
        #     3's status exactly, none and incomplete
        20: (M._day(1990, 5, 25), None, None),
        # 21 and 22: duplicates, 21 Atlantic side (two qualifying observations), 22 none and alive at the endpoint: the group is
        #            Atlantic side and its follow-up COMPLETE, because a member has the event
        21: (JUNE1 + 30.0, None, None),
        22: (JUNE1 + 30.0, None, None),
    }
    shared = [round(5.0 - 0.2 * k, 1) for k in range(24)]                                    # twenty-four close African steps, so the mean separation stays under one degree
    tracks[21] = (JUNE1 + 30.0, [13.0] * 26, shared + [-9.0, -9.5])
    n22 = int(round((endpoint + 0.5 - (JUNE1 + 30.0)) / 0.25)) + 1
    tracks[22] = (JUNE1 + 30.0, [13.2] * n22, shared + [shared[-1]] * (n22 - 24))          # holds its last African position past the endpoint
    n20 = int(round((M._day(1990, 10, 5) - tracks[20][0]) / 0.25)) + 1
    tracks[20] = (tracks[20][0], [10.2] * n20, [15.0] * n20)                              # beside track 3 through its September steps
    n15 = int(round((endpoint - tracks[15][0]) / 0.25)) + 1
    tracks[15] = (tracks[15][0], [11.0] * n15, [20.0] * (n15 - 1) + [-9.0])            # ends exactly at the endpoint, on the Atlantic side
    for i, lat, lons3 in ((17, 7.2, [5.0, 4.0, 3.0, 2.0, 1.0]), (19, 21.2, [25.0, 24.0, 23.0])):
        n = int(round((endpoint + 0.5 - tracks[i][0]) / 0.25)) + 1                      # runs two steps past the endpoint
        tracks[i] = (tracks[i][0], [lat] * n, lons3 + [lons3[-1]] * (n - len(lons3)))   # then holds its last African position
    n3 = int(round((dec31 - tracks[3][0]) / 0.25)) + 1
    lons3 = np.full(n3, 15.0)
    times3 = tracks[3][0] + 0.25 * np.arange(n3)
    lons3[times3 > endpoint + 1.0] = -20.0                                     # the Atlantic side only after the endpoint
    tracks[3] = (tracks[3][0], [10.0] * n3, list(lons3))
    t1 = JUNE1 + 12.0
    times9 = np.arange(M._day(1990, 5, 30), t1 + 0.75 + 1e-9, 0.25)
    lons9 = np.full(times9.size, 20.0)
    lons9[-4:] = [20.0, 18.0, 16.0, 14.0]                                       # the four steps shared with track 1
    times9 = np.concatenate([times9, times9[-1] + 0.25 * np.arange(1, 4)])
    lons9 = np.concatenate([lons9, [5.0, -3.0, -10.0]])                        # then on to the Atlantic side
    tracks[9] = (float(times9[0]), [12.2] * times9.size, list(lons9))
    payload = {"n": float(len(tracks)), "case_id": np.array(["c" * 32]), "producer_json": np.array(["{}"])}
    for i, (t0, lats, lons) in tracks.items():
        payload[f"time{i}"] = t0 + 0.25 * np.arange(len(lons))
        payload[f"lat{i}"], payload[f"lon{i}"] = np.array(lats, float), np.array(lons, float)
    savemat(str(ev / "tracker_port.mat"), payload)
    (ev / "tracking_era5_1990.json").write_text(json.dumps({"dataset_specific": {"dataset": "era5", "year": 1990, "tracks_sha256": T._sha(ev / "tracker_port.mat")}}))
    qdir = tmp_path / "with_epac"
    qdir.mkdir()
    # QTrack: 1 reaches the Atlantic side; 2 starts at the file's first step (left-censored); 3 is alive at the last step without an event
    # (incomplete); 4 and 5 share an identical four-step tail, 4 in the cohort and Atlantic side, 5 starting at 28 N (outside the cohort)
    systems = [(40, 12.0, 3.0, 4), (0, 12.0, 15.0, 4), (464, 12.0, 15.0, 148), (100, 12.0, 3.0, 6), (100, 28.0, 3.0, 6)]   # 3 starts September 25, alive to the last step
    T._qtrack_file(str(qdir / "ERA5_AEW_tracks_with_basins_1990.nc"), systems)
    import netCDF4 as nc
    d = nc.Dataset(str(qdir / "ERA5_AEW_tracks_with_basins_1990.nc"), "a")
    d.variables["AEW_lon"][0, 40:44] = [3.0, 1.0, -5.0, -10.0]
    d.variables["AEW_lon"][3, 100:106] = [3.0, 1.0, -9.0, -11.0, -13.0, -15.0]
    d.variables["AEW_lon"][4, 100:106] = [3.0, 1.0, -9.0, -11.0, -13.0, -15.0]
    d.variables["AEW_lat"][4, 100:106] = [28.0, 27.0, 26.0, 25.0, 24.0, 23.0]
    d.variables["AEW_lat"][3, 100:106] = [12.0, 12.0, 26.0, 25.0, 24.0, 23.0]   # identical positions on the last four steps
    d.close()
    T0._regions(str(tmp_path / "regions"))
    return str(tmp_path / "evidence"), str(qdir), str(tmp_path / "regions")


def test_events_endpoint_censoring_bands_and_denominators(tmp_path):
    C = _load()
    evidence, qdir, regions = _world(tmp_path)
    out = tmp_path / "coast.json"
    assert C.main(["--year", "1990", "--campaign-evidence", evidence, "--qtrack-dir", qdir, "--regions-dir", regions, "--out", str(out)]) == 0
    art = json.load(open(out))
    ours = art["sides"]["this_record"]
    r = {t["id"]: t for t in ours["tracks"]}
    assert sorted(r) == [0, 1, 2, 3, 7, 8, 10, 11, 14, 15, 16, 17, 18, 19, 21, 22] and ours["outside_cohort_latitude_ids"] == [5] and ours["left_censored_excluded_ids"] == []
    assert ours["archive_rule_n"] == 17                                                       # 20 starts in May, so it is not counted
    assert r[21]["outcome"] == "atlantic_side" and not r[21]["single_observation_entrant"] and r[22]["outcome"] == "none" and r[22]["follow_up_incomplete"]
    assert r[14]["outcome"] == "gulf_only" and r[14]["first_gulf"]["lon"] == -7.4               # raw longitude, not the rounded one
    assert r[15]["outcome"] == "atlantic_side" and r[15]["first_atlantic_side"]["time"].startswith("1990-10-31 18")
    assert r[16]["outcome"] == "gulf_only" and not r[16]["follow_up_incomplete"] and r[17]["outcome"] == "none" and r[17]["follow_up_incomplete"]
    assert r[18]["outcome"] == "none" and not r[18]["follow_up_incomplete"] and r[19]["follow_up_incomplete"]
    assert r[0]["outcome"] == "atlantic_side" and r[0]["sectors_visited"] == "both" and r[0]["first_atlantic_side"]["lon"] == -7.5 and r[0]["n_atlantic_side_observations"] == 2
    assert r[1]["outcome"] == "none" and not r[1]["follow_up_incomplete"]
    assert r[2]["outcome"] == "gulf_only" and r[2]["sectors_visited"] == "gulf" and not r[2]["follow_up_incomplete"]
    assert r[3]["outcome"] == "none" and r[3]["follow_up_incomplete"] and r[3]["alive_at_endpoint"] and r[3]["n_observations_after_endpoint"] > 0
    assert r[7]["outcome"] == "atlantic_side" and r[7]["single_observation_entrant"] and not r[0]["single_observation_entrant"]
    assert r[10]["band"] == "20 to 25" and r[11]["band"] == "5 to 10"
    assert all(t["first"]["lon"] is not None and t["first"]["lat"] is not None for t in ours["tracks"])
    tot = ours["by_band"]["total"]
    assert tot["cohort"] == 16 and tot["atlantic_side"] == 5 and tot["gulf_only"] == 3 and tot["none"] == 4 and tot["follow_up_incomplete"] == 4
    assert tot["atlantic_side"] + tot["gulf_only"] + tot["none"] + tot["follow_up_incomplete"] == tot["cohort"]
    assert tot["denominators"] == {"cohort": 16, "complete_follow_up": 12} and abs(tot["fraction_atlantic_side_over_complete_follow_up"] - 5 / 12) < 1e-12
    assert tot["single_observation_entrants"] == 2                                             # 7 and 15
    assert sum(row["cohort"] for lab, row in ours["by_band"].items() if lab != "total") == 16
    q = art["sides"]["qtrack"]
    qr = {t["id"]: t for t in q["tracks"]}
    assert sorted(qr) == [1, 3, 4] and q["left_censored_excluded_ids"] == [2] and q["outside_cohort_latitude_ids"] == [5]
    assert qr[1]["outcome"] == "atlantic_side" and qr[3]["follow_up_incomplete"] and qr[4]["outcome"] == "atlantic_side"
    assert art["rules"]["endpoint"].startswith("October 31 18Z")
    with pytest.raises(SystemExit):                                              # never overwritten
        C.main(["--year", "1990", "--campaign-evidence", evidence, "--qtrack-dir", qdir, "--regions-dir", regions, "--out", str(out)])


def test_grouping_sensitivity_uses_cohort_members_only(tmp_path):
    C = _load()
    evidence, qdir, regions = _world(tmp_path)
    out = tmp_path / "coast.json"
    assert C.main(["--year", "1990", "--campaign-evidence", evidence, "--qtrack-dir", qdir, "--regions-dir", regions, "--out", str(out)]) == 0
    art = json.load(open(out))
    ours = art["sides"]["this_record"]
    g = ours["grouping_sensitivity"]
    groups = g["groups_with_cohort_members"]
    # group {0, 8} has both members in the cohort, one unit, Atlantic side. In group {1, 9}, 9 is excluded (May start) though it reaches the Atlantic side
    by_members = {tuple(sorted(v["members_in_cohort"] + v["members_outside_cohort"])): v for v in groups.values()}
    assert (0, 8) in by_members and by_members[(0, 8)]["members_outside_cohort"] == []
    assert (1, 9) in by_members and by_members[(1, 9)]["members_in_cohort"] == [1] and by_members[(1, 9)]["members_outside_cohort"] == [9]
    assert g["units"] == 12 and g["grouped_units"] == 6                         # 16 cohort tracks: six singletons and six group units
    assert len(groups) == 6 and all(v["members_in_cohort"] for v in groups.values())   # only groups with a cohort member are listed
    unit_21 = [u for u in g["group_units"] if sorted(u["members"]) == [21, 22]][0]
    assert unit_21["outcome"] == "atlantic_side" and not unit_21["follow_up_incomplete"]     # an Atlantic-side member keeps the group complete
    unit_3 = [u for u in g["group_units"] if u["members"] == [3]][0]
    assert unit_3["outcome"] == "none" and unit_3["follow_up_incomplete"]                    # one cohort member: its status, exactly
    assert by_members[(3, 20)]["members_outside_cohort"] == [20]
    unit_1 = [u for u in g["group_units"] if u["members"] == [1]][0]
    assert unit_1["first"]["time"].startswith("1990-06-13") and unit_1["outcome"] == "none"   # first position and outcome from the cohort member, not the excluded May starter
    unit_16 = [u for u in g["group_units"] if sorted(u["members"]) == [16, 17]][0]
    assert unit_16["outcome"] == "gulf_only" and unit_16["follow_up_incomplete"]              # sector Gulf-only, follow-up incomplete (section 0a)
    unit_18 = [u for u in g["group_units"] if sorted(u["members"]) == [18, 19]][0]
    assert unit_18["outcome"] == "none" and unit_18["follow_up_incomplete"]                    # none with an alive member is incomplete
    tot = g["by_band"]["total"]
    assert tot["cohort"] == 12 and tot["atlantic_side"] == 4 and tot["none"] == 3 and tot["gulf_only"] == 2 and tot["follow_up_incomplete"] == 3
    assert tot["atlantic_side"] + tot["gulf_only"] + tot["none"] + tot["follow_up_incomplete"] == tot["cohort"]
    q = art["sides"]["qtrack"]["grouping_sensitivity"]
    qg = list(q["groups_with_cohort_members"].values())
    assert len(qg) == 1 and qg[0]["members_in_cohort"] == [4] and qg[0]["members_outside_cohort"] == [5]
    assert q["units"] == 3 and q["by_band"]["total"]["atlantic_side"] == 2


def test_band_edges_and_cutoff_convention():
    C = _load()
    assert C.band_of(0.0) == "0 to 5" and C.band_of(5.0) == "5 to 10" and C.band_of(24.999) == "20 to 25" and C.band_of(25.0) == "20 to 25"
    with pytest.raises(SystemExit):
        C.band_of(25.001)
    assert C.CUTOFF_LON == -7.5 and C.COHORT_LAT == (0.0, 25.0)
