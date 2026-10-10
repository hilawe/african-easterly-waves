"""The cohort assessment reads correspondence classes from the comparison's pairing, reads
membership changes only across identical or mutual counterparts, keeps ambiguous and
unmatched members apart, and never lets an exact history show a differing summary."""
import importlib.util
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def _load(name):
    spec = importlib.util.spec_from_file_location(name + "_under_test", os.path.join(ROOT, "scripts", name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _finished(birth, obs):
    steps = [o[0] for o in obs]
    return {"birth": birth, "steps": steps, "raw_lat": [o[1] for o in obs], "raw_lon": [o[2] for o in obs], "time": [100 + 0.25 * s for s in steps],
            "lat": [o[1] for o in obs], "lon": [o[2] for o in obs], "n_points": [1] * len(obs), "region_sha256": ["r"] * len(obs)}


def test_correspondence_classes_cover_every_finished_track_once():
    C = _load("pilot_margin_cohort")
    a = [(0, 10.0, 30.0), (1, 10.0, 28.5), (2, 10.0, 27.0)]
    b = [(0, 10.0, 30.0), (1, 10.0, 28.5), (2, 9.0, 26.0)]
    c = [(5, 20.0, 20.0), (6, 20.0, 18.5)]
    f = [(9, 0.0, -50.0)]
    native = [_finished("0:0", a), _finished("5:0", c), _finished("9:0", f)]
    margin = [_finished("0:1", b), _finished("5:2", c)]
    cls_n, cls_m = C.correspondence(native, margin)
    assert cls_n == {0: ("mutual_diverging", 0), 1: ("identical", 1), 2: ("unmatched", None)}
    assert cls_m == {0: ("mutual_diverging", 0), 1: ("identical", 1)}


def test_membership_changes_are_read_across_counterparts_and_exact_histories_cannot_differ():
    C = _load("pilot_margin_cohort")
    C.REGIONS = []                                                                             # source_region returns OTH for every track, fine for this read
    native = [_finished("0:0", [(0, 10.0, 30.0), (1, 10.0, 28.5)]), _finished("5:0", [(5, 20.0, 20.0), (6, 20.0, 18.5)]), _finished("8:0", [(8, 5.0, 25.0), (9, 5.0, 23.5)])]
    margin = [_finished("0:1", [(0, 10.0, 30.0), (1, 10.5, 27.0), (2, 10.5, 25.5)]), _finished("5:2", [(5, 20.0, 20.0), (6, 20.0, 18.5)]), _finished("8:3", [(8, 5.0, 25.0), (9, 4.0, 22.0)])]
    cls_n, cls_m = C.correspondence(native, margin)
    rec = lambda i, outcome, band, n: {"id": i, "outcome": outcome, "follow_up_incomplete": False, "band": band, "n_observations": n, "first": {}}
    meas_n = {"tracks": [rec(0, "none", "10 to 15", 2), rec(1, "none", "20 to 25", 2), rec(2, "atlantic_side", "5 to 10", 2)]}
    meas_m = {"tracks": [rec(0, "atlantic_side", "10 to 15", 3), rec(1, "none", "20 to 25", 2)]}                 # native 2's counterpart (margin 2) is not in the margin cohort
    tn, tm = C.as_tracks(native), C.as_tracks(margin)
    out = C.cohort_correspondence(meas_n, meas_m, cls_n, cls_m, tn, tm)
    assert out["native_cohort_by_class"] == {"mutual_diverging": 2, "identical": 1} and out["margin_cohort_by_class"] == {"mutual_diverging": 1, "identical": 1}
    assert out["members_in_both_cohorts_through_a_counterpart"] == {"exact_histories": 1, "mutual_pairs": 1}
    assert out["left_the_cohort_through_a_counterpart"]["count"] == 1 and out["left_the_cohort_through_a_counterpart"]["rows"][0]["native"] == 2
    assert out["entered_the_cohort_through_a_counterpart"]["count"] == 0
    d = out["mutual_pairs_in_both_cohorts"]["summaries_differing"]
    assert d == {"outcome": 1, "follow_up_incomplete": 0, "single_observation_entrant": 0, "first_latitude_band": 0, "genesis_longitude_band": 0, "any_summary": 1}
    assert out["mutual_pairs_in_both_cohorts"]["observations_difference_margin_minus_native"]["bins_by_absolute_value"] == {"0": 0, "1 to 3": 1, "4 to 7": 0, "8 or more": 0}
    # an identical pair that disagrees in any summary, in its cohort membership or in its finished arrays is refused
    for field, value in (("outcome", "atlantic_side"), ("follow_up_incomplete", True), ("single_observation_entrant", True), ("band", "15 to 20"), ("n_observations", 3)):
        bad = {"tracks": [rec(0, "atlantic_side", "10 to 15", 3), {**rec(1, "none", "20 to 25", 2), field: value}]}
        with pytest.raises(SystemExit, match="differs in a summary"):
            C.cohort_correspondence(meas_n, bad, cls_n, cls_m, tn, tm)
    only_one_side = {"tracks": [rec(0, "atlantic_side", "10 to 15", 3)]}                        # the identical history missing from the margin cohort
    with pytest.raises(SystemExit, match="in one cohort and not the other"):
        C.cohort_correspondence(meas_n, only_one_side, cls_n, cls_m, tn, tm)
    tm_moved = [dict(t) for t in tm]
    tm_moved[1] = {**tm_moved[1], "lat": tm_moved[1]["lat"] + 0.5}
    with pytest.raises(SystemExit, match="finished lat array"):
        C.cohort_correspondence(meas_n, meas_m, cls_n, cls_m, tn, tm_moved)
    tm_time = [dict(t) for t in tm]
    tm_time[1] = {**tm_time[1], "time": tm_time[1]["time"] + 0.25}
    with pytest.raises(SystemExit, match="finished time array"):
        C.cohort_correspondence(meas_n, meas_m, cls_n, cls_m, tn, tm_time)
    tm_lon = [dict(t) for t in tm]
    tm_lon[1] = {**tm[1], "lon": tm[1]["lon"] + np.array([0.0] + [0.5] * (tm[1]["lon"].size - 1))}   # the first longitude kept, so the genesis band is unchanged
    with pytest.raises(SystemExit, match="finished lon array"):
        C.cohort_correspondence(meas_n, meas_m, cls_n, cls_m, tn, tm_lon)
    tm_band = [dict(t) for t in tm]
    tm_band[1] = {**tm[1], "lon": tm[1]["lon"] - 0.5}                                             # 20.0 E to 19.5 E, across the genesis-band edge at 20 E
    with pytest.raises(SystemExit, match="'genesis_longitude_band': True"):                      # the genesis summary itself refuses, before the array guard
        C.cohort_correspondence(meas_n, meas_m, cls_n, cls_m, tn, tm_band)


def test_an_ambiguous_counterpart_carries_no_membership_and_an_entrant_is_read_through_its_counterpart():
    C = _load("pilot_margin_cohort")
    C.REGIONS = []
    shared = [(1, 10.0, 30.0), (2, 10.0, 28.5)]
    native = [_finished("0:0", [(0, 11.0, 32.0)] + shared), _finished("0:1", [(0, 9.0, 32.0)] + shared), _finished("6:0", [(6, 27.0, 20.0), (7, 26.0, 18.0)])]
    margin = [_finished("0:0", [(0, 10.0, 31.0)] + shared), _finished("6:1", [(6, 24.0, 20.0), (7, 26.0, 18.0)])]
    cls_n, cls_m = C.correspondence(native, margin)
    assert cls_n[0] == ("ambiguous", None) and cls_n[1] == ("ambiguous", None) and cls_m[0] == ("ambiguous", None)
    assert cls_n[2] == ("mutual_converging", 1) and cls_m[1] == ("mutual_converging", 2)
    rec = lambda i, outcome, band, n: {"id": i, "outcome": outcome, "follow_up_incomplete": False, "band": band, "n_observations": n, "first": {}}
    meas_n = {"tracks": [rec(0, "none", "10 to 15", 3), rec(1, "none", "5 to 10", 3)]}            # native 2 starts at 27 N, outside the cohort
    meas_m = {"tracks": [rec(0, "none", "10 to 15", 3), rec(1, "none", "20 to 25", 2)]}            # margin 1 starts at 24 N, inside
    out = C.cohort_correspondence(meas_n, meas_m, cls_n, cls_m, C.as_tracks(native), C.as_tracks(margin))
    assert out["native_cohort_by_class"] == {"ambiguous": 2} and out["margin_cohort_by_class"] == {"ambiguous": 1, "mutual_converging": 1}
    assert out["left_the_cohort_through_a_counterpart"]["count"] == 0 and out["members_in_both_cohorts_through_a_counterpart"] == {"exact_histories": 0, "mutual_pairs": 0}
    e = out["entered_the_cohort_through_a_counterpart"]
    assert e["count"] == 1 and e["rows"][0]["margin"] == 1 and e["rows"][0]["native_first"] == [27.0, 20.0] and e["by_kind"] == {"mutual_converging": 1}
    assert out["without_a_counterpart"] == {"native": {"ambiguous": 2, "unmatched": 0}, "margin": {"ambiguous": 1, "unmatched": 0}}


def test_the_command_measures_each_runs_finished_arrays_with_its_own_indices_year_and_first_day(tmp_path, monkeypatch):
    C = _load("pilot_margin_cohort")
    import gzip, hashlib, json
    native_fin = [_finished("0:0", [(0, 10.0, 30.0), (1, 10.0, 28.5)])]
    margin_fin = [_finished("0:1", [(0, 10.0, 30.0), (1, 10.5, 27.0)])]
    native_fin[0]["lat"], native_fin[0]["lon"] = [10.2, 10.2], [29.9, 28.6]                       # smoothed arrays unequal to the raw ones
    margin_fin[0]["lat"], margin_fin[0]["lon"] = [10.3, 10.3], [29.8, 27.2]
    def gz(name, payload):
        p = tmp_path / name
        blob = json.dumps(payload).encode()
        with gzip.open(p, "wb") as fh:
            fh.write(blob)
        return str(p), hashlib.sha256(blob).hexdigest()
    native, sha_n = gz("n.json.gz", {"year": 2006, "preparation": "native", "archive_comparison": {"passed": True}, "finished": native_fin})
    margin, sha_m = gz("m.json.gz", {"year": 2006, "preparation": "margin", "archive_comparison": {"passed": False}, "finished": margin_fin})
    cmp = tmp_path / "c.json"
    cmp.write_text(json.dumps({"inputs": {"native": {"sha256": sha_n}, "margin": {"sha256": sha_m}}}))
    calls, group_calls = [], []
    def fake_measure(tracks, year, regions, first_day, membership):
        calls.append({"ids": [t["id"] for t in tracks], "lat": [t["lat"].tolist() for t in tracks], "lon": [t["lon"].tolist() for t in tracks],
                      "time": [t["time"].tolist() for t in tracks], "year": year, "first_day": first_day, "membership": membership})
        rows = [{"id": t["id"], "first": {}, "band": "10 to 15", "n_observations": int(t["time"].size), "group": None, "outcome": "none", "follow_up_incomplete": False, "single_observation_entrant": False} for t in tracks]
        empty = {lab: {"cohort": 0, "atlantic_side": 0, "gulf_only": 0, "none": 0, "follow_up_incomplete": 0, "single_observation_entrants": 0,
                       "fraction_atlantic_side_over_cohort": None, "fraction_atlantic_side_over_complete_follow_up": None, "denominators": {"cohort": 0, "complete_follow_up": 0}} for lab in C.CC.band_labels() + ["total"]}
        return {"endpoint": "e", "cohort_n": len(rows), "outside_cohort_latitude_ids": [], "left_censored_excluded_ids": [], "archive_rule_n": len(rows), "by_band": empty,
                "grouping_sensitivity": {"units": 0, "grouped_units": 0, "by_band": empty, "groups_with_cohort_members": {}, "group_units": []}, "tracks": rows}
    def fake_groups(tracks):
        group_calls.append([t["lon"].tolist() for t in tracks])
        return {"grouping": len(group_calls)}                                                       # a distinct result per call, so each run must receive its own
    monkeypatch.setattr(C.CC, "measure", fake_measure)
    monkeypatch.setattr(C.CC, "groups_ours", fake_groups)
    monkeypatch.setattr(C.S, "load_regions", lambda d: ([], {}))
    monkeypatch.setattr(C.S, "spacing", lambda tracks: {"duration_days_percentiles": None})
    out = tmp_path / "o.json"
    assert C.main(["--native", native, "--margin", margin, "--compare", str(cmp), "--regions-dir", "d", "--out", str(out)]) == 0
    assert [c["ids"] for c in calls] == [[0], [0]] and calls[0]["lat"] == [[10.2, 10.2]] and calls[1]["lat"] == [[10.3, 10.3]]     # finished arrays, each run's own
    assert calls[0]["lon"] == [[29.9, 28.6]] and calls[1]["lon"] == [[29.8, 27.2]] and calls[0]["time"] == [[100.0, 100.25]] == calls[1]["time"]
    assert all(c["year"] == 2006 and c["first_day"] == C.M._day(2006, 1, 1) for c in calls)
    assert group_calls == [[[29.9, 28.6]], [[29.8, 27.2]]] and calls[0]["membership"] == {"grouping": 1} and calls[1]["membership"] == {"grouping": 2}
    rec = json.loads(out.read_text())
    assert rec["runs"]["native"]["cohort_n"] == 1 and rec["correspondence"]["native_cohort_by_class"] == {"mutual_diverging": 1}


def test_an_existing_output_and_a_mismatched_comparison_are_refused(tmp_path):
    C = _load("pilot_margin_cohort")
    out = tmp_path / "o.json"
    out.write_text("{}")
    with pytest.raises(SystemExit, match="never overwritten"):
        C.main(["--native", "a", "--margin", "b", "--compare", "c", "--regions-dir", "d", "--out", str(out)])
    import gzip, json
    def gz(name, payload):
        p = tmp_path / name
        with gzip.open(p, "wb") as fh:
            fh.write(json.dumps(payload).encode())
        return str(p)
    native = gz("n.json.gz", {"year": 1990, "preparation": "native", "archive_comparison": {"passed": True}, "finished": []})
    margin = gz("m.json.gz", {"year": 1990, "preparation": "margin", "archive_comparison": {"passed": False}, "finished": []})
    cmp = tmp_path / "c.json"
    cmp.write_text(json.dumps({"inputs": {"native": {"sha256": "x"}, "margin": {"sha256": "y"}}}))
    with pytest.raises(SystemExit, match="not made from these replay records"):
        C.main(["--native", native, "--margin", margin, "--compare", str(cmp), "--regions-dir", "d", "--out", str(tmp_path / "o2.json")])


def test_the_cohort_gate_follows_the_declared_contrast():
    C = _load("pilot_margin_cohort")
    nat = {"preparation": "native", "archive_comparison": {"passed": True}}
    mar = {"preparation": "margin", "archive_comparison": {"passed": False}}
    assert C.admit("preparation", nat, mar) == "preparation"
    with pytest.raises(SystemExit, match="did not reproduce"):
        C.admit("preparation", {**nat, "archive_comparison": {"passed": False}}, mar)
    assert C.admit("calibration", mar, mar) == "calibration"
    assert C.admit("extent", mar, mar) == "extent"
    with pytest.raises(SystemExit, match="calibration contrast"):                                     # the reference is a margin run there
        C.admit("calibration", nat, mar)
    with pytest.raises(SystemExit, match="did not reproduce"):
        C.admit("preparation_and_calibration", {**nat, "archive_comparison": {"passed": False}}, mar)
    with pytest.raises(SystemExit, match="unknown contrast"):
        C.admit("x", nat, mar)
    with pytest.raises(SystemExit, match="candidate is not a margin"):
        C.admit("preparation", nat, nat)


def test_an_extent_cohort_carries_its_comparisons_verification_identity_and_refuses_one_without():
    C = _load("pilot_margin_cohort")
    ident = {"path": "v.json", "sha256": "x" * 64, "script_sha256": "v", "replays": {"a": "1" * 64, "b": "2" * 64}}
    assert C.verification_identity({"contrast": "extent", "verification": ident}, "extent") == ident
    assert C.verification_identity({"contrast": "calibration"}, "calibration") is None
    with pytest.raises(SystemExit, match="extent comparison carries no verification"):
        C.verification_identity({"contrast": "extent"}, "extent")
    with pytest.raises(SystemExit, match="extent comparison carries no verification"):
        C.verification_identity({"contrast": "extent", "verification": {"path": "v.json"}}, "extent")


def test_an_extent_cohort_command_requires_and_carries_the_comparisons_verification(tmp_path, monkeypatch):
    C = _load("pilot_margin_cohort")
    import gzip, hashlib, json
    fin = [_finished("0:0", [(0, 10.0, 30.0), (1, 10.0, 28.5)])]
    def gz(name, payload):
        p = tmp_path / name
        blob = json.dumps(payload).encode()
        with gzip.open(p, "wb") as fh:
            fh.write(blob)
        return str(p), hashlib.sha256(blob).hexdigest()
    native, sha_n = gz("n.json.gz", {"year": 2006, "preparation": "margin", "archive_comparison": {"passed": False}, "finished": fin})
    margin, sha_m = gz("m.json.gz", {"year": 2006, "preparation": "margin", "archive_comparison": {"passed": False}, "finished": fin})
    ident = {"path": "v.json", "sha256": "x" * 64, "script_sha256": "v", "replays": {"a": sha_n, "b": sha_m}, "steps": 1460, "extent_is_the_only_difference": True}
    def cmp_file(name, **extra):
        p = tmp_path / name
        p.write_text(json.dumps({"inputs": {"native": {"sha256": sha_n}, "margin": {"sha256": sha_m}}, "contrast": "extent", **extra}))
        return str(p)
    def fake_measure(tracks, year, regions, first_day, membership):
        rows = [{"id": t["id"], "first": {}, "band": "10 to 15", "n_observations": int(t["time"].size), "group": None, "outcome": "none", "follow_up_incomplete": False, "single_observation_entrant": False} for t in tracks]
        empty = {lab: {"cohort": 0, "atlantic_side": 0, "gulf_only": 0, "none": 0, "follow_up_incomplete": 0, "single_observation_entrants": 0,
                       "fraction_atlantic_side_over_cohort": None, "fraction_atlantic_side_over_complete_follow_up": None, "denominators": {"cohort": 0, "complete_follow_up": 0}} for lab in C.CC.band_labels() + ["total"]}
        return {"endpoint": "e", "cohort_n": len(rows), "outside_cohort_latitude_ids": [], "left_censored_excluded_ids": [], "archive_rule_n": len(rows), "by_band": empty,
                "grouping_sensitivity": {"units": 0, "grouped_units": 0, "by_band": empty, "groups_with_cohort_members": {}, "group_units": []}, "tracks": rows}
    monkeypatch.setattr(C.CC, "measure", fake_measure)
    monkeypatch.setattr(C.CC, "groups_ours", lambda tracks: {})
    monkeypatch.setattr(C.S, "load_regions", lambda d: ([], {}))
    monkeypatch.setattr(C.S, "spacing", lambda tracks: {"duration_days_percentiles": None})
    out = tmp_path / "o.json"
    assert C.main(["--native", native, "--margin", margin, "--compare", cmp_file("c.json", verification=ident), "--regions-dir", "d", "--out", str(out)]) == 0
    rec = json.loads(out.read_text())
    assert rec["verification"] == ident and rec["contrast"] == "extent"                                   # the identity is carried as read
    with pytest.raises(SystemExit, match="carries no verification identity"):
        C.main(["--native", native, "--margin", margin, "--compare", cmp_file("c2.json"), "--regions-dir", "d", "--out", str(tmp_path / "o2.json")])
    with pytest.raises(SystemExit, match="not bound to these two replays"):
        C.main(["--native", native, "--margin", margin, "--compare", cmp_file("c3.json", verification={**ident, "replays": {"a": sha_n, "b": "0" * 64}}), "--regions-dir", "d", "--out", str(tmp_path / "o3.json")])
