"""The 2006 baseline measurement and its lineage reading, on hand-built fixtures.

What is tested: the distance to a read longitude given as a value or a range, the west of
40 E split, the claim status of a candidate near the read geometry, declared transitions
and intervals in place of R1's fixed join and band, finished coverage per declared
segment, and the lineage reading's two proximities, its stage per axis in the merges' own
order, a mixed map where inputs fail at different stages, and refusals on missing
evidence."""
import gzip
import hashlib
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import pilot_eastern_lineage_reading as L  # noqa: E402
import pilot_eastern_measurement as E  # noqa: E402
import pilot_r1_measurement as M  # noqa: E402


def D(name):
    return hashlib.sha256(name.encode()).hexdigest()


def entry(date, step, cands, seeds=(), claims=(), live=()):
    return {"step": step, "date": date + ":00Z", "n_live_before": len(live),
            "candidates": [{"index": i, "lat_mean": la, "lon_mean": lo, "n_points": 10, "region_sha256": dg} for i, (la, lo, dg) in enumerate(cands)],
            "live_before": [{"birth": b, "n_obs": 3, "last_step": step - 2, "last_lat": la, "last_lon": lo, "due": None} for b, la, lo in live],
            "seeds": [{"birth": b, "candidate": c, "lat": cands[c][0], "lon": cands[c][1]} for b, c in seeds],
            "claims": [{"birth": b, "candidate": c, "n_obs": 4, "lat": cands[c][0], "lon": cands[c][1]} for b, c in claims]}


def test_the_distance_takes_a_longitude_value_or_range_and_reduces_to_the_r1_distance():
    m = {"axis_lon": 44.0, "lat_range": [5.0, 11.0]}
    for lat, lon in ((8.0, 44.0), (12.0, 47.0), (2.0, 40.0)):
        assert E.ref_distance(lat, lon, m) == pytest.approx(M.axis_distance(lat, lon, 44.0, [5.0, 11.0]))
    r = {"axis_lon_range": [50.0, 53.0], "lat_range": [15.0, 30.0]}
    assert E.ref_distance(20.0, 51.5, r) == 0.0 and E.ref_distance(20.0, 56.0, r) == pytest.approx(3.0) and E.ref_distance(11.0, 47.0, r) == pytest.approx(5.0)
    assert E.west_of_40({"axis_lon": 39.0}) and not E.west_of_40({"axis_lon": 40.0}) and E.west_of_40({"axis_lon_range": [24.0, 36.0]})


def test_a_candidate_within_tolerance_reports_whether_a_history_holds_it():
    m = {"segment": "S", "scored": True, "axis_lon": 47.0, "lat_range": [10.0, 20.0]}
    e = entry("2006-08-01T00", 10, [(15.0, 46.0, D("a")), (15.0, 48.5, D("b")), (2.0, 10.0, D("c"))], seeds=[("10:0", 0)])
    r = E.measure_map(e, m, {"lat": [5.0, 25.0], "lon": [40.0, 55.0]}, 3.0, 5.0)
    held = {c["index"]: c["held_by"] for c in r["detection"]["within"]}
    assert held == {0: [{"birth": "10:0", "kind": "seed"}], 1: []} and r["detection"]["within_unclaimed"] == [1]
    assert r["detection"]["candidates_in_case_box"] == 2 and r["west_of_40E"] is False


def _write_gz(path, obj):
    blob = json.dumps(obj).encode()
    with gzip.open(path, "wb") as fh:
        fh.write(blob)
    return hashlib.sha256(blob).hexdigest()


def _declaration():
    return {"families": [{"name": "W", "window": ["2006-08-01T00", "2006-08-02T12"], "box": {"lat": [0, 30], "lon": [20, 62]}}],
            "lineage_dates": ["2006-08-01T12", "2006-08-02T00", "2006-08-02T12"],
            "reference": {"segments": {"S1": {"scored": True, "case_box": {"lat": [5.0, 25.0], "lon": [40.0, 55.0]}},
                                       "S2": {"scored": True, "case_box": {"lat": [5.0, 25.0], "lon": [30.0, 50.0]}}},
                          "maps": [{"date": "2006-08-01T12", "segment": "S1", "axis_lon": 47.0, "lat_range": [10.0, 20.0], "scored": True},
                                   {"date": "2006-08-02T00", "segment": "S2", "axis_lon": 41.0, "lat_range": [10.0, 20.0], "scored": True},
                                   {"date": "2006-08-02T12", "segment": "S2", "axis_lon": 39.0, "lat_range": [10.0, 20.0], "scored": True}],
                          "transitions": [{"name": "S1_S2", "from": {"date": "2006-08-01T12", "segment": "S1"}, "to": {"date": "2006-08-02T00", "segment": "S2"}}],
                          "intervals": [{"name": "I", "window": ["2006-08-01T00", "2006-08-01T12"], "lat": [5.0, 25.0], "lon": [40.0, 55.0]}]},
            "tolerances": {"positional_deg": 3.0, "alternative_deg": 5.0}}


def _fixture(tmp_path, damage=None):
    decl = _declaration()
    ref_path = tmp_path / "declaration.json"
    ref_path.write_text(json.dumps(decl))
    finished = [{"birth": "0:0", "time": [1.0, 1.5, 2.0, 2.5], "lat": [15.0, 15.0, 15.0, 15.0], "lon": [48.0, 46.5, 44.0, 41.5], "steps": [0, 2, 4, 6],
                 "raw_lat": [15.0, 15.0, 15.0, 15.0], "raw_lon": [48.0, 47.0, 44.0, 41.5], "n_points": [10, 10, 10, 10], "region_sha256": [D("c0"), D("c2"), D("c4"), D("c6")]}]
    retained = {"finished": finished, "configuration": "recomputed calibration, the domain's own", "preparation": "margin", "margin_edges": "all",
                "tracker_flags": {"absorb": False, "exclusive": False}, "year": 2006, "thresholds": {"coarse": 1e-7, "fine": 2e-6},
                "runs": {"B": {"case_sha256": "b" * 64}, "wide": {"case_sha256": "w" * 64}}}
    fin_path = tmp_path / "retained.json.gz"
    fin_digest = _write_gz(fin_path, retained)
    steps = [entry("2006-08-01T00", 0, [(15.0, 48.0, D("c0"))], seeds=[("0:0", 0)]),
             entry("2006-08-01T12", 2, [(15.0, 47.0, D("c2"))], claims=[("0:0", 0)], live=[("0:0", 15.0, 48.0)]),
             entry("2006-08-02T00", 4, [(15.0, 44.0, D("c4")), (15.0, 40.5, D("c4b"))], claims=[("0:0", 0)], live=[("0:0", 15.0, 47.0)]),
             entry("2006-08-02T12", 6, [(15.0, 41.5, D("c6"))], claims=[("0:0", 0)], live=[("0:0", 15.0, 44.0)])]
    histories = {"0:0": {"steps": [0, 2, 4, 6], "lon_claimed": [48.0, 47.0, 44.0, 41.5], "lat_claimed": [15.0, 15.0, 15.0, 15.0], "n_points": [10, 10, 10, 10],
                         "region_sha256": [D("c0"), D("c2"), D("c4"), D("c6")]}}
    trace = {"trace": {"families_sha256": M.sha256_of(str(ref_path)), "steps": steps, "histories": histories, "fates": {"0:0": "finished"}, "pruned_at": {}, "finished_index": {"0:0": 0}},
             "reproduces": {"finished_identical": True, "sha256": fin_digest}, "finished": json.loads(json.dumps(finished)),
             **{k: retained[k] for k in M.CONFIGURATION_KEYS}, "thresholds": dict(retained["thresholds"]), "runs": json.loads(json.dumps(retained["runs"])),
             "elapsed_seconds": 1.0}
    if damage is not None:
        damage(trace)
    trace_path = tmp_path / "trace.json.gz"
    digest = _write_gz(trace_path, trace)
    out = tmp_path / "measurement.json"
    return ["--reference", str(ref_path), "--trace", str(trace_path), "--finished", str(fin_path), "--out", str(out), "--trace-digest", digest[:12]], out


def test_entry_point_reports_segments_split_at_40E_transitions_intervals_and_finished_coverage(tmp_path):
    argv, out = _fixture(tmp_path)
    assert E.main(argv) == 0
    rec = json.loads(out.read_text())
    tol = rec["summary"]["tolerance"]
    assert tol["segments"]["S1"]["maps_with_a_candidate_within"] == ["2006-08-01T12"]
    assert tol["segments"]["S2"]["maps_with_a_candidate_within"] == ["2006-08-02T00", "2006-08-02T12"] and tol["segments"]["S2"]["west_of_40E_maps"] == ["2006-08-02T12"]
    split = rec["results"]["tolerance"]["segments"]["S2"]
    assert split["east_of_40E"]["inspected_maps"] == ["2006-08-02T00"] and split["west_of_40E"]["inspected_maps"] == ["2006-08-02T12"]
    assert "east_of_40E" not in rec["results"]["tolerance"]["segments"]["S1"]
    assert tol["segments"]["S2"]["maps_with_an_unclaimed_candidate_within"] == ["2006-08-02T00"]          # the 40.5 E candidate nobody holds
    assert tol["transitions"] == {"S1_S2": "continues"}
    assert rec["summary"]["intervals"] == {"I": {"3.0": {"west_moving": 1}, "5.0": {"west_moving": 1}}}
    fin = rec["finished_stage"]["finished_histories_near_a_scored_map"]
    assert len(fin) == 1 and set(fin[0]["coverage"]) == {"S1", "S2"} and fin[0]["coverage"]["S2"]["share"] == 1.0


@pytest.mark.parametrize("label, damage, message", [
    ("binding", lambda t: t["trace"].__setitem__("families_sha256", "0" * 64), "declared reference"),
    ("map missing", lambda t: t["trace"]["steps"].pop(3), "not among the trace's logged steps"),
    ("interval raw history missing", lambda t: (t["trace"]["histories"].pop("0:0"), t["trace"]["finished_index"].pop("0:0")), "raw history|finished index"),
])
def test_entry_point_refuses_missing_or_inconsistent_evidence(tmp_path, label, damage, message):
    argv, out = _fixture(tmp_path, damage)
    with pytest.raises(SystemExit, match=message):
        E.main(argv)
    assert not out.exists(), label


# --- the lineage reading -------------------------------------------------------------------

M0 = {"date": "2006-09-20T00", "segment": "R", "scored": True, "axis_lon": 50.0, "lat_range": [15.0, 25.0]}


def axis(i, lats, lons):
    return {"index": i, "lat": lats, "lon": lons, "lat_mean": sum(lats) / len(lats), "lon_mean": sum(lons) / len(lons), "n_vertices": len(lats)}


def sel(lat, lon, level=0, exhausted=False):
    return {"levels": [], "level_landed": level, "last_level": 5, "exhausted": exhausted, "selected_cells": 9, "selected_lat_extent": 4.0, "selected_lon_extent": 4.0,
            "region": {"cells": 9, "region_sha256": D(f"{lat},{lon}"), "median_lat": lat, "median_lon": lon}, "base_region_cells": 9}


def test_the_two_proximities_are_kept_apart_and_a_far_center_is_its_own_stage():
    a = axis(0, [20.0, 32.0, 34.0, 36.0], [50.0, 50.5, 52.0, 53.0])               # a vertex on the read axis, its mean pulled north
    lin = {"axes": [a], "coarse": [{"input": 0, "lat_mean": a["lat_mean"], "lon_mean": a["lon_mean"], "select": sel(20.0, 50.0), "merged": 0,
                                    "kept_after_pass_two": True, "output": 0, "rejected_min_extent": False}],
           "fine": [{"input": 0, "lat_mean": 20.0, "lon_mean": 50.0, "select": sel(20.0, 50.0), "merged": 0, "kept_after_pass_two": True, "output": 0, "rejected_min_extent": False}]}
    r = L.follow(a, lin, [{"index": 0, "lat_mean": 20.0, "lon_mean": 50.0}], M0, 3.0)
    assert r["vertex_distance_deg"] == 0.0 and r["center_distance_deg"] > 3.0
    assert r["stage"] == "the merge-input center lies beyond tolerance"
    assert r["positions"][-1]["stage"] == "final candidate" and r["positions"][-1]["distance_deg"] == 0.0      # the later return stays visible


def test_a_map_with_inputs_failing_at_different_stages_is_mixed_and_a_dropped_input_names_its_survivor():
    a0, a1, a2 = axis(0, [17.0, 19.0], [49.0, 49.0]), axis(1, [21.0, 23.0], [51.0, 51.0]), axis(2, [5.0, 6.0], [10.0, 10.0])
    lin = {"axes": [a0, a1, a2],
           "coarse": [{"input": 0, "lat_mean": 18.0, "lon_mean": 49.0, "select": sel(23.0, 54.0, level=5, exhausted=True), "merged": 0, "kept_after_pass_two": True, "output": 0, "rejected_min_extent": False},
                      {"input": 1, "lat_mean": 22.0, "lon_mean": 51.0, "select": sel(22.0, 51.0), "merged": 1, "kept_after_pass_two": False, "output": None, "rejected_min_extent": False},
                      {"input": 2, "lat_mean": 5.5, "lon_mean": 10.0, "select": sel(5.5, 10.0), "merged": 2, "kept_after_pass_two": True, "output": 1, "rejected_min_extent": False}],
           "fine": [{"input": 0, "lat_mean": 23.0, "lon_mean": 54.0, "select": sel(23.0, 54.0), "merged": 0, "kept_after_pass_two": True, "output": 0, "rejected_min_extent": False},
                    {"input": 1, "lat_mean": 5.5, "lon_mean": 10.0, "select": sel(5.5, 10.0), "merged": 1, "kept_after_pass_two": True, "output": 1, "rejected_min_extent": False}]}
    e = {"candidates": [{"index": 0, "lat_mean": 23.0, "lon_mean": 54.0}, {"index": 1, "lat_mean": 5.5, "lon_mean": 10.0}], "lineage": lin}
    r = L.read_map(e, M0, 3.0)
    assert r["stage"] == "mixed" and len(r["relevant_axes"]) == 2
    by = {x["axis"]: x for x in r["relevant_axes"]}
    assert by[0]["stage"] == "coarse merge, the selected region's position lies beyond tolerance" and by[0]["positions"][0]["exhausted"] is True
    assert by[1]["stage"] == "coarse merge, dropped in the duplicate pass" and by[1]["dropped_near"][0]["input"] == 0
    far = L.read_map({"candidates": [], "lineage": {"axes": [a2], "coarse": [], "fine": []}}, M0, 3.0)
    assert far["stage"] == L.NONE_NEAR and far["nearest_axis_vertex_distance_deg"] > 30.0


def test_a_broken_chain_is_unresolved_and_a_map_without_lineage_refuses():
    a = axis(0, [20.0, 20.0], [50.0, 50.0])
    r = L.follow(a, {"axes": [a], "coarse": [{"input": 0, "lat_mean": 20.0, "lon_mean": 50.0, "select": sel(20.0, 50.0), "merged": 0,
                                               "kept_after_pass_two": True, "output": 3, "rejected_min_extent": False}], "fine": []}, [], M0, 3.0)
    assert r["stage"].startswith("unresolved")
    with pytest.raises(SystemExit, match="carries no lineage"):
        L.read_map({"candidates": []}, M0, 3.0)


# --- an alternative reading (EASTERN_ALTERNATIVE_READINGS_2026-10-13.md, section 1) --------

def test_a_map_with_several_geometries_takes_the_nearest():
    m = {"axis_lon": 44.0, "lat_range": [10.0, 20.0], "geometries": [{"axis_lon": 44.0, "lat_range": [10.0, 20.0]}, {"axis_lon_range": [58.0, 60.0], "lat_range": [10.0, 20.0]}]}
    assert E.ref_distance(15.0, 57.0, m) == pytest.approx(1.0) and E.ref_distance(15.0, 46.0, m) == pytest.approx(2.0)


def test_an_alternative_reading_replaces_geometry_where_it_scores_and_sets_the_rest_aside(tmp_path):
    argv, out = _fixture(tmp_path)
    alt = {"source": "synthetic", "maps": {"2006-08-01T12": [{"status": "identified", "axis_lon": 48.0, "lat_range": [10.0, 20.0]}],
                                           "2006-08-02T00": [{"status": "identified", "axis_lon": 44.0, "lat_range": [10.0, 20.0]},
                                                             {"status": "weak", "axis_lon": 40.5, "lat_range": [10.0, 20.0]}],
                                           "2006-08-02T12": [{"status": "indeterminate"}]}}
    alt_path = tmp_path / "alt.json"
    alt_path.write_text(json.dumps(alt))
    assert E.main(argv + ["--alternative-reading", str(alt_path)]) == 0
    rec = json.loads(out.read_text())
    block = rec["alternative_reading"]
    assert block["source"] == "synthetic" and block["maps_scored"] == 2 and [u["date"] for u in block["maps_unscored"]] == ["2006-08-02T12"]
    assert block["transitions_skipped"] == [] and block["weak_only_maps"] == []
    tol = rec["summary"]["tolerance"]["segments"]
    assert tol["S1"]["maps"] == 1 and tol["S2"]["maps"] == 1 and tol["S2"]["maps_unscored_by_reading"] == ["2006-08-02T12"]
    assert tol["S2"]["maps_with_an_unclaimed_candidate_within"] == ["2006-08-02T00"], "the 40.5 E candidate is within tolerance of the second geometry"
    per = {m["date"]: m for m in rec["results"]["tolerance"]["maps"]}
    assert set(per) == {"2006-08-01T12", "2006-08-02T00"} and per["2006-08-02T00"]["detection"]["nearest"]["distance_deg"] == 0.0
    assert rec["summary"]["tolerance"]["transitions"] == {"S1_S2": "continues"} and rec["reference_declared"]["reference"]["maps"][2]["axis_lon"] == 39.0, "the declaration is recorded unchanged"
    alt["maps"]["2006-08-01T12"] = [{"status": "none"}]
    alt_path.write_text(json.dumps(alt))
    out2 = tmp_path / "m2.json"
    argv2 = [a if a != str(out) else str(out2) for a in argv]
    assert E.main(argv2 + ["--alternative-reading", str(alt_path)]) == 0
    rec2 = json.loads(out2.read_text())
    assert rec2["alternative_reading"]["transitions_skipped"] == ["S1_S2"] and rec2["summary"]["tolerance"]["transitions"] == {}
