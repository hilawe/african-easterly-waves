"""The R1 measurement pilot's helpers and its entry point, on hand-built fixtures.

The design's distinctions are what is tested: distance to the axis segment, both
tolerances, sequential coverage listed without a fragmentation label, an overlap that is a
confirmed duplicate only with the same well-formed region claimed, the join carrying
positions and lifecycle, raw and smoothed positions kept apart at the finished stage, the
standing band's maps enumerated from the declaration with refusals on missing evidence
and numerical displacement categories at both parameters, the binding as complete record
consistency, and the input contract enforced at the entry point so that missing or
malformed evidence refuses rather than reading as an inspected absence."""
import gzip
import hashlib
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import pilot_r1_measurement as M  # noqa: E402


def D(name):
    """A well-formed region digest for a fixture."""
    return hashlib.sha256(name.encode()).hexdigest()


REF = {
    "case_box": {"lat": [5.0, 20.0], "lon": [20.0, 55.0]},
    "segments": {"A": ["1990-09-08T00", "1990-09-08T12"], "B": ["1990-09-09T00", "1990-09-10T12"]},
    "maps": [
        {"date": "1990-09-08T00", "segment": "A", "axis_lon": 47.0, "lat_range": [5.0, 11.0], "scored": True},
        {"date": "1990-09-08T12", "segment": "A", "axis_lon": 44.0, "lat_range": [5.0, 11.0], "scored": True},
        {"date": "1990-09-09T00", "segment": "B", "axis_lon": 40.0, "lat_range": [12.0, 15.0], "scored": True},
        {"date": "1990-09-09T12", "segment": "B", "axis_lon": 35.0, "lat_range": [9.0, 15.0], "scored": True},
        {"date": "1990-09-10T00", "segment": "B", "axis_lon": 31.0, "lat_range": [11.0, 17.0], "scored": True},
        {"date": "1990-09-10T12", "segment": "B", "axis_lon": 28.0, "lat_range": [11.0, 17.0], "scored": True},
    ],
    "standing_band": {"window": ["1990-09-05T00", "1990-09-06T12"], "lon": [40.0, 48.0], "lat": [8.0, 25.0], "scored": False},
}


def entry(date, step, cands, seeds=(), claims=(), live=()):
    return {"step": step, "date": date + ":00Z", "n_live_before": len(live),
            "candidates": [{"index": i, "lat_mean": la, "lon_mean": lo, "n_points": 10, "region_sha256": dg} for i, (la, lo, dg) in enumerate(cands)],
            "live_before": [{"birth": b, "n_obs": 3, "last_step": step - 2, "last_lat": la, "last_lon": lo, "due": None} for b, la, lo in live],
            "seeds": [{"birth": b, "candidate": c, "lat": cands[c][0], "lon": cands[c][1]} for b, c in seeds],
            "claims": [{"birth": b, "candidate": c, "n_obs": 4, "lat": cands[c][0], "lon": cands[c][1]} for b, c in claims]}


def test_axis_distance_measures_to_the_segment_not_the_point():
    assert M.axis_distance(8.0, 47.0, 47.0, [5, 11]) == 0.0
    assert M.axis_distance(10.0, 47.0, 47.0, [5, 11]) == 0.0           # anywhere along the segment
    assert M.axis_distance(8.0, 50.0, 47.0, [5, 11]) == pytest.approx(3.0)
    assert M.axis_distance(14.0, 47.0, 47.0, [5, 11]) == pytest.approx(3.0)   # north of the segment
    assert M.axis_distance(14.0, 51.0, 47.0, [5, 11]) == pytest.approx(5.0)   # the 3-4-5 corner


def test_classify_honors_both_tolerances_inclusively():
    assert M.classify(3.0, 3.0, 5.0) == "within"
    assert M.classify(3.001, 3.0, 5.0) == "ambiguous"
    assert M.classify(5.0, 3.0, 5.0) == "ambiguous"
    assert M.classify(5.001, 3.0, 5.0) == "none"


def test_inspected_dates_are_enumerated_from_the_window_at_twelve_hours():
    assert M.inspected_dates(["1990-09-05T00", "1990-09-06T12"]) == ["1990-09-05T00", "1990-09-05T12", "1990-09-06T00", "1990-09-06T12"]
    assert M.inspected_dates(["1990-09-05T06", "1990-09-05T18"]) == ["1990-09-05T12"]


def test_measure_map_keeps_detection_and_association_apart():
    e = entry("1990-09-08T00", 1000, [(8.0, 47.0, D("r0")), (8.0, 51.0, D("r1")), (30.0, 60.0, D("r2"))],
              claims=[("990:0", 0)], live=[("990:0", 8.0, 48.0), ("980:1", 9.0, 45.0)])
    r = M.measure_map(e, REF["maps"][0], REF, 3.0, 5.0)
    d, a = r["detection"], r["association"]
    assert d["candidates_in_case_box"] == 2 and d["candidates_total"] == 3
    assert [c["index"] for c in d["within"]] == [0] and [c["index"] for c in d["ambiguous"]] == [1]
    assert d["nearest"]["index"] == 0
    assert [o["birth"] for o in a["observations_within"]] == ["990:0"] and a["observations_within"][0]["region_sha256"] == D("r0")
    assert a["observations_ambiguous"] == []
    assert [x["birth"] for x in a["alive_within_without_observation"]] == ["980:1"]   # alive near the axis, no observation this map


def test_sequential_coverage_and_ambiguous_proximity_are_listed_per_history_and_never_called_fragmentation():
    maps = [M.measure_map(entry("1990-09-08T00", 1000, [(8.0, 47.0, D("r0")), (8.0, 51.0, D("r1"))], seeds=[("1000:0", 0), ("1000:1", 1)], live=[("980:0", 9.0, 45.0)]), REF["maps"][0], REF, 3.0, 5.0),
            M.measure_map(entry("1990-09-08T12", 1002, [(8.0, 44.0, D("r2"))], seeds=[("1002:0", 0)]), REF["maps"][1], REF, 3.0, 5.0)]
    s = M.segment_summary(maps, "A", "tolerance")
    assert s["sequential_coverage"] == {"1000:0": ["1990-09-08T00"], "1002:0": ["1990-09-08T12"]}
    assert s["one_history_observes_every_map"] == [] and s["maps_uncovered"] == [] and s["coverage_share_any_history"] == 1.0
    assert s["overlapping_coverage"] == []
    amb = s["ambiguous_proximity"]
    assert amb["observations_between_tolerance_and_alternative"] == {"1000:1": ["1990-09-08T00"]}     # 4 degrees east, an observation
    assert amb["alive_within_tolerance_without_observation"] == {"980:0": ["1990-09-08T00"]}            # alive nearby, no observation
    assert amb["histories_in_either"] == 2 and s["histories_touching_count"] == 4

    def keys(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield k
                yield from keys(v)
        elif isinstance(obj, list):
            for v in obj:
                yield from keys(v)
    assert not any("fragment" in k.lower() for k in keys(s))      # the word appears only in the note's text, never as a measurement


def test_an_overlap_is_a_confirmed_duplicate_only_with_the_same_well_formed_region():
    different = M.measure_map(entry("1990-09-08T00", 1000, [(8.0, 47.0, D("r0")), (9.0, 46.0, D("r1"))], claims=[("990:0", 0), ("992:1", 1)]), REF["maps"][0], REF, 3.0, 5.0)
    s = M.segment_summary([different], "A", "tolerance")
    assert len(s["overlapping_coverage"]) == 1 and s["confirmed_duplicates"] == []
    same = M.measure_map(entry("1990-09-08T00", 1000, [(8.0, 47.0, D("r0"))], claims=[("990:0", 0), ("992:1", 0)]), REF["maps"][0], REF, 3.0, 5.0)
    s = M.segment_summary([same], "A", "tolerance")
    assert len(s["confirmed_duplicates"]) == 1 and s["confirmed_duplicates"][0]["region_sha256"] == D("r0")
    for bad in (None, "", True, "r0", 7):
        malformed = M.measure_map(entry("1990-09-08T00", 1000, [(8.0, 47.0, bad)], claims=[("990:0", 0), ("992:1", 0)]), REF["maps"][0], REF, 3.0, 5.0)
        s = M.segment_summary([malformed], "A", "tolerance")
        assert len(s["overlapping_coverage"]) == 1 and s["confirmed_duplicates"] == [], bad   # no identity evidence, no confirmation


def _join_fixture(a_claims, b_claims, b_cands):
    maps = [M.measure_map(entry("1990-09-08T00", 1000, [(8.0, 47.0, D("a0"))]), REF["maps"][0], REF, 3.0, 5.0),
            M.measure_map(entry("1990-09-08T12", 1002, [(8.0, 44.0, D("a1"))], claims=a_claims), REF["maps"][1], REF, 3.0, 5.0),
            M.measure_map(entry("1990-09-09T00", 1004, b_cands, claims=b_claims), REF["maps"][2], REF, 3.0, 5.0)]
    return maps


def test_join_report_carries_positions_lifecycle_and_the_transition():
    hist = {"x": {"steps": [998, 1002, 1004], "lon_claimed": [46, 44, 40], "lat_claimed": [8, 8, 13]},
            "y": {"steps": [1002], "lon_claimed": [44], "lat_claimed": [8]},
            "z": {"steps": [1004], "lon_claimed": [40], "lat_claimed": [13]}}
    fates = {"x": "finished", "y": "pruned_in_loop", "z": "live"}
    # x continues, y ends, z is a newcomer claiming the same region as x: a split
    maps = _join_fixture(a_claims=[("x", 0), ("y", 0)], b_claims=[("x", 0), ("z", 0)], b_cands=[(13.0, 40.0, D("b0"))])
    j = M.join_report(maps, hist, fates, {"y": {"step": 1005}}, {"x": 7})
    outcomes = {r["birth"]: r["outcome"] for r in j["histories_on_last_A_map"]}
    assert outcomes == {"x": "continues", "y": "ends"}
    row_x = next(r for r in j["histories_on_last_A_map"] if r["birth"] == "x")
    assert row_x["position_on_last_A_map"] == [8.0, 44.0] and row_x["position_on_first_B_map"] == [13.0, 40.0] and row_x["finished_index"] == 7
    assert j["newcomers_on_first_B_map"] == ["z"] and j["splits"] == [{"continuing": "x", "newcomer": "z", "region_sha256": D("b0")}]
    assert j["transition"] == "continues, with a split" and "not a score" in j["note"]
    # nobody continues, a newcomer takes the first B map: replaced
    maps = _join_fixture(a_claims=[("y", 0)], b_claims=[("z", 0)], b_cands=[(13.0, 40.0, D("b0"))])
    j = M.join_report(maps, hist, fates, {}, {})
    assert j["transition"] == "replaced" and [r["outcome"] for r in j["histories_on_last_A_map"]] == ["ends"]
    # a history with later observations but none within tolerance on the first B map
    hist2 = {"x": {"steps": [1002, 1004], "lon_claimed": [44, 30], "lat_claimed": [8, 8]}}
    maps = _join_fixture(a_claims=[("x", 0)], b_claims=[], b_cands=[(13.0, 40.0, D("b0"))])
    j = M.join_report(maps, hist2, {}, {}, {})
    assert j["histories_on_last_A_map"][0]["outcome"] == "observes elsewhere after the last A map"
    assert j["transition"].startswith("ends or observes elsewhere")
    maps = _join_fixture(a_claims=[], b_claims=[], b_cands=[(13.0, 40.0, D("b0"))])
    assert M.join_report(maps, {}, {}, {}, {})["transition"] == "no history within tolerance on either map"


def test_finished_report_keeps_raw_and_smoothed_apart_and_reads_the_crossing_from_raw():
    step_of_date = {m["date"]: 1000 + 2 * i for i, m in enumerate(REF["maps"])}
    near = {"birth": "998:0", "steps": [1000, 1002, 1004], "raw_lat": [8.0, 8.0, 13.0], "raw_lon": [47.0, 44.0, 39.5],
            "lat": [8.0, 8.0, 13.0], "lon": [51.0, 48.0, 43.5]}      # smoothed sits 4 degrees east
    far = {"birth": "900:0", "steps": [1000], "raw_lat": [25.0], "raw_lon": [10.0], "lat": [25.0], "lon": [10.0]}
    out = M.finished_report([far, near], REF, step_of_date, 3.0, 5.0)
    assert [h["finished_index"] for h in out] == [1]
    h = out[0]
    assert [r["raw_class"] for r in h["at_inspected_maps"]] == ["within", "within", "within"]
    assert [r["smoothed_class"] for r in h["at_inspected_maps"]] == ["ambiguous", "ambiguous", "ambiguous"]
    assert h["crosses_40E_raw"] is True
    assert h["coverage"]["A"]["share"] == 1.0 and h["coverage"]["B"]["share"] == 0.25


def _history(steps, lons, lats):
    return {"steps": steps, "lon_claimed": lons, "lat_claimed": lats, "n_points": [10] * len(steps), "region_sha256": [D(f"h{s}") for s in steps]}


def _band_steps(drop=None):
    steps = [entry("1990-09-05T00", 988, [(12.0, 44.0, D("b0")), (30.0, 10.0, D("b9"))], claims=[("980:0", 0)]),
             entry("1990-09-05T06", 989, [(12.0, 44.0, D("b0"))], claims=[("980:0", 0)]),       # a 6-hourly step, not inspected
             entry("1990-09-05T12", 990, [(12.0, 44.0, D("b0"))], claims=[("980:0", 0)]),
             entry("1990-09-06T00", 992, [(12.0, 43.5, D("b1"))], claims=[("980:0", 0), ("990:3", 0)]),
             entry("1990-09-06T12", 994, [(12.0, 43.0, D("b1"))], claims=[("980:0", 0)])]
    return [s for s in steps if s["date"][:13] != drop]


def test_standing_band_report_enumerates_its_maps_counts_candidates_and_categorizes_at_both_parameters():
    hist = {"980:0": _history([980, 988, 989, 990, 992, 994], [45.0, 44.0, 44.0, 44.0, 43.5, 43.0], [12.0] * 6),
            "990:3": _history([990, 992], [37.0, 43.5], [12.0, 12.0])}
    a_entry = entry("1990-09-08T00", 1000, [(8.0, 51.0, D("r0"))], claims=[("980:0", 0)])     # 4 degrees east of the axis
    a_results = [M.measure_map(a_entry, REF["maps"][0], REF, 3.0, 5.0)]
    a_results_alt = [M.measure_map(a_entry, REF["maps"][0], REF, 5.0, 5.0)]
    r = M.standing_band_report(_band_steps(), REF, hist, {"980:0": "pruned_in_loop"}, {"980:0": {"step": 1003}}, {}, a_results, 3.0, a_results_alt, 5.0)
    assert r["inspected_maps"] == ["1990-09-05T00", "1990-09-05T12", "1990-09-06T00", "1990-09-06T12"]
    assert r["candidates_in_band_per_map"] == {"1990-09-05T00": 1, "1990-09-05T12": 1, "1990-09-06T00": 1, "1990-09-06T12": 1}
    p = {h["birth"]: h for h in r["produced_histories"]}
    assert p["980:0"]["maps_observed_in_band"] == r["inspected_maps"]
    assert p["980:0"]["per_step_lon_displacement_deg"] == [-1.0, 0.0, 0.0, -0.5, -0.5] and p["980:0"]["net_lon_displacement_deg"] == -2.0
    assert p["980:0"]["displacement_category_by_stationary_parameter"] == {"3.0": "small_net_displacement", "5.0": "small_net_displacement"}
    assert p["990:3"]["net_lon_displacement_deg"] == 6.5 and p["990:3"]["displacement_category_by_stationary_parameter"] == {"3.0": "east_moving", "5.0": "east_moving"}
    assert all(h["section_6_class"] == "undetermined pending the field reading" for h in r["produced_histories"])
    assert r["displacement_category_counts_by_stationary_parameter"] == {"3.0": {"small_net_displacement": 1, "east_moving": 1}, "5.0": {"small_net_displacement": 1, "east_moving": 1}}
    assert p["980:0"]["continues_into_A_within_tolerance"] is False and p["980:0"]["continues_into_A_within_alternative"] is True
    assert p["980:0"]["fate"] == "pruned_in_loop" and "not labeled false" in r["note"]
    assert M.displacement_category(-14.0, 3.0) == "west_moving"
    assert M.displacement_category(3.0, 3.0) == "small_net_displacement" and M.displacement_category(-3.0, 3.0) == "small_net_displacement"   # inclusive
    assert M.displacement_category(3.001, 3.0) == "east_moving" and M.displacement_category(-3.001, 3.0) == "west_moving"
    r3 = M.standing_band_report(_band_steps(), REF, hist, {}, {}, {}, a_results, 3.0)
    assert r3["produced_histories"][0]["continues_into_A_within_alternative"] is None


def test_standing_band_report_refuses_missing_maps_and_missing_or_malformed_raw_histories():
    hist = {"980:0": _history([988], [44.0], [12.0]), "990:3": _history([992], [43.5], [12.0])}
    with pytest.raises(SystemExit, match="inspected maps .* not among"):
        M.standing_band_report(_band_steps(drop="1990-09-06T00"), REF, hist, {}, {}, {}, [], 3.0)
    with pytest.raises(SystemExit, match="raw history of produced history 990:3"):
        M.standing_band_report(_band_steps(), REF, {"980:0": hist["980:0"]}, {}, {}, {}, [], 3.0)
    truncated = {"980:0": {**hist["980:0"], "lon_claimed": []}, "990:3": hist["990:3"]}
    with pytest.raises(SystemExit, match="empty or not aligned"):
        M.standing_band_report(_band_steps(), REF, truncated, {}, {}, {}, [], 3.0)


def _retained():
    return {"finished": [{"birth": "996:0", "time": [33117.0, 33117.5, 33118.0], "lat": [12.0, 10.7, 8.0], "lon": [44.0, 45.3, 47.0], "steps": [996, 998, 1000],
                          "raw_lat": [12.0, 12.0, 8.0], "raw_lon": [44.0, 45.0, 47.0], "n_points": [10, 10, 10], "region_sha256": [D("c996"), D("c998"), D("c1000")]}],
            "configuration": "recomputed calibration, the domain's own", "preparation": "margin", "margin_edges": "all", "tracker_flags": {"absorb": False, "exclusive": False},
            "year": 1990, "thresholds": {"coarse": 1e-7, "fine": 2e-6}, "runs": {"B": {"case_sha256": "b" * 64}, "wide": {"case_sha256": "w" * 64}}}


def _write_gz(path, obj):
    blob = json.dumps(obj).encode()
    with gzip.open(path, "wb") as fh:
        fh.write(blob)
    return hashlib.sha256(blob).hexdigest()


def _fixture(tmp_path, damage=None):
    """A retained-style reference, trace and finished record, consistent with each other,
    with an optional mutation of the trace before it is written."""
    ref = {"families": [{"name": "R1", "window": ["1990-09-07T00", "1990-09-09T12"], "box": {"lat": [0, 30], "lon": [20, 62]}}],
           "reference": {"case_box": {"lat": [5.0, 20.0], "lon": [20.0, 55.0]}, "segments": {"A": ["1990-09-08T00", "1990-09-08T12"], "B": ["1990-09-09T00", "1990-09-09T12"]},
                         "maps": [{"date": "1990-09-08T00", "segment": "A", "axis_lon": 47.0, "lat_range": [5.0, 11.0], "scored": True},
                                  {"date": "1990-09-08T12", "segment": "A", "axis_lon": 44.0, "lat_range": [5.0, 11.0], "scored": True},
                                  {"date": "1990-09-09T00", "segment": "B", "axis_lon": 40.0, "lat_range": [12.0, 15.0], "scored": True},
                                  {"date": "1990-09-09T12", "segment": "B", "axis_lon": 35.0, "lat_range": [9.0, 15.0], "scored": True}],
                         "standing_band": {"window": ["1990-09-07T00", "1990-09-07T12"], "lon": [40.0, 48.0], "lat": [8.0, 25.0], "scored": False}},
           "tolerances": {"positional_deg": 3.0, "alternative_deg": 5.0}}
    ref_path = tmp_path / "reference.json"
    ref_path.write_text(json.dumps(ref))
    retained = _retained()
    fin_path = tmp_path / "retained.json.gz"
    fin_digest = _write_gz(fin_path, retained)
    steps = [entry("1990-09-07T00", 996, [(12.0, 44.0, D("c996"))], seeds=[("996:0", 0)]),
             entry("1990-09-07T12", 998, [(12.0, 45.0, D("c998"))], claims=[("996:0", 0)], live=[("996:0", 12.0, 44.0)]),
             entry("1990-09-08T00", 1000, [(8.0, 47.0, D("c1000"))], claims=[("996:0", 0)], live=[("996:0", 12.0, 45.0)]),
             entry("1990-09-08T12", 1002, [(21.0, 47.0, D("c1002"))], live=[("996:0", 8.0, 47.0)]),
             entry("1990-09-09T00", 1004, [(15.75, 38.0, D("c1004"))], seeds=[("1003:1", 0)], live=[("996:0", 8.0, 47.0)]),
             entry("1990-09-09T12", 1006, [(20.0, 42.0, D("c1006"))], live=[("1003:1", 15.75, 38.0)])]
    histories = {"996:0": {"steps": [996, 998, 1000], "lon_claimed": [44.0, 45.0, 47.0], "lat_claimed": [12.0, 12.0, 8.0], "n_points": [10, 10, 10],
                           "region_sha256": [D("c996"), D("c998"), D("c1000")]},
                 "1003:1": {"steps": [1004], "lon_claimed": [38.0], "lat_claimed": [15.75], "n_points": [10], "region_sha256": [D("c1004")]}}
    trace = {"trace": {"families_sha256": M.sha256_of(str(ref_path)), "steps": steps, "histories": histories,
                       "fates": {"996:0": "finished", "1003:1": "pruned_in_loop"}, "pruned_at": {"1003:1": {"step": 1007}}, "finished_index": {"996:0": 0}},
             "reproduces": {"finished_identical": True, "sha256": fin_digest}, "finished": json.loads(json.dumps(retained["finished"])),
             **{k: retained[k] for k in M.CONFIGURATION_KEYS}, "thresholds": dict(retained["thresholds"]), "runs": json.loads(json.dumps(retained["runs"])),
             "elapsed_seconds": 1.0, "configuration": retained["configuration"]}
    if damage is not None:
        damage(trace)
    trace_path = tmp_path / "trace.json.gz"
    digest = _write_gz(trace_path, trace)
    out = tmp_path / "measurement.json"
    return ["--reference", str(ref_path), "--trace", str(trace_path), "--finished", str(fin_path), "--out", str(out), "--trace-digest", digest[:12]], out


def test_entry_point_measures_a_consistent_fixture(tmp_path):
    argv, out = _fixture(tmp_path)
    assert M.main(argv) == 0
    rec = json.loads(out.read_text())
    assert rec["input_contract_checked"] is True
    assert rec["summary"]["tolerance"]["segment_coverage"] == {"A": ["1990-09-08T00"], "B": ["1990-09-09T00"]}
    assert rec["summary"]["tolerance"]["join_transition"] == "nothing within tolerance on the last A map, newcomers on the first B map"
    band = rec["standing_band"]
    assert band["inspected_maps"] == ["1990-09-07T00", "1990-09-07T12"] and [p["birth"] for p in band["produced_histories"]] == ["996:0"]
    assert band["produced_histories"][0]["continues_into_A_within_tolerance"] is True
    assert rec["summary"]["standing_band_displacement_categories"] == {"3.0": {"small_net_displacement": 1}, "5.0": {"small_net_displacement": 1}}
    assert rec["finished_stage"]["finished_histories_near_an_inspected_map"][0]["finished_index"] == 0


@pytest.mark.parametrize("label, damage, message", [
    ("binding", lambda t: t["trace"].__setitem__("families_sha256", "0" * 64), "declared reference"),
    ("claims missing from a band map", lambda t: t["trace"]["steps"][0].pop("claims"), "lacks claims"),
    ("seeds missing from an A map", lambda t: t["trace"]["steps"][2].pop("seeds"), "lacks seeds"),
    ("claims not a list", lambda t: t["trace"]["steps"][1].__setitem__("claims", None), "not a list|holds no seed or claim"),
    ("truncated raw history", lambda t: t["trace"]["histories"]["996:0"].__setitem__("lon_claimed", [44.0]), "empty or not aligned"),
    ("raw history disagrees with its bound finished record",
     lambda t: (t["trace"]["histories"]["996:0"]["lon_claimed"].__setitem__(2, 46.0), t["trace"]["steps"][2]["claims"][0].__setitem__("lon", 46.0)),
     "disagrees with its bound finished record"),     # the logged claim agrees with the raw history, only the bound finished record differs
    ("observation disagrees with its raw history", lambda t: t["trace"]["steps"][1]["claims"][0].__setitem__("lon", 45.5), "disagrees with its raw history|holds no seed or claim for it at that position"),
    ("boolean region digest", lambda t: t["trace"]["steps"][2]["candidates"][0].__setitem__("region_sha256", True), "malformed region digest"),
    ("malformed raw history digest", lambda t: t["trace"]["histories"]["1003:1"]["region_sha256"].__setitem__(0, "r0"), "malformed region digest"),
    ("non-finite candidate position", lambda t: t["trace"]["steps"][2]["candidates"][0].__setitem__("lon_mean", float("nan")), "non-finite position"),
    ("finished history without a raw history", lambda t: (t["trace"]["histories"].pop("996:0"), t["trace"]["finished_index"].pop("996:0")), "has no raw history|retained in the trace without|raw history of produced history"),
    ("inspected map missing", lambda t: t["trace"]["steps"].pop(3), "not among the trace's logged steps"),
    ("contradictory empty lists erase retained observations", lambda t: [e.update({"seeds": [], "claims": []}) for e in t["trace"]["steps"][:2]], "holds no seed or claim"),
    ("finished history retained without its finished index", lambda t: t["trace"]["finished_index"].pop("996:0"), "retained in the trace without its finished index"),
    ("candidate digest swapped under an unchanged history", lambda t: t["trace"]["steps"][2]["candidates"][0].__setitem__("region_sha256", D("c998")), "region or point count"),
    ("candidate point count swapped", lambda t: t["trace"]["steps"][2]["candidates"][0].__setitem__("n_points", 11), "region or point count"),
    ("duplicate logged step under another date", lambda t: t["trace"]["steps"].append({**json.loads(json.dumps(t["trace"]["steps"][4])), "date": "1990-09-09T12:00Z"}), "duplicate logged step"),
    ("logged steps out of order", lambda t: t["trace"]["steps"].reverse(), "duplicate logged step|out of order"),
    ("digest with a trailing newline", lambda t: t["trace"]["steps"][2]["candidates"][0].__setitem__("region_sha256", D("c1000") + "\n"), "malformed region digest"),
    ("raw history steps not increasing", lambda t: t["trace"]["histories"].__setitem__("1003:1", {"steps": [1004, 1003], "lon_claimed": [38.0, 37.5], "lat_claimed": [15.75, 16.5], "n_points": [10, 10], "region_sha256": [D("c1004"), D("c1003")]}), "not increasing integers"),
    ("finished record region differs from the raw history", lambda t: t["trace"]["histories"]["996:0"]["region_sha256"].__setitem__(0, D("other")), "holds no seed or claim|disagrees with its raw history|bound finished record"),
])
def test_entry_point_refuses_damaged_evidence(tmp_path, label, damage, message):
    argv, out = _fixture(tmp_path, damage)
    with pytest.raises(SystemExit, match=message):
        M.main(argv)
    assert not out.exists(), label


def test_entry_point_requires_and_binds_the_trace_to_the_replays_logged_digest(tmp_path):
    argv, out = _fixture(tmp_path)
    base = argv[:argv.index("--trace-digest")]
    trace_path = argv[argv.index("--trace") + 1]
    digest = M.sha256_decompressed(trace_path)
    with pytest.raises(SystemExit):                              # the fingerprint is required, not optional
        M.main(base)
    with pytest.raises(SystemExit, match="does not begin with the one the replay logged"):
        M.main(base + ["--trace-digest", "0" * 12])
    with pytest.raises(SystemExit, match="does not begin with the one the replay logged"):
        M.main(base + ["--trace-digest", digest[:8]])          # shorter than the log keeps is not a binding
    assert not out.exists()
    assert M.main(base + ["--trace-digest", digest[:12]]) == 0
    rec = json.loads(out.read_text())
    assert rec["inputs"]["trace"]["bound_to_declared_digest"] is True and rec["inputs"]["trace"]["declared_digest"] == digest[:12]
    assert rec["inputs"]["trace"]["sha256_decompressed"] == digest and "not the correctness" in rec["inputs"]["trace"]["limit"]


def test_check_binding_requires_complete_record_consistency(tmp_path):
    ref = tmp_path / "ref.json"
    ref.write_text("{}")
    rec = _retained()
    path = str(tmp_path / "retained.json.gz")
    digest = _write_gz(path, rec)
    good = {"trace": {"families_sha256": M.sha256_of(str(ref))}, "reproduces": {"finished_identical": True, "sha256": digest},
            "finished": json.loads(json.dumps(rec["finished"])), **{k: rec[k] for k in M.CONFIGURATION_KEYS}, "thresholds": dict(rec["thresholds"]), "runs": json.loads(json.dumps(rec["runs"]))}
    M.check_binding(good, str(ref), rec, path)
    with pytest.raises(SystemExit, match="declared reference"):
        M.check_binding({**good, "trace": {"families_sha256": "0" * 64}}, str(ref), rec, path)
    with pytest.raises(SystemExit, match="did not reproduce"):
        M.check_binding({**good, "reproduces": {"finished_identical": False, "sha256": digest}}, str(ref), rec, path)
    with pytest.raises(SystemExit, match="reproduction digest"):
        M.check_binding({**good, "reproduces": {"finished_identical": True, "sha256": "0" * 64}}, str(ref), rec, path)
    for key in M.FINISHED_KEYS:
        altered = json.loads(json.dumps(good))
        altered["finished"][0][key] = "altered"
        with pytest.raises(SystemExit, match=f"every field \\({key}\\)"):
            M.check_binding(altered, str(ref), rec, path)
    for key in M.CONFIGURATION_KEYS:
        with pytest.raises(SystemExit, match=f"configuration differs .*{key}"):
            M.check_binding({**good, key: "altered"}, str(ref), rec, path)
    with pytest.raises(SystemExit, match="fine threshold"):
        M.check_binding({**good, "thresholds": {"coarse": 1e-7, "fine": 3e-6}}, str(ref), rec, path)
    with pytest.raises(SystemExit, match="wide case differs"):
        M.check_binding({**good, "runs": {"B": {"case_sha256": "b" * 64}, "wide": {"case_sha256": "x" * 64}}}, str(ref), rec, path)
    with pytest.raises(SystemExit, match="not .*retained.json.gz's$"):
        M.check_binding({**good, "finished": []}, str(ref), rec, path)
