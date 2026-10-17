"""One usable reference per scored map from the owning sequence: two answers, the other
sequence's rows preserved and unused, indeterminate, none and missing kept distinct, and
the output read by the measurement's alternative-reading mode unchanged."""
import importlib.util
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


U = load("reading_usable_under_test", "pilot_reading_usable.py")
E = load("eastern_measurement_for_usable_test", "pilot_eastern_measurement.py")

G = lambda seq, lon, status="identified": {"sequence": seq, "status": status, "axis_lon": lon, "lat_range": [10.0, 20.0], "signature": "x"}
PARSED = {"source": "synthetic", "label": "l", "sequences": {"A": ["d1", "d2", "d3"], "B": ["d3", "d4", "d5", "d6"], "C": ["d4", "d5", "d6"]}, "form": {},
          "maps": {"d1": [G("A", 40.0)],
                   "d2": [G("A", 22.0), G("A", 58.0)],
                   "d3": [G("A", 30.0, "weak"), G("B", 50.0)],
                   "d4": [{"sequence": "B", "status": "indeterminate", "signature": "diffuse"}, G("C", 45.0)],
                   "d5": [{"sequence": "B", "status": "none", "signature": ""}, G("C", 44.0)],
                   "d6": [G("C", 42.0)]}}
DECL = {"reference": {"maps": [{"date": "d1", "segment": "S1", "scored": True, "axis_lon": 48.0, "lat_range": [10.0, 20.0]},
                               {"date": "d2", "segment": "S1", "scored": True, "axis_lon": 48.0, "lat_range": [10.0, 20.0]},
                               {"date": "d3", "segment": "S1", "scored": False, "axis_lon": 47.0, "lat_range": [10.0, 20.0]},
                               {"date": "d4", "segment": "S2", "scored": True, "axis_lon": 57.0, "lat_range": [15.0, 28.0]},
                               {"date": "d5", "segment": "S2", "scored": True, "axis_lon": 54.0, "lat_range": [15.0, 28.0]},
                               {"date": "d6", "segment": "S2", "scored": True, "axis_lon": 51.0, "lat_range": [15.0, 28.0]}],
                      "transitions": [], "segments": {"S1": {"scored": True}, "S2": {"scored": True}}},
        "tolerances": {"positional_deg": 3.0, "alternative_deg": 5.0}}
OWNERS = {"S1": "A", "S2": "B"}


def test_one_usable_reference_from_the_owning_sequence_with_the_rest_kept_distinct():
    maps, other, table = U.build(PARSED, DECL, OWNERS)
    assert maps["d1"] == [G("A", 40.0)]
    assert maps["d2"][0]["status"] == "two answers" and [a["axis_lon"] for a in maps["d2"][0]["answers"]] == [22.0, 58.0] and "axis_lon" not in maps["d2"][0]
    assert maps["d3"] == [G("A", 30.0, "weak")] and other["d3"] == [G("B", 50.0)], "the other sequence's row is preserved and not used"
    assert maps["d4"][0]["status"] == "indeterminate" and other["d4"] == [G("C", 45.0)]
    assert maps["d5"][0]["status"] == "none" and other["d5"] == [G("C", 44.0)]
    assert maps["d6"] == [{"status": "missing", "sequence": "B"}] and other["d6"] == [G("C", 42.0)]
    assert table["S1"]["usable"] == ["d1"] and table["S1"]["two answers"] == ["d2"] and table["S1"]["counts"]["missing"] == 0, "the unscored d3 is not counted"
    assert table["S2"]["counts"] == {"usable": 0, "two answers": 0, "indeterminate": 1, "none": 1, "missing": 1}


def test_the_measurement_reads_the_usable_file_and_names_every_map_without_a_reference(tmp_path):
    maps, other, table = U.build(PARSED, DECL, OWNERS)
    p = tmp_path / "usable.json"
    p.write_text(json.dumps({"source": "synthetic", "maps": maps, "usable": table}))
    ref, block = E.apply_alternative_reading(DECL["reference"], str(p))
    assert [m["date"] for m in ref["maps"]] == ["d1", "d3"] and ref["maps"][0]["axis_lon"] == 40.0 and len(ref["maps"][0]["geometries"]) == 1
    assert {u["date"]: u["status"] for u in block["maps_unscored"]} == {"d2": ["two answers"], "d4": ["indeterminate"], "d5": ["none"], "d6": ["missing"]}


def test_a_segment_without_an_owner_is_refused():
    import pytest
    with pytest.raises(SystemExit, match="no owning sequence"):
        U.build(PARSED, DECL, {"S1": "A"})
