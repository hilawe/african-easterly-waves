"""The eastern descriptive layer's builder (`EASTERN_DESCRIPTIVE_LAYER_2026-10-17.md`,
sections 4 and 6): one table row per listed row and step, the frozen reader's pair class
and motion label joined onto the right row-step, steps outside chains left empty, the
listing's minima and maxima flattened as declared, and a reader output whose recorded
listing digest does not match the listing refused."""
import hashlib
import importlib.util
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def _load():
    spec = importlib.util.spec_from_file_location("descriptive_layer_builder_under_test", os.path.join(ROOT, "scripts", "build_eastern_descriptive_layer.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write(tmp_path):
    ann = {"i1": {"lon": [44, 63], "rows": [12, 30]}}
    def row(lat, sel, mins, status="assigned", maxima=()):
        m = {"row_lat": lat, "status": status, "wind_minima": [{"lon": a, "v": b, "label": c} for a, b, c in mins],
             "zeta_maxima": [{"lon": a, "zeta_1e5": b, "label": "other"} for a, b in maxima], "i1": None, "i1_alternatives": [],
             "edge": {"primary": {"lon": None}, "sensitivity": {"lon": None}}, "reversal": {"lon": None, "status": "none within the box"}, "za": None, "zs": None}
        if status == "assigned":
            m["i1"] = {"lon": sel, "v": -10.0}; m["i1_alternatives"] = [{"lon": a, "v": b} for a, b, c in mins if c == "I1" and a != sel]
            m["edge"] = {"primary": {"lon": sel}, "sensitivity": {"lon": sel - 1}}; m["reversal"] = {"lon": sel + 6.25, "status": "available"}
        return m
    steps = [{"time": "2000-07-01T00", "annotation": ann, "rows": [row(20, 52.0, [(41.0, -9.0, "W"), (52.0, -10.0, "I1"), (55.0, -8.5, "I1")], maxima=[(53.5, 3.0)]), row(19, None, [], status="abstained")]},
             {"time": "2000-07-01T06", "annotation": ann, "rows": [row(20, 51.0, [(51.0, -10.0, "I1")]), row(19, 50.0, [(50.0, -9.0, "I1")])]},
             {"time": "2000-07-01T12", "annotation": ann, "rows": [row(20, 50.0, [(50.0, -10.0, "I1")]), row(19, 58.0, [(58.0, -9.0, "I1")])]}]
    listing = {"generated_by": "x", "script_sha256": "l" * 64, "year": 2000, "level_hpa": 700, "declared": {"grid": "native-lon", "lon_spacing_deg": 0.5, "rows": [20, 19], "every_hours": 6}, "steps": steps}
    lp = tmp_path / "listing.json"; lp.write_text(json.dumps(listing))
    sha = hashlib.sha256(lp.read_bytes()).hexdigest()
    res = {"params": {"depth_ms": -8.0, "pair_bound_deg": 1.5, "margin_deg": 2.0, "lag_steps": 8, "rate_deg_per_day": 1.0, "min_standing_hours": 24, "long_phase_hours": 48, "overlap_hours": 24, "adjacent_rows": 4, "step_hours": 6},
           "pairs": {"20": [{"row": 20, "k": 0, "from": "2000-07-01T00", "to": "2000-07-01T06", "lon_from": 52.0, "lon_to": 51.0, "move": -1.0, "class": "continuous_unique"},
                            {"row": 20, "k": 1, "from": "2000-07-01T06", "to": "2000-07-01T12", "lon_from": 51.0, "lon_to": 50.0, "move": -1.0, "class": "continuous_unique"}],
                     "19": [{"row": 19, "k": 0, "from": "2000-07-01T00", "to": "2000-07-01T06", "lon_from": None, "lon_to": None, "move": None, "class": "unassessable"},
                            {"row": 19, "k": 1, "from": "2000-07-01T06", "to": "2000-07-01T12", "lon_from": 50.0, "lon_to": 58.0, "move": 8.0, "class": "jump"}]},
           "chains": {"20": [{"row": 20, "i_start": 0, "i_end": 2, "start": "2000-07-01T00", "end": "2000-07-01T12", "pairs": 2, "lons": [52.0, 51.0, 50.0], "classes": ["continuous_unique", "continuous_unique"]}], "19": []},
           "phases": {"20": [{"row": 20, "direction": "west", "i_start": 0, "i_end": 2, "start": "2000-07-01T00", "end": "2000-07-01T12", "hours": 12, "n_steps": 3, "lon_start": 52.0, "lon_end": 50.0, "lon_min": 50.0, "lon_max": 52.0, "net_move": -2.0, "rate_deg_per_day": -4.0, "ambiguous_pairs": 0}], "19": []},
           "standing": {"20": [], "19": []}, "motion": {"20": [[0, "west"], [1, "west"], [2, "west"]], "19": []},
           "groups": {}, "displacement_set": {"rows": [], "n_rows": 0, "phases": [], "meets_criterion": False}}
    phases = {"generated_by": "y", "script_sha256": "r" * 64, "record": {"path": str(lp), "sha256": sha}, "results": {"primary": res}}
    pp = tmp_path / "phases.json"; pp.write_text(json.dumps(phases))
    return lp, pp


def test_the_table_joins_the_reader_onto_the_right_row_step_and_flattens_the_listing(tmp_path):
    B = _load()
    lp, pp = _write(tmp_path)
    rows = B.build_table(str(lp), str(pp), "synthetic")
    assert len(rows) == 6
    by = {(r["row_lat"], r["time"]): r for r in rows}
    r = by[(20, "2000-07-01T00")]
    assert r["selected_lon"] == 52.0 and r["alternatives"] == "55:-8.5" and r["all_minima"] == "41:-9:W;52:-10:I1;55:-8.5:I1"
    assert r["zeta_maxima"] == "53.5:3" and r["window_lon_lo"] == 44 and r["window_lon_hi"] == 63 and r["supported_primary"] is True
    assert r["pair_class_to_next"] == "continuous_unique" and r["pair_move_to_next"] == -1.0 and r["motion_label"] == "west" and r["chain_index"] == 0 and r["phase_index"] == 0 and r["phase_direction"] == "west"
    last = by[(20, "2000-07-01T12")]
    assert last["pair_class_to_next"] == "" and last["pair_move_to_next"] == "" and last["motion_label"] == "west"
    a = by[(19, "2000-07-01T00")]
    assert a["status"] == "abstained" and a["selected_lon"] == "" and a["supported_primary"] is False and a["pair_class_to_next"] == "unassessable" and a["motion_label"] == "" and a["chain_index"] == ""
    j = by[(19, "2000-07-01T06")]
    assert j["pair_class_to_next"] == "jump" and j["pair_move_to_next"] == 8.0 and j["motion_label"] == ""
    m = B.manifest_entry(str(lp), str(pp), "synthetic", "a role", rows)
    assert m["row_steps"] == 6 and m["by_status"]["assigned"] == 5 and m["by_status"]["abstained"] == 1 and m["supported_primary"] == 5
    assert m["pairs_by_class"]["continuous_unique"] == 2 and m["pairs_by_class"]["jump"] == 1 and m["pairs_by_class"]["unassessable"] == 1
    assert m["chains"] == 1 and m["phases"] == 1 and m["standing_intervals"] == 0 and m["window_bounds_by_step"][0] == {"time": "2000-07-01T00", "lon_lo": 44, "lon_hi": 63}


def test_a_reader_output_whose_listing_digest_does_not_match_is_refused(tmp_path):
    import pytest
    B = _load()
    lp, pp = _write(tmp_path)
    lp.write_text(lp.read_text() + "\n")
    with pytest.raises(SystemExit):
        B.build_table(str(lp), str(pp), "synthetic")
