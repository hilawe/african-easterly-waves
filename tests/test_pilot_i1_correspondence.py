"""The correspondence reader's helpers: nearest integer row with both rows at a tie,
counterpart precedence (reversal before the vorticity maximum), the on, near and off
classes, hits on another structure, and the lineage summary, which must expose a retained
parent outside the box that shares a selected-region digest with an input inside it. The
history-selection and verdict branches of `main` are not exercised here."""
import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def _load():
    spec = importlib.util.spec_from_file_location("correspondence_under_test", os.path.join(ROOT, "scripts", "pilot_i1_correspondence.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_rows_classes_and_counterpart_precedence():
    C = _load()
    assert C.rows_of(24.0) == [24] and C.rows_of(24.4) == [24] and C.rows_of(24.5) == [24, 25]
    assert C.classify(2.9) == "on" and C.classify(-4.9) == "near" and C.classify(5.1) == "off" and C.classify(None) == "unassessable"
    row = {"status": "assigned", "reversal": {"lon": 50.0}, "zs": {"lon": 47.0}, "wind_minima": [{"lon": 43.0, "label": "W"}, {"lon": 60.0, "label": "E"}]}
    assert C.counterpart(row) == (50.0, "reversal")
    row2 = {"status": "assigned", "reversal": {"lon": None}, "zs": {"lon": 47.0}, "wind_minima": []}
    assert C.counterpart(row2)[0] == 47.0 and "vorticity" in C.counterpart(row2)[1]
    assert C.counterpart({"status": "abstained"})[0] is None
    assert C.other_structure(row, 44.0) == ["W 43E"] and C.other_structure(row, 52.0) == []


def _entry(i, lat, lon, digest, output):
    return {"input": i, "lat_mean": lat, "lon_mean": lon, "seed": {"lat": lat, "lon": lon}, "kept_after_pass_two": output is not None, "merged": i, "output": output,
            "rejected_min_extent": False, "select": {"selected_cells": 20, "selected_lat_extent": 8, "selected_lon_extent": 10, "level_landed": 2, "exhausted": False,
                                                    "region": {"cells": 20, "region_sha256": digest, "median_lat": 22.0, "median_lon": 51.0}}}


def test_lineage_summary_exposes_a_retained_parent_outside_the_box_that_shares_the_region():
    C = _load()
    L = {"axes": [{"index": 1, "lat": [24.0, 26.0], "lon": [49.0, 50.0], "lat_mean": 25.0, "lon_mean": 49.5, "n_vertices": 2},
                  {"index": 2, "lat": [5.0], "lon": [10.0], "lat_mean": 5.0, "lon_mean": 10.0, "n_vertices": 1}],
         "coarse": [_entry(11, 24.7, 50.0, "abc", None), _entry(6, 31.4, 46.3, "abc", 4), _entry(40, 5.0, 10.0, "zzz", 9)],
         "fine": []}
    out = C.lineage_in_box(L, {"lat": [10, 30], "lon": [38, 65]})
    assert [a["index"] for a in out["axes"]] == [1]
    by = {e["input"]: e for e in out["coarse"]}
    assert set(by) == {11, 6}, "the inside input and its parent outside the box are listed, the unrelated one is not"
    assert by[6]["parent_outside_box"] is True and by[6]["inside_box"] is False and by[6]["output"] == 4
    assert by[11]["shares_region_with"] == [6] and by[11]["retained_input_of_group"] == [6] and by[11]["pass_one_index"] == 11
    assert "region" not in by[11] and by[11]["region_median"] == [22.0, 51.0]
