"""The contour-divergence classifier on synthetic fields: the crossing classes as local
vertex properties, candidate selection as a heuristic apart from geometric identity,
overlap as an audited count, the print tolerance, and the field gate."""
import importlib.util
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
pytest.importorskip("scipy")


def _load():
    spec = importlib.util.spec_from_file_location(
        "characterize_contour_divergence", os.path.join(ROOT, "scripts", "characterize_contour_divergence.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _block(grid, rows=(0.0, 2.5, 5.0, 7.5), cols=(0.0, 2.5, 5.0, 7.5)):
    return {"rows_lat": list(rows), "cols_lon": list(cols), "masked_smoothed_advection": grid, "grid": grid}


def test_closed_open_and_off_are_local_vertex_properties():
    C = _load()
    from trace_sahara_case import crossing_neighborhoods, zero_crossings
    block = _block([[1.0, 1.0, -1.0, -1.0] for _ in range(4)])
    cr = zero_crossings(block)
    nb = crossing_neighborhoods(block["rows_lat"], block["cols_lon"], block["grid"], cr, domain_complete=True)
    axis = {"lat": [2.5, 5.0], "lon": [3.75, 3.75], "centroid": (3.75, 3.75), "points": 2}
    klass, detail = C.classify(axis, block, cr, nb)
    assert klass == "OPEN CROSSING" and detail["open"] == 2
    block = _block([[None] * 4, [1.0, 1.0, -1.0, -1.0], [None] * 4, [None] * 4])
    cr = zero_crossings(block)
    nb = crossing_neighborhoods(block["rows_lat"], block["cols_lon"], block["grid"], cr, domain_complete=True)
    klass, detail = C.classify({"lat": [2.5], "lon": [3.75]}, block, cr, nb)
    assert klass == "CLOSED CROSSING" and detail["closed"] == 1
    klass, detail = C.classify({"lat": [1.0, 1.2], "lon": [0.3, 0.4]}, block, cr, nb)
    assert klass == "OFF CROSSING" and detail["off"] == 2


def test_centroid_selection_is_not_geometric_identity():
    """A review's two probes: the same segment sampled at more points is centroid-unmatched,
    and two perpendicular segments through one point are centroid-matched. The distinct
    vertex test tells both apart, and the record labels the selection a heuristic."""
    C = _load()
    same_line_v1 = [(np.array([0.0, 10.0]), np.array([0.0, 0.0]))]
    same_line_port = [(np.array([0.0, 1.0, 2.0, 10.0]), np.array([0.0, 0.0, 0.0, 0.0]))]
    out = C.unmatched_axes(same_line_v1, same_line_port)
    assert len(out) == 1 and out[0]["geometrically_identical_to_a_port_axis"] is False
    perpendicular_v1 = [(np.array([-2.0, 2.0]), np.array([0.0, 0.0]))]
    perpendicular_port = [(np.array([0.0, 0.0]), np.array([-2.0, 2.0]))]
    assert C.unmatched_axes(perpendicular_v1, perpendicular_port) == []      # the heuristic's known blindness
    assert C.geometric_match({"lat": [-2.0, 2.0], "lon": [0.0, 0.0]}, perpendicular_port) is None
    # geometric identity ignores order and repeated endpoints, which version 1 writes
    assert C.geometric_match({"lat": [10.0, 10.0, 0.0], "lon": [0.0, 0.0, 0.0]}, same_line_v1) == 0
    assert C.distinct_vertices([10.0, 10.0, 0.0], [0.0, 0.0, 0.0]) == ([(10.0, 0.0), (0.0, 0.0)], 1)
    # one to one: two vertices a hair apart on one side and one vertex plus a far one on the
    # other have equal counts and must not be called identical through a many-to-one match
    mine = {"lat": [0.0, 0.0001], "lon": [0.0, 0.0]}        # two distinct vertices, 1e-4 apart
    theirs = [(np.array([0.00004, 5.0]), np.array([0.0, 0.0]))]   # one within tolerance of BOTH, one far
    assert C.distinct_vertices(mine["lat"], mine["lon"])[0] == [(0.0, 0.0), (0.0001, 0.0)]
    assert C.geometric_match(mine, theirs) is None
    assert C.geometric_match({"lat": [0.00004, 5.0], "lon": [0.0, 0.0]}, [(np.array(mine["lat"]), np.array(mine["lon"]))]) is None


def test_overlap_is_counted_at_the_print_tolerance_with_its_carriers():
    C = _load()
    axis = {"lat": [2.5, 2.5, 5.0], "lon": [3.75, 3.75, 3.75]}
    port = [(np.array([2.5, 9.0]), np.array([3.75049, 9.0])), (np.array([2.5]), np.array([3.75051]))]
    o = C.vertex_overlap(axis, port)
    # 3.75049 is 4.9e-4 away, beyond the print tolerance; nothing is shared
    assert o["overlap"] == "none" and o["shared"] == 0 and o["distinct_vertices"] == 2 and o["repeated_vertices"] == 1
    port = [(np.array([2.5, 9.0]), np.array([3.75004, 9.0])), (np.array([5.0]), np.array([3.75]))]
    o = C.vertex_overlap(axis, port)
    assert o["overlap"] == "complete" and o["shared"] == 2 and o["port_axes_carrying_each_vertex"] == [[0], [1]]
    assert o["port_axes_involved"] == [0, 1]
    o = C.vertex_overlap(axis, [(np.array([5.0]), np.array([3.75]))])
    assert o["overlap"] == "partial" and o["shared"] == 1


def test_the_primary_reading_is_gated_on_the_field_comparison(monkeypatch, tmp_path):
    """A field that differs gives FIELD, and no comparison gives UNDETERMINED, whatever the
    candidate's crossings say."""
    C = _load()
    grid = [[1.0, 1.0, -1.0, -1.0] for _ in range(4)]
    case = {"lat_c": np.array([0.0, 2.5, 5.0, 7.5]), "lon_c": np.array([0.0, 2.5, 5.0, 7.5]), "time": np.array([1.0])}
    monkeypatch.setattr(C, "port_field_at", lambda case, step: np.array(grid))
    monkeypatch.setattr(C.B, "axes_at", lambda case, step: [])
    v1 = [(np.array([2.5, 5.0]), np.array([3.75, 3.75]))]
    monkeypatch.setattr(C.S, "ordered_capture", lambda text, times: ({"1.0000": v1}, {}, {}))
    def fields(text, step):
        cells = {(round(la, 4), round(lo, 4)): float(grid[i][j]) for i, la in enumerate(case["lat_c"]) for j, lo in enumerate(case["lon_c"])}
        if text == "differ":
            cells[(0.0, 0.0)] = 2.0
        return {"cells": {} if text == "absent" else cells, "records": len(cells), "problems": []}
    monkeypatch.setattr(C, "reference_field_at_text", fields)
    same = C.read_step(case, "same", 1.0, (3.75, 3.75))
    assert same["primary"] == "fields agree within tolerance" and same["candidate"]["class"] == "OPEN CROSSING"
    assert "1e-09" in same["fields_agreement"]
    differ = C.read_step(case, "differ", 1.0, (3.75, 3.75))
    assert differ["primary"].startswith("FIELD") and differ["candidate"]["class"].startswith("FIELD")
    assert differ["fields_agreement"].startswith("masks differ or values beyond")
    absent = C.read_step(case, "absent", 1.0, (3.75, 3.75))
    assert absent["primary"].startswith("UNDETERMINED") and absent["candidate"]["class"].startswith("UNDETERMINED")
    assert absent["fields_agreement"] == "not compared"
