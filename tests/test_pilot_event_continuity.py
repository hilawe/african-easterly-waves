"""The continuity assessment's contracts. The staged recomputation gives the same final
candidates as the replay's detection from prepared fields, the trace gate tells a
reproduced candidate population from a changed one, the hypothesis path is a start
position moved at a declared speed, and a strip reading reports the maximum, its
position and the finite count or none."""
import importlib.util
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def _load(name, folder="scripts"):
    spec = importlib.util.spec_from_file_location(name + "_under_test", os.path.join(ROOT, folder, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_staged_recomputation_reproduces_the_replays_detection_and_keeps_its_intermediates():
    pytest.importorskip("scipy")
    C = _load("pilot_event_continuity")
    T = _load("test_pilot_alledge_replay", "tests")
    A = C.A
    case_b, wide = T.analytic_case(35.0, -140.0, 39.0), T.analytic_case(39.0, -144.0, 43.0)
    grids, sel = A.MR.control_grids(case_b), A.selection(case_b, wide)
    fields = A.prepared_fields(wide, 3, *sel)
    t = float(np.asarray(case_b["time"], float).ravel()[3])
    st = C.stages_from_prepared(fields, t, 4.0e-7, 2.5e-6, grids)
    check = A.detect_from_prepared(fields, t, 4.0e-7, 2.5e-6, grids)
    assert C.signature_of(st["final"]) == C.signature_of(check) and st["final"]
    assert st["axes"] and st["axis_candidates"] and st["coarse"] and st["westerly"].shape == st["curvature_masked"].shape
    assert np.isnan(st["curvature_masked"][st["westerly"]]).all()                                     # westerly cells are blank after the mask


def test_the_trace_gate_tells_a_reproduced_population_from_a_changed_one_and_the_path_moves_at_the_declared_speed():
    C = _load("pilot_event_continuity")
    entry = {"candidates": [{"lat_mean": 14.5, "lon_mean": 27.0, "n_points": 90, "region_sha256": "a" * 64}, {"lat_mean": 8.0, "lon_mean": 11.25, "n_points": 20, "region_sha256": "b" * 64}]}
    same = C.trace_signature(entry)
    assert same == [(14.5, 27.0, 90, "a" * 64), (8.0, 11.25, 20, "b" * 64)]
    changed = {"candidates": [dict(entry["candidates"][0], n_points=91), entry["candidates"][1]]}
    assert C.trace_signature(changed) != same
    hyp = {"lat": 14.5, "lon": 27.0, "t0": 100.0, "deg_per_day": -5.0}
    assert C.hypothesis_position(hyp, 100.0) == (14.5, 27.0) and C.hypothesis_position(hyp, 101.0) == (14.5, 22.0) and C.hypothesis_position(hyp, 100.25) == (14.5, 25.75)


def test_a_strip_reading_reports_the_maximum_and_its_position_or_none():
    C = _load("pilot_event_continuity")
    lat = np.arange(5.0, 26.0, 1.0); lon = np.arange(10.0, 36.0, 1.0)
    latgrid, longrid = np.meshgrid(lat, lon, indexing="ij")
    field = np.full(latgrid.shape, np.nan)
    field[(latgrid == 14.0) & (longrid == 24.0)] = 3.0e-7
    field[(latgrid == 12.0) & (longrid == 30.0)] = 9.0e-7                                             # outside the strip, must not win
    r = C.strip_reading(field, latgrid, longrid, {"lat": [5.0, 25.0], "lon": [10.0, 35.0]}, [20.0, 28.0], [9.5, 19.5])
    assert r == {"finite_cells": 1, "max": 3.0e-7, "max_lat": 14.0, "max_lon": 24.0}
    empty = C.strip_reading(np.full(latgrid.shape, np.nan), latgrid, longrid, {"lat": [5.0, 25.0], "lon": [10.0, 35.0]}, [20.0, 28.0], [9.5, 19.5])
    assert empty == {"finite_cells": 0, "max": None, "max_lat": None, "max_lon": None}
    assert C.near([(14.0, 24.0), (14.0, 40.0)], 14.5, 27.0, 10.0) == [{"lat": 14.0, "lon": 24.0, "distance_deg": 3.041}]


def test_the_holder_of_a_cell_is_the_merged_candidate_whose_region_contains_it_or_none():
    pytest.importorskip("scipy")
    C = _load("pilot_event_continuity")
    T = _load("test_pilot_alledge_replay", "tests")
    A = C.A
    case_b, wide = T.analytic_case(35.0, -140.0, 39.0), T.analytic_case(39.0, -144.0, 43.0)
    grids, sel = A.MR.control_grids(case_b), A.selection(case_b, wide)
    t = float(np.asarray(case_b["time"], float).ravel()[3])
    st = C.stages_from_prepared(A.prepared_fields(wide, 3, *sel), t, 4.0e-7, 2.5e-6, grids)
    latgrid, longrid = grids["latgrid_c"], grids["longrid_c"]
    region = np.asarray(st["coarse"][0]["region"], bool)
    r, c = np.argwhere(region)[0]
    h = C.holder_of(st["coarse"], latgrid, longrid, float(latgrid[r, c]), float(longrid[r, c]))
    assert h is not None and h["n_points"] == int(region.sum()) and h["cell"] == {"lat": float(latgrid[r, c]), "lon": float(longrid[r, c])}
    assert h["region_sha256"] == C.R.region_digest(region)
    outside = np.argwhere(~np.any([np.asarray(w["region"], bool) for w in st["coarse"]], axis=0))[0]
    assert C.holder_of(st["coarse"], latgrid, longrid, float(latgrid[outside[0], outside[1]]), float(longrid[outside[0], outside[1]])) is None


def _ladder_grid():
    lat = np.arange(0.0, 41.0, 1.0); lon = np.arange(0.0, 61.0, 1.0)
    return np.meshgrid(lat, lon, indexing="ij")


def test_the_ladder_detail_reports_each_level_the_level_landed_on_and_exhaustion_of_the_selected_region():
    pytest.importorskip("scipy")
    C = _load("pilot_event_continuity")
    from aew.v1port.contours import _binary_masks, _select_region
    ct = 1.0e-7
    latgrid, longrid = _ladder_grid()
    # S1: a band 11 rows tall whose longitude extent is 30 at the base level, 26 at 1.5 times, 25
    # at 2 times and 24 from 2.5 times on, so the cascade steps three times and lands on the
    # fourth level, which is under the trigger: not exhausted.
    field = np.full(latgrid.shape, np.nan)
    rows = (latgrid >= 10.0) & (latgrid <= 20.0)
    field[rows & (longrid <= 24.0)] = 4.0 * ct
    field[rows & (longrid == 25.0)] = 2.2 * ct
    field[rows & (longrid == 26.0)] = 1.7 * ct
    field[rows & (longrid >= 27.0) & (longrid <= 30.0)] = 1.2 * ct
    d = C.ladder_detail(field, ct, 15.0, 12.0, latgrid, longrid)
    assert d["seed_cell"] == {"lat": 15.0, "lon": 12.0}
    assert len(d["levels"]) == 6 and d["levels"][-1]["multiple"] == 3.5
    assert [l["cells"] for l in d["levels"]] == [341, 297, 286, 275, 275, 275]
    assert [l["lon_extent"] for l in d["levels"]] == [30.0, 26.0, 25.0, 24.0, 24.0, 24.0]
    assert d["level_landed"] == 3 and d["last_level"] == 5 and d["cascade_reached_last_level"] is False
    assert d["selected"] == {"cells": 275, "lat_extent": 10.0, "lon_extent": 24.0,
                             "region_sha256": C.R.region_digest(_select_region(_binary_masks(field, ct), (15, 12), latgrid, longrid))}
    assert d["exhausted"] is False                     # the base level spans 30, the selected one 24
    # S2: the same band at 4 times the base everywhere with a longitude extent of exactly 25, so
    # every level triggers, the cascade reaches the last level, and the retained region still
    # reaches the trigger: exhausted.
    field = np.full(latgrid.shape, np.nan)
    field[rows & (longrid <= 25.0)] = 4.0 * ct
    d = C.ladder_detail(field, ct, 15.0, 12.0, latgrid, longrid)
    assert [l["cells"] for l in d["levels"]] == [286] * 6 and d["level_landed"] == 5 and d["cascade_reached_last_level"] is True
    assert d["selected"]["lon_extent"] == 25.0 and d["extent_trigger_deg"] == {"lat": 25.0, "lon": 25.0}
    assert d["exhausted"] is True
    # S3: two above-threshold cells tied in distance to the position on separate components; the
    # seed is the column-first one, the cell with the smaller longitude index.
    field = np.full(latgrid.shape, np.nan)
    field[(latgrid == 9.0) & (longrid == 11.0)] = 2.0 * ct
    field[(latgrid == 11.0) & (longrid == 9.0)] = 2.0 * ct
    d = C.ladder_detail(field, ct, 10.0, 10.0, latgrid, longrid)
    assert d["seed_cell"] == {"lat": 11.0, "lon": 9.0} and d["selected"]["cells"] == 1 and d["exhausted"] is False
    assert C.ladder_detail(np.full(latgrid.shape, np.nan), ct, 10.0, 10.0, latgrid, longrid) is None


def test_an_axis_detail_counts_vertices_spans_and_vertices_on_above_threshold_cells():
    C = _load("pilot_event_continuity")
    lat = np.arange(5.0, 26.0, 1.0); lon = np.arange(10.0, 36.0, 1.0)
    latgrid, longrid = np.meshgrid(lat, lon, indexing="ij")
    masked = np.full(latgrid.shape, np.nan)
    masked[(latgrid >= 12.0) & (latgrid <= 16.0) & (longrid >= 20.0) & (longrid <= 26.0)] = 1.0e-6
    d = C.axis_detail([12.0, 14.0, 16.0, 18.0], [21.0, 23.0, 25.0, 27.0], masked, latgrid, longrid)
    assert d == {"n_vertices": 4, "lat_span": [12.0, 18.0], "lon_span": [21.0, 27.0], "vertices_on_above_threshold_cells": 3}


def test_the_step_figure_s_profile_panels_sit_exactly_under_their_maps_and_a_misaligned_layout_is_refused(tmp_path):
    pytest.importorskip("scipy"); pytest.importorskip("matplotlib")
    C = _load("pilot_event_continuity")
    T = _load("test_pilot_alledge_replay", "tests")
    A = C.A
    case_b, wide = T.analytic_case(35.0, -140.0, 39.0), T.analytic_case(39.0, -144.0, 43.0)
    grids, sel = A.MR.control_grids(case_b), A.selection(case_b, wide)
    t = float(np.asarray(case_b["time"], float).ravel()[3])
    fields = A.prepared_fields(wide, 3, *sel)
    st = C.stages_from_prepared(fields, t, 4.0e-7, 2.5e-6, grids)
    latgrid_f, longrid_f = grids["latgrid_f"], grids["longrid_f"]
    box = {"lat": [float(latgrid_f.min()), float(latgrid_f.max())], "lon": [float(longrid_f.min()), float(longrid_f.max())]}
    lon0, lon1 = box["lon"]
    strip, band = [lon0 + 1.0, lon0 + 3.0], [box["lat"][0] + 1.0, box["lat"][0] + 3.0]
    hyp = {"lat": band[0] + 1.0, "lon": lon0 + 2.0, "t0": t, "deg_per_day": -5.0}
    reading = C.read_step(st, grids, {"latgrid_f": latgrid_f, "longrid_f": longrid_f}, box, strip, band, hyp, t, 4.0e-7, 2.5e-6, 10.0)
    case_fields = {"u": np.asarray(wide["u"][3], float)[sel[2], :][:, sel[3]], "v": np.asarray(wide["v"][3], float)[sel[2], :][:, sel[3]], "latgrid": latgrid_f, "longrid": longrid_f}
    lons = np.linspace(lon0, lon1, 25)
    out = tmp_path / "step.png"
    alignment = C.draw_step(st, case_fields, grids, box, strip, band, hyp, t, reading, np.sin(lons), None, lons, str(out), "layout test")
    assert out.exists() and alignment["aligned"] is True
    for c in alignment["columns"]:
        assert c["map_x0"] == c["profile_x0"] and c["map_x1"] == c["profile_x1"] and c["map_xlim"] == c["profile_xlim"] == [lon0, lon1]
    # the lazy layout, a colorbar attached to the map axes, narrows the map and is refused
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(14, 10))
    axes, caxes = C.panel_axes(fig)
    m = axes[0, 0].pcolormesh(np.arange(5), np.arange(4), np.zeros((4, 5)), shading="auto")
    fig.colorbar(m, ax=axes[0, 0])
    assert C.panel_alignment(axes)["aligned"] is False
    plt.close(fig)
    # and draw_step itself refuses to save when the alignment check fails
    C.panel_alignment = lambda axes, tol=1e-6: {"aligned": False, "columns": []}
    refused = tmp_path / "refused.png"
    with pytest.raises(RuntimeError, match="REFUSED"):
        C.draw_step(st, case_fields, grids, box, strip, band, hyp, t, reading, np.sin(lons), None, lons, str(refused), "layout test")
    assert not refused.exists()
