"""The stage diagnostic recomputes the detector's stages in its order, compares two runs
stage by stage inside a box and the common domain, reads a named track's position only at
its own observation times, and refuses an existing output directory or a case the record
does not name."""
import importlib.util
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def _load():
    spec = importlib.util.spec_from_file_location("pilot_stage_diagnostic_under_test", os.path.join(ROOT, "scripts", "pilot_stage_diagnostic.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _case(lon_hi, trough_lon=20.0):
    """A one-step case on a 2-degree coarse grid and 1-degree fine grid, with one trough:
    a positive anomaly ridge along `trough_lon` and an advection field crossing zero there."""
    lat_c = np.arange(34.0, -35.0, -2.0)
    lon_c = np.arange(-139.0, lon_hi, 2.0)
    latgrid, longrid = np.meshgrid(np.arange(35.0, -36.0, -1.0), np.arange(-140.0, lon_hi + 0.5, 1.0), indexing="ij")
    LA, LO = np.meshgrid(lat_c, lon_c, indexing="ij")
    anom = 2e-5 * np.exp(-((LO - trough_lon) / 4.0) ** 2) * np.exp(-((LA - 12.0) / 6.0) ** 2)
    anom = np.where(LA < 0, -anom, anom)                                                   # southern rows cyclonic negative, as stored
    adv = 1e-10 * (LO - trough_lon) * np.exp(-((LA - 12.0) / 6.0) ** 2)                   # zero crossing on the ridge
    anom_f = 2e-5 * np.exp(-((longrid - trough_lon) / 4.0) ** 2) * np.exp(-((latgrid - 12.0) / 6.0) ** 2)
    return {"time": np.array([100.0]), "lat_c": lat_c, "lon_c": lon_c, "latgrid": latgrid, "longrid": longrid,
            "u_c": np.full((1,) + LA.shape, -5.0), "v_c": np.zeros((1,) + LA.shape), "currv_anom_c": anom[None], "advcurrv_anom_c": adv[None],
            "currv_anom": anom_f[None], "u": np.full((1,) + latgrid.shape, -5.0), "v": np.zeros((1,) + latgrid.shape), "sha256": "x" * 64}


def test_stages_find_the_trough_and_identical_runs_show_no_difference():
    D = _load()
    case_b, case_c = _case(39.0), _case(59.0)
    sb = D.stages(case_b, 0, 4.0e-7, 2.5e-6)
    sc = D.stages(case_c, 0, 4.0e-7, 2.5e-6)
    assert sb["axes"] and sb["final"] and abs(sb["final"][0][1] - 20.0) < 2.5 and abs(sb["final"][0][0] - 12.0) < 4.0
    assert np.isnan(sb["prepared"]).any() and not np.isnan(sb["prepared"][11, 79])            # masked far from the ridge, kept on it (12 N, 19 E)
    box = {"lat": [0.0, 25.0], "lon": [5.0, 35.0]}
    cmp = D.compare_step(sb, sc, case_b, case_c, box)
    assert cmp["prepared_field"]["cells_differing"] == 0 and cmp["axes"]["vertices_b"] == 13 == cmp["axes"]["vertices_c"]
    assert cmp["axes"]["b_vertices_without_c_counterpart"] == 0 == cmp["axes"]["c_vertices_without_b_counterpart"]   # exact matches at distance 0 are matches
    assert cmp["final_candidates"]["only_b"] == [] and cmp["final_candidates"]["only_c"] == [] and cmp["final_candidates"]["b"] == [[12.0, 20.0]]
    assert cmp["axes"]["vertex_sets_identical"] is True
    moved = D.stages(_case(59.0, trough_lon=30.0), 0, 4.0e-7, 2.5e-6)
    cmp2 = D.compare_step(sb, moved, case_b, case_c, box)
    assert cmp2["prepared_field"]["cells_differing"] > 0 and cmp2["final_candidates"]["only_b"] and cmp2["final_candidates"]["only_c"]
    nudged = D.stages(_case(59.0, trough_lon=20.5), 0, 4.0e-7, 2.5e-6)
    cmp3 = D.compare_step(sb, nudged, case_b, case_c, box)
    assert cmp3["axes"]["b_vertices_without_c_counterpart"] == 0 and cmp3["axes"]["vertex_sets_identical"] is False   # within 1 degree is matched, identity says otherwise
    tiny = dict(sc, axes=[(la, lo + 4e-7) for la, lo in sc["axes"]])                    # a sub-microdegree shift is still not identity
    assert D.compare_step(sb, tiny, case_b, case_c, box)["axes"]["vertex_sets_identical"] is False


def test_every_prepared_input_is_compared_on_its_own_grid():
    D = _load()
    case_b, case_c = _case(39.0), _case(59.0)
    box = {"lat": [0.0, 25.0], "lon": [5.0, 35.0]}
    sb = D.stages(case_b, 0, 4.0e-7, 2.5e-6)
    same = D.compare_step(sb, D.stages(case_c, 0, 4.0e-7, 2.5e-6), case_b, case_c, box)
    assert sorted(same["prepared_inputs"]) == sorted(n for n, _k, _g in D.PREPARED_INPUTS) and len(same["prepared_inputs"]) == 6
    assert all(v["cells_differing"] == 0 for v in same["prepared_inputs"].values())
    assert same["coordinate_support"]["coarse_common_columns"] == 16 and same["coordinate_support"]["fine_common_columns"] == 31
    assert same["coordinate_support"]["fine_easternmost_lon_b"] == 39.0 and same["coordinate_support"]["coarse_easternmost_lon_c"] == 57.0
    fine_only = _case(59.0)
    fine_only["currv_anom"][0, 20:24, 160:163] += 1e-5                                    # 15 to 12 N, 20 to 22 E, fine grid only
    cmp = D.compare_step(sb, D.stages(fine_only, 0, 4.0e-7, 2.5e-6), case_b, case_c, box)
    assert cmp["prepared_inputs"]["coarse_curvature"]["cells_differing"] == 0 == cmp["prepared_field"]["cells_differing"]
    assert cmp["prepared_inputs"]["fine_curvature"]["cells_differing"] > 0 and cmp["prepared_inputs"]["fine_curvature"]["westernmost_differing_lon"] == 19.0
    assert cmp["prepared_inputs"]["fine_zonal_wind_smoothed"]["cells_differing"] == 0
    wind_only = _case(59.0)
    wind_only["u"][0, 20:24, 160:163] = 3.0
    cmp = D.compare_step(sb, D.stages(wind_only, 0, 4.0e-7, 2.5e-6), case_b, case_c, box)
    assert cmp["prepared_inputs"]["fine_zonal_wind_smoothed"]["cells_differing"] > 0 and cmp["prepared_inputs"]["fine_meridional_wind_smoothed"]["cells_differing"] == 0
    assert cmp["prepared_inputs"]["fine_curvature"]["cells_differing"] == 0 and cmp["prepared_inputs"]["coarse_zonal_wind_smoothed"]["cells_differing"] == 0
    v_only = _case(59.0)
    v_only["v"][0, 20:24, 160:163] = 3.0
    cmp = D.compare_step(sb, D.stages(v_only, 0, 4.0e-7, 2.5e-6), case_b, case_c, box)
    assert cmp["prepared_inputs"]["fine_meridional_wind_smoothed"]["cells_differing"] > 0
    assert all(cmp["prepared_inputs"][n]["cells_differing"] == 0 for n in cmp["prepared_inputs"] if n != "fine_meridional_wind_smoothed")
    other_rows = _case(59.0)
    other_rows["lat_c"] = other_rows["lat_c"] + 0.5
    with pytest.raises(SystemExit, match="latitude rows"):
        D.compare_step(sb, D.stages(other_rows, 0, 4.0e-7, 2.5e-6), case_b, other_rows, box)
    other_fine_rows = _case(59.0)
    other_fine_rows["latgrid"] = other_fine_rows["latgrid"] + 0.5
    with pytest.raises(SystemExit, match="latitude rows"):
        D.compare_step(sb, D.stages(other_fine_rows, 0, 4.0e-7, 2.5e-6), case_b, other_fine_rows, box)


def test_the_westerly_mask_is_applied_in_the_stage():
    import numpy as np
    D = _load()
    case = _case(59.0)
    case["u_c"][0, :, 79:81] = 10.0                                                        # westerlies on the ridge's columns (19 and 21 E)
    s = D.stages(case, 0, 4.0e-7, 2.5e-6)
    assert np.isnan(s["prepared"][11, 79]) and np.isnan(s["prepared"][11, 80])              # masked by the smoothed wind
    assert not s["axes"] or all(not (18.0 <= float(np.mean(lo)) <= 22.0 and 8.0 <= float(np.mean(la)) <= 16.0) for la, lo in s["axes"])   # no axis on the masked ridge


def test_track_rows_read_positions_only_at_observation_times_and_name_the_other_runs_track():
    D = _load()
    case_b, case_c = _case(39.0), _case(59.0)
    sb, sc = D.stages(case_b, 0, 4.0e-7, 2.5e-6), D.stages(case_c, 0, 4.0e-7, 2.5e-6)
    tr = {"time": np.array([99.75, 100.0]), "lat": np.array([12.0, 12.0]), "lon": np.array([22.0, 20.5])}
    other = [{"time": np.array([100.0]), "lat": np.array([13.0]), "lon": np.array([20.0])}]
    row = D.track_row("B0", tr, 100.0, sb, sc, other)
    assert row["position"] == [12.0, 20.5] and row["nearest_final_candidate_own_run_deg"] is not None and row["nearest_stored_position_other_run_track"] == 0
    assert D.track_row("B0", tr, 100.25, sb, sc, other) == {"track": "B0", "position": None}
    assert D.nearest((0.0, 0.0), [(3.0, 4.0), (1.0, 0.0)]) == (1.0, (1.0, 0.0)) and D.nearest((0.0, 0.0), []) == (None, None)
    assert D.in_box([(1.0, 1.0), (30.0, 1.0)], {"lat": [0, 10], "lon": [0, 10]}) == [(1.0, 1.0)]


def test_an_existing_output_directory_and_a_missing_coastline_cache_are_refused(tmp_path, monkeypatch):
    import json
    D = _load()
    (tmp_path / "out").mkdir()
    fams = tmp_path / "f.json"; fams.write_text(json.dumps({"families": []}))
    with pytest.raises(SystemExit, match="figure sets are never overwritten"):
        D.main(["--families", str(fams), "--control-run", "x", "--control-case", "x", "--treatment-run", "x", "--treatment-case", "x", "--year", "1990", "--out-dir", str(tmp_path / "out")])
    import cartopy
    monkeypatch.setitem(cartopy.config, "data_dir", str(tmp_path / "nowhere"))
    with pytest.raises(SystemExit, match="does not download"):
        D.coastlines()
