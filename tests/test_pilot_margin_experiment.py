"""The margin experiment builds three preparations of one step, verifies that the margin
preparation equals the wider one in every common column and differs from the 40 E one only
in the edge columns, reproduces the detector's own candidates on the native bundle, and
reads a mechanical verdict against the retained native-only candidates."""
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


def _case(lon_hi, troughs, steps=2):
    """A case on the stage diagnostic's grids with a trough ridge at each longitude in
    `troughs`, the same analytic fields on either grid so the common columns are equal."""
    lat_c = np.arange(34.0, -35.0, -2.0)
    lon_c = np.arange(-139.0, lon_hi, 2.0)
    latgrid, longrid = np.meshgrid(np.arange(35.0, -36.0, -1.0), np.arange(-140.0, lon_hi + 0.5, 1.0), indexing="ij")
    LA, LO = np.meshgrid(lat_c, lon_c, indexing="ij")
    anom = sum(2e-5 * np.exp(-((LO - c) / 3.0) ** 2) * np.exp(-((LA - 12.0) / 6.0) ** 2) for c in troughs)
    adv = sum(1e-10 * (LO - c) * np.exp(-((LO - c) / 6.0) ** 2) * np.exp(-((LA - 12.0) / 6.0) ** 2) for c in troughs)
    anom_f = sum(2e-5 * np.exp(-((longrid - c) / 3.0) ** 2) * np.exp(-((latgrid - 12.0) / 6.0) ** 2) for c in troughs)
    u_c = -7.0 + 0.1 * LO / 100.0
    u_f = -7.0 + 0.1 * longrid / 100.0
    v_f = 0.5 * np.sin(latgrid / 10.0)
    return {"time": 100.0 + 0.25 * np.arange(steps), "lat_c": lat_c, "lon_c": lon_c, "latgrid": latgrid, "longrid": longrid,
            "u_c": np.repeat(u_c[None], steps, 0), "v_c": np.zeros((steps,) + LA.shape), "currv_anom_c": np.repeat(np.where(LA < 0, -anom, anom)[None], steps, 0),
            "advcurrv_anom_c": np.repeat(adv[None], steps, 0), "currv_anom": np.repeat(anom_f[None], steps, 0),
            "u": np.repeat(u_f[None], steps, 0), "v": np.repeat(v_f[None], steps, 0), "sha256": "x" * 64}


def test_the_margin_preparation_equals_the_wider_one_in_common_columns_and_differs_from_p40_only_at_the_edge():
    M = _load("pilot_margin_experiment")
    case_b, case_c = _case(39.0, [36.0]), _case(59.0, [36.0])                                  # one trough one column inside the edge, the same analytic fields on both grids
    cols_c, fcols_c = M.common_columns(case_b, case_c)
    raw = M.raw_equality(case_b, case_c, 0, cols_c, fcols_c)
    assert all(v["differing_all_steps"] == 0 and v["differing_at_step"] == 0 for v in raw.values()) and raw["u"]["steps"] == 2
    tails = M.raw_equality(case_b, _case(59.0, [36.0, 50.0]), 0, cols_c, fcols_c)             # a second trough's tails reach the common columns
    assert tails["advcurrv_anom_c"]["differing_all_steps"] > 0
    p40, l40, _ = M.prepared_bundle(case_b, 0, 4.0e-7, 2.5e-6)
    p60, l60, _ = M.prepared_bundle(case_c, 0, 4.0e-7, 2.5e-6)
    pm, lm, _ = M.margin_bundle(case_b, case_c, 0, 4.0e-7, 2.5e-6, cols_c, fcols_c)
    cmp = M.prepared_comparison(p40, pm, p60, case_b, cols_c, fcols_c)
    assert all(v["cells_differing"] == 0 for v in cmp["PMARGIN_vs_P60_common_columns"].values())
    last_coarse, last_fine = float(case_b["lon_c"][-1]), float(case_b["longrid"][0, -1])
    for name, v in cmp["P40_vs_PMARGIN"].items():
        assert set(v["columns_differing"]) <= {last_coarse, last_fine}, name
    assert cmp["P40_vs_PMARGIN"]["coarse_curvature"]["columns_differing"] == [last_coarse]           # the ridge at 36 E is live in the edge column
    assert cmp["P40_vs_PMARGIN"]["fine_curvature"]["columns_differing"] == [last_fine]
    assert pm["extent"] == p40["extent"] and pm["digests"]["lon_c"] == p40["digests"]["lon_c"] and pm["digests"]["coarse_curvature"] != p40["digests"]["coarse_curvature"]
    assert len(l60) == 1 and len(lm) == 1 and len(l40) in (0, 1)                               # the edge-smoothed advection may push the zero crossing off the control grid


def test_the_native_bundle_reproduces_the_detector_and_the_margin_merge_runs_on_a_fixed_list():
    M = _load("pilot_margin_experiment")
    from aew.v1port.detection import detect_troughs
    case_b, case_c = _case(39.0, [20.0]), _case(59.0, [20.0, 50.0])
    cols_c, fcols_c = M.common_columns(case_b, case_c)
    p40, l40, _ = M.prepared_bundle(case_b, 0, 4.0e-7, 2.5e-6)
    pm, lm, _ = M.margin_bundle(case_b, case_c, 0, 4.0e-7, 2.5e-6, cols_c, fcols_c)
    p60, l60, _ = M.prepared_bundle(case_c, 0, 4.0e-7, 2.5e-6)
    waves = detect_troughs(100.0, *np.meshgrid(case_b["lat_c"], case_b["lon_c"], indexing="ij"), case_b["u_c"][0], case_b["currv_anom_c"][0], case_b["advcurrv_anom_c"][0],
                           case_b["latgrid"], case_b["longrid"], case_b["currv_anom"][0], coarse_threshold=4.0e-7, fine_threshold=2.5e-6)
    native = M.CI.combination(p40, l40, "as_given")
    assert M.CI.native_check(native, [(w["lat_mean"], w["lon_mean"], M.R.region_digest(w["region"])) for w in waves])["reproduces_replay_candidates"]
    on_margin = M.CI.combination(pm, l60, "as_given")                                            # the treatment's list on the cropped grid
    assert on_margin["candidates_used"] == 2 and not on_margin["candidates"][1]["inside_receiving_grid"] and on_margin["candidates"][1]["seed_lon"] <= 39.0
    withheld = M.CI.combination(pm, l60, "inside_receiving_grid")
    assert withheld["candidates_used"] == 1 and withheld["candidates_withheld"][0]["index"] == 1
    assert [f["lat_mean"] for f in withheld["final"]] == [f["lat_mean"] for f in M.CI.combination(pm, lm, "as_given")["final"]]


def test_the_verdict_reads_the_margin_finals_against_the_retained_native_only_candidates():
    M = _load("pilot_margin_experiment")
    reference = {"B": [[10.5, 36.0]], "C": [[5.5, 39.0]]}
    def results(margin_final):
        out = {}
        for contrast, lists in (("edge_ring", ("L40", "LMARGIN")), ("extent", ("LMARGIN", "L60"))):
            for lst in lists:
                for variant in ("as_given", "inside_receiving_grid"):
                    out[(contrast, "PMARGIN", lst, variant)] = {"final": margin_final}
                    out[(contrast, "P40" if contrast == "edge_ring" else "P60", lst, variant)] = {"final": [{"lat_mean": 0.0, "lon_mean": 0.0}]}
        return out
    assert M.verdict(results([{"lat_mean": 5.5, "lon_mean": 39.0}]), reference)["verdict"] == "smoothing"
    assert M.verdict(results([{"lat_mean": 10.5, "lon_mean": 36.0}]), reference)["verdict"] == "extent"
    assert M.verdict(results([{"lat_mean": 5.5, "lon_mean": 39.0}, {"lat_mean": 10.5, "lon_mean": 36.0}]), reference)["verdict"] == "mixed_or_null"
    assert M.verdict(results([{"lat_mean": 20.0, "lon_mean": 20.0}]), reference)["verdict"] == "mixed_or_null"
    assert M.verdict(results([{"lat_mean": 5.5, "lon_mean": 39.6}]), reference)["verdict"] == "mixed_or_null"       # within a degree but not exact
    assert "not one each" in M.verdict({}, {"B": [], "C": [[5.5, 39.0]]})["note"]


def test_common_columns_select_by_membership_and_refuse_a_missing_or_reordered_column():
    M = _load("pilot_margin_experiment")
    case_b = _case(39.0, [20.0])
    wider_west = _case(59.0, [20.0])
    wider_west["lon_c"] = np.concatenate([[-141.0], wider_west["lon_c"]])                     # an extra column WEST of the control's first, so a prefix is wrong
    for name in ("u_c", "v_c", "currv_anom_c", "advcurrv_anom_c"):
        wider_west[name] = np.concatenate([wider_west[name][:, :, :1], wider_west[name]], axis=2)
    lat_f = wider_west["latgrid"][:, :1]
    wider_west["longrid"] = np.concatenate([np.full_like(lat_f, -141.0), wider_west["longrid"]], axis=1)   # and one WEST of the control's first fine column
    wider_west["latgrid"] = np.concatenate([lat_f, wider_west["latgrid"]], axis=1)
    for name in ("u", "v", "currv_anom"):
        wider_west[name] = np.concatenate([wider_west[name][:, :, :1], wider_west[name]], axis=2)
    cols_c, fcols_c = M.common_columns(case_b, wider_west)
    assert not cols_c[0] and cols_c.sum() == case_b["lon_c"].size and np.array_equal(wider_west["lon_c"][cols_c], case_b["lon_c"])
    assert not fcols_c[0] and fcols_c.sum() == case_b["longrid"].shape[1] and np.array_equal(wider_west["longrid"][0, fcols_c], case_b["longrid"][0, :])
    raw = M.raw_equality(case_b, wider_west, 0, cols_c, fcols_c)                                 # the selection, not a prefix, is what makes the fields equal
    assert all(v["differing_all_steps"] == 0 for v in raw.values())
    missing = _case(59.0, [20.0])
    keep = missing["lon_c"] != 1.0
    missing["lon_c"] = missing["lon_c"][keep]
    for name in ("u_c", "v_c", "currv_anom_c", "advcurrv_anom_c"):
        missing[name] = missing[name][:, :, keep]
    with pytest.raises(SystemExit, match="not a subset"):
        M.common_columns(case_b, missing)
    missing_fine = _case(59.0, [20.0])
    keep_f = missing_fine["longrid"][0, :] != 1.0
    missing_fine["longrid"], missing_fine["latgrid"] = missing_fine["longrid"][:, keep_f], missing_fine["latgrid"][:, keep_f]
    for name in ("u", "v", "currv_anom"):
        missing_fine[name] = missing_fine[name][:, :, keep_f]
    with pytest.raises(SystemExit, match="not a subset"):
        M.common_columns(case_b, missing_fine)
    other_rows = _case(59.0, [20.0])
    other_rows["lat_c"] = other_rows["lat_c"] + 0.5
    with pytest.raises(SystemExit, match="latitude rows"):
        M.common_columns(case_b, other_rows)


def test_the_margin_bundle_takes_the_control_thresholds_and_the_wider_one_the_treatments():
    M = _load("pilot_margin_experiment")
    case_b, case_c = _case(39.0, [20.0]), _case(59.0, [20.0])
    cols_c, fcols_c = M.common_columns(case_b, case_c)
    thr = {"B": (4.0e-7, 2.5e-6), "C": (4.4e-7, 2.9e-6)}
    preps, lists, sizes, thresholds = M.build_bundles({"B": case_b, "C": case_c}, thr, 0, cols_c, fcols_c)
    assert thresholds == {"P40": thr["B"], "PMARGIN": thr["B"], "P60": thr["C"]}
    assert (preps["PMARGIN"]["ct"], preps["PMARGIN"]["ft"]) == thr["B"] and (preps["P60"]["ct"], preps["P60"]["ft"]) == thr["C"]
    assert set(lists) == {"L40", "LMARGIN", "L60"} and set(sizes) == set(lists)


def test_the_verdict_reads_bind_both_declared_radii():
    M = _load("pilot_margin_experiment")
    reference = {"B": [[10.5, 36.0]], "C": [[5.5, 39.0]]}
    def results(margin_final):
        return {(c, "PMARGIN", l, v): {"final": margin_final} for c, ls in (("edge_ring", ("L40", "LMARGIN")), ("extent", ("LMARGIN", "L60"))) for l in ls for v in ("as_given", "inside_receiving_grid")}
    def read(offset_lon, tol_name):
        r = M.verdict(results([{"lat_mean": 5.5, "lon_mean": 39.0 + offset_lon}]), reference)["reads"]["extent/L60/as_given/" + tol_name]
        return r["holds_treatment_only"]
    assert read(0.999, "within_match_deg") is True and read(1.001, "within_match_deg") is False
    assert read(5e-10, "exact") is True and read(2e-9, "exact") is False
    assert M.verdict(results([{"lat_mean": 5.5, "lon_mean": 39.999}]), reference)["verdict"] == "mixed_or_null"        # loose holds, exact does not


def test_content_and_order_are_reported_separately():
    M = _load("pilot_margin_experiment")
    a = {"final": [{"lat_mean": 1.0, "lon_mean": 2.0, "region_sha256": "x"}, {"lat_mean": 3.0, "lon_mean": 4.0, "region_sha256": "y"}]}
    b = {"final": a["final"][::-1]}
    assert M.content_and_order(a, a) == {"same_content": True, "same_order": True, "count_a": 2, "count_b": 2}
    assert M.content_and_order(a, b) == {"same_content": True, "same_order": False, "count_a": 2, "count_b": 2}
    c = {"final": a["final"][:1]}
    assert M.content_and_order(a, c)["same_content"] is False


def test_an_undeclared_date_and_an_existing_output_are_refused(tmp_path):
    M = _load("pilot_margin_experiment")
    args = ["--control-run", "a", "--control-case", "b", "--treatment-run", "c", "--treatment-case", "d", "--families", "f", "--family", "g",
            "--replay-control", "h", "--replay-treatment", "i", "--crossed-input", "j", "--year", "1990"]
    out = tmp_path / "x.json"
    with pytest.raises(SystemExit, match="no expectation was declared"):
        M.main(args + ["--date", "1990-07-02T12", "--out", str(out)])
    out.write_text("{}")
    with pytest.raises(SystemExit, match="never overwritten"):
        M.main(args + ["--date", "1990-07-01T12", "--out", str(out)])
