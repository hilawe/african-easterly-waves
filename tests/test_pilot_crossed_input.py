"""The crossed-input experiment reproduces the detector's merge on a native combination,
binds its replicated seed cell to the production merge, lists a candidate outside the
receiving grid with its displaced seed in one variant and withholds it in the other, and
reads where a native-only candidate appears across the four combinations."""
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


def _case(lon_hi, troughs):
    """A one-step case with a trough ridge at each longitude in `troughs`."""
    lat_c = np.arange(34.0, -35.0, -2.0)
    lon_c = np.arange(-139.0, lon_hi, 2.0)
    latgrid, longrid = np.meshgrid(np.arange(35.0, -36.0, -1.0), np.arange(-140.0, lon_hi + 0.5, 1.0), indexing="ij")
    LA, LO = np.meshgrid(lat_c, lon_c, indexing="ij")
    anom = sum(2e-5 * np.exp(-((LO - c) / 3.0) ** 2) * np.exp(-((LA - 12.0) / 6.0) ** 2) for c in troughs)
    adv = sum(1e-10 * (LO - c) * np.exp(-((LO - c) / 6.0) ** 2) * np.exp(-((LA - 12.0) / 6.0) ** 2) for c in troughs)
    anom_f = sum(2e-5 * np.exp(-((longrid - c) / 3.0) ** 2) * np.exp(-((latgrid - 12.0) / 6.0) ** 2) for c in troughs)
    return {"time": np.array([100.0]), "lat_c": lat_c, "lon_c": lon_c, "latgrid": latgrid, "longrid": longrid,
            "u_c": np.full((1,) + LA.shape, -7.0), "v_c": np.zeros((1,) + LA.shape), "currv_anom_c": np.where(LA < 0, -anom, anom)[None],
            "advcurrv_anom_c": adv[None], "currv_anom": anom_f[None], "u": np.full((1,) + latgrid.shape, -7.0), "v": np.zeros((1,) + latgrid.shape), "sha256": "x" * 64}


def test_native_merge_reproduces_detect_troughs_and_an_outside_candidate_is_displaced_or_withheld():
    C = _load("pilot_crossed_input")
    from aew.v1port.detection import detect_troughs
    case_b, case_c = _case(39.0, [20.0]), _case(59.0, [20.0, 50.0])
    prep_b, cands_b, _ = C.bundles(case_b, 0, 4.0e-7, 2.5e-6)
    prep_c, cands_c, _ = C.bundles(case_c, 0, 4.0e-7, 2.5e-6)
    assert len(cands_b) == 1 and len(cands_c) == 2 and cands_c[1]["lon_mean"] > 39.0
    lat_c, lon_c = case_c["lat_c"], case_c["lon_c"]
    waves = detect_troughs(100.0, *np.meshgrid(lat_c, lon_c, indexing="ij"), case_c["u_c"][0], case_c["currv_anom_c"][0], case_c["advcurrv_anom_c"][0],
                           case_c["latgrid"], case_c["longrid"], case_c["currv_anom"][0], coarse_threshold=4.0e-7, fine_threshold=2.5e-6)
    native = C.combination(prep_c, cands_c, "as_given")
    assert [(f["lat_mean"], f["lon_mean"], f["region_sha256"]) for f in native["final"]] == [(w["lat_mean"], w["lon_mean"], C.R.region_digest(w["region"])) for w in waves]
    assert native["candidates"][0]["seed_displacement_deg"] < 2.0 and native["candidates"][0]["inside_receiving_grid"]
    row0 = native["candidates"][0]
    assert row0["region_lon_extent"][0] <= 20.0 <= row0["region_lon_extent"][1] and row0["base_level_region_cells"] == row0["region_cells"] and row0["region_smaller_than_base_level"] is False
    assert C.combination(prep_c, cands_c, "inside_receiving_grid")["final"] == native["final"]
    crossed = C.combination(prep_b, cands_c, "as_given")                                    # C's candidates on B's grid
    row = crossed["candidates"][1]
    assert not row["inside_receiving_grid"] and row["seed_lon"] <= 39.0 and row["seed_displacement_deg"] >= 50.0 - 39.0
    assert crossed["candidates_withheld"] == [] and crossed["candidates_used"] == 2
    inside_only = C.combination(prep_b, cands_c, "inside_receiving_grid")
    assert inside_only["candidates_used"] == 1 and inside_only["candidates_withheld"][0]["index"] == 1
    assert [f["lat_mean"] for f in inside_only["final"]] == [f["lat_mean"] for f in C.combination(prep_b, cands_b, "as_given")["final"]]


def test_a_wide_weak_ridge_makes_the_ladder_step_up_and_the_region_shrink():
    """A ridge above the base threshold but below the next level, 36 degrees wide, around a
    strong core: the base-level component spans the ridge, the extent rule steps the ladder
    up, and the selected region is the core alone."""
    C = _load("pilot_crossed_input")
    case = _case(39.0, [20.0])
    lat_c, lon_c = case["lat_c"], case["lon_c"]
    LA, LO = np.meshgrid(lat_c, lon_c, indexing="ij")
    ridge = (np.abs(LA - 12.0) <= 2.0) & (LO >= 2.0) & (LO <= 38.0)
    case["currv_anom_c"][0][ridge] = np.maximum(case["currv_anom_c"][0][ridge], 5.0e-7)       # above 4e-7, below 1.5 x 4e-7
    prep, cands, _ = C.bundles(case, 0, 4.0e-7, 2.5e-6)
    rows = C.candidate_rows(prep, cands)
    row = rows[0]
    assert row["base_level_lon_extent"][1] - row["base_level_lon_extent"][0] >= 25.0
    assert row["region_cells"] < row["base_level_region_cells"] and row["region_smaller_than_base_level"] is True
    assert row["region_lon_extent"][1] - row["region_lon_extent"][0] < 25.0


def test_presence_is_a_euclidean_distance_and_a_counterpart_outside_the_box_still_counts():
    C = _load("pilot_crossed_input")
    assert C.present((10.0, 30.0), [{"lat_mean": 10.8, "lon_mean": 30.8}], 1.0) is False          # 1.13 degrees away, not a counterpart
    assert C.present((10.0, 30.0), [{"lat_mean": 10.6, "lon_mean": 30.6}], 1.0) is True           # 0.85 degrees away
    assert C.present((10.0, 30.0), [{"lat_mean": 10.0, "lon_mean": 30.0 + 5e-10}], 1e-9) is True
    # the reading under the 1-degree rule with a diagonal offset in one crossed combination
    def results(where):
        return {(p, c): {"as_given": {"final": [{"lat_mean": 10.0 + where[p + c][0], "lon_mean": 30.0 + where[p + c][1]}] if where[p + c] else []}} for p in "BC" for c in "BC"}
    r = C.reading((10.0, 30.0), "B", "B", results({"BB": (0.0, 0.0), "BC": (0.8, 0.8), "CB": None, "CC": None}), "as_given", 1.0)
    assert r["follows"] == "interaction_or_unresolved"
    r = C.reading((10.0, 30.0), "C", "C", results({"CC": (0.0, 0.0), "CB": (0.6, 0.6), "BC": None, "BB": None}), "as_given", 1.0)
    assert r["follows"] == "preparation" and r["present_in"]["prep_C_cand_B"] is True and r["present_in"]["prep_B_cand_C"] is False


def test_a_counterpart_outside_the_box_or_east_of_the_common_edge_still_disqualifies_native_only():
    """The selection `main` uses: a control candidate at the box's western edge whose
    treatment counterpart lies 0.2 degrees outside the box is not native-only, and one whose
    counterpart lies just east of the common edge is not either."""
    C = _load("pilot_crossed_input")
    def results(b_finals, c_finals):
        empty = {"as_given": {"final": []}}
        return {("B", "B"): {"as_given": {"final": b_finals}}, ("C", "C"): {"as_given": {"final": c_finals}}, ("B", "C"): empty, ("C", "B"): empty}
    box = {"lat": [0.0, 25.0], "lon": [30.0, 50.0]}
    finals, native = C.native_only_sets(results([{"lat_mean": 10.0, "lon_mean": 30.0}], [{"lat_mean": 10.0, "lon_mean": 29.8}]), box)
    assert finals["B"] == [{"lat_mean": 10.0, "lon_mean": 30.0}] and finals["C"] == [] and native == {"B": [], "C": []}
    finals, native = C.native_only_sets(results([{"lat_mean": 10.0, "lon_mean": 39.9}], [{"lat_mean": 10.0, "lon_mean": 40.3}]), box)
    assert finals["C"] == [] and native == {"B": [], "C": []}                                   # the counterpart is east of the common edge
    finals, native = C.native_only_sets(results([{"lat_mean": 10.0, "lon_mean": 35.0}], [{"lat_mean": 10.0, "lon_mean": 36.5}]), box)
    assert native == {"B": [{"lat_mean": 10.0, "lon_mean": 35.0}], "C": [{"lat_mean": 10.0, "lon_mean": 36.5}]}     # 1.5 degrees apart, both native-only


def test_the_native_endpoint_check_reads_centers_regions_and_order():
    C = _load("pilot_crossed_input")
    result = {"final": [{"lat_mean": 1.0, "lon_mean": 2.0, "region_sha256": "a"}, {"lat_mean": 3.0, "lon_mean": 4.0, "region_sha256": "b"}]}
    logged = [(1.0, 2.0, "a"), (3.0, 4.0, "b")]
    assert C.native_check(result, logged)["reproduces_replay_candidates"] is True
    assert C.native_check(result, [(1.0, 2.0, "a"), (3.0, 4.0, "c")])["reproduces_replay_candidates"] is False     # one region digest
    assert C.native_check(result, logged[::-1])["reproduces_replay_candidates"] is False                           # order
    assert C.native_check(result, logged[:1])["reproduces_replay_candidates"] is False                             # count


def test_the_seed_bind_refuses_a_seed_that_grows_another_region(monkeypatch):
    C = _load("pilot_crossed_input")
    case = _case(39.0, [20.0])
    prep, cands, _ = C.bundles(case, 0, 4.0e-7, 2.5e-6)
    real = C.seed_cell

    def elsewhere(prep_, cand):
        masks, seed, disp = real(prep_, cand)
        return masks, (0, 0), disp                                                          # the north-west corner, below threshold
    monkeypatch.setattr(C, "seed_cell", elsewhere)
    with pytest.raises(SystemExit, match="production merge does not return"):
        C.candidate_rows(prep, cands)


def test_the_reading_names_what_a_native_only_candidate_follows():
    C = _load("pilot_crossed_input")

    def results(where):
        return {(p, c): {"as_given": {"final": [{"lat_mean": 1.0, "lon_mean": 1.0}] if where[p + c] else []}} for p in "BC" for c in "BC"}
    point = (1.0, 1.0)
    assert C.reading(point, "B", "B", results({"BB": 1, "BC": 1, "CB": 0, "CC": 0}), "as_given", 1e-9)["follows"] == "preparation"
    assert C.reading(point, "B", "B", results({"BB": 1, "BC": 0, "CB": 1, "CC": 0}), "as_given", 1e-9)["follows"] == "candidates"
    assert C.reading(point, "B", "B", results({"BB": 1, "BC": 0, "CB": 0, "CC": 0}), "as_given", 1e-9)["follows"] == "interaction_or_unresolved"
    assert C.reading(point, "B", "B", results({"BB": 1, "BC": 1, "CB": 1, "CC": 0}), "as_given", 1e-9)["follows"] == "interaction_or_unresolved"


def test_an_existing_output_is_refused(tmp_path):
    C = _load("pilot_crossed_input")
    out = tmp_path / "x.json"
    out.write_text("{}")
    with pytest.raises(SystemExit, match="never overwritten"):
        C.main(["--control-run", "a", "--control-case", "b", "--treatment-run", "c", "--treatment-case", "d", "--date", "1990-07-01T12", "--families", "f",
                "--family", "g", "--replay-control", "h", "--replay-treatment", "i", "--out", str(out), "--year", "1990"])
