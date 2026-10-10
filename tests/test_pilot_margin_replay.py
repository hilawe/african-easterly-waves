"""The margin replay's native mode reproduces `detect_troughs` step for step and the
production loop's finished tracks, its margin mode runs the same loop on the wider case
cropped to the control grids with every finished track's raw history retained, and the
comparison applies the declared pairing and divergence rules."""
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


def moving_case(lon_hi, steps=14, start_lon=36.0, speed=-1.5):
    """A trough drifting west from near the control's edge, the same analytic fields on
    either grid, with a uniform easterly."""
    lat_c = np.arange(34.0, -35.0, -2.0)
    lon_c = np.arange(-139.0, lon_hi, 2.0)
    latgrid, longrid = np.meshgrid(np.arange(35.0, -36.0, -1.0), np.arange(-140.0, lon_hi + 0.5, 1.0), indexing="ij")
    LA, LO = np.meshgrid(lat_c, lon_c, indexing="ij")
    anom, adv, anom_f = [], [], []
    for k in range(steps):
        c = start_lon + k * speed
        a = 2e-5 * np.exp(-((LO - c) / 4.0) ** 2) * np.exp(-((LA - 12.0) / 6.0) ** 2)
        anom.append(np.where(LA < 0, -a, a))
        adv.append(1e-10 * (LO - c) * np.exp(-((LA - 12.0) / 6.0) ** 2))
        anom_f.append(2e-5 * np.exp(-((longrid - c) / 4.0) ** 2) * np.exp(-((latgrid - 12.0) / 6.0) ** 2))
    shape = (steps,) + LA.shape
    return {"time": 100.0 + 0.25 * np.arange(steps), "lat_c": lat_c, "lon_c": lon_c, "latgrid": latgrid, "longrid": longrid,
            "u_c": np.full(shape, -7.0), "v_c": np.zeros(shape), "currv_anom_c": np.stack(anom), "advcurrv_anom_c": np.stack(adv),
            "currv_anom": np.stack(anom_f), "u": np.full((steps,) + latgrid.shape, -7.0), "v": np.zeros((steps,) + latgrid.shape), "sha256": "x" * 64}


FLAGS = {"exclusive": False, "absorb": False}


def test_native_mode_binds_to_detect_troughs_and_the_production_loop_and_retains_raw_histories():
    M = _load("pilot_margin_replay")
    E = _load("export_protocol_case")
    case = moving_case(39.0)
    grids = M.control_grids(case)
    all_c, all_f = np.ones(case["lon_c"].size, bool), np.ones(case["longrid"].shape[1], bool)
    finished, counts, bound = M.run_season(case, grids, all_c, all_f, 4.0e-7, 2.5e-6, FLAGS, bind_case=case)
    assert bound == 14 and finished and counts["finished"] == len(finished)
    ref = E.track_case({k: v for k, v in case.items() if k != "sha256"}, 4.0e-7, 2.5e-6, exclusive=False, absorb=False)
    archived = {"tracks": [{"time": np.asarray(t["time"]), "lat": np.asarray(t["meanlat"]), "lon": np.asarray(t["meanlon"])} for t in ref]}
    assert M.archive_gate(finished, archived)["passed"]
    long = max(finished, key=lambda f: len(f["steps"]))
    assert long["steps"][0] == 0 and len(long["raw_lat"]) == len(long["steps"]) == len(long["region_sha256"]) and long["raw_lon"][0] > long["raw_lon"][-1]
    assert long["lon"] != long["raw_lon"]                                                          # the output is smoothed, the raw history is not


def test_margin_mode_crops_the_wider_preparation_and_the_bind_refuses_a_changed_preparation():
    M = _load("pilot_margin_replay")
    ME = _load("pilot_margin_experiment")
    case_b, case_c = moving_case(39.0), moving_case(59.0)
    grids = M.control_grids(case_b)
    cols_c, fcols_c = ME.common_columns(case_b, case_c)
    waves_m, u_s, v_s = M.prepare_and_detect(case_c, 0, 4.0e-7, 2.5e-6, cols_c, fcols_c, grids)
    assert u_s.shape == case_b["latgrid"].shape and waves_m and waves_m[0]["region"].shape == case_b["latgrid"].shape
    finished, counts, bound = M.run_season(case_c, grids, cols_c, fcols_c, 4.0e-7, 2.5e-6, FLAGS)
    assert bound == 0 and finished and all(len(f["steps"]) == len(f["raw_lat"]) for f in finished)
    nudged = {k: v for k, v in case_b.items()}
    nudged["currv_anom"] = np.roll(case_b["currv_anom"], 3, axis=2)                              # a native run whose preparation is not the oracle's: the fine ridge three columns east
    all_c, all_f = np.ones(case_b["lon_c"].size, bool), np.ones(case_b["longrid"].shape[1], bool)
    with pytest.raises(SystemExit, match="does not reproduce detect_troughs"):
        M.run_season(nudged, grids, all_c, all_f, 4.0e-7, 2.5e-6, FLAGS, bind_case=case_b)


def test_the_margin_fine_winds_are_smoothed_on_the_wider_grid_before_the_crop_and_reach_the_association(monkeypatch):
    M = _load("pilot_margin_replay")
    ME = _load("pilot_margin_experiment")
    from aew.v1port import pipeline as PL
    from aew.v1port.climatology import smooth9
    case_b, case_c = moving_case(39.0, steps=3), moving_case(59.0, steps=3)
    for case in (case_b, case_c):
        case["u"] = -7.0 + 0.05 * case["longrid"][None] + 0.02 * case["latgrid"][None] + np.zeros((3, 1, 1))     # winds that vary, so the edge smoothing shows
        case["v"] = 0.5 * np.sin(case["longrid"][None] / 7.0) + np.zeros((3, 1, 1))
    grids = M.control_grids(case_b)
    cols_c, fcols_c = ME.common_columns(case_b, case_c)
    waves, u_s, v_s = M.prepare_and_detect(case_c, 0, 4.0e-7, 2.5e-6, cols_c, fcols_c, grids)
    expected_u = smooth9(case_c["u"][0])[:, fcols_c]
    assert np.array_equal(u_s, expected_u) and np.array_equal(v_s, smooth9(case_c["v"][0])[:, fcols_c])
    native_u = smooth9(case_b["u"][0])
    assert not np.array_equal(u_s[:, -1], native_u[:, -1]) and np.array_equal(u_s[:, :-1], native_u[:, :-1])     # the crop comes after the smoother: only the edge column differs
    seen = []
    real = PL._median_over
    def spy(field):
        seen.append(np.array(field, copy=True))
        return real(field)
    monkeypatch.setattr(PL, "_median_over", spy)
    M.run_season(case_c, grids, cols_c, fcols_c, 4.0e-7, 2.5e-6, FLAGS)
    assert seen and np.array_equal(seen[0], expected_u)                                           # the association received the cropped, wide-smoothed wind


def test_publication_is_exclusive_against_a_competing_writer_and_leaves_no_partial_record(tmp_path, monkeypatch):
    M = _load("pilot_margin_replay")
    import gzip as real_gzip
    real_open = real_gzip.open
    target = tmp_path / "r.json.gz"
    competitor = b"competitor"
    def delayed_open(path, mode, **kw):
        if not target.exists():
            target.write_bytes(competitor)                                                        # a completed record lands between the check and the publication
        return real_open(path, mode, **kw)
    monkeypatch.setattr(M.gzip, "open", delayed_open)
    with pytest.raises(SystemExit, match="never overwritten"):
        M.write_gz(str(target), {"x": 1})
    assert target.read_bytes() == competitor and sorted(os.listdir(tmp_path)) == ["r.json.gz"]      # unchanged, and no temporary file left
    monkeypatch.setattr(M.gzip, "open", real_open)
    fresh = tmp_path / "s.json.gz"
    digest = M.write_gz(str(fresh), {"x": 2})
    with real_open(fresh, "rb") as fh:
        blob = fh.read()
    import hashlib
    assert hashlib.sha256(blob).hexdigest() == digest and sorted(os.listdir(tmp_path)) == ["r.json.gz", "s.json.gz"]


def _track(birth, obs, smoothed_lon_shift=0.0):
    steps = [o[0] for o in obs]
    return {"birth": birth, "steps": steps, "raw_lat": [o[1] for o in obs], "raw_lon": [o[2] for o in obs], "time": [100 + 0.25 * s for s in steps],
            "lat": [o[1] for o in obs], "lon": [o[2] + smoothed_lon_shift for o in obs], "n_points": [1] * len(obs), "region_sha256": ["r"] * len(obs)}


def test_pairing_requires_a_mutual_unique_best_match_and_divergence_reads_raw_positions_at_the_band_edge():
    C = _load("pilot_margin_compare")
    a, b, c, d = (1, 10.0, 30.0), (2, 10.0, 28.5), (3, 10.0, 27.0), (4, 10.0, 25.5)
    n1, n2 = _track("0:0", [a, b]), _track("0:1", [a, b, c])
    m = _track("0:0", [a, b, c, d])
    p = C.pair([n1, n2], [m])
    assert p["mutual"] == [(1, 0, 3)] and [x["native"] for x in p["ambiguous_native"]] == [0] and p["ambiguous_margin"] == []   # n1's best is m, but m's best is n2
    # the divergence longitude is the raw last common longitude, not the smoothed one
    shared = [(0, 12.0, 39.0), (1, 12.0, 38.5)]
    nat = _track("0:0", shared + [(2, 12.0, 37.0)], smoothed_lon_shift=-2.0)
    mar = _track("0:1", shared + [(2, 11.0, 36.0)], smoothed_lon_shift=-2.0)
    d1 = C.divergence(nat, mar)
    assert d1["divergence_lon"] == 38.5 and d1["within_band"] is True and nat["lon"][1] == 36.5
    for lon, inside in ((37.5, False), (38.0, True), (38.5, True)):
        nat = _track("0:0", [(0, 12.0, lon), (1, 12.0, lon - 1.5)])
        mar = _track("0:1", [(0, 12.0, lon), (1, 11.0, lon - 2.5)])
        assert C.divergence(nat, mar)["within_band"] is inside, lon
        conv = C.divergence(_track("0:0", [(0, 12.0, lon), (1, 12.0, 30.0)]), _track("0:1", [(0, 13.0, lon), (1, 12.0, 30.0)]))
        assert conv["kind"] == "converging" and conv["within_band"] == {"native_birth": inside, "margin_birth": inside}, lon
        s = C.summarize([_track("0:0", [(5, 12.0, lon)])], [])
        assert s["unmatched_native"][0]["birth_within_band"] is inside, lon
    s = C.summarize([_track("0:0", [(0, 12.0, 39.0), (1, 12.0, 30.0)]), _track("0:1", [(0, 12.0, 36.0), (2, 12.0, 20.0)])],
                    [_track("0:0", [(0, 13.0, 37.0), (1, 12.0, 30.0)]), _track("0:1", [(0, 12.0, 35.0), (2, 12.0, 20.0)])])
    assert s["converging_birth_longitude_bins"] == {"native": {"[35, 40)": 2}, "margin": {"[35, 40)": 2}} and sum(s["converging_birth_longitude_bins"]["native"].values()) == s["band"]["converging_pairs"]


def test_the_comparison_pairs_identical_histories_by_count_and_the_rest_by_mutual_best_overlap():
    C = _load("pilot_margin_compare")
    a = [(0, 10.0, 39.0), (1, 10.0, 37.5), (2, 10.0, 36.0), (3, 10.0, 34.5)]
    b = [(0, 10.0, 39.0), (1, 10.0, 37.5), (2, 9.0, 35.0), (3, 9.0, 33.5)]          # diverges after two shared observations at 37.5 E
    c = [(5, 20.0, 20.0), (6, 20.0, 18.5)]                                           # identical on both sides, twice in native, once in margin
    d = [(8, 5.0, 30.0), (9, 5.0, 28.5), (10, 5.0, 27.0)]
    e = [(7, 6.0, 33.0), (8, 6.0, 31.5), (9, 5.0, 28.5), (10, 5.0, 27.0)]            # converging: different start, shared tail with d
    f = [(12, 0.0, -50.0)]                                                            # unmatched native
    native = [_track("0:1", a), _track("5:0", c), _track("5:0", c), _track("8:2", d), _track("12:0", f)]
    margin = [_track("0:3", b), _track("5:1", c), _track("7:2", e)]
    s = C.summarize(native, margin)
    assert s["counts"] == {"native_finished": 5, "margin_finished": 3, "identical_pairs": 1, "mutual_changed_pairs": 2, "ambiguous_native": 0, "ambiguous_margin": 0, "unmatched_native": 2, "unmatched_margin": 0}
    assert s["content_and_order"]["raw_histories"]["equal_as_multiset"] is False and s["content_and_order"]["raw_histories"]["shared_with_multiplicity"] == 1
    div = s["diverging_pairs"][0]
    assert div["common_prefix_observations"] == 2 and div["divergence_lon"] == 37.5 and div["within_band"] is False
    assert div["first_differing_native"]["lon"] == 36.0 and div["first_differing_margin"]["lon"] == 35.0 and div["native_birth"] == "0:1" and div["margin_birth"] == "0:3"
    conv = s["converging_pairs"][0]
    assert conv["kind"] == "converging" and conv["first_shared"]["step"] == 9 and conv["divergence_lon"] == {"native_birth": 30.0, "margin_birth": 33.0}
    assert [u["birth_lon"] for u in s["unmatched_native"]] == [20.0, -50.0]                       # the second copy of c, and f
    assert s["band"]["diverging_pairs_west_of_band"] == 1 and s["band"]["unmatched_native_births_west"] == 2
    assert C.bins([39.0, 37.5, -50.0]) == {"[-50, -45)": 1, "[35, 40)": 2}


def test_a_tied_best_match_is_ambiguous_and_never_forced():
    C = _load("pilot_margin_compare")
    shared = [(1, 10.0, 30.0), (2, 10.0, 28.5)]
    n1 = _track("0:0", [(0, 11.0, 32.0)] + shared)
    n2 = _track("0:1", [(0, 9.0, 32.0)] + shared)
    m = _track("0:0", [(0, 10.0, 31.0)] + shared)
    p = C.pair([n1, n2], [m])
    assert p["mutual"] == [] and p["unmatched_native"] == [] and p["unmatched_margin"] == []
    assert len(p["ambiguous_native"]) == 2 and p["ambiguous_margin"][0]["candidates"] == [0, 1] and p["ambiguous_margin"][0]["overlap"] == 2


def test_the_comparison_refuses_a_failed_native_gate_and_mismatched_inputs(tmp_path):
    C = _load("pilot_margin_compare")
    import gzip, json
    def gz(name, payload):
        p = tmp_path / name
        with gzip.open(p, "wb") as fh:
            fh.write(json.dumps(payload).encode())
        return str(p)
    base = {"year": 1990, "runs": {"B": "x"}, "thresholds": {"coarse": 1.0}, "script_sha256": "s", "native_bind": None, "finished": []}
    native = gz("n.json.gz", {**base, "preparation": "native", "archive_comparison": {"passed": False}})
    margin = gz("m.json.gz", {**base, "preparation": "margin", "archive_comparison": {"passed": False}})
    with pytest.raises(SystemExit, match="did not reproduce"):
        C.main(["--native", native, "--margin", margin, "--out", str(tmp_path / "o.json")])
    native_ok = gz("n2.json.gz", {**base, "preparation": "native", "archive_comparison": {"passed": True}})
    other_year = gz("m2.json.gz", {**base, "year": 2006, "preparation": "margin", "archive_comparison": {"passed": False}})
    with pytest.raises(SystemExit, match="not a native and a margin replay of one season"):
        C.main(["--native", native_ok, "--margin", other_year, "--out", str(tmp_path / "o2.json")])


def test_the_contrast_gates_admit_only_the_declared_pair_and_the_band_is_a_parameter(tmp_path):
    C = _load("pilot_margin_compare")
    import gzip, json
    def gz(name, payload):
        p = tmp_path / name
        with gzip.open(p, "wb") as fh:
            fh.write(json.dumps(payload).encode())
        return str(p)
    runs40 = {"B": {"dir": "b", "record_sha256": "r", "tracks_sha256": "t", "case_sha256": "c", "control_domain": {"lat": [-35.0, 35.0], "lon": [-140.0, 40.0]}},
              "wide": {"case_sha256": "w", "box": {"lat": [-39.0, 39.0], "lon": [-144.0, 44.0]}}}
    runs60 = {"B": {"dir": "d", "record_sha256": "r2", "tracks_sha256": "t2", "case_sha256": "c2", "control_domain": {"lat": [-35.0, 35.0], "lon": [-140.0, 60.0]}},
              "wide": {"case_sha256": "w2", "box": {"lat": [-39.0, 39.0], "lon": [-144.0, 64.0]}}}
    released = {"coarse": 1.0, "fine": 2.0, "from": "the control run's record"}
    recomputed = {"coarse": 0.9, "fine": 1.9, "from": "the calibration audit's smooth_then_crop order", "calibration_audit": {"sha256": "a" * 64}, "transfer": False}
    base = {"year": 1990, "script_sha256": "s", "native_bind": None, "finished": [], "margin_edges": "all", "archive_comparison": {"passed": False}}
    m_rel = gz("m_rel.json.gz", {**base, "preparation": "margin", "runs": runs40, "thresholds": released})
    m_rec = gz("m_rec.json.gz", {**base, "preparation": "margin", "runs": runs40, "thresholds": recomputed})
    m60 = gz("m60.json.gz", {**base, "preparation": "margin", "runs": runs60, "thresholds": recomputed})
    n60 = gz("n60.json.gz", {**base, "preparation": "native", "runs": runs60, "thresholds": released, "archive_comparison": {"passed": True}})
    out = tmp_path / "cal.json"
    assert C.main(["--native", m_rel, "--margin", m_rec, "--out", str(out), "--contrast", "calibration"]) == 0
    r = json.loads(out.read_text())
    assert r["contrast"] == "calibration" and r["band_west_edge_lon"] == 38.0 and "released" in r["roles"]["native"] and "recomputed" in r["roles"]["margin"]
    assert r["thresholds"] == {"native": released, "margin": recomputed} and r["runs"] == {"native": runs40, "margin": runs40}
    with pytest.raises(SystemExit, match="calibration contrast"):                                     # the same pair is not a calibration contrast
        C.main(["--native", m_rel, "--margin", m_rel, "--out", str(tmp_path / "x1.json"), "--contrast", "calibration"])
    with pytest.raises(SystemExit, match="calibration contrast"):                                     # a native reference is not either
        C.main(["--native", n60, "--margin", m60, "--out", str(tmp_path / "x2.json"), "--contrast", "calibration"])
    out2 = tmp_path / "ext.json"
    def digest_of(path):
        import hashlib
        with gzip.open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    verification = tmp_path / "verification.json"
    verification.write_text(json.dumps({"generated_by": "scripts/pilot_extent_verification.py", "script_sha256": "v", "year": 1990, "steps": 1460,
                                        "inputs": {"replay_a": {"sha256": digest_of(m_rec)}, "replay_b": {"sha256": digest_of(m60)}},
                                        "declared": {"both_margin": True, "one_applied_pair": True, "one_calibration_audit": True, "same_tracker_flags": True, "same_year": True, "same_inputs": True, "same_climatology": True, "boxes": {"same_latitudes_and_western_edge": True}},
                                        "raw_shared_cells_differing": {k: 0 for k in ("u_c", "v_c", "currv_anom_c", "advcurrv_anom_c", "u", "v", "currv_anom")},
                                        "prepared_control_a_cells_differing": {k: 0 for k in ("coarse_curvature", "coarse_advection", "coarse_zonal_wind_smoothed", "fine_curvature", "fine_zonal_wind_smoothed", "fine_meridional_wind_smoothed")},
                                        "extent_is_the_only_difference": True}))
    assert C.main(["--native", m_rec, "--margin", m60, "--out", str(out2), "--contrast", "extent", "--verification", str(verification)]) == 0
    r2 = json.loads(out2.read_text())
    assert r2["contrast"] == "extent" and r2["runs"]["native"] == runs40 and r2["runs"]["margin"] == runs60
    with pytest.raises(SystemExit, match="extent contrast"):                                          # one control is not an extent contrast
        C.main(["--native", m_rec, "--margin", m_rec, "--out", str(tmp_path / "x3.json"), "--contrast", "extent"])
    with pytest.raises(SystemExit, match="extent contrast"):                                          # two pairs are not either
        C.main(["--native", m_rel, "--margin", m60, "--out", str(tmp_path / "x4.json"), "--contrast", "extent"])
    out3 = tmp_path / "pc.json"
    assert C.main(["--native", n60, "--margin", m60, "--out", str(out3), "--contrast", "preparation_and_calibration", "--band-west-edge", "58"]) == 0
    r3 = json.loads(out3.read_text())
    assert r3["contrast"] == "preparation_and_calibration" and r3["band_west_edge_lon"] == 58.0
    with pytest.raises(SystemExit, match="do not share"):                                             # under the default contrast the pairs must agree
        C.main(["--native", n60, "--margin", m60, "--out", str(tmp_path / "x5.json")])
    a = _track("0:0", [(0, 10.0, 57.0), (1, 10.0, 56.0)])
    b = _track("0:0", [(0, 10.0, 57.0), (1, 10.5, 56.0)])
    assert C.divergence(a, b, 58.0)["within_band"] is False and C.divergence(a, b, 56.0)["within_band"] is True
    assert C.summarize([a], [b], 56.0)["band"]["diverging_pairs_within_band"] == 1 and C.summarize([a], [b], 58.0)["band"]["diverging_pairs_within_band"] == 0
    assert C.bins([44.0, 59.5, 60.0]) == {"[40, 45)": 1, "[55, 60)": 1, "[60, 65)": 1}                  # bins reach the 60 E domain
    with pytest.raises(SystemExit, match="calibration contrast's reference applies the control"):     # the roles reversed
        C.main(["--native", m_rec, "--margin", m_rel, "--out", str(tmp_path / "x6.json"), "--contrast", "calibration"])
    m60_ctl = gz("m60_ctl.json.gz", {**base, "preparation": "margin", "runs": runs60, "thresholds": {"coarse": 0.9, "fine": 1.9, "from": "the control run's record"}})
    with pytest.raises(SystemExit, match="from one audit"):                                           # the same numbers, not from the audit
        C.main(["--native", m_rec, "--margin", m60_ctl, "--out", str(tmp_path / "x7.json"), "--contrast", "extent"])
    m60_other = gz("m60_other.json.gz", {**base, "preparation": "margin", "runs": runs60, "thresholds": {**recomputed, "calibration_audit": {"sha256": "b" * 64}}})
    with pytest.raises(SystemExit, match="from one audit"):                                           # another audit with the same numbers
        C.main(["--native", m_rec, "--margin", m60_other, "--out", str(tmp_path / "x8.json"), "--contrast", "extent"])
    with pytest.raises(SystemExit, match="preparation_and_calibration contrast's candidate applies the audit"):
        C.main(["--native", n60, "--margin", m60_ctl, "--out", str(tmp_path / "x9.json"), "--contrast", "preparation_and_calibration"])
    n40 = gz("n40.json.gz", {**base, "preparation": "native", "runs": runs40, "thresholds": released, "archive_comparison": {"passed": True}})
    m40_audit_same = gz("m40_as.json.gz", {**base, "preparation": "margin", "runs": runs40, "thresholds": {**recomputed, "coarse": 1.0, "fine": 2.0}})
    with pytest.raises(SystemExit, match="do not share their runs and thresholds"):                   # the same numbers, but the candidate's come from an audit
        C.main(["--native", n40, "--margin", m40_audit_same, "--out", str(tmp_path / "x10.json")])
    n40_audit = gz("n40_a.json.gz", {**base, "preparation": "native", "runs": runs40, "thresholds": {**recomputed, "coarse": 1.0, "fine": 2.0}, "archive_comparison": {"passed": True}})
    with pytest.raises(SystemExit, match="preparation contrast's reference applies the control"):     # a reference under an audit's pair
        C.main(["--native", n40_audit, "--margin", m_rel, "--out", str(tmp_path / "x11.json")])
    with pytest.raises(SystemExit, match="which must differ"):                                        # an audit candidate with the reference's own numbers
        C.main(["--native", n40, "--margin", m40_audit_same, "--out", str(tmp_path / "x12.json"), "--contrast", "preparation_and_calibration"])
    m_rec_nodigest = gz("m_rec_nd.json.gz", {**base, "preparation": "margin", "runs": runs40, "thresholds": {**recomputed, "calibration_audit": {"path": "audit-A.json"}}})
    m60_nodigest = gz("m60_nd.json.gz", {**base, "preparation": "margin", "runs": runs60, "thresholds": {**recomputed, "calibration_audit": {"path": "audit-B.json"}}})
    with pytest.raises(SystemExit, match="calibration provenance"):                                   # an audit without a digest is not a provenance
        C.main(["--native", m_rec_nodigest, "--margin", m60_nodigest, "--out", str(tmp_path / "x13.json"), "--contrast", "extent"])
    with pytest.raises(SystemExit, match="calibration provenance"):
        C.main(["--native", m_rel, "--margin", m_rec_nodigest, "--out", str(tmp_path / "x14.json"), "--contrast", "calibration"])
    assert C.provenance({"thresholds": recomputed}) == ("audit", "a" * 64) and C.provenance({"thresholds": released}) == ("control", None)
    with pytest.raises(SystemExit, match="calibration provenance"):
        C.provenance({"thresholds": {**recomputed, "calibration_audit": "a" * 64}})                   # a bare string is malformed, not a control pair


def test_an_extent_comparison_needs_a_passing_verification_bound_to_its_two_replays_and_every_contrast_refuses_differing_flags(tmp_path):
    C = _load("pilot_margin_compare")
    import gzip, hashlib, json
    def gz(name, payload):
        blob = json.dumps(payload).encode()
        p = tmp_path / name
        with gzip.open(p, "wb") as fh:
            fh.write(blob)
        return str(p), hashlib.sha256(blob).hexdigest()
    runs40 = {"B": {"dir": "b", "record_sha256": "r", "tracks_sha256": "t", "case_sha256": "c", "control_domain": {"lat": [-35.0, 35.0], "lon": [-140.0, 40.0]}},
              "wide": {"case_sha256": "w", "box": {"lat": [-39.0, 39.0], "lon": [-144.0, 44.0]}}}
    runs60 = {"B": {"dir": "d", "record_sha256": "r2", "tracks_sha256": "t2", "case_sha256": "c2", "control_domain": {"lat": [-35.0, 35.0], "lon": [-140.0, 60.0]}},
              "wide": {"case_sha256": "w2", "box": {"lat": [-39.0, 39.0], "lon": [-144.0, 64.0]}}}
    recomputed = {"coarse": 0.9, "fine": 1.9, "from": "the calibration audit's smooth_then_crop order", "calibration_audit": {"sha256": "a" * 64}, "transfer": False}
    base = {"year": 1990, "script_sha256": "s", "native_bind": None, "finished": [], "margin_edges": "all", "archive_comparison": {"passed": False}, "tracker_flags": {"exclusive": False, "absorb": False}}
    m40, sha40 = gz("m40.json.gz", {**base, "preparation": "margin", "runs": runs40, "thresholds": recomputed})
    m60, sha60 = gz("m60.json.gz", {**base, "preparation": "margin", "runs": runs60, "thresholds": {**recomputed, "transfer": True}})
    zeros7 = {k: 0 for k in ("u_c", "v_c", "currv_anom_c", "advcurrv_anom_c", "u", "v", "currv_anom")}
    zeros6 = {k: 0 for k in ("coarse_curvature", "coarse_advection", "coarse_zonal_wind_smoothed", "fine_curvature", "fine_zonal_wind_smoothed", "fine_meridional_wind_smoothed")}
    good = {"generated_by": "scripts/pilot_extent_verification.py", "script_sha256": "v", "year": 1990, "steps": 1460,
            "inputs": {"replay_a": {"sha256": sha40}, "replay_b": {"sha256": sha60}},
            "declared": {"both_margin": True, "one_applied_pair": True, "one_calibration_audit": True, "same_tracker_flags": True, "same_year": True, "same_inputs": True, "same_climatology": True,
                         "boxes": {"same_latitudes_and_western_edge": True}},
            "raw_shared_cells_differing": zeros7, "prepared_control_a_cells_differing": zeros6, "extent_is_the_only_difference": True}
    def ver(name, payload):
        p = tmp_path / name
        p.write_text(json.dumps(payload))
        return str(p)
    args = ["--native", m40, "--margin", m60, "--contrast", "extent"]
    with pytest.raises(SystemExit, match="needs its verification record"):
        C.main(args + ["--out", str(tmp_path / "v1.json")])
    with pytest.raises(SystemExit, match="not bound to these two replays"):                           # a stale or wrong-replay verification
        C.main(args + ["--out", str(tmp_path / "v2.json"), "--verification", ver("stale.json", {**good, "inputs": {"replay_a": {"sha256": sha40}, "replay_b": {"sha256": "0" * 64}}})])
    with pytest.raises(SystemExit, match="did not find the extent the only difference"):              # a failed verification, whatever its flag says
        C.main(args + ["--out", str(tmp_path / "v3.json"), "--verification", ver("failed.json", {**good, "raw_shared_cells_differing": {**zeros7, "u": 1}})])
    with pytest.raises(SystemExit, match="did not find the extent the only difference"):
        C.main(args + ["--out", str(tmp_path / "v4.json"), "--verification", ver("declared.json", {**good, "declared": {**good["declared"], "same_climatology": False}})])
    with pytest.raises(SystemExit, match="did not find the extent the only difference"):              # a true flag over a nonzero count is not read
        C.main(args + ["--out", str(tmp_path / "v5.json"), "--verification", ver("flagged.json", {**good, "prepared_control_a_cells_differing": {**zeros6, "fine_curvature": 3}, "extent_is_the_only_difference": True})])
    with pytest.raises(SystemExit, match="not the verification instrument's record"):
        C.main(args + ["--out", str(tmp_path / "v6.json"), "--verification", ver("other.json", {**good, "generated_by": "scripts/other.py"})])
    out = tmp_path / "ok.json"
    vpath = ver("good.json", good)
    assert C.main(args + ["--out", str(out), "--verification", vpath]) == 0
    rec = json.loads(out.read_text())
    assert rec["verification"]["path"] == vpath and rec["verification"]["sha256"] == hashlib.sha256((tmp_path / "good.json").read_bytes()).hexdigest()
    assert rec["verification"]["replays"] == {"a": sha40, "b": sha60} and rec["verification"]["script_sha256"] == "v" and rec["tracker_flags"] == base["tracker_flags"]
    m60_flag, _ = gz("m60_flag.json.gz", {**base, "preparation": "margin", "runs": runs60, "thresholds": {**recomputed, "transfer": True}, "tracker_flags": {"exclusive": False, "absorb": True}})
    with pytest.raises(SystemExit, match="tracker flags"):                                            # differing flags refused before any verification is read
        C.main(["--native", m40, "--margin", m60_flag, "--contrast", "extent", "--out", str(tmp_path / "v7.json"), "--verification", vpath])
    m40_flag, _ = gz("m40_flag.json.gz", {**base, "preparation": "margin", "runs": runs40, "thresholds": {"coarse": 1.0, "fine": 2.0, "from": "the control run's record"}, "tracker_flags": {"exclusive": True, "absorb": False}})
    with pytest.raises(SystemExit, match="tracker flags"):                                            # and under every other contrast
        C.main(["--native", m40_flag, "--margin", m40, "--contrast", "calibration", "--out", str(tmp_path / "v8.json")])
    with pytest.raises(SystemExit, match="not bound to these two replays"):                           # the reference's digest wrong, the candidate's right
        C.main(args + ["--out", str(tmp_path / "v9.json"), "--verification", ver("stale_a.json", {**good, "inputs": {"replay_a": {"sha256": "0" * 64}, "replay_b": {"sha256": sha60}}})])
    with pytest.raises(SystemExit, match="not bound to these two replays"):                           # another season's verification of the same replays
        C.main(args + ["--out", str(tmp_path / "v10.json"), "--verification", ver("stale_year.json", {**good, "year": 2006})])
    padded = {**good, "raw_shared_cells_differing": {f"other_{i}": 0 for i in range(7)}}
    with pytest.raises(SystemExit, match="by name"):                                                  # thirteen zeros under other names are not the counts
        C.main(args + ["--out", str(tmp_path / "v11.json"), "--verification", ver("padded.json", padded)])
    short = {**good, "raw_shared_cells_differing": {k: 0 for k in list(zeros7)[:6]}}
    with pytest.raises(SystemExit, match="by name"):                                                  # a missing field is not a zero
        C.main(args + ["--out", str(tmp_path / "v12.json"), "--verification", ver("short.json", short)])
    padded_prepared = {**good, "prepared_control_a_cells_differing": {f"other_{i}": 0 for i in range(6)}}
    with pytest.raises(SystemExit, match="by name"):                                                  # the same for the six prepared fields
        C.main(args + ["--out", str(tmp_path / "v12b.json"), "--verification", ver("padded_prepared.json", padded_prepared)])
    short_prepared = {**good, "prepared_control_a_cells_differing": {k: 0 for k in list(zeros6)[:5]}}
    with pytest.raises(SystemExit, match="by name"):
        C.main(args + ["--out", str(tmp_path / "v12c.json"), "--verification", ver("short_prepared.json", short_prepared)])
    for i, key in enumerate(("both_margin", "one_applied_pair", "one_calibration_audit", "same_tracker_flags", "same_year", "same_inputs", "same_climatology")):
        with pytest.raises(SystemExit, match="did not find the extent the only difference"):          # each declared equality on its own
            C.main(args + ["--out", str(tmp_path / f"v13_{i}.json"), "--verification", ver(f"decl_{i}.json", {**good, "declared": {**good["declared"], key: False}})])
    with pytest.raises(SystemExit, match="did not find the extent the only difference"):
        C.main(args + ["--out", str(tmp_path / "v14.json"), "--verification", ver("boxes.json", {**good, "declared": {**good["declared"], "boxes": {"same_latitudes_and_western_edge": False}}})])


def test_the_calibration_audits_contrast_admits_a_transferred_reference_against_the_domains_own_pair_only(tmp_path):
    C = _load("pilot_margin_compare")
    import gzip, json
    def gz(name, payload):
        p = tmp_path / name
        with gzip.open(p, "wb") as fh:
            fh.write(json.dumps(payload).encode())
        return str(p)
    runs60 = {"B": {"dir": "d", "record_sha256": "r2", "tracks_sha256": "t2", "case_sha256": "c2", "control_domain": {"lat": [-35.0, 35.0], "lon": [-140.0, 60.0]}},
              "wide": {"case_sha256": "w2", "box": {"lat": [-39.0, 39.0], "lon": [-144.0, 64.0]}}}
    base = {"year": 1990, "script_sha256": "s", "native_bind": None, "finished": [], "margin_edges": "all", "archive_comparison": {"passed": False}, "tracker_flags": {"exclusive": False, "absorb": False}}
    transferred = {"coarse": 0.9, "fine": 1.9, "from": "the calibration audit's smooth_then_crop order", "transfer": True, "domain_calibration": False,
                   "calibration_audit": {"sha256": "a" * 64, "domain": {"lat_range": [-35.0, 35.0], "lon_range": [-140.0, 40.0]}}}
    own = {"coarse": 0.8, "fine": 1.8, "from": "the calibration audit's smooth_then_crop order", "transfer": False, "domain_calibration": True,
           "calibration_audit": {"sha256": "b" * 64, "domain": {"lat_range": [-35.0, 35.0], "lon_range": [-140.0, 60.0]}}}
    ref = gz("ref.json.gz", {**base, "preparation": "margin", "runs": runs60, "thresholds": transferred})
    cand = gz("cand.json.gz", {**base, "preparation": "margin", "runs": runs60, "thresholds": own})
    out = tmp_path / "ca.json"
    assert C.main(["--native", ref, "--margin", cand, "--out", str(out), "--contrast", "calibration_audits", "--band-west-edge", "58"]) == 0
    r = json.loads(out.read_text())
    assert r["contrast"] == "calibration_audits" and "transferred" in r["roles"]["native"] and "own" in r["roles"]["margin"] and r["band_west_edge_lon"] == 58.0
    with pytest.raises(SystemExit, match="calibration_audits contrast"):                              # reversed
        C.main(["--native", cand, "--margin", ref, "--out", str(tmp_path / "x1.json"), "--contrast", "calibration_audits"])
    foreign = gz("foreign.json.gz", {**base, "preparation": "margin", "runs": runs60, "thresholds": {**own, "calibration_audit": {"sha256": "b" * 64, "domain": {"lat_range": [-35.0, 35.0], "lon_range": [-140.0, 40.0]}}}})
    with pytest.raises(SystemExit, match="calibration_audits contrast"):                              # the candidate's audit is not of the control's domain
        C.main(["--native", ref, "--margin", foreign, "--out", str(tmp_path / "x2.json"), "--contrast", "calibration_audits"])
    same_audit = gz("same.json.gz", {**base, "preparation": "margin", "runs": runs60, "thresholds": {**own, "calibration_audit": {**own["calibration_audit"], "sha256": "a" * 64}}})
    with pytest.raises(SystemExit, match="calibration_audits contrast"):                              # one audit on both sides
        C.main(["--native", ref, "--margin", same_audit, "--out", str(tmp_path / "x3.json"), "--contrast", "calibration_audits"])
    ctl = gz("ctl.json.gz", {**base, "preparation": "margin", "runs": runs60, "thresholds": {"coarse": 1.0, "fine": 2.0, "from": "the control run's record"}})
    with pytest.raises(SystemExit, match="calibration_audits contrast"):                              # a reference under the control's pair belongs to the calibration contrast
        C.main(["--native", ctl, "--margin", cand, "--out", str(tmp_path / "x4.json"), "--contrast", "calibration_audits"])
    CH = _load("pilot_margin_cohort")
    mar = {"preparation": "margin", "archive_comparison": {"passed": False}}
    assert CH.admit("calibration_audits", mar, mar) == "calibration_audits"


def test_the_calibration_audits_contrast_refuses_a_reference_whose_audit_is_of_the_controls_own_domain(tmp_path):
    C = _load("pilot_margin_compare")
    import gzip, json
    def gz(name, payload):
        p = tmp_path / name
        with gzip.open(p, "wb") as fh:
            fh.write(json.dumps(payload).encode())
        return str(p)
    runs60 = {"B": {"dir": "d", "record_sha256": "r2", "tracks_sha256": "t2", "case_sha256": "c2", "control_domain": {"lat": [-35.0, 35.0], "lon": [-140.0, 60.0]}},
              "wide": {"case_sha256": "w2", "box": {"lat": [-39.0, 39.0], "lon": [-144.0, 64.0]}}}
    base = {"year": 1990, "script_sha256": "s", "native_bind": None, "finished": [], "margin_edges": "all", "archive_comparison": {"passed": False}, "tracker_flags": {"exclusive": False, "absorb": False}}
    own_domain = {"lat_range": [-35.0, 35.0], "lon_range": [-140.0, 60.0]}
    ref = gz("ref.json.gz", {**base, "preparation": "margin", "runs": runs60, "thresholds": {"coarse": 0.9, "fine": 1.9, "from": "the calibration audit's smooth_then_crop order", "transfer": True, "domain_calibration": False, "calibration_audit": {"sha256": "a" * 64, "domain": own_domain}}})
    cand = gz("cand.json.gz", {**base, "preparation": "margin", "runs": runs60, "thresholds": {"coarse": 0.8, "fine": 1.8, "from": "the calibration audit's smooth_then_crop order", "transfer": False, "domain_calibration": True, "calibration_audit": {"sha256": "b" * 64, "domain": own_domain}}})
    with pytest.raises(SystemExit, match="reference applies a pair transferred"):                     # a transfer flag over an audit of the control's own domain is not a transfer
        C.main(["--native", ref, "--margin", cand, "--out", str(tmp_path / "x.json"), "--contrast", "calibration_audits"])


def test_the_calibration_audits_contrast_refuses_a_reference_without_an_audit_domain_and_checks_each_side_against_its_audit_file(tmp_path):
    C = _load("pilot_margin_compare")
    import gzip, hashlib, json
    def gz(name, payload):
        p = tmp_path / name
        with gzip.open(p, "wb") as fh:
            fh.write(json.dumps(payload).encode())
        return str(p)
    def audit_file(name, coarse, fine):
        p = tmp_path / name
        p.write_text(json.dumps({"generated_by": "scripts/pilot_alledge_calibration.py",
                                 "orders": {"crop_then_smooth": {"coarse": {"threshold": 9.0}, "fine": {"threshold": 9.5}}, "smooth_then_crop": {"coarse": {"threshold": coarse}, "fine": {"threshold": fine}}}}))
        return str(p), hashlib.sha256(p.read_bytes()).hexdigest()
    runs60 = {"B": {"dir": "d", "record_sha256": "r2", "tracks_sha256": "t2", "case_sha256": "c2", "control_domain": {"lat": [-35.0, 35.0], "lon": [-140.0, 60.0]}},
              "wide": {"case_sha256": "w2", "box": {"lat": [-39.0, 39.0], "lon": [-144.0, 64.0]}}}
    base = {"year": 1990, "script_sha256": "s", "native_bind": None, "finished": [], "margin_edges": "all", "archive_comparison": {"passed": False}, "tracker_flags": {"exclusive": False, "absorb": False}}
    ref_path, ref_sha = audit_file("audit40.json", 0.9, 1.9)
    own_path, own_sha = audit_file("audit60.json", 0.8, 1.8)
    transferred = {"coarse": 0.9, "fine": 1.9, "from": "the calibration audit's smooth_then_crop order", "transfer": True, "domain_calibration": False,
                   "calibration_audit": {"sha256": ref_sha, "path": ref_path, "domain": {"lat_range": [-35.0, 35.0], "lon_range": [-140.0, 40.0]}}}
    own = {"coarse": 0.8, "fine": 1.8, "from": "the calibration audit's smooth_then_crop order", "transfer": False, "domain_calibration": True,
           "calibration_audit": {"sha256": own_sha, "path": own_path, "domain": {"lat_range": [-35.0, 35.0], "lon_range": [-140.0, 60.0]}}}
    ref = gz("ref.json.gz", {**base, "preparation": "margin", "runs": runs60, "thresholds": transferred})
    cand = gz("cand.json.gz", {**base, "preparation": "margin", "runs": runs60, "thresholds": own})
    out = tmp_path / "ok.json"
    assert C.main(["--native", ref, "--margin", cand, "--out", str(out), "--contrast", "calibration_audits"]) == 0
    audits = json.loads(out.read_text())["audits"]
    assert audits["native"]["found"] is True and audits["margin"]["found"] is True and audits["margin"]["applied_pair_is_the_audits_smooth_then_crop_pair"] is True
    nodom = gz("nodom.json.gz", {**base, "preparation": "margin", "runs": runs60, "thresholds": {**transferred, "calibration_audit": {"sha256": ref_sha, "path": ref_path}}})
    with pytest.raises(SystemExit, match="records no audit domain"):                                 # a reference without a recorded audit domain is not read as a transfer
        C.main(["--native", nodom, "--margin", cand, "--out", str(tmp_path / "x1.json"), "--contrast", "calibration_audits"])
    wrong = gz("wrong.json.gz", {**base, "preparation": "margin", "runs": runs60, "thresholds": {**own, "coarse": 0.7}})
    with pytest.raises(SystemExit, match="not the audit's smooth_then_crop pair"):                   # the applied pair is not what the named audit gives
        C.main(["--native", ref, "--margin", wrong, "--out", str(tmp_path / "x2.json"), "--contrast", "calibration_audits"])
    forged = gz("forged.json.gz", {**base, "preparation": "margin", "runs": runs60, "thresholds": {**own, "calibration_audit": {**own["calibration_audit"], "sha256": "b" * 64}}})
    with pytest.raises(SystemExit, match="the replay recorded"):                                      # the file at the recorded path is not the recorded audit
        C.main(["--native", ref, "--margin", forged, "--out", str(tmp_path / "x3.json"), "--contrast", "calibration_audits"])
    absent = gz("absent.json.gz", {**base, "preparation": "margin", "runs": runs60, "thresholds": {**own, "calibration_audit": {**own["calibration_audit"], "path": str(tmp_path / "gone.json")}}})
    out2 = tmp_path / "absent.json"
    assert C.main(["--native", ref, "--margin", absent, "--out", str(out2), "--contrast", "calibration_audits"]) == 0
    a = json.loads(out2.read_text())["audits"]["margin"]
    assert a["found"] is False and "applied_pair_is_the_audits_smooth_then_crop_pair" not in a    # an absent file is recorded as not found and claims nothing
