"""The visual audit's case selection is the predeclared rule and nothing else: at most six
longest candidate prefixes, three longest unpaired eastern starts, three identical
near-edge comparison cases, deterministic order, identities carried from the report."""
import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def _load():
    spec = importlib.util.spec_from_file_location("pilot_visual_audit_cases_under_test", os.path.join(ROOT, "scripts", "pilot_visual_audit_cases.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _attr(start_lon, start_time, observations, censored=False, month=8, lat=10.0):
    return {"start_lon": start_lon, "start_lat": lat, "start_time": start_time, "observations": observations, "potentially_censored_start": censored, "start_month": month}


def _report():
    attrs_a = [_attr(35.0, 100.0, 10), _attr(38.0, 110.0, 12), _attr(20.0, 120.0, 9), _attr(37.0, 130.0, 8)] + [_attr(30.0, 140.0 + i, 10) for i in range(8)] \
              + [_attr(36.0, 300.0, 9, lat=8.0), _attr(37.0, 310.0, 20, lat=5.0), _attr(30.0, 320.0, 10)]
    attrs_b = [_attr(37.0, 99.0, 14), _attr(38.0, 110.0, 12), _attr(22.0, 119.5, 11), _attr(37.0, 130.0, 8)] + [_attr(33.0, 139.0 + i, 10 + i) for i in range(8)] \
              + [_attr(55.0, 200.0, 7, True), _attr(50.0, 210.0, 9), _attr(45.0, 220.0, 9), _attr(58.0, 230.0, 5, True), _attr(30.0, 240.0, 20)] \
              + [_attr(36.0, 300.0, 9, lat=8.0), _attr(37.0, 310.0, 20, lat=6.0), _attr(45.0, 320.0, 12)]
    pairs = [{"a": 0, "b": 0, "class": "prefix changed", "candidate_prefix_steps": 4, "candidate_prefix_degrees": 2.0, "start_shift_days": -1.0},
             {"a": 1, "b": 1, "class": "identical", "candidate_prefix_steps": 0, "candidate_prefix_degrees": 0.0, "start_shift_days": 0.0},
             {"a": 2, "b": 2, "class": "prefix changed", "candidate_prefix_steps": 2, "candidate_prefix_degrees": 2.0, "start_shift_days": -0.5},
             {"a": 3, "b": 3, "class": "identical", "candidate_prefix_steps": 0, "candidate_prefix_degrees": 0.0, "start_shift_days": 0.0}]
    pairs += [{"a": 4 + i, "b": 4 + i, "class": "prefix changed", "candidate_prefix_steps": 4 + i, "candidate_prefix_degrees": 3.0, "start_shift_days": -1.0 - 0.25 * i} for i in range(8)]
    pairs += [{"a": 12, "b": 17, "class": "otherwise changed", "candidate_prefix_steps": 0, "candidate_prefix_degrees": 0.0, "start_shift_days": 0.0},   # same start, later positions differ
              {"a": 13, "b": 18, "class": "identical", "candidate_prefix_steps": 0, "candidate_prefix_degrees": 0.0, "start_shift_days": 0.0},         # same time and longitude, another latitude
              {"a": 14, "b": 19, "class": "otherwise changed", "candidate_prefix_steps": 5, "candidate_prefix_degrees": 15.0, "start_shift_days": 0.0}]  # east of 40 E and farther east, not earlier
    return {"year": 1990, "pairs": pairs, "attributes_a": attrs_a, "attributes_b": attrs_b, "unpaired_b": [12, 13, 14, 15, 16], "runs": {}}


def test_the_selection_is_the_predeclared_rule_in_a_deterministic_order():
    V = _load()
    out = V.select(_report())
    kinds = [c["kind"] for c in out["cases"]]
    assert kinds.count("candidate recovered prefix") == 6 and kinds.count("unpaired eastern start") == 3 and kinds.count("comparison, same start near the old edge") == 3
    prefixes = [c for c in out["cases"] if c["kind"] == "candidate recovered prefix"]
    assert [c["prefix_steps"] for c in prefixes] == [11, 10, 9, 8, 7, 6]                    # the six longest
    assert [c["b"] for c in prefixes] == [11, 10, 9, 8, 7, 6]
    unpaired = [c for c in out["cases"] if c["kind"] == "unpaired eastern start"]
    assert [c["b"] for c in unpaired] == [13, 14, 12] and all(c["start_lon"] > 40 for c in unpaired)   # east of 40 E, longest first, 16 at 30 E excluded
    assert unpaired[2]["potentially_censored_start"] is True
    comparison = [c for c in out["cases"] if c["kind"].startswith("comparison")]
    assert [c["a"] for c in comparison] == [1, 12, 3] and comparison[1]["pair_class"] == "otherwise changed"   # same first time, latitude and longitude, any class, longest first; 13 differs in latitude
    assert out["candidates_available"] == {"prefix": 10, "unpaired_eastern": 4, "comparison": 3}
    assert len(out["cases"]) <= 12 and out["readings"].startswith("to be entered by the human reader")


def test_ties_in_prefix_length_go_to_the_earlier_treatment_start():
    V = _load()
    r = _report()
    r["pairs"][0]["candidate_prefix_steps"] = 11                                           # ties with pair 11 (steps 11), starts earlier (99 against 146)
    out = V.select(r)
    prefixes = [c for c in out["cases"] if c["kind"] == "candidate recovered prefix"]
    assert prefixes[0]["b"] == 0 and prefixes[1]["b"] == 11


def test_steps_cover_the_prefix_and_two_steps_either_side():
    import numpy as np
    V = _load()
    times = np.arange(100.0, 110.0, 0.25)
    track_a = {"time": times[8:14]}
    track_b = {"time": times[4:14]}                                                           # a four-step prefix
    steps = V.steps_for({"kind": "candidate recovered prefix"}, track_b, track_a, times)
    assert steps == list(range(2, 11))                                                        # 4 - 2 to 8 + 2
    steps = V.steps_for({"kind": "unpaired eastern start"}, track_b, None, times)
    assert steps == list(range(4, 10))


def test_the_loose_prefix_rule_is_opt_in_and_declared_in_the_cases_file():
    V = _load()
    r = _report()
    for p in r["pairs"]:
        if p["class"] == "prefix changed":
            p["class"] = "otherwise changed"                                                   # positions altered, same end, earlier start
    strict = V.select(r)
    assert [c for c in strict["cases"] if c["kind"] == "candidate recovered prefix"] == [] and strict["rule"]["prefix_rule"] == "strict"
    for i in range(4, 12):                                                                  # the eight long ones now start east of 40 E
        r["attributes_b"][i]["start_lon"] = 45.0
    r["attributes_b"][0]["start_lon"], r["attributes_a"][0]["start_lon"] = 45.0, 46.0        # east of 40 E but not farther east than the control
    loose = V.select(r, "loose")
    prefixes = [c for c in loose["cases"] if c["kind"] == "candidate recovered prefix"]
    assert [c["prefix_steps"] for c in prefixes] == [11, 10, 9, 8, 7, 6] and loose["rule"]["prefix_rule_note"].startswith("SUPERSEDED")
    assert all(c["b"] >= 4 for c in prefixes) and loose["candidates_available"]["prefix"] == 8   # pair 0 (not farther east), pair 2 (22 E) and pair 14 (not earlier) excluded
    assert all(c["start_shift_days"] < 0 for c in prefixes)


def test_the_eastern_rule_is_item_2_as_written_restricted_to_eastern_starts():
    V = _load()
    r = _report()
    for i in range(4, 12):
        r["attributes_b"][i]["start_lon"] = 45.0                                            # the eight long ones start east of 40 E
    r["pairs"][4]["class"] = "endpoint changed"                                             # an endpoint-changed pair stays in, unlike "loose"
    out = V.select(r, "eastern")
    prefixes = [c for c in out["cases"] if c["kind"] == "candidate recovered prefix"]
    assert [c["prefix_steps"] for c in prefixes] == [11, 10, 9, 8, 7, 6] and out["candidates_available"]["prefix"] == 8
    assert out["rule"]["prefix_rule_note"].startswith("the brief's item 2 class as written")
    assert 4 not in {c["b"] for c in [c for c in V.select(r, "loose")["cases"] if c["kind"] == "candidate recovered prefix"]} \
        or V.select(r, "loose")["candidates_available"]["prefix"] == 7                       # "loose" drops the endpoint-changed pair
    r["attributes_b"][6]["start_time"] = r["attributes_a"][6]["start_time"]                 # not earlier: out
    r["attributes_b"][7]["start_lon"] = 25.0                                                 # earlier and farther east, but west of 40 E: out
    assert V.select(r, "eastern")["candidates_available"]["prefix"] == 6
    import pytest
    with pytest.raises(SystemExit, match="unknown prefix rule"):
        V.select(r, "wide")


def test_the_season_only_selection_is_supplementary_and_keeps_only_june_to_september_starts():
    V = _load()
    r = _report()
    for i in (11, 10):                                                                      # the two longest prefixes start in November on both sides
        r["attributes_a"][i]["start_month"] = r["attributes_b"][i]["start_month"] = 11
    r["attributes_a"][9]["start_month"] = 11                                               # the third: control in November, treatment in August, qualifies
    r["attributes_b"][13]["start_month"] = 2                                               # the longest unpaired eastern start, in February
    out = V.select(r, "strict", season_only=True)
    prefixes = [c for c in out["cases"] if c["kind"] == "candidate recovered prefix"]
    assert [c["prefix_steps"] for c in prefixes] == [9, 8, 7, 6, 5, 4] and all(c["b"] <= 9 for c in prefixes)
    unpaired = [c for c in out["cases"] if c["kind"] == "unpaired eastern start"]
    assert [c["b"] for c in unpaired] == [14, 12, 15] and out["candidates_available"]["unpaired_eastern"] == 3
    assert out["rule"]["season_only"] is True and out["rule"]["season_note"].startswith("SUPPLEMENTARY")
    assert V.select(r)["rule"]["season_only"] is False and V.select(r)["candidates_available"]["unpaired_eastern"] == 4


def _render_inputs(tmp_path, imagery_stamps=(), mismatched_stamp=None):
    """A 12-step case over 0 to 30 N and 20 to 60 E from 1990-08-01 00Z, a control track
    starting at step 4 at 35 E, a treatment track starting at step 0 at 45 E with the same
    end, an unpaired eastern track, both runs recorded with digests (the treatment naming
    the case), a cases file selected from them, and synthetic cropped imagery at the named
    stamps (one of which may carry a wrong internal timestamp)."""
    import datetime as dt
    import hashlib
    import json
    import numpy as np
    import xarray as xr
    from scipy.io import savemat
    t0 = float((dt.date(1990, 8, 1) - dt.date(1900, 1, 1)).days)
    times = t0 + 0.25 * np.arange(12)
    lat_c, lon_c = np.arange(0.0, 31.0), np.arange(20.0, 61.0)
    rng = np.random.default_rng(0)
    case_path = tmp_path / "tracker_case.mat"
    savemat(str(case_path), {"time": times, "lat_c": lat_c, "lon_c": lon_c, "u_c": rng.normal(size=(12, 31, 41)),
                             "v_c": rng.normal(size=(12, 31, 41)), "currv_anom_c": rng.normal(size=(12, 31, 41)) * 1e-6,
                             "advcurrv_anom_c": rng.normal(size=(12, 31, 41)) * 1e-11})
    case_digest = hashlib.sha256(case_path.read_bytes()).hexdigest()
    control = {"time": times[4:10], "lat": np.full(6, 12.0), "lon": 35.0 - 2.0 * np.arange(6)}
    treatment = {"time": times[0:10], "lat": np.full(10, 12.0), "lon": 45.0 - 2.5 * np.arange(10)}
    eastern = {"time": times[2:8], "lat": np.full(6, 10.0), "lon": 55.0 - 1.0 * np.arange(6)}
    runs = {}
    for name, tracks, lon_hi, manifest, extra in (("B", [control], 40.0, "m" * 64, {}), ("C", [treatment, eastern], 60.0, "n" * 64, {"case_sha256": case_digest, "coarse_threshold": 4.0e-7}),
                                                  ("C_other", [treatment, eastern], 60.0, "o" * 64, {"case_sha256": case_digest, "coarse_threshold": 4.0e-7})):
        d = tmp_path / name
        d.mkdir()
        mat = {"n": float(len(tracks)), "case_id": name.ljust(32, "x")}
        for i, t in enumerate(tracks):
            mat[f"lat{i}"], mat[f"lon{i}"], mat[f"time{i}"] = t["lat"], t["lon"], t["time"]
        savemat(str(d / "tracker_port.mat"), mat)
        digest = hashlib.sha256((d / "tracker_port.mat").read_bytes()).hexdigest()
        record = json.dumps({"dataset_specific": {"dataset": "era5", "year": 1990, "tracks_sha256": digest, **extra},
                             "protocol_settings": {"manifest_sha256": manifest, "domain": {"lat": [-35.0, 35.0], "lon": [-140.0, lon_hi]}}})
        (d / "tracking_era5_1990.json").write_text(record)
        runs[name] = (str(d), {"tracks_sha256": digest, "manifest_sha256": manifest, "record_sha256": hashlib.sha256(record.encode()).hexdigest()})
    cases = {"report": {"a": runs["B"][1], "b": runs["C"][1]}, "year": 1990,
             "cases": [{"kind": "candidate recovered prefix", "a": 0, "b": 0, "prefix_steps": 4, "prefix_degrees": 10.0, "start_shift_days": -1.0},
                       {"kind": "unpaired eastern start", "a": None, "b": 1, "observations": 6, "start_lon": 55.0, "potentially_censored_start": False}]}
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(json.dumps(cases))
    imagery = tmp_path / "imagery"
    imagery.mkdir()
    lat, lon = np.arange(0.0, 20.5, 0.5), np.arange(30.0, 60.5, 0.5)
    for stamp in imagery_stamps:
        tb = np.full((1, lat.size, lon.size), 300.0)
        tb[0][np.ix_((lat >= 11) & (lat <= 13), (lon >= 43) & (lon <= 45))] = 210.0            # a cold patch beside the treatment's first position
        inner = "1990-08-01T03:00" if stamp == mismatched_stamp else f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}T{stamp[8:10]}:00"
        xr.Dataset({"irwin_cdr": (("time", "lat", "lon"), tb)}, coords={"time": np.array([np.datetime64(inner)]), "lat": lat, "lon": lon}).to_netcdf(imagery / f"gridsat_{stamp}.nc")
    return str(case_path), runs["B"][0], runs["C"][0], str(cases_path), str(imagery), runs["C_other"][0]


def test_render_binds_to_the_runs_and_the_case_and_records_every_panels_satellite_coverage(tmp_path):
    import json
    V = _load()
    case_path, b_dir, c_dir, cases_path, imagery, _ = _render_inputs(tmp_path, imagery_stamps=("1990080100", "1990080106"), mismatched_stamp="1990080106")
    out_dir = str(tmp_path / "figures")
    made = V.render(cases_path, case_path, b_dir, c_dir, 1990, imagery, out_dir)
    assert [m["steps"] for m in made] == [list(range(0, 7)), list(range(2, 8))] and all(os.path.exists(pg["figure"]) for m in made for pg in m["pages"])
    assert [len(m["pages"]) for m in made] == [1, 1] and made[0]["pages"][0]["steps"] == list(range(0, 7))
    rec = json.load(open(os.path.join(out_dir, "audit_figures_1990.json")))
    assert rec["runs"]["treatment"] == {"dir": c_dir, **json.load(open(cases_path))["report"]["b"]} and rec["case_file_sha256"]
    panels = rec["figures"][0]["panels"]
    assert rec["field"]["coarse_threshold"] == 4.0e-7 and all("trough_axes" in q for q in panels)
    assert rec["figures"][0]["gridsat_usable_steps"] == 1 == rec["figures"][0]["gridsat_covered_steps"]
    assert panels[0]["gridsat"]["coverage"] == "retained imagery" and panels[0]["gridsat"]["cold_cells"] > 0 and panels[0]["gridsat"]["spatial_coverage"] == "complete"
    assert panels[1]["gridsat"]["coverage"] == "file timestamp does not match the requested timestep"      # named 06Z, holding 03Z
    assert panels[2]["gridsat"]["coverage"] == "not covered" and rec["figures"][0]["gridsat_covered_steps"] == 1
    assert panels[0]["position_lat_lon"] == [12.0, 45.0] and rec["readings"].startswith("to be entered by the human reader")


def test_render_claims_a_fresh_directory_and_never_overwrites_an_existing_page(tmp_path):
    import pytest
    V = _load()
    case_path, b_dir, c_dir, cases_path, imagery, _ = _render_inputs(tmp_path)
    out_dir = tmp_path / "figures"
    V.render(cases_path, case_path, b_dir, c_dir, 1990, imagery, str(out_dir))
    pages = sorted(p for p in out_dir.iterdir() if p.suffix == ".png")
    before = {p.name: p.read_bytes() for p in out_dir.iterdir()}
    with pytest.raises(SystemExit, match="figure sets are never overwritten"):
        V.render(cases_path, case_path, b_dir, c_dir, 1990, imagery, str(out_dir))
    assert {p.name: p.read_bytes() for p in out_dir.iterdir()} == before and len(pages) == 2    # every byte as it was
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit, match="figure sets are never overwritten"):
        V.render(cases_path, case_path, b_dir, c_dir, 1990, imagery, str(empty))
    assert list(empty.iterdir()) == []


def test_render_refuses_a_case_file_or_runs_the_cases_were_not_selected_from(tmp_path):
    import json
    import pytest
    V = _load()
    case_path, b_dir, c_dir, cases_path, imagery, c_other = _render_inputs(tmp_path)
    with pytest.raises(SystemExit, match="whose record_sha256 is not the b run's"):             # the same track bytes under another record and manifest
        V.render(cases_path, case_path, b_dir, c_other, 1990, imagery, str(tmp_path / "f0"))
    with open(case_path, "ab") as fh:
        fh.write(b"\\0")
    with pytest.raises(SystemExit, match="is not the case the treatment record names"):
        V.render(cases_path, case_path, b_dir, c_dir, 1990, imagery, str(tmp_path / "f1"))
    cases = json.load(open(cases_path))
    cases["report"]["a"]["tracks_sha256"] = "0" * 64
    other = tmp_path / "cases_other.json"
    other.write_text(json.dumps(cases))
    with pytest.raises(SystemExit, match="whose tracks_sha256 is not the a run's"):
        V.render(str(other), case_path, b_dir, c_dir, 1990, imagery, str(tmp_path / "f2"))
    del cases["report"]["a"]["record_sha256"]
    older = tmp_path / "cases_older.json"
    older.write_text(json.dumps(cases))
    with pytest.raises(SystemExit, match="does not carry the a run's record_sha256"):
        V.render(str(older), case_path, b_dir, c_dir, 1990, imagery, str(tmp_path / "f3"))


def test_the_tracker_view_is_the_detectors_coarse_stage():
    import numpy as np
    V = _load()
    lat_c, lon_c = np.array([-4.0, -2.0, 0.0, 2.0, 4.0]), np.arange(0.0, 20.0, 2.0)
    anom = np.full((5, 10), 1.0e-5)
    anom[:2] *= -1                                                                          # the southern rows hold cyclonic (negative) anomalies
    u = np.zeros((5, 10)); u[:, 7] = 10.0                                                   # one westerly column, 5 m/s after smoothing, 2.5 beside it
    adv = np.tile(np.linspace(-1.0, 1.0, 10), (5, 1)) * 1e-11                               # zero crossing between columns 4 and 5
    curvature, axes = V.tracker_view(u, anom, adv, lat_c, lon_c, coarse_threshold=4.0e-7)
    assert np.all(curvature[:2, :5] > 0) and np.nanmin(curvature) >= 5.0e-6 - 1e-12 and np.nanmax(curvature) <= 1.0e-5 + 1e-12
    # southern rows made positive; the smoother runs before the sign flip, so the rows beside the equator hold half the value
    assert np.isnan(curvature[2, 7]) and not np.isnan(curvature[2, 6])                      # the westerly column is masked, its neighbor at exactly 2.5 m/s is not
    assert axes and all(8.0 < float(np.mean(lons)) < 11.0 for _, lons in axes)             # the axis sits at the advection's zero crossing
    weak = anom.copy(); weak[2:, :] = 1.0e-8
    curvature, _ = V.tracker_view(np.zeros((5, 10)), weak, adv, lat_c, lon_c, coarse_threshold=4.0e-7)
    assert np.isnan(curvature[3, 3]) and not np.isnan(curvature[0, 3])                       # below the coarse threshold is masked


def test_nearest_axis_distance_is_point_to_segment():
    import numpy as np
    V = _load()
    axes = [(np.array([0.0, 10.0]), np.array([5.0, 5.0]))]                                  # a north-south axis at 5 E from 0 to 10 N
    assert V.nearest_axis_deg(5.0, 7.0, axes) == 2.0                                         # 2 degrees east of the segment's middle
    assert abs(V.nearest_axis_deg(13.0, 9.0, axes) - 5.0) < 1e-12                            # beyond the end: to the end point (3, 4 triangle)
    assert V.nearest_axis_deg(0.0, 0.0, []) is None


def test_proximity_records_every_step_with_its_reference_position_and_tallies_the_prefix(tmp_path):
    import json
    V = _load()
    case_path, b_dir, c_dir, cases_path, imagery, _ = _render_inputs(tmp_path)
    out = V.proximity(cases_path, case_path, b_dir, c_dir, 1990, str(tmp_path / "prox.json"))
    first = out["cases"][0]
    assert [r["step"] for r in first["steps"]] == list(range(0, 7)) and out["tolerance_deg"] == 2.0
    assert first["steps"][0]["position_source"] == "treatment" and first["steps"][0]["in_prefix"] and not first["steps"][5]["in_prefix"]
    assert first["summary"]["prefix"]["positioned_steps"] == 4 and first["summary"]["positioned"]["positioned_steps"] == 7
    for r in first["steps"]:
        assert (r["nearest_axis_deg"] is None) == (r["trough_axes"] == 0)
        assert r["axis_within_tolerance"] in (None, r["nearest_axis_deg"] is not None and r["nearest_axis_deg"] <= 2.0)
    second = out["cases"][1]
    assert second["steps"][0]["position_source"] == "treatment" and "prefix" not in second["summary"]
    g = first["treatment_geometry"]
    assert g["start_lat_lon"] == [12.0, 45.0] and g["end_lat_lon"] == [12.0, 22.5] and g["observations"] == 10
    assert g["whole_track_dlon"] == -22.5 and g["shown_steps_dlon"] == -15.0                # positions at steps 0 to 6, 2.5 degrees a step
    assert json.load(open(tmp_path / "prox.json"))["runs"]["treatment"]["tracks_sha256"]


def test_reference_measures_every_track_against_its_own_runs_axes(tmp_path):
    import pytest
    V = _load()
    case_path, b_dir, c_dir, cases_path, imagery, _ = _render_inputs(tmp_path)
    out = V.reference(case_path, c_dir, 1990, str(tmp_path / "ref.json"))
    assert out["all_tracks"]["tracks"] == 2 and out["tolerance_deg"] == 2.0                  # the treatment run holds two tracks
    assert set(out["all_tracks"]["share_percentiles"]) == {"10", "25", "50", "75", "90"}
    assert out["african_season_tracks"]["tracks"] == 0 and out["african_season_tracks"]["share_percentiles"]["50"] is None   # both start east of 40 E
    with pytest.raises(SystemExit, match="is not the case the record in"):
        V.reference(case_path, b_dir, 1990, str(tmp_path / "ref2.json"))                    # the control record names no such case
    assert V._share_and_gap([True, False, False, True, False]) == (0.4, 2)
    assert out["positions_measured"] == out["positions_stored"] == 16 and len(out["per_track"]) == 2
    m = out["all_tracks"]["meeting_both_cutoffs"]
    assert m["tracks"] == sum(1 for r in out["per_track"] if r["share"] >= m["share_at_least"] and r["longest_run_without"] <= m["longest_run_at_most"])


def test_reference_refuses_a_stored_position_off_the_cases_time_axis(tmp_path):
    import hashlib
    import json
    import numpy as np
    import pytest
    from scipy.io import loadmat, savemat
    V = _load()
    case_path, b_dir, c_dir, cases_path, imagery, _ = _render_inputs(tmp_path)
    times = np.asarray(loadmat(case_path, variable_names=["time"])["time"], float).ravel()
    d = tmp_path / "C_off"
    d.mkdir()
    savemat(str(d / "tracker_port.mat"), {"n": 1.0, "case_id": "C_off".ljust(32, "x"), "lat0": np.array([12.0, 12.0, 12.0]),
                                           "lon0": np.array([45.0, 44.0, 43.0]), "time0": np.array([times[0], times[1] + 0.1, times[2]])})
    record = {"dataset_specific": {"dataset": "era5", "year": 1990, "tracks_sha256": hashlib.sha256((d / "tracker_port.mat").read_bytes()).hexdigest(),
                                   "case_sha256": hashlib.sha256(open(case_path, "rb").read()).hexdigest(), "coarse_threshold": 4.0e-7},
              "protocol_settings": {"manifest_sha256": "n" * 64, "domain": {"lat": [-35.0, 35.0], "lon": [-140.0, 60.0]}}}
    (d / "tracking_era5_1990.json").write_text(json.dumps(record))
    with pytest.raises(SystemExit, match="is not on the case's time axis"):
        V.reference(case_path, str(d), 1990, str(tmp_path / "ref_off.json"))
    assert not (tmp_path / "ref_off.json").exists()
