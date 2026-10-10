"""The wide case's crop and bind, and the all-edge replay: the selection of the control
grids inside the wide grids, the raw-field equality, the preparation verified against the
native one with differences confined to the edge ring, the detector reproduced from
prepared fields, the native gate against the production loop, and a margin run on a
synthetic wide case."""
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


def analytic_case(lat_hi, lon_lo, lon_hi, steps=12, start_lon=30.0, speed=-1.5):
    """A trough drifting west with the same analytic fields on any box, winds that vary,
    on a one-degree fine grid and a two-degree coarse grid whose coordinates are the
    control fixture's (odd longitudes, even latitudes) inside the box."""
    lat_f = np.arange(lat_hi, -lat_hi - 0.5, -1.0)
    lon_f = np.arange(lon_lo, lon_hi + 0.5, 1.0)
    lat_c = np.array([x for x in np.arange(34.0 + 2.0 * 10, -35.0 - 2.0 * 10, -2.0) if -lat_hi <= x <= lat_hi])      # even latitudes within the box
    lon_c = np.array([x for x in np.arange(-139.0 - 2.0 * 10, lon_hi + 2.0 * 10, 2.0) if lon_lo <= x <= lon_hi])      # odd longitudes within the box
    latgrid, longrid = np.meshgrid(lat_f, lon_f, indexing="ij")
    LA, LO = np.meshgrid(lat_c, lon_c, indexing="ij")
    anom, adv, anom_f = [], [], []
    for k in range(steps):
        c = start_lon + k * speed
        a = 2e-5 * np.exp(-((LO - c) / 4.0) ** 2) * np.exp(-((LA - 12.0) / 6.0) ** 2)
        anom.append(np.where(LA < 0, -a, a))
        adv.append(1e-10 * (LO - c) * np.exp(-((LA - 12.0) / 6.0) ** 2))
        anom_f.append(2e-5 * np.exp(-((longrid - c) / 4.0) ** 2) * np.exp(-((latgrid - 12.0) / 6.0) ** 2))
    shape = (steps,) + LA.shape
    u_c = -7.0 + 0.5 * np.sin(LO / 9.0) + 0.3 * np.cos(LA / 5.0)                                     # nonlinear, so the edge smoothing shows at the corners too
    return {"time": 100.0 + 0.25 * np.arange(steps), "lat_c": lat_c, "lon_c": lon_c, "latgrid": latgrid, "longrid": longrid,
            "u_c": np.repeat(u_c[None], steps, 0), "v_c": np.zeros(shape), "currv_anom_c": np.stack(anom), "advcurrv_anom_c": np.stack(adv), "currv_anom": np.stack(anom_f),
            "u": np.repeat((-7.0 + 0.5 * np.sin(longrid / 9.0) + 0.3 * np.cos(latgrid / 5.0))[None], steps, 0), "v": np.repeat((0.5 * np.sin(longrid / 7.0))[None], steps, 0), "sha256": "x" * 64}


FLAGS = {"exclusive": False, "absorb": False}


def test_the_wide_case_crop_and_bind_reproduce_the_control_box():
    W = _load("pilot_wide_case")
    steps = 3
    lat = np.arange(50.0, -51.0, -1.0); lon = np.arange(-155.0, 76.0, 1.0)
    latgrid, longrid = np.meshgrid(lat, lon, indexing="ij")
    lat_c, lon_c = np.meshgrid(lat[::2], lon[::2], indexing="ij")
    field = lambda g: np.repeat((np.sin(g[0] / 9.0) * np.cos(g[1] / 11.0))[None], steps, 0)
    pre = {"times": 100.0 + 0.25 * np.arange(steps), "latgrid": latgrid, "longrid": longrid, "u": field((latgrid, longrid)), "v": 2 * field((latgrid, longrid)),
           "anomaly": 3 * field((latgrid, longrid)), "lat_c": lat_c, "lon_c": lon_c,
           "coarse": {"u": field((lat_c, lon_c)), "v": 2 * field((lat_c, lon_c)), "anomaly": 3 * field((lat_c, lon_c)), "advection": 4 * field((lat_c, lon_c))}}
    control = W.crop(pre, (-35.0, 35.0), (-140.0, 40.0))
    wide = W.crop(pre, W.WIDE_LAT, W.WIDE_LON)
    assert control["latgrid"].shape == (71, 181) and control["lat_c"].size == 35 and control["lon_c"].size == 90
    assert wide["latgrid"].shape == (79, 189) and wide["lat_c"].size == 39 and wide["lon_c"].size == 94
    case_b = dict(control); case_b["sha256"] = "x" * 64
    bound = W.bind_to_control(control, case_b)
    assert all(bound.values()) and set(bound) == {"lat_c", "lon_c", "latgrid", "longrid", "time", *W.RAW_FIELDS}
    moved = dict(control); moved["u"] = control["u"] + 1e-12
    assert W.bind_to_control(moved, case_b)["u"] is False and W.bind_to_control(moved, case_b)["v"] is True


def test_selection_raw_equality_and_the_preparation_verification_confine_differences_to_the_ring():
    A = _load("pilot_alledge_replay")
    case_b, wide = analytic_case(35.0, -140.0, 39.0), analytic_case(39.0, -144.0, 43.0)
    sel = A.selection(case_b, wide)
    assert [int(s.sum()) for s in sel] == [case_b["lat_c"].size, case_b["lon_c"].size, case_b["latgrid"].shape[0], case_b["longrid"].shape[1]]
    assert all(v == 0 for v in A.raw_equality(case_b, wide, sel).values())
    all_c = (np.ones(case_b["lat_c"].size, bool), np.ones(case_b["lon_c"].size, bool))
    all_f = (np.ones(case_b["latgrid"].shape[0], bool), np.ones(case_b["longrid"].shape[1], bool))
    native = A.prepared_fields(case_b, 0, all_c[0], all_c[1], all_f[0], all_f[1])
    margin = A.prepared_fields(wide, 0, *sel)
    for name, _g in A.PREPARED:
        c = A.ring_counts(native[name], margin[name])
        assert c["interior"] == 0, name
        assert margin[name].shape == native[name].shape
    assert A.ring_counts(native["fine_zonal_wind_smoothed"], margin["fine_zonal_wind_smoothed"])["ring"] > 0        # the winds the association reads differ on the ring
    assert A.ring_counts(native["fine_zonal_wind_smoothed"], margin["fine_zonal_wind_smoothed"])["corners"] == 4
    assert A.ring_counts(native["coarse_zonal_wind_smoothed"], margin["coarse_zonal_wind_smoothed"])["corners"] == 4
    shifted = analytic_case(39.0, -144.0, 43.0)
    shifted["currv_anom"] = np.roll(shifted["currv_anom"], 2, axis=2)
    assert A.ring_counts(native["fine_curvature"], A.prepared_fields(shifted, 0, *sel)["fine_curvature"])["interior"] > 0
    missing = analytic_case(39.0, -144.0, 43.0)
    keep = missing["lat_c"] != 0.0
    missing["lat_c"] = missing["lat_c"][keep]
    for name in ("u_c", "v_c", "currv_anom_c", "advcurrv_anom_c"):
        missing[name] = missing[name][:, keep, :]
    with pytest.raises(SystemExit, match="not a subset"):
        A.selection(case_b, missing)


def test_native_mode_reproduces_detect_troughs_and_the_production_loop_and_margin_mode_runs_and_verifies():
    A = _load("pilot_alledge_replay")
    E = _load("export_protocol_case")
    case_b, wide = analytic_case(35.0, -140.0, 39.0), analytic_case(39.0, -144.0, 43.0)
    grids = A.MR.control_grids(case_b)
    finished, counts, bound, verification = A.run_season(case_b, grids, 4.0e-7, 2.5e-6, FLAGS)
    assert bound == 12 and verification is None and finished
    ref = E.track_case({k: v for k, v in case_b.items() if k != "sha256"}, 4.0e-7, 2.5e-6, exclusive=False, absorb=False)
    archived = {"tracks": [{"time": np.asarray(t["time"]), "lat": np.asarray(t["meanlat"]), "lon": np.asarray(t["meanlon"])} for t in ref]}
    assert A.MR.archive_gate(finished, archived)["passed"]
    sel = A.selection(case_b, wide)
    finished_m, counts_m, bound_m, ver = A.run_season(case_b, grids, 4.0e-7, 2.5e-6, FLAGS, wide=wide, sel=sel)
    assert bound_m == 0 and ver["steps"] == 12 and ver["interior_differing_max"] == 0 and finished_m
    assert all(ver["ring_differing_steps"][n] == 12 for n in ("fine_zonal_wind_smoothed", "fine_meridional_wind_smoothed", "coarse_zonal_wind_smoothed"))
    assert all(len(f["steps"]) == len(f["raw_lat"]) == len(f["region_sha256"]) for f in finished_m)
    broken = analytic_case(39.0, -144.0, 43.0)
    broken["u"] = broken["u"] + 0.3                                                                  # an interior difference in the association's wind
    with pytest.raises(SystemExit, match="interior cells of fine_zonal_wind_smoothed"):
        A.run_season(case_b, grids, 4.0e-7, 2.5e-6, FLAGS, wide=broken, sel=sel)


def test_thresholds_come_from_the_control_record_or_a_bound_calibration_audit_with_a_declared_transfer(tmp_path):
    import hashlib, json
    A = _load("pilot_alledge_replay")
    ct, ft, rec = A.thresholds_for_run((1.0, 2.0))
    assert (ct, ft) == (1.0, 2.0) and rec == {"coarse": 1.0, "fine": 2.0, "from": "the control run's record"}
    audit = {"generated_by": "scripts/pilot_alledge_calibration.py", "domain": {"lat_range": [-35.0, 35.0], "lon_range": [-140.0, 40.0]},
             "orders": {"crop_then_smooth": {"coarse": {"threshold": 1.0}, "fine": {"threshold": 2.0}}, "smooth_then_crop": {"coarse": {"threshold": 0.9}, "fine": {"threshold": 1.9}}},
             "bind": {"archived_order_reproduces_released_pair": True, "got": [1.0, 2.0], "released": [1.0, 2.0]}, "released_artifact": {"coarse": 1.0, "fine": 2.0, "case_id": "c"}}
    path = tmp_path / "audit.json"
    blob = json.dumps(audit).encode()
    path.write_bytes(blob)
    ct, ft, rec = A.thresholds_for_run((1.0, 2.0), str(path))
    assert (ct, ft) == (0.9, 1.9) and rec["transfer"] is False and rec["control_run_pair"] == [1.0, 2.0]
    assert rec["calibration_audit"]["sha256"] == hashlib.sha256(blob).hexdigest() and rec["calibration_audit"]["audited_released_pair"] == [1.0, 2.0]
    assert rec["calibration_audit"]["domain"] == audit["domain"] and rec["from"] == "the calibration audit's smooth_then_crop order"
    with pytest.raises(SystemExit, match="not this control run's pair"):                              # another run's pair without a declared transfer
        A.thresholds_for_run((1.0, 2.5), str(path))
    ct, ft, rec = A.thresholds_for_run((1.0, 2.5), str(path), transfer=True)
    assert (ct, ft) == (0.9, 1.9) and rec["transfer"] is True and rec["control_run_pair"] == [1.0, 2.5] and rec["calibration_audit"]["audited_released_pair"] == [1.0, 2.0]
    with pytest.raises(SystemExit, match="nothing is transferred"):                                   # a transfer declared where the pairs are the same
        A.thresholds_for_run((1.0, 2.0), str(path), transfer=True)
    with pytest.raises(SystemExit, match="needs a calibration input"):
        A.thresholds_for_run((1.0, 2.0), None, transfer=True)
    unbound = tmp_path / "unbound.json"
    unbound.write_text(json.dumps({**audit, "bind": {**audit["bind"], "archived_order_reproduces_released_pair": False}}))
    with pytest.raises(SystemExit, match="did not reproduce"):                                        # the audit's own bind is still required under a transfer
        A.thresholds_for_run((1.0, 2.5), str(unbound), transfer=True)


def test_native_mode_refuses_a_calibration_input_before_reading_anything(tmp_path):
    A = _load("pilot_alledge_replay")
    args = ["--control-run", "a", "--control-case", "b", "--wide-dir", "c", "--year", "1990", "--out", str(tmp_path / "o.json.gz")]
    with pytest.raises(SystemExit, match="released configuration"):
        A.main(args + ["--preparation", "native", "--calibration", "x"])
    with pytest.raises(SystemExit, match="released configuration"):
        A.main(args + ["--preparation", "native", "--transfer"])


def test_a_domain_calibration_applies_the_domains_own_audit_pair_anchored_to_another_released_pair(tmp_path):
    import json
    A = _load("pilot_alledge_replay")
    audit = {"generated_by": "scripts/pilot_alledge_calibration.py", "domain": {"lat_range": [-35.0, 35.0], "lon_range": [-140.0, 60.0]},
             "orders": {"crop_then_smooth": {"coarse": {"threshold": 3.0}, "fine": {"threshold": 4.0}}, "smooth_then_crop": {"coarse": {"threshold": 2.9}, "fine": {"threshold": 3.9}}},
             "bind": {"archived_order_reproduces_released_pair": True, "got": [3.0, 4.0], "released": [3.0, 4.0]}, "released_artifact": {"coarse": 3.0, "fine": 4.0, "case_id": "d"}}
    path = tmp_path / "audit60.json"
    path.write_text(json.dumps(audit))
    domain = {"lat": [-35.0, 35.0], "lon": [-140.0, 60.0]}
    ct, ft, rec = A.thresholds_for_run((1.0, 2.0), str(path), domain_calibration=True, control_domain=domain)   # the control carries another run's pair
    assert (ct, ft) == (2.9, 3.9) and rec["domain_calibration"] is True and rec["transfer"] is False
    assert rec["control_run_pair"] == [1.0, 2.0] and rec["calibration_audit"]["audited_released_pair"] == [3.0, 4.0] and rec["calibration_audit"]["domain"] == audit["domain"]
    with pytest.raises(SystemExit, match="not this control run's domain"):                            # the audit of another domain is not a domain calibration
        A.thresholds_for_run((1.0, 2.0), str(path), domain_calibration=True, control_domain={"lat": [-35.0, 35.0], "lon": [-140.0, 40.0]})
    with pytest.raises(SystemExit, match="one of"):                                                   # transfer and domain calibration exclude each other
        A.thresholds_for_run((1.0, 2.0), str(path), transfer=True, domain_calibration=True, control_domain=domain)
    with pytest.raises(SystemExit, match="needs a calibration input"):
        A.thresholds_for_run((1.0, 2.0), None, domain_calibration=True, control_domain=domain)
    with pytest.raises(SystemExit, match="no flag is needed"):                                        # the control's own pair needs no declaration
        A.thresholds_for_run((3.0, 4.0), str(path), domain_calibration=True, control_domain=domain)
    with pytest.raises(SystemExit, match="did not reproduce"):                                        # the audit's own bind is still required
        bad = tmp_path / "unbound60.json"
        bad.write_text(json.dumps({**audit, "bind": {**audit["bind"], "archived_order_reproduces_released_pair": False}}))
        A.thresholds_for_run((1.0, 2.0), str(bad), domain_calibration=True, control_domain=domain)
    with pytest.raises(SystemExit, match="released configuration"):
        A.main(["--control-run", "a", "--control-case", "b", "--wide-dir", "c", "--year", "1990", "--out", str(tmp_path / "o.json.gz"), "--preparation", "native", "--domain-calibration"])


def test_a_domain_calibration_refuses_an_audit_whose_latitude_range_alone_differs(tmp_path):
    import json
    A = _load("pilot_alledge_replay")
    audit = {"generated_by": "scripts/pilot_alledge_calibration.py", "domain": {"lat_range": [-30.0, 35.0], "lon_range": [-140.0, 60.0]},
             "orders": {"crop_then_smooth": {"coarse": {"threshold": 3.0}, "fine": {"threshold": 4.0}}, "smooth_then_crop": {"coarse": {"threshold": 2.9}, "fine": {"threshold": 3.9}}},
             "bind": {"archived_order_reproduces_released_pair": True, "got": [3.0, 4.0], "released": [3.0, 4.0]}, "released_artifact": {"coarse": 3.0, "fine": 4.0}}
    path = tmp_path / "audit_lat.json"
    path.write_text(json.dumps(audit))
    with pytest.raises(SystemExit, match="not this control run's domain"):
        A.thresholds_for_run((1.0, 2.0), str(path), domain_calibration=True, control_domain={"lat": [-35.0, 35.0], "lon": [-140.0, 60.0]})


def test_a_trace_is_a_side_record_that_leaves_the_loop_unchanged_and_names_every_track_it_saw(tmp_path):
    A = _load("pilot_alledge_replay")
    case_b, wide = analytic_case(35.0, -140.0, 39.0), analytic_case(39.0, -144.0, 43.0)
    grids, sel = A.MR.control_grids(case_b), A.selection(case_b, wide)
    plain, counts, _b, _v = A.run_season(case_b, grids, 4.0e-7, 2.5e-6, FLAGS, wide=wide, sel=sel)
    assert plain
    point = (plain[0]["lat"][0], plain[0]["lon"][0])
    fam = {"name": "T", "box": {"lat": [-90.0, 90.0], "lon": [-180.0, 180.0]}, "window": ["synthetic", "synthetic"]}
    trace = {"steps": set(range(12)), "box": fam["box"], "families": [fam], "point": point, "radius_deg": 5.0}
    traced, counts_t, _b2, _v2 = A.run_season(case_b, grids, 4.0e-7, 2.5e-6, FLAGS, wide=wide, sel=sel, trace=trace)
    assert traced == plain and counts_t == counts                                                      # the trace never touches the live state
    res = trace["result"]
    assert len(res["steps"]) == 12 and res["born"] and set(res["born"]) == set(res["fates"]) and set(res["histories"]) >= set(res["born"])
    assert any(s["seeds"] for s in res["steps"]) and all("nearest_to_point" in s and "matching" in s and "pruned" in s for s in res["steps"])
    first = res["steps"][0]["nearest_to_point"]
    assert first is not None and first["distance_deg"] == 0.0 and first["held_by"] in res["born"] and first["within_radius"] is True
    assert all(row["fate"] in ("finished", "pruned_in_loop", "removed_as_single_observation", "removed_by_final_speed_filter") for row in res["summary"])
    assert sum(1 for row in res["summary"] if row["fate"] == "finished") == len(res["finished_index"]) == sum(1 for b in res["born"] if b in {f["birth"] for f in traced})
