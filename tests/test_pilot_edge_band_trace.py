"""The edge-band trace finds a control track's best treatment partner with no separation
cutoff and classes it as extended, displaced or replaced by the declared measures."""
import hashlib
import importlib.util
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def _load():
    spec = importlib.util.spec_from_file_location("pilot_edge_band_trace_under_test", os.path.join(ROOT, "scripts", "pilot_edge_band_trace.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _day(y, m, d):
    import datetime as dt
    return float((dt.date(y, m, d) - dt.date(1900, 1, 1)).days)


def _track(t0, lat, lon, n, step=-0.5):
    return {"time": t0 + 0.25 * np.arange(n), "lat": np.full(n, float(lat)), "lon": lon + step * np.arange(n)}


def _run(tmp_path, name, tracks, domain_lon):
    from scipy.io import savemat
    d = tmp_path / name
    d.mkdir(parents=True)
    mat = {"n": float(len(tracks)), "case_id": name.ljust(32, "x")}
    for i, t in enumerate(tracks):
        mat[f"lat{i}"], mat[f"lon{i}"], mat[f"time{i}"] = t["lat"], t["lon"], t["time"]
    savemat(str(d / "tracker_port.mat"), mat)
    record = {"dataset_specific": {"dataset": "era5", "year": 1990, "tracks_sha256": hashlib.sha256((d / "tracker_port.mat").read_bytes()).hexdigest()},
              "protocol_settings": {"manifest_sha256": "m" * 64, "domain": {"lat": [-35.0, 35.0], "lon": [-140.0, domain_lon]}}}
    (d / "tracking_era5_1990.json").write_text(json.dumps(record))
    return str(d)


def test_the_trace_classes_extended_displaced_and_replaced_controls(tmp_path):
    T = _load()
    t0 = _day(1990, 8, 1)
    same = _track(t0, 12.0, 35.0, 10)                                          # identical in both runs
    extended = _track(t0 + 3, 10.0, 36.0, 10)                                  # the treatment starts 4 steps earlier at 38 E... see below
    extended_b = {"time": np.concatenate([t0 + 3 - 0.25 * np.arange(4, 0, -1), extended["time"]]),
                  "lat": np.full(14, 10.0), "lon": np.concatenate([44.0 - 0.5 * np.arange(4, 0, -1) + 0.0, extended["lon"]])}   # earlier start at 42 E, then the control's positions
    displaced = _track(t0 + 6, 8.0, 38.0, 10)
    displaced_b = dict(displaced, lat=displaced["lat"] + 2.0)                  # the same times, 2 degrees north
    replaced = _track(t0 + 9, 14.0, 33.0, 10)                                   # no treatment track at any of its times
    outside = _track(t0 + 12, 12.0, 20.0, 10)                                   # starts outside the band
    a = T.P.load_run(_run(tmp_path, "B", [same, extended, displaced, replaced, outside], 40.0), year=1990)
    b = T.P.load_run(_run(tmp_path, "C", [same, extended_b, displaced_b, outside], 60.0), year=1990)
    a["dir"], b["dir"] = "B", "C"
    r = T.trace(a, b, 1990, (30.0, 40.0))
    assert r["controls_in_band"] == 4 and r["controls_with_no_treatment_track_at_any_of_their_times"] == [3]
    s = r["all_months"]
    assert s["controls"] == 3 and s["controls_covered_within_near"] == 2 and s["controls_covered_within_close"] == 3
    assert s["partners_starting_east_of_edge"] == 1 and s["partners_starting_earlier"] == 1 and s["partners_starting_same_time"] == 2
    assert s["partners_east_and_earlier_covering_within_near"] == 1 == s["partners_east_and_earlier_covering_within_close"]
    by = {row["a"]: row for row in r["rows"]}
    assert by[2]["partner"]["mean_separation_deg"] == 2.0 and by[2]["partner"]["steps_within_near"] == 0 and by[2]["partner"]["steps_within_close"] == 10
    assert by[1]["partner"]["b"] == 1 and by[1]["partner"]["shared_steps"] == 10 and by[1]["partner"]["mean_separation_deg"] == 0.0
    assert r["june_to_september"]["controls"] == 3 and r["rule"]["close_deg"] == 3.0


def test_an_empty_band_gives_a_stable_summary_and_the_command_exits_cleanly(tmp_path, capsys):
    T = _load()
    t0 = _day(1990, 8, 1)
    a_dir = _run(tmp_path, "B", [_track(t0, 12.0, 20.0, 10)], 40.0)                        # starts at 20 E, outside 30 to 40 E
    b_dir = _run(tmp_path, "C", [_track(t0, 12.0, 20.0, 10)], 60.0)
    a, b = T.P.load_run(a_dir, year=1990), T.P.load_run(b_dir, year=1990)
    a["dir"], b["dir"] = "B", "C"
    r = T.trace(a, b, 1990, (30.0, 40.0))
    assert r["controls_in_band"] == 0 and r["all_months"]["controls"] == 0 and r["all_months"]["controls_covered_within_close"] == 0
    assert set(r["all_months"]) == set(r["june_to_september"]) and r["all_months"]["partner_mean_separation_deg_percentiles"] == {"10": None, "50": None, "90": None}
    out = tmp_path / "trace.json"
    assert T.main(["--a", a_dir, "--b", b_dir, "--year", "1990", "--band", "30", "40", "--out", str(out)]) == 0
    assert "0 control tracks start in" in capsys.readouterr().out and out.exists()


def test_the_partner_is_chosen_by_close_steps_then_the_smaller_separation(tmp_path):
    T = _load()
    t0 = _day(1990, 8, 1)
    control = _track(t0, 12.0, 35.0, 10)
    many_far = dict(control, lat=control["lat"] + 2.5)                                      # all 10 steps within 3 degrees, mean 2.5
    few_near = {"time": control["time"][:4], "lat": control["lat"][:4] + 0.2, "lon": control["lon"][:4]}   # 4 steps, mean 0.2
    many_farther = dict(control, lat=control["lat"] + 2.9)                                   # all 10 steps within 3 degrees, mean 2.9
    a = T.P.load_run(_run(tmp_path, "B", [control], 40.0), year=1990)
    b = T.P.load_run(_run(tmp_path, "C", [few_near, many_farther, many_far], 60.0), year=1990)
    a["dir"], b["dir"] = "B", "C"
    row = T.trace(a, b, 1990, (30.0, 40.0))["rows"][0]
    assert row["partner"]["b"] == 2 and row["partner"]["steps_within_close"] == 10 and row["partner"]["mean_separation_deg"] == 2.5
    b2 = T.P.load_run(_run(tmp_path, "C2", [few_near], 60.0), year=1990)
    b2["dir"] = "C2"
    assert T.trace(a, b2, 1990, (30.0, 40.0))["rows"][0]["partner"]["b"] == 0
