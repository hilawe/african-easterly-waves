"""The retained review checks: the tropical eastern summer starts are the attributes'
tracks east of 40 E, 5 to 20 N, June to September; the union coverage counts any treatment
position at the same time within 3 degrees; a crosswalk from other runs is refused."""
import hashlib
import importlib.util
import json
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def _load():
    spec = importlib.util.spec_from_file_location("pilot_review_checks_under_test", os.path.join(ROOT, "scripts", "pilot_review_checks.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _day(y, m, d):
    import datetime as dt
    return float((dt.date(y, m, d) - dt.date(1900, 1, 1)).days)


def _track(t0, lat, lon, n, step=-1.0):
    return {"time": t0 + 0.25 * np.arange(n), "lat": np.full(n, float(lat)), "lon": lon + step * np.arange(n)}


def test_tropical_starts_and_union_coverage():
    R = _load()
    attrs = [{"start_lon": 45.0, "start_lat": 12.0, "start_month": 8, "end_lon": 30.0, "observations": 10},   # qualifies
             {"start_lon": 45.0, "start_lat": 25.0, "start_month": 8, "end_lon": 30.0, "observations": 10},   # too far north
             {"start_lon": 45.0, "start_lat": 12.0, "start_month": 10, "end_lon": 30.0, "observations": 10},  # out of season
             {"start_lon": 35.0, "start_lat": 12.0, "start_month": 8, "end_lon": 30.0, "observations": 10}]   # west of 40 E
    assert R.tropical_eastern_summer_starts({"attributes_b": attrs}) == [0]
    t0 = _day(1990, 8, 1)
    control = [_track(t0, 12.0, 35.0, 8), _track(t0, 12.0, 20.0, 8), _track(_day(1990, 3, 1), 12.0, 35.0, 8)]  # in band; out of band; out of season
    treatment = [_track(t0, 12.5, 35.0, 4), _track(t0 + 1.0, 12.5, 31.0, 4)]                                    # two tracks that together cover all 8 steps
    u = R.union_coverage(control, treatment)
    assert u["controls"] == 1 and u["rows"][0]["share_within_close"] == 1.0 and u["controls_covered_at_least_80_percent"] == 1
    u2 = R.union_coverage(control, treatment[:1])
    assert u2["rows"][0]["share_within_close"] == 0.5 and u2["controls_covered_at_least_80_percent"] == 0 and u2["median_share"] == 0.5


def test_a_crosswalk_from_other_runs_is_refused(tmp_path):
    from scipy.io import savemat
    R = _load()
    d = tmp_path / "B"; d.mkdir()
    savemat(str(d / "tracker_port.mat"), {"n": 1.0, "case_id": "B".ljust(32, "x"), "lat0": np.array([1.0, 1.0]), "lon0": np.array([1.0, 0.5]), "time0": np.array([1.0, 1.25])})
    digest = hashlib.sha256((d / "tracker_port.mat").read_bytes()).hexdigest()
    (d / "tracking_era5_1990.json").write_text(json.dumps({"dataset_specific": {"dataset": "era5", "year": 1990, "tracks_sha256": digest},
                                                             "protocol_settings": {"manifest_sha256": "m" * 64, "domain": {"lat": [-35.0, 35.0], "lon": [-140.0, 40.0]}}}))
    cw = tmp_path / "cw.json"
    cw.write_text(json.dumps({"runs": {"a": {"tracks_sha256": "0" * 64}, "b": {"tracks_sha256": digest}}, "attributes_b": []}))
    with pytest.raises(SystemExit, match="was not made from the a run"):
        R.main(["--crosswalk", str(cw), "--control-run", str(d), "--treatment-run", str(d), "--year", "1990", "--out", str(tmp_path / "o.json")])
