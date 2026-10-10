"""The mask-equivalence check counts a changed decision only where a prepared value lies
between the two thresholds (or between their ladder multiples), reads the case it hashes,
and refuses a case the record does not name."""
import hashlib
import importlib.util
import json
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def _load():
    spec = importlib.util.spec_from_file_location("check_threshold_mask_equivalence_under_test", os.path.join(ROOT, "scripts", "check_threshold_mask_equivalence.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _case_blob(values, u=None):
    """One timestep per value, the field constant over the 4 by 5 northern-hemisphere grid
    at that step, so the smoother returns it unchanged and no sign flip applies."""
    from scipy.io import savemat
    import io
    lat_c = np.array([8.0, 6.0, 4.0, 2.0])
    n = len(values)
    anom = np.zeros((n, 4, 5)) + np.asarray(values, float)[:, None, None]
    u = np.zeros((n, 4, 5)) if u is None else u
    buf = io.BytesIO()
    savemat(buf, {"time": 1.0 + 0.25 * np.arange(n), "lat_c": lat_c, "u_c": u, "currv_anom_c": anom})
    return buf.getvalue()


def test_decisions_differ_only_where_a_value_lies_between_the_two_numbers():
    C = _load()
    thr_a, thr_b = 4.0e-7, 4.0e-7 + 2e-21                                                   # two ulps apart, as the pilot's
    blob = _case_blob([1e-5, 5e-7, 1e-7, 3.9e-7, 4.0e-7 + 1e-21])                           # the last step's value lies between the two numbers
    r = C.check(blob, "lat_c", thr_a, thr_b)
    assert r["ladder_multipliers"] == [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    assert r["base_mask_cells_differing"] == 20 and r["ladder_mask_cells_differing"][0] == 20 and r["ladder_mask_cells_differing"][1:] == [0] * 5
    assert not r["passed"] and r["finite_prepared_coarse_values"] == 100
    on_level = C.check(_case_blob([thr_a * 1.5, thr_b * 2.0]), "lat_c", thr_a, thr_b)        # values exactly on a level of one number
    # the ladder keeps a cell AT or above a level: on a's 1.5 level a keeps and b (higher) discards, on b's 2.0
    # level both keep; a ladder that kept only strictly above would also differ at the 2.0 level
    assert on_level["ladder_mask_cells_differing"] == [0, 20, 0, 0, 0, 0] and on_level["base_mask_cells_differing"] == 0
    same = C.check(_case_blob([1e-5, 5e-7, 1e-7, 3.9e-7]), "lat_c", thr_a, thr_b)
    assert same["passed"] and same["base_mask_cells_differing"] == 0 and same["ladder_mask_cells_differing"] == [0] * 6
    assert same["min_abs_distance_to_any_level_of_b"] == pytest.approx(min(abs(v - thr_b * m) for v in (1e-5, 5e-7, 1e-7, 3.9e-7) for m in (1, 1.5, 2, 2.5, 3, 3.5)))


def test_westerly_cells_are_not_decided_and_a_wrong_case_is_refused(tmp_path):
    C = _load()
    u = np.zeros((2, 4, 5)); u[:, :, 2] = 10.0                                              # a westerly column, 5 m/s after smoothing, 2.5 beside it
    r = C.check(_case_blob([1e-5, 4.0e-7 + 1e-21], u=u), "lat_c", 4.0e-7, 4.0e-7 + 2e-21)
    assert r["finite_prepared_coarse_values"] == 2 * 4 * 4 and r["base_mask_cells_differing"] == 16  # the masked column is not decided
    from scipy.io import savemat
    d = tmp_path / "run"; d.mkdir()
    blob = _case_blob([1e-5, 1e-7])
    (d / "tracker_case.mat").write_bytes(blob)
    savemat(str(d / "tracker_port.mat"), {"n": 1.0, "case_id": "r".ljust(32, "x"), "lat0": np.array([1.0, 1.0]), "lon0": np.array([1.0, 0.5]), "time0": np.array([1.0, 1.25])})
    record = {"dataset_specific": {"dataset": "era5", "year": 1990, "tracks_sha256": hashlib.sha256((d / "tracker_port.mat").read_bytes()).hexdigest(),
                                   "case_sha256": "0" * 64, "coarse_threshold": 4.0e-7, "fine_threshold": 2.5e-6},
              "protocol_settings": {"manifest_sha256": "m" * 64, "domain": {"lat": [-35.0, 35.0], "lon": [-140.0, 40.0]}}}
    (d / "tracking_era5_1990.json").write_text(json.dumps(record))
    with pytest.raises(SystemExit, match="is not the case the record in"):
        C.main(["--run", str(d), "--case", str(d / "tracker_case.mat"), "--year", "1990", "--threshold-a", "4e-7", "--threshold-b", "4e-7", "--out", str(tmp_path / "o.json")])
    record["dataset_specific"]["case_sha256"] = hashlib.sha256(blob).hexdigest()
    (d / "tracking_era5_1990.json").write_text(json.dumps(record))
    assert C.main(["--run", str(d), "--case", str(d / "tracker_case.mat"), "--year", "1990", "--threshold-a", "4e-7", "--threshold-b", "4e-7", "--out", str(tmp_path / "o.json")]) == 0
    assert json.load(open(tmp_path / "o.json"))["passed"]
