"""The prepared-input comparison passes only on the same arrays, compares missing cells as
cells, reports a difference with its size and place, and refuses a case file that is not
the one its record names or a pair that lacks a field."""
import hashlib
import importlib.util
import json
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def _load():
    spec = importlib.util.spec_from_file_location("compare_case_fields_under_test", os.path.join(ROOT, "scripts", "compare_case_fields.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fields(seed=0):
    rng = np.random.default_rng(seed)
    f = {"time": np.arange(4.0), "lat_c": np.arange(3.0), "lon_c": np.arange(5.0), "latgrid": np.ones((6, 9)), "longrid": np.ones((6, 9)),
         "level": np.array([[700.0]])}
    for k in ("u_c", "v_c"):
        f[k] = rng.normal(size=(4, 3, 5))
    f["currv_anom_c"] = rng.normal(size=(4, 3, 5)) * 1e-6                                   # the real fields' scales, so a 1e-19 change is representable
    f["advcurrv_anom_c"] = rng.normal(size=(4, 3, 5)) * 1e-11
    for k in ("u", "v", "currv_anom"):
        f[k] = rng.normal(size=(4, 6, 9))
    f["u_c"][0, 0, 0] = np.nan
    return f


def _run(tmp_path, name, fields, producer="p", extra=None):
    from scipy.io import savemat
    d = tmp_path / name
    d.mkdir()
    case = d / "tracker_case.mat"
    savemat(str(case), {**fields, **(extra or {}), "rean": "era5", "case_id": name.ljust(32, "x"), "producer_json": producer})
    savemat(str(d / "tracker_port.mat"), {"n": 1.0, "case_id": name.ljust(32, "x"), "lat0": np.array([10.0, 11.0]), "lon0": np.array([20.0, 19.0]), "time0": np.array([1.0, 1.25])})
    record = {"dataset_specific": {"dataset": "era5", "year": 1990, "tracks_sha256": hashlib.sha256((d / "tracker_port.mat").read_bytes()).hexdigest(),
                                   "case_sha256": hashlib.sha256(case.read_bytes()).hexdigest()},
              "protocol_settings": {"manifest_sha256": "m" * 64, "domain": {"lat": [-35.0, 35.0], "lon": [-140.0, 40.0]}}}
    (d / "tracking_era5_1990.json").write_text(json.dumps(record))
    return str(d), str(case)


def test_the_same_fields_pass_and_provenance_strings_are_not_compared(tmp_path):
    C = _load()
    a_dir, a_case = _run(tmp_path, "A", _fields(), producer="laptop")
    b_dir, b_case = _run(tmp_path, "B", _fields(), producer="cluster")                     # another producer string, the same arrays
    r = C.compare(a_dir, a_case, b_dir, b_case, 1990)
    assert r["passed"] and set(r["fields"]) == set(C.FIELDS) and r["not_compared"] == ["rean", "case_id", "producer_json"]
    assert r["fields"]["u_c"]["missing_cells_a"] == 1 == r["fields"]["u_c"]["missing_cells_b"]


def test_a_value_or_a_missing_cell_that_differs_fails_with_its_size_and_place(tmp_path):
    C = _load()
    a_dir, a_case = _run(tmp_path, "A", _fields())
    changed = _fields()
    changed["advcurrv_anom_c"][2, 1, 3] += 1e-19
    changed["currv_anom"][1, 2, 2] = np.nan
    b_dir, b_case = _run(tmp_path, "B", changed)
    r = C.compare(a_dir, a_case, b_dir, b_case, 1990)
    adv, fine = r["fields"]["advcurrv_anom_c"], r["fields"]["currv_anom"]
    assert not r["passed"] and not adv["passed"] and adv["value_cells_differing"] == 1 and adv["first_differing_index"] == [2, 1, 3]
    assert 0 < adv["max_abs_difference"] < 1e-18
    assert not fine["passed"] and fine["missing_mask_cells_differing"] == 1 and fine["value_cells_differing"] == 0
    assert r["fields"]["u_c"]["passed"]


def test_a_case_not_named_by_its_record_or_a_missing_field_is_refused(tmp_path):
    from scipy.io import savemat
    C = _load()
    a_dir, a_case = _run(tmp_path, "A", _fields())
    b_dir, b_case = _run(tmp_path, "B", _fields())
    with pytest.raises(SystemExit, match="is not the case the record in"):
        C.compare(a_dir, b_case, b_dir, b_case, 1990)
    short = _fields()
    del short["advcurrv_anom_c"]
    c_dir, c_case = _run(tmp_path, "C", short)
    with pytest.raises(SystemExit, match="do not both hold"):
        C.compare(a_dir, a_case, c_dir, c_case, 1990)
    assert C.compare_variable(np.zeros((2, 3)), np.zeros((3, 2)))["problem"] == "shapes differ"


def test_every_numeric_variable_is_compared_and_unequal_inventories_are_refused(tmp_path):
    C = _load()
    a_dir, a_case = _run(tmp_path, "A", _fields(), extra={"new_numeric": np.zeros(3)})
    b_dir, b_case = _run(tmp_path, "B", _fields(), extra={"new_numeric": np.ones(3)})       # a variable outside the known list, different
    r = C.compare(a_dir, a_case, b_dir, b_case, 1990)
    assert not r["passed"] and not r["fields"]["new_numeric"]["passed"] and r["other_variables_not_compared"] == []
    c_dir, c_case = _run(tmp_path, "C", _fields())                                            # lacks it
    with pytest.raises(SystemExit, match="hold different numeric variables"):
        C.compare(a_dir, a_case, c_dir, c_case, 1990)


def test_the_bytes_compared_are_the_bytes_hashed(tmp_path, monkeypatch):
    """The case file is replaced on disk after its check: the comparison must still be of
    the checked bytes, so it passes against an identical A and reports the checked digest."""
    import scipy.io
    from scipy.io import savemat
    C = _load()
    a_dir, a_case = _run(tmp_path, "A", _fields())
    b_dir, b_case = _run(tmp_path, "B", _fields())
    real = scipy.io.loadmat
    swapped = {"done": False}

    def swapping_loadmat(*args, **kwargs):
        if not swapped["done"] and kwargs.get("variable_names") == ["time"]:                # the first case field read, after both checks
            changed = _fields()
            changed["currv_anom_c"] += 1.0
            savemat(b_case, {**changed, "rean": "era5", "case_id": "B".ljust(32, "x"), "producer_json": "p"})
            swapped["done"] = True
        return real(*args, **kwargs)

    monkeypatch.setattr(scipy.io, "loadmat", swapping_loadmat)
    r = C.compare(a_dir, a_case, b_dir, b_case, 1990)
    assert swapped["done"] and r["passed"] and r["fields"]["currv_anom_c"]["passed"]
    assert r["b"]["case_sha256"] == json.load(open(os.path.join(b_dir, "tracking_era5_1990.json")))["dataset_specific"]["case_sha256"]


def test_a_difference_reports_its_column_spread_and_the_median_magnitude(tmp_path):
    C = _load()
    a = np.ones((2, 3, 4)) * 2e-6
    b = a.copy(); b[:, :, 1] += 1e-19; b[0, 0, 3] += 1e-19
    r = C.compare_variable(a, b)
    assert r["median_abs_value_a"] == 2e-6 and r["value_cells_differing"] == 7
    assert r["differing_cells_per_last_axis_column"] == {"columns": 4, "min": 0, "max": 6, "columns_with_none": 2}
