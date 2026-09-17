"""The basin-code derivation reads code changes along tracks, not positions alone.

MUTATION LIST, written before the assertions: record transitions at the step BEFORE the
change instead of the entry step; count a NaN gap as a transition; drop the code-1
sample; refuse nothing when the directory is empty.
"""
import importlib.util
import json
import os
import sys

import numpy as np
import pytest

nc = pytest.importorskip("netCDF4")

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "diagnose_qtrack_basins", os.path.join(_HERE, "..", "scripts",
                                           "diagnose_qtrack_basins.py"))
diag = importlib.util.module_from_spec(_spec)
sys.modules["diagnose_qtrack_basins"] = diag
_spec.loader.exec_module(diag)


def write_tracks(path, systems):
    """systems: list of (lon list, lat list, code list), NaN for gaps."""
    n_t = max(len(s[0]) for s in systems)
    d = nc.Dataset(path, "w")
    d.createDimension("time", n_t)
    d.createDimension("system", len(systems))
    for name in ("AEW_lon", "AEW_lat", "basin_des"):
        v = d.createVariable(name, "f8", ("system", "time"), fill_value=np.nan)
        for i, s in enumerate(systems):
            arr = np.full(n_t, np.nan)
            src = {"AEW_lon": s[0], "AEW_lat": s[1], "basin_des": s[2]}[name]
            arr[:len(src)] = src
            v[i, :] = arr
    fb = d.createVariable("first_basin_des", "f8", ("system",), fill_value=np.nan)
    fb[:] = [next(c for c in s[2] if c == c) for s in systems]
    d.close()


@pytest.fixture
def tree(tmp_path):
    # system A: Africa (2) then Atlantic (7), crossing at -17 exactly on the entry step
    a = ([-10.0, -14.0, -17.5, -22.0], [10.0, 10.5, 11.0, 11.5], [2, 2, 7, 7])
    # system B: Caribbean (6) into land (1), with a NaN gap that must NOT count
    b = ([-58.0, -61.0, np.nan, -85.0, -88.0], [15.0, 15.5, np.nan, 16.0, 16.5],
         [7, 6, np.nan, 1, 1])
    write_tracks(str(tmp_path / "y1.nc"), [a, b])
    return tmp_path


def test_transitions_are_recorded_at_the_entry_step_and_gaps_do_not_count(tree):
    files = [str(tree / "y1.nc")]
    r = diag.derive(files, sample_seed=0, sample_size=5)
    assert set(r["transitions"]) == {"2->7", "7->6", "6->1"}
    assert r["transitions"]["2->7"]["entry_lon"]["p50"] == -17.5
    assert r["transitions"]["6->1"]["entry_lon"]["p50"] == -85.0
    assert r["transitions"]["6->1"]["count"] == 1
    assert r["codes"]["2"]["n_steps"] == 2 and r["codes"]["1"]["n_steps"] == 2
    assert r["codes"]["2"]["first_basin_systems"] == 1
    assert r["codes"]["7"]["first_basin_systems"] == 1
    assert sorted(r["code1_sample_lon_lat"]) == [[-88.0, 16.5], [-85.0, 16.0]]


def test_an_empty_directory_is_refused(tmp_path):
    assert diag.main(["--directory", str(tmp_path)]) == 2


def test_the_artifact_carries_provenance(tree, tmp_path):
    out = str(tmp_path / "a.json")
    assert diag.main(["--directory", str(tree), "--out", out]) == 0
    with open(out) as fh:
        a = json.load(fh)
    assert set(a["input_file_sha256"]) == {"y1.nc"}
    assert len(a["source_sha256"]["scripts/diagnose_qtrack_basins.py"]) == 64
    assert a["what_this_establishes"].startswith("The geographic extent")
