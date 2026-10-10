"""The calibration audit's population loop is the producer's loop under the archived
order (bit-equal thresholds), differs from it only on the edge ring under the declared
order, and refuses a released artifact whose domain or years are not the run's."""
import importlib.util
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from test_thresholds import COARSE_RES, LAT_RANGE, LON_RANGE, build, tree  # noqa: E402,F401

nc = pytest.importorskip("netCDF4")


def _load():
    spec = importlib.util.spec_from_file_location("pilot_alledge_calibration_under_test", os.path.join(ROOT, "scripts", "pilot_alledge_calibration.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_archived_order_is_the_producers_loop_and_the_declared_order_differs_only_on_the_ring(tree, tmp_path):
    C = _load()
    from aew.v1port import load as L
    from aew.v1port import thresholds as T
    climatology = L.build_climatology([1981, 1982, 1983], tree, "era5")
    os.makedirs(tmp_path / "producer")
    producer = build(tree, str(tmp_path / "producer"), [1982, 1983], [1981, 1982, 1983])
    archived = C.build_population([1982, 1983], tree, "era5", climatology, str(tmp_path / "archived"), order="crop_then_smooth", lat_range=LAT_RANGE, lon_range=LON_RANGE, coarse_resolution=COARSE_RES)
    for which in ("fine", "coarse"):
        a = np.fromfile(getattr(producer, which + "_path"), dtype=np.float64)
        b = np.fromfile(getattr(archived, which + "_path"), dtype=np.float64)
        assert np.array_equal(a, b, equal_nan=True), which                                        # the loop is the producer's loop
    assert C.thresholds_for(archived)["coarse"]["threshold"] == T.exact_threshold(producer, "coarse", T.COARSE_Q)[0]
    declared = C.build_population([1982, 1983], tree, "era5", climatology, str(tmp_path / "declared"), order="smooth_then_crop", lat_range=LAT_RANGE, lon_range=LON_RANGE, coarse_resolution=COARSE_RES)
    for which, shape in (("fine", archived.fine_shape), ("coarse", archived.coarse_shape)):
        fa = np.fromfile(getattr(archived, which + "_path"), dtype=np.float64).reshape(archived.steps, *shape)
        fb = np.fromfile(getattr(declared, which + "_path"), dtype=np.float64).reshape(declared.steps, *shape)
        ring = np.zeros(shape, bool); ring[0, :] = ring[-1, :] = True; ring[:, 0] = ring[:, -1] = True
        assert np.array_equal(fa[:, ~ring], fb[:, ~ring], equal_nan=True), which
        assert not np.array_equal(fa[:, ring], fb[:, ring], equal_nan=True), which
    with pytest.raises(ValueError, match="order must be one of"):
        C.build_population([1982], tree, "era5", climatology, str(tmp_path / "x"), order="crop_twice", lat_range=LAT_RANGE, lon_range=LON_RANGE)


def test_the_command_refuses_a_released_artifact_for_another_domain_and_an_existing_output(tree, tmp_path):
    import json
    C = _load()
    out = tmp_path / "o.json"
    out.write_text("{}")
    args = ["--directory", tree, "--prefix", "era5", "--climatology-years", "1981", "1983", "--population-years", "1982", "1983",
            "--lat-range", str(LAT_RANGE[0]), str(LAT_RANGE[1]), "--lon-range", str(LON_RANGE[0]), str(LON_RANGE[1]), "--scratch", str(tmp_path / "s")]
    with pytest.raises(SystemExit, match="never overwritten"):
        C.main(args + ["--released", "r", "--out", str(out)])
    released = tmp_path / "r.json"
    released.write_text(json.dumps({"coarse": {"threshold": 1.0}, "fine": {"threshold": 2.0}, "domain": {"lat_range": [0.0, 1.0], "lon_range": list(LON_RANGE)},
                                    "climatology_years": [1981, 1983], "population_years": [1982, 1983], "input_file_sha256": {}}))
    with pytest.raises(SystemExit, match="domain or years"):
        C.main(args + ["--released", str(released), "--out", str(tmp_path / "o2.json")])
