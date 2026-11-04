"""Tests for scripts/eastern_phase_envelope_probes.py (`EASTERN_PHASE_ENVELOPE_MATCHED_2026-10-27.md`, section 12)."""

import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
sys.path.insert(0, HERE)
import eastern_phase_envelope as PE  # noqa: E402
import eastern_phase_envelope_matched as M  # noqa: E402
import eastern_phase_envelope_probes as PR  # noqa: E402
import pilot_eastern_propagation as P  # noqa: E402
from test_eastern_phase_envelope_matched import seasons_on  # noqa: E402


# ------------------------------------------------------------------ the reference probe

def test_a_reference_equal_to_the_base_column_gives_the_frozen_reading():
    seasons = seasons_on(M.MLONS, -15.0, 0.0, n=4)
    col = int(np.flatnonzero(np.isclose(M.MLONS, -15.0))[0])
    got = PR.reference_results(seasons, [S[:, col] for S in seasons], -15.0, n_boot=9)
    want = M.results(seasons, (-15.0,), M.MLONS, n_boot=9)["-15"]
    assert M._same(json.loads(json.dumps(got)), json.loads(json.dumps(want))) == []


def test_a_different_reference_changes_the_reading_and_keeps_the_field_shape():
    seasons = seasons_on(M.MLONS, -15.0, 0.0, n=4)
    col = int(np.flatnonzero(np.isclose(M.MLONS, -15.0))[0])
    s = PR.sums_with_reference(seasons[0], seasons[0][:, col + 4])
    assert s["sb"].shape == (len(PE.LAGS), M.MLONS.size) and s["sab"].shape == s["sbb"].shape == s["sb"].shape
    got = PR.reference_results(seasons, [S[:, col + 4] for S in seasons], -15.0, n_boot=9)
    want = M.results(seasons, (-15.0,), M.MLONS, n_boot=9)["-15"]
    assert M._same(json.loads(json.dumps(got)), json.loads(json.dumps(want))) != []


def fake_block(v, sp):
    return lambda year, level, rows, cols: {"v": np.full((488, 1, 1), v), "sp": np.full((488, 1, 1), sp)}


@pytest.fixture
def grid(monkeypatch):
    netCDF4 = pytest.importorskip("netCDF4")

    class D:
        def __init__(self, path):
            self.variables = {"latitude": np.arange(35.0, -25.01, -0.5), "longitude": np.arange(-45.0, 75.01, 0.5)}

        def close(self):
            pass
    monkeypatch.setattr(netCDF4, "Dataset", D)


def test_the_point_series_is_rounded_and_refused_below_ground(monkeypatch, grid):
    monkeypatch.setattr(P, "read_season", fake_block(1.234, 101000.0))
    s = PR.point_series(1990, 850, -15.0)
    assert s.shape == (488,) and np.all(s == 1.23)
    monkeypatch.setattr(P, "read_season", fake_block(1.0, 85000.0))
    with pytest.raises(SystemExit, match="not valid at 850 hPa"):
        PR.point_series(1990, 850, -15.0)
    with pytest.raises(SystemExit, match="no single grid point"):
        PR.point_series(1990, 850, -15.25)


def test_the_point_series_refuses_the_held_out_season_before_any_file_is_opened(monkeypatch):
    netCDF4 = pytest.importorskip("netCDF4")

    def opened(*a, **k):
        raise AssertionError("a file was opened")
    monkeypatch.setattr(netCDF4, "Dataset", opened)
    with pytest.raises(SystemExit, match="held-out season"):
        PR.point_series(2005, 850, -15.0)


def test_run_reference_checks_the_band_mean_reading_against_the_stored_record(monkeypatch, tmp_path):
    seasons = seasons_on(M.MLONS, -15.0, 0.0, n=3)
    years = (1990, 1991, 1992)
    col = int(np.flatnonzero(np.isclose(M.MLONS, -15.0))[0])
    monkeypatch.setattr(M, "SHARED_YEARS", years)
    monkeypatch.setattr(M, "PRIMARY", ((850, -15.0),))
    monkeypatch.setattr(M, "band_means", lambda y, level, band: seasons[years.index(y)])
    monkeypatch.setattr(PR, "point_series", lambda y, level, lon: seasons[years.index(y)][:, col + 2])
    band = json.loads(json.dumps(M.results(seasons, (-15.0,), M.MLONS, n_boot=5)))
    path = tmp_path / "matched.json"
    path.write_text(json.dumps({"results": {"primary and transect": {"850": band}}}))
    out = PR.run_reference(str(path), n_boot=5)["850 hPa, -15 E"]
    assert out["band_mean_reference"] == band["-15"] and M._same(out["point_reference"], band["-15"]) != []
    assert -1.0 <= out["reference_correlation"] < 1.0
    band["-15"]["envelope_speed"] += 1e-9
    path.write_text(json.dumps({"results": {"primary and transect": {"850": band}}}))
    with pytest.raises(SystemExit, match="does not reproduce the stored matched record"):
        PR.run_reference(str(path), n_boot=5)


# ------------------------------------------------------------------ the exposure probe

def rd(lags, positions):
    return {"lags": lags, "positions": positions}


def test_touches_reads_the_base_window_and_every_attempted_window():
    lons = np.arange(-10.0, 75.01, 0.5)
    assert PR.touches(rd([], []), 5.0, lons)                         # the window at lag 0 reaches -10
    assert not PR.touches(rd([0, 1], [20.0, 21.0]), 20.0, lons)
    assert PR.touches(rd([0, 1, 2], [50.0, 55.0, 60.0]), 50.0, lons)    # the window around 60 E, tried at lag 3, reaches 75
    last = PR.MAX_LAG
    assert not PR.touches(rd([0, last], [30.0, 60.5]), 30.0, lons)      # no window is tried beyond the largest lag
    assert PR.touches(rd([0, last - 1], [30.0, 60.0]), 30.0, lons)      # exactly at the end column counts


def test_exposure_counts_and_refuses_a_stored_mismatch():
    seasons = seasons_on(PE.LONS, 60.0, 0.0, n=4)
    Zs = [PE.analytic(S) for S in seasons]
    col = int(np.flatnonzero(np.isclose(PE.LONS, 60.0))[0])
    C, E, N = PE.maps(PE.stack([PE.season_sums(Z, col) for Z in Zs]))
    r = json.loads(json.dumps(PE.readings(C, N, col, None, PE.LONS)))
    stored = dict(r, envelope_class="standing")
    out = PR.exposure_one(Zs, 60.0, stored, n_boot=7)
    assert out["pooled_touches"] and out["resamples_touching"] == 7 and out["envelope_class"] == "standing"
    bad = json.loads(json.dumps(stored)); bad["positions"][0] += 1e-9
    with pytest.raises(SystemExit, match="does not reproduce the stored positions"):
        PR.exposure_one(Zs, 60.0, bad, n_boot=7)


def test_exposure_counts_the_resampled_paths_read_in_the_pooled_direction(monkeypatch):
    calls = []

    def fake(C, N, col, sign, lons):
        calls.append(sign)
        if sign is None:
            return {"envelope_speed": 0.0, "positions": [30.0], "lags": [0], "sign": -1}
        i = len(calls) - 2
        return {"envelope_speed": 0.0, "positions": [30.0, 60.0 if i % 3 == 0 else 31.0], "lags": [0, 1], "sign": sign}
    monkeypatch.setattr(PE, "readings", fake)
    Zs = [PE.analytic(S) for S in seasons_on(PE.LONS, 30.0, 0.0, n=3)]
    stored = {"envelope_speed": 0.0, "positions": [30.0], "lags": [0], "sign": -1, "envelope_class": "standing"}
    out = PR.exposure_one(Zs, 30.0, stored, n_boot=7)
    assert not out["pooled_touches"] and out["resamples_touching"] == 3 and calls[1:] == [-1] * 7


# ------------------------------------------------------------------ text and main

def test_the_texts_name_no_mechanism():
    r = {"envelope_class": "westward", "envelope_speed": -1.0, "envelope_interval": [-2.0, -0.5], "envelope_undefined": 0.0,
         "phase_speed": -7.0, "phase_interval": [-7.5, -6.5]}
    ref = PR.reference_text({"850 hPa, -15 E": {"band_mean_reference": r, "point_reference": r, "reference_correlation": 0.89}})
    assert "| 850 hPa, -15 E | point, 10 N | westward | -1.00 (-2.00 to -0.50) | 0.000 | -7.00 (-7.50 to -6.50) |" in ref
    cell = {"envelope_class": "standing", "pooled_touches": True, "resamples_touching": 12}
    exp = PR.exposure_text({"700": {"10 to 14.5 N": {f"{b:g}": cell for b in PE.BASES}}})
    assert "standing, yes, 12" in exp
    for word in ("group", "dispers", "packet", "propagat", "energy", "mechanism"):
        assert word not in (ref + exp).lower()


@pytest.mark.parametrize("mode", ("reference", "exposure"))
def test_records_are_never_overwritten_and_inputs_are_required(tmp_path, mode):
    with pytest.raises(SystemExit, match="required"):
        PR.main([mode, "--out-dir", str(tmp_path)])
    (tmp_path / f"eval60_phase_env_{mode}.json").write_text("{}")
    with pytest.raises(SystemExit, match="never overwritten"):
        PR.main([mode, "--out-dir", str(tmp_path)])
