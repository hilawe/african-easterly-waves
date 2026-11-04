"""Tests for scripts/eastern_phase_envelope_edge.py (`EASTERN_PHASE_ENVELOPE_MATCHED_2026-10-27.md`, section 10)."""

import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
sys.path.insert(0, HERE)
import eastern_lag_regression as LR  # noqa: E402
import eastern_phase_envelope as PE  # noqa: E402
import eastern_phase_envelope_edge as ED  # noqa: E402
import eastern_phase_envelope_matched as M  # noqa: E402
from test_eastern_phase_envelope_matched import seasons_on  # noqa: E402


def reading(cls, speed, positions):
    return {"envelope_class": cls, "envelope_speed": speed, "envelope_interval": [speed - 0.5, speed + 0.5] if speed is not None else [None, None],
            "phase_speed": -7.0, "positions": positions, "pos_lag_days": 3.0}


def test_the_declared_diagnostic():
    assert ED.LEVEL == 700 and ED.EDGE_BASES == (0.0, 10.0)
    assert ED.EDGE_BANDS == {"10 to 14.5 N": (10.0, 14.5, 10), "15 to 19.5 N": (15.0, 19.5, 10), "20 to 24.5 N": (20.0, 24.5, 10)}
    assert ED.frozen_span() == (-10.0, 75.0)


def test_compare_reports_the_speed_change_and_a_class_change():
    out = ED.compare({"0": reading("westward", -2.0, [1.0, -5.0]), "10": reading("unresolved", None, [])},
                     {"0": reading("westward", -4.0, [1.0, -20.0]), "10": reading("westward", -2.5, [9.0, 3.0])})
    assert out["0"]["speed_change"] == -2.0 and not out["0"]["class_changed"]
    assert out["0"]["frozen"]["westmost"] == -5.0 and out["0"]["wide"]["westmost"] == -20.0
    assert out["10"]["speed_change"] is None and out["10"]["class_changed"] and out["10"]["frozen"]["westmost"] is None


def test_the_text_names_no_mechanism():
    res = {"10 to 14.5 N": ED.compare({"0": reading("westward", -2.0, [1.0]), "10": reading("standing", 0.1, [9.0])},
                                      {"0": reading("westward", -4.0, [-3.0]), "10": reading("westward", -1.5, [4.0])})}
    text = ED.edge_text(res).lower()
    assert "| 10 to 14.5 n | 0 | westward -2.00 (-2.50 to -1.50) | +1.00 | westward -4.00 (-4.50 to -3.50) | -3.00 | no | -2.00 |" in text
    assert "| 10 to 14.5 n | 10 | standing +0.10 (-0.40 to +0.60) | +9.00 | westward -1.50 (-2.00 to -1.00) | +4.00 | yes | -1.60 |" in text
    for word in ("group", "dispers", "packet", "propagat", "energy", "mechanism"):
        assert word not in text


@pytest.fixture(scope="module")
def frozen_case():
    seasons = seasons_on(PE.LONS, 8.0, 0.0, n=4)
    res = json.loads(json.dumps(M.results(seasons, ED.EDGE_BASES, PE.LONS, n_boot=5)))
    return seasons, {"10 to 14.5 N": res}


def test_check_stored_passes_the_same_readings(monkeypatch, frozen_case):
    seasons, stored = frozen_case
    by = {y: {"i": i} for i, y in enumerate(M.ALL_YEARS)}
    seasons = (seasons * 6)[:len(M.ALL_YEARS)]
    monkeypatch.setattr(LR, "season_samples", lambda recs, band: seasons[recs["i"]])
    stored = {"10 to 14.5 N": json.loads(json.dumps(M.results(seasons, ED.EDGE_BASES, PE.LONS, n_boot=5)))}
    assert ED.check_stored(by, "10 to 14.5 N", seasons, stored, n_boot=5) == stored["10 to 14.5 N"]
    changed = json.loads(json.dumps(stored)); changed["10 to 14.5 N"]["0"]["envelope_speed"] += 1e-9
    with pytest.raises(SystemExit, match="do not reproduce the stored ones"):
        ED.check_stored(by, "10 to 14.5 N", seasons, changed, n_boot=5)
    monkeypatch.setattr(LR, "season_samples", lambda recs, band: seasons[recs["i"]] + 0.01)
    with pytest.raises(SystemExit, match="does not reproduce the stored band means"):
        ED.check_stored(by, "10 to 14.5 N", seasons, stored, n_boot=5)


def test_season_means_adds_the_bands_and_uses_the_matched_builder(monkeypatch):
    seen = []
    monkeypatch.setattr(M, "BANDS", dict(M.BANDS))
    monkeypatch.setattr(M, "band_means", lambda y, level, band, span: seen.append((y, level, band, span, M.BANDS[band])) or np.zeros(1))
    ED.season_means("20 to 24.5 N", (-10.0, 75.0))
    assert len(seen) == 24 and all(s[1:] == (700, "20 to 24.5 N", (-10.0, 75.0), (20.0, 24.5, 10)) for s in seen)
    assert [s[0] for s in seen] == list(M.ALL_YEARS)


def test_records_are_never_overwritten(tmp_path):
    (tmp_path / "eval60_phase_env_edge.md").write_text("")
    with pytest.raises(SystemExit, match="never overwritten"):
        ED.main(["--xband", str(tmp_path), "--stored", str(tmp_path / "x.json"), "--out-dir", str(tmp_path)])
