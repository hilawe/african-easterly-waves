"""The post-hoc maxima reader of the QTrack look-back (section 10 of its record)."""

import importlib.util
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load():
    spec = importlib.util.spec_from_file_location("lookback_maxima_under_test", os.path.join(ROOT, "scripts", "eastern_qtrack_lookback_maxima.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _grid(spec, M):
    C = np.zeros((len(M.LAGS), len(M.DREL)))
    for k, (d, v) in spec.items():
        C[M.LAGS.index(k), int(np.flatnonzero(M.DREL == d)[0])] = v
    return [[None if not np.isfinite(x) else float(x) for x in row] for row in C]


def test_the_maxima_their_locations_and_the_shifts(tmp_path):
    M = _load()
    obs = _grid({-3: (17.0, 0.61), -1: (8.5, 0.98), 0: (3.5, 1.69)}, M)
    obs[M.LAGS.index(-1)][int(np.flatnonzero(M.DREL == 2.0)[0])] = 0.98
    obs[M.LAGS.index(-5)] = [None] * len(M.DREL)
    shifts = [_grid({-3: (0.0, 0.5)}, M)] * 21 + [_grid({-3: (5.0, 0.61)}, M), _grid({-3: (5.0, 0.62), 0: (1.0, 2.0)}, M)]
    table = M.maxima_table({"results": {"RB1": {"composite": obs, "shift_composites": shifts}}})["RB1"]
    row = {k: (m, d, n, s) for k, m, d, n, s in table}
    assert row[-3] == (0.61, 17.0, 2, 0.62), "a shift equal to the maximum counts as at or above"
    assert row[-1][1] == 2.0 and row[-1][2] == 0, "the westernmost location on a tie"
    assert row[0][2] == 1 and row[-5] == (None, None, None, None)
    text = M.maxima_text({"RB1": table}, "a.json")
    assert "| RB1 | -3 | 0.61 | 17 | 2 | 0.62 |" in text and "| RB1 | -5 |  |  |  |  |" in text
    p = tmp_path / "a.json"; out = tmp_path / "m.md"
    p.write_text(json.dumps({"results": {"RB1": {"composite": obs, "shift_composites": shifts}}}))
    assert M.main(["--archive", str(p), "--out", str(out)]) == 0 and out.read_text() == M.maxima_text({"RB1": table}, "a.json")
    try:
        M.main(["--archive", str(p), "--out", str(out)]); assert False
    except SystemExit as ex:
        assert "never overwritten" in str(ex)


def test_a_missing_shift_leaves_the_count_unavailable():
    M = _load()
    obs = _grid({-2: (10.0, 1.0)}, M)
    shifts = [_grid({-2: (0.0, 0.5)}, M)] * 22 + [[[None] * len(M.DREL) if i == M.LAGS.index(-2) else row for i, row in enumerate(_grid({}, M))]]
    row = {k: (m, d, n, s) for k, m, d, n, s in M.maxima_table({"results": {"RB1": {"composite": obs, "shift_composites": shifts}}})["RB1"]}
    assert row[-2] == (1.0, 10.0, None, 0.5) and row[-1][2] == 23, "unavailable at the lag with a missing shift, counted elsewhere (all 23 tie at zero)"
