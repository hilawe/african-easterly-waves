"""The year-matched sensitivity: its same-sample row and the cells each row states."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

import sensitivity_yearstrat as Y
from aew.environment import cluster_bootstrap_diff

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def eligible():
    """Two years, three months, two longitude bins. One year holds 40 troughs a cell and the other 12."""
    rng = np.random.default_rng(5)
    rows, wave = [], 0
    for year, n in ((1990, 40), (1991, 12)):
        for month in (7, 8, 9):
            for lon0 in (-29.5, 10.5):
                for k in range(n + (3 if (year, month, lon0) == (1991, 9, 10.5) else 0)):   # one cell of 15, so the median is not the mean
                    if k % 3 == 0:
                        wave += 1
                    resp = float(rng.integers(0, 30)) + (8.0 if year == 1990 else 0.0)
                    rows.append(dict(wave=wave, time=pd.Timestamp(year=year, month=month, day=1 + k % 28, hour=6 * (k % 4)),
                                     lon=lon0 + 9.0 * rng.random(), response=resp, inflow_rh=50.0 + 0.3 * resp + rng.normal(0.0, 3.0)))
    rows.append(dict(wave=wave + 1, time=pd.Timestamp("1990-07-05 00:00"), lon=40.0, response=3.0, inflow_rh=51.0))   # in no cell
    return pd.DataFrame(rows)


def run(tmp_path, monkeypatch, table):
    table.to_csv(tmp_path / "eligible_troughs.csv", index=False)
    monkeypatch.setattr(sys, "argv", ["sensitivity_yearstrat.py", "--eligible", str(tmp_path / "eligible_troughs.csv"), "--outdir", str(tmp_path)])
    monkeypatch.setattr(Y, "N_BOOT", 300)
    Y.main()
    return pd.read_csv(tmp_path / "yearstrat_sensitivity.csv").set_index("split")


def test_the_same_sample_row_uses_the_canonical_classes_on_the_troughs_of_the_kept_cells(tmp_path, monkeypatch):
    e = eligible()
    out = run(tmp_path, monkeypatch, e)
    assert list(out.index) == ["monthlon", "yearmonthlon", "monthlon_kept_cells"]
    d = e.rename(columns={"wave": "traj_id", "inflow_rh": "H"}).copy()
    d["year"], d["month"] = d.time.dt.year, d.time.dt.month
    d["lonbin"] = pd.cut(d.lon, Y.EDGES, right=False, labels=False)
    in_kept = ((d.year == 1990) & d.lonbin.notna()).values                  # the cells of 40, the others hold 12
    lo, hi = Y.stratified_terciles_cells(d, ["month", "lonbin"])
    r = out.loc["monthlon_kept_cells"]
    assert (r.n_quiet, r.n_active) == (int((lo & in_kept).sum()), int((hi & in_kept).sum())) and 0 < r.n_quiet < lo.sum()
    assert r["diff"] == pytest.approx(d.H[hi & in_kept].mean() - d.H[lo & in_kept].mean(), abs=1e-6)
    ly, hy = Y.stratified_terciles_cells(d, ["year", "month", "lonbin"])
    assert (out.loc["yearmonthlon"].n_quiet, out.loc["yearmonthlon"].n_active) == (int(ly.sum()), int(hy.sum()))
    assert not ly[~in_kept].any() and not hy[~in_kept].any()               # the year-matched classes lie in the same cells
    assert r["diff"] != pytest.approx(out.loc["yearmonthlon"]["diff"], abs=1e-3)


def test_the_first_two_rows_are_what_they_were_before_the_third_was_added(tmp_path, monkeypatch):
    e = eligible()
    out = run(tmp_path, monkeypatch, e)
    d = e.rename(columns={"wave": "traj_id", "inflow_rh": "H"}).copy()
    d["year"], d["month"] = d.time.dt.year, d.time.dt.month
    d["lonbin"] = pd.cut(d.lon, Y.EDGES, right=False, labels=False)
    rng = np.random.default_rng(0)
    for label, cols in (("monthlon", ["month", "lonbin"]), ("yearmonthlon", ["year", "month", "lonbin"])):
        lo, hi = Y.stratified_terciles_cells(d, cols)
        diff, a, b, _, _ = cluster_bootstrap_diff(d.traj_id[lo], d.H[lo], d.traj_id[hi], d.H[hi], rng, n_boot=300)
        r = out.loc[label]
        assert (r["diff"], r.ci_lo, r.ci_hi) == pytest.approx((diff, a, b), abs=1e-6)


def test_every_row_states_its_cells(tmp_path, monkeypatch):
    out = run(tmp_path, monkeypatch, eligible())
    r = out.loc["monthlon"]
    assert (r.n_cells, r.n_cells_kept, r.n_in_kept_cells, r.median_cell, r.smallest_cell) == (6, 6, 6 * 52 + 3, 52.0, 52)
    for split in ("yearmonthlon", "monthlon_kept_cells"):
        r = out.loc[split]
        assert (r.n_cells, r.n_cells_kept, r.n_in_kept_cells, r.median_cell, r.smallest_cell) == (12, 6, 6 * 40, 27.5, 12)


def test_the_registry_carries_the_cells_of_each_row(tmp_path, monkeypatch):
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import build_registry as R
    out = run(tmp_path, monkeypatch, eligible())
    monkeypatch.chdir(tmp_path)
    os.makedirs("deposit")
    out.reset_index().to_csv("deposit/yearstrat_sensitivity.csv", index=False)
    reg = R.rows_from_yearstrat_sensitivity().set_index("statistic")
    r = reg.loc["yearstrat_monthlon_kept_cells"]
    assert (r.n_cells, r.n_cells_kept, r.n_in_kept_cells, r.median_cell, r.n_total) == (12, 6, 240, 27.5, r.n_quiet + r.n_active)
    assert reg.loc["yearstrat_monthlon", "smallest_cell"] == 52
    out.reset_index()[["split", "diff", "ci_lo", "ci_hi", "n_quiet", "n_active", "year_gap", "significant"]].to_csv("deposit/yearstrat_sensitivity.csv", index=False)
    assert "n_cells" not in R.rows_from_yearstrat_sensitivity().columns     # a table written before the cells were stated
