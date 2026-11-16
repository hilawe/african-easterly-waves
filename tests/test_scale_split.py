"""Tests of scripts/scale_split.py on made-up data.

The identity is checked against a hand case and against a direct computation, the
bootstrap against a sample rebuilt wave by wave for the same multiplicities, the merge of
same-time detections on small constructed cases, and the binding checks on a small
made-up deposit that is then damaged in three ways.
"""
import json
import os

import numpy as np
import pandas as pd
import pytest

import scale_split as S
from aew.environment import stratified_terciles


# ------------------------------------------------------------------ a direct reference
def direct_parts(df, background):
    """The four parts computed with plain group means. ``df`` has wave, year, cell, x, lo, hi."""
    d = df.copy()
    bg = d[background & np.isfinite(d.x.values)]
    ym = bg.groupby("year").x.mean()
    cm = bg.groupby("cell").x.mean()
    wm = bg.groupby(["wave", "cell"]).x.mean()
    d["ym"] = d.year.map(ym)
    d["cm"] = d.cell.map(cm)
    d["wm"] = [wm.get((w, c), np.nan) for w, c in zip(d.wave, d.cell)]
    fin = np.isfinite(d.x.values)
    hi, lo = d[d.hi.values & fin], d[d.lo.values & fin]

    def diff(col):
        return hi[col].mean() - lo[col].mean()
    out = dict(total=diff("x"), years=diff("ym"))
    out["cells"] = diff("cm") - diff("ym")
    out["waves"] = diff("wm") - diff("cm")
    out["passages"] = diff("x") - diff("wm")
    return out


def rebuilt(df, mult, background):
    """The sample with every wave repeated by its multiplicity, as separate waves."""
    frames = []
    for w, m in mult.items():
        for k in range(int(m)):
            g = df[df.wave == w].copy()
            g["wave"] = f"{w}-{k}"
            g["_bg"] = background[df.wave.values == w]
            frames.append(g)
    out = pd.concat(frames, ignore_index=True)
    return out, out.pop("_bg").values


def made_up(rng, n_waves=40, with_nan=False):
    rows = []
    for w in range(n_waves):
        year = 1990 + int(rng.integers(0, 4))
        for _ in range(int(rng.integers(1, 9))):
            cell = year * 10 + int(rng.integers(0, 3))
            rows.append((w, year, cell, float(rng.normal(60 + 3 * (year - 1990), 8))))
    d = pd.DataFrame(rows, columns=["wave", "year", "cell", "x"])
    u = rng.random(len(d))
    d["lo"], d["hi"] = u < 0.35, u > 0.6
    if with_nan:
        d.loc[rng.random(len(d)) < 0.1, "x"] = np.nan
    return d


def mats_of(d, background):
    s = S.Sample(d.wave.values, d.year.values, d.cell.values)
    return s, s.measure(d.x.values, d.lo.values, d.hi.values, background)


# ------------------------------------------------------------------ the identity
def test_hand_case_gives_the_parts_worked_out_on_paper():
    d = pd.DataFrame(dict(
        wave=["w1", "w1", "w2", "w3", "w3", "w4", "w5", "w5"],
        year=[1, 1, 1, 1, 1, 2, 2, 2],
        cell=["A", "A", "A", "B", "B", "C", "C", "C"],
        x=[10.0, 14.0, 20.0, 30.0, 34.0, 50.0, 70.0, 60.0],
        hi=[True, False, True, False, True, False, True, False],
        lo=[False, True, False, True, False, True, False, False]))
    s, m = mats_of(d, np.ones(len(d), bool))
    v = {q: float(a[0]) for q, a in S.split_values(m, np.ones((1, s.W))).items()}
    assert v["total"] == pytest.approx(33.5 - 94.0 / 3.0)
    assert v["years"] == pytest.approx(31.2 - 34.4)
    assert v["cells"] == pytest.approx((91.0 / 3.0 - 31.2) - (320.0 / 9.0 - 34.4))
    assert v["waves"] == pytest.approx((32.25 - 91.0 / 3.0) - (94.0 / 3.0 - 320.0 / 9.0))
    assert v["passages"] == pytest.approx(1.25)
    assert v["years and cells"] == pytest.approx(v["years"] + v["cells"])
    assert v["waves and passages"] == pytest.approx(v["waves"] + v["passages"])
    assert sum(v[p] for p in S.PARTS) == pytest.approx(v["total"])


@pytest.mark.parametrize("classified_only", [False, True])
@pytest.mark.parametrize("with_nan", [False, True])
def test_point_values_match_a_direct_computation(classified_only, with_nan):
    d = made_up(np.random.default_rng(3), with_nan=with_nan)
    bg = (d.lo | d.hi).values if classified_only else np.ones(len(d), bool)
    s, m = mats_of(d, bg)
    got = S.split_values(m, np.ones((1, s.W)))
    want = direct_parts(d, bg)
    for q in ("total",) + S.PARTS:
        assert float(got[q][0]) == pytest.approx(want[q], abs=1e-10), q


def test_the_background_changes_the_parts_but_not_the_total():
    d = made_up(np.random.default_rng(5))
    s, a = mats_of(d, np.ones(len(d), bool))
    _, b = mats_of(d, (d.lo | d.hi).values)
    va, vb = S.split_values(a, np.ones((1, s.W))), S.split_values(b, np.ones((1, s.W)))
    assert float(va["total"][0]) == pytest.approx(float(vb["total"][0]))
    assert abs(float(va["years"][0]) - float(vb["years"][0])) > 1e-6


def test_a_classified_observation_outside_the_background_is_refused():
    d = made_up(np.random.default_rng(7))
    s = S.Sample(d.wave.values, d.year.values, d.cell.values)
    bg = np.ones(len(d), bool)
    bg[np.flatnonzero(d.hi.values)[0]] = False
    with pytest.raises(ValueError, match="outside the background"):
        s.measure(d.x.values, d.lo.values, d.hi.values, bg)
    both = d.lo.values.copy()
    both[np.flatnonzero(d.hi.values)[0]] = True
    with pytest.raises(ValueError, match="both classes"):
        s.measure(d.x.values, both, d.hi.values, np.ones(len(d), bool))


# ------------------------------------------------------------------ the bootstrap
@pytest.mark.parametrize("classified_only", [False, True])
def test_every_replicate_equals_the_sample_rebuilt_wave_by_wave(classified_only):
    rng = np.random.default_rng(11)
    d = made_up(rng, n_waves=25, with_nan=True)
    bg = (d.lo | d.hi).values if classified_only else np.ones(len(d), bool)
    s, m = mats_of(d, bg)
    M = rng.multinomial(s.W, np.full(s.W, 1.0 / s.W), size=6).astype(float)
    got = S.split_values(m, M)
    for r in range(M.shape[0]):
        mult = dict(zip(s.wave_ids, M[r]))
        big, big_bg = rebuilt(d, mult, bg)
        want = direct_parts(big, big_bg)
        for q in ("total",) + S.PARTS:
            assert float(got[q][r]) == pytest.approx(want[q], abs=1e-9), (r, q)
        assert sum(float(got[p][r]) for p in S.PARTS) == pytest.approx(float(got["total"][r]), abs=1e-9)


def test_a_cell_emptied_by_the_draw_leaves_no_nan():
    d = made_up(np.random.default_rng(13), n_waves=20)
    s, m = mats_of(d, np.ones(len(d), bool))
    gone = set(d.wave[d.cell == d.cell.iloc[0]])
    M = np.array([[0.0 if w in gone else 2.0 for w in s.wave_ids]])
    got = S.split_values(m, M)
    assert all(np.isfinite(got[q][0]) for q in S.QUANTITIES)
    big, big_bg = rebuilt(d, dict(zip(s.wave_ids, M[0])), np.ones(len(d), bool))
    want = direct_parts(big, big_bg)
    for q in ("total",) + S.PARTS:
        assert float(got[q][0]) == pytest.approx(want[q], abs=1e-9)


def test_a_replicate_without_a_class_is_nan_and_is_not_counted():
    d = made_up(np.random.default_rng(17), n_waves=12)
    s, m = mats_of(d, np.ones(len(d), bool))
    has_hi = set(d.wave[d.hi])
    M = np.array([[0.0 if w in has_hi else 1.0 for w in s.wave_ids], np.ones(s.W)])
    got = S.split_values(m, M)
    assert np.isnan(got["total"][0]) and np.isfinite(got["total"][1])
    held = S.split_values_held(m, M)
    assert np.isnan(held["total"][0]) and np.isnan(held["years"][0]) and np.isfinite(held["total"][1])
    out = S.interval(1.0, np.array([np.nan, 0.0, 1.0, 2.0]))
    assert out["draws_used"] == 3 and out["low"] == pytest.approx(0.05) and out["high"] == pytest.approx(1.95)


def test_multiplicities_follow_the_name_and_the_seed_and_sum_to_the_waves():
    a = np.concatenate(list(S.multiplicities(30, 2500, "whole sample", 0)))
    b = np.concatenate(list(S.multiplicities(30, 2500, "whole sample", 0)))
    c = np.concatenate(list(S.multiplicities(30, 2500, "1983 to 1993", 0)))
    e = np.concatenate(list(S.multiplicities(30, 2500, "whole sample", 1)))
    assert a.shape == (2500, 30) and np.array_equal(a, b)
    assert not np.array_equal(a, c) and not np.array_equal(a, e)
    assert (a.sum(axis=1) == 30).all()


def test_paired_rows_use_one_set_of_replicates():
    d = made_up(np.random.default_rng(19), n_waves=30)
    s, m = mats_of(d, np.ones(len(d), bool))
    d2 = d.assign(x=d.x + 2.0 * d.hi)                       # the active class two units higher
    _, m2 = mats_of(d2, np.ones(len(d), bool))
    rows, reps, paired = S.run_sample(s, "t", {"a": m, "b": m, "c": m2}, 400, 0, pairs=[("a", "b"), ("c", "a")])
    same = paired[("a", "b")]["total"]
    assert same["diff"] == 0.0 and same["low"] == 0.0 and same["high"] == 0.0
    shifted = paired[("c", "a")]["total"]
    assert shifted["diff"] == pytest.approx(2.0) and shifted["low"] == pytest.approx(2.0) and shifted["high"] == pytest.approx(2.0)
    assert same["held_low"] == 0.0 and same["held_high"] == 0.0 and same["replicate_median"] == 0.0
    assert shifted["held_low"] == pytest.approx(2.0) and shifted["held_high"] == pytest.approx(2.0)
    assert rows["a"]["quantities"]["total"]["low"] < rows["a"]["quantities"]["total"]["diff"] < rows["a"]["quantities"]["total"]["high"]
    assert reps["recomputed"]["a"]["years"].shape == (400,) and reps["held"]["a"]["years"].shape == (400,)
    assert rows["a"]["n_quiet"] == int(d.lo.sum()) and rows["a"]["waves_resampled"] == s.W


# ------------------------------------------------------------------ the amendment
def held_direct(d, mult, background):
    """Class differences of the sample's own four terms, each wave weighted by its multiplicity."""
    bg = d[background & np.isfinite(d.x.values)]
    ym = d.year.map(bg.groupby("year").x.mean())
    cm = d.cell.map(bg.groupby("cell").x.mean())
    wm_ = bg.groupby(["wave", "cell"]).x.mean()
    wm = pd.Series([wm_.get((w, c), np.nan) for w, c in zip(d.wave, d.cell)], index=d.index)
    terms = dict(years=ym, cells=cm - ym, waves=wm - cm, passages=d.x - wm, total=d.x)
    wgt = d.wave.map(mult).values.astype(float)
    fin = np.isfinite(d.x.values)
    out = {}
    for q, v in terms.items():
        hi, lo = d.hi.values & fin, d.lo.values & fin
        out[q] = np.average(v.values[hi], weights=wgt[hi]) - np.average(v.values[lo], weights=wgt[lo])
    return out


@pytest.mark.parametrize("classified_only", [False, True])
def test_held_replicates_keep_the_sample_means_and_resample_the_classes(classified_only):
    rng = np.random.default_rng(41)
    d = made_up(rng, n_waves=25, with_nan=True)
    bg = (d.lo | d.hi).values if classified_only else np.ones(len(d), bool)
    s, m = mats_of(d, bg)
    M = rng.multinomial(s.W, np.full(s.W, 1.0 / s.W), size=5).astype(float)
    got = S.split_values_held(m, M)
    again = S.split_values(m, M)
    for r in range(M.shape[0]):
        want = held_direct(d, dict(zip(s.wave_ids, M[r])), bg)
        for q in ("total",) + S.PARTS:
            assert float(got[q][r]) == pytest.approx(want[q], abs=1e-9), (r, q)
        assert float(got["total"][r]) == pytest.approx(float(again["total"][r]), abs=1e-9)
        assert float(got["passages"][r]) == pytest.approx(float(again["passages"][r]), abs=1e-9)
        assert abs(float(got["cells"][r]) - float(again["cells"][r])) > 1e-6
    ones = np.ones((1, s.W))
    for q in S.QUANTITIES:
        assert float(S.split_values_held(m, ones)[q][0]) == pytest.approx(float(S.split_values(m, ones)[q][0]), abs=1e-9)


def test_a_contrast_made_between_waves_alone_moves_toward_cells_under_resampling():
    """The property the first run exposed, on a case built to have no cell effect at all."""
    rng = np.random.default_rng(43)
    rows = []
    for w in range(300):
        effect = rng.normal(0.0, 1.0)
        for _ in range(4):
            rows.append((w, 1990 + w % 3, (1990 + w % 3) * 100 + (w // 3) % 20, 60.0 + effect + rng.normal(0.0, 0.3), effect))
    d = pd.DataFrame(rows, columns=["wave", "year", "cell", "x", "effect"])       # five waves in a cell, placed at random
    d["hi"], d["lo"] = d.effect > 0.4, d.effect < -0.4
    s, m = mats_of(d, np.ones(len(d), bool))
    rows_, reps, _ = S.run_sample(s, "built", {"a": m}, 1500, 0)
    q = rows_["a"]["quantities"]
    assert q["cells"]["diff"] > 0.1 * q["total"]["diff"]                           # the sample's own cell means carry part of it
    shift = q["cells"]["replicate_median"] - q["cells"]["diff"]
    assert shift > 0.25 * q["cells"]["diff"]                                       # and a replicate's carry more
    assert q["waves"]["replicate_median"] < q["waves"]["diff"] - 0.25 * q["cells"]["diff"]
    assert abs(float(np.median(reps["held"]["a"]["cells"])) - q["cells"]["diff"]) < 0.05 * q["cells"]["diff"]
    assert q["cells"]["held_low"] < q["cells"]["diff"] < q["cells"]["held_high"]
    assert abs(q["total"]["replicate_median"] - q["total"]["diff"]) < 0.02 * q["total"]["diff"]
    assert q["total"]["held_low"] == pytest.approx(q["total"]["low"]) and q["passages"]["held_high"] == pytest.approx(q["passages"]["high"])
    # reflecting the declared replicates about the point value gives back the built truth, nothing between cells
    assert abs(2 * q["cells"]["diff"] - q["cells"]["replicate_median"]) < 0.05 * q["total"]["diff"]
    assert 2 * q["cells"]["diff"] - q["cells"]["high"] < 0.0 < 2 * q["cells"]["diff"] - q["cells"]["low"]
    assert 2 * q["waves"]["diff"] - q["waves"]["high"] < q["total"]["diff"] < 2 * q["waves"]["diff"] - q["waves"]["low"]


# ------------------------------------------------------------------ labels
def test_cell_terciles_reproduce_the_canonical_routine_and_its_half_open_bins():
    rng = np.random.default_rng(23)
    n = 3000
    lon = rng.uniform(-30, 40, n)
    lon[:5] = 40.0                                           # on the closed end, in no bin
    month = rng.integers(7, 10, n).astype(float)
    resp = rng.poisson(12, n).astype(float)
    lo, hi = stratified_terciles(resp, lon, S.EDGES, min_bin=30, strat2=month, edges2=S.M_EDGES)
    lo2, hi2 = S.cell_terciles(resp, [month, S.lon_bin(lon)])
    assert np.array_equal(lo, lo2) and np.array_equal(hi, hi2)
    assert not lo2[:5].any() and not hi2[:5].any()
    assert np.isnan(S.lon_bin([40.0, -30.1])).all() and list(S.lon_bin([-30.0, 39.99, 0.0])) == [0.0, 6.0, 3.0]


def test_a_longitude_a_rounding_error_below_an_edge_stays_in_the_bin_below():
    tiny = -1.1102230246251565e-16                           # a trough of the record
    lons = np.array([tiny, 0.0, np.nextafter(10.0, 0.0), 10.0, np.nextafter(-30.0, -40.0), np.nextafter(40.0, 0.0)])
    got = S.lon_bin(lons)
    assert list(got[:4]) == [2.0, 3.0, 3.0, 4.0] and np.isnan(got[4]) and got[5] == 6.0
    for i in range(S.EDGES.size - 1):
        assert np.array_equal((lons >= S.EDGES[i]) & (lons < S.EDGES[i + 1]), got == i)


def test_a_cell_under_the_minimum_takes_no_part():
    resp = np.arange(100.0)
    key = np.r_[np.zeros(71), np.ones(29)]
    lo, hi = S.cell_terciles(resp, [key])
    assert lo[:71].sum() > 0 and hi[:71].sum() > 0
    assert not lo[71:].any() and not hi[71:].any()
    lo3, hi3 = S.cell_terciles(resp, [key], min_bin=29)
    assert lo3[71:].sum() > 0 and hi3[71:].sum() > 0


def test_rows_with_a_missing_key_lie_in_no_cell_however_many_they_are():
    resp = np.arange(100.0)
    key = np.r_[np.zeros(60), np.full(40, np.nan)]
    lo, hi = S.cell_terciles(resp, [key, np.ones(100)])
    assert lo[:60].sum() > 0 and hi[:60].sum() > 0
    assert not lo[60:].any() and not hi[60:].any()


# ------------------------------------------------------------------ the merge
def stamp(h):
    return np.datetime64("2000-08-01T00:00") + np.timedelta64(int(h), "h")


def test_merge_joins_close_detections_of_one_time_stamp_only():
    t = np.array([stamp(0), stamp(0), stamp(3), stamp(3)])
    lon = np.array([20.0, 20.3, 20.0, 25.0])
    lat = np.array([10.0, 10.0, 10.0, 10.0])
    mt, mx, my, removed = S.merge_same_time(t, lon, lat, 1.0)
    assert removed == 1 and mt.size == 3
    first = mx[mt == stamp(0)]
    assert first.size == 1 and first[0] == pytest.approx(20.15)
    assert sorted(mx[mt == stamp(3)]) == [20.0, 25.0]


def test_merge_follows_chains_and_respects_the_distance():
    t = np.array([stamp(0)] * 5)
    lon = np.array([10.0, 10.8, 11.6, 20.0, 21.2])
    lat = np.full(5, 0.0)
    _, mx, _, removed = S.merge_same_time(t, lon, lat, 1.0)
    assert removed == 2 and sorted(np.round(mx, 6)) == [10.8, 20.0, 21.2]
    _, mx2, _, removed2 = S.merge_same_time(t, lon, lat, 2.0)
    assert removed2 == 3 and sorted(np.round(mx2, 6)) == [10.8, 20.6]


def test_merge_scales_longitude_by_latitude_and_leaves_the_outside_alone():
    t = np.array([stamp(0)] * 4)
    lon = np.array([10.0, 11.04, 10.0, 10.2])
    lat = np.array([18.0, 18.0, 30.0, 30.0])                # the second pair is outside the box
    mt, mx, my, removed = S.merge_same_time(t, lon, lat, 1.0)
    assert removed == 1 and mt.size == 3                     # 1.04 degrees of longitude at 18 N is 0.989
    assert sorted(mx[my == 30.0]) == [10.0, 10.2]
    _, _, _, flat = S.merge_same_time(t, lon, np.array([0.0, 0.0, 30.0, 30.0]), 1.0)
    assert flat == 0                                         # the same pair on the equator is 1.04 apart


# ------------------------------------------------------------------ the binding checks
REG_COLS = ["tier", "level", "statistic", "time_rel_h", "unit", "kind", "n_quiet", "n_active", "diff"]


def small_deposit(tmp_path, n_years=3):
    """A made-up deposit that the checks accept: 21 cells of 60 troughs, dense in one year."""
    rng = np.random.default_rng(29)
    rows = []
    wave = 0
    for b in range(7):
        for mo in (7, 8, 9):
            for k in range(60):
                year = 1983 if k < 40 else 1984 + (k % (n_years - 1))
                if k % 3 == 0:
                    wave += 1
                t = pd.Timestamp(year=year, month=mo, day=3 + (k % 25), hour=6 * (k % 4)) + pd.Timedelta(minutes=wave)
                rows.append(dict(time=t, lon=-30 + 10 * b + rng.uniform(0, 9.9), wave=wave, year=year))
    e = pd.DataFrame(rows)
    e["response"] = rng.poisson(12, len(e))
    for col, mu in (("inflow_rh", 58.0), ("box_rh", 68.0), ("tcwv", 45.0)):
        e[col] = rng.normal(mu, 6, len(e)) + 0.2 * e.response
    e["antecedent"] = rng.poisson(15, len(e)).astype(float)
    month = e.time.dt.month.values.astype(float)
    lo, hi = stratified_terciles(e.response.values.astype(float), e.lon.values, S.EDGES, min_bin=30, strat2=month, edges2=S.M_EDGES)
    ly, hy = S.cell_terciles(e.response.values, [e.year.values, month, S.lon_bin(e.lon.values)])
    assert ly.sum() > 0 and hy.sum() > 0
    e.to_csv(tmp_path / "control_model_design_eligible.csv", index=False)
    label = np.where(lo, "MCS-quiet", np.where(hi, "MCS-active", "mid-tercile-or-thin-cell"))
    pd.DataFrame(dict(traj_id=e.wave, time=e.time, lat=10.0, lon=e.lon, year=e.year, month=month, response=e.response.astype(float),
                      label=label)).to_csv(tmp_path / "troughs_pooled.csv", index=False)
    pd.DataFrame(dict(split=["monthlon", "yearmonthlon"], diff=[S.contrast(e.inflow_rh, lo, hi), S.contrast(e.inflow_rh, ly, hy)],
                      n_quiet=[lo.sum(), ly.sum()], n_active=[hi.sum(), hy.sum()])).to_csv(tmp_path / "yearstrat_sensitivity.csv", index=False)
    reg = []
    for col, level, stat, hour in (("inflow_rh", 700.0, "lagrangian_rh", -72.0), ("box_rh", 700.0, "eulerian_box", -24.0),
                                   ("tcwv", np.nan, "tcwv_box", -24.0), ("antecedent", 700.0, "antecedent_convection", -48.0)):
        reg.append(["pooled", level, stat, hour, "", "contrast", lo.sum(), hi.sum(), S.contrast(e[col], lo, hi)])
    for _, a, b in S.EPOCHS:
        m = ((e.year >= a) & (e.year <= b)).values
        reg.append(["pooled", 700.0, f"lagrangian_rh_epoch_{a}_{b}", -72.0, "", "contrast", (lo & m).sum(), (hi & m).sum(),
                    S.contrast(e.inflow_rh, lo & m, hi & m) if (hi & m).any() and (lo & m).any() else 0.0])
    pd.DataFrame(reg, columns=REG_COLS).to_csv(tmp_path / "canonical_numbers.csv", index=False)
    return e, (lo, hi), (ly, hy)


def run_checks(tmp_path, counts=None, n=None):
    e = S.load_inputs(str(tmp_path))
    month = e.month.values
    lab = {"canonical": stratified_terciles(e.response.values.astype(float), e.lon.values, S.EDGES, min_bin=30, strat2=month, edges2=S.M_EDGES),
           "year-matched": S.cell_terciles(e.response.values, [e.year.values, month, e.lonbin.values])}
    if counts is None:
        counts = (e.response.values.astype(float), e.antecedent.values.astype(float))
    expect = (len(e), e.wave.nunique()) if n is None else n
    return S.check_inputs(e, lab, str(tmp_path), counts, expect=expect)


def test_the_checks_accept_a_consistent_deposit(tmp_path):
    small_deposit(tmp_path, n_years=12)
    out = run_checks(tmp_path)
    assert out["the cell-tercile routine reproduces the canonical split"] is True
    assert len(out["registry lagrangian_rh"]) == 2


def test_the_checks_refuse_a_changed_label_a_changed_count_and_a_changed_registry(tmp_path):
    e, (lo, hi), _ = small_deposit(tmp_path, n_years=12)
    with pytest.raises(SystemExit, match="expected"):
        run_checks(tmp_path, n=(11457, 1002))
    tr = pd.read_csv(tmp_path / "troughs_pooled.csv")
    saved = tr.copy()
    i = int(np.flatnonzero(lo)[0])
    tr.loc[i, "label"] = "MCS-active"
    tr.to_csv(tmp_path / "troughs_pooled.csv", index=False)
    with pytest.raises(SystemExit, match="canonical labels differ"):
        run_checks(tmp_path)
    saved.to_csv(tmp_path / "troughs_pooled.csv", index=False)
    resp = e.response.values.astype(float).copy()
    resp[3] += 1
    with pytest.raises(SystemExit, match="response recounted"):
        run_checks(tmp_path, counts=(resp, e.antecedent.values.astype(float)))
    ante = e.antecedent.values.astype(float).copy()
    ante[3] += 1
    with pytest.raises(SystemExit, match="antecedent count recounted"):
        run_checks(tmp_path, counts=(e.response.values.astype(float), ante))
    reg = pd.read_csv(tmp_path / "canonical_numbers.csv")
    keep = reg.copy()
    reg.loc[reg.statistic == "eulerian_box", "diff"] += 1e-3
    reg.to_csv(tmp_path / "canonical_numbers.csv", index=False)
    with pytest.raises(SystemExit, match="box_rh"):
        run_checks(tmp_path)
    keep.to_csv(tmp_path / "canonical_numbers.csv", index=False)
    ys = pd.read_csv(tmp_path / "yearstrat_sensitivity.csv")
    ys.loc[ys.split == "yearmonthlon", "n_quiet"] += 1
    ys.to_csv(tmp_path / "yearstrat_sensitivity.csv", index=False)
    with pytest.raises(SystemExit, match="year-matched labels"):
        run_checks(tmp_path)


def test_the_checks_refuse_a_longitude_bin_that_departs_from_the_canonical_rule(tmp_path):
    small_deposit(tmp_path, n_years=12)
    e = S.load_inputs(str(tmp_path))
    month = e.month.values
    lab = {"canonical": stratified_terciles(e.response.values.astype(float), e.lon.values, S.EDGES, min_bin=30, strat2=month, edges2=S.M_EDGES),
           "year-matched": S.cell_terciles(e.response.values, [e.year.values, month, e.lonbin.values])}
    counts = (e.response.values.astype(float), e.antecedent.values.astype(float))
    S.check_inputs(e, lab, str(tmp_path), counts, expect=(len(e), e.wave.nunique()))
    e.loc[5, "lonbin"] = (e.lonbin[5] + 1) % 7
    with pytest.raises(SystemExit, match="longitude bin"):
        S.check_inputs(e, lab, str(tmp_path), counts, expect=(len(e), e.wave.nunique()))


def test_the_registry_reader_refuses_a_missing_or_doubled_row(tmp_path):
    small_deposit(tmp_path, n_years=12)
    get = S.registry(str(tmp_path))
    assert len(get(700, "eulerian_box", -24.0)) == 3 and len(get(None, "tcwv_box", -24.0)) == 3
    with pytest.raises(SystemExit, match="found 0 times"):
        get(850, "eulerian_box", -24.0)
    reg = pd.read_csv(tmp_path / "canonical_numbers.csv")
    pd.concat([reg, reg[reg.statistic == "tcwv_box"]]).to_csv(tmp_path / "canonical_numbers.csv", index=False)
    with pytest.raises(SystemExit, match="found 2 times"):
        S.registry(str(tmp_path))(None, "tcwv_box", -24.0)


def test_case_tables_are_bound_to_the_classes_and_to_the_registry(tmp_path):
    e, (lo, hi), _ = small_deposit(tmp_path, n_years=12)
    e = S.load_inputs(str(tmp_path))
    rng = np.random.default_rng(31)
    sel = lo | hi
    c = pd.DataFrame(dict(traj_id=e.wave[sel].values, time=e.time[sel].values, label=e.label[sel].values))
    for col, _, _, _ in S.classified_measures(850):
        c[col] = np.round(rng.normal(70, 5, len(c)), 4)
    c.loc[c.index[:7], "rh_m72"] = np.nan
    c.to_csv(tmp_path / "cases_pooled_850.csv", index=False)
    j = e[["wave", "time"]].merge(c.rename(columns={"traj_id": "wave"}), on=["wave", "time"], how="left")
    truth = {}
    for col, _, stat, hour in S.classified_measures(850):
        x = j[col].values
        truth[(stat, hour)] = (S.contrast(x, lo, hi), int((lo & np.isfinite(x)).sum()), int((hi & np.isfinite(x)).sum()))

    def get(level, stat, hour):
        assert level == 850
        return truth[(stat, hour)]
    vals, checks = S.attach_cases(e, str(tmp_path), 850, lo, hi, get)
    assert len(vals) == len(e) and vals.rh_m72.notna().sum() == sel.sum() - 7
    assert len(checks) == len(S.classified_measures(850))

    def off(level, stat, hour):
        d, nq, na = truth[(stat, hour)]
        return (d + 1e-3, nq, na) if stat == "decomp_T" else (d, nq, na)
    with pytest.raises(SystemExit, match="T_m72"):
        S.attach_cases(e, str(tmp_path), 850, lo, hi, off)

    def short(level, stat, hour):
        d, nq, na = truth[(stat, hour)]
        return (d, nq + 1, na) if (stat, hour) == ("lagrangian_rh", -72.0) else (d, nq, na)
    with pytest.raises(SystemExit, match="rh_m72"):
        S.attach_cases(e, str(tmp_path), 850, lo, hi, short)
    c.iloc[:-1].to_csv(tmp_path / "cases_pooled_850.csv", index=False)
    with pytest.raises(SystemExit, match="exactly the classified"):
        S.attach_cases(e, str(tmp_path), 850, lo, hi, get)


def test_finished_rows_are_bound_to_the_registry():
    H, BOX, TCWV, _, ANTE = [name for _, name in S.ELIGIBLE]
    truth = {}

    def get(level, stat, hour):
        return truth.setdefault((level, stat, hour), (round(0.1 * (len(truth) + 1), 6), 100 + len(truth), 90 + len(truth)))
    rows = []

    def add(sample, measure, labels, bg, ref):
        rows.append(dict(sample=sample, measure=measure, labels=labels, background=bg, quantity="total", diff=ref[0], n_quiet=ref[1], n_active=ref[2]))
        rows.append(dict(sample=sample, measure=measure, labels=labels, background=bg, quantity="years", diff=9.9, n_quiet=ref[1], n_active=ref[2]))
    every, cls = "every eligible trough", "classified observations"
    add("whole sample", H, "canonical", every, get(700, "lagrangian_rh", -72.0))
    add("whole sample", H, "canonical", cls, get(700, "lagrangian_rh", -72.0))
    add("whole sample", BOX, "canonical", every, get(700, "eulerian_box", -24.0))
    add("whole sample", TCWV, "canonical", every, get(None, "tcwv_box", -24.0))
    add("whole sample", ANTE, "canonical", every, get(700, "antecedent_convection", -48.0))
    matched = (1.5, 50, 49)
    add("whole sample", H, "year-matched", every, matched)
    for name, a, b in S.EPOCHS:
        add(name, H, "canonical", every, get(700, f"lagrangian_rh_epoch_{a}_{b}", -72.0))
    for level in (700, 850):
        for _, name, stat, hour in S.classified_measures(level):
            if name != H:
                add("whole sample", name, "canonical", cls, get(level, stat, hour))
    n = S.bind_outputs(rows, get, matched)
    assert n == 6 + len(S.EPOCHS) + len(S.classified_measures(700)) - 1 + len(S.classified_measures(850))
    for i in (0, 4, 10, len(rows) - 2):
        assert rows[i]["quantity"] == "total"
        for field, step in (("diff", 1e-3), ("n_quiet", 1)):
            bad = [dict(r) for r in rows]
            bad[i][field] += step
            with pytest.raises(SystemExit, match="against the registry"):
                S.bind_outputs(bad, get, matched)
    with pytest.raises(SystemExit, match="no row for"):
        S.bind_outputs(rows[2:], get, matched)


def test_the_record_is_plain_json(tmp_path):
    d = made_up(np.random.default_rng(37), n_waves=30)
    s, m = mats_of(d, np.ones(len(d), bool))
    rows, _, paired = S.run_sample(s, "t", {"a": m}, 200, 0)
    text = json.dumps(dict(rows=[dict(quantity=q, **v) for q, v in rows["a"]["quantities"].items()]))
    assert set(json.loads(text)["rows"][0]) == {"quantity", "diff", "low", "high", "draws_used", "replicate_median", "held_low", "held_high"}
    assert os.path.basename(S.__file__) == "scale_split.py"


# ------------------------------------------------------------------ added after the independent check
def test_finite_close_refuses_a_missing_number_on_either_side():
    assert S.finite_close(1.0, 1.0 + 5e-6, 1e-5) and not S.finite_close(1.0, 1.0 + 2e-5, 1e-5)
    assert not S.finite_close(np.nan, 1.0, 1e-5) and not S.finite_close(1.0, np.nan, 1e-5) and not S.finite_close(np.inf, np.inf, 1e-5)
    with pytest.raises(ValueError, match="no finite replicate"):
        S.interval(1.0, np.array([np.nan, np.nan]))
    with pytest.raises(ValueError, match="no finite replicate"):
        S.interval(np.nan, np.array([0.0, 1.0]))


def test_the_checks_refuse_a_missing_registry_contrast(tmp_path):
    small_deposit(tmp_path, n_years=12)
    reg = pd.read_csv(tmp_path / "canonical_numbers.csv")
    reg["diff"] = reg["diff"].astype(float)
    reg.loc[reg.statistic == "eulerian_box", "diff"] = np.nan
    reg.to_csv(tmp_path / "canonical_numbers.csv", index=False)
    with pytest.raises(SystemExit, match="box_rh"):
        run_checks(tmp_path)


def test_finished_rows_with_a_missing_total_or_a_second_copy_are_refused():
    H = S.ELIGIBLE[0][1]
    every = "every eligible trough"
    truth = {}

    def get(level, stat, hour):
        return truth.setdefault((level, stat, hour), (0.5, 10, 9))
    rows = []

    def add(sample, measure, labels, bg, ref=(0.5, 10, 9)):
        rows.append(dict(sample=sample, measure=measure, labels=labels, background=bg, quantity="total", diff=ref[0], n_quiet=ref[1], n_active=ref[2]))
    for _, name in S.ELIGIBLE:
        if name != S.ELIGIBLE[3][1]:
            add("whole sample", name, "canonical", every)
    add("whole sample", H, "canonical", "classified observations")
    add("whole sample", H, "year-matched", every)
    for name, _, _ in S.EPOCHS:
        add(name, H, "canonical", every)
    for level in (700, 850):
        for _, name, _, _ in S.classified_measures(level):
            if name != H:
                add("whole sample", name, "canonical", "classified observations")
    assert S.bind_outputs(rows, get, (0.5, 10, 9)) > 20
    missing = [dict(r) for r in rows]
    missing[0]["diff"] = float("nan")
    with pytest.raises(SystemExit, match="against the registry"):
        S.bind_outputs(missing, get, (0.5, 10, 9))
    twice = [dict(rows[0], diff=9.9)] + [dict(r) for r in rows]          # a wrong copy ahead of the right one
    with pytest.raises(SystemExit, match="two rows for"):
        S.bind_outputs(twice, get, (0.5, 10, 9))
    with pytest.raises(ValueError):
        json.dumps(dict(x=float("nan")), allow_nan=False)


def test_a_case_table_with_a_wrong_label_is_refused(tmp_path):
    e, (lo, hi), _ = small_deposit(tmp_path, n_years=12)
    e = S.load_inputs(str(tmp_path))
    sel = lo | hi
    c = pd.DataFrame(dict(traj_id=e.wave[sel].values, time=e.time[sel].values, label=e.label[sel].values))
    for col, _, _, _ in S.classified_measures(850):
        c[col] = 70.0 + np.arange(len(c)) % 7
    j = e[["wave", "time"]].merge(c.rename(columns={"traj_id": "wave"}), on=["wave", "time"], how="left")

    def get(level, stat, hour):
        col = {(m[2], m[3]): m[0] for m in S.classified_measures(850)}[(stat, hour)]
        return S.contrast(j[col].values, lo, hi), int(lo.sum()), int(hi.sum())
    c.to_csv(tmp_path / "cases_pooled_850.csv", index=False)
    S.attach_cases(e, str(tmp_path), 850, lo, hi, get)
    c.loc[c.index[3], "label"] = "MCS-active" if c.label[c.index[3]] == "MCS-quiet" else "MCS-quiet"
    c.to_csv(tmp_path / "cases_pooled_850.csv", index=False)
    with pytest.raises(SystemExit, match="case labels differ"):
        S.attach_cases(e, str(tmp_path), 850, lo, hi, get)


def test_merge_leaves_an_empty_or_wholly_outside_catalog_alone():
    t, x, y, removed = S.merge_same_time(np.array([], dtype="datetime64[ns]"), [], [], 1.0)
    assert t.size == 0 and x.size == 0 and y.size == 0 and removed == 0
    t, x, y, removed = S.merge_same_time(np.array([stamp(0), stamp(0)]), [10.0, 10.1], [30.0, 30.0], 1.0)
    assert removed == 0 and list(x) == [10.0, 10.1] and list(y) == [30.0, 30.0]


def test_merge_uses_the_mean_latitude_of_the_pair_and_the_declared_box():
    t = np.array([stamp(0)] * 2)
    lat = np.array([19.5, 19.0])                                           # the cosine of 19.25, 19.5 and 19.0 differ in the third decimal
    assert S.merge_same_time(t, np.array([10.0, 10.9165]), lat, 1.0)[3] == 1    # 0.9997 by the pair's mean latitude
    assert S.merge_same_time(t, np.array([10.0, 10.9180]), lat, 1.0)[3] == 0    # 1.0010 by the mean, 0.9994 by the higher endpoint alone
    flat = np.array([10.0, 10.0])
    assert S.merge_same_time(t, np.array([-30.0, -29.7]), flat, 1.0)[3] == 1    # well inside the western edge at 45 W
    assert S.merge_same_time(t, np.array([-45.0, -44.8]), flat, 1.0)[3] == 1
    assert S.merge_same_time(t, np.array([-45.4, -45.2]), flat, 1.0)[3] == 0
    assert S.merge_same_time(t, np.array([-45.1, -44.9]), flat, 1.0)[3] == 0    # one of the pair is outside
    assert S.merge_same_time(t, np.array([54.8, 55.0]), flat, 1.0)[3] == 1
    assert S.merge_same_time(t, np.array([10.0, 10.2]), np.array([20.0, 19.8]), 1.0)[3] == 1
    assert S.merge_same_time(t, np.array([10.0, 10.2]), np.array([20.1, 19.9]), 1.0)[3] == 0


def test_tied_counts_leave_the_classes_of_a_cell_unequal():
    resp = np.r_[np.zeros(18), np.ones(12)]
    lo, hi = S.cell_terciles(resp, [np.zeros(30)])
    assert (int(lo.sum()), int(hi.sum())) == (18, 12)
    e = pd.DataFrame(dict(year=[1990] * 30))
    out = S.label_balance(e, {"x": (lo, hi)}, np.zeros(30))["x"]
    assert out["cells_with_unequal_classes"] == 1 and out["largest_count_difference"] == 6
    assert out["active_share_lowest"] == pytest.approx(0.4) and out["active_share_before_1998"] == pytest.approx(0.4)


# ------------------------------------------------------------------ the whole assembly on a made-up world
def brute_count(times_h, lons, ct_h, cx, cy, start_h, length_h, dlon):
    """Detections in [t + start, t + start + length) within dlon of the trough and between 5 and 15 N."""
    order = np.argsort(ct_h, kind="stable")
    st, sx, sy = ct_h[order], cx[order], cy[order]
    out = np.zeros(len(times_h))
    for i, (t, L) in enumerate(zip(times_h, lons)):
        a, b = np.searchsorted(st, [t + start_h, t + start_h + length_h], side="left")
        out[i] = ((np.abs(sx[a:b] - L) <= dlon) & (sy[a:b] >= 5.0) & (sy[a:b] <= 15.0)).sum()
    return out


def brute_merge(ct_h, cx, cy, deg):
    """One detection for every group of same-time detections chained by pairs within ``deg``.

    Only detections between the equator and 20 N and between 45 W and 55 E are merged, the
    declared box, and the rest are passed through.
    """
    inside = (cy >= 0.0) & (cy <= 20.0) & (cx >= -45.0) & (cx <= 55.0)
    keep_t, keep_x, keep_y = list(ct_h[~inside]), list(cx[~inside]), list(cy[~inside])
    order = np.argsort(ct_h[inside], kind="stable")
    st, sx, sy = ct_h[inside][order], cx[inside][order], cy[inside][order]
    cuts = np.flatnonzero(np.r_[True, st[1:] != st[:-1], True])
    for a, b in zip(cuts[:-1], cuts[1:]):
        x, y = sx[a:b], sy[a:b]
        dx = (x[:, None] - x[None, :]) * np.cos(np.deg2rad(0.5 * (y[:, None] + y[None, :])))
        near = np.hypot(dx, y[:, None] - y[None, :]) <= deg
        group = np.arange(x.size)
        while True:                                                         # pass the smallest label along every link until nothing moves
            spread = np.where(near, group[None, :], x.size).min(axis=1)
            if np.array_equal(spread, group):
                break
            group = spread
        for g in np.unique(group):
            keep_t.append(st[a]); keep_x.append(x[group == g].mean()); keep_y.append(y[group == g].mean())
    return np.array(keep_t), np.array(keep_x), np.array(keep_y)


def plain_terciles(resp, keys):
    """Thirds within cells of at least 30 rows, written out plainly as a second opinion."""
    lo, hi = np.zeros(resp.size, bool), np.zeros(resp.size, bool)
    for k in set(keys):
        idx = np.flatnonzero([kk == k for kk in keys])
        if idx.size < 30:
            continue
        a, b = np.quantile(resp[idx], [1.0 / 3.0, 2.0 / 3.0])
        lo[idx[resp[idx] <= a]] = True
        hi[idx[resp[idx] >= b]] = True
    return lo, hi


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """A catalog, a deposit recounted from it, and case tables, small enough to count by hand."""
    tmp = tmp_path_factory.mktemp("world")
    rng = np.random.default_rng(47)
    years = (1992, 1997, 1998, 1999)                                        # 1997 sits just under the era boundary
    ct, cx, cy, sat = [], [], [], []
    for y in years:
        for h in range(0, 92 * 24, 3):
            n = int(rng.integers(20, 30))
            x, la = rng.uniform(-40.0, 50.0, n), rng.uniform(0.0, 20.0, n)
            t = np.datetime64(f"{y}-07-01T00:00") + np.timedelta64(h, "h")
            ct += [t] * n; cx += list(x); cy += list(la); sat += ["MET"] * n
            if y >= 1998:                                                   # a second record of each eastern detection, close by
                twin = x >= 10.0
                ct += [t] * int(twin.sum()); cx += list(x[twin] + 0.3); cy += list(la[twin] + 0.1); sat += ["INS"] * int(twin.sum())
    ct = np.array(ct, dtype="datetime64[ns]")
    raw_x, raw_y = np.array(cx), np.array(cy)
    # the file's own way of storing a position: single precision, a western longitude being the eastern value less 360
    f32 = np.float32
    cx = np.array([float(f32(v + 360.0) - f32(360.0)) if v < 0 else float(f32(v)) for v in raw_x])
    cy = np.array([float(f32(v)) for v in raw_y])
    import xarray as xr
    xr.Dataset(dict(lon=("time", cx), lat=("time", cy)), coords=dict(time=ct)).to_netcdf(tmp / "catalog.nc")
    # the pulled rows: every detection with the position that recorded it, the second records of the east being
    # another position's, with smaller rows the catalog does not hold, rows of a year the deposit lacks, and lines of other kinds
    sat = np.array(sat)
    side = np.random.default_rng(48)
    radius = side.integers(101, 400, ct.size)
    radius[:500] = 101
    stamp_of = pd.DatetimeIndex(ct)
    lines = ["T MET 1992 7 5", "N 1992 36 100", "Y 1992 100 0", "S 1992 MET 07 0", "S 1992 GLOBAL 19921231/ISCCP 0"]
    for k in range(ct.size):
        s = stamp_of[k]
        lines.append(f"R {sat[k]} {s.year} {s.month} {s.day} {s.hour} {k % 900} {radius[k]} {float(raw_y[k])!r} {float(raw_x[k])!r}")
    for k in range(3000):
        s = stamp_of[int(side.integers(0, ct.size))]
        lines.append(f"R {('MET', 'INS', 'GOE')[k % 3]} {s.year} {s.month} {s.day} {s.hour} {k} {int(side.integers(37, 101))} "
                     f"{float(side.uniform(0.0, 20.0))!r} {float(side.uniform(-40.0, 50.0))!r}")
    for k in range(200):
        lines.append(f"R MET 1990 8 {1 + k % 28} {3 * (k % 8)} {k} 150 {float(side.uniform(5.0, 15.0))!r} {float(side.uniform(-30.0, 40.0))!r}")
    import gzip
    with gzip.open(tmp / "pulled.txt.gz", "wt") as f:
        f.write("\n".join(lines) + "\n")
    rows, wave = [], 0
    for y in years:
        for mo in (7, 8, 9):
            for b in range(7):
                for k in range(34):
                    if k % 4 == 0:
                        wave += 1
                    t = pd.Timestamp(year=y, month=mo, day=3 + (k % 24), hour=6 * (k % 4))
                    rows.append(dict(time=t, lon=-30.0 + 10.0 * b + 9.9 * rng.random(), wave=wave, year=y))
    e = pd.DataFrame(rows)
    origin = np.datetime64("1990-01-01T00:00")
    th = (e.time.values.astype("datetime64[ns]") - origin) / np.timedelta64(1, "h")
    ch = (ct - origin) / np.timedelta64(1, "h")
    e["response"] = brute_count(th, e.lon.values, ch, cx, cy, 0.0, 24.0, 8.0).astype(int)
    e["antecedent"] = brute_count(th, e.lon.values, ch, cx, cy, -48.0, 24.0, 10.0)
    e["inflow_rh"] = 55.0 + 0.4 * e.response + 0.5 * (e.year - 1992) + rng.normal(0.0, 4.0, len(e))
    e["box_rh"] = 66.0 + 0.1 * e.response + rng.normal(0.0, 4.0, len(e))
    e["tcwv"] = 45.0 + 0.05 * e.response + rng.normal(0.0, 2.0, len(e))
    month = e.time.dt.month.values.astype(float)
    lo, hi = stratified_terciles(e.response.values.astype(float), e.lon.values, S.EDGES, min_bin=30, strat2=month, edges2=S.M_EDGES)
    lonbin = np.floor((e.lon.values + 30.0) / 10.0)
    ly, hy = plain_terciles(e.response.values.astype(float), list(zip(e.year, month, lonbin)))
    e.to_csv(tmp / "control_model_design_eligible.csv", index=False)
    label = np.where(lo, "MCS-quiet", np.where(hi, "MCS-active", "mid-tercile-or-thin-cell"))
    pd.DataFrame(dict(traj_id=e.wave, time=e.time, lat=10.0, lon=e.lon, year=e.year, month=month, response=e.response.astype(float),
                      label=label)).to_csv(tmp / "troughs_pooled.csv", index=False)
    pd.DataFrame(dict(split=["monthlon", "yearmonthlon"], diff=[S.contrast(e.inflow_rh, lo, hi), S.contrast(e.inflow_rh, ly, hy)],
                      n_quiet=[lo.sum(), ly.sum()], n_active=[hi.sum(), hy.sum()])).to_csv(tmp / "yearstrat_sensitivity.csv", index=False)
    reg = []
    for col, level, stat, hour in (("inflow_rh", 700.0, "lagrangian_rh", -72.0), ("box_rh", 700.0, "eulerian_box", -24.0),
                                   ("tcwv", np.nan, "tcwv_box", -24.0), ("antecedent", 700.0, "antecedent_convection", -48.0)):
        reg.append(["pooled", level, stat, hour, "", "contrast", lo.sum(), hi.sum(), S.contrast(e[col], lo, hi)])
    for _, a, b in S.EPOCHS:
        m = ((e.year >= a) & (e.year <= b)).values
        reg.append(["pooled", 700.0, f"lagrangian_rh_epoch_{a}_{b}", -72.0, "", "contrast", (lo & m).sum(), (hi & m).sum(), S.contrast(e.inflow_rh, lo & m, hi & m)])
    sel = lo | hi
    for level in (700, 850):
        c = pd.DataFrame(dict(traj_id=e.wave[sel].values, time=e.time[sel].values, label=label[sel]))
        for col, _, stat, hour in S.classified_measures(level):
            if (level, col) == (700, "rh_m72"):
                c[col] = np.round(e.inflow_rh[sel].values, 4)
            else:
                c[col] = np.round(60.0 + 0.2 * e.response[sel].values + rng.normal(0.0, 3.0, int(sel.sum())), 4)
            if (level, stat, hour) != (700, "lagrangian_rh", -72.0):
                reg.append(["pooled", float(level), stat, hour, "", "contrast", lo.sum(), hi.sum(), S.contrast(c[col].values, lo[sel], hi[sel])])
        c.to_csv(tmp / f"cases_pooled_{level}.csv", index=False)
    pd.DataFrame(reg, columns=REG_COLS).to_csv(tmp / "canonical_numbers.csv", index=False)
    record = S.build(str(tmp), str(tmp / "catalog.nc"), n_boot=40, seed=0, expect=(len(e), e.wave.nunique()),
                     positions=str(tmp / "pulled.txt.gz"))
    return dict(tmp=tmp, e=e, lo=lo, hi=hi, month=month, lonbin=lonbin, th=th, ch=ch, cx=cx, cy=cy, sat=sat, record=record,
                lines=lines)


def row(record, sample, measure, labels, quantity, bg="every eligible trough"):
    hit = [r for r in record["rows"] if (r["sample"], r["measure"], r["labels"], r["background"], r["quantity"]) == (sample, measure, labels, bg, quantity)]
    assert len(hit) == 1
    return hit[0]


def test_the_assembly_splits_by_year_month_and_longitude(world):
    e, H = world["e"], S.ELIGIBLE[0][1]
    d = pd.DataFrame(dict(wave=e.wave, year=e.year, cell=[f"{y}-{m}-{b}" for y, m, b in zip(e.year, world["month"], world["lonbin"])],
                          x=e.inflow_rh, lo=world["lo"], hi=world["hi"]))
    want = direct_parts(d, np.ones(len(d), bool))
    for q in ("total",) + S.PARTS:
        assert row(world["record"], "whole sample", H, "canonical", q)["diff"] == pytest.approx(want[q], abs=1e-9), q
    coarse = direct_parts(d.assign(cell=[f"{y}-{b}" for y, b in zip(e.year, world["lonbin"])]), np.ones(len(d), bool))
    assert abs(coarse["cells"] - want["cells"]) > 1e-3                      # a cell without its month would give another split


def test_the_assembly_sets_era_thresholds_either_side_of_1998(world):
    e, H = world["e"], S.ELIGIBLE[0][1]
    resp = e.response.values.astype(float)
    lo, hi = plain_terciles(resp, list(zip(e.year >= 1998, world["month"], world["lonbin"])))
    got = row(world["record"], "whole sample", H, "era", "total")
    assert (got["n_quiet"], got["n_active"]) == (int(lo.sum()), int(hi.sum()))
    assert got["diff"] == pytest.approx(S.contrast(e.inflow_rh, lo, hi), abs=1e-9)
    assert world["record"]["labels"]["era"]["quiet"] == int(lo.sum())
    other, _ = plain_terciles(resp, list(zip(e.year >= 1993, world["month"], world["lonbin"])))
    assert not np.array_equal(other, lo) and not np.array_equal(world["lo"], lo)   # another year, or no era at all, would differ
    for name, a, b in S.ERAS:
        m = ((e.year >= a) & (e.year <= b)).values
        assert row(world["record"], name, H, "canonical", "total")["diff"] == pytest.approx(S.contrast(e.inflow_rh, world["lo"] & m, world["hi"] & m), abs=1e-9)


def test_the_assembly_recounts_both_counts_from_the_merged_catalog(world):
    e = world["e"]
    mt, mx, my = brute_merge(world["ch"], world["cx"], world["cy"], 1.0)
    resp_m = brute_count(world["th"], e.lon.values, mt, mx, my, 0.0, 24.0, 8.0)
    ante_m = brute_count(world["th"], e.lon.values, mt, mx, my, -48.0, 24.0, 10.0)
    lo, hi = stratified_terciles(resp_m, e.lon.values, S.EDGES, min_bin=30, strat2=world["month"], edges2=S.M_EDGES)
    rec = world["record"]
    got = row(rec, "whole sample", "antecedent count", "merged 1", "total")
    assert (got["n_quiet"], got["n_active"]) == (int(lo.sum()), int(hi.sum()))
    assert got["diff"] == pytest.approx(S.contrast(ante_m, lo, hi), abs=1e-9)
    assert row(rec, "whole sample", "forward response, count", "merged 1", "total")["diff"] == pytest.approx(S.contrast(resp_m, lo, hi), abs=1e-9)
    assert abs(S.contrast(e.antecedent.values, lo, hi) - S.contrast(ante_m, lo, hi)) > 1e-3   # the unmerged count would give another number
    assert rec["merge"]["1"]["removed"] == world["ch"].size - mt.size > 0
    east_late = ((e.lon >= 10) & (e.year >= 1998)).values
    assert resp_m[east_late].mean() < 0.75 * e.response.values[east_late].mean()


def test_the_assembly_refuses_a_deposit_counted_over_another_window(world, tmp_path):
    import shutil
    for f in os.listdir(world["tmp"]):
        shutil.copy(world["tmp"] / f, tmp_path / f)
    e = pd.read_csv(tmp_path / "control_model_design_eligible.csv")
    e["antecedent"] = brute_count(world["th"], e.lon.values, world["ch"], world["cx"], world["cy"], -48.0, 12.0, 10.0)
    e.to_csv(tmp_path / "control_model_design_eligible.csv", index=False)
    with pytest.raises(SystemExit, match="antecedent"):
        S.build(str(tmp_path), str(tmp_path / "catalog.nc"), n_boot=10, seed=0, expect=(len(e), e.wave.nunique()))


# ------------------------------------------------------------------ the labelings from one satellite position
EV = "every eligible trough"


def one_position_counts(world, code="MET"):
    e, own = world["e"], world["sat"] == code
    resp = brute_count(world["th"], e.lon.values, world["ch"][own], world["cx"][own], world["cy"][own], 0.0, 24.0, 8.0)
    ante = brute_count(world["th"], e.lon.values, world["ch"][own], world["cx"][own], world["cy"][own], -48.0, 24.0, 10.0)
    return resp, ante


def test_stored_coordinates_are_formed_as_the_file_forms_them():
    f32 = np.float32
    x, y = S.stored_coordinates([-58.8, 12.3, -0.1, 180.0, 0.0], [5.0, 14.9, 7.3, 1.1, 0.0])
    assert x[0] == float(f32(301.2) - f32(360.0)) and x[0] != float(f32(-58.8))     # not the western value taken directly
    assert x[1] == float(f32(12.3)) and x[2] == float(f32(359.9) - f32(360.0)) and x[3] == 180.0 and x[4] == 0.0
    assert list(y) == [float(f32(v)) for v in (5.0, 14.9, 7.3, 1.1, 0.0)]


def test_the_pulled_rows_are_bound_to_the_catalog(world):
    rec, sat = world["record"], world["sat"]
    ck = rec["checks"]
    assert ck["positions: the rows of every position equal the catalog inside the pulled box, one for one"] == world["ch"].size
    assert ck["positions: rows by position"] == {"MET": int((sat == "MET").sum()), "INS": int((sat == "INS").sum())}
    assert ck["positions: the rows of every position give the deposited response and antecedent count"] == len(world["e"])
    assert rec["position"]["rows"] == world["ch"].size and rec["position"]["rows_of_position"] == int((sat == "MET").sum())
    assert (sat == "INS").sum() > 1000 and rec["position"]["position"] == "MET" and rec["position"]["smallest_radius_km"] == 101


def test_the_assembly_recounts_both_counts_from_one_position(world):
    e, rec = world["e"], world["record"]
    resp_p, ante_p = one_position_counts(world)
    lo, hi = stratified_terciles(resp_p, e.lon.values, S.EDGES, min_bin=30, strat2=world["month"], edges2=S.M_EDGES)
    for measure, x in (("forward response, count", resp_p), ("antecedent count", ante_p), (S.ELIGIBLE[0][1], e.inflow_rh.values),
                       (S.ELIGIBLE[1][1], e.box_rh.values), (S.ELIGIBLE[2][1], e.tcwv.values)):
        got = row(rec, "whole sample", measure, "MET alone", "total")
        assert (got["n_quiet"], got["n_active"]) == (int(lo.sum()), int(hi.sum())), measure
        assert got["diff"] == pytest.approx(S.contrast(x, lo, hi), abs=1e-9), measure
    assert rec["labels"]["MET alone"]["quiet"] == int(lo.sum()) and rec["labels"]["MET alone"]["active"] == int(hi.sum())
    # the pooled counts, the pooled classes or the other position would each give other numbers
    assert abs(S.contrast(e.antecedent.values, lo, hi) - S.contrast(ante_p, lo, hi)) > 1e-3
    assert abs(S.contrast(e.response.values, lo, hi) - S.contrast(resp_p, lo, hi)) > 1e-3
    assert not np.array_equal(lo, world["lo"])
    other, _ = one_position_counts(world, "INS")
    assert not np.array_equal(stratified_terciles(other, e.lon.values, S.EDGES, min_bin=30, strat2=world["month"], edges2=S.M_EDGES)[0], lo)
    east_late = ((e.lon >= 10) & (e.year >= 1998)).values
    assert resp_p[east_late].mean() < 0.6 * e.response.values[east_late].mean()
    for name, a, b in S.ERAS:                                               # the eras are split on the same classes
        m = ((e.year >= a) & (e.year <= b)).values
        assert row(rec, name, S.ELIGIBLE[0][1], "MET alone", "total")["diff"] == pytest.approx(S.contrast(e.inflow_rh, lo & m, hi & m), abs=1e-9)


def test_the_year_matched_classes_from_one_position(world):
    e, rec = world["e"], world["record"]
    resp_p, ante_p = one_position_counts(world)
    lo, hi = plain_terciles(resp_p, list(zip(e.year, world["month"], world["lonbin"])))
    for measure, x in ((S.ELIGIBLE[0][1], e.inflow_rh.values), ("antecedent count", ante_p), ("forward response, count", resp_p)):
        got = row(rec, "whole sample", measure, "MET alone, year-matched", "total")
        assert (got["n_quiet"], got["n_active"]) == (int(lo.sum()), int(hi.sum())), measure
        assert got["diff"] == pytest.approx(S.contrast(x, lo, hi), abs=1e-9), measure
    pooled, _ = plain_terciles(e.response.values.astype(float), list(zip(e.year, world["month"], world["lonbin"])))
    without_year, _ = plain_terciles(resp_p, list(zip(world["month"], world["lonbin"])))
    assert not np.array_equal(pooled, lo) and not np.array_equal(without_year, lo)


def test_what_one_position_did_to_the_counts_and_the_classes(world):
    e, rec = world["e"], world["record"]
    resp_p, ante_p = one_position_counts(world)
    lo, hi = stratified_terciles(resp_p, e.lon.values, S.EDGES, min_bin=30, strat2=world["month"], edges2=S.M_EDGES)
    ly, hy = plain_terciles(resp_p, list(zip(e.year, world["month"], world["lonbin"])))
    late = (e.year >= 1998).values
    b = world["lonbin"] == 5
    got = rec["position"]["by_bin"]["20 to 30"]
    assert got["response"]["before"] == pytest.approx(resp_p[b & ~late].mean()) and got["response"]["after"] == pytest.approx(resp_p[b & late].mean())
    assert got["response"]["pooled_after"] == pytest.approx(e.response.values[b & late].mean())
    assert got["antecedent"]["after"] == pytest.approx(ante_p[b & late].mean()) and got["antecedent"]["pooled_after"] == pytest.approx(e.antecedent.values[b & late].mean())
    assert got["response"]["pooled_after"] > 1.5 * got["response"]["after"]
    m = ((e.lon >= 10) & (e.lon < 40)).values & late
    share = rec["position"]["active_share"]["10 E to 40 E, 1998 to 2007"]
    assert share["MET alone"] == pytest.approx(hi[m].sum() / (lo[m] | hi[m]).sum())
    assert share["MET alone, year-matched"] == pytest.approx(hy[m].sum() / (ly[m] | hy[m]).sum())
    assert share["canonical"] == pytest.approx(world["hi"][m].sum() / (world["lo"][m] | world["hi"][m]).sum())
    assert share["canonical"] > share["MET alone"] + 0.2
    tab = rec["position"]["crosstab"]
    assert sum(tab.values()) == len(e)
    assert tab["MCS-quiet to MCS-active"] == int((world["lo"] & hi).sum()) and tab["MCS-active to MCS-quiet"] == int((world["hi"] & lo).sum())
    assert tab["MCS-active to neither class"] == int((world["hi"] & ~lo & ~hi).sum())
    same = (world["lo"] == lo) & (world["hi"] == hi)
    assert rec["position"]["agree_with_canonical"] == pytest.approx(same.mean()) and 0.3 < same.mean() < 0.999


def test_the_paired_rows_from_one_position(world):
    rec, H, BOX = world["record"], S.ELIGIBLE[0][1], S.ELIGIBLE[1][1]
    paired = {(p["sample"], tuple(p["first"]), tuple(p["second"]), p["quantity"]): p for p in rec["paired"]}
    for L in S.ONE_POSITION:
        g = paired[("whole sample", (H, L, EV), (BOX, L, EV), "total")]
        assert g["diff"] == pytest.approx(row(rec, "whole sample", H, L, "total")["diff"] - row(rec, "whole sample", BOX, L, "total")["diff"], abs=1e-12)
        g = paired[("1983 to 1997 less 1998 to 2007", (H, L, EV), (H, L, EV), "waves")]
        assert g["diff"] == pytest.approx(row(rec, "1983 to 1997", H, L, "waves")["diff"] - row(rec, "1998 to 2007", H, L, "waves")["diff"], abs=1e-12)
    g = paired[("whole sample", (H, "canonical", EV), (H, "MET alone", EV), "total")]
    want = row(rec, "whole sample", H, "canonical", "total")["diff"] - row(rec, "whole sample", H, "MET alone", "total")["diff"]
    assert g["diff"] == pytest.approx(want, abs=1e-12) and abs(want) > 1e-6
    assert g["low"] <= g["replicate_median"] <= g["high"]


def test_without_the_pulled_rows_nothing_comes_from_one_position_and_no_other_row_moves(world):
    e = world["e"]
    rec = S.build(str(world["tmp"]), str(world["tmp"] / "catalog.nc"), n_boot=40, seed=0, expect=(len(e), e.wave.nunique()))
    assert "position" not in rec and not any(k.startswith("positions:") for k in rec["checks"])
    assert not any(r["labels"] in S.ONE_POSITION for r in rec["rows"]) and set(rec["labels"]) == set(world["record"]["labels"]) - set(S.ONE_POSITION)
    assert rec["rows"] == [r for r in world["record"]["rows"] if r["labels"] not in S.ONE_POSITION]
    assert rec["paired"] == [p for p in world["record"]["paired"] if p["first"][1] not in S.ONE_POSITION and p["second"][1] not in S.ONE_POSITION]
    assert rec["merge"] == world["record"]["merge"]


def pulled_variant(world, tmp_path, change):
    """The made-up deposit beside a changed copy of the pulled rows; returns the arguments of the build."""
    import gzip
    import shutil
    for f in os.listdir(world["tmp"]):
        shutil.copy(world["tmp"] / f, tmp_path / f)
    with gzip.open(tmp_path / "pulled.txt.gz", "wt") as f:
        f.write("\n".join(change(list(world["lines"]))) + "\n")
    e = world["e"]
    return dict(dep=str(tmp_path), csct=str(tmp_path / "catalog.nc"), n_boot=10, seed=0, expect=(len(e), e.wave.nunique()),
                positions=str(tmp_path / "pulled.txt.gz"))


def first_row(lines, wanted):
    """Index of the first pulled row of at least 101 km whose latitude satisfies ``wanted``."""
    for i, line in enumerate(lines):
        p = line.split()
        if p[0] == "R" and int(p[7]) >= 101 and p[2] != "1990" and wanted(float(p[8])):
            return i
    raise AssertionError("no such row")


def test_pulled_rows_that_are_not_the_catalog_are_refused(world, tmp_path):
    outside = lambda lat: lat > 16.0 or lat < 4.0                           # a row no count can see, so only the comparison with the catalog can refuse it

    def dropped(lines):
        del lines[first_row(lines, outside)]
        return lines

    def moved(lines):
        i = first_row(lines, outside)
        p = lines[i].split()
        p[9] = repr(float(p[9]) + 0.5)
        lines[i] = " ".join(p)
        return lines

    def shrunk(lines):
        i = first_row(lines, outside)
        p = lines[i].split()
        p[7] = "100"
        lines[i] = " ".join(p)
        return lines

    def added(lines):
        return lines + ["R MET 1998 8 1 0 1 150 18.5 20.0"]

    for k, change in enumerate((dropped, moved, shrunk, added)):
        sub = tmp_path / f"v{k}"
        sub.mkdir()
        with pytest.raises(SystemExit, match="do not equal the catalog"):
            S.build(**pulled_variant(world, sub, change))


def test_a_pulled_row_of_an_unknown_position_or_a_broken_row_is_refused(world, tmp_path):
    def unknown(lines):
        i = first_row(lines, lambda lat: True)
        lines[i] = lines[i].replace("R MET", "R XYZ", 1).replace("R INS", "R XYZ", 1)
        return lines

    def short(lines):
        i = first_row(lines, lambda lat: True)
        lines[i] = " ".join(lines[i].split()[:9])
        return lines

    for k, (change, why) in enumerate(((unknown, "unknown position"), (short, "fields"))):
        sub = tmp_path / f"v{k}"
        sub.mkdir()
        with pytest.raises(SystemExit, match=why):
            S.build(**pulled_variant(world, sub, change))


def test_counts_that_reach_outside_the_pulled_box_are_refused(world, monkeypatch):
    e = world["e"]
    monkeypatch.setattr(S, "PULL_BOX", (-2.0, 22.0, -39.0, 60.0))
    with pytest.raises(SystemExit, match="reach outside"):
        S.build(str(world["tmp"]), str(world["tmp"] / "catalog.nc"), n_boot=10, seed=0, expect=(len(e), e.wave.nunique()),
                positions=str(world["tmp"] / "pulled.txt.gz"))


def test_build_reproduces_the_retained_record_when_the_inputs_are_named():
    """Opt-in, since the deposit and the catalog are outside the tree the suite runs in.

    With AEW_A1_POSITIONS naming the pulled source rows, the labelings from one position are compared as well.
    """
    deposit, catalog = os.environ.get("AEW_A1_DEPOSIT"), os.environ.get("AEW_A1_CSCT")
    if not deposit or not catalog:
        pytest.skip("set AEW_A1_DEPOSIT and AEW_A1_CSCT to run the A1 assembly against its retained record")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ref = json.load(open(os.path.join(root, "docs", "submission", "revision1", "a1", "scale_split_record.json")))
    positions = os.environ.get("AEW_A1_POSITIONS")
    got = S.build(deposit, catalog, n_boot=200, seed=0, positions=positions)
    if not positions:                                                       # without the pulled rows, the record less what comes from them
        ref.pop("position", None)
        ref["checks"] = {k: v for k, v in ref["checks"].items() if not k.startswith("positions:")}
        for section in ("labels", "balance"):
            ref[section] = {k: v for k, v in ref[section].items() if k not in S.ONE_POSITION}
        ref["rows"] = [r for r in ref["rows"] if r["labels"] not in S.ONE_POSITION]
        ref["paired"] = [r for r in ref["paired"] if r["first"][1] not in S.ONE_POSITION and r["second"][1] not in S.ONE_POSITION]
    for section in ("checks", "labels", "merge", "balance") + (("position",) if positions else ()):
        assert got[section] == ref[section], section
    drawn = {"low", "high", "held_low", "held_high", "replicate_median", "draws_used"}
    for section in ("rows", "paired"):
        assert len(got[section]) == len(ref[section])
        for a, b in zip(got[section], ref[section]):
            assert a.keys() == b.keys()
            for k in b.keys() - drawn:
                assert a[k] == pytest.approx(b[k], abs=1e-9) if isinstance(b[k], float) else a[k] == b[k], (section, k)


# ------------------------------------------------------------------ a second catalog on the same troughs
ALT = "another catalog"


def alt_world(world, tmp_path, change=None, counts_rows=True):
    """A copy of the made-up deposit with a second catalog, the rows of one position, and the deposit's build on it."""
    import shutil
    import xarray as xr
    for name in os.listdir(world["tmp"]):
        if name.endswith(".csv") or name == "catalog.nc":
            shutil.copy(world["tmp"] / name, tmp_path / name)
    own = world["sat"] == "MET"
    origin = np.datetime64("1990-01-01T00:00")
    ct = origin + (world["ch"][own] * 3600.0).astype("timedelta64[s]")
    xr.Dataset(dict(lon=("time", world["cx"][own]), lat=("time", world["cy"][own])),
               coords=dict(time=ct.astype("datetime64[ns]"))).to_netcdf(tmp_path / "second.nc")
    e = world["e"]
    resp = brute_count(world["th"], e.lon.values, world["ch"][own], world["cx"][own], world["cy"][own], 0.0, 24.0, 8.0)
    ante = brute_count(world["th"], e.lon.values, world["ch"][own], world["cx"][own], world["cy"][own], -48.0, 24.0, 10.0)
    table = pd.DataFrame(dict(traj_id=e.wave, time=e.time, response=resp))
    if change is not None:
        table = change(table)
    os.makedirs(tmp_path / S.ALT_DEPOSIT, exist_ok=True)
    if table is not None:
        table.to_csv(tmp_path / S.ALT_DEPOSIT / "troughs_pooled.csv", index=False)
    if counts_rows:
        reg = pd.read_csv(tmp_path / "canonical_numbers.csv")
        extra = pd.DataFrame([["pooled", np.nan, "n_corridor_troughs", np.nan, "count", "point", np.nan, np.nan, float(len(e))],
                              ["pooled", np.nan, "n_waves", np.nan, "count", "point", np.nan, np.nan, float(e.wave.nunique())],
                              ["dev", np.nan, "n_corridor_troughs", np.nan, "count", "point", np.nan, np.nan, 7.0],
                              ["dev", np.nan, "n_waves", np.nan, "count", "point", np.nan, np.nan, 3.0]], columns=REG_COLS)
        pd.concat([reg, extra]).to_csv(tmp_path / "canonical_numbers.csv", index=False)
    return dict(dep=str(tmp_path), csct=str(tmp_path / "catalog.nc"), alt=str(tmp_path / "second.nc"), resp=resp, ante=ante)


@pytest.fixture(scope="module")
def second(world, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("second")
    w = alt_world(world, tmp)
    w["record"] = S.build(w["dep"], w["csct"], n_boot=40, seed=0, alt_catalog=w["alt"], alt_label=ALT)
    return w


def test_the_assembly_recounts_both_counts_from_the_second_catalog(world, second):
    e, rec = world["e"], second["record"]
    lo, hi = stratified_terciles(second["resp"], e.lon.values, S.EDGES, min_bin=30, strat2=world["month"], edges2=S.M_EDGES)
    assert not np.array_equal(second["resp"], e.response.values) and not np.array_equal(second["ante"], e.antecedent.values)
    assert (rec["labels"][ALT]["quiet"], rec["labels"][ALT]["active"]) == (int(lo.sum()), int(hi.sum()))
    for measure, values in (("forward response, count", second["resp"]), ("antecedent count", second["ante"]),
                            ("inflow RH at -72 h, 700 hPa", e.inflow_rh.values)):
        r = row(rec, "whole sample", measure, ALT, "total")
        assert r["diff"] == pytest.approx(S.contrast(values, lo, hi), abs=1e-9) and (r["n_quiet"], r["n_active"]) == (int(lo.sum()), int(hi.sum()))
    # the same measures under the deposit's own labels still use the deposit's own counts
    assert row(rec, "whole sample", "forward response, count", "canonical", "total")["diff"] == pytest.approx(
        S.contrast(e.response.values, world["lo"], world["hi"]), abs=1e-9)


def test_the_year_matched_classes_from_the_second_catalog(world, second):
    e, name = world["e"], S.alt_names(ALT)[1]
    ly, hy = plain_terciles(second["resp"].astype(float), list(zip(e.year, world["month"], world["lonbin"])))
    r = row(second["record"], "whole sample", "inflow RH at -72 h, 700 hPa", name, "total")
    assert (r["n_quiet"], r["n_active"]) == (int(ly.sum()), int(hy.sum())) and r["diff"] == pytest.approx(S.contrast(e.inflow_rh, ly, hy), abs=1e-9)
    r = row(second["record"], "whole sample", "forward response, count", name, "total")
    assert r["diff"] == pytest.approx(S.contrast(second["resp"], ly, hy), abs=1e-9)


def test_the_second_catalog_adds_its_rows_and_moves_no_other(world, second):
    e, names = world["e"], S.alt_names(ALT)
    plain = S.build(second["dep"], second["csct"], n_boot=40, seed=0)
    rec = second["record"]
    assert set(rec["labels"]) == set(plain["labels"]) | set(names) and "alt_catalog" not in plain
    assert [r for r in rec["rows"] if r["labels"] not in names] == plain["rows"]
    assert [p for p in rec["paired"] if p["first"][1] not in names and p["second"][1] not in names] == plain["paired"]
    H, BOX, bg = "inflow RH at -72 h, 700 hPa", "fixed box RH at -24 h, 700 hPa", "every eligible trough"
    firsts = {(tuple(p["first"]), tuple(p["second"])) for p in rec["paired"] if p["sample"] == "whole sample"}
    assert {((H, L, bg), (BOX, L, bg)) for L in names} <= firsts and ((H, "canonical", bg), (H, names[0], bg)) in firsts
    for sample in ("1983 to 1993", "1998 to 2007", "10 E to 40 E"):
        assert row(rec, sample, H, names[0], "total")["n_quiet"] > 0


def test_what_the_second_catalog_did_to_the_classes(world, second):
    import hashlib
    e, a = world["e"], second["record"]["alt_catalog"]
    lo, hi = stratified_terciles(second["resp"], e.lon.values, S.EDGES, min_bin=30, strat2=world["month"], edges2=S.M_EDGES)
    was = np.where(world["lo"], 0, np.where(world["hi"], 2, 1))
    now = np.where(lo, 0, np.where(hi, 2, 1))
    assert sum(a["crosstab"].values()) == len(e) and a["crosstab"]["MCS-active to MCS-quiet"] == int(((was == 2) & (now == 0)).sum())
    assert a["agree_with_canonical"] == pytest.approx(float((was == now).mean())) and a["agree_with_canonical"] < 1.0
    assert a["label"] == ALT and a["detections"] == int((world["sat"] == "MET").sum())
    assert a["sha256"] == hashlib.sha256(open(second["alt"], "rb").read()).hexdigest()
    assert a["catalog_sha256"] == hashlib.sha256(open(second["csct"], "rb").read()).hexdigest() != a["sha256"]
    late = e.year.values >= S.ERA_YEAR
    m = world["lonbin"] == 5
    assert a["by_bin"]["20 to 30"]["response"]["after"] == pytest.approx(float(second["resp"][m & late].mean()))
    assert a["by_bin"]["20 to 30"]["response"]["pooled_after"] == pytest.approx(float(e.response.values[m & late].mean()))


def swap_two(table):
    i, k = 0, int(np.flatnonzero(table.response.values != table.response.values[0])[0])
    table = table.copy()
    table.loc[[i, k], "response"] = table.response.values[[k, i]]
    return table


@pytest.mark.parametrize("change, message", [
    (lambda t: t.assign(response=np.r_[t.response.values[0] + 1, t.response.values[1:]]), "differs from the deposit's build"),
    (swap_two, "differs from the deposit's build"),
    (lambda t: t.iloc[1:], "do not pair one to one"),
    (lambda t: None, "is absent"),
])
def test_a_second_catalog_that_is_not_the_deposits_build_is_refused(world, tmp_path, change, message):
    w = alt_world(world, tmp_path, change=change)
    with pytest.raises(SystemExit, match=message):
        S.build(w["dep"], w["csct"], n_boot=10, seed=0, alt_catalog=w["alt"], alt_label=ALT, check_only=True)


@pytest.mark.parametrize("label", ["canonical", "year-matched", "era", "merged 1", S.ONE_POSITION[0]])
def test_a_second_catalog_under_the_name_of_another_labeling_is_refused(world, tmp_path, label):
    w = alt_world(world, tmp_path)
    with pytest.raises(SystemExit, match="belongs to another labeling"):
        S.build(w["dep"], w["csct"], n_boot=10, seed=0, alt_catalog=w["alt"], alt_label=label, check_only=True)


def test_the_sample_size_is_read_from_the_deposits_own_count_rows(world, tmp_path):
    e = world["e"]
    w = alt_world(world, tmp_path)
    assert S.expected_sample(w["dep"]) == (len(e), e.wave.nunique())
    reg = pd.read_csv(tmp_path / "canonical_numbers.csv")
    pooled_n = (reg.tier == "pooled") & (reg.statistic == "n_corridor_troughs")
    for plant, message in ((reg.assign(diff=np.where(pooled_n, len(e) - 1.0, reg["diff"])), "expected"),
                           (reg[~((reg.tier == "pooled") & (reg.statistic == "n_waves"))], "n_waves 0 times"),
                           (pd.concat([reg, reg[pooled_n]]), "n_corridor_troughs 2 times"),
                           (reg.assign(diff=np.where(pooled_n, np.nan, reg["diff"])), "without a value")):
        plant.to_csv(tmp_path / "canonical_numbers.csv", index=False)
        with pytest.raises(SystemExit, match=message):
            S.build(w["dep"], w["csct"], n_boot=10, seed=0, check_only=True)
    reg.to_csv(tmp_path / "canonical_numbers.csv", index=False)
    assert "checks" in S.build(w["dep"], w["csct"], n_boot=10, seed=0, check_only=True)
    with pytest.raises(SystemExit, match="expected"):
        S.build(w["dep"], w["csct"], n_boot=10, seed=0, check_only=True, expect=(len(e), e.wave.nunique() + 1))
