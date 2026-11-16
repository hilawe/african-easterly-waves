"""The four small analyses a reviewer asked for: counts before and after deduplication, the
moist tail at other percentiles, an interval for a centroid difference, and the alignment
check on a regional grid. Every test runs on made-up data small enough to work by hand."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

import check_track_alignment_region as A
import dedup_counts as D
import fig_ct_wave_following as F
import sensitivity_exceedance as E
from aew.data.aewc import Troughs
from aew.environment import cluster_bootstrap_diff

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))


# ------------------------------------------------------------------ counts before and after deduplication
def test_a_stage_counts_observations_trajectories_and_distinct_time_positions():
    t = np.array(["2000-07-01T00", "2000-07-01T00", "2000-07-01T00", "2000-07-01T06", "2000-07-01T06"], dtype="datetime64[ns]")
    lat = np.array([10.0, 10.0, 10.0, 10.0, 11.0])
    lon = np.array([5.0, 5.0, 6.0, 5.0, 5.0])
    c = D.count_stage(t, lat, lon, np.array([1, 2, 3, 1, 1]))
    assert c == dict(observations=5, trajectories=3, distinct_time_positions=4)   # one pair of time and position is held twice
    d = D.derived(c, dict(observations=4, trajectories=2, distinct_time_positions=4))
    assert d["pct_archived_repeats"] == pytest.approx(100.0 * 1 / 5) and d["pct_waves_of_trajectories"] == pytest.approx(100.0 * 2 / 3)
    with pytest.raises(SystemExit, match="no observation"):
        D.derived(dict(observations=0, trajectories=0, distinct_time_positions=0), c)


def fake_loader(n_dedup):
    t = pd.date_range("2000-07-05", periods=6, freq="6h").values

    def load(_glob, dedup=True):
        k = n_dedup if dedup else 6
        return Troughs(t[:k], np.full(k, 10.0), np.linspace(0.0, 5.0, 6)[:k], {"traj_id": np.arange(k) // 2})
    return load


def test_the_counts_are_written_only_when_they_equal_the_deposit(tmp_path, monkeypatch):
    pd.DataFrame([("pooled", "n_corridor_troughs_all_season", 4.0)], columns=["tier", "statistic", "diff"]).to_csv(tmp_path / "canonical_numbers.csv", index=False)
    monkeypatch.setattr(sys, "argv", ["dedup_counts.py", "--deposit", str(tmp_path)])
    monkeypatch.setattr(D, "load_aewc_troughs", fake_loader(4))
    D.main()
    out = pd.read_csv(tmp_path / "dedup_counts.csv").set_index("stage")
    assert (out.loc["as_archived", "observations"], out.loc["deduplicated", "observations"]) == (6, 4)
    assert (out.loc["as_archived", "trajectories"], out.loc["deduplicated", "trajectories"]) == (3, 2)
    assert list(out.pct_waves_of_trajectories) == pytest.approx([100.0 * 2 / 3] * 2)
    os.remove(tmp_path / "dedup_counts.csv")
    monkeypatch.setattr(D, "load_aewc_troughs", fake_loader(5))            # one observation more than the deposit states
    with pytest.raises(SystemExit, match="the deposit states 4"):
        D.main()
    assert not os.path.exists(tmp_path / "dedup_counts.csv")
    reg = pd.DataFrame([("pooled", "n_corridor_troughs_all_season", 4.0)] * 2, columns=["tier", "statistic", "diff"])
    reg.to_csv(tmp_path / "canonical_numbers.csv", index=False)
    with pytest.raises(SystemExit, match="2 times"):
        D.deposit_count(str(tmp_path))


# ------------------------------------------------------------------ the moist tail at other percentiles
def cases_world(n_waves=60, seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    for w in range(n_waves):
        base = rng.normal(60.0, 6.0)
        for _ in range(int(rng.integers(2, 7))):
            active = rng.random() < 0.5
            rows.append(dict(traj_id=w, label=E.ACTIVE if active else E.QUIET, rh_m72=round(base + (2.5 if active else 0.0) + rng.normal(0.0, 3.0), 4)))
    c = pd.DataFrame(rows)
    c.loc[3, "rh_m72"] = np.nan
    return c


def test_the_exceedance_is_the_share_of_each_class_above_the_pooled_percentile():
    c = cases_world()
    r = E.exceedance(c.rh_m72.values, c.label.values, c.traj_id.values, 80.0)
    ok = np.isfinite(c.rh_m72.values)
    thr = np.percentile(c.rh_m72.values[ok], 80.0)
    q, a = (c.label.values == E.QUIET) & ok, (c.label.values == E.ACTIVE) & ok
    pq, pa = (c.rh_m72.values[q] > thr).mean(), (c.rh_m72.values[a] > thr).mean()
    assert r["threshold"] == pytest.approx(thr) and (r["n_quiet"], r["n_active"]) == (int(q.sum()), int(a.sum()))
    assert (r["mean_quiet"], r["mean_active"], r["diff"], r["ratio"]) == pytest.approx((100 * pq, 100 * pa, 100 * (pa - pq), pa / pq))
    assert pa > pq and r["ratio"] > 1.0 and 0.0 < pq < 1.0
    # a value equal to the threshold is not above it
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    lab = np.array([E.QUIET] * 5 + [E.ACTIVE] * 5)
    r = E.exceedance(x, lab, np.arange(10), 50.0)
    assert r["threshold"] == 3.0 and r["mean_quiet"] == r["mean_active"] == 40.0
    with pytest.raises(SystemExit, match="holds no observation"):
        E.exceedance(x, np.array([E.QUIET] * 10), np.arange(10), 50.0)


def test_the_interval_resamples_waves_with_a_stream_of_its_own_for_each_percentile():
    c = cases_world()
    ok = np.isfinite(c.rh_m72.values)
    q, a = (c.label.values == E.QUIET) & ok, (c.label.values == E.ACTIVE) & ok
    got = {}
    for p in (60.0, 75.0):
        r = E.exceedance(c.rh_m72.values, c.label.values, c.traj_id.values, p, n_boot=300, seed=0)
        exc = (c.rh_m72.values > r["threshold"]).astype(float)
        rng = np.random.default_rng([0, 7, int(round(10 * p))])
        d, lo, hi, nq, na = cluster_bootstrap_diff(c.traj_id.values[q], exc[q], c.traj_id.values[a], exc[a], rng, n_boot=300)
        assert (r["diff"], r["ci_lo"], r["ci_hi"]) == pytest.approx((100 * d, 100 * lo, 100 * hi)) and r["ci_lo"] < r["diff"] < r["ci_hi"]
        assert (r["clusters_quiet"], r["clusters_active"], r["n_boot"]) == (nq, na, 300)
        assert r["significant"] == (not (lo <= 0.0 <= hi))
        got[p] = r
    again = E.exceedance(c.rh_m72.values, c.label.values, c.traj_id.values, 60.0, n_boot=300, seed=0)
    assert (again["ci_lo"], again["ci_hi"]) == (got[60.0]["ci_lo"], got[60.0]["ci_hi"])
    other = E.exceedance(c.rh_m72.values, c.label.values, c.traj_id.values, 60.0, n_boot=300, seed=1)
    assert (other["ci_lo"], other["ci_hi"]) != (got[60.0]["ci_lo"], got[60.0]["ci_hi"])


def deposit_for(c, tmp_path, change=None):
    r = E.exceedance(c.rh_m72.values, c.label.values, c.traj_id.values, E.CANONICAL)
    c = c.assign(exceed_p70=r["flags"])
    vals = dict(mean_quiet=r["mean_quiet"], mean_active=r["mean_active"], diff=r["diff"], threshold=r["threshold"], ratio=r["ratio"],
                n_quiet=r["n_quiet"], n_active=r["n_active"])
    if change:
        c, vals = change(c, vals)
    rows = [("pooled", 700.0, "exceedance_p70", -72.0, vals["n_quiet"], vals["n_active"], vals["mean_quiet"], vals["mean_active"], vals["diff"]),
            ("pooled", 700.0, "exceedance_threshold", -72.0, np.nan, np.nan, np.nan, np.nan, vals["threshold"]),
            ("pooled", 700.0, "exceedance_ratio", -72.0, np.nan, np.nan, np.nan, np.nan, vals["ratio"]),
            ("dev", 700.0, "exceedance_p70", -72.0, 1, 1, 0.0, 0.0, 0.0)]
    pd.DataFrame(rows, columns=["tier", "level", "statistic", "time_rel_h", "n_quiet", "n_active", "mean_quiet", "mean_active", "diff"]).to_csv(
        tmp_path / "canonical_numbers.csv", index=False)
    c.to_csv(tmp_path / "cases_pooled_700.csv", index=False)
    return c


def test_the_construction_must_reproduce_the_deposit_at_the_canonical_percentile(tmp_path, monkeypatch):
    c = deposit_for(cases_world(), tmp_path)
    assert set(E.bind_to_deposit(c, str(tmp_path))) == {"mean_quiet", "mean_active", "diff", "threshold", "ratio"}
    monkeypatch.setattr(sys, "argv", ["sensitivity_exceedance.py", "--deposit", str(tmp_path), "--n-boot", "200"])
    E.main()
    out = pd.read_csv(tmp_path / "exceedance_sensitivity.csv")
    assert list(out.percentile) == list(E.PERCENTILES) and E.CANONICAL not in set(out.percentile) and "flags" not in out.columns
    assert (out.threshold.diff().dropna() > 0).all() and (out.mean_quiet.diff().dropna() < 0).all()

    def flag(c, v):
        c = c.copy()
        i = int(np.flatnonzero(np.isfinite(c.rh_m72.values))[5])
        c.loc[i, "exceed_p70"] = 1.0 - c.exceed_p70.values[i]
        return c, v
    plants = [(flag, "differ from the deposited flags"),
              (lambda c, v: (c, dict(v, mean_active=v["mean_active"] + 0.01)), "mean_active at the canonical percentile"),
              (lambda c, v: (c, dict(v, threshold=v["threshold"] + 0.01)), "threshold at the canonical percentile"),
              (lambda c, v: (c, dict(v, ratio=np.nan)), "ratio at the canonical percentile"),
              (lambda c, v: (c, dict(v, n_quiet=v["n_quiet"] + 1)), "class sizes differ")]
    for change, message in plants:
        c = deposit_for(cases_world(), tmp_path, change)
        with pytest.raises(SystemExit, match=message):
            E.bind_to_deposit(c, str(tmp_path))


# ------------------------------------------------------------------ the centroid difference with seasons resampled
def season_profiles(n=8, seed=11):
    rng = np.random.default_rng(seed)
    a = rng.integers(20, 60, (n, F.REL_C.size)).astype(float)
    d = np.round(0.3 * a * rng.uniform(0.5, 1.5, a.shape))
    d[:, F.REL_C > 0] += 6.0                                               # the second curve leans east
    return a, d


def test_the_centroid_difference_resamples_whole_seasons_for_both_curves_together():
    a, d = season_profiles()
    r = F.centroid_difference_interval(a, d, n_boot=400, seed=0)
    ca = float((F.REL_C * a.sum(0)).sum() / a.sum())
    cd = float((F.REL_C * d.sum(0)).sum() / d.sum())
    assert (r["centroid_all"], r["centroid_deep"], r["diff"]) == pytest.approx((ca, cd, cd - ca)) and r["diff"] > 0
    rng = np.random.default_rng([0, 3, 5])
    w = rng.multinomial(a.shape[0], np.full(a.shape[0], 1.0 / a.shape[0]), size=400)
    reps = []
    for row in w:                                                           # one replicate at a time, written out plainly
        pick = np.repeat(np.arange(a.shape[0]), row)
        sa, sd = a[pick].sum(0), d[pick].sum(0)
        reps.append((F.REL_C * sd).sum() / sd.sum() - (F.REL_C * sa).sum() / sa.sum())
    lo, hi = np.percentile(reps, [2.5, 97.5])
    assert (r["ci_lo"], r["ci_hi"]) == pytest.approx((lo, hi)) and r["ci_lo"] < r["diff"] < r["ci_hi"]
    assert (r["n_seasons"], r["n_boot"], r["significant"]) == (8, 400, not (lo <= 0.0 <= hi))
    with pytest.raises(SystemExit, match="do not share one shape"):
        F.centroid_difference_interval(a, d[:-1])
    with pytest.raises(SystemExit, match="do not share one shape"):
        F.centroid_difference_interval(a[:1], d[:1])


def test_two_curves_of_one_shape_have_no_centroid_difference_in_any_replicate():
    a, _ = season_profiles()
    r = F.centroid_difference_interval(a, 0.25 * a, n_boot=200)
    assert r["diff"] == pytest.approx(0.0, abs=1e-12) and r["ci_lo"] == pytest.approx(0.0, abs=1e-12) and r["ci_hi"] == pytest.approx(0.0, abs=1e-12)


# ------------------------------------------------------------------ the alignment composite on a grid that does not wrap
def bump_world(lon, shift, tr_lon, width=2.0):
    """One field per trough time whose curvature vorticity peaks ``shift`` degrees east of the trough."""
    lat = np.arange(0.0, 20.5, 0.5) if np.median(np.diff(lon)) < 1.0 else np.arange(0.0, 21.0, 1.5)
    times = pd.date_range("2000-07-01", periods=len(tr_lon), freq="6h")
    crv = np.zeros((len(tr_lon), lat.size, lon.size))
    near = (np.abs(lat - 10.0) <= 1.0)[:, None]                             # the bump lies in the rows about the trough's latitude alone
    for i, lo in enumerate(tr_lon):
        crv[i] = np.where(near, np.exp(-0.5 * ((lon - (lo + np.ravel(shift)[i % np.size(shift)])) / width) ** 2)[None, :], 0.0)
    return times, lat, crv


def test_the_composite_finds_a_displacement_planted_in_the_field():
    lon = np.arange(-45.0, 75.5, 0.5)
    tr_lon = np.array([-20.0, -5.0, 0.0, 10.0, 25.0, 38.0])
    times, lat, crv = bump_world(lon, 1.0, tr_lon)
    c = A.Composite(0.5, wrap=False)
    c.add(times, lat, lon, crv, times.values, np.full(tr_lon.size, 10.0), tr_lon)
    s, prof = c.summary()
    assert (s["peak_deg"], s["median_argmax_deg"], s["n_matched"], s["n_window_outside_grid"]) == (1.0, 1.0, 6, 0)
    assert s["centroid_deg"] == pytest.approx(1.0, abs=1e-6) and s["grid_step_deg"] == 0.5 and s["n_argmax"] == 6
    assert list(prof.offset_deg[[0, len(prof) - 1]]) == [-12.0, 12.0] and len(prof) == 49 and set(prof.n) == {6.0}
    # an asymmetric lobe: the centroid of the half-peak lobe is not the peak
    crv2 = crv + 0.7 * bump_world(lon, 3.0, tr_lon, width=1.0)[2]
    c2 = A.Composite(0.5, wrap=False)
    c2.add(times, lat, lon, crv2, times.values, np.full(tr_lon.size, 10.0), tr_lon)
    s2, p2 = c2.summary()
    lobe = p2.mean_curvature_vorticity > 0.5 * p2.mean_curvature_vorticity.max()
    want = float((p2.offset_deg[lobe] * p2.mean_curvature_vorticity[lobe]).sum() / p2.mean_curvature_vorticity[lobe].sum())
    assert s2["centroid_deg"] == pytest.approx(want) and abs(s2["centroid_deg"] - s2["peak_deg"]) > 0.1
    positive = float((p2.offset_deg * p2.mean_curvature_vorticity).sum() / p2.mean_curvature_vorticity.sum())
    assert abs(s2["centroid_deg"] - positive) > 0.05                        # not the centroid of the whole window


def test_the_median_of_the_troughs_own_maxima_is_a_median_and_reads_the_troughs_latitude():
    lon = np.arange(-45.0, 75.5, 0.5)
    tr_lon = np.array([-10.0, 0.0, 10.0])
    times, lat, crv = bump_world(lon, np.array([0.0, 0.0, 3.0]), tr_lon)    # two troughs aligned, one displaced by three degrees
    c = A.Composite(0.5, wrap=False)
    c.add(times, lat, lon, crv, times.values, np.full(3, 10.0), tr_lon)
    s, _ = c.summary()
    assert (s["median_argmax_deg"], s["n_argmax"]) == (0.0, 3) and s["peak_deg"] in (0.0, 0.5)   # a mean of the three would be 1.0
    far = A.Composite(0.5, wrap=False)                                      # read at a latitude the bump does not reach, the composite is flat
    far.add(times, lat, lon, crv, times.values, np.full(3, 18.0), tr_lon)
    assert float(np.abs(far.total).max()) == 0.0 and far.matched == 3


def test_a_trough_whose_window_leaves_a_grid_that_does_not_wrap_takes_no_part():
    lon = np.arange(-45.0, 75.5, 0.5)
    tr_lon = np.array([-44.0, 0.0, 70.0])                                   # the first and the last lie within 12 degrees of an edge
    times, lat, crv = bump_world(lon, 0.0, tr_lon)
    c = A.Composite(0.5, wrap=False)
    c.add(times, lat, lon, crv, times.values, np.full(3, 10.0), tr_lon)
    s, prof = c.summary()
    assert (s["n_matched"], s["n_window_outside_grid"], s["peak_deg"]) == (1, 2, 0.0) and set(prof.n) == {1.0}
    # a time the fields do not hold is passed over, and a grid of another step is refused
    c.add(times, lat, lon, crv, np.array(["1999-01-01T00"], dtype="datetime64[ns]"), np.array([10.0]), np.array([0.0]))
    assert c.matched == 1
    with pytest.raises(SystemExit, match="grid step differs"):
        c.add(times, lat, np.arange(-45.0, 76.0, 1.5), crv[:, :, :81], times.values, np.full(3, 10.0), tr_lon)
    with pytest.raises(SystemExit, match="does not divide"):
        A.Composite(0.7, wrap=False)


def test_on_a_grid_that_wraps_the_window_continues_across_the_seam():
    lon = np.arange(-180.0, 180.0, 1.5)
    tr_lon = np.array([-178.5, 0.0, 177.0])
    times, lat, _ = bump_world(lon, 0.0, tr_lon)
    crv = np.zeros((3, lat.size, lon.size))
    for i, lo in enumerate(tr_lon):                                         # a bump three steps east of the trough, measured around the globe
        dist = (lon - (lo + 4.5) + 180.0) % 360.0 - 180.0
        crv[i] = np.exp(-0.5 * (dist / 2.0) ** 2)[None, :]                  # in every row, so the row chosen does not matter here
    c = A.Composite(1.5, wrap=True)
    c.add(times, lat, lon, crv, times.values, np.full(3, 10.5), tr_lon)
    s, prof = c.summary()
    assert (s["n_matched"], s["n_window_outside_grid"], s["peak_deg"], s["median_argmax_deg"]) == (3, 0, 4.5, 4.5) and len(prof) == 17
    assert c.argmax == [4.5, 4.5, 4.5]                                      # the trough at 177 E finds its bump past the seam, at 178.5 W


def test_the_function_must_reproduce_the_earlier_check_before_anything_is_written(tmp_path):
    ok = dict(peak_deg=0.0, median_argmax_deg=0.0, centroid_deg=0.5512)
    A.require_earlier_check(ok, 0.55)
    for plant in (dict(ok, peak_deg=1.5), dict(ok, median_argmax_deg=-1.5), dict(ok, centroid_deg=0.561), dict(ok, centroid_deg=np.nan)):
        with pytest.raises(SystemExit, match="the paper states"):
            A.require_earlier_check(plant, 0.55)
    pd.DataFrame([("validation", "val_align_centroid_deg", 0.55), ("validation", "other", 1.0)], columns=["tier", "statistic", "diff"]).to_csv(tmp_path / "c.csv", index=False)
    assert A.constants(str(tmp_path / "c.csv")) == 0.55
    pd.DataFrame([("validation", "other", 1.0)], columns=["tier", "statistic", "diff"]).to_csv(tmp_path / "c.csv", index=False)
    with pytest.raises(SystemExit, match="0 times"):
        A.constants(str(tmp_path / "c.csv"))


# ------------------------------------------------------------------ the registry
def test_the_registry_takes_the_four_analyses(tmp_path, monkeypatch):
    import build_registry as R
    monkeypatch.chdir(tmp_path)
    assert R.rows_from_dedup_counts() is None and R.rows_from_exceedance_sensitivity() is None and R.rows_from_alignment_region() is None
    os.makedirs("deposit")
    pd.DataFrame([dict(stage="as_archived", observations=20, trajectories=8, distinct_time_positions=12, pct_archived_repeats=40.0, pct_waves_of_trajectories=50.0),
                  dict(stage="deduplicated", observations=10, trajectories=4, distinct_time_positions=10, pct_archived_repeats=40.0, pct_waves_of_trajectories=50.0)]
                 ).to_csv("deposit/dedup_counts.csv", index=False)
    d = R.rows_from_dedup_counts().set_index("statistic")["diff"]
    assert (d["n_archived_observations"], d["n_archived_trajectories"], d["n_archived_distinct_time_positions"]) == (20, 8, 12)
    assert (d["n_dedup_observations"], d["n_dedup_trajectories"], d["pct_archived_repeats"], d["pct_waves_of_trajectories"]) == (10, 4, 40.0, 50.0)
    pd.DataFrame([dict(percentile=50.0, threshold=58.0, mean_quiet=45.0, mean_active=55.0, diff=10.0, ratio=1.22, n_quiet=30, n_active=31, ci_lo=6.0, ci_hi=14.0,
                       clusters_quiet=9, clusters_active=8, n_boot=200, significant=True),
                  dict(percentile=90.0, threshold=71.0, mean_quiet=8.0, mean_active=12.0, diff=4.0, ratio=1.5, n_quiet=30, n_active=31, ci_lo=-1.0, ci_hi=9.0,
                       clusters_quiet=9, clusters_active=8, n_boot=200, significant=False)]).to_csv("deposit/exceedance_sensitivity.csv", index=False)
    e = R.rows_from_exceedance_sensitivity().set_index("statistic")
    assert list(e.index) == ["exceedsens_p50", "exceedsens_p90"]
    r = e.loc["exceedsens_p90"]
    assert (r["diff"], r.ci_lo, r.ci_hi, r.mean_active, r.mean_quiet, r.ratio, r.threshold, r.level, r.time_rel_h, bool(r.significant)) == (4.0, -1.0, 9.0, 12.0, 8.0, 1.5, 71.0, 700.0, -72.0, False)
    pd.DataFrame([dict(grid="global band", seasons=25, grid_step_deg=1.5, n_matched=100, n_window_outside_grid=0, peak_deg=0.0, centroid_deg=0.55, median_argmax_deg=0.0, n_argmax=99),
                  dict(grid="regional", seasons=25, grid_step_deg=0.5, n_matched=98, n_window_outside_grid=2, peak_deg=0.5, centroid_deg=0.31, median_argmax_deg=1.0, n_argmax=97)]
                 ).to_csv("deposit/alignment_region.csv", index=False)
    a = R.rows_from_alignment_region().set_index("statistic")["diff"]
    assert (a["align_region_peak_deg"], a["align_region_centroid_deg"], a["align_region_median_argmax_deg"], a["align_region_grid_step_deg"],
            a["align_region_n_matched"], a["align_region_n_window_outside_grid"]) == (0.5, 0.31, 1.0, 0.5, 98, 2)
    assert (a["align_region_n_argmax"], a["align_band_n_argmax"]) == (97, 99)          # each grid's own count, not the other's
    base = dict(figure="F3", n_repeats_dropped=0, n_matches=50.0, bandmean_profile_total=10.0, peak_count=3.0, peak_rel_lon=0.0, peak_frac_pct=4.0,
                share_core_pct=16.0, share_west_pct=48.0)
    pd.DataFrame([dict(base, subset="all", n_events=10, centroid_rel_lon=-0.1), dict(base, subset="deep_at_detection", n_events=4, centroid_rel_lon=1.7)]
                 ).to_csv("deposit/fig3_descriptive_stats.csv", index=False)
    pd.DataFrame([dict(centroid_all=-0.1, centroid_deep=1.7, diff=1.8, ci_lo=0.9, ci_hi=2.6, n_seasons=24, n_boot=20000, significant=True)]
                 ).to_csv("deposit/fig3_centroid_interval.csv", index=False)
    f = R.rows_from_fig3_descriptive().set_index("statistic")
    r = f.loc["fig3_centroid_difference"]
    assert (r["diff"], r.ci_lo, r.ci_hi, r.n_seasons, r.n_boot, bool(r.significant)) == (1.8, 0.9, 2.6, 24, 20000, True)
    assert f.loc["fig3_centroid_rel_lon_deep_at_detection", "diff"] == 1.7
