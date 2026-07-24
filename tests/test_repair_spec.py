"""Tests for the REPAIR_SPEC.md R2/R3 machinery (terrain masks, window eligibility)."""

import numpy as np
import pandas as pd

from aew.environment import complete_window_mask
from aew.terrain import apply_terrain_mask, masked_box_mean, valid_fraction
from aew.trajectory import Gridded, back_trajectories


def test_complete_window_mask_bounds():
    times = pd.to_datetime([
        "2001-07-01 00:00", "2001-07-02 18:00",   # antecedent window incomplete
        "2001-07-03 00:00",                        # earliest eligible
        "2001-08-15 12:00",                        # interior
        "2001-09-29 18:00", "2001-09-30 00:00",   # last two eligible times
        "2001-09-30 06:00",                        # forward window incomplete
        "2001-06-30 18:00", "2001-10-01 00:00",   # outside the season
    ])
    got = complete_window_mask(times)
    assert got.tolist() == [False, False, True, True, True, True, False, False, False]


def test_terrain_mask_threshold_and_alignment():
    sp = np.array([[[101000.0, 88000.0, 87000.0]]])   # Pa; threshold at 850 is 87500
    vals = np.ones_like(sp)
    out = apply_terrain_mask(vals, sp, level_hpa=850.0)
    assert np.isfinite(out[0, 0, 0]) and np.isfinite(out[0, 0, 1])
    assert np.isnan(out[0, 0, 2])
    # 700 hPa threshold 72500: every point valid (the corridor expectation)
    assert np.isfinite(apply_terrain_mask(vals, sp, level_hpa=700.0)).all()
    try:
        apply_terrain_mask(np.ones((1, 2, 2)), sp, 850.0)
        raised = False
    except ValueError:
        raised = True
    assert raised
    assert np.allclose(valid_fraction(sp, 850.0), [2.0 / 3.0])


def test_masked_box_mean_half_rule():
    box = np.array([[1.0, 1.0, np.nan, np.nan]])
    assert np.isclose(masked_box_mean(box)[0], 1.0)          # exactly half -> kept
    box2 = np.array([[1.0, np.nan, np.nan, np.nan]])
    assert np.isnan(masked_box_mean(box2)[0])                # under half -> missing


def test_gridded_stencil_propagates_mask():
    # one masked grid corner makes any sample whose bilinear stencil touches it NaN
    t = pd.to_datetime(["2001-07-10 00:00", "2001-07-10 06:00"]).values
    lat = np.array([10.0, 11.0, 12.0])
    lon = np.array([0.0, 1.0, 2.0])
    v = np.ones((2, 3, 3))
    v[:, 0, 0] = np.nan                                       # masked corner
    g = Gridded(t, lat, lon, v)
    th = g.t[0] + 3.0
    assert np.isnan(g.sample(np.array([th]), np.array([10.4]), np.array([0.4])))[0]
    assert np.isfinite(g.sample(np.array([th]), np.array([11.5]), np.array([1.5])))[0]


def test_trajectory_nan_holds_once_masked():
    # uniform westward wind, but a masked band at lon > 1.5: a parcel integrated
    # backward (moving east) goes NaN at the band and stays NaN at earlier times
    t = pd.date_range("2001-07-10", periods=9, freq="6h").values
    lat = np.arange(5.0, 16.0, 1.0)
    lon = np.arange(-5.0, 6.0, 1.0)
    shape = (t.size, lat.size, lon.size)
    u = np.full(shape, -5.0)
    u[:, :, lon > 1.5] = np.nan                               # terrain-masked winds
    v = np.zeros(shape)
    gu = Gridded(t, lat, lon, u)
    gv = Gridded(t, lat, lon, v)
    seed_t = np.array([t[-1]])
    elapsed, plat, plon = back_trajectories(gu, gv, seed_t, np.array([10.0]),
                                            np.array([0.0]), hours=36.0, dt_hours=1.0)
    assert np.isfinite(plon[0, 0])                            # seed is fine
    bad = np.where(np.isnan(plon[:, 0]))[0]
    assert bad.size > 0                                       # the band was reached
    assert np.isnan(plon[bad[0]:, 0]).all()                   # and NaN holds after it


def test_lead_field_box_half_rule():
    from aew.environment import lead_field_box
    t = pd.to_datetime(["2001-07-10 00:00"]).values
    ft = pd.to_datetime(["2001-07-09 00:00"]).values
    lat = np.arange(5.0, 15.1, 5.0)      # 3 lats
    lon = np.arange(-5.0, 5.1, 5.0)      # 3 lons -> 9 box cells
    good = np.ones((1, 3, 3))
    out = lead_field_box(t, np.array([0.0]), ft, lat, lon, good, lead_h=24.0)
    assert np.isclose(out[0], 1.0)
    mostly_masked = good.copy()
    mostly_masked[0, :, :2] = np.nan     # 6 of 9 masked -> under half valid
    out2 = lead_field_box(t, np.array([0.0]), ft, lat, lon, mostly_masked, lead_h=24.0)
    assert np.isnan(out2[0])


def test_mask_level_inplace_mutates():
    from aew.terrain import mask_level_inplace
    sp = np.array([[[101000.0, 87000.0]]])
    f = np.ones((1, 1, 2), dtype=np.float32)
    out = mask_level_inplace(f, sp, 850.0)
    assert out is f and np.isfinite(f[0, 0, 0]) and np.isnan(f[0, 0, 1])


def test_anchor_permutation_preserves_shape_and_distribution():
    from aew.composites import anchor_permutation
    rng = np.random.default_rng(1)
    # two waves in July, two in August, distinct anchors, 3 obs each
    tid = np.repeat([10, 11, 20, 21], 3)
    times = pd.to_datetime(
        ["2001-07-05 00:00", "2001-07-05 06:00", "2001-07-05 12:00",
         "2001-07-20 00:00", "2001-07-20 06:00", "2001-07-20 12:00",
         "2001-08-05 00:00", "2001-08-05 06:00", "2001-08-05 12:00",
         "2001-08-20 00:00", "2001-08-20 06:00", "2001-08-20 12:00"]).values
    lons = np.array([0.0, 1.0, 2.0, 10.0, 11.0, 12.0,
                     20.0, 21.0, 22.0, 30.0, 31.0, 32.0])
    perm, apply_draw = anchor_permutation(tid, times, lons, n_draws=50, rng=rng)
    anchors = {10: 1.0, 11: 11.0, 20: 21.0, 21: 31.0}
    swapped = 0
    for b in range(50):
        lb = apply_draw(b)
        # internal shape preserved: each wave's obs stay anchor + (-1, 0, +1)
        for w, sl in ((0, slice(0, 3)), (1, slice(3, 6)),
                      (2, slice(6, 9)), (3, slice(9, 12))):
            assert np.allclose(np.diff(lb[sl]), 1.0)
        # anchors are a permutation of the observed anchors, within stratum
        got = sorted([lb[1], lb[4]])            # July wave centers
        assert np.allclose(got, [1.0, 11.0])
        got = sorted([lb[7], lb[10]])           # August wave centers
        assert np.allclose(got, [21.0, 31.0])
        if not np.isclose(lb[1], 1.0):
            swapped += 1
        # never a cross-month anchor
        assert perm[b, 0] in (0, 1) and perm[b, 2] in (2, 3)
    assert 0 < swapped < 50                     # permutation actually permutes


def test_randomization_test_selection_aware():
    from aew.composites import randomization_test
    rng = np.random.default_rng(2)
    rel_c = np.arange(-30.0, 30.1, 2.0)
    null = rng.normal(100.0, 5.0, size=(999, rel_c.size))
    obs = np.full(rel_c.size, 100.0)
    obs[rel_c == -2.0] += 40.0                  # a real peak 2 deg west
    r = randomization_test(obs, null, rel_c)
    assert r["p_value"] == 1.0 / 1000.0         # never matched in 999 draws
    assert r["peak_rel_lon"] == -2.0
    assert 35.0 < r["peak_excess"] < 45.0
    # a flat observation is not significant even at the null's pointwise edge
    flat = np.full(rel_c.size, 100.0)
    flat[5] += 10.0                             # ~2 sigma at one bin
    r2 = randomization_test(flat, null, rel_c, search=(-30.0, 30.0))
    assert r2["p_value"] > 0.05                 # max statistic prices in selection
    assert r2["simult_half_width"] > 2.0 * 5.0  # simultaneous wider than pointwise


def _synthetic_cases(n_waves=60, obs_per_wave=6, seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    for w in range(n_waves):
        base_h = 55.0 + (w % 3) * 2.0            # wave-level H differences
        lon = -20.0 + (w % 2) * 10.0             # two cells, both above the 10-wave min
        month = 7 + (w % 2)
        kind = w % 3                              # 0 quiet-only, 1 active-only, 2 mixed
        for o in range(obs_per_wave):
            if kind == 0:
                lab = "MCS-quiet"
            elif kind == 1:
                lab = "MCS-active"
            else:
                lab = "MCS-active" if o % 2 else "MCS-quiet"
            h = base_h + (1.0 if lab == "MCS-active" else 0.0) + rng.normal(0, 0.1)
            rows.append(dict(traj_id=w, time=f"2001-{month:02d}-10 {6*(o%4):02d}:00",
                             lon=lon + o * 0.5, label=lab, rh_m72=h))
    return pd.DataFrame(rows)


def test_paired_contrast_recovers_within_wave_offset():
    from aew.wave_level import paired_contrast
    cases = _synthetic_cases()
    r = paired_contrast(cases, rng=np.random.default_rng(0), n_boot=500)
    assert r["n_waves"] == 20                      # the mixed third
    assert 0.8 < r["diff"] < 1.2                   # the built-in +1 offset
    assert r["ci_lo"] > 0.5


def _waves_full_from_cases(cases):
    from aew.wave_level import wave_table
    d = cases.rename(columns={"rh_m72": "H"}).copy()
    d["response"] = 0.0
    return wave_table(d[["traj_id", "time", "lon", "response", "H"]])


def test_between_contrast_standardizes_cells():
    from aew.wave_level import between_contrast
    cases = _synthetic_cases()
    waves_full = _waves_full_from_cases(cases)
    r = between_contrast(cases, waves_full, rng=np.random.default_rng(0), n_boot=500)
    assert r["n_active_only"] == 20 and r["n_quiet_only"] == 20
    assert r["diff"] > 0.0
    # standardization must be REAL: add a cell-level offset that separates the two
    # groups only through cell composition; a working standardization removes it
    biased = cases.copy()
    biased.loc[biased.lon > -15.0, "rh_m72"] += 50.0
    r2 = between_contrast(biased, _waves_full_from_cases(biased),
                          rng=np.random.default_rng(0), n_boot=500)
    assert abs(r2["diff"] - r["diff"]) < 5.0     # not dragged by the +50 cell shift


def test_paired_presence_decided_before_missing_H():
    from aew.wave_level import paired_contrast
    cases = _synthetic_cases()
    # blank every quiet H of wave 2 (a mixed wave): it must stay MIXED and become
    # attrition, never be reclassified active-only
    m = (cases.traj_id == 2) & (cases.label == "MCS-quiet")
    cases.loc[m, "rh_m72"] = np.nan
    r = paired_contrast(cases, rng=np.random.default_rng(0), n_boot=300)
    assert r["n_waves"] == 19 and r["n_mixed_unusable"] == 1


def test_wave_unit_contrast_mechanics():
    from aew.wave_level import wave_table, wave_unit_contrast
    rng = np.random.default_rng(4)
    rows = []
    for w in range(40):
        resp = float(w % 4) * 5.0                  # response terciles
        h = 50.0 + (w % 4) * 1.0                   # H rises with response
        for o in range(4):
            rows.append(dict(traj_id=w, time=f"2001-08-{10+o:02d} 00:00",
                             lon=0.0 + o, response=resp + rng.normal(0, 0.1),
                             H=h + rng.normal(0, 0.05)))
    waves = wave_table(pd.DataFrame(rows))
    assert len(waves) == 40 and (waves["n_obs"] == 4).all()
    r = wave_unit_contrast(waves, rng=np.random.default_rng(0), n_boot=300)
    assert r["n_waves"] == 40 and r["n_dropped_small_cell"] == 0
    assert r["n_dropped_missing_H"] == 0
    assert 2.0 < r["diff"] < 4.0                   # top (h=53) minus bottom (h=50)
    assert r["ci_lo"] > 1.0 and r["sens_season_lo"] > 0.0


def test_mask_inplace_rejects_nan_surface_pressure():
    from aew.terrain import mask_level_inplace
    sp = np.array([[[101000.0, np.nan]]])
    f = np.ones((1, 1, 2))
    mask_level_inplace(f, sp, 850.0)
    assert np.isfinite(f[0, 0, 0]) and np.isnan(f[0, 0, 1])


def test_terciles_use_exact_thirds():
    from aew.environment import stratified_terciles
    # 3001 values 0..3000: exact-1/3 quantile is 1000; the 33.3 percentile is 999.something,
    # so the boundary observation 1000 is IN the bottom class only under exact thirds
    resp = np.arange(3001, dtype=float)
    lon = np.zeros(3001)
    low, high = stratified_terciles(resp, lon, np.array([-5.0, 5.0]), min_bin=30)
    assert low[1000] and not low[1001]
    assert high[2000] and not high[1999]


def test_aggregate_parcels_min_valid():
    from aew.trajectory import aggregate_parcels
    vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, np.nan, np.nan, np.nan, np.nan,  # 5 valid
                     1.0, 2.0, 3.0, 4.0, np.nan, np.nan, np.nan, np.nan, np.nan])  # 4
    mean, nv = aggregate_parcels(vals, 2, 9)
    assert np.isclose(mean[0], 3.0) and nv[0] == 5
    assert np.isnan(mean[1]) and nv[1] == 4
