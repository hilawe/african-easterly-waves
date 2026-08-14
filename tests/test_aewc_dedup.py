"""Tests for AEWC trough deduplication (aew.data.aewc.deduplicate)."""

import numpy as np
import pandas as pd

from aew.data.aewc import Troughs, deduplicate


def _troughs(time, lat, lon, tid):
    return Troughs(time=np.asarray(time), lat=np.asarray(lat, float),
                   lon=np.asarray(lon, float),
                   variables={"traj_id": np.asarray(tid, np.int64)})


def test_deduplicate_removes_exact_duplicates_and_merges_fragments():
    # one physical wave tracked as two identical fragment trajectories (0 and 1), plus a
    # distinct wave (2) that shares no points
    t = pd.to_datetime(["2004-08-04 00:00", "2004-08-04 06:00",
                        "2004-08-04 12:00", "2004-08-04 18:00"]).values
    time = np.concatenate([t, t, t])
    lat = np.array([10., 11., 12., 13.] * 2 + [20., 21., 22., 23.])
    lon = np.array([-5., -6., -7., -8.] * 2 + [0., 1., 2., 3.])
    tid = np.array([0] * 4 + [1] * 4 + [2] * 4)

    out = deduplicate(_troughs(time, lat, lon, tid))

    assert len(out) == 8                                   # 12 raw -> 8 unique observations
    assert np.unique(out.variables["wave_id"]).size == 2   # fragments 0,1 merged; 2 separate
    assert np.unique(out.variables["traj_id"]).size == 2   # traj_id overwritten with wave_id
    assert "orig_traj_id" in out.variables                 # pre-merge id preserved


def test_deduplicate_keeps_distinct_waves_sharing_one_point():
    # two distinct waves that cross at a single shared point stay separate (below
    # min_shared), and BOTH keep their full tracks. The old rule silently deleted the
    # shared point from one of them; the longest-trajectory rule only ever drops whole
    # fragments of a merged component, never single points of unmerged waves.
    t = pd.to_datetime(["2004-08-04 00:00", "2004-08-04 06:00"]).values
    time = np.concatenate([t, t])
    lat = np.array([10., 11., 10., 20.])   # both have (t0, 10, -5)
    lon = np.array([-5., -6., -5., 0.])
    tid = np.array([0, 0, 1, 1])

    out = deduplicate(_troughs(time, lat, lon, tid), min_shared=3)

    assert len(out) == 4                                   # both tracks intact
    assert np.unique(out.variables["wave_id"]).size == 2   # not merged on a single point


def _one_position_per_wave_time(out):
    wt = list(zip(out.variables["traj_id"].tolist(),
                  pd.DatetimeIndex(out.time).asi8.tolist()))
    return len(wt) == len(set(wt))


def test_deduplicate_diverging_branches_keep_only_the_longest_fragment():
    # THE ROUND-9 DEFECT, as a fixture. Two fragments share three points then diverge,
    # so they merge into one wave; the old rule kept BOTH divergent endpoints and the
    # merged wave occupied two positions at the final timestamp. The repaired rule keeps
    # only the longest fragment, so the wave has one position at every time.
    t = pd.to_datetime(["2004-08-04 00:00", "2004-08-04 06:00", "2004-08-04 12:00",
                        "2004-08-04 18:00", "2004-08-05 00:00"]).values
    # fragment 0: five observations; fragment 1: the same first three, then it diverges
    time = np.concatenate([t, t[:4]])
    lat = np.array([10., 11., 12., 13., 14.] + [10., 11., 12., 18.])
    lon = np.array([-5., -6., -7., -8., -9.] + [-5., -6., -7., -2.])
    tid = np.array([0] * 5 + [1] * 4)

    out = deduplicate(_troughs(time, lat, lon, tid))

    assert np.unique(out.variables["wave_id"]).size == 1   # merged into one wave
    assert len(out) == 5                                   # only fragment 0 survives
    assert set(out.variables["orig_traj_id"].tolist()) == {0}
    assert _one_position_per_wave_time(out)
    # the divergent branch point (18, -2) is gone
    assert not ((out.lat == 18.0) & (out.lon == -2.0)).any()


def test_deduplicate_transitive_union_keeps_one_fragment():
    # A shares points with B, B with C, A and C share nothing: all three are one
    # component, and only the longest of the three survives.
    t = pd.to_datetime(["2004-08-04 00:00", "2004-08-04 06:00", "2004-08-04 12:00",
                        "2004-08-04 18:00", "2004-08-05 00:00",
                        "2004-08-05 06:00", "2004-08-05 12:00"]).values
    lat = np.arange(10., 17.)
    lon = -np.arange(5., 12.)
    a = slice(0, 4); b = slice(1, 6); c = slice(3, 7)      # A:4 obs, B:5 obs, C:4 obs
    time = np.concatenate([t[a], t[b], t[c]])
    lats = np.concatenate([lat[a], lat[b], lat[c]])
    lons = np.concatenate([lon[a], lon[b], lon[c]])
    tid = np.array([0] * 4 + [1] * 5 + [2] * 4)

    out = deduplicate(_troughs(time, lats, lons, tid))

    assert np.unique(out.variables["wave_id"]).size == 1
    assert set(out.variables["orig_traj_id"].tolist()) == {1}   # B is longest
    assert len(out) == 5
    assert _one_position_per_wave_time(out)


def test_deduplicate_tie_breaks_on_smallest_original_id():
    # equal-length fragments: the smallest original id wins, deterministically
    t = pd.to_datetime(["2004-08-04 00:00", "2004-08-04 06:00", "2004-08-04 12:00",
                        "2004-08-04 18:00"]).values
    time = np.concatenate([t, t])
    lat = np.array([10., 11., 12., 13.] + [10., 11., 12., 19.])   # share 3, diverge at end
    lon = np.array([-5., -6., -7., -8.] + [-5., -6., -7., -1.])
    tid = np.array([7] * 4 + [3] * 4)

    out = deduplicate(_troughs(time, lat, lon, tid))

    assert set(out.variables["orig_traj_id"].tolist()) == {3}
    assert _one_position_per_wave_time(out)


def test_deduplicate_noop_when_already_unique():
    t = pd.to_datetime(["2004-08-04 00:00", "2004-08-04 06:00"]).values
    out = deduplicate(_troughs(np.concatenate([t, t]),
                               [10., 11., 20., 21.], [-5., -6., 0., 1.],
                               [0, 0, 1, 1]))
    assert len(out) == 4
    assert np.unique(out.variables["wave_id"]).size == 2
