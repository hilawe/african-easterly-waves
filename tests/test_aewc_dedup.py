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
    # two distinct waves that cross at a single shared point stay separate (below min_shared)
    t = pd.to_datetime(["2004-08-04 00:00", "2004-08-04 06:00"]).values
    time = np.concatenate([t, t])
    lat = np.array([10., 11., 10., 20.])   # both have (t0, 10, -5)
    lon = np.array([-5., -6., -5., 0.])
    tid = np.array([0, 0, 1, 1])

    out = deduplicate(_troughs(time, lat, lon, tid), min_shared=3)

    assert len(out) == 3                                   # the one shared point collapses
    assert np.unique(out.variables["wave_id"]).size == 2   # not merged on a single point


def test_deduplicate_noop_when_already_unique():
    t = pd.to_datetime(["2004-08-04 00:00", "2004-08-04 06:00"]).values
    out = deduplicate(_troughs(np.concatenate([t, t]),
                               [10., 11., 20., 21.], [-5., -6., 0., 1.],
                               [0, 0, 1, 1]))
    assert len(out) == 4
    assert np.unique(out.variables["wave_id"]).size == 2
