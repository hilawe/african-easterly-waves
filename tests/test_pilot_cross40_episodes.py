"""The cross-40 E episode study's contracts, on the complete rule with synthetic seasons
and its null of unrelated seasons joined across the line: a traveling signal crosses both
lines westward beyond chance joining, a standing oscillation does not, western waves
beside an eastern standing pattern cross 20 E but not 40 E beyond chance joining (the case
that defeated a plain westward-minus-eastward count), and the periodic signal that
defeated the correlation test is resolved, because successive episodes are told apart by
six-hourly continuity rather than by a choice among peaks.

These pin the basic behavior only. The realistic validation, in which the instrument
missed genuine crossings that weaken at 40 E and reported false directions from single
crossings, is the `--validate` record described in CROSS40_EPISODE_STUDY_2026-10-10.md.
The instrument was not run on real data."""
import importlib.util
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def _load():
    spec = importlib.util.spec_from_file_location("x40e_under_test", os.path.join(ROOT, "scripts", "pilot_cross40_episodes.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _smooth_noise(rng, T, lons, scale):
    """Noise correlated over about 3 degrees of longitude, white in time."""
    raw = rng.standard_normal((T, lons.size))
    k = np.exp(-0.5 * (np.arange(-12, 13) * 0.5 / 3.0) ** 2); k /= k.sum()
    return scale * np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 1, raw) * 3.0


def _fields(E, kind, seed=0, seasons=24):
    P = E.P
    rng = np.random.default_rng(seed)
    w = P.lanczos_bandpass()
    lons = np.arange(E.LON_RANGE[0], E.LON_RANGE[1] + 0.25, 0.5)
    t = np.arange(488) * P.DT_DAYS
    out = []
    for _ in range(seasons):
        ph = rng.uniform(0, 2 * np.pi)
        trav = np.sin(2 * np.pi * (lons[None, :] + 7.0 * t[:, None]) / 28.0 + ph)
        stand = np.cos(2 * np.pi * (lons[None, :] - 40.0) / 28.0) * np.sin(2 * np.pi * t[:, None] / 4.0 + ph)
        if kind == "traveling":
            f = trav + _smooth_noise(rng, 488, lons, 0.3)
        elif kind == "periodic":
            f = trav + _smooth_noise(rng, 488, lons, 0.05)
        elif kind == "standing":
            f = stand + _smooth_noise(rng, 488, lons, 0.3)
        else:                                                         # separate regimes with a quiet gap at 36 to 44 E
            west = np.clip((36.0 - lons) / 6.0, 0, 1)[None, :]
            east = np.clip((lons - 44.0) / 6.0, 0, 1)[None, :]
            f = west * trav + east * stand + _smooth_noise(rng, 488, lons, 0.15)
        out.append(P.filter_supported(f, w))
    return out, lons


W = "westward crossing episodes beyond chance joining"


def test_a_traveling_signal_crosses_both_lines_westward():
    E = _load()
    r = E.analyze(*_fields(E, "traveling"), n_perm=100)
    assert r["primary"]["category"] == W and r["control"]["category"] == W
    assert r["primary"]["east_per_season"] == 0.0 and r["primary"]["west_per_season"] > 10
    assert r["primary"]["west_per_season"] > r["primary"]["null_west_per_season"]["q95"]


def test_a_standing_oscillation_gives_no_resolved_direction():
    E = _load()
    r = E.analyze(*_fields(E, "standing"), n_perm=100)
    for key in ("primary", "control"):
        assert r[key]["category"] in ("no crossing episodes identified", "crossing episodes not distinguished from chance joining"), (key, r[key])


def test_western_waves_beside_an_eastern_standing_pattern_cross_20E_but_not_40E():
    E = _load()
    r = E.analyze(*_fields(E, "separate"), n_perm=100)
    assert r["control"]["category"] == W
    assert r["primary"]["category"] != W, r["primary"]


def test_the_periodic_signal_that_defeated_the_correlation_test_is_resolved_here():
    E = _load()
    r = E.analyze(*_fields(E, "periodic"), n_perm=100)
    assert r["primary"]["category"] == W and r["primary"]["east_per_season"] == 0.0


def test_linking_never_passes_between_successive_troughs_and_crossings_need_both_margins():
    E = _load()
    steps = [[50.0, 22.0], [48.5, 20.5], [], [45.5, 17.5], [44.0]]   # one missing step, then within twice the link distance
    eps = E.link_episodes(steps)
    assert sorted(len(e) for e in eps) == [3, 4] and all(abs(e[-1][1] - e[0][1]) < 7.0 for e in eps)
    assert E.link_episodes([[50.0], [43.0]]) == [[(0, 50.0)], [(1, 43.0)]]          # a 7-degree step is two episodes
    assert E.crossings([(0, 43.0), (1, 41.0), (2, 39.5), (3, 37.9)], 40.0) == (1, 0, [3])
    assert E.crossings([(0, 41.9), (1, 38.1)], 40.0) == (0, 0, [])                   # inside both margins, no crossing
    assert E.crossings([(0, 37.0), (1, 39.0), (2, 42.5)], 40.0) == (0, 1, [2])


def test_the_joined_null_keeps_variance_and_never_pairs_a_season_with_itself():
    E = _load()
    lons = np.arange(30.0, 50.25, 0.5)
    rng = np.random.default_rng(5)
    a, b = rng.standard_normal((20000, lons.size)), rng.standard_normal((20000, lons.size))
    h = E.hybrid(a, b, lons, 40.0)
    assert np.allclose(h[:, lons <= 38.0], b[:, lons <= 38.0]) and np.allclose(h[:, lons >= 42.0], a[:, lons >= 42.0])
    assert np.all(np.abs(h.std(axis=0) - 1.0) < 0.03)                              # independent sources keep unit variance in the zone
    perms = E.derangements(24, 30, seed=2)
    assert not np.any(perms == np.arange(24)) and all(sorted(p) == list(range(24)) for p in perms)
