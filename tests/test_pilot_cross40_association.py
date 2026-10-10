"""The cross-40 E association test's contracts, on the complete decision rule with its
shuffled-season null: delayed propagation is resolved east-leading, a shared standing
oscillation is association without clear direction, independent filtered variability is
no resolved association, and repeated correlation peaks from a periodic signal are
ambiguous rather than east-leading."""
import importlib.util
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def _load():
    spec = importlib.util.spec_from_file_location("x40_under_test", os.path.join(ROOT, "scripts", "pilot_cross40_association.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _case(C, kind, seed=1, n_perm=300):
    P = C.P
    rng = np.random.default_rng(seed)
    w = P.lanczos_bandpass()
    S, T, pad = 24, 488, 40
    delays = (6, 9, 11)                                  # six-hourly steps to 35, 30 and 25 E, about 7 degrees per day
    east, west = [], []
    for s in range(S):
        t = np.arange(T + pad)
        if kind == "periodic":
            phase = rng.uniform(0, 2 * np.pi)
            base = np.sin(2 * np.pi * t * P.DT_DAYS / 4.0 + phase)
            e = base + 0.2 * rng.standard_normal(T + pad)
            ws = [np.sin(2 * np.pi * (t - d) * P.DT_DAYS / 4.0 + phase) + 0.2 * rng.standard_normal(T + pad) for d in delays]
        else:
            source = rng.standard_normal(T + pad)
            e = source
            if kind == "delayed":
                ws = [np.roll(source, d) + 0.7 * rng.standard_normal(T + pad) for d in delays]
            elif kind == "standing":
                ws = [source + 0.7 * rng.standard_normal(T + pad) for _ in delays]
            else:
                ws = [rng.standard_normal(T + pad) for _ in delays]
        east.append(P.filter_supported(e[pad:], w))
        west.append(np.stack([P.filter_supported(x[pad:], w) for x in ws], axis=1))
    perms = C.derangements(S, n_perm, seed=7)
    return C.run_case(np.array(east), np.array(west), perms, C.ALPHA / C.PRIMARY_TESTS)


def test_delayed_propagation_is_resolved_east_leading():
    C = _load()
    r = _case(C, "delayed")
    assert r["category"] == "resolved east-leading association"
    assert [p["lag_days"] for p in r["observed"]["peaks"]] == [1.5, 2.25, 2.75] and not any(p["ambiguous"] for p in r["observed"]["peaks"])


def test_a_shared_standing_oscillation_is_association_without_clear_direction():
    C = _load()
    r = _case(C, "standing")
    assert r["category"] == "association without clear direction"
    assert all(p["lag_days"] == 0.0 for p in r["observed"]["peaks"]) and r["observed"]["east_criterion"] is False


def test_independent_filtered_variability_is_no_resolved_association():
    C = _load()
    r = _case(C, "independent")
    assert r["category"] == "no resolved association" and r["p_A"] > C.ALPHA / C.PRIMARY_TESTS


def test_repeated_peaks_from_a_periodic_signal_are_ambiguous_not_east_leading():
    C = _load()
    r = _case(C, "periodic")
    assert any(p["ambiguous"] for p in r["observed"]["peaks"]) and r["observed"]["east_criterion"] is False
    assert r["category"] == "association without clear direction"


def test_the_null_never_pairs_a_season_with_itself_and_uses_one_permutation_for_all_points():
    C = _load()
    perms = C.derangements(24, 50, seed=3)
    assert perms.shape == (50, 24) and not np.any(perms == np.arange(24)) and all(sorted(p) == list(range(24)) for p in perms)
    lags = np.array(C.LAGS) * C.P.DT_DAYS
    r = np.zeros((len(C.LAGS), 3)); i0 = list(C.LAGS).index(0)
    for k, d in enumerate((4, 8, 12)):                                   # main peaks at 1, 2 and 3 days, single peaked
        r[i0 + d, k] = 0.5
    st = C.statistics(r, lags)
    assert st["east_criterion"] is True and st["S_E"] == 0.5
    r[i0 + 2, 2], r[i0 + 12, 2] = 0.5, 0.0                               # 25 E now peaks at 0.5 days, before 30 E
    assert C.statistics(r, lags)["east_criterion"] is False
