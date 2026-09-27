"""The pairing module under the specification: grid validation, coverage, exact-time
sharing with great-circle separation, the median as the separation, the minimum overlap,
the tolerance, maximum-cardinality one-to-one assignment (the three-edge case where a
greedy matcher returns one pair while two exist), and the reporting of tracks that had
a candidate but lost the assignment."""
import numpy as np
import pytest

from aew import pairing as P

D = 32872.0        # January 1 1990 in days since 1900


def _track(tid, t0, lat, lon, n=4, dlon=-0.5):
    t = t0 + 0.25 * np.arange(n)
    return {"id": tid, "time": t, "lat": np.full(n, float(lat)), "lon": lon + dlon * np.arange(n)}


def test_grid_validation_accepts_quarter_days_and_refuses_anything_else():
    assert P.on_grid([D, D + 0.25, D + 100.5]) and P.on_grid([])
    assert not P.on_grid([D, D + 0.2]) and not P.on_grid([D + 1 / 6])
    assert not P.on_grid([D + 1e-7])                                       # any offset at all shares nothing later, so it is refused here
    assert P.on_grid([D, np.nan])                                         # a missing time is not off the grid


def test_coverage_restricts_to_window_and_region_and_keeps_time_order():
    t = {"id": 1, "time": np.array([D + 2, D, D + 1, D + 3]), "lat": np.array([10.0, 10.0, 40.0, 10.0]), "lon": np.array([0.0, 0.0, 0.0, 50.0])}
    m = P.coverage(t, (D, D + 2), (-20, 35, -140, 40))
    assert m.tolist() == [True, True, False, False]                        # D+1 is out by latitude, D+3 by window and longitude
    r = P.restrict(t, m)
    assert r["time"].tolist() == [D, D + 2] and r["id"] == 1


def test_shared_separation_is_exact_in_time_and_great_circle_in_km():
    a = _track(1, D, 10.0, 0.0, dlon=0.0)
    b = _track(2, D + 0.25, 10.0, 1.0, dlon=0.0)                          # shares three of a's four times, one degree east
    shared, d = P.shared_separation(a, b)
    assert shared.size == 3 and abs(d[0] - 109.5) < 2.0                    # one degree of longitude at 10 N is about 109.5 km
    c = _track(3, D + 0.1, 10.0, 0.0)                                     # off the grid by 2.4 hours: nothing shared
    assert P.shared_separation(a, c)[0].size == 0


def test_candidates_need_three_shared_and_a_median_within_tolerance():
    a = _track(1, D, 10.0, 0.0, n=6)
    near = _track(2, D, 10.0, 0.5, n=6)                                    # about 55 km away throughout
    short = _track(3, D + 1.0, 10.0, 0.5, n=2)                             # two shared observations only
    far = _track(4, D, 10.0, 6.0, n=6)                                     # about 650 km away
    c = P.candidates([a], [near, short, far], 500.0, 3)
    assert set(c) == {(0, 0)} and c[(0, 0)]["n_shared"] == 6 and c[(0, 0)]["fraction_within"] == 1.0
    assert set(P.candidates([a], [near, short, far], 350.0, 3)) == {(0, 0)}
    assert set(P.candidates([a], [far], 1000.0, 3)) == {(0, 0)}


def test_the_median_not_the_mean_is_the_separation():
    a = _track(1, D, 10.0, 0.0, n=5, dlon=0.0)
    b = {"id": 2, "time": a["time"].copy(), "lat": np.full(5, 10.0), "lon": np.array([0.5, 0.5, 0.5, 0.5, 20.0])}   # one wild step
    c = P.candidates([a], [b], 500.0, 3)
    assert (0, 0) in c and c[(0, 0)]["median_km"] < 60 and c[(0, 0)]["fraction_within"] == pytest.approx(0.8)


def test_maximum_cardinality_takes_two_pairs_where_greedy_would_take_one():
    """The three-edge case: A1 is closest to B1, but A2 can only pair with B1, while A1
    could also pair with B2. Greedy assigns A1-B1 and leaves A2 alone; maximum
    cardinality assigns A1-B2 and A2-B1."""
    a1 = _track("A1", D, 10.0, 0.0, n=6, dlon=0.0)
    a2 = _track("A2", D, 10.0, 1.0, n=6, dlon=0.0)
    b1 = _track("B1", D, 10.0, 0.5, n=6, dlon=0.0)                         # 55 km from A1, 55 km from A2
    b2 = _track("B2", D, 10.0, -4.0, n=6, dlon=0.0)                        # 438 km from A1, 547 km from A2, so A2 cannot pair with it
    b1["lon"] = np.full(6, 0.2)                                            # now closer to A1 (22 km) than to A2 (88 km)
    r = P.pair([a1, a2], [b1, b2], 500.0, 3)
    assigned = {(p["a"], p["b"]) for p in r["pairs"]}
    assert assigned == {("A1", "B2"), ("A2", "B1")}
    assert r["a"]["assigned"] == 2 and r["b"]["assigned"] == 2 and r["candidate_pairs"] == 3


def test_lost_assignment_and_no_candidate_are_reported_separately_with_counts():
    a = _track("A", D, 10.0, 0.0, n=6, dlon=0.0)
    b1 = _track("B1", D, 10.0, 0.2, n=6, dlon=0.0)
    b2 = _track("B2", D, 10.0, 0.6, n=6, dlon=0.0)                         # also a candidate for A, but A pairs once
    b3 = _track("B3", D, 10.0, 30.0, n=6, dlon=0.0)                        # no candidate anywhere
    r = P.pair([a], [b1, b2, b3], 500.0, 3)
    assert [p["b"] for p in r["pairs"]] == ["B1"] and r["pairs"][0]["a_candidates"] == 2
    assert r["b"]["lost_assignment"] == [{"id": "B2", "candidates": 1}]
    assert r["b"]["no_candidate"][0]["id"] == "B3" and r["b"]["no_candidate"][0]["candidates"] == 0
    assert 3200 < r["b"]["no_candidate"][0]["nearest_median_km"] < 3400              # the nearest miss, 30 degrees away at 10 N
    assert r["a"]["lost_assignment"] == [] and r["a"]["no_candidate"] == []
    lonely = _track("L", D + 50, 10.0, 0.0, n=6, dlon=0.0)                           # coexists with nothing
    r2 = P.pair([lonely], [b1], 500.0, 3)
    assert r2["a"]["no_candidate"] == [{"id": "L", "candidates": 0, "nearest_median_km": None, "nearest_id": None, "nearest_shared": 0}]
    assert r["b"]["no_candidate"][0]["nearest_id"] == "A" and r["b"]["no_candidate"][0]["nearest_shared"] == 6
    d = P.distributions(r["pairs"])
    assert d["shared_observations"]["50"] == 6 and d["separation_km"]["50"] < 30 and d["mean_fraction_within"] == 1.0
    assert P.distributions([]) is None


def test_ties_are_deterministic_in_the_given_order():
    a = _track("A", D, 10.0, 0.0, n=6, dlon=0.0)
    b1 = _track("B1", D, 10.0, 0.3, n=6, dlon=0.0)
    b2 = _track("B2", D, 10.0, -0.3, n=6, dlon=0.0)                        # exactly the same separation as B1
    first = P.pair([a], [b1, b2], 500.0, 3)["pairs"][0]["b"]
    again = P.pair([a], [b1, b2], 500.0, 3)["pairs"][0]["b"]
    assert first == again


def test_exactly_three_shared_observations_and_exactly_the_tolerance_are_eligible():
    a = _track(1, D, 10.0, 0.0, n=3, dlon=0.0)
    b = _track(2, D, 10.0, 0.5, n=3, dlon=0.0)                               # exactly three shared, about 55 km apart
    c = P.candidates([a], [b], 500.0, 3)
    assert (0, 0) in c and c[(0, 0)]["n_shared"] == 3
    assert P.candidates([a], [b], 500.0, 4) == {}                          # four required: not eligible
    shared, d = P.shared_separation(a, b)
    exact = float(np.median(d))
    assert (0, 0) in P.candidates([a], [b], exact, 3)                      # a median exactly at the tolerance is within it
    assert P.candidates([a], [b], np.nextafter(exact, 0.0), 3) == {}       # just below it is not


def test_the_secondary_objective_is_the_total_median_separation_not_a_transform_of_it():
    """A complete two-by-two candidate matrix [[1, 6], [6, 10]] in units of about 110 km per
    degree: the maximum-cardinality solutions are the diagonal (1 + 10 = 11) and the
    anti-diagonal (6 + 6 = 12); the declared objective takes the diagonal, and a squared
    cost would take the anti-diagonal (1 + 100 against 36 + 36)."""
    t = D + 0.25 * np.arange(6)
    a1 = {"id": "A1", "time": t}
    a1 = {"id": "A1", "time": a1["time"], "lat": np.zeros(6), "lon": np.zeros(6)}
    a2 = {"id": "A2", "time": a1["time"], "lat": np.zeros(6), "lon": np.full(6, 6.0)}
    b1 = {"id": "B1", "time": a1["time"], "lat": np.zeros(6), "lon": np.full(6, 1.0)}          # A1-B1 1, A2-B1 5
    b2 = {"id": "B2", "time": a1["time"], "lat": np.full(6, 6.0), "lon": np.zeros(6)}          # A1-B2 6, A2-B2 sqrt(72) = 8.5
    # totals: diagonal 1 + 8.5 = 9.5, anti-diagonal 6 + 5 = 11; squared: 1 + 72 = 73 against 36 + 25 = 61, which reverses the choice
    r = P.pair([a1, a2], [b1, b2], 2000.0, 3)
    assert {(p["a"], p["b"]) for p in r["pairs"]} == {("A1", "B1"), ("A2", "B2")}


def test_distribution_tails_are_the_tenth_and_ninetieth_percentiles_not_the_median():
    pairs = [{"median_km": s, "n_shared": n, "fraction_within": 1.0} for s, n in ((10, 3), (20, 5), (30, 8), (40, 13), (400, 30))]
    d = P.distributions(pairs)
    assert d["separation_km"]["50"] == 30 and d["separation_km"]["10"] < 20 and d["separation_km"]["90"] > 200
    assert d["shared_observations"]["50"] == 8 and d["shared_observations"]["90"] > 20 and d["shared_observations"]["10"] < 5
