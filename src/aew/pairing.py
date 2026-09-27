"""Pairing two track records under the predeclared specification, for stored-track
correspondences and nothing more.

The rules are rules 1 to 7 of the project's matcher specification, composed
from pieces that already exist: the great-circle distance of `aew.v1port.geometry`, the
exact-timestamp intersection the oracle comparer uses, and the maximum-cardinality
assignment by the Hungarian method that comparer uses, with the median over the shared
observations as the separation and a minimum of three shared observations. Times must
sit on the six-hour grid, as quarter-day multiples since 1900-01-01 00Z, and a record
that does not is refused rather than snapped.

A track here is a dict with `time` (days since 1900), `lat` and `lon` arrays of one
length, and an `id` the caller chooses. Only observations IN COVERAGE (inside a declared
time window and region) take part in a pairing, so the two records are compared where
both could have seen the same thing. A pair is a correspondence between two stored
tracks under these rules. It is not a recovered physical wave.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

from aew.v1port.geometry import great_circle_distance

GRID_DAYS = 0.25


def on_grid(times):
    """Every finite time an EXACT multiple of a quarter day since 1900-01-01 00Z. Quarter
    days are exactly representable, and the intersection later is exact, so any offset at
    all, however small, would silently share nothing: it is refused here instead."""
    t = np.asarray(times, dtype=float)
    t = t[np.isfinite(t)]
    return bool(np.all(np.mod(t, GRID_DAYS) == 0.0))


def coverage(track, window, region):
    """Which observations of a track lie in the common window and region: a boolean per
    observation. `window` is (first, last) in days since 1900, inclusive; `region` is
    (lat_min, lat_max, lon_min, lon_max), inclusive."""
    t, la, lo = (np.asarray(track[k], dtype=float) for k in ("time", "lat", "lon"))
    finite = np.isfinite(t) & np.isfinite(la) & np.isfinite(lo)
    return (finite & (t >= window[0]) & (t <= window[1])
            & (la >= region[0]) & (la <= region[1]) & (lo >= region[2]) & (lo <= region[3]))


def restrict(track, mask):
    """The track's observations under `mask`, in time order."""
    order = np.argsort(np.asarray(track["time"], dtype=float)[mask])
    out = {k: np.asarray(track[k], dtype=float)[mask][order] for k in ("time", "lat", "lon")}
    out["id"] = track.get("id")
    return out


def shared_separation(a, b):
    """The timestamps two tracks share exactly, and the great-circle separation in
    kilometers at each. Both tracks are assumed restricted to coverage already."""
    shared, ia, ib = np.intersect1d(a["time"], b["time"], return_indices=True)
    if shared.size == 0:
        return shared, np.array([])
    d = np.array([great_circle_distance(a["lat"][i], a["lon"][i], b["lat"][j], b["lon"][j], "km")
                  for i, j in zip(ia, ib)], dtype=float)
    return shared, d


def coexisting(side_a, side_b, min_shared=3):
    """Every pair with at least `min_shared` shared observations, whatever its separation:
    {(i, j): (n_shared, median_km, distances)}. The tolerance is applied on top."""
    out = {}
    for i, a in enumerate(side_a):
        for j, b in enumerate(side_b):
            shared, d = shared_separation(a, b)
            if shared.size >= min_shared:
                out[(i, j)] = (int(shared.size), float(np.median(d)), d)
    return out


def candidates(side_a, side_b, tolerance_km, min_shared=3, coexist=None):
    """Every eligible pair: at least `min_shared` shared observations and a median
    separation at or below the tolerance. Returns {(i, j): {n_shared, median_km,
    fraction_within}} over the indices of the two lists."""
    coexist = coexisting(side_a, side_b, min_shared) if coexist is None else coexist
    return {(i, j): {"n_shared": n, "median_km": median, "fraction_within": float(np.mean(d <= tolerance_km))}
            for (i, j), (n, median, d) in coexist.items() if median <= tolerance_km}


def assign(cands, n_a, n_b, tolerance_km):
    """One-to-one, maximum cardinality first and minimum total median separation second,
    by the Hungarian method: an ineligible cell costs more than any set of eligible
    assignments can total, so leaving an eligible pair unassigned never pays. Ties
    between equal-cost solutions fall to the solver given the rows and columns in the
    order the caller gave, which the caller records."""
    if not cands or n_a == 0 or n_b == 0:
        return []
    forbidden = tolerance_km * (min(n_a, n_b) + 1) + 1.0
    cost = np.full((n_a, n_b), forbidden, dtype=float)
    for (i, j), c in cands.items():
        cost[i, j] = c["median_km"]
    rows, cols = linear_sum_assignment(cost)
    return [(int(i), int(j)) for i, j in zip(rows, cols) if (int(i), int(j)) in cands]


def pair(side_a, side_b, tolerance_km, min_shared=3):
    """The pairing of two records already restricted to coverage. Returns the assigned
    pairs with their separation and overlap, and for each side the tracks with no
    eligible counterpart and the tracks that had one or more eligible counterparts but
    lost the one-to-one assignment, each with its candidate count."""
    coexist = coexisting(side_a, side_b, min_shared)
    cands = candidates(side_a, side_b, tolerance_km, min_shared, coexist)
    assigned = assign(cands, len(side_a), len(side_b), tolerance_km)
    nearest_a, nearest_b = {}, {}
    for (i, j), (n, median, d) in coexist.items():             # the nearest miss: the closest coexisting track, at any separation
        if median < nearest_a.get(i, (np.inf,))[0]:
            nearest_a[i] = (median, side_b[j]["id"], n)
        if median < nearest_b.get(j, (np.inf,))[0]:
            nearest_b[j] = (median, side_a[i]["id"], n)
    a_assigned = {i for i, _ in assigned}
    b_assigned = {j for _, j in assigned}
    a_counts = {i: 0 for i in range(len(side_a))}
    b_counts = {j: 0 for j in range(len(side_b))}
    for i, j in cands:
        a_counts[i] += 1
        b_counts[j] += 1
    pairs = [{"a": side_a[i]["id"], "b": side_b[j]["id"], **cands[(i, j)],
              "a_candidates": a_counts[i], "b_candidates": b_counts[j]} for i, j in assigned]

    def unassigned(side, counts, taken, nearest):
        none, lost = [], []
        for k, t in enumerate(side):
            if k in taken:
                continue
            entry = {"id": t["id"], "candidates": counts[k]}
            if counts[k] == 0:
                if k in nearest:                                        # the closest coexisting track, named, and how long they coexist
                    entry["nearest_median_km"], entry["nearest_id"], entry["nearest_shared"] = float(nearest[k][0]), nearest[k][1], int(nearest[k][2])
                else:
                    entry["nearest_median_km"], entry["nearest_id"], entry["nearest_shared"] = None, None, 0   # nothing shares three observations
                none.append(entry)
            else:
                lost.append(entry)
        return none, lost
    a_none, a_lost = unassigned(side_a, a_counts, a_assigned, nearest_a)
    b_none, b_lost = unassigned(side_b, b_counts, b_assigned, nearest_b)
    return {"tolerance_km": tolerance_km, "min_shared": min_shared, "pairs": pairs,
            "candidate_pairs": len(cands),
            "a": {"n": len(side_a), "assigned": len(a_assigned), "no_candidate": a_none, "lost_assignment": a_lost},
            "b": {"n": len(side_b), "assigned": len(b_assigned), "no_candidate": b_none, "lost_assignment": b_lost}}


def distributions(pairs):
    """Median, 10th and 90th percentiles of the separation and of the overlap over the
    assigned pairs, and the mean fraction of shared observations within the tolerance."""
    if not pairs:
        return None
    sep = np.array([p["median_km"] for p in pairs])
    ovl = np.array([p["n_shared"] for p in pairs])
    return {"separation_km": {"10": float(np.percentile(sep, 10)), "50": float(np.median(sep)), "90": float(np.percentile(sep, 90))},
            "shared_observations": {"10": float(np.percentile(ovl, 10)), "50": float(np.median(ovl)), "90": float(np.percentile(ovl, 90))},
            "mean_fraction_within": float(np.mean([p["fraction_within"] for p in pairs]))}
