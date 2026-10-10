#!/usr/bin/env python3
"""A bounded test of east-leading statistical association across 40 E.

Declared in `CROSS40_ASSOCIATION_TEST_2026-10-09.md` before any result. It reuses the
propagation study's preparation unchanged (`pilot_eastern_propagation.py`: the seasons,
levels, bands, terrain rule, filter support and lag sums). For each eastern base (45 and
50 E) and each band and level, the correlation of the band-passed base series with the
band-passed series at 35, 30 and 25 E is computed at lags from -4 to +4 days, positive lag
meaning the western series is later, from season sums pooled over same-season pairs. The
null pairs each eastern season with a different season's western series by one random
derangement common to the three western points, preserving calendar alignment and valid
support, and the complete selection is applied to every realization:

- association, T_A, the largest positive correlation over every lag and western point;
- east-leading, S_E, the smallest of the three points' main-peak correlations when, at
  every point, the main peak (largest positive correlation over all lags) lies at a
  positive lag, no other local maximum reaches AMBIGUITY_RATIO of it, and the main-peak
  lags do not decrease from 35 to 30 to 25 E and increase overall; otherwise -infinity.

Each p-value is (1 + realizations at least as large) / (1 + realizations). Categories:
resolved east-leading association when both p-values are at most the adjusted alpha,
association without clear direction when only the first is, no resolved association
otherwise. Failure to reject is not physical disconnection.

    python3 scripts/pilot_cross40_association.py --benchmark
    python3 scripts/pilot_cross40_association.py --out <record.json> --png <figure.png>
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pilot_eastern_propagation as P  # noqa: E402

EAST_BASES = (45.0, 50.0)
WEST_POINTS = (35.0, 30.0, 25.0)               # ordered eastward to westward
LAGS = tuple(range(-16, 17))                   # -4 to +4 days, six-hourly; positive: the western series later
AMBIGUITY_RATIO = 0.8
N_PERM = 2000
SEED = 20261010
PRIMARY_TESTS = len(P.BANDS) * len(EAST_BASES)  # 700 hPa, two bands, two bases
ALPHA = 0.05


def local_maxima(v):
    """Indices of local maxima, an end point counting when it exceeds its one neighbor."""
    v = np.asarray(v, float)
    out = []
    for i in range(len(v)):
        left = v[i - 1] if i > 0 else -np.inf
        right = v[i + 1] if i < len(v) - 1 else -np.inf
        if v[i] > left and v[i] > right:
            out.append(i)
    return out


def point_peak(r, lags_days):
    """One western point's main peak, the largest positive correlation over all lags, its
    lag, and whether another local maximum reaches AMBIGUITY_RATIO of it."""
    r = np.asarray(r, float)
    m = int(np.nanargmax(r))
    peak = float(r[m])
    others = [i for i in local_maxima(r) if i != m]
    ambiguous = bool(peak <= 0 or any(r[i] >= AMBIGUITY_RATIO * peak for i in others))
    return {"lag_days": float(lags_days[m]), "r": peak, "ambiguous": ambiguous,
            "second_r": max((float(r[i]) for i in others), default=None)}


def statistics(rmap, lags_days):
    """T_A and S_E for one realization, from correlations of shape (lags, western points)."""
    rmap = np.asarray(rmap, float)
    t_a = float(np.nanmax(rmap))
    peaks = [point_peak(rmap[:, k], lags_days) for k in range(rmap.shape[1])]
    lags = [p["lag_days"] for p in peaks]
    east = all(p["lag_days"] > 0 and not p["ambiguous"] for p in peaks) and all(a <= b for a, b in zip(lags, lags[1:])) and lags[-1] > lags[0]
    s_e = min(p["r"] for p in peaks) if east else -np.inf
    return {"T_A": t_a, "S_E": s_e, "east_criterion": bool(east), "peaks": peaks}


def derangements(n_seasons, n, seed=SEED):
    """`n` random permutations of the seasons with no season paired with itself."""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        p = rng.permutation(n_seasons)
        if not np.any(p == np.arange(n_seasons)):
            out.append(p)
    return np.array(out)


def pair_sums(east, west, lags=LAGS):
    """Lag sums for every eastern season with every western season: east (S, T), west
    (S, T, points) -> (S, S, 6, lags, points)."""
    S = len(east)
    out = np.zeros((S, S, 6, len(lags), west.shape[2]))
    for i in range(S):
        for j in range(S):
            out[i, j] = P.lag_sums(east[i], west[j], lags)
    return out


def correlations(sums):
    return P.regression_map(sums)[1]


def run_case(east, west, perms, alpha_adj, lags=LAGS):
    """The observed statistics, the null over the realizations, the p-values and the category."""
    lags_days = np.array(lags) * P.DT_DAYS
    ps = pair_sums(east, west, lags)
    S = len(east)
    r_obs = correlations(ps[np.arange(S), np.arange(S)].sum(axis=0))
    obs = statistics(r_obs, lags_days)
    null_a, null_e, null_east = [], [], 0
    for p in perms:
        st = statistics(correlations(ps[np.arange(S), p].sum(axis=0)), lags_days)
        null_a.append(st["T_A"]); null_e.append(st["S_E"]); null_east += st["east_criterion"]
    null_a, null_e = np.array(null_a), np.array(null_e)
    n = len(perms)
    p_a = (1 + int(np.sum(null_a >= obs["T_A"]))) / (1 + n)
    p_e = (1 + int(np.sum(null_e >= obs["S_E"]))) / (1 + n) if np.isfinite(obs["S_E"]) else 1.0
    if p_a <= alpha_adj and p_e <= alpha_adj:
        category = "resolved east-leading association"
    elif p_a <= alpha_adj:
        category = "association without clear direction"
    else:
        category = "no resolved association"
    return {"observed": {**obs, "S_E": None if not np.isfinite(obs["S_E"]) else obs["S_E"]}, "r": np.round(r_obs, 4).tolist(),
            "p_A": p_a, "p_E": p_e, "null_T_A_quantiles": {q: float(np.quantile(null_a, q)) for q in (0.5, 0.95, 1 - alpha_adj)},
            "null_east_criterion_rate": null_east / n, "category": category}


def season_series(year, level, rows, cols, lons, weights):
    """Per band, the filtered series at the eastern bases and the western points."""
    allrows = np.concatenate(list(rows.values()))
    data = P.read_season(year, level, allrows, cols)
    ci = np.asarray(cols) - data["col0"]
    out = {}
    for band, r in rows.items():
        ri = np.asarray(r) - data["row0"]
        series, _ = P.band_series(data["v"][:, ri][:, :, ci], data["sp"][:, ri][:, :, ci], level)
        filt = P.filter_supported(series, weights)
        pick = lambda lon: filt[:, int(np.argmin(np.abs(lons - lon)))]
        out[band] = {"east": {b: pick(b) for b in EAST_BASES}, "west": np.stack([pick(w) for w in WEST_POINTS], axis=1),
                     "supported_pairs": int(np.sum(np.isfinite(pick(EAST_BASES[0])) & np.isfinite(pick(WEST_POINTS[-1]))))}
    return out


def draw(results, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lags_days = np.array(LAGS) * P.DT_DAYS
    fig, axes = plt.subplots(len(P.LEVELS) * len(P.BANDS), len(EAST_BASES), figsize=(12, 13), sharex=True, sharey=True)
    for i, (level, band) in enumerate([(l, b) for l in P.LEVELS for b in P.BANDS]):
        for j, base in enumerate(EAST_BASES):
            ax = axes[i, j]; e = results[str(level)][band][f"{base:g}"]
            r = np.array(e["r"])
            for k, w in enumerate(WEST_POINTS):
                ax.plot(lags_days, r[:, k], label=f"{w:g} E")
            ax.axhline(e["null_T_A_quantiles"][str(1 - ALPHA / PRIMARY_TESTS)] if str(1 - ALPHA / PRIMARY_TESTS) in e["null_T_A_quantiles"] else e["null_T_A_quantiles"][1 - ALPHA / PRIMARY_TESTS],
                       color="k", ls="--", lw=0.8)
            ax.axvline(0, color="0.5", lw=0.6)
            ax.set_title(f"{level} hPa {band}, base {base:g} E: {e['category']}\np_A {e['p_A']:.4f}, p_E {e['p_E']:.4f}", fontsize=8)
            if i == 0 and j == 0:
                ax.legend(fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("lag (days), positive: western point later")
    for ax in axes[:, 0]:
        ax.set_ylabel("correlation")
    fig.suptitle("Band-passed v, eastern base against points west of 40 E; dashed: the shuffled-season null's adjusted critical value of T_A", fontsize=9)
    fig.savefig(png, dpi=90, bbox_inches="tight"); plt.close(fig)
    return png


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--benchmark", action="store_true", help="time the preparation of one season and one case's permutations, printing only times")
    ap.add_argument("--out"); ap.add_argument("--png")
    args = ap.parse_args(argv)
    weights = P.lanczos_bandpass()
    _, lons, rows, cols = P.grid_indices()
    perms = derangements(len(P.SEASONS), N_PERM)
    alpha_adj = ALPHA / PRIMARY_TESTS
    if args.benchmark:
        t0 = time.time()
        s = season_series(P.SEASONS[0], P.PRIMARY_LEVEL, rows, cols, lons, weights)
        t1 = time.time()
        rng = np.random.default_rng(0)
        east = rng.standard_normal((len(P.SEASONS), 488)); west = rng.standard_normal((len(P.SEASONS), 488, len(WEST_POINTS)))
        run_case(east, west, perms, alpha_adj)
        print(f"benchmark: one season's preparation {t1 - t0:.1f} s, one case with {N_PERM} realizations on random series {time.time() - t1:.1f} s")
        return 0
    if not args.out or not args.png:
        raise SystemExit("REFUSED: the full pass needs --out and --png")
    if os.path.exists(args.out):
        raise SystemExit(f"REFUSED: {args.out} exists and a record is never overwritten")
    t0 = time.time()
    record = {"generated_by": "scripts/pilot_cross40_association.py", "seasons": list(P.SEASONS), "held_out": P.HELD_OUT,
              "declared": {"east_bases": list(EAST_BASES), "west_points": list(WEST_POINTS), "lags_days": [l * P.DT_DAYS for l in LAGS],
                           "ambiguity_ratio": AMBIGUITY_RATIO, "n_perm": N_PERM, "seed": SEED, "primary_tests": PRIMARY_TESTS, "alpha": ALPHA,
                           "alpha_adjusted": alpha_adj, "levels": list(P.LEVELS), "bands": P.BANDS, "periods_days": list(P.PERIODS_DAYS)},
              "results": {}, "supported_pairs_per_season": {}}
    for level in P.LEVELS:
        per = [season_series(y, level, rows, cols, lons, weights) for y in P.SEASONS]
        record["results"][str(level)] = {}
        for band in P.BANDS:
            west = np.stack([s[band]["west"] for s in per])
            record["supported_pairs_per_season"][f"{level}_{band}"] = sorted({s[band]["supported_pairs"] for s in per})
            record["results"][str(level)][band] = {}
            for base in EAST_BASES:
                east = np.stack([s[band]["east"][base] for s in per])
                res = run_case(east, west, perms, alpha_adj)
                res["null_T_A_quantiles"] = {str(k): v for k, v in res["null_T_A_quantiles"].items()}
                record["results"][str(level)][band][f"{base:g}"] = res
    record["figure"] = draw(record["results"], args.png)
    record["elapsed_seconds"] = round(time.time() - t0, 1)
    with open(args.out, "x") as fh:
        json.dump(record, fh)
    print(f"done in {record['elapsed_seconds']} s, record {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
