#!/usr/bin/env python3
"""Do individual trough episodes cross 40 E? A bounded episode-continuity study.

NOT VALIDATED FOR THE REALISTIC REGIME AND NOT RUN ON REAL DATA (2026-10-10). On synthetic
seasons it resolves a uniform traveling signal, but a synthetic crossing whose amplitude
dips at 40 E, a stress test motivated by the measured variability minimum there, is not
distinguished from the joined null, and single crossings against a null of almost none
gave false directions. `--validate` reproduces those cases. See
`CROSS40_EPISODE_STUDY_2026-10-10.md`.

Declared in `CROSS40_EPISODE_STUDY_2026-10-10.md` before any result. It reuses the
propagation study's preparation unchanged (`pilot_eastern_propagation.py`): the 24 seasons
without 2005, each processed on its own, the band means over terrain-valid cells, the
2.5 to 6 day band-pass inside supported runs with edges dropped, the southern and northern
bands, 700 hPa primary and 600 hPa the single sensitivity.

What tells successive episodes apart is time resolution. At every six-hourly step the
trough axes are the longitudes where the band-passed meridional wind turns from northerly
on the west to southerly on the east, kept where the local amplitude reaches half the
median band-passed standard deviation over the analysis longitudes, one threshold per band
and level. A threshold scaled to each longitude's own variability let noise-level axes in
a quiet zone carry episodes across it on synthetic data, so a weak crossing through the
amplitude minimum near 36 to 40 E can be missed, and a failure to find one is not absence. Axes are linked from one
step to the next within LINK_DEG, either direction, with at most one missing step. One
trough moves at most a few degrees in six hours while successive troughs lie a wavelength
apart, so a link cannot pass from one episode to the next, unlike a choice among
correlation peaks a period apart. No direction is assumed.

An episode crosses a line westward when it lies at least CROSS_MARGIN_DEG east of it and
later at least as far west, and eastward the other way round. Westward and eastward
crossings are counted at 40 E (primary) and 20 E (the western comparison).

A westward excess alone is not evidence. On synthetic western waves beside an eastern
standing pattern, episodes chained through the noise between them and were carried west,
giving westward crossings where none exist. So the observed counts are compared with a
null of unrelated seasons joined across the line, with east of the line from one season
and west of it from a different season by a derangement, blended over the margin zone
with weights that keep the variance of independent inputs, and the complete rule run on
every realization. The join was meant to keep chance chaining and remove same-season
continuity. That is unvalidated, since variance normalization does not preserve spatial
structure and blending can create an axis absent from either input.

    python3 scripts/pilot_cross40_episodes.py --benchmark
    python3 scripts/pilot_cross40_episodes.py --out <record.json> --png <figure.png>
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pilot_eastern_propagation as P  # noqa: E402

LON_RANGE = (5.0, 60.0)
AMPLITUDE_HALF_WIDTH_DEG = 2.5
AMPLITUDE_FRACTION = 0.5
LINK_DEG = 3.0                    # per six-hourly step, either direction
MAX_GAP_STEPS = 1                 # at most one missing step, the link distance doubling across it
CROSS_MARGIN_DEG = 2.0
PRIMARY_LINE = 40.0
CONTROL_LINE = 20.0
N_PERM = 400
SEED = 20261011
PRIMARY_TESTS = len(P.BANDS)      # 700 hPa, two bands, at 40 E
ALPHA = 0.05


def axes_by_step(vf, lons, threshold):
    """Trough axes at every six-hourly step of a field (steps, longitudes) on a regular
    grid: sign changes between adjacent finite points from negative on the west to zero or
    positive on the east, located by linear interpolation, kept where half the difference
    across +/- AMPLITUDE_HALF_WIDTH_DEG, both ends finite and inside the grid, reaches
    `threshold`, one value for the whole field. Returns one list of positions per step."""
    v = np.asarray(vf, float)
    lons = np.asarray(lons, float)
    dx = lons[1] - lons[0]
    a, b = v[:, :-1], v[:, 1:]
    cond = np.isfinite(a) & np.isfinite(b) & (a < 0) & (b >= 0)
    t_idx, j_idx = np.nonzero(cond)
    x = lons[j_idx] + (0.0 - a[t_idx, j_idx]) / (b[t_idx, j_idx] - a[t_idx, j_idx]) * dx

    def at(xq):
        fi = (xq - lons[0]) / dx
        i0 = np.floor(fi).astype(int)
        ok = (i0 >= 0) & (i0 + 1 < lons.size)
        i0c = np.clip(i0, 0, lons.size - 2)
        fr = fi - i0c
        val = v[t_idx, i0c] * (1 - fr) + v[t_idx, i0c + 1] * fr
        return np.where(ok, val, np.nan)
    inside = (x - AMPLITUDE_HALF_WIDTH_DEG >= lons[0]) & (x + AMPLITUDE_HALF_WIDTH_DEG <= lons[-1])
    s = 0.5 * (at(x + AMPLITUDE_HALF_WIDTH_DEG) - at(x - AMPLITUDE_HALF_WIDTH_DEG))
    keep = inside & np.isfinite(s) & (s >= threshold)
    out = [[] for _ in range(v.shape[0])]
    for t, xx in zip(t_idx[keep], x[keep]):
        out[t].append(float(xx))
    return out


def link_episodes(axes_by_step):
    """Episodes from per-step axis positions: each axis joins the nearest open episode
    seen at the previous step within LINK_DEG, or two steps back within twice that,
    one to one and nearest first, else starts a new episode. Returns lists of
    (step, position)."""
    episodes, open_ids = [], []
    for t, positions in enumerate(axes_by_step):
        cands = []
        for k in open_ids:
            last_t, last_x = episodes[k][-1]
            gap = t - last_t
            if gap < 1 or gap > MAX_GAP_STEPS + 1:
                continue
            for i, x in enumerate(positions):
                d = abs(x - last_x)
                if d <= LINK_DEG * gap:
                    cands.append((d, k, i))
        used_k, used_i = set(), set()
        for d, k, i in sorted(cands):
            if k in used_k or i in used_i:
                continue
            episodes[k].append((t, positions[i])); used_k.add(k); used_i.add(i)
        new_ids = []
        for i, x in enumerate(positions):
            if i not in used_i:
                episodes.append([(t, x)]); new_ids.append(len(episodes) - 1)
        open_ids = [k for k in open_ids + new_ids if t - episodes[k][-1][0] <= MAX_GAP_STEPS]
    return episodes


def crossings(episode, line):
    """Westward and eastward crossings of `line` by one episode, with the six-hourly
    steps taken between the last position on one side and the first on the other."""
    west, east, spans = 0, 0, []
    side, side_t = None, None
    for t, x in episode:
        here = "east" if x >= line + CROSS_MARGIN_DEG else "west" if x <= line - CROSS_MARGIN_DEG else None
        if here is None:
            continue
        if side is not None and here != side:
            if here == "west":
                west += 1
            else:
                east += 1
            spans.append(t - side_t)
        side, side_t = here, t
    return west, east, spans


def season_counts(vf, lons, threshold, line):
    """For one season's band-passed field (steps, longitudes): the westward and eastward
    crossings of `line`, their spans, the number of episodes and of episodes ever east of
    the line."""
    eps = link_episodes(axes_by_step(vf, lons, threshold))
    w = e = 0; spans = []
    for ep in eps:
        a, b, s = crossings(ep, line)
        w += a; e += b; spans += s
    return {"west": w, "east": e, "spans_steps": spans, "episodes": len(eps),
            "eastern_episodes": sum(1 for ep in eps if any(x >= line + CROSS_MARGIN_DEG for _, x in ep))}


def hybrid(east_src, west_src, lons, line):
    """East of the margin zone from `east_src`, west of it from `west_src`, blended across
    it with weights whose squares sum to one, so independent fields keep their variance."""
    w = np.clip((np.asarray(lons, float) - (line - CROSS_MARGIN_DEG)) / (2 * CROSS_MARGIN_DEG), 0.0, 1.0)
    return (w * east_src + (1 - w) * west_src) / np.sqrt(w ** 2 + (1 - w) ** 2)


def derangements(n_seasons, n, seed=SEED):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        p = rng.permutation(n_seasons)
        if not np.any(p == np.arange(n_seasons)):
            out.append(p)
    return np.array(out)


def joining_test(fields, lons, threshold, line, perms, alpha_adj):
    """Observed crossings of `line` against unrelated seasons joined across it, the
    complete rule run on every realization. Returns the counts, the null summary, the
    p-values and the category."""
    obs = [season_counts(f, lons, threshold, line) for f in fields]
    w_obs = sum(o["west"] for o in obs); e_obs = sum(o["east"] for o in obs)
    w_null, e_null = [], []
    for p in perms:
        cs = [season_counts(hybrid(fields[i], fields[j], lons, line), lons, threshold, line) for i, j in enumerate(p)]
        w_null.append(sum(c["west"] for c in cs)); e_null.append(sum(c["east"] for c in cs))
    w_null, e_null = np.array(w_null), np.array(e_null)
    n = len(perms)
    p_w = (1 + int(np.sum(w_null >= w_obs))) / (1 + n)
    p_e = (1 + int(np.sum(e_null >= e_obs))) / (1 + n)
    if w_obs + e_obs == 0:
        category = "no crossing episodes identified"
    elif p_w <= alpha_adj:
        category = "westward crossing episodes beyond chance joining"
    elif p_e <= alpha_adj:
        category = "eastward crossing episodes beyond chance joining"
    else:
        category = "crossing episodes not distinguished from chance joining"
    S = len(fields)
    spans = [s for o in obs for s in o["spans_steps"]]
    return {"west_per_season": w_obs / S, "east_per_season": e_obs / S,
            "null_west_per_season": {"mean": float(w_null.mean()) / S, "q95": float(np.quantile(w_null, 0.95)) / S},
            "null_east_per_season": {"mean": float(e_null.mean()) / S, "q95": float(np.quantile(e_null, 0.95)) / S},
            "p_west": p_w, "p_east": p_e, "category": category,
            "episodes_per_season": float(np.mean([o["episodes"] for o in obs])),
            "episodes_ever_east_of_line_per_season": float(np.mean([o["eastern_episodes"] for o in obs])),
            "crossing_span_steps": {"median": float(np.median(spans)) if spans else None, "n": len(spans)},
            "per_season": [{"west": o["west"], "east": o["east"]} for o in obs]}


def prepare(level, rows, cols, lons_all, weights):
    """Per band, the band-passed season fields on the analysis longitudes."""
    sel = (lons_all >= LON_RANGE[0] - 1e-6) & (lons_all <= LON_RANGE[1] + 1e-6)
    out = {band: [] for band in P.BANDS}
    allrows = np.concatenate(list(rows.values()))
    for year in P.SEASONS:
        data = P.read_season(year, level, allrows, cols)
        ci = np.asarray(cols) - data["col0"]
        for band, r in rows.items():
            ri = np.asarray(r) - data["row0"]
            series, _ = P.band_series(data["v"][:, ri][:, :, ci], data["sp"][:, ri][:, :, ci], level)
            out[band].append(P.filter_supported(series, weights)[:, sel])
    return out, lons_all[sel]


def pooled_sigma(fields):
    stack = np.concatenate(fields, axis=0)
    return np.sqrt(np.nanmean(stack ** 2, axis=0))


def analyze(fields, lons, n_perm=N_PERM, seed=SEED, alpha_adj=ALPHA / PRIMARY_TESTS):
    sigma = pooled_sigma(fields)
    threshold = AMPLITUDE_FRACTION * float(np.median(sigma))
    perms = derangements(len(fields), n_perm, seed)
    return {"primary": joining_test(fields, lons, threshold, PRIMARY_LINE, perms, alpha_adj),
            "control": joining_test(fields, lons, threshold, CONTROL_LINE, perms, alpha_adj),
            "amplitude_threshold_m_s": threshold, "sigma_median_m_s": float(np.median(sigma)),
            "sigma_at_40E": float(np.interp(40.0, lons, sigma)), "sigma_at_20E": float(np.interp(20.0, lons, sigma))}


def synthetic_case(kind, seed=0, raw=0.3, floor=1.0, locked=True, seasons=24):
    """Band-passed synthetic seasons on the analysis longitudes. `traveling`, westward at
    7 degrees per day with a 28-degree wavelength under an amplitude envelope dipping to
    `floor` at 40 E; `standing`, an antinode at 40 E; `separate`, the traveling signal
    west of 36 E beside the standing one east of 44 E, phase-locked when `locked`. Noise
    is smooth over about 3 degrees with raw standard deviation `raw`."""
    rng = np.random.default_rng(seed)
    w = P.lanczos_bandpass()
    lons = np.arange(LON_RANGE[0], LON_RANGE[1] + 0.25, 0.5)
    t = np.arange(488) * P.DT_DAYS
    k = np.exp(-0.5 * (np.arange(-12, 13) * 0.5 / 3.0) ** 2); k /= k.sum()
    env = (1 - (1 - floor) * np.exp(-0.5 * ((lons - 40.0) / 3.0) ** 2))[None, :]
    out = []
    for _ in range(seasons):
        ph = rng.uniform(0, 2 * np.pi); ph2 = ph if locked else rng.uniform(0, 2 * np.pi)
        trav = np.sin(2 * np.pi * (lons[None, :] + 7.0 * t[:, None]) / 28.0 + ph)
        stand = np.cos(2 * np.pi * (lons[None, :] - 40.0) / 28.0) * np.sin(2 * np.pi * t[:, None] / 4.0 + ph2)
        noise = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 1, rng.standard_normal((488, lons.size)))
        noise = noise / noise.std() * raw
        if kind == "traveling":
            f = env * trav
        elif kind == "standing":
            f = stand
        else:
            f = np.clip((36.0 - lons) / 6.0, 0, 1)[None, :] * trav + np.clip((lons - 44.0) / 6.0, 0, 1)[None, :] * stand
        out.append(P.filter_supported(f + noise, w))
    return out, lons


VALIDATION_CASES = (
    ("uniform traveling", dict(kind="traveling")),
    ("standing", dict(kind="standing")),
    ("separate, phase-locked, gap at 0.46 of the median", dict(kind="separate", raw=1.1)),
    ("separate, phase-locked, gap at 0.56 of the median", dict(kind="separate", raw=1.4)),
    ("separate, independent phases, gap at 0.56 of the median", dict(kind="separate", raw=1.4, locked=False)),
    ("traveling, amplitude dipping to 0.5 at 40 E", dict(kind="traveling", floor=0.5, raw=0.5)),
    ("traveling, amplitude dipping to 0.5 at 40 E, more noise", dict(kind="traveling", floor=0.5, raw=0.8)),
    ("traveling, amplitude dipping to 0.3 at 40 E, more noise", dict(kind="traveling", floor=0.3, raw=0.8)),
)


def validate(n_perm=100):
    rows = []
    for name, kw in VALIDATION_CASES:
        fields, lons = synthetic_case(**kw)
        r = analyze(fields, lons, n_perm=n_perm)
        rows.append({"case": name, "parameters": kw, "sigma_40_over_median": r["sigma_at_40E"] / r["sigma_median_m_s"],
                     **{key: {k: r[key][k] for k in ("category", "west_per_season", "east_per_season", "null_west_per_season", "null_east_per_season", "p_west", "p_east")}
                        for key in ("primary", "control")}})
    return rows


def draw(results, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    cases = [(l, b) for l in P.LEVELS for b in P.BANDS]
    for ax, key, line in ((axes[0], "primary", PRIMARY_LINE), (axes[1], "control", CONTROL_LINE)):
        for i, (level, band) in enumerate(cases):
            e = results[str(level)][band][key]
            ax.plot([i - 0.1], [e["west_per_season"]], "o", color="C0", label="westward, observed" if i == 0 else None)
            ax.plot([i - 0.1], [e["null_west_per_season"]["q95"]], "_", color="C0", ms=14, label="westward, null 95th percentile" if i == 0 else None)
            ax.plot([i + 0.1], [e["east_per_season"]], "s", color="C3", label="eastward, observed" if i == 0 else None)
            ax.plot([i + 0.1], [e["null_east_per_season"]["q95"]], "_", color="C3", ms=14, label="eastward, null 95th percentile" if i == 0 else None)
            ax.text(i, max(e["west_per_season"], e["null_west_per_season"]["q95"]) * 1.05 + 0.05, f"p_W {e['p_west']:.3f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(range(len(cases))); ax.set_xticklabels([f"{l} hPa\n{b}" for l, b in cases], fontsize=8)
        ax.set_title(f"crossing episodes of {line:g} E per season, observed and unrelated seasons joined", fontsize=9)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("crossing episodes per season")
    fig.savefig(png, dpi=90, bbox_inches="tight"); plt.close(fig)
    return png


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--benchmark", action="store_true", help="time one level's preparation and analysis and print only times")
    ap.add_argument("--validate", help="run the synthetic validation cases only and write their record here; no real data")
    ap.add_argument("--out"); ap.add_argument("--png")
    args = ap.parse_args(argv)
    if args.validate:
        if os.path.exists(args.validate):
            raise SystemExit(f"REFUSED: {args.validate} exists and a record is never overwritten")
        t0 = time.time()
        rec = {"generated_by": "scripts/pilot_cross40_episodes.py --validate", "script_sha256": __import__("exact_tracks").digest(os.path.abspath(__file__)),
               "real_data": "none", "n_perm": 100, "cases": validate()}
        rec["elapsed_seconds"] = round(time.time() - t0, 1)
        with open(args.validate, "x") as fh:
            json.dump(rec, fh, indent=1)
        for c in rec["cases"]:
            print(f"{c['case']}: 40 E {c['primary']['category']}, 20 E {c['control']['category']}")
        return 0
    weights = P.lanczos_bandpass()
    _, lons_all, rows, cols = P.grid_indices()
    if args.benchmark:
        t0 = time.time()
        fields, lons = prepare(P.PRIMARY_LEVEL, rows, cols, lons_all, weights)
        t1 = time.time()
        for band in P.BANDS:
            analyze(fields[band], lons)
        print(f"benchmark: one level's preparation {t1 - t0:.1f} s, its analysis in both bands {time.time() - t1:.1f} s")
        return 0
    if not args.out or not args.png:
        raise SystemExit("REFUSED: the full pass needs --out and --png")
    if os.path.exists(args.out):
        raise SystemExit(f"REFUSED: {args.out} exists and a record is never overwritten")
    t0 = time.time()
    record = {"generated_by": "scripts/pilot_cross40_episodes.py", "seasons": list(P.SEASONS), "held_out": P.HELD_OUT,
              "declared": {"lon_range": list(LON_RANGE), "amplitude_half_width_deg": AMPLITUDE_HALF_WIDTH_DEG, "amplitude_fraction": AMPLITUDE_FRACTION,
                           "link_deg_per_step": LINK_DEG, "max_gap_steps": MAX_GAP_STEPS, "cross_margin_deg": CROSS_MARGIN_DEG,
                           "primary_line": PRIMARY_LINE, "control_line": CONTROL_LINE, "n_perm": N_PERM, "seed": SEED, "primary_tests": PRIMARY_TESTS,
                           "alpha": ALPHA, "levels": list(P.LEVELS), "bands": P.BANDS, "periods_days": list(P.PERIODS_DAYS)},
              "results": {}}
    for level in P.LEVELS:
        fields, lons = prepare(level, rows, cols, lons_all, weights)
        record["results"][str(level)] = {band: analyze(fields[band], lons) for band in P.BANDS}
    record["figure"] = draw(record["results"], args.png)
    record["elapsed_seconds"] = round(time.time() - t0, 1)
    with open(args.out, "x") as fh:
        json.dump(record, fh)
    print(f"done in {record['elapsed_seconds']} s, record {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
