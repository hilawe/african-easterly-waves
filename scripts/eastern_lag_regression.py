#!/usr/bin/env python3
"""The lag-regression diagnostic (`EASTERN_LAG_REGRESSION_2026-10-21.md`): the 700 hPa band
means on 10 W to 75 E, band-passed to 2 to 10 days at 6-hourly resolution per season, and
for base longitudes across the band the correlation of the base series with the filtered
series at every longitude and at lags of up to 4 days, pooled over the 24 retained seasons.
The ridge of maximum correlation is followed from the base as the lag grows, and its slope is
the speed at which the filtered signal moves. A westward slope means the 2 to 10 day signal
travels west, a flat ridge means it stays where it is, and nothing here identifies a feature
or attributes a single history.

    python3 scripts/eastern_lag_regression.py constructions --out-dir DIR
    python3 scripts/eastern_lag_regression.py archive --xband DIR --constructions PATH --out-dir DIR
"""

import argparse
import glob
import gzip
import hashlib
import json
import math
import os
import sys
from fractions import Fraction

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
import eastern_anomaly_minima as A  # noqa: E402
import eastern_crossing_test as X  # noqa: E402
import eastern_minima_histories as HI  # noqa: E402
import eastern_standing_climatology as SC  # noqa: E402
from aew.filtering import lanczos_bandpass  # noqa: E402

YEARS = SC.YEARS
MONTHS = SC.MONTHS
LONS = X.XLONS
OBS_PER_DAY = 4
PERIODS = (2.0, 10.0)
BUFFER_DAYS = 15
MAX_LAG_DAYS = 4
LAGS = np.arange(-MAX_LAG_DAYS * OBS_PER_DAY, MAX_LAG_DAYS * OBS_PER_DAY + 1)
BASES = (0.0, 10.0, 20.0, 30.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0)
R_MIN = 0.3
SEARCH_DEG = 4.0
EDGE_DEG = 1.0
BAND_LAT = {"10 to 14.5 N": 12.25, "15 to 19.5 N": 17.25, "20 to 24.5 N": 22.25, "25 to 29.5 N": 27.25}


def sha256_of(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


SCRIPT_SHA256_AT_LOAD = sha256_of(os.path.abspath(__file__))


# ------------------------------------------------------------------ the filtered seasons

def season_samples(recs_by_month, band):
    """The season's 6-hourly band means, June to September in order, as a (samples, longitudes)
    array, refused unless the season holds the expected number of samples, every one finite."""
    S = np.concatenate([np.array(recs_by_month[m]["band_means_700"][band], float) for m in MONTHS], axis=0)
    if S.shape[0] != A.SEASON_DAYS * OBS_PER_DAY:
        raise SystemExit(f"REFUSED: {S.shape[0]} samples in a season, not {A.SEASON_DAYS * OBS_PER_DAY}")
    if not np.isfinite(S).all():
        raise SystemExit("REFUSED: a season holds a missing sample, and the declared input is complete")
    return S


def check_inputs(records):
    """Each extended record is on the declared grid and band set, holds the six-hourly times of
    its own month in order, and has a complete finite array of the right shape in every band
    (added on the review, section 8)."""
    import calendar
    for r in records:
        y, m = int(r["year"]), int(r["month"])
        if not np.array_equal(np.asarray(r["lons"], float), LONS):
            raise SystemExit(f"REFUSED: {y}-{m:02d} is not on the declared grid of 10 W to 75 E at half a degree")
        if list(r["bands"]) != list(BAND_LAT):
            raise SystemExit(f"REFUSED: {y}-{m:02d} does not hold the declared bands")
        times = [f"{y}-{m:02d}-{d:02d}T{h:02d}" for d in range(1, calendar.monthrange(y, m)[1] + 1) for h in (0, 6, 12, 18)]
        if list(r["times"]) != times:
            raise SystemExit(f"REFUSED: {y}-{m:02d} does not hold its month's six-hourly times in order")
        for b in BAND_LAT:
            a = np.asarray(r["band_means_700"][b], float)
            if a.shape != (len(times), len(LONS)) or not np.isfinite(a).all():
                raise SystemExit(f"REFUSED: {y}-{m:02d} {b} is not a complete array of its month's samples")


def filtered(S):
    """The 2 to 10 day Lanczos band-pass along time, NaN on the BUFFER_DAYS at each end."""
    return lanczos_bandpass(S, PERIODS[0], PERIODS[1], obs_per_day=OBS_PER_DAY, buffer_days=BUFFER_DAYS, axis=0)


def valid_block(F):
    """The first and one-past-last rows of the season's finite block, refused unless the finite
    rows are one block with every longitude finite on them."""
    idx = np.flatnonzero(np.isfinite(F).all(axis=1))
    if idx.size == 0 or idx[-1] - idx[0] + 1 != idx.size or np.isfinite(F[idx[0]:idx[-1] + 1]).sum() != (idx.size * F.shape[1]):
        raise SystemExit("REFUSED: a season's filtered rows are not one finite block")
    return int(idx[0]), int(idx[-1]) + 1


def lag_correlation(seasons, base_col, shift=0):
    """The Pearson correlation, pooled over seasons, of the base column of season i with every
    column of season i + shift (cyclic) at each lag in LAGS, a positive lag meaning the field
    later than the base, over the times both lie in their seasons' finite blocks. Returns a
    (lags, longitudes) array."""
    n_s = len(seasons)
    blocks = [valid_block(F) for F in seasons]
    out = np.full((len(LAGS), seasons[0].shape[1]), np.nan)
    for li, L in enumerate(LAGS):
        L = int(L)
        n = 0.0; sa = 0.0; saa = 0.0
        sb = np.zeros(seasons[0].shape[1]); sab = np.zeros_like(sb); sbb = np.zeros_like(sb)
        for i in range(n_s):
            j = (i + shift) % n_s
            (e0, e1), (f0, f1) = blocks[i], blocks[j]
            t0, t1 = max(e0, f0 - L), min(e1, f1 - L)
            if t1 <= t0:
                continue
            a = seasons[i][t0:t1, base_col]
            B = seasons[j][t0 + L:t1 + L]
            n += t1 - t0; sa += a.sum(); saa += a @ a
            sb += B.sum(axis=0); sab += a @ B; sbb += (B * B).sum(axis=0)
        if n <= 2:
            continue
        cov = sab - sa * sb / n
        va = saa - sa * sa / n
        vb = sbb - sb * sb / n
        with np.errstate(invalid="ignore", divide="ignore"):
            out[li] = np.where((va > 0) & (vb > 0), cov / np.sqrt(va * vb), np.nan)
    return out


# ------------------------------------------------------------------ the ridge

def track_ridge(R, base_col, lons=LONS, r_min=R_MIN):
    """The ridge of maximum correlation followed from the base at lag 0 in both directions of
    lag: at each next lag, the column of largest correlation within SEARCH_DEG of the previous
    peak, kept while that correlation is at least R_MIN and the peak lies more than EDGE_DEG
    inside the grid, at R_MIN unless `r_min` is given for the post hoc sensitivity of section 7.
    Returns the tracked lags (in samples), peak longitudes and correlations,
    the slope of peak longitude against lag time by least squares (degrees per day), and the
    westmost and eastmost peaks."""
    zero = int(np.flatnonzero(LAGS == 0)[0])
    peaks = {0: (float(lons[base_col]), float(R[zero, base_col]))}
    for direction in (1, -1):
        col = base_col
        li = zero + direction
        while 0 <= li < len(LAGS):
            win = np.flatnonzero(np.abs(lons - lons[col]) <= SEARCH_DEG + 1e-9)
            vals = R[li, win]
            if not np.isfinite(vals).any():
                break
            j = win[int(np.nanargmax(vals))]
            r = float(R[li, j])
            if r < r_min or lons[j] <= lons[0] + EDGE_DEG or lons[j] >= lons[-1] - EDGE_DEG:
                break
            peaks[int(LAGS[li])] = (float(lons[j]), r)
            col = j
            li += direction
    lags = sorted(peaks)
    x = [peaks[k][0] for k in lags]
    exact = exact_slope(lags, x) if len(lags) >= 3 else None
    return {"lags": lags, "peak_lon": x, "peak_r": [peaks[k][1] for k in lags], "slope_deg_per_day": None if exact is None else float(exact),
            "slope_exact": None if exact is None else [exact.numerator, exact.denominator],
            "westmost": float(min(x)), "eastmost": float(max(x)), "positive_lag_days": max(lags) / OBS_PER_DAY, "negative_lag_days": -min(lags) / OBS_PER_DAY}


def exact_slope(lags, lons):
    """The least-squares slope of peak longitude against lag in days, exact on the integer lags
    and half-degree longitudes, so a ridge on an inclusive boundary is classed the same at every
    base (added on the review, section 8). With these lags and longitudes an exact slope that is
    not on a boundary differs from it by far more than a float's precision, so the float of the
    exact slope decides the classes."""
    t = [Fraction(int(k), OBS_PER_DAY) for k in lags]
    x = [Fraction(v).limit_denominator(2) for v in lons]
    tm, xm = sum(t) / len(t), sum(x) / len(x)
    return sum((a - tm) * (b - xm) for a, b in zip(t, x)) / sum((a - tm) ** 2 for a in t)


SENSITIVITY_R = (0.25, 0.2)


def base_autocorrelation(R, base_col):
    """The base's own filtered autocorrelation at positive lags: the lag in days and the value of
    its first local minimum (a reader added after the archive was read, section 7)."""
    zero = int(np.flatnonzero(LAGS == 0)[0])
    r = R[zero:, base_col]
    for k in range(1, len(r) - 1):
        if r[k] < r[k - 1] and r[k] <= r[k + 1]:
            return {"first_min_lag_days": k / OBS_PER_DAY, "first_min_r": float(r[k])}
    return {"first_min_lag_days": None, "first_min_r": None}


def speed_ms(slope_deg_per_day, lat):
    return None if slope_deg_per_day is None else slope_deg_per_day * 111195.0 * math.cos(math.radians(lat)) / 86400.0


# ------------------------------------------------------------------ constructions

CONSTRUCTION_NOISE_STD = 3.03
PHASE_SEED = 20261021
CASES = ("R1", "R2", "R3", "R4", "R5", "R6")
CASE_NOTES = {
    "R1": "a feature of depth 10 at 50 E wobbling 3 degrees with an 8-day period, a phase per season",
    "R2": "a westward wave train of amplitude 3 m/s, wavelength 30 degrees, moving 7 degrees per day over the whole domain",
    "R3": "R1 and R2 together",
    "R4": "two features at 47 and 54 E taking turns as the deeper (10 and 7 m/s) every 4 days",
    "R5": "R2 confined east of 45 E, its amplitude tapering to zero between 50 and 45 E",
    "R6": "noise alone",
}
WAVE = {"amplitude": 3.0, "wavelength": 30.0, "speed": 7.0}
CONSTRUCTION_NULL_BASES = (10.0, 50.0, 60.0)


def construction_samples(case, i, lons=LONS):
    """One season of six-hourly samples of a construction, rounded to two decimals."""
    t = (np.arange(A.SEASON_DAYS * OBS_PER_DAY) / OBS_PER_DAY)[:, None]
    x = lons[None, :]
    rng = np.random.default_rng(PHASE_SEED + i)
    ph = float(rng.uniform(0.0, 2.0 * np.pi))
    V = np.zeros((t.shape[0], len(lons)))
    if case in ("R1", "R3"):
        centers = 50.0 + 3.0 * np.sin(2.0 * np.pi * t[:, 0] / 8.0 + ph)
        V += np.array([A.gauss(lons, c, 10.0) for c in centers])
    if case in ("R2", "R3", "R5"):
        wave = WAVE["amplitude"] * np.sin(2.0 * np.pi * (x + WAVE["speed"] * t) / WAVE["wavelength"] + ph)
        if case == "R5":
            wave = wave * np.clip((x - 45.0) / 5.0, 0.0, 1.0)
        V += wave
    if case == "R4":
        deep_a = (np.floor(t[:, 0] / 4.0) % 2 == 0)
        V += np.array([A.gauss(lons, 47.0, 10.0 if d else 7.0) + A.gauss(lons, 54.0, 7.0 if d else 10.0) for d in deep_a])
    if case not in CASES:
        raise SystemExit(f"REFUSED: unknown construction {case}")
    V += A.noise_field(CONSTRUCTION_NOISE_STD, seed=HI.NOISE_SEEDS[i], n_steps=V.shape[0], lons=lons)
    return np.round(V, 2)


def summarize(seasons_f, bases=BASES, lat=17.25, null_bases=BASES):
    """For each base, the tracked ridge, its speed in m/s at `lat`, and, for the bases in
    `null_bases`, the null of the season shifts, the largest correlation at any longitude and
    any lag for each of the shifts, given as its median and maximum."""
    out = {}
    for b in bases:
        col = int(np.flatnonzero(np.isclose(LONS, b))[0])
        R = lag_correlation(seasons_f, col)
        tr = track_ridge(R, col)
        tr["speed_ms"] = speed_ms(tr["slope_deg_per_day"], lat)
        tr["sensitivity"] = {f"{r:g}": track_ridge(R, col, r_min=r) for r in SENSITIVITY_R}
        tr["base_autocorrelation"] = base_autocorrelation(R, col)
        if b in null_bases:
            mx = []
            for k in range(1, len(seasons_f)):
                Rk = lag_correlation(seasons_f, col, shift=k)
                mx.append(float(np.nanmax(Rk)))
            tr["shift_max_r"] = {"median": float(np.median(mx)), "max": float(np.max(mx))}
        out[f"{b:g}"] = tr
    return out


# ------------------------------------------------------------------ classes and the rule

SLOPE_MOVING = 2.0
SLOPE_STANDING = 1.0
MIN_REACH = 5.0
MIN_LAG_DAYS = 1.0
S2_WEST = 44.5
S2_EAST_BASES = (60.0, 65.0, 70.0)


def classify(tr, base):
    """'westward' when the slope is at most -2 degrees per day, the ridge is tracked over at least
    a day of positive lag, and its westmost peak lies at least 5 degrees west of the base;
    'eastward' symmetrically; 'standing' when the slope is under 1 degree per day in size or the
    ridge is too short to fit; 'other' otherwise."""
    s = tr["slope_deg_per_day"]
    if s is None or abs(s) < SLOPE_STANDING:
        return "standing"
    if s <= -SLOPE_MOVING and tr["positive_lag_days"] >= MIN_LAG_DAYS and base - tr["westmost"] >= MIN_REACH:
        return "westward"
    if s >= SLOPE_MOVING and tr["positive_lag_days"] >= MIN_LAG_DAYS and tr["eastmost"] - base >= MIN_REACH:
        return "eastward"
    return "other"


def annotate(res):
    """Each base's class and, for a westward ridge, whether it reaches S2's western edge (44.5 E)
    and 40 E."""
    for b, tr in res.items():
        base = float(b)
        for t in [tr] + list(tr.get("sensitivity", {}).values()):
            t["class"] = classify(t, base)
            w = t["class"] == "westward"
            t["reaches_s2_west"] = bool(w and base > S2_WEST and t["westmost"] <= S2_WEST)
            t["crosses_40"] = bool(w and base > 40.0 and t["westmost"] <= 40.0)
    return res


def score_case(case):
    seasons = [filtered(construction_samples(case, i)) for i in range(len(YEARS))]
    return annotate(summarize(seasons, null_bases=CONSTRUCTION_NULL_BASES if case == "R6" else ()))


def run_constructions():
    return {c: score_case(c) for c in CASES}


def apply_rule(results):
    """Section 4: no westward or eastward base in R1, R4 and R6; every base westward in R2 with its
    slope within 25 percent of the constructed -7 degrees per day; the bases at 60, 65 and 70 E
    westward and reaching S2's western edge in R3; and in R5, the bases at 55 E and east of it
    westward and none reaching S2's western edge."""
    fails = []
    for c in ("R1", "R4", "R6"):
        moving = [b for b, v in results[c].items() if v["class"] in ("westward", "eastward")]
        if moving:
            fails.append(f"{c} moving at {moving}")
    for b, v in results["R2"].items():
        s = v["slope_deg_per_day"]
        if v["class"] != "westward" or s is None or abs(s + WAVE["speed"]) > 0.25 * WAVE["speed"]:
            fails.append(f"R2 base {b} not westward at -7 within 25 percent ({s})")
    for b in S2_EAST_BASES:
        v = results["R3"][f"{b:g}"]
        if not v["reaches_s2_west"]:
            fails.append(f"R3 base {b:g} does not reach S2's western edge")
    for b, v in results["R5"].items():
        if float(b) >= 55.0 and v["class"] != "westward":
            fails.append(f"R5 base {b} not westward")
        if v["reaches_s2_west"]:
            fails.append(f"R5 base {b} reaches S2's western edge")
    return {"adequate": not fails, "failures": fails}


def check_constructions(path, rerun=None):
    """The archive runs only on constructions run by this exact script whose stored results equal
    a fresh run and pass the rule recomputed from them."""
    with open(path, encoding="utf-8") as fh:
        rec = json.load(fh)
    if rec["generated_by"]["sha256"] != SCRIPT_SHA256_AT_LOAD:
        raise SystemExit("REFUSED: the constructions were run by a different version of this script")
    fresh = json.loads(json.dumps({"results": (rerun or run_constructions)()}))["results"]
    if fresh != rec.get("results"):
        raise SystemExit("REFUSED: the constructions record differs from a fresh run of the same constructions")
    verdict = apply_rule(rec["results"])
    if not verdict["adequate"]:
        raise SystemExit(f"REFUSED: the constructions do not pass the rule of section 4 ({verdict['failures']})")
    if verdict != rec.get("verdict"):
        raise SystemExit("REFUSED: the stored summary disagrees with the rule applied to the stored cases")
    return rec


# ------------------------------------------------------------------ the archive

READINGS = {"passes": "a westward ridge followed from 60, 65 or 70 E reaches S2's western edge at 44.5 E within 4 days at correlation 0.3",
            "stops": "no westward ridge followed from 60, 65 or 70 E reaches S2's western edge at 44.5 E within 4 days at correlation 0.3"}


def band_reading(res):
    """Section 2.5's fixed reading for one band, from the bases at 60, 65 and 70 E."""
    return READINGS["passes"] if any(res[f"{b:g}"]["reaches_s2_west"] for b in S2_EAST_BASES) else READINGS["stops"]


def archive_results(by):
    out = {}
    for band, lat in BAND_LAT.items():
        seasons = [filtered(season_samples(by[y], band)) for y in YEARS]
        res = annotate(summarize(seasons, lat=lat))
        out[band] = {"bases": res, "reading": band_reading(res)}
    return out


def draw(path, by, bases=(10.0, 50.0, 60.0)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(BAND_LAT), len(bases), figsize=(4.2 * len(bases), 2.9 * len(BAND_LAT)), sharex=True, sharey=True, constrained_layout=True)
    lagd = LAGS / OBS_PER_DAY
    for r_i, band in enumerate(BAND_LAT):
        seasons = [filtered(season_samples(by[y], band)) for y in YEARS]
        for c_i, b in enumerate(bases):
            col = int(np.flatnonzero(np.isclose(LONS, b))[0])
            R = lag_correlation(seasons, col)
            ax = axes[r_i, c_i]
            cs = ax.contourf(LONS, lagd, R, levels=np.arange(-0.8, 0.85, 0.1), cmap="RdBu_r", extend="both")
            ax.contour(LONS, lagd, R, levels=[R_MIN], colors="k", linewidths=0.6)
            tr = track_ridge(R, col)
            ax.plot(tr["peak_lon"], np.array(tr["lags"]) / OBS_PER_DAY, "k.", ms=2)
            ax.axvline(S2_WEST, color="0.4", lw=0.6, ls="--"); ax.axvline(40.0, color="0.4", lw=0.6, ls=":")
            ax.set_title(f"{band}, base {b:g} E", fontsize=8)
            if c_i == 0:
                ax.set_ylabel("lag (days)")
            if r_i == len(BAND_LAT) - 1:
                ax.set_xlabel("longitude (deg E)")
    fig.colorbar(cs, ax=axes, shrink=0.6, label="correlation with the base, 2 to 10 day band-pass")
    fig.suptitle("Lag correlation of the 700 hPa band-mean meridional wind with a base longitude, 24 seasons. Dots: the tracked ridge. Dashed: 44.5 E. Dotted: 40 E.", fontsize=8)
    fig.savefig(path, dpi=90); plt.close(fig)


# ------------------------------------------------------------------ text and main

HEAD = ("| case or band | base (deg E) | class | slope (deg per day) | speed (m/s) | westmost ridge peak (deg E) | positive lag tracked (days) | reaches 44.5 E | crosses 40 E | shifted null, median and max r |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")


def rows(name, res):
    out = []
    for b, v in res.items():
        s, sp = v["slope_deg_per_day"], v.get("speed_ms")
        null = v.get("shift_max_r")
        out.append(f"| {name} | {b} | {v['class']} | {'' if s is None else format(s, '.2f')} | {'' if sp is None else format(sp, '.1f')} | {v['westmost']:.1f} | "
                   f"{v['positive_lag_days']:.2f} | {'yes' if v['reaches_s2_west'] else 'no'} | {'yes' if v['crosses_40'] else 'no'} | "
                   f"{'' if null is None else format(null['median'], '.3f') + ', ' + format(null['max'], '.3f')} |")
    return out


def constructions_text(results, verdict):
    L = ["# lag regression, the constructions (section 3)", "", HEAD]
    for c in CASES:
        L += rows(c, results[c])
    L += ["", f"adequate under the rule of section 4, {'yes' if verdict['adequate'] else 'no'}"] + [f"- {x}" for x in verdict["failures"]]
    L += ["", "## the ridge at lower thresholds and the base autocorrelation, readers added after the archive was read (section 7)", "", SENS_HEAD]
    for c in CASES:
        L += sensitivity_rows(c, results[c])
    return "\n".join(L) + "\n"


def sensitivity_rows(name, res):
    out = []
    for b, v in res.items():
        a = v["base_autocorrelation"]
        cells = []
        for r in SENSITIVITY_R:
            t = v["sensitivity"][f"{r:g}"]
            s = t["slope_deg_per_day"]
            cells.append(f"{t['class']}, {'' if s is None else format(s, '.2f')}, {t['westmost']:.1f}, {t['positive_lag_days']:.2f}, {'reaches' if t['reaches_s2_west'] else 'no'}")
        out.append(f"| {name} | {b} | " + " | ".join(cells) + f" | {'' if a['first_min_lag_days'] is None else format(a['first_min_lag_days'], '.2f')} | {'' if a['first_min_r'] is None else format(a['first_min_r'], '.2f')} |")
    return out


SENS_HEAD = ("| case or band | base (deg E) | at r 0.25, class, slope, westmost, positive lag, 44.5 E | at r 0.2, the same | base autocorrelation, first minimum lag (days) | its value |\n"
             "| --- | --- | --- | --- | --- | --- |")


def archive_text(res):
    L = ["# lag regression, the archive (section 5)", "", HEAD]
    for band, v in res.items():
        L += rows(band, v["bases"])
    L += [""] + [f"- {band}, reading under section 2.5, {v['reading']}" for band, v in res.items()]
    L += ["", "## the ridge at lower thresholds and the base autocorrelation, readers added after the archive was read (section 7)", "", SENS_HEAD]
    for band, v in res.items():
        L += sensitivity_rows(band, v["bases"])
    return "\n".join(L) + "\n"


def refuse_existing(paths):
    for p in paths:
        if os.path.exists(p):
            raise SystemExit(f"REFUSED: {p} exists and a record is never overwritten")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("constructions", "archive"))
    ap.add_argument("--out-dir", required=True)
    for opt in ("--xband", "--constructions"):
        ap.add_argument(opt, default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    gen = {"script": "scripts/eastern_lag_regression.py", "sha256": SCRIPT_SHA256_AT_LOAD, "declared": "EASTERN_LAG_REGRESSION_2026-10-21.md",
           "filtering_sha256": sha256_of(os.path.join(os.path.dirname(HERE), "src", "aew", "filtering.py"))}
    if a.mode == "constructions":
        pj, pm = (os.path.join(a.out_dir, f"eval60_lagreg_constructions{a.tag}.{e}") for e in ("json", "md"))
        refuse_existing([pj, pm])
        results = json.loads(json.dumps(run_constructions())); verdict = apply_rule(results)
        with open(pj, "w", encoding="utf-8") as fh:
            json.dump({"generated_by": gen, "results": results, "verdict": verdict}, fh)
        text = constructions_text(results, verdict)
        with open(pm, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(text)
        return 0
    if not (a.xband and a.constructions):
        raise SystemExit("REFUSED: --xband and --constructions are required for the archive")
    check_constructions(a.constructions)
    paths = {k: os.path.join(a.out_dir, f"eval60_lagreg_{k}{a.tag}.{e}") for k, e in (("archive", "json"), ("tables", "md"), ("lags", "png"))}
    refuse_existing(paths.values())
    files = sorted(glob.glob(os.path.join(a.xband, "eval60_xband_????_??.json.gz")))
    recs = []
    for p in files:
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            recs.append(json.load(fh))
    X.check_xband_identity(files, recs)
    check_inputs(recs)
    by = {}
    for r in recs:
        by.setdefault(int(r["year"]), {})[int(r["month"])] = r
    res = archive_results(by)
    draw(paths["lags"], by)
    gen.update({"xband": {os.path.basename(p): sha256_of(p) for p in files}, "constructions": {"path": a.constructions, "sha256": sha256_of(a.constructions)}})
    with open(paths["archive"], "w", encoding="utf-8") as fh:
        json.dump({"generated_by": gen, "results": res, "figure": {"path": paths["lags"], "sha256": sha256_of(paths["lags"])}}, fh)
    text = archive_text(res)
    with open(paths["tables"], "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
