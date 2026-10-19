#!/usr/bin/env python3
"""The QTrack look-back (`EASTERN_QTRACK_LOOKBACK_2026-10-19.md`, section 2): a
track-relative lag composite of the own-month (RB1) and nine-day (RB9) band-mean anomalies
around QTrack's tracks that begin east of 30 E, against the same composite in 23 season
shifts, with an instrument check on the start day and an approach reading over the three
days before. The constructions of section 3 run first, and the archive refuses to run
unless they passed under this exact script. A composite is an average of positions and
dates. Nothing is identified, connected or explained.

    python3 scripts/eastern_qtrack_lookback.py constructions --out-dir DIR [--tag r2]
    python3 scripts/eastern_qtrack_lookback.py archive --xband DIR --qtrack DIR \\
        --constructions PATH --out-dir DIR [--tag r1]
"""

import argparse
import datetime as dt
import glob
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eastern_regime_views as V  # noqa: E402
import eastern_anomaly_minima as A  # noqa: E402
import eastern_minima_histories as HI  # noqa: E402
import eastern_standing_climatology as SC  # noqa: E402
import eastern_crossing_test as X  # noqa: E402

YEARS = SC.YEARS
MONTHS = SC.MONTHS
LAGS = tuple(range(-5, 3))
DREL = np.arange(-15.0, 30.0 + 1e-9, 0.5)
REPS = {"RB1": "B1", "RB9": "B2_9"}
APPROACH_LAGS = (-3, -2, -1)
MIN_WEST_DEG = 4.5
R1_LAGS = (-5, -4, -3, -2, -1)        # the one repair (section 7 of the record), declared at 69b03d9 before it was scored
R1_MIN_WEST_DEG = 7.5
TIE_EPS = 1e-9
CONTRIB_THRESHOLD = -2.0
CONTRIB_HALF_WIDTH = 1.0
SPLIT_LON = 36.0
LOW_BAND = "10 to 14.5 N"
CONSTRUCTION_NOISE_STD = 2.48          # the spread-matched level of 10 to 14.5 N, eval60_hist_noise_r1.json
TRACK_SEED = 20261020
TRACKS_PER_SEASON = 4
START_DAYS = (30, 110)
TRANSIENT_DEPTH = 5.0
WOBBLE = {"at": 44.0, "amplitude": 3.0, "period": 14.0, "depth": 10.0, "track_lon": 38.0}
WOBBLE_AMPLITUDE = {"K5": 3.0, "K8": 5.0, "K9": 5.0}      # K8 added after the constructions were first read, K9 with the repair (section 7 of the record)
WOBBLE_PERIOD = {"K5": 14.0, "K8": 14.0, "K9": 20.0}
CASES = ("K1", "K2", "K3", "K4", "K5", "K6", "K7", "K8", "K9")
RULE = {"shown": ("K1", "K2"), "not_shown": ("K4", "K5", "K6"), "start_registered": ("K1", "K2", "K4")}
RULE_R1 = {"shown": ("K1", "K2"), "not_shown": ("K4", "K5", "K6", "K8"), "start_registered": ("K1", "K2", "K4")}


def sha256_of(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


SCRIPT_SHA256_AT_LOAD = sha256_of(os.path.abspath(__file__))


# ------------------------------------------------------------------ tracks

def band_of(lat):
    """The band whose center is nearest the latitude, the southern one on a tie."""
    centers = [(abs((lo + hi) / 2.0 - lat), k, b) for k, (b, (lo, hi)) in enumerate(V.BANDS.items())]
    return min(centers)[2]


def round_half_east(x):
    """The nearest half degree, ties to the east."""
    return float(np.floor(float(x) * 2.0 + 0.5) / 2.0)


def season_day(date):
    return (date - dt.date(date.year, 6, 1)).days


def track_table(tracks):
    """The eligible tracks as dicts with season, start day index, start longitude rounded to
    the half degree, the first longitude and latitude, and the nearest-center band."""
    out = []
    for k, v in sorted(X.eligible_tracks(tracks).items(), key=lambda kv: (kv[1][0], kv[0])):
        t0, la0, lo0 = v[0], v[1], v[2]
        out.append({"key": list(k), "year": t0.year, "day": season_day(t0.date()), "x0": round_half_east(lo0), "first_lon": float(lo0), "first_lat": float(la0), "band": band_of(la0)})
    return out


# ------------------------------------------------------------------ the fields

def anomaly_values(H, month_of, kind, gap=None):
    """The exact anomaly in m/s as floats, NaN where the background is unavailable or the
    longitude is a gap. Values differ from -2 by at least one over the scale when not equal
    to it, far above float resolution, so a comparison with -2 is exact."""
    rows, scales = A.exact_anomaly(kind, H, month_of)
    out = np.full(np.shape(H), np.nan)
    for i, (r, s) in enumerate(zip(rows, scales)):
        if r is not None:
            out[i] = np.asarray(r, dtype=np.int64) / s
    if gap is not None:
        out[:, np.asarray(gap, bool)] = np.nan
    return out


def season_fields(xrecords):
    """{(year, band): {rep: values}} for the retained seasons from the extended records."""
    by = {}
    for r in xrecords:
        by.setdefault(int(r["year"]), {})[int(r["month"])] = r
    month_of = A.month_index(); out = {}
    for y in YEARS:
        for b in xrecords[0]["bands"]:
            _, H, gap = X.season_hundredths_with_gaps(by[y], b)
            out[(y, b)] = {rep: anomaly_values(H, month_of, kind, gap) for rep, kind in REPS.items()}
    return out


# ------------------------------------------------------------------ the composite

def track_values(field, day, x0r, lons=X.XLONS):
    """The values at lag LAGS and relative longitude DREL around one start, NaN outside the
    season or the grid."""
    out = np.full((len(LAGS), len(DREL)), np.nan)
    idx = np.rint((x0r + DREL - lons[0]) / 0.5).astype(int)
    ok = (idx >= 0) & (idx < len(lons))
    for i, k in enumerate(LAGS):
        d = day + k
        if 0 <= d < field.shape[0]:
            out[i, ok] = field[d, idx[ok]]
    return out


def track_stack(fields, tracks, rep, shift=0, years=YEARS):
    """The tracks' value arrays, each read from the season `shift` places along the retained
    seasons, keeping month and day, band and start longitude."""
    return np.array([track_values(fields[(years[(years.index(t["year"]) + shift) % len(years)], t["band"])][rep], t["day"], t["x0"]) for t in tracks])


def composite(stack):
    n = np.isfinite(stack).sum(axis=0)
    with np.errstate(invalid="ignore"):
        C = np.where(n > 0, np.nansum(stack, axis=0) / np.maximum(n, 1), np.nan)
    return C, n


def lag_minima(C):
    """Per lag, the composite minimum over DREL and its westernmost location (None if empty)."""
    out = []
    for row in C:
        if not np.isfinite(row).any():
            out.append((None, None))
        else:
            j = int(np.nanargmin(row)); out.append((float(row[j]), float(DREL[j])))
    return out


def reading(C, shift_Cs):
    """Section 2.4: per lag the minimum, its location and its rank among the shifts' minima
    (a shift at or below when within TIE_EPS), the instrument check on the start day, the
    approach over lags -3 to -1, and the day-before-only pattern. The rank is unavailable,
    and so fails every condition, unless every shift has a minimum at that lag (review fold,
    section 12 of the record)."""
    obs = lag_minima(C); shm = [lag_minima(Cs) for Cs in shift_Cs]
    lags = {}
    for i, k in enumerate(LAGS):
        m, d = obs[i]
        ms = [s[i][0] for s in shm]
        valid = sum(1 for v in ms if v is not None)
        rank = None if (m is None or valid < len(ms)) else int(sum(1 for v in ms if v <= m + TIE_EPS))
        lags[k] = {"min": m, "D_star": d, "rank": rank, "shift_minima": ms, "valid_shifts": valid}
    r = lambda k: lags[k]["rank"]
    D = lambda k: lags[k]["D_star"]
    ranks_ok = all(r(k) == 0 for k in APPROACH_LAGS)
    have = all(D(k) is not None for k in (-3, -2, -1, 0))
    order_ok = have and D(-3) >= D(-2) >= D(-1) >= D(0)
    west_ok = have and (D(-3) - D(0)) >= MIN_WEST_DEG
    approach = bool(ranks_ok and order_ok and west_ok)
    day_before = bool(not approach and r(-1) == 0 and D(-1) is not None and D(0) is not None and D(-1) > D(0))
    ranks_r1 = all(r(k) == 0 for k in R1_LAGS)
    ds = [D(k) for k in R1_LAGS + (0,)]
    have_r1 = all(d is not None for d in ds)
    order_r1 = have_r1 and all(a >= b for a, b in zip(ds, ds[1:]))
    west_r1 = have_r1 and (ds[0] - ds[-1]) >= R1_MIN_WEST_DEG
    return {"lags": lags, "instrument": r(0) == 0, "approach": approach, "parts": {"ranks_zero_at_minus3_to_minus1": ranks_ok, "not_moving_east": bool(order_ok), "west_at_least_4p5": bool(west_ok)}, "day_before_only": day_before,
            "approach_r1": bool(ranks_r1 and order_r1 and west_r1), "parts_r1": {"ranks_zero_at_minus5_to_minus1": ranks_r1, "not_moving_east_from_minus5": bool(order_r1), "west_at_least_7p5": bool(west_r1)}}


def contribution_count(stack, d_star, k):
    """The tracks whose own value within CONTRIB_HALF_WIDTH of d_star at lag k is at or below
    CONTRIB_THRESHOLD."""
    if d_star is None:
        return None
    i = LAGS.index(k); cols = np.abs(DREL - d_star) <= CONTRIB_HALF_WIDTH + 1e-9
    v = stack[:, i][:, cols]
    with np.errstate(invalid="ignore"):
        low = np.where(np.isfinite(v).any(axis=1), np.nanmin(np.where(np.isfinite(v), v, np.inf), axis=1), np.inf)
    return int(np.sum(low <= CONTRIB_THRESHOLD))


def look_back(fields, tracks, rep, years=YEARS):
    """The composite, its 23 shifts, the reading, the pointwise rank, and the contribution
    counts (each shift counted at its own minimum's location, the same statistic)."""
    st = track_stack(fields, tracks, rep, years=years)
    C, n = composite(st)
    shifts = [track_stack(fields, tracks, rep, s, years=years) for s in range(1, len(years))]
    Cs = [composite(x)[0] for x in shifts]
    rd = reading(C, Cs)
    with np.errstate(invalid="ignore"):
        complete = np.isfinite(C) & np.all([np.isfinite(c) for c in Cs], axis=0)
        prank = np.where(complete, np.sum([np.where(complete, c <= C + TIE_EPS, False) for c in Cs], axis=0), np.nan)
    contrib = {}
    for k in APPROACH_LAGS:
        sh = [contribution_count(x, lag_minima(c)[LAGS.index(k)][1], k) for x, c in zip(shifts, Cs)]
        shv = [v for v in sh if v is not None]
        contrib[k] = {"observed": contribution_count(st, rd["lags"][k]["D_star"], k), "shifts": sh, "shift_mean": (float(np.mean(shv)) if shv else None), "shift_max": (int(max(shv)) if shv else None)}
    return {"composite": C, "n": n, "shift_composites": Cs, "reading": rd, "pointwise_rank": prank, "contributions": contrib, "tracks": len(tracks)}


def archive_verdicts(per_rep):
    """The archive's reading under the repaired rule and, beside it, under the declared one,
    both stopped when the start day is not registered under both representations."""
    if not all(per_rep[rep]["reading"]["instrument"] for rep in per_rep):
        stop = "stopped, the start day is not registered under both representations"
        return stop, stop
    return verdict_of(per_rep, "approach_r1"), verdict_of(per_rep)


def verdict_of(per_rep, key="approach"):
    """The reading under the declared rule (key "approach") or the repaired one ("approach_r1"),
    which no longer reports the day-before-only pattern (section 7 of the record)."""
    a = {rep: per_rep[rep]["reading"][key] for rep in per_rep}
    if all(a.values()):
        return "SHOWN"
    if any(a.values()):
        return "under " + " and ".join(k for k, v in a.items() if v) + " only"
    if key == "approach" and any(per_rep[rep]["reading"]["day_before_only"] for rep in per_rep):
        return "a signal on the day before only, under " + " and ".join(k for k in per_rep if per_rep[k]["reading"]["day_before_only"])
    return "not shown"


# ------------------------------------------------------------------ constructions (section 3)

def construction_days(i, case):
    """Four start days per synthetic season, drawn by a fixed seed in START_DAYS, or for K5
    the days of fastest westward wobble for the season's phase, with that phase."""
    rng = np.random.default_rng(TRACK_SEED + i)
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    if case in WOBBLE_AMPLITUDE:
        P = WOBBLE_PERIOD[case]
        t_fast = [((np.pi - phase) / (2.0 * np.pi)) * P + P * m for m in range(-2, 12)]
        days = sorted({int(round(t - 0.375)) for t in t_fast if START_DAYS[0] <= round(t - 0.375) <= START_DAYS[1]})
        days = sorted(rng.choice(days, size=min(TRACKS_PER_SEASON, len(days)), replace=False).tolist())
    else:
        days = sorted(rng.choice(np.arange(START_DAYS[0], START_DAYS[1] + 1), size=TRACKS_PER_SEASON, replace=False).tolist())
    return [int(d) for d in days], phase


def wobble_center(t, phase, amplitude=WOBBLE["amplitude"], period=WOBBLE["period"]):
    return WOBBLE["at"] + amplitude * np.sin(2.0 * np.pi * np.asarray(t) / period + phase)


def construction_transients(case, days):
    """Each transient as (path {time: lon}, depth), passing 35 E as a daily mean on its start
    day (K1 to K3, K7 for the first two tracks), or appearing there that day (K4)."""
    speed = {"K1": 2.0, "K2": 5.0, "K3": 1.5, "K7": 2.0}
    out = []
    for j, d in enumerate(days):
        if case in speed and (case != "K7" or j < 2):
            s = speed[case]; lead = int(np.ceil(40.0 / s))
            out.append((X.path_positions(35.0 + s * (lead + 0.375), float(d - lead), (lambda v: (lambda x: v))(s)), TRANSIENT_DEPTH))
        elif case == "K4":
            out.append((X.path_positions(35.0, float(d), lambda x: 4.0), TRANSIENT_DEPTH))
    return out


def construction_season(case, i, lons=X.XLONS):
    """One synthetic season: the fixed channel trough and southerly region, seeded noise at
    CONSTRUCTION_NOISE_STD, the case's transients or wobble, rounded to hundredths. Returns the
    daily sums, the tracks (start day, start longitude) and each transient's daily-mean center."""
    days, phase = construction_days(i, case)
    t = np.arange(A.SEASON_DAYS * 4) / 4.0
    F = A.gauss(lons, X.CHANNEL_AT, A.STANDING_DEPTH)[None, :] + A.gauss(lons, X.SOUTHERLY_AT, -6.0, width=3.0)[None, :]
    F = F + A.noise_field(CONSTRUCTION_NOISE_STD, seed=HI.NOISE_SEEDS[i], lons=lons)
    centers = []
    for path, depth in construction_transients(case, days):
        pos = np.full(len(t), np.nan)
        for kk, tk in enumerate(t):
            if tk in path:
                pos[kk] = path[tk]; F[kk] = F[kk] + A.gauss(lons, path[tk], depth)
        centers.append(pos.reshape(A.SEASON_DAYS, 4).mean(axis=1))
    if case in WOBBLE_AMPLITUDE:
        wc = wobble_center(t, phase, WOBBLE_AMPLITUDE[case], WOBBLE_PERIOD[case])
        F = F + np.array([A.gauss(lons, c, WOBBLE["depth"]) for c in wc])
    x0 = WOBBLE["track_lon"] if case in WOBBLE_AMPLITUDE else 35.0
    return A.hundredths_from_samples(np.round(F, 2)), [{"year": YEARS[i], "day": d, "x0": x0, "band": LOW_BAND} for d in days], centers


def run_constructions():
    month_of = A.month_index(); out = {}
    for case in CASES:
        fields, tracks = {}, []
        for i, y in enumerate(YEARS):
            H, tr, _ = construction_season(case, i)
            fields[(y, LOW_BAND)] = {rep: anomaly_values(H, month_of, kind) for rep, kind in REPS.items()}
            tracks += tr
        res = {rep: look_back(fields, tracks, rep) for rep in REPS}
        out[case] = {rep: {"reading": r["reading"], "contributions": r["contributions"], "tracks": r["tracks"]} for rep, r in res.items()}
        out[case]["verdict"] = verdict_of(res)
        out[case]["verdict_r1"] = verdict_of(res, "approach_r1")
    return out


def apply_rule(results, repaired=False):
    """Section 3: under each representation, K1 and K2 approach with the start registered, K4
    registers its start and does not approach, and K5 and K6 do not approach. The repaired
    rule of section 7 reads the five-day approach and adds K8 to the cases that must not."""
    fails = []; rule = RULE_R1 if repaired else RULE; key = "approach_r1" if repaired else "approach"
    for rep in REPS:
        for c in rule["shown"]:
            if not results[c][rep]["reading"][key]:
                fails.append(f"{c} under {rep} does not approach")
        for c in rule["not_shown"]:
            if results[c][rep]["reading"][key]:
                fails.append(f"{c} under {rep} approaches")
        for c in rule["start_registered"]:
            if not results[c][rep]["reading"]["instrument"]:
                fails.append(f"{c} under {rep} does not register its start")
    return {"adequate": not fails, "failures": fails}


def reading_rows(label, rep, rd, contrib=None):
    L = rd["lags"]
    f = lambda x: "" if x is None else f"{x:.2f}"
    g = lambda x: "" if x is None else f"{x:g}"
    cells = [", ".join(f"{g(L[k]['D_star'])} ({L[k]['rank']})" for k in (-5, -4))]
    cells += [f"{f(L[k]['min'])} at {g(L[k]['D_star'])} (rank {L[k]['rank']})" for k in (-3, -2, -1, 0)]
    cells += ["yes" if rd["instrument"] else "no", "yes" if rd["approach"] else "no", "yes" if rd["day_before_only"] else "no", "yes" if rd["approach_r1"] else "no"]
    if contrib is not None:
        for k in APPROACH_LAGS:
            c = contrib[k]
            cells.append(f"{c['observed']} ({f(c['shift_mean'])}, {c['shift_max']})")
    return f"| {label} | {rep} | " + " | ".join(cells) + " |"


HEADER = "| case | representation | D at lags -5 and -4 (rank), added | lag -3, minimum at D (rank) | lag -2 | lag -1 | lag 0 | start registered | approach | day before only | approach, repaired five-day |"


def constructions_text(results, verdict, verdict_r1=None):
    L = ["# QTrack look-back, the constructions (section 3)", "",
         "minimum of the composite in m/s at its relative longitude D in degrees, rank = shifts of 23 at or below it, K8 and the lags -5 and -4 added after the constructions were first read, K9 and the repaired five-day reading with the repair (section 7)", "", HEADER, "| --- |" + " --- |" * 10]
    for c in CASES:
        for rep in REPS:
            L.append(reading_rows(c, rep, results[c][rep]["reading"]))
    L += ["", "| case | verdict, declared | verdict, repaired |", "| --- | --- | --- |"] + [f"| {c} | {results[c]['verdict']} | {results[c]['verdict_r1']} |" for c in CASES]
    L += ["", f"adequate under the rule of section 3, {'yes' if verdict['adequate'] else 'no'}"] + [f"- {x}" for x in verdict["failures"]]
    if verdict_r1 is not None:
        L += [f"adequate under the repaired rule of section 7, {'yes' if verdict_r1['adequate'] else 'no'}"] + [f"- {x}" for x in verdict_r1["failures"]]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ the archive

def subsets(tracks):
    return {"first position west of 36 E": [t for t in tracks if t["first_lon"] < SPLIT_LON], "first position at or east of 36 E": [t for t in tracks if t["first_lon"] >= SPLIT_LON],
            "band 10 to 14.5 N": [t for t in tracks if t["band"] == LOW_BAND], "other bands": [t for t in tracks if t["band"] != LOW_BAND]}


def archive_results(fields, tracks):
    full = {rep: look_back(fields, tracks, rep) for rep in REPS}
    subs = {}
    for name, tr in subsets(tracks).items():
        subs[name] = {rep: {"tracks": len(tr), "reading": look_back(fields, tr, rep)["reading"]} for rep in REPS} if tr else None
    return full, subs


def archive_text(full, subs, tracks, verdict):
    L = ["# QTrack look-back, the archive (section 2)", "",
         f"{len(tracks)} eligible QTrack tracks, reading {verdict}, with the minimum of the composite in m/s at its relative longitude D in degrees, rank = shifts of 23 at or below it", "",
         HEADER + " tracks at or below -2 m/s within 1 degree, lag -3 (shift mean, max) | lag -2 | lag -1 |", "| --- |" + " --- |" * 13]
    for rep in REPS:
        L.append(reading_rows("all", rep, full[rep]["reading"], full[rep]["contributions"]))
    L += ["", "## tracks with a value, by lag (the composite's count at D = 0)", "", "| representation | " + " | ".join(f"lag {k}" for k in LAGS) + " |", "| --- |" + " --- |" * len(LAGS)]
    j0 = int(np.flatnonzero(np.isclose(DREL, 0.0))[0])
    for rep in REPS:
        L.append(f"| {rep} | " + " | ".join(str(int(full[rep]["n"][i, j0])) for i in range(len(LAGS))) + " |")
    L += ["", "## subsets, descriptive, no criterion", "", HEADER.replace("| case |", "| subset (tracks) |"), "| --- |" + " --- |" * 10]
    for name, d in subs.items():
        for rep in REPS:
            if d is not None:
                L.append(reading_rows(f"{name} ({d[rep]['tracks']})", rep, d[rep]["reading"]))
    return "\n".join(L) + "\n"


def draw(path, full):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, len(REPS), figsize=(6.0 * len(REPS), 7.6), sharex=True, sharey=True, constrained_layout=True)
    for j, rep in enumerate(REPS):
        C = full[rep]["composite"]; pr = full[rep]["pointwise_rank"]
        lim = float(np.nanmax(np.abs(C))) if np.isfinite(C).any() else 1.0
        a = axes[0, j].pcolormesh(DREL, LAGS, C, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="nearest")
        fig.colorbar(a, ax=axes[0, j], label="composite anomaly (m/s)")
        b = axes[1, j].pcolormesh(DREL, LAGS, pr, cmap="viridis_r", vmin=0, vmax=23, shading="nearest")
        fig.colorbar(b, ax=axes[1, j], label="shifts of 23 at or below")
        for ax in axes[:, j]:
            ds = [full[rep]["reading"]["lags"][k]["D_star"] for k in LAGS]
            ax.plot([d for d in ds], LAGS, "k^", ms=5)
            ax.axvline(0.0, color="k", lw=0.7, ls="--"); ax.set_title(rep, fontsize=10)
        axes[1, j].set_xlabel("longitude relative to the start (degrees)")
    for i in range(2):
        axes[i, 0].set_ylabel("lag (days from the start day)")
    fig.suptitle("QTrack look-back, composite band-mean anomaly around the eastern starts (top) and its pointwise rank among 23 season shifts (bottom). Triangles mark each lag's composite minimum. Positions and dates only.", fontsize=9, wrap=True)
    fig.savefig(path, dpi=100); plt.close(fig)


def jsonable(full):
    out = {}
    for rep, r in full.items():
        out[rep] = {"tracks": r["tracks"], "reading": r["reading"], "contributions": r["contributions"], "composite": np.where(np.isfinite(r["composite"]), np.round(r["composite"], 6), None).tolist(),
                    "n": r["n"].tolist(), "pointwise_rank": np.where(np.isfinite(r["pointwise_rank"]), r["pointwise_rank"], None).tolist(),
                    "shift_composites": [np.where(np.isfinite(c), np.round(c, 6), None).tolist() for c in r["shift_composites"]]}
    return out


# ------------------------------------------------------------------ main

def refuse_existing(paths):
    for p in paths:
        if os.path.exists(p):
            raise SystemExit(f"REFUSED: {p} exists and a record is never overwritten")


def check_constructions(path, rerun=None):
    """The archive runs only after constructions that passed, run by this exact script. The
    record's case results must equal a fresh run of the same constructions, which binds every
    imported calculation, and the repaired rule is applied to those results rather than read
    from the stored summary (review fold, section 12 of the record)."""
    with open(path, encoding="utf-8") as fh:
        rec = json.load(fh)
    if rec["generated_by"]["sha256"] != SCRIPT_SHA256_AT_LOAD:
        raise SystemExit("REFUSED: the constructions were run by a different version of this script")
    fresh = json.loads(json.dumps({"results": (rerun or run_constructions)()}, default=float))["results"]
    if fresh != rec.get("results"):
        raise SystemExit("REFUSED: the constructions record differs from a fresh run of the same constructions")
    verdict = apply_rule(rec["results"], repaired=True)
    if not verdict["adequate"]:
        raise SystemExit(f"REFUSED: the constructions do not pass the repaired rule of section 7 ({verdict['failures']})")
    if verdict != rec.get("verdict_r1"):
        raise SystemExit("REFUSED: the stored summary disagrees with the repaired rule applied to the stored cases")
    return rec


def qtrack_paths(qtrack_dir):
    """QTrack's files of the retained seasons, refused unless there is one per season. The
    paper's catalog is not read (review fold, section 12 of the record)."""
    paths = sorted(p for p in glob.glob(os.path.join(qtrack_dir, "ERA5_AEW_tracks_atlantic_????.nc")) if X.qtrack_year(p) in YEARS)
    X.check_western_coverage(paths, X.qtrack_year, "QTrack")
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("constructions", "archive"))
    ap.add_argument("--out-dir", required=True)
    for opt in ("--xband", "--qtrack", "--constructions"):
        ap.add_argument(opt, default=None)
    ap.add_argument("--tag", default=None, help="constructions are written as eval60_lookback_constructions_<tag> (default r1), an archive under <name>_<tag> (default no suffix)")
    a = ap.parse_args(argv)
    gen = {"script": "scripts/eastern_qtrack_lookback.py", "sha256": SCRIPT_SHA256_AT_LOAD, "declared": "EASTERN_QTRACK_LOOKBACK_2026-10-19.md",
           "anomaly_script_sha256": sha256_of(os.path.join(HERE, "eastern_anomaly_minima.py")), "crossing_script_sha256": sha256_of(os.path.join(HERE, "eastern_crossing_test.py"))}
    if a.mode == "constructions":
        tag = a.tag or "r1"
        pj, pm = os.path.join(a.out_dir, f"eval60_lookback_constructions_{tag}.json"), os.path.join(a.out_dir, f"eval60_lookback_constructions_{tag}.md")
        refuse_existing([pj, pm])
        results = run_constructions(); verdict = apply_rule(results); verdict_r1 = apply_rule(results, repaired=True)
        with open(pj, "w", encoding="utf-8") as fh:
            json.dump({"generated_by": gen, "results": results, "verdict": verdict, "verdict_r1": verdict_r1}, fh, default=float)
        text = constructions_text(results, verdict, verdict_r1)
        with open(pm, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(text)
        return 0
    if not all((a.xband, a.qtrack, a.constructions)):
        raise SystemExit("REFUSED: --xband, --qtrack and --constructions are required for the archive")
    check_constructions(a.constructions)
    sfx = f"_{a.tag}" if a.tag else ""
    paths = {k: os.path.join(a.out_dir, v) for k, v in (("json", f"eval60_lookback_archive{sfx}.json"), ("tables", f"eval60_lookback_tables{sfx}.md"), ("figure", f"eval60_lookback_composite{sfx}.png"))}
    refuse_existing(paths.values())
    xfiles = sorted(glob.glob(os.path.join(a.xband, "eval60_xband_????_??.json.gz")))
    xrecs = [SC.load_record(p) for p in xfiles]
    X.check_xband_identity(xfiles, xrecs)
    q_paths = qtrack_paths(a.qtrack)
    _, qtracks = X.qtrack_points(q_paths)
    tracks = track_table(qtracks)
    fields = season_fields(xrecs)
    full, subs = archive_results(fields, tracks)
    verdict, verdict_declared = archive_verdicts(full)
    draw(paths["figure"], full)
    gen.update({"xband": {os.path.basename(p): sha256_of(p) for p in xfiles}, "qtrack": {os.path.basename(p): sha256_of(p) for p in q_paths},
                "constructions": {"path": a.constructions, "sha256": sha256_of(a.constructions)}})
    with open(paths["json"], "w", encoding="utf-8") as fh:
        json.dump({"generated_by": gen, "verdict": verdict, "verdict_declared_rule": verdict_declared, "tracks": tracks, "results": jsonable(full), "subsets": subs,
                   "figure": {"path": paths["figure"], "sha256": sha256_of(paths["figure"])}}, fh, default=float)
    text = archive_text(full, subs, tracks, f"{verdict} under the repaired rule ({verdict_declared} under the declared rule)")
    with open(paths["tables"], "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
