#!/usr/bin/env python3
"""The two-sided look-back (`EASTERN_TWO_SIDED_LOOKBACK_2026-10-19.md`, sections 2 and 3):
the QTrack look-back's composite read in both flanks, its minimum (northerly) and a mirrored
maximum (southerly), on constructions of two-sided waves first, then blind on the QTrack
tracks that begin at 20 to 30 E and, labeled as not blind, on the look-back's 86. The
maximum reading is the minimum reading applied to the negated composite and shifts. A
composite is an average of positions and dates. Nothing is identified, connected or
explained.

    python3 scripts/eastern_two_sided_lookback.py constructions --out-dir DIR [--tag r1]
    python3 scripts/eastern_two_sided_lookback.py archive --xband DIR --qtrack DIR \\
        --constructions PATH --out-dir DIR [--tag r1]
    python3 scripts/eastern_two_sided_lookback.py verify-lookback --xband DIR \\
        --archive PATH --out PATH
"""

import argparse
import collections
import glob
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eastern_anomaly_minima as A  # noqa: E402
import eastern_minima_histories as HI  # noqa: E402
import eastern_standing_climatology as SC  # noqa: E402
import eastern_crossing_test as X  # noqa: E402
import eastern_qtrack_lookback as LB  # noqa: E402

YEARS = SC.YEARS
LAGS = LB.LAGS
DREL = np.arange(-15.0, 35.0 + 1e-9, 0.5)
REPS = LB.REPS
FLANKS = {"minimum": 1.0, "maximum": -1.0}     # the sign that turns the flank into a minimum
APPROACH_LAGS = (-5, -4, -3, -2, -1)
MIN_WEST_DEG = 7.5
TIE_EPS = 1e-9
SCALES = {"RB1": 372000, "RB9": 3600}   # common integer scales (second review fold, section 10): B1 at 400 n for n of 30 or 31, B2_9 at 3600
LATTICE_TOL = 1e-6
CONTRIB_THRESHOLD = -2.0
CONTRIB_HALF_WIDTH = 1.0
NEW_POPULATION = (20.0, 30.0)
OLD_POPULATION_FROM = 30.0
START_LON = 25.0
WAVE = {"amplitude": 6.0, "wavelength": 20.0, "envelope": 6.0}
TRACK_SEED = 20261021
SPEED_SEED = 20261022
WOBBLE_SEED = 20261023
SPEEDS = {"T1": 4.5, "T2": 2.0, "T3": 6.0, "T4": 4.5, "T8": 4.5}
STANDING_NORTHERLY = {"at": 38.0, "amplitude": 3.0, "period": 14.0, "depth": 10.0}
TWO_SIDED_WOBBLE = {"at": 32.0, "amplitude": 5.0, "period": 14.0}
CASES = ("T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9")
RULE = {"shown": ("T1", "T2"), "not_shown": ("T5", "T6", "T7"), "start_registered": ("T1", "T2", "T5")}


def sha256_of(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


SCRIPT_SHA256_AT_LOAD = sha256_of(os.path.abspath(__file__))


# ------------------------------------------------------------------ populations

def population(tracks, lon_from, lon_to=None):
    """The tracks of the retained seasons whose first position lies in June to September at
    lon_from or east of it and west of lon_to (when given), at a first latitude inside some
    band's padded window, as the look-back's track dicts."""
    out = []
    for k, v in sorted(tracks.items(), key=lambda kv: (kv[1][0], kv[0])):
        t0, la0, lo0 = v[0], v[1], v[2]
        if t0.year in YEARS and 6 <= t0.month <= 9 and lo0 >= lon_from and (lon_to is None or lo0 < lon_to) and any(X.lat_window(b)[0] <= la0 <= X.lat_window(b)[1] for b in X.V.BANDS):
            out.append({"key": list(k), "year": t0.year, "day": LB.season_day(t0.date()), "x0": LB.round_half_east(lo0), "first_lon": float(lo0), "first_lat": float(la0), "band": LB.band_of(la0)})
    return out


# ------------------------------------------------------------------ the composite and the readings

def track_values(field, day, x0r, lons=X.XLONS):
    out = np.full((len(LAGS), len(DREL)), np.nan)
    idx = np.rint((x0r + DREL - lons[0]) / 0.5).astype(int)
    ok = (idx >= 0) & (idx < len(lons))
    for i, k in enumerate(LAGS):
        d = day + k
        if 0 <= d < field.shape[0]:
            out[i, ok] = field[d, idx[ok]]
    return out


def track_stack(fields, tracks, rep, shift=0, years=YEARS):
    return np.array([track_values(fields[(years[(years.index(t["year"]) + shift) % len(years)], t["band"])][rep], t["day"], t["x0"]) for t in tracks])


ExactComposite = collections.namedtuple("ExactComposite", "S n scale")


def exact_composite(stack, scale):
    """The composite as integer numerator sums over a common scale and the contributing
    counts, so that extrema are chosen exactly (second review fold, section 10). Each value
    of a representation is an integer over its own scale, which divides `scale`, so the
    value times `scale` is an integer, and a value off that lattice is refused."""
    fin = np.isfinite(stack)
    x = np.where(fin, stack, 0.0) * scale
    N = np.rint(x)
    if fin.any() and np.max(np.abs(x - N)[fin]) > LATTICE_TOL:
        raise SystemExit(f"REFUSED: values are not integers over the scale {scale}")
    return ExactComposite(np.where(fin, N, 0).astype(np.int64).sum(axis=0), fin.sum(axis=0), int(scale))


def exact_from_values(C, scale=1000):
    """A composite of one value per cell, for a matrix already on the lattice of `scale`."""
    return exact_composite(np.asarray(C, float)[None, ...], scale)


def composite_values(e):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(e.n > 0, e.S / (np.maximum(e.n, 1) * float(e.scale)), np.nan)


def signed(e, sign):
    return ExactComposite(int(sign) * e.S, e.n, e.scale)


def lag_minima(e, drel=DREL):
    """Per lag, the minimum of the composite over drel chosen exactly by cross multiplication
    of numerator sums and counts, its westernmost location on an exact tie, and its value."""
    out = []
    for i in range(e.S.shape[0]):
        best = None
        for j in range(e.S.shape[1]):
            if e.n[i, j] == 0:
                continue
            if best is None or int(e.S[i, j]) * int(e.n[i, best]) < int(e.S[i, best]) * int(e.n[i, j]):
                best = j
        out.append((None, None) if best is None else (int(e.S[i, best]) / (int(e.n[i, best]) * e.scale), float(drel[best])))
    return out


def reading(C, shift_Cs):
    """The minimum reading of section 2.3 on exact composites already turned into minima: per
    lag the minimum, its westernmost location and its rank (shifts at or below within
    TIE_EPS), the start registration, and the five-day approach. The rank is unavailable,
    and so fails every condition, unless every shift has a value at that lag (the look-back
    review's third finding, applied here before anything is scored)."""
    obs = lag_minima(C); shm = [lag_minima(Cs) for Cs in shift_Cs]
    lags = {}
    for i, k in enumerate(LAGS):
        m, d = obs[i]
        ms = [s[i][0] for s in shm]
        valid = sum(1 for v in ms if v is not None)
        lags[k] = {"value": m, "location": d, "rank": None if (m is None or valid < len(ms)) else int(sum(1 for v in ms if v <= m + TIE_EPS)), "shift_values": ms, "valid_shifts": valid}
    r = lambda k: lags[k]["rank"]
    ds = [lags[k]["location"] for k in APPROACH_LAGS + (0,)]
    have = all(d is not None for d in ds)
    ranks_ok = all(r(k) == 0 for k in APPROACH_LAGS)
    order_ok = have and all(a >= b for a, b in zip(ds, ds[1:]))
    west_ok = have and (ds[0] - ds[-1]) >= MIN_WEST_DEG
    return {"lags": lags, "registered": r(0) == 0, "approach": bool(ranks_ok and order_ok and west_ok),
            "parts": {"ranks_zero_at_minus5_to_minus1": ranks_ok, "not_moving_east": bool(order_ok), "west_at_least_7p5": bool(west_ok)}}


def flank_values(reading_, sign):
    """The reading's values in m/s for the flank (a maximum is reported as positive)."""
    out = {}
    for k, L in reading_["lags"].items():
        out[k] = dict(L, value=None if L["value"] is None else sign * L["value"], shift_values=[None if v is None else sign * v for v in L["shift_values"]])
    return dict(reading_, lags=out)


def contribution_count(stack, location, k):
    """Tracks whose own (flank-signed) value within CONTRIB_HALF_WIDTH of location at lag k is
    at or below CONTRIB_THRESHOLD."""
    if location is None:
        return None
    i = LAGS.index(k); cols = np.abs(DREL - location) <= CONTRIB_HALF_WIDTH + 1e-9
    v = stack[:, i][:, cols]
    low = np.min(np.where(np.isfinite(v), v, np.inf), axis=1) if v.size else np.full(len(stack), np.inf)
    return int(np.sum(low <= CONTRIB_THRESHOLD))


def look_back(fields, tracks, rep, years=YEARS):
    """Both flanks of one representation: the composite, its 23 shifts, each flank's reading
    on the flank-signed composite, and each flank's contribution counts."""
    st = track_stack(fields, tracks, rep, years=years)
    e = exact_composite(st, SCALES[rep]); C = composite_values(e)
    shifts = [track_stack(fields, tracks, rep, s, years=years) for s in range(1, len(years))]
    es = [exact_composite(x, SCALES[rep]) for x in shifts]
    Cs = [composite_values(x) for x in es]
    out = {"composite": C, "n": e.n, "shift_composites": Cs, "tracks": len(tracks), "flanks": {}}
    for flank, sign in FLANKS.items():
        rd = reading(signed(e, sign), [signed(x, sign) for x in es])
        contrib = {}
        for k in APPROACH_LAGS:
            sh = [contribution_count(sign * x, lag_minima(signed(c, sign))[LAGS.index(k)][1], k) for x, c in zip(shifts, es)]
            shv = [v for v in sh if v is not None]
            contrib[k] = {"observed": contribution_count(sign * st, rd["lags"][k]["location"], k), "shifts": sh, "shift_mean": float(np.mean(shv)) if shv else None, "shift_max": int(max(shv)) if shv else None}
        out["flanks"][flank] = {"reading": flank_values(rd, sign), "contributions": contrib}
    return out


def flank_verdict(per_rep, flank):
    a = {rep: per_rep[rep]["flanks"][flank]["reading"]["approach"] for rep in per_rep}
    if all(a.values()):
        return "SHOWN"
    if any(a.values()):
        return "under " + " and ".join(k for k, v in a.items() if v) + " only"
    return "not shown"


# ------------------------------------------------------------------ constructions (section 3)

def two_sided(lons, trough, amplitude=WAVE["amplitude"]):
    d = np.asarray(lons) - trough
    return amplitude * np.sin(2.0 * np.pi * d / WAVE["wavelength"]) * np.exp(-d ** 2 / (2.0 * WAVE["envelope"] ** 2))


def construction_days(i, case):
    """Four start days per synthetic season drawn by a fixed seed between day 30 and day 110,
    or for T6 the days of fastest westward wobble for the season's phase, with that phase."""
    rng = np.random.default_rng(TRACK_SEED + i)
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    if case == "T6":
        P = TWO_SIDED_WOBBLE["period"]
        t_fast = [((np.pi - phase) / (2.0 * np.pi)) * P + P * m for m in range(-2, 12)]
        days = sorted({int(round(t - 0.375)) for t in t_fast if LB.START_DAYS[0] <= round(t - 0.375) <= LB.START_DAYS[1]})
        days = sorted(rng.choice(days, size=min(LB.TRACKS_PER_SEASON, len(days)), replace=False).tolist())
    else:
        days = sorted(rng.choice(np.arange(LB.START_DAYS[0], LB.START_DAYS[1] + 1), size=LB.TRACKS_PER_SEASON, replace=False).tolist())
    return [int(d) for d in days], phase


def track_speeds(i, case):
    if case == "T9":
        return [float(v) for v in np.random.default_rng(SPEED_SEED + i).uniform(2.0, 6.0, size=LB.TRACKS_PER_SEASON)]
    return [SPEEDS.get(case)] * LB.TRACKS_PER_SEASON


def wave_paths(case, days, speeds):
    """Each wave's trough path {time: lon}, at START_LON as a daily mean on its start day, or
    for T5 appearing there that day."""
    out = []
    for j, (d, s) in enumerate(zip(days, speeds)):
        if case in ("T7", "T6") or (case == "T8" and j >= 2):
            continue
        if case == "T5":
            out.append(X.path_positions(START_LON, float(d), lambda x: 4.5))
            continue
        lead = int(np.ceil(50.0 / s))
        out.append(X.path_positions(START_LON + s * (lead + 0.375), float(d - lead), (lambda v: (lambda x: v))(s)))
    return out


def construction_season(case, i, lons=X.XLONS):
    """One synthetic season, the look-back's fixed features and seeded noise plus the case's
    two-sided waves, standing northerly wobble or two-sided wobble, rounded to hundredths.
    Returns the daily sums, the tracks and each wave's daily-mean trough position."""
    days, phase = construction_days(i, case)
    speeds = track_speeds(i, case)
    t = np.arange(A.SEASON_DAYS * 4) / 4.0
    F = A.gauss(lons, X.CHANNEL_AT, A.STANDING_DEPTH)[None, :] + A.gauss(lons, X.SOUTHERLY_AT, -6.0, width=3.0)[None, :]
    F = F + A.noise_field(LB.CONSTRUCTION_NOISE_STD, seed=HI.NOISE_SEEDS[i], lons=lons)
    centers = []
    for path in wave_paths(case, days, speeds):
        pos = np.full(len(t), np.nan)
        for kk, tk in enumerate(t):
            if tk in path:
                pos[kk] = path[tk]; F[kk] = F[kk] + two_sided(lons, path[tk])
        centers.append(pos.reshape(A.SEASON_DAYS, 4).mean(axis=1))
    if case == "T4":
        ph = float(np.random.default_rng(WOBBLE_SEED + i).uniform(0.0, 2.0 * np.pi))
        c = STANDING_NORTHERLY["at"] + STANDING_NORTHERLY["amplitude"] * np.sin(2.0 * np.pi * t / STANDING_NORTHERLY["period"] + ph)
        F = F + np.array([A.gauss(lons, x, STANDING_NORTHERLY["depth"]) for x in c])
    if case == "T6":
        c = TWO_SIDED_WOBBLE["at"] + TWO_SIDED_WOBBLE["amplitude"] * np.sin(2.0 * np.pi * t / TWO_SIDED_WOBBLE["period"] + phase)
        F = F + np.array([two_sided(lons, x) for x in c])
    tracks = [{"year": YEARS[i], "day": d, "x0": START_LON, "band": LB.LOW_BAND} for d in days]
    return A.hundredths_from_samples(np.round(F, 2)), tracks, centers


def run_constructions():
    month_of = A.month_index(); out = {}
    for case in CASES:
        fields, tracks = {}, []
        for i, y in enumerate(YEARS):
            H, tr, _ = construction_season(case, i)
            fields[(y, LB.LOW_BAND)] = {rep: LB.anomaly_values(H, month_of, kind) for rep, kind in REPS.items()}
            tracks += tr
        res = {rep: look_back(fields, tracks, rep) for rep in REPS}
        out[case] = {rep: {"flanks": res[rep]["flanks"], "tracks": res[rep]["tracks"]} for rep in REPS}
        out[case]["verdicts"] = {f: flank_verdict(res, f) for f in FLANKS}
    return out


def apply_rule(results):
    """Section 3, per flank: under each representation T1 and T2 approach with the start
    registered, T5 registers its start and does not approach, and T6 and T7 do not approach."""
    out = {}
    for flank in FLANKS:
        fails = []
        for rep in REPS:
            rd = lambda c: results[c][rep]["flanks"][flank]["reading"]
            fails += [f"{c} under {rep} does not approach" for c in RULE["shown"] if not rd(c)["approach"]]
            fails += [f"{c} under {rep} approaches" for c in RULE["not_shown"] if rd(c)["approach"]]
            fails += [f"{c} under {rep} does not register its start" for c in RULE["start_registered"] if not rd(c)["registered"]]
        out[flank] = {"adequate": not fails, "failures": fails}
    return out


# ------------------------------------------------------------------ text, figure, main

HEADER = "| case | representation | flank | location at lags -5, -4, -3, -2, -1, 0 (rank) | value at lag -3, -1, 0 (m/s) | registered | approach |"


def reading_row(label, rep, flank, rd):
    L = rd["lags"]; g = lambda x: "" if x is None else f"{x:g}"; f = lambda x: "" if x is None else f"{x:.2f}"
    locs = ", ".join(f"{g(L[k]['location'])} ({L[k]['rank']})" for k in APPROACH_LAGS + (0,))
    vals = ", ".join(f(L[k]["value"]) for k in (-3, -1, 0))
    return f"| {label} | {rep} | {flank} | {locs} | {vals} | {'yes' if rd['registered'] else 'no'} | {'yes' if rd['approach'] else 'no'} |"


def constructions_text(results, verdict):
    L = ["# two-sided look-back, the constructions (section 3)", "", "location D in degrees relative to the start, rank = shifts of 23 at or beyond the composite's value in the flank", "", HEADER, "| --- |" + " --- |" * 6]
    for c in CASES:
        for rep in REPS:
            for flank in FLANKS:
                L.append(reading_row(c, rep, flank, results[c][rep]["flanks"][flank]["reading"]))
    L += ["", "| case | minimum | maximum |", "| --- | --- | --- |"] + [f"| {c} | {results[c]['verdicts']['minimum']} | {results[c]['verdicts']['maximum']} |" for c in CASES]
    for flank, v in verdict.items():
        L += ["", f"adequate for the {flank} flank, {'yes' if v['adequate'] else 'no'}"] + [f"- {x}" for x in v["failures"]]
    return "\n".join(L) + "\n"


def subsets(tracks):
    return {"band 10 to 14.5 N": [t for t in tracks if t["band"] == LB.LOW_BAND], "other bands": [t for t in tracks if t["band"] != LB.LOW_BAND],
            "first position west of 25 E": [t for t in tracks if t["first_lon"] < 25.0], "first position at or east of 25 E": [t for t in tracks if t["first_lon"] >= 25.0]}


def archive_text(results, verdicts, sizes):
    L = ["# two-sided look-back, the archive (section 2)", "", "location D in degrees relative to the start, rank = shifts of 23 at or beyond the composite's value in the flank", ""]
    for pop, res in results.items():
        L += [f"## {pop} ({sizes[pop]} tracks), minimum {verdicts[pop]['minimum']}, maximum {verdicts[pop]['maximum']}", "", HEADER, "| --- |" + " --- |" * 6]
        for rep in REPS:
            for flank in FLANKS:
                L.append(reading_row("all", rep, flank, res["full"][rep]["flanks"][flank]["reading"]))
        L += ["", "| flank | representation | tracks beyond 2 m/s within 1 degree, lag -5 (shift mean, max) | lag -4 | lag -3 | lag -2 | lag -1 |", "| --- |" + " --- |" * 6]
        for flank in FLANKS:
            for rep in REPS:
                c = res["full"][rep]["flanks"][flank]["contributions"]
                L.append(f"| {flank} | {rep} | " + " | ".join(f"{c[k]['observed']} ({'' if c[k]['shift_mean'] is None else format(c[k]['shift_mean'], '.2f')}, {c[k]['shift_max']})" for k in APPROACH_LAGS) + " |")
        L += ["", "subsets, descriptive, no criterion", "", HEADER.replace("| case |", "| subset (tracks) |"), "| --- |" + " --- |" * 6]
        for name, d in res["subsets"].items():
            if d is None:
                continue
            for rep in REPS:
                for flank in FLANKS:
                    L.append(reading_row(f"{name} ({d[rep]['tracks']})", rep, flank, d[rep]["flanks"][flank]["reading"]))
        L.append("")
    return "\n".join(L)


def draw(path, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pops = list(results)
    fig, axes = plt.subplots(len(pops) * 2, len(REPS), figsize=(6.0 * len(REPS), 3.6 * 2 * len(pops)), sharex=True, constrained_layout=True)
    for p, pop in enumerate(pops):
        for j, rep in enumerate(REPS):
            full = results[pop]["full"][rep]; C = full["composite"]
            lim = float(np.nanmax(np.abs(C))) if np.isfinite(C).any() else 1.0
            a = axes[2 * p, j].pcolormesh(DREL, LAGS, C, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="nearest")
            fig.colorbar(a, ax=axes[2 * p, j], label="composite anomaly (m/s)")
            complete = np.isfinite(C) & np.all([np.isfinite(c) for c in full["shift_composites"]], axis=0)
            with np.errstate(invalid="ignore"):
                lo = np.sum([np.where(complete, c <= C + TIE_EPS, False) for c in full["shift_composites"]], axis=0)
                hi = np.sum([np.where(complete, c >= C - TIE_EPS, False) for c in full["shift_composites"]], axis=0)
            b = axes[2 * p + 1, j].pcolormesh(DREL, LAGS, np.where(complete, np.minimum(lo, hi), np.nan), cmap="viridis_r", vmin=0, vmax=23, shading="nearest")
            fig.colorbar(b, ax=axes[2 * p + 1, j], label="shifts at or beyond, nearer flank")
            for ax in (axes[2 * p, j], axes[2 * p + 1, j]):
                for flank, mk in (("minimum", "k^"), ("maximum", "kv")):
                    ax.plot([full["flanks"][flank]["reading"]["lags"][k]["location"] for k in LAGS], LAGS, mk, ms=5)
                ax.axvline(0.0, color="k", lw=0.7, ls="--"); ax.set_title(f"{pop}, {rep}", fontsize=9); ax.set_ylabel("lag (days)")
    for j in range(len(REPS)):
        axes[-1, j].set_xlabel("longitude relative to the start (degrees)")
    fig.suptitle("Two-sided look-back, composite band-mean anomaly around QTrack starts and its rank among 23 season shifts in the nearer flank. Up triangles mark the minimum, down triangles the maximum. Positions and dates only.", fontsize=9, wrap=True)
    fig.savefig(path, dpi=90); plt.close(fig)


def check_constructions(path, rerun=None):
    """The archive runs only on constructions run by this exact script whose stored cases equal
    a fresh run, which binds every imported calculation, and it takes each flank's adequacy
    from the rule applied to those cases, refusing a stored summary that disagrees (the
    look-back review's fifth finding, applied here before anything is scored)."""
    with open(path, encoding="utf-8") as fh:
        rec = json.load(fh)
    if rec["generated_by"]["sha256"] != SCRIPT_SHA256_AT_LOAD:
        raise SystemExit("REFUSED: the constructions were run by a different version of this script")
    fresh = json.loads(json.dumps({"results": (rerun or run_constructions)()}, default=float))["results"]
    if fresh != rec.get("results"):
        raise SystemExit("REFUSED: the constructions record differs from a fresh run of the same constructions")
    verdict = apply_rule(rec["results"])
    if verdict != rec.get("verdict"):
        raise SystemExit("REFUSED: the stored summary disagrees with the rule applied to the stored cases")
    return {f: v["adequate"] for f, v in verdict.items()}


def verify_lookback(archive, fields):
    """The look-back's committed readings recomputed with exact extrema on its own range, for
    the full population and its subsets under both representations: the observed minimum's
    location and value and every shift's minimum value at every lag (second review fold,
    section 10). Returns the counts checked and the mismatches."""
    tracks = archive["tracks"]
    groups = {"all": tracks, **{k: v for k, v in LB.subsets(tracks).items() if v}}
    checked, mismatches = 0, []
    for name, tr in groups.items():
        for rep in REPS:
            stored = archive["results"][rep]["reading"] if name == "all" else archive["subsets"][name][rep]["reading"]
            obs = lag_minima(exact_composite(LB.track_stack(fields, tr, rep), SCALES[rep]), LB.DREL)
            shm = [lag_minima(exact_composite(LB.track_stack(fields, tr, rep, s), SCALES[rep]), LB.DREL) for s in range(1, len(YEARS))]
            for i, k in enumerate(LAGS):
                L = stored["lags"][str(k)]
                checked += 1 + len(shm)
                if obs[i][1] != L["D_star"] or (obs[i][0] is None) != (L["min"] is None) or (obs[i][0] is not None and abs(obs[i][0] - L["min"]) > 1e-12):
                    mismatches.append(f"{name}, {rep}, lag {k}, observed")
                for s, (a, b) in enumerate(zip(shm, L["shift_minima"]), start=1):
                    if (a[i][0] is None) != (b is None) or (b is not None and abs(a[i][0] - b) > 1e-12):
                        mismatches.append(f"{name}, {rep}, lag {k}, shift {s}")
    return checked, mismatches


def jsonable(full):
    out = {}
    for rep, r in full.items():
        out[rep] = {"tracks": r["tracks"], "flanks": r["flanks"], "n": r["n"].tolist(),
                    "composite": np.where(np.isfinite(r["composite"]), np.round(r["composite"], 6), None).tolist(),
                    "shift_composites": [np.where(np.isfinite(c), np.round(c, 6), None).tolist() for c in r["shift_composites"]]}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("constructions", "archive", "verify-lookback"))
    ap.add_argument("--out-dir", default=None)
    for opt in ("--xband", "--qtrack", "--constructions", "--archive", "--out", "--tag"):
        ap.add_argument(opt, default=None)
    a = ap.parse_args(argv)
    sfx = f"_{a.tag}" if a.tag else ""
    if a.mode == "verify-lookback":
        if not all((a.xband, a.archive, a.out)):
            raise SystemExit("REFUSED: --xband, --archive and --out are required to verify the look-back")
        LB.refuse_existing([a.out])
        xfiles = sorted(glob.glob(os.path.join(a.xband, "eval60_xband_????_??.json.gz")))
        xrecs = [SC.load_record(p) for p in xfiles]
        X.check_xband_identity(xfiles, xrecs)
        with open(a.archive, encoding="utf-8") as fh:
            archive = json.load(fh)
        checked, mismatches = verify_lookback(archive, LB.season_fields(xrecs))
        text = "\n".join([f"# the look-back's committed readings with exact extrema (two-sided record, section 10)", "",
                          f"from {os.path.basename(a.archive)} (sha256 {sha256_of(a.archive)}), script sha256 {SCRIPT_SHA256_AT_LOAD}", "",
                          f"extrema checked {checked}, mismatches {len(mismatches)}"] + [f"- {m}" for m in mismatches]) + "\n"
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(text)
        return 0 if not mismatches else 1
    if not a.out_dir:
        raise SystemExit("REFUSED: --out-dir is required")
    gen = {"script": "scripts/eastern_two_sided_lookback.py", "sha256": SCRIPT_SHA256_AT_LOAD, "declared": "EASTERN_TWO_SIDED_LOOKBACK_2026-10-19.md",
           "lookback_script_sha256": sha256_of(os.path.join(HERE, "eastern_qtrack_lookback.py")), "anomaly_script_sha256": sha256_of(os.path.join(HERE, "eastern_anomaly_minima.py"))}
    if a.mode == "constructions":
        pj, pm = os.path.join(a.out_dir, f"eval60_twosided_constructions{sfx}.json"), os.path.join(a.out_dir, f"eval60_twosided_constructions{sfx}.md")
        LB.refuse_existing([pj, pm])
        results = run_constructions(); verdict = apply_rule(results)
        with open(pj, "w", encoding="utf-8") as fh:
            json.dump({"generated_by": gen, "results": results, "verdict": verdict}, fh, default=float)
        text = constructions_text(results, verdict)
        with open(pm, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(text)
        return 0
    if not all((a.xband, a.qtrack, a.constructions)):
        raise SystemExit("REFUSED: --xband, --qtrack and --constructions are required for the archive")
    adequate = check_constructions(a.constructions)
    paths = {k: os.path.join(a.out_dir, v) for k, v in (("json", f"eval60_twosided_archive{sfx}.json"), ("tables", f"eval60_twosided_tables{sfx}.md"), ("figure", f"eval60_twosided_composite{sfx}.png"))}
    LB.refuse_existing(paths.values())
    xfiles = sorted(glob.glob(os.path.join(a.xband, "eval60_xband_????_??.json.gz")))
    xrecs = [SC.load_record(p) for p in xfiles]
    X.check_xband_identity(xfiles, xrecs)
    q_paths = LB.qtrack_paths(a.qtrack)
    _, qtracks = X.qtrack_points(q_paths)
    pops = {"new, 20 to 30 E, blind": population(qtracks, *NEW_POPULATION), "the look-back's 86, not blind": population(qtracks, OLD_POPULATION_FROM)}
    fields = LB.season_fields(xrecs)
    results, verdicts = {}, {}
    for name, tr in pops.items():
        full = {rep: look_back(fields, tr, rep) for rep in REPS}
        subs = {s: ({rep: {"tracks": len(t), "flanks": look_back(fields, t, rep)["flanks"]} for rep in REPS} if t else None) for s, t in subsets(tr).items()} if name.startswith("new") else {}
        results[name] = {"full": full, "subsets": subs}
        verdicts[name] = {f: (flank_verdict(full, f) if adequate[f] else f"{flank_verdict(full, f)}, the flank's constructions inadequate so it decides nothing") for f in FLANKS}
    draw(paths["figure"], results)
    gen.update({"xband": {os.path.basename(p): sha256_of(p) for p in xfiles}, "qtrack": {os.path.basename(p): sha256_of(p) for p in q_paths},
                "constructions": {"path": a.constructions, "sha256": sha256_of(a.constructions), "adequate": adequate}})
    with open(paths["json"], "w", encoding="utf-8") as fh:
        json.dump({"generated_by": gen, "verdicts": verdicts, "populations": pops,
                   "results": {name: {"full": jsonable(r["full"]), "subsets": r["subsets"]} for name, r in results.items()},
                   "figure": {"path": paths["figure"], "sha256": sha256_of(paths["figure"])}}, fh, default=float)
    text = archive_text(results, verdicts, {k: len(v) for k, v in pops.items()})
    with open(paths["tables"], "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
