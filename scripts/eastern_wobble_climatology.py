#!/usr/bin/env python3
"""The wobble climatology of the standing elements (`EASTERN_WOBBLE_CLIMATOLOGY_2026-10-19.md`):
the daily northerly core of S1, S2 and S3 inside fixed windows over the 24 retained seasons,
on the exact path, with presence, position, monthly swing, timescale, westward excursions
at the look-backs' criteria, proximity with the westward histories, and the constructions
of section 3 run first. A core is a position of strong northerly flow inside a window, and
nothing here separates a wobble of an element from a transient passing through it.

    python3 scripts/eastern_wobble_climatology.py constructions --out-dir DIR [--tag _r1]
    python3 scripts/eastern_wobble_climatology.py archive --records DIR --histories PATH \\
        --constructions PATH --out-dir DIR [--tag _r1] [--floor]
    python3 scripts/eastern_wobble_climatology.py posthoc --records DIR --archive PATH --out-dir DIR
    python3 scripts/eastern_wobble_climatology.py baseline --records DIR --histories PATH --archive PATH --out-dir DIR
    python3 scripts/eastern_wobble_climatology.py lagcontrol --records DIR --histories PATH --archive PATH --out-dir DIR
"""

import argparse
import glob
import hashlib
import json
import os
import sys
from fractions import Fraction

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eastern_anomaly_minima as A  # noqa: E402
import eastern_minima_histories as HI  # noqa: E402
import eastern_standing_climatology as SC  # noqa: E402

YEARS = SC.YEARS
MONTHS = SC.MONTHS
LONS = A.LONS
ELEMENTS = {
    "S1": {"bands": ("10 to 14.5 N",), "window": (38.0, 45.0)},
    "S2": {"bands": ("15 to 19.5 N", "20 to 24.5 N", "25 to 29.5 N"), "window": (44.5, 57.0)},
    "S3": {"bands": ("20 to 24.5 N", "25 to 29.5 N"), "window": (57.5, 70.0)},
}
THRESHOLD = -4.0
SENSITIVITY = (-3.0, -5.0)
MIN_MONTH_DAYS = 20
MIN_SEASON_DAYS = 80
MAX_LAG = 20
EXCURSIONS = {3: Fraction(9, 2), 5: Fraction(15, 2)}
NEAR_DEG = Fraction(3, 2)
HISTORIES = ("RB9", "LN")      # the declared combination, its stored count read from the histories record
CONSTRUCTION_BAND = "15 to 19.5 N"
CONSTRUCTION_NOISE_STD = 3.03          # the spread-matched level of 15 to 19.5 N, eval60_hist_noise_r1.json
PHASE_SEED = 20261024
CASES = {
    "W0": {"at": 50.0},
    "W1": {"at": 50.0, "amplitude": 3.0, "period": 14.0},
    "W2": {"at": 50.0, "amplitude": 5.0, "period": 14.0},
    "W3": {"at": 50.0, "amplitude": 3.0, "period": 28.0},
    "W4": {"at": 50.0, "transient": (35.0, 72.0, 2.0, 5.0)},
    "W5": {"at": 56.0},
}
RULE_TOL = 0.25


def sha256_of(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


SCRIPT_SHA256_AT_LOAD = sha256_of(os.path.abspath(__file__))


# ------------------------------------------------------------------ the daily measures

def window_cols(window, lons=LONS):
    return np.flatnonzero((lons >= window[0] - 1e-9) & (lons <= window[1] + 1e-9))


def day_measure(h_row, cols, lons=LONS, threshold=THRESHOLD):
    """The core, strongest cell and edge flags of one day from its integer sums of hundredths
    (400 times the daily mean), or None when no cell in the window is at or below the
    threshold. The core is the exact centroid, weighted by the negated sums, of the connected
    run of qualifying cells that contains the strongest cell (the repair of section 6), and
    the region-edge flag says whether that run reaches a window edge."""
    thr = int(round(threshold * 400))
    h = np.asarray(h_row, dtype=np.int64)[cols]
    q = h <= thr
    if not q.any():
        return None
    j = int(np.argmin(h))
    lo = hi = j
    while lo > 0 and q[lo - 1]:
        lo -= 1
    while hi < len(h) - 1 and q[hi + 1]:
        hi += 1
    w = [int(-v) for v in h[lo:hi + 1]]
    x2 = [int(round(2 * lons[c])) for c in cols[lo:hi + 1]]
    core = Fraction(sum(a * b for a, b in zip(x2, w)), 2 * sum(w))
    return {"core": core, "strongest": float(lons[cols[j]]), "strongest_value": int(h[j]) / 400.0,
            "edge": j in (0, len(cols) - 1), "region_edge": lo == 0 or hi == len(cols) - 1}


def season_measures(H, cols, lons=LONS, threshold=THRESHOLD):
    return [day_measure(row, cols, lons, threshold) for row in H]


# ------------------------------------------------------------------ the statistics

def quartiles(values):
    v = [float(x) for x in values]
    return [float(np.percentile(v, q)) for q in (25, 50, 75)] if v else None


def monthly_swing(days, month_of):
    """(amplitude estimate, half-range) for each calendar month with at least MIN_MONTH_DAYS
    present days, the amplitude estimate the square root of two times the population
    standard deviation of the daily core."""
    out = {}
    for m in MONTHS:
        cores = [float(d["core"]) for d, mm in zip(days, month_of) if mm == m and d is not None]
        if len(cores) >= MIN_MONTH_DAYS:
            out[m] = (float(np.sqrt(2.0) * np.std(cores)), (max(cores) - min(cores)) / 2.0)
    return out


def autocorrelation(series, lag):
    pairs = [(series[i], series[i + lag]) for i in range(len(series) - lag) if series[i] is not None and series[i + lag] is not None]
    if len(pairs) < 3:
        return None
    a = np.array(pairs, float)
    if a[:, 0].std() == 0 or a[:, 1].std() == 0:
        return None
    return float(np.corrcoef(a[:, 0], a[:, 1])[0, 1])


def season_timescale(days, month_of):
    """The period estimate 2L from the first local minimum L of the autocorrelation of the core
    anomaly (core minus its calendar-month mean over present days), or None when the season
    has fewer than MIN_SEASON_DAYS present days or no minimum within MAX_LAG."""
    present = [d for d in days if d is not None]
    if len(present) < MIN_SEASON_DAYS:
        return None
    means = {}
    for m in MONTHS:
        v = [float(d["core"]) for d, mm in zip(days, month_of) if mm == m and d is not None]
        if v:
            means[m] = float(np.mean(v))
    anom = [None if d is None else float(d["core"]) - means[mm] for d, mm in zip(days, month_of)]
    r = {0: 1.0}
    for L in range(1, MAX_LAG + 2):
        r[L] = autocorrelation(anom, L)
    for L in range(1, MAX_LAG + 1):
        if r[L] is None or r[L - 1] is None or r[L + 1] is None:
            continue
        if r[L] < r[L - 1] and r[L] <= r[L + 1]:
            return 2 * L
    return None


def excursions(days, steps, west):
    """(windows meeting the criterion, complete windows) over windows of steps + 1
    consecutive present days, a window meeting it when the core never moves east and ends
    at least `west` degrees west of where it began, decided on the exact cores."""
    hits = total = 0
    for i in range(len(days) - steps):
        w = days[i:i + steps + 1]
        if any(d is None for d in w):
            continue
        total += 1
        c = [d["core"] for d in w]
        if all(b <= a for a, b in zip(c, c[1:])) and c[0] - c[-1] >= west:
            hits += 1
    return hits, total


def largest_westward(days, steps=5):
    best = None
    for i in range(len(days) - steps):
        w = days[i:i + steps + 1]
        if any(d is None for d in w):
            continue
        v = w[0]["core"] - w[-1]["core"]
        best = v if best is None or v > best else best
    return None if best is None else float(best)


def summarize(seasons, month_of):
    """The section 2.4 statistics over the seasons' daily measures, every one with its count."""
    out = {"presence": {}, "core_quartiles": {}}
    for m in MONTHS:
        idx = [i for i, mm in enumerate(month_of) if mm == m]
        pres = sum(1 for days in seasons for i in idx if days[i] is not None)
        out["presence"][m] = {"present": pres, "days": len(idx) * len(seasons)}
        out["core_quartiles"][m] = quartiles([days[i]["core"] for days in seasons for i in idx if days[i] is not None])
    sw = [v for days in seasons for v in monthly_swing(days, month_of).values()]
    out["swing"] = {"season_months": len(sw), "amplitude_q": quartiles([a for a, _ in sw]), "half_range_q": quartiles([h for _, h in sw])}
    ts = [season_timescale(days, month_of) for days in seasons]
    qualifying = sum(1 for days in seasons if sum(1 for d in days if d is not None) >= MIN_SEASON_DAYS)
    found = [t for t in ts if t is not None]
    out["timescale"] = {"qualifying_seasons": qualifying, "with_minimum": len(found), "period_q": quartiles(found)}
    out["excursions"] = {}
    for steps, west in EXCURSIONS.items():
        hits = total = 0
        for days in seasons:
            h, t = excursions(days, steps, west); hits += h; total += t
        out["excursions"][f"{steps}_day"] = {"west_deg": float(west), "windows": total, "meeting": hits}
    lw = [largest_westward(days) for days in seasons]
    out["largest_five_day_west_q"] = quartiles([v for v in lw if v is not None])
    pres_days = [d for days in seasons for d in days if d is not None]
    out["edges"] = {"present_days": len(pres_days), "edge": sum(1 for d in pres_days if d["edge"]), "region_edge": sum(1 for d in pres_days if d["region_edge"])}
    return out


# ------------------------------------------------------------------ constructions (section 3)

def construction_season(case, i, lons=LONS):
    spec = CASES[case]
    phase = float(np.random.default_rng(PHASE_SEED + i).uniform(0.0, 2.0 * np.pi))
    if "amplitude" in spec:
        standing = (lambda a, p, ph, c: (lambda t: c + a * np.sin(2.0 * np.pi * t / p + ph)))(spec["amplitude"], spec["period"], phase, spec["at"])
    else:
        standing = (lambda c: (lambda t: c))(spec["at"])
    H, s_c, tr_c = A.synth_hundredths(standing=standing, transient=spec.get("transient"), noise=A.noise_field(CONSTRUCTION_NOISE_STD, seed=HI.NOISE_SEEDS[i], lons=lons), lons=lons)
    return H, s_c, tr_c


def run_constructions():
    month_of = A.month_index(); cols = window_cols(ELEMENTS["S2"]["window"]); out = {}
    for case in CASES:
        seasons = [season_measures(construction_season(case, i)[0], cols) for i in range(len(YEARS))]
        out[case] = summarize(seasons, month_of)
    return out


def apply_rule(results):
    """Section 3: median amplitude within RULE_TOL of the constructed amplitude in W1 to W3,
    median period within RULE_TOL of the constructed period, and W0 below 1 degree."""
    fails = []
    for case in ("W1", "W2", "W3"):
        a, p = CASES[case]["amplitude"], CASES[case]["period"]
        amp = (results[case]["swing"]["amplitude_q"] or [None, None])[1]
        per = (results[case]["timescale"]["period_q"] or [None, None])[1]
        if amp is None or abs(amp - a) > RULE_TOL * a:
            fails.append(f"{case} median amplitude {amp} not within {RULE_TOL:g} of {a:g}")
        if per is None or abs(per - p) > RULE_TOL * p:
            fails.append(f"{case} median period {per} not within {RULE_TOL:g} of {p:g}")
    a0 = (results["W0"]["swing"]["amplitude_q"] or [None, None])[1]
    if a0 is None or a0 >= 1.0:
        fails.append(f"W0 median amplitude {a0} not below 1")
    return {"adequate": not fails, "failures": fails}


# ------------------------------------------------------------------ the archive

def element_seasons(by, element, band, threshold=THRESHOLD):
    cols = window_cols(ELEMENTS[element]["window"])
    return [season_measures(A.season_hundredths(by[y], band)[1], cols, threshold=threshold) for y in YEARS]


def stored_history_count(histories_record):
    rep, linker = HISTORIES
    return len(histories_record["tables"][HI.combo(rep, linker)]["westward_histories"])


def recompute_histories(by, expected):
    """The westward histories of the declared combination per (year, band), with their daily
    members, refused unless their number equals the stored count."""
    rep, linker = HISTORIES
    month_of = A.month_index(); out = {}
    bands = list(by[YEARS[0]][MONTHS[0]]["bands"])
    for y in YEARS:
        for b in bands:
            H = A.season_hundredths(by[y], b)[1]
            out[(y, b)] = [h for h in HI.link(HI.rep_minima(rep, H, month_of, LONS), linker) if HI.classify(h) == "westward"]
    n = sum(len(v) for v in out.values())
    if n != expected:
        raise SystemExit(f"REFUSED: {n} recomputed westward histories under {rep} with {linker}, not the stored {expected}")
    return out


def proximity(histories, cores_by, element="S2"):
    """Member-days of westward histories in the element's bands lying inside its window on
    a day it is present, how many lie within NEAR_DEG of the core, and how many histories
    have at least half of such member-days near it."""
    lo, hi = ELEMENTS[element]["window"]
    days_total = near = 0; hist_with = hist_half = 0
    for (y, b), hs in histories.items():
        if b not in ELEMENTS[element]["bands"]:
            continue
        cores = cores_by[(element, y, b)]
        for h in hs:
            n_in = n_near = 0
            for d, x in zip(h["days"], h["lons"]):
                if lo <= x <= hi and cores[d] is not None:
                    n_in += 1
                    if abs(Fraction(x).limit_denominator(2) - cores[d]["core"]) <= NEAR_DEG:
                        n_near += 1
            days_total += n_in; near += n_near
            if n_in:
                hist_with += 1
                if 2 * n_near >= n_in:
                    hist_half += 1
    return {"member_days_in_window": days_total, "near_core": near, "histories_in_window": hist_with, "histories_half_near": hist_half}


def archive_results(by):
    month_of = A.month_index(); out = {}; cores_by = {}
    for el, spec in ELEMENTS.items():
        for b in spec["bands"]:
            seasons = element_seasons(by, el, b)
            res = {"declared": summarize(seasons, month_of), "sensitivity": {}}
            for thr in SENSITIVITY:
                s2 = summarize(element_seasons(by, el, b, thr), month_of)
                res["sensitivity"][f"{thr:g}"] = {k: s2[k] for k in ("presence", "swing", "excursions")}
            res["daily"] = {str(y): [None if d is None else round(float(d["core"]), 2) for d in days] for y, days in zip(YEARS, seasons)}
            out[f"{el}|{b}"] = res
            for y, days in zip(YEARS, seasons):
                cores_by[(el, y, b)] = days
    return out, cores_by


# ------------------------------------------------------------------ the season-shift baseline (section 12)

def shifted_cores(cores_by, k, element="S2"):
    """The element's cores of season i + k, cyclic over the retained seasons, placed under season i."""
    out = {}
    for (el, y, b) in cores_by:
        if el == element:
            out[(el, y, b)] = cores_by[(el, YEARS[(YEARS.index(y) + k) % len(YEARS)], b)]
    return out


def member_distances(histories, cores_by, element="S2"):
    """The exact distances between the members inside the element's window and its core, on
    the days it is present, over the histories in its bands."""
    lo, hi = ELEMENTS[element]["window"]
    out = []
    for (y, b), hs in histories.items():
        if b not in ELEMENTS[element]["bands"]:
            continue
        cores = cores_by[(element, y, b)]
        for h in hs:
            for d, x in zip(h["days"], h["lons"]):
                if lo <= x <= hi and cores[d] is not None:
                    out.append(abs(Fraction(x).limit_denominator(2) - cores[d]["core"]))
    return out


def exact_median(values):
    v = sorted(values)
    if not v:
        return None
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2


def pairing_stats(histories, cores_by, bands):
    hs = {key: v for key, v in histories.items() if key[1] in bands}
    p = proximity(hs, cores_by)
    n, near = p["member_days_in_window"], p["near_core"]
    return {"member_days": n, "near": near, "share": Fraction(near, n) if n else None, "histories": p["histories_in_window"],
            "half_near": p["histories_half_near"], "median_distance": exact_median(member_distances(hs, cores_by))}


def shift_summary(obs, shifts):
    """Mean, range and rank (shifts at or beyond the observed value) of the near share and the
    median distance, a rank unavailable when any shift lacks the statistic."""
    out = {}
    for stat, beyond in (("share", lambda v, o: v >= o), ("median_distance", lambda v, o: v <= o)):
        vals = [sh[stat] for sh in shifts]
        if obs[stat] is None or any(v is None for v in vals):
            out[stat] = {"observed": None if obs[stat] is None else float(obs[stat]), "rank": None, "mean": None, "range": None}
            continue
        out[stat] = {"observed": float(obs[stat]), "rank": sum(1 for v in vals if beyond(v, obs[stat])), "mean": float(sum(vals) / len(vals)),
                     "range": [float(min(vals)), float(max(vals))], "inside_range": min(vals) <= obs[stat] <= max(vals)}
    return out


BASELINE_READINGS = {"nearer": "the members lie nearer S2's core on the same day than in any shifted pairing",
                     "matched": "the proximity is matched by other seasons' cores on the same day of the season"}
LAG_READINGS = {"nearer": "the members lie nearer S2's core on their own day than on the same season's days 10 to 20 days away",
                "matched": "the proximity is matched by the same season's core 10 to 20 days away"}


def fixed_reading(summary, readings):
    """The fixed reading of sections 12 and 15: 'nearer' with rank 0 on both pooled statistics,
    'matched' with both own values inside the references' ranges, otherwise none."""
    sh, md = summary["share"], summary["median_distance"]
    if sh["rank"] is None or md["rank"] is None:
        return None
    if sh["rank"] == 0 and md["rank"] == 0:
        return readings["nearer"]
    if sh["inside_range"] and md["inside_range"]:
        return readings["matched"]
    return None


def baseline_reading(summary):
    """Section 12's fixed reading of the pooled statistics."""
    return fixed_reading(summary, BASELINE_READINGS)


def _plain(r):
    return {k: (float(v) if isinstance(v, Fraction) else v) for k, v in r.items()}


def season_shift_baseline(histories, cores_by):
    groups = {"pooled": ELEMENTS["S2"]["bands"], **{b: (b,) for b in ELEMENTS["S2"]["bands"]}}
    out = {}
    for g, bands in groups.items():
        obs = pairing_stats(histories, cores_by, bands)
        shifts = [pairing_stats(histories, shifted_cores(cores_by, k), bands) for k in range(1, len(YEARS))]
        summ = shift_summary(obs, shifts)
        out[g] = {"observed": _plain(obs), "shifts": [_plain(x) for x in shifts], "summary": summ}
    out["reading"] = baseline_reading(out["pooled"]["summary"])
    return out


def check_observed(base, archive_prox):
    """The observed pairing must reproduce the archive record's proximity count."""
    o = base["pooled"]["observed"]
    if (o["member_days"], o["near"], o["histories"], o["half_near"]) != (archive_prox["member_days_in_window"], archive_prox["near_core"],
                                                                          archive_prox["histories_in_window"], archive_prox["histories_half_near"]):
        raise SystemExit("REFUSED: the observed pairing does not reproduce the archive record's proximity count")


def comparison_rows(base, n_ref, ref, own="observed"):
    """The table of an own pairing against its reference pairings, pooled and per band."""
    L = [f"| bands | {own} near, of member-days | {own} half-near, of histories | {ref} near share, mean and range | near-share rank | "
         f"{own} median distance (deg) | {ref} median distance, mean and range (deg) | distance rank |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for g in ("pooled",) + ELEMENTS["S2"]["bands"]:
        o, sm = base[g]["observed"], base[g]["summary"]
        sh, md = sm["share"], sm["median_distance"]
        rng = lambda x, n=3: "" if x["range"] is None else f"{x['mean']:.{n}f} ({x['range'][0]:.{n}f} to {x['range'][1]:.{n}f})"
        L.append(f"| {g} | {o['near']} of {o['member_days']} | {o['half_near']} of {o['histories']} | {rng(sh)} | {sh['rank']} of {n_ref} | "
                 f"{'' if md['observed'] is None else format(md['observed'], '.3f')} | {rng(md)} | {md['rank']} of {n_ref} |")
    return L


def baseline_text(base):
    L = ["# wobble climatology, the season-shift baseline of the proximity count (section 12)", ""] + comparison_rows(base, 23, "shifts'")
    L += ["", f"reading under section 12, {base['reading'] or 'none, the result reported by its ranks'}"]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ the within-season lag control (section 15)

LAGS = tuple(range(-20, -9)) + tuple(range(10, 21))
LAG_CENTERS = (47.5, 48.0, 47.0, 46.5, 56.0, 53.5, 55.0, 51.0, 56.5, 52.0, 49.5, 49.0,
               53.0, 48.5, 45.5, 45.0, 51.5, 52.5, 46.0, 50.0, 54.0, 50.5, 55.5, 54.5)
LAG_OFFSETS = (5.0, 2.0, 1.5, 1.0, 0.5, -0.5, -1.0, -1.5, -4.5)
LAG_FIXED = -2000                      # the fixed core, -5 m/s as an integer sum of hundredths
LAG_CASES = {"C1": -1200, "C2": -2400}  # the moving contribution, -3 m/s (weaker) and -6 m/s (stronger)
LAG_EXPECTED = {"C1": "matched", "C2": "nearer"}          # the declared rule of section 15
EDGE_CORE = 51.0
EDGE_NEAR = (55.0, 52.0, 51.5, 50.5, 49.5, 46.5)
EDGE_FAR = (64.0, 61.0, 58.0, 55.0, 54.5, 54.0, 53.5, 47.5)
EDGE_STARTS = {"C3": (4, 50, 112), "C4": (25, 50, 80)}   # the two nearer paths at the season's edges, then inside it
MATCHED_EXPECTED = {"C1": "matched", "C2": "nearer", "C3": "matched", "C4": "matched"}   # the corrected rule of section 17
MATCHED_READINGS = {"nearer": "on the same member-days, the members lie nearer S2's core on their own day than at every lag of 10 to 20 days",
                    "matched": "on the same member-days, every lag of 10 to 20 days brings S2's core as near as the own day"}


def lagged_cores(cores_by, L, element="S2"):
    """The element's cores of day d + L in the same season and band placed under day d, absent
    where d + L falls outside the season."""
    out = {}
    for (el, y, b), days in cores_by.items():
        if el == element:
            out[(el, y, b)] = [days[d + L] if 0 <= d + L < len(days) else None for d in range(len(days))]
    return out


def lag_control(histories, cores_by):
    groups = {"pooled": ELEMENTS["S2"]["bands"], **{b: (b,) for b in ELEMENTS["S2"]["bands"]}}
    out = {}
    for g, bands in groups.items():
        obs = pairing_stats(histories, cores_by, bands)
        lags = [pairing_stats(histories, lagged_cores(cores_by, L), bands) for L in LAGS]
        out[g] = {"observed": _plain(obs), "lags": [dict(_plain(x), lag=L) for x, L in zip(lags, LAGS)], "summary": shift_summary(obs, lags)}
    out["reading"] = fixed_reading(out["pooled"]["summary"], LAG_READINGS)
    return out


def lag_construction(case, start=50, band="15 to 19.5 N"):
    """Section 15's constructions through the real readers: each year's core fixed at its own
    longitude and one westward-moving contribution per year, weaker than the core in C1 and
    stronger in C2."""
    col = lambda x: int(np.flatnonzero(np.isclose(LONS, x))[0])
    cols = window_cols(ELEMENTS["S2"]["window"]); month_of = A.month_index()
    histories, cores = {}, {}
    for year, center in zip(YEARS, LAG_CENTERS):
        H = np.zeros((len(month_of), len(LONS)), dtype=np.int64)
        H[:, col(center)] = LAG_FIXED
        for day, x in enumerate([center + o for o in LAG_OFFSETS], start):
            H[day, col(x)] = LAG_CASES[case]
        histories[(year, band)] = [h for h in HI.link(HI.rep_minima("RB9", H, month_of, LONS), "LN") if HI.classify(h) == "westward"]
        cores[("S2", year, band)] = season_measures(H, cols)
    return histories, cores


def edge_construction(starts, band="15 to 19.5 N"):
    """Section 17's constructions, the review's: one season with a core fixed at 51 E on every
    day and three weaker westward paths through the real readers, the two nearer ones starting
    on the given days and the farther one on day 50."""
    col = lambda x: int(np.flatnonzero(np.isclose(LONS, x))[0])
    month_of = A.month_index()
    H = np.zeros((len(month_of), len(LONS)), dtype=np.int64)
    H[:, col(EDGE_CORE)] = LAG_FIXED
    for start, path in zip(starts, (EDGE_NEAR, EDGE_FAR, EDGE_NEAR)):
        for day, x in enumerate(path, start):
            H[day, col(x)] = LAG_CASES["C1"]
    key = (YEARS[0], band)
    hist = [h for h in HI.link(HI.rep_minima("RB9", H, month_of, LONS), "LN") if HI.classify(h) == "westward"]
    return {key: hist}, {("S2",) + key: season_measures(H, window_cols(ELEMENTS["S2"]["window"]))}


def construction_case(case):
    return lag_construction(case) if case in LAG_CASES else edge_construction(EDGE_STARTS[case])


def matched_lag_comparison(histories, cores_by, element="S2"):
    """Section 17: at each lag, the own-day and lagged statistics on the same member-days, those
    inside the element's window with the element present on both the own and the lagged day,
    with the member-days present on only one of the two counted apart."""
    lo, hi = ELEMENTS[element]["window"]
    groups = {"pooled": ELEMENTS[element]["bands"], **{b: (b,) for b in ELEMENTS[element]["bands"]}}
    out = {}
    for g, bands in groups.items():
        rows = []
        for L in LAGS:
            r = {"lag": L, "common": 0, "own_only": 0, "lag_only": 0}
            d_own, d_lag = [], []
            for (y, b), hs in histories.items():
                if b not in bands:
                    continue
                cores = cores_by[(element, y, b)]
                for h in hs:
                    for d, x in zip(h["days"], h["lons"]):
                        if not lo <= x <= hi:
                            continue
                        xo = Fraction(x).limit_denominator(2)
                        own, lag = cores[d], (cores[d + L] if 0 <= d + L < len(cores) else None)
                        if own is not None and lag is not None:
                            r["common"] += 1
                            d_own.append(abs(xo - own["core"])); d_lag.append(abs(xo - lag["core"]))
                        elif own is not None:
                            r["own_only"] += 1
                        elif lag is not None:
                            r["lag_only"] += 1
            r.update(own_near=sum(1 for v in d_own if v <= NEAR_DEG), lag_near=sum(1 for v in d_lag if v <= NEAR_DEG),
                     own_median=exact_median(d_own), lag_median=exact_median(d_lag))
            rows.append(r)
        avail = all(r["common"] for r in rows)
        summary = {"share_at_or_beyond": sum(1 for r in rows if r["lag_near"] >= r["own_near"]) if avail else None,
                   "distance_at_or_beyond": sum(1 for r in rows if r["lag_median"] <= r["own_median"]) if avail else None}
        out[g] = {"lags": [_plain(r) for r in rows], "summary": summary}
    out["reading"] = matched_reading(out["pooled"]["summary"])
    return out


def matched_reading(summary):
    """Section 17's corrected reading: 'nearer' when no lag reaches the own day on either
    statistic over the same member-days, 'matched' when every lag does on both, otherwise none."""
    a, b = summary["share_at_or_beyond"], summary["distance_at_or_beyond"]
    if a is None or b is None:
        return None
    if a == 0 and b == 0:
        return MATCHED_READINGS["nearer"]
    if a == len(LAGS) and b == len(LAGS):
        return MATCHED_READINGS["matched"]
    return None


def check_lag_constructions():
    """The archive is read only when C1 and C2 give their declared readings under section 15's
    rule and all four constructions give their expected readings under section 17's matched
    rule. C3's reading under the declared rule is recorded, since it is that rule's failure."""
    out = {}
    for case in MATCHED_EXPECTED:
        hist, cores = construction_case(case)
        res, mat = lag_control(hist, cores), matched_lag_comparison(hist, cores)
        out[case] = {"reading": res["reading"], "share_rank": res["pooled"]["summary"]["share"]["rank"],
                     "distance_rank": res["pooled"]["summary"]["median_distance"]["rank"], "observed": res["pooled"]["observed"],
                     "matched_reading": mat["reading"], "matched_summary": mat["pooled"]["summary"]}
        if case in LAG_EXPECTED and res["reading"] != LAG_READINGS[LAG_EXPECTED[case]]:
            raise SystemExit(f"REFUSED: construction {case} reads {res['reading']!r}, not the expected {LAG_EXPECTED[case]} reading of section 15")
        if mat["reading"] != MATCHED_READINGS[MATCHED_EXPECTED[case]]:
            raise SystemExit(f"REFUSED: construction {case} reads {mat['reading']!r} on matched member-days, not the expected {MATCHED_EXPECTED[case]} reading of section 17")
    return out


def lag_text(res, cons, matched=None):
    L = ["# wobble climatology, the within-season lag control of the proximity count (section 15)", "",
         "constructions, read before the archive"]
    for c, v in cons.items():
        line = f"- {c}, {v['reading']}, share rank {v['share_rank']} and distance rank {v['distance_rank']} of {len(LAGS)}, own pairing {v['observed']['near']} near of {v['observed']['member_days']}"
        if "matched_reading" in v:
            line += f", and on matched member-days {v['matched_reading']}"
        L.append(line)
    L += [""] + comparison_rows(res, len(LAGS), "lags'", own="own-day")
    L += ["", f"reading under section 15, {res['reading'] or 'none, the result reported by its ranks'}"]
    if matched is not None:
        L += ["", "## the same member-days at each lag (section 17)", "",
              "| bands | matched member-days | own-only, lag-only | own-day near share | lagged near share | own-day median distance (deg) | lagged median distance (deg) | lags at or beyond the own day, share and distance |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        span = lambda v, n=3: f"{min(v):.{n}f} to {max(v):.{n}f}" if v else ""
        for g in ("pooled",) + ELEMENTS["S2"]["bands"]:
            rows, sm = matched[g]["lags"], matched[g]["summary"]
            ok = [r for r in rows if r["common"]]
            L.append(f"| {g} | {min(r['common'] for r in rows)} to {max(r['common'] for r in rows)} | "
                     f"{min(r['own_only'] for r in rows)} to {max(r['own_only'] for r in rows)}, {min(r['lag_only'] for r in rows)} to {max(r['lag_only'] for r in rows)} | "
                     f"{span([r['own_near'] / r['common'] for r in ok])} | {span([r['lag_near'] / r['common'] for r in ok])} | "
                     f"{span([r['own_median'] for r in ok])} | {span([r['lag_median'] for r in ok])} | {sm['share_at_or_beyond']} and {sm['distance_at_or_beyond']} of {len(LAGS)} |")
        L += ["", f"reading under section 17, {matched['reading'] or 'none, the result reported by its counts'}"]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ post hoc reader (section 10)

def runs(h_row, cols, threshold=THRESHOLD):
    """The connected runs of qualifying cells in the window, as (first, last) column positions."""
    thr = int(round(threshold * 400))
    q = np.asarray(h_row, dtype=np.int64)[cols] <= thr
    out, i = [], 0
    while i < len(q):
        if q[i]:
            k = i
            while k + 1 < len(q) and q[k + 1]:
                k += 1
            out.append((i, k)); i = k + 1
        else:
            i += 1
    return out


def run_summary(seasons_h, cols, lons=LONS, daily_out=None):
    """Over the present days of the seasons' integer sums: the days with two or more runs in
    the window, the side of the window the strongest cell's run reaches, its span in degrees,
    and the absolute day-to-day change of the core between consecutive present days. With
    `daily_out`, each season's daily core rounded as the archive stores it is appended."""
    present = multi = west = east = both = 0
    spans, steps = [], []
    for H in seasons_h:
        prev = None
        series = []
        if daily_out is not None:
            daily_out.append(series)
        for row in H:
            d = day_measure(row, cols, lons)
            series.append(None if d is None else round(float(d["core"]), 2))
            if d is None:
                prev = None
                continue
            present += 1
            rs = runs(row, cols)
            multi += len(rs) >= 2
            j = int(np.argmin(np.asarray(row, dtype=np.int64)[cols]))
            a, b = next(r for r in rs if r[0] <= j <= r[1])
            w, e = a == 0, b == len(cols) - 1
            both += w and e; west += w and not e; east += e and not w
            spans.append(float(lons[cols[b]] - lons[cols[a]]))
            if prev is not None:
                steps.append(abs(float(d["core"] - prev)))
            prev = d["core"]
    pct = lambda v: [float(np.percentile(v, q)) for q in (25, 50, 75, 90)] if v else None
    return {"present_days": present, "two_or_more_runs": multi, "run_reaches_west_only": west, "run_reaches_east_only": east,
            "run_spans_window": both, "run_span_q": pct(spans), "steps": len(steps), "step_q": pct(steps)}


def check_record_hashes(current, archive_generated_by):
    """The reader reads the same month records the archive read, by name and digest."""
    if archive_generated_by.get("records") != current:
        raise SystemExit("REFUSED: the month records differ from those the archive record names")


def check_against_archive(arch, dailies, archive_results_):
    """The reader's present days, region-edge counts and rounded daily cores must equal the
    archive record's for every element and band."""
    for k, r in arch.items():
        dec = archive_results_[k]["declared"]
        if r["present_days"] != sum(v["present"] for v in dec["presence"].values()):
            raise SystemExit(f"REFUSED: the present days of {k} differ from the archive record")
        if r["run_reaches_west_only"] + r["run_reaches_east_only"] + r["run_spans_window"] != dec["edges"]["region_edge"]:
            raise SystemExit(f"REFUSED: the region-edge days of {k} differ from the archive record")
        if dailies[k] != [archive_results_[k]["daily"][str(y)] for y in YEARS]:
            raise SystemExit(f"REFUSED: the daily cores of {k} differ from the archive record")


def threshold_orders(results):
    """Section 2.5: the order of the element-bands by median amplitude estimate and by the
    five-day excursion share at each threshold, and whether either order changes."""
    out = {}
    for stat in ("median_amplitude", "five_day_share"):
        orders = {}
        for thr in ("-3", "-4", "-5"):
            vals = {}
            for k, v in results.items():
                s = v["declared"] if thr == "-4" else v["sensitivity"][thr]
                if stat == "median_amplitude":
                    vals[k] = (s["swing"]["amplitude_q"] or [None, float("nan")])[1]
                else:
                    e = s["excursions"]["5_day"]
                    vals[k] = e["meeting"] / e["windows"] if e["windows"] else float("nan")
            orders[thr] = sorted(vals, key=lambda k: (-vals[k], k))
        out[stat] = {"orders": orders, "threshold_dependent": len({tuple(o) for o in orders.values()}) > 1}
    return out


def posthoc_text(arch, cons, orders):
    L = ["# wobble climatology, the post hoc reader (section 10, written after the archive was read)", "",
         "| element and band, or construction | present days | days with two or more runs | strongest cell's run reaches west edge only, east only, both | run span, quartiles and 90th percentile (deg) | day-to-day core change, quartiles and 90th percentile (deg), of steps |",
         "| --- | --- | --- | --- | --- | --- |"]
    for k, r in list(cons.items()) + list(arch.items()):
        L.append(f"| {label(k)} | {r['present_days']} | {r['two_or_more_runs']} | {r['run_reaches_west_only']}, {r['run_reaches_east_only']}, {r['run_spans_window']} | "
                 f"{fmt(r['run_span_q'])} | {fmt(r['step_q'])}, of {r['steps']} |")
    L += ["", "## orders across thresholds (section 2.5)", ""]
    for stat, v in orders.items():
        L.append(f"- {stat}, threshold-dependent {'yes' if v['threshold_dependent'] else 'no'}")
        L += [f"  - {thr}: {'; '.join(label(x) for x in o)}" for thr, o in v["orders"].items()]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ text, figure, main

def label(key):
    """An element and band key for display, without the vertical bar that would split a Markdown cell."""
    return key.replace("|", ", ")


def fmt(q, n=2):
    return "" if q is None else ", ".join(f"{x:.{n}f}" for x in q)


def summary_rows(name, s):
    pres = sum(v["present"] for v in s["presence"].values()); days = sum(v["days"] for v in s["presence"].values())
    e3, e5 = s["excursions"]["3_day"], s["excursions"]["5_day"]
    return (f"| {label(name)} | {pres} of {days} | {s['swing']['season_months']} | {fmt(s['swing']['amplitude_q'])} | {fmt(s['swing']['half_range_q'])} | "
            f"{s['timescale']['with_minimum']} of {s['timescale']['qualifying_seasons']} | {fmt(s['timescale']['period_q'], 1)} | {e3['meeting']} of {e3['windows']} | "
            f"{e5['meeting']} of {e5['windows']} | {fmt(s['largest_five_day_west_q'])} | {s['edges']['edge']}, {s['edges']['region_edge']} of {s['edges']['present_days']} |")


HEADER = ("| case | present days | season-months with 20 or more | amplitude estimate, quartiles (deg) | half-range, quartiles (deg) | seasons with a minimum, of qualifying | "
          "period, quartiles (days) | three-day excursions, of windows | five-day excursions, of windows | largest five-day westward displacement, quartiles (deg) | edge, region-edge, of present days |")


def constructions_text(results, verdict):
    L = ["# wobble climatology, the constructions (section 3)", "", HEADER, "| --- |" + " --- |" * 10]
    L += [summary_rows(c, results[c]) for c in CASES]
    L += ["", f"adequate under the rule of section 3, {'yes' if verdict['adequate'] else 'no'}"] + [f"- {x}" for x in verdict["failures"]]
    return "\n".join(L) + "\n"


def archive_text(results, prox, floor=None):
    L = ["# wobble climatology, the archive (section 2)", "", "declared threshold -4 m/s", ""]
    if floor is not None:
        L += ["read under section 8, the rule of section 3 not adequate on W0's bar alone, with W0's row from the constructions as the stated floor, "
              "measured in the S2 window at 3.03 m/s with noise correlated over the feature's width and unmatched for S1, S3 and the other bands", ""]
    L += [HEADER.replace("| case |", "| element and band |"), "| --- |" + " --- |" * 10]
    if floor is not None:
        L.append(summary_rows("W0, constructions (the floor)", floor))
    L += [summary_rows(k, v["declared"]) for k, v in results.items()]
    L += ["", "## presence and core position by calendar month", "", "| element and band | month | present, of days | core quartiles (deg E) |", "| --- | --- | --- | --- |"]
    for k, v in results.items():
        for m in MONTHS:
            p = v["declared"]["presence"][m]
            L.append(f"| {label(k)} | {m} | {p['present']} of {p['days']} | {fmt(v['declared']['core_quartiles'][m])} |")
    L += ["", "## threshold sensitivity", "", "| element and band | threshold (m/s) | present days | amplitude estimate, quartiles | five-day excursions, of windows |", "| --- | --- | --- | --- | --- |"]
    for k, v in results.items():
        for thr, s in [("-4", v["declared"])] + list(v["sensitivity"].items()):
            pres = sum(x["present"] for x in s["presence"].values()); days = sum(x["days"] for x in s["presence"].values())
            L.append(f"| {label(k)} | {thr} | {pres} of {days} | {fmt(s['swing']['amplitude_q'])} | {s['excursions']['5_day']['meeting']} of {s['excursions']['5_day']['windows']} |")
    L += ["", "## proximity of the westward histories to the S2 core (positions only)", "",
          f"{prox['near_core']} of {prox['member_days_in_window']} member-days inside the S2 window on present days lie within 1.5 degrees of the core, and {prox['histories_half_near']} of {prox['histories_in_window']} such histories have at least half of those days near it"]
    return "\n".join(L) + "\n"


def draw(path, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = list(results)
    fig, axes = plt.subplots(len(keys), 1, figsize=(10, 2.3 * len(keys)), sharex=True, constrained_layout=True)
    for ax, k in zip(np.atleast_1d(axes), keys):
        for y, series in results[k]["daily"].items():
            ax.plot(range(len(series)), [np.nan if v is None else v for v in series], lw=0.5, alpha=0.5)
        el = k.split("|")[0]; lo, hi = ELEMENTS[el]["window"]
        ax.set_ylim(lo - 0.5, hi + 0.5); ax.set_ylabel("core (deg E)"); ax.set_title(k, fontsize=9)
    np.atleast_1d(axes)[-1].set_xlabel("day of the season from 1 June")
    fig.suptitle("Daily northerly core of the standing elements inside their windows, one line per season (24 seasons). Positions only.", fontsize=9)
    fig.savefig(path, dpi=90); plt.close(fig)


def check_constructions(path, rerun=None, floor=False):
    """The archive runs only on constructions run by this exact script whose stored results
    equal a fresh run and pass the rule recomputed from them, or, with `floor` (section 8),
    fail it on W0's bar alone."""
    with open(path, encoding="utf-8") as fh:
        rec = json.load(fh)
    if rec["generated_by"]["sha256"] != SCRIPT_SHA256_AT_LOAD:
        raise SystemExit("REFUSED: the constructions were run by a different version of this script")
    fresh = json.loads(json.dumps({"results": (rerun or run_constructions)()}, default=float))["results"]
    if fresh != rec.get("results"):
        raise SystemExit("REFUSED: the constructions record differs from a fresh run of the same constructions")
    verdict = apply_rule(rec["results"])
    if not verdict["adequate"] and not floor:
        raise SystemExit(f"REFUSED: the constructions do not pass the rule of section 3 ({verdict['failures']})")
    if [f for f in verdict["failures"] if not f.startswith("W0 median amplitude")]:
        raise SystemExit(f"REFUSED: the constructions fail the rule beyond W0's bar, which section 8 does not cover ({verdict['failures']})")
    if verdict != rec.get("verdict"):
        raise SystemExit("REFUSED: the stored summary disagrees with the rule applied to the stored cases")
    return rec


def refuse_existing(paths):
    for p in paths:
        if os.path.exists(p):
            raise SystemExit(f"REFUSED: {p} exists and a record is never overwritten")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("constructions", "archive", "posthoc", "baseline", "lagcontrol"))
    ap.add_argument("--out-dir", required=True)
    for opt in ("--records", "--histories", "--constructions", "--archive"):
        ap.add_argument(opt, default=None)
    ap.add_argument("--tag", default="", help="a suffix for the output names, so a repaired run sits beside the declared one")
    ap.add_argument("--floor", action="store_true", help="read the archive under section 8, accepting a failure on W0's bar alone")
    a = ap.parse_args(argv)
    gen = {"script": "scripts/eastern_wobble_climatology.py", "sha256": SCRIPT_SHA256_AT_LOAD, "declared": "EASTERN_WOBBLE_CLIMATOLOGY_2026-10-19.md",
           "anomaly_script_sha256": sha256_of(os.path.join(HERE, "eastern_anomaly_minima.py")), "histories_script_sha256": sha256_of(os.path.join(HERE, "eastern_minima_histories.py"))}
    if a.mode == "constructions":
        pj, pm = os.path.join(a.out_dir, f"eval60_wobble_constructions{a.tag}.json"), os.path.join(a.out_dir, f"eval60_wobble_constructions{a.tag}.md")
        refuse_existing([pj, pm])
        results = run_constructions(); verdict = apply_rule(results)
        with open(pj, "w", encoding="utf-8") as fh:
            json.dump({"generated_by": gen, "results": results, "verdict": verdict}, fh, default=float)
        text = constructions_text(results, verdict)
        with open(pm, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(text)
        return 0
    if a.mode in ("baseline", "lagcontrol"):
        if not (a.records and a.histories and a.archive):
            raise SystemExit(f"REFUSED: --records, --histories and --archive are required for the {a.mode}")
        stem = "eval60_wobble_proximity_baseline" if a.mode == "baseline" else "eval60_wobble_lag_control"
        pj, pm = os.path.join(a.out_dir, f"{stem}{a.tag}.json"), os.path.join(a.out_dir, f"{stem}{a.tag}.md")
        refuse_existing([pj, pm])
        cons = check_lag_constructions() if a.mode == "lagcontrol" else None
        with open(a.archive, encoding="utf-8") as fh:
            arch_rec = json.load(fh)
        files = sorted(glob.glob(os.path.join(a.records, "eval60_regime_????_??.json.gz")))
        check_record_hashes({os.path.basename(p): sha256_of(p) for p in files}, arch_rec["generated_by"])
        recs = [SC.load_record(p) for p in files]
        SC.check_archive_identity(files, recs)
        by = {}
        for r in recs:
            by.setdefault(int(r["year"]), {})[int(r["month"])] = r
        _, cores_by = archive_results(by)
        with open(a.histories, encoding="utf-8") as fh:
            expected = stored_history_count(json.load(fh))
        hist = recompute_histories(by, expected)
        base = season_shift_baseline(hist, cores_by) if a.mode == "baseline" else lag_control(hist, cores_by)
        matched = matched_lag_comparison(hist, cores_by) if a.mode == "lagcontrol" else None
        check_observed(base, arch_rec["proximity"])
        gen.update({"records": {os.path.basename(p): sha256_of(p) for p in files}, "histories": {"path": a.histories, "sha256": sha256_of(a.histories)},
                    "archive": {"path": a.archive, "sha256": sha256_of(a.archive)}})
        with open(pj, "w", encoding="utf-8") as fh:
            json.dump({"generated_by": gen, "baseline": base} if a.mode == "baseline" else {"generated_by": gen, "constructions": cons, "lag_control": base, "matched": matched}, fh, default=float)
        text = baseline_text(base) if a.mode == "baseline" else lag_text(base, cons, matched)
        with open(pm, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(text)
        return 0
    if a.mode == "posthoc":
        if not (a.records and a.archive):
            raise SystemExit("REFUSED: --records and --archive are required for the post hoc reader")
        pj, pm = os.path.join(a.out_dir, f"eval60_wobble_posthoc{a.tag}.json"), os.path.join(a.out_dir, f"eval60_wobble_posthoc{a.tag}.md")
        refuse_existing([pj, pm])
        with open(a.archive, encoding="utf-8") as fh:
            arch_rec = json.load(fh)
        files = sorted(glob.glob(os.path.join(a.records, "eval60_regime_????_??.json.gz")))
        recs = [SC.load_record(p) for p in files]
        SC.check_archive_identity(files, recs)
        by = {}
        for r in recs:
            by.setdefault(int(r["year"]), {})[int(r["month"])] = r
        check_record_hashes({os.path.basename(p): sha256_of(p) for p in files}, arch_rec["generated_by"])
        arch, dailies = {}, {}
        for el, spec in ELEMENTS.items():
            cols = window_cols(spec["window"])
            for b in spec["bands"]:
                dailies[f"{el}|{b}"] = []
                arch[f"{el}|{b}"] = run_summary([A.season_hundredths(by[y], b)[1] for y in YEARS], cols, daily_out=dailies[f"{el}|{b}"])
        check_against_archive(arch, dailies, arch_rec["results"])
        cols = window_cols(ELEMENTS["S2"]["window"])
        cons = {c: run_summary([construction_season(c, i)[0] for i in range(len(YEARS))], cols) for c in CASES}
        orders = threshold_orders(arch_rec["results"])
        gen.update({"records": {os.path.basename(p): sha256_of(p) for p in files}, "archive": {"path": a.archive, "sha256": sha256_of(a.archive)}})
        with open(pj, "w", encoding="utf-8") as fh:
            json.dump({"generated_by": gen, "archive": arch, "constructions": cons, "threshold_orders": orders}, fh, default=float)
        text = posthoc_text(arch, cons, orders)
        with open(pm, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(text)
        return 0
    if not all((a.records, a.histories, a.constructions)):
        raise SystemExit("REFUSED: --records, --histories and --constructions are required for the archive")
    crec = check_constructions(a.constructions, floor=a.floor)
    floor = crec["results"]["W0"] if a.floor else None
    paths = {k: os.path.join(a.out_dir, v) for k, v in (("json", f"eval60_wobble_archive{a.tag}.json"), ("tables", f"eval60_wobble_tables{a.tag}.md"), ("figure", f"eval60_wobble_cores{a.tag}.png"))}
    refuse_existing(paths.values())
    files = sorted(glob.glob(os.path.join(a.records, "eval60_regime_????_??.json.gz")))
    recs = [SC.load_record(p) for p in files]
    SC.check_archive_identity(files, recs)
    by = {}
    for r in recs:
        by.setdefault(int(r["year"]), {})[int(r["month"])] = r
    results, cores_by = archive_results(by)
    with open(a.histories, encoding="utf-8") as fh:
        expected = stored_history_count(json.load(fh))
    hist = recompute_histories(by, expected)
    prox = proximity(hist, cores_by)
    draw(paths["figure"], results)
    gen.update({"records": {os.path.basename(p): sha256_of(p) for p in files}, "histories": {"path": a.histories, "sha256": sha256_of(a.histories)},
                "constructions": {"path": a.constructions, "sha256": sha256_of(a.constructions), "rule_verdict": crec["verdict"],
                                  "read_under": "section 8, W0's row as the stated floor" if a.floor else "section 3"}})
    with open(paths["json"], "w", encoding="utf-8") as fh:
        json.dump({"generated_by": gen, "results": results, "proximity": prox, "floor": floor, "figure": {"path": paths["figure"], "sha256": sha256_of(paths["figure"])}}, fh, default=float)
    text = archive_text(results, prox, floor)
    with open(paths["tables"], "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
