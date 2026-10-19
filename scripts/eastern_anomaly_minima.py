#!/usr/bin/env python3
"""Background removal for eastern minima (`EASTERN_BACKGROUND_REMOVAL_2026-10-19.md`,
section 2): minima of the daily-mean 700 hPa band mean after a background is subtracted,
with four backgrounds (B0 the other seasons' month mean, B1 the season's own month mean,
B2 centered running means of 5 and 9 days), scored first on synthetic constructions under
a pass rule and a selection rule fixed in the declaration, and applied to the 96 month
records only if a background is selected. An anomaly minimum is a minimum of the band mean
after a stated background is subtracted, and nothing is tracked, linked or explained.

    python3 scripts/eastern_anomaly_minima.py constructions --floor declared --out-dir DIR
    python3 scripts/eastern_anomaly_minima.py constructions --floor repaired --out-dir DIR
    python3 scripts/eastern_anomaly_minima.py archive --constructions PATH --records DIR \\
        --flag-ledger PATH --out-dir DIR
"""
import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eastern_standing_climatology as SC  # noqa: E402
import eastern_t1_ledger as T  # noqa: E402
import eastern_candidate_positions as CP  # noqa: E402
import eastern_standing_flag as F  # noqa: E402

YEARS = SC.YEARS
MONTHS = SC.MONTHS
LONS = np.arange(20.0, 75.0 + 1e-9, 0.5)
MONTH_DAYS = {6: 30, 7: 31, 8: 31, 9: 30}
SEASON_DAYS = sum(MONTH_DAYS.values())
BACKGROUNDS = ("B0", "B1", "B2_5", "B2_9")
WINDOWS = {"B2_5": 5, "B2_9": 9}
DECLARED_FLOOR = -2.0      # section 2, scored once in section 4
REPAIRED_FLOOR = 0.0       # section 5: the one repair, so that a prominence of 2 m/s means about 2 m/s below the zero line
FLOOR_RULES = {"declared": (DECLARED_FLOOR, "eval60_anom_constructions"), "repaired": (REPAIRED_FLOOR, "eval60_anom_constructions_r1")}
PROMINENCE = 2.0
NEAR_DEG = 5.0
TOL_DEG = 1.0
TOL_SENS_DEG = 2.0
REGION_DEG = 3.0
PRESENT = (24.0, 71.0)
PASS_DISPLACED = 0.10
PASS_RECOVERY = 0.80
RULE_SPEEDS = (2.0, 3.0, 5.0)
STANDING_DEPTH = 10.0
WIDTH_DEG = 2.0
STANDING_AT = 50.0
TRANSIENT_START_DAY = 35.0
TRANSIENT_FROM = 72.0
NOISE_SEED = 20261019
NOISE_STD = 2.0
NOISE_VARIANT_STD = 1.5
BIN_DEG = 2.0
EAST_OF = 40.0


def sha256_of(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def month_index():
    return np.array([m for m in MONTHS for _ in range(MONTH_DAYS[m])])


# ------------------------------------------------------------------ backgrounds

def month_means(D, month_of):
    """{month: the mean over the month's days of the daily series, per longitude}."""
    return {m: D[month_of == m].mean(axis=0) for m in MONTHS}


def loo_month_means(all_month_means, year):
    """{month: the mean over the other seasons of their month mean}, leaving `year` out."""
    others = [mm for y, mm in all_month_means.items() if y != year]
    return {m: np.mean([o[m] for o in others], axis=0) for m in MONTHS}


def background(kind, D, month_of, clim=None):
    """The background per day and longitude, NaN on days where it is unavailable. B0 needs
    `clim`, the leave-one-out month means. B2 is the centered running mean over its window,
    unavailable where the window leaves the season."""
    n = len(D)
    if kind == "B0":
        return np.array([clim[m] for m in month_of])
    if kind == "B1":
        mm = month_means(D, month_of)
        return np.array([mm[m] for m in month_of])
    if kind in WINDOWS:
        h = WINDOWS[kind] // 2
        out = np.full(D.shape, np.nan)
        for i in range(h, n - h):
            out[i] = D[i - h: i + h + 1].mean(axis=0)
        return out
    raise SystemExit(f"REFUSED: unknown background {kind}")


def anomaly_minima(D, Bg, lons, floor, context=""):
    """Per day, the prominent anomaly minima (anomaly at or below `floor`, prominence at
    least 2 m/s, measured by the candidate layer's rule) of the daily series minus the
    background, or None where the background is unavailable at any longitude."""
    out = []
    for i in range(len(D)):
        if not np.isfinite(Bg[i]).all():
            out.append(None)
            continue
        cs = CP.candidates_on(D[i] - Bg[i], lons, floor=floor, context=f"{context} day {i}")
        out.append([{"lon": float(c["lon"]), "depth": float(c["depth"]), "prominence": float(c["prominence"])} for c in cs if c["prominence"] >= PROMINENCE])
    return out


# ------------------------------------------------------------------ constructions

def gauss(x, center, depth, width=WIDTH_DEG):
    return -depth * np.exp(-0.5 * ((x - center) / width) ** 2)


def noise_field(std, seed=NOISE_SEED, n_steps=SEASON_DAYS * 4, lons=LONS, ar=0.9, width=WIDTH_DEG):
    """A seeded field correlated in longitude (Gaussian smoothing of `width` degrees) and in
    time (a six-hourly autoregressive coefficient `ar`), scaled to the standard deviation."""
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((n_steps, len(lons)))
    k = np.arange(-int(4 * width / 0.5), int(4 * width / 0.5) + 1) * 0.5
    ker = np.exp(-0.5 * (k / width) ** 2); ker /= ker.sum()
    S = np.array([np.convolve(row, ker, mode="same") for row in W])
    N = np.zeros_like(S); N[0] = S[0]
    for i in range(1, n_steps):
        N[i] = ar * N[i - 1] + np.sqrt(1.0 - ar ** 2) * S[i]
    return N / N.std() * std


def synth_season(standing=None, transient=None, noise=None, lons=LONS):
    """A synthetic season reduced to daily means by the same four-sample daily averaging rule
    as the real data (a reshape of complete regular samples, where the archive's
    `daily_means` also checks the hours and missing cells): six-hourly samples of the
    standing trough (a function of time in days giving its center, or None) and the
    transient ((start day, start longitude, speed in degrees per day, depth), or None),
    plus noise, rounded to the records' two decimals. Returns the daily series and the
    daily-mean centers of the standing feature and the transient (the mean of the four
    six-hourly positions, NaN on a day the feature is absent at any sample)."""
    V, s_c, tr_c = synth_samples(standing, transient, noise, lons)
    D = V.reshape(SEASON_DAYS, 4, -1).mean(axis=1)
    return D, s_c, tr_c


def synth_samples(standing=None, transient=None, noise=None, lons=LONS):
    """The six-hourly samples of `synth_season`, rounded to two decimals, with the daily-mean
    centers of the standing feature and the transient."""
    t = np.arange(SEASON_DAYS * 4) / 4.0
    V = np.zeros((len(t), len(lons)))
    s_pos = np.full(len(t), np.nan); tr_pos = np.full(len(t), np.nan)
    for k, tk in enumerate(t):
        if standing is not None:
            s_pos[k] = standing(tk); V[k] += gauss(lons, s_pos[k], STANDING_DEPTH)
        if transient is not None and tk >= transient[0]:
            tr_pos[k] = transient[1] - transient[2] * (tk - transient[0]); V[k] += gauss(lons, tr_pos[k], transient[3])
    if noise is not None:
        V = V + noise
    V = np.round(V, 2)
    return V, s_pos.reshape(SEASON_DAYS, 4).mean(axis=1), tr_pos.reshape(SEASON_DAYS, 4).mean(axis=1)


# ------------------------------------------------------------------ the exact stored-precision path

# Added 2026-10-19 on the review of the minima-histories effort (F1). The stored band means
# are hundredths of a m/s, so a daily mean is a quarter of an integer sum of hundredths and
# every background difference is an integer at a known scale. Minima found on those integers
# with the floor and the prominence scaled to match decide every comparison exactly, where
# floating averaging put an exact 2 m/s prominence at 1.9999999999999991. The float path above
# is kept unchanged, because the background-removal records were made with it.

def hundredths_from_samples(V):
    """Per day, the integer sum of four consecutive six-hourly samples in hundredths. The
    samples must lie on the hundredth lattice and be complete."""
    X = np.asarray(V, float) * 100.0
    if not np.isfinite(X).all():
        raise SystemExit("REFUSED: a missing sample has no exact daily sum")
    Hs = np.rint(X)
    if not np.allclose(Hs, X, atol=1e-6, rtol=0.0):
        raise SystemExit("REFUSED: samples are not on the hundredth lattice")
    return Hs.reshape(-1, 4, Hs.shape[-1]).sum(axis=1).astype(np.int64)


def synth_hundredths(standing=None, transient=None, noise=None, lons=LONS):
    """`synth_season` on the exact path: the daily sums in hundredths and the centers."""
    V, s_c, tr_c = synth_samples(standing, transient, noise, lons)
    return hundredths_from_samples(V), s_c, tr_c


def daily_hundredths(record, band):
    """{date: the integer sum in hundredths of the four stored six-hourly band-mean samples},
    with the ledger's check that a date holds exactly the hours 00, 06, 12 and 18."""
    S = np.array(record["band_means_700"][band], float)
    days = [t[:10] for t in record["times"]]; hours = [t[11:13] for t in record["times"]]
    out = {}
    for d in sorted(set(days)):
        idx = [i for i, t in enumerate(days) if t == d]
        if sorted(hours[i] for i in idx) != list(T.EXPECTED_HOURS):
            raise SystemExit(f"REFUSED: {d} does not hold exactly the four six-hourly samples")
        out[d] = hundredths_from_samples(S[idx])[0]
    return out


def season_hundredths(recs_by_month, band):
    """The dates and the daily sums in hundredths of one season and band, June to September."""
    dates, rows = [], []
    for m in MONTHS:
        dh = daily_hundredths(recs_by_month[m], band)
        for d in sorted(dh):
            dates.append(d); rows.append(dh[d])
    if len(dates) != SEASON_DAYS:
        raise SystemExit(f"REFUSED: {len(dates)} days in a season, not {SEASON_DAYS}")
    return dates, np.array(rows, dtype=np.int64)


def exact_anomaly(kind, H, month_of, other_month_sums=None):
    """Per day, the anomaly times its scale as integers (None where the background is
    unavailable), and the scale, so that the anomaly in m/s is the row over the scale. The
    daily mean is H / 400. 'zero' gives H at 400. B1 gives n H minus the month's sum at 400 n
    for a month of n days. A running mean of w days gives w H minus the window's sum at
    400 w. B0 gives k n H minus the other seasons' summed month sums at 400 k n for k other
    seasons, with `other_month_sums` {month: (summed month sums, k)}."""
    H = np.asarray(H, dtype=np.int64); n = len(H)
    if kind == "zero":
        return [H[i] for i in range(n)], [400] * n
    if kind == "B1":
        sums = {m: H[month_of == m].sum(axis=0) for m in MONTHS}; cnt = {m: int((month_of == m).sum()) for m in MONTHS}
        return [cnt[m] * H[i] - sums[m] for i, m in enumerate(month_of)], [400 * cnt[m] for m in month_of]
    if kind == "B0":
        out, sc = [], []
        for i, m in enumerate(month_of):
            tot, k = other_month_sums[m]; nm = int((month_of == m).sum())
            out.append(k * nm * H[i] - tot); sc.append(400 * nm * k)
        return out, sc
    if kind in WINDOWS:
        w = WINDOWS[kind]; h = w // 2
        rows = [None if (i < h or i >= n - h) else w * H[i] - H[i - h: i + h + 1].sum(axis=0) for i in range(n)]
        return rows, [400 * w] * n
    raise SystemExit(f"REFUSED: unknown background {kind}")


def exact_anomaly_minima(rows, scales, lons, floor, context=""):
    """`anomaly_minima` on the exact path. The candidate function runs on the integer rows
    with the floor times the scale, a minimum is prominent when its integer prominence is at
    least 2 m/s times the scale, and depth and prominence are reported in m/s."""
    out = []
    for i, (row, s) in enumerate(zip(rows, scales)):
        if row is None:
            out.append(None)
            continue
        cs = CP.candidates_on(np.asarray(row, dtype=float), lons, floor=floor * s, context=f"{context} day {i}")
        out.append([{"lon": float(c["lon"]), "depth": float(c["depth"]) / s, "prominence": float(c["prominence"]) / s} for c in cs if c["prominence"] >= PROMINENCE * s])
    return out


def share_near(mins_by_day, centers, near=NEAR_DEG):
    """The share of available days (background available, center defined) with a
    prominent anomaly minimum within `near` degrees of the center, and the day count."""
    n = k = 0
    for mins, c in zip(mins_by_day, centers):
        if mins is None or not np.isfinite(c):
            continue
        n += 1; k += any(abs(m["lon"] - c) <= near for m in mins)
    return (k / n if n else None), n


def recovery(mins_by_day, tr_centers, standing_at=None, tol=TOL_DEG):
    """The share of the transient's present days (center between 24 and 71 E, background
    available) with a prominent anomaly minimum within `tol` of its center, and the same
    over the in-region days (center within 3 degrees of `standing_at`), with counts."""
    n = k = nr = kr = 0
    for mins, c in zip(mins_by_day, tr_centers):
        if mins is None or not np.isfinite(c) or not (PRESENT[0] <= c <= PRESENT[1]):
            continue
        hit = any(abs(m["lon"] - c) <= tol for m in mins)
        n += 1; k += hit
        if standing_at is not None and abs(c - standing_at) <= REGION_DEG:
            nr += 1; kr += hit
    return {"present_days": n, "share": (k / n if n else None), "in_region_days": nr, "in_region_share": (kr / nr if nr else None)}


def construction_cases():
    """The declared cases, in order: (name, kind, synth arguments, meta)."""
    cases = []
    for shift in (52.0, 54.0):
        cases.append((f"C1 displaced to {shift:g} E", "displaced", {"standing": (lambda s: (lambda t: s))(shift)}, {"shift": shift}))
    cases.append(("C2 drift 48 to 52 E", "displaced", {"standing": lambda t: 48.0 + 4.0 * t / SEASON_DAYS}, {}))
    for amp in (1.5, 3.0):
        for per in (14.0, 28.0):
            cases.append((f"C3 wobble {amp:g} deg, {per:g} days", "wobble", {"standing": (lambda a, p: (lambda t: STANDING_AT + a * np.sin(2 * np.pi * t / p)))(amp, per)}, {"amplitude": amp, "period": per}))
    for which in ("C4", "C5"):
        for depth in (5.0, 10.0):
            for c in (1.0, 2.0, 3.0, 5.0):
                kw = {"transient": (TRANSIENT_START_DAY, TRANSIENT_FROM, c, depth)}
                if which == "C4":
                    kw["standing"] = lambda t: STANDING_AT
                cases.append((f"{which} depth {depth:g}, {c:g} deg/day", "passage", kw, {"over_standing": which == "C4", "depth": depth, "speed": c}))
    cases.append(("C0 noise alone", "noise", {"noise": noise_field(NOISE_STD)}, {}))
    nv = noise_field(NOISE_VARIANT_STD)
    cases.append(("C1 displaced to 52 E, with noise", "displaced", {"standing": lambda t: 52.0, "noise": nv}, {"shift": 52.0, "variant": True}))
    cases.append(("C4 depth 5, 2 deg/day, with noise", "passage", {"standing": lambda t: STANDING_AT, "transient": (TRANSIENT_START_DAY, TRANSIENT_FROM, 2.0, 5.0), "noise": nv},
                  {"over_standing": True, "depth": 5.0, "speed": 2.0, "variant": True}))
    return cases


def run_constructions(floor):
    month_of = month_index()
    D_clim, _, _ = synth_season(standing=lambda t: STANDING_AT)
    clim = month_means(D_clim, month_of)                 # the 23 other seasons are identical, so their mean is this
    out = {}
    for name, kind, kw, meta in construction_cases():
        D, s_c, tr_c = synth_season(**kw)
        res = {"kind": kind, "meta": {k: v for k, v in meta.items()}}
        for bg in BACKGROUNDS:
            mins = anomaly_minima(D, background(bg, D, month_of, clim), LONS, floor, context=f" ({name}, {bg})")
            if kind in ("displaced", "wobble"):
                sh, n = share_near(mins, s_c)
                res[bg] = {"share_near_standing": sh, "days": n}
            elif kind == "passage":
                res[bg] = recovery(mins, tr_c, STANDING_AT if meta["over_standing"] else None)
            else:
                avail = [m for m in mins if m is not None]
                res[bg] = {"minima_per_day": (sum(len(m) for m in avail) / len(avail) if avail else None), "days": len(avail)}
        out[name] = res
    return out


def apply_rule(results):
    """Per background, whether it passes the declared rule on the noise-free cases, with
    the failing cases named, and the selection (B1 if it passes, else the shorter B2 window
    that passes, else none)."""
    verdict = {}
    for bg in BACKGROUNDS:
        fails = []
        for name, r in results.items():
            if r["meta"].get("variant"):
                continue
            if r["kind"] == "displaced":
                if r[bg]["share_near_standing"] is None or r[bg]["share_near_standing"] > PASS_DISPLACED:
                    fails.append(name)
            elif r["kind"] == "passage" and r["meta"]["speed"] in RULE_SPEEDS:
                d = r[bg]
                if d["share"] is None or d["share"] < PASS_RECOVERY:
                    fails.append(name)
                elif r["meta"]["over_standing"] and (d["in_region_share"] is None or d["in_region_share"] < PASS_RECOVERY):
                    fails.append(name + " (in region)")
        verdict[bg] = {"passes": not fails, "failing_cases": fails}
    order = ["B1"] + sorted(WINDOWS, key=lambda k: WINDOWS[k])
    selected = next((bg for bg in order if verdict[bg]["passes"]), None)
    return verdict, selected


def constructions_text(results, verdict, selected, floor):
    f = lambda x: "" if x is None else f"{x:.2f}"
    L = [f"# background removal, the constructions, anomaly floor {floor:g} m/s",
         f"prominent anomaly minima are at or below {floor:g} m/s with prominence at least 2 m/s (the candidate layer's prominence); near is within 5 degrees of the standing center, recovered within 1 degree of the transient's center, in-region within 3 degrees of 50 E", ""]
    L += ["## the rule", "", "| background | passes | failing cases |", "| --- | --- | --- |"]
    for bg in BACKGROUNDS:
        L.append(f"| {bg} | {'yes' if verdict[bg]['passes'] else 'no'} | {', '.join(verdict[bg]['failing_cases']) or 'none'} |")
    L += ["", f"selected: {selected or 'none, so the effort stops after the constructions'}", ""]
    L += ["## displacement, drift and wobble, the share of available days with a prominent anomaly minimum near the standing center", "", "| case | " + " | ".join(BACKGROUNDS) + " |", "| --- |" + " --- |" * len(BACKGROUNDS)]
    for name, r in results.items():
        if r["kind"] in ("displaced", "wobble"):
            L.append(f"| {name} | " + " | ".join(f"{f(r[bg]['share_near_standing'])} ({r[bg]['days']})" for bg in BACKGROUNDS) + " |")
    L += ["", "## passage, the share of present days recovered (in-region share in brackets, day counts after)", "", "| case | " + " | ".join(BACKGROUNDS) + " |", "| --- |" + " --- |" * len(BACKGROUNDS)]
    for name, r in results.items():
        if r["kind"] == "passage":
            cells = []
            for bg in BACKGROUNDS:
                d = r[bg]
                cells.append(f"{f(d['share'])} [{f(d['in_region_share'])}] ({d['present_days']}, {d['in_region_days']})" if r["meta"]["over_standing"] else f"{f(d['share'])} ({d['present_days']})")
            L.append(f"| {name} | " + " | ".join(cells) + " |")
    L += ["", "## noise alone, prominent anomaly minima per available day", "", "| case | " + " | ".join(BACKGROUNDS) + " |", "| --- |" + " --- |" * len(BACKGROUNDS)]
    for name, r in results.items():
        if r["kind"] == "noise":
            L.append(f"| {name} | " + " | ".join(f"{f(r[bg]['minima_per_day'])} ({r[bg]['days']})" for bg in BACKGROUNDS) + " |")
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ the archive

def season_daily(recs_by_month, band):
    """The dates and the daily series of one season and band, June to September joined."""
    dates, rows = [], []
    for m in MONTHS:
        dm = T.daily_means(recs_by_month[m], band)
        for d in sorted(dm):
            dates.append(d); rows.append(dm[d])
    if len(dates) != SEASON_DAYS:
        raise SystemExit(f"REFUSED: {len(dates)} days in a season, not {SEASON_DAYS}")
    return dates, np.array(rows)


def archive_minima(records, floor):
    """{(year, band): (dates, {background: minima per day})} for every season and band."""
    by = {}
    for r in records:
        by.setdefault(int(r["year"]), {})[int(r["month"])] = r
    bands = list(records[0]["bands"]); lons = np.array(records[0]["lons"], float); month_of = month_index()
    series = {(y, b): season_daily(by[y], b) for y in YEARS for b in bands}
    allmm = {b: {y: month_means(series[(y, b)][1], month_of) for y in YEARS} for b in bands}
    out = {}
    for (y, b), (dates, D) in series.items():
        clim = loo_month_means(allmm[b], y)
        out[(y, b)] = (dates, {bg: anomaly_minima(D, background(bg, D, month_of, clim), lons, floor, context=f" ({y}, {b}, {bg})") for bg in BACKGROUNDS})
    return out


def category(c):
    if c["flags"]["primary"]:
        return "on the mask"
    if c["mask_distance_deg"] is None:
        return "no mask in the band-month"
    return "off, within 5 degrees" if c["mask_distance_deg"] <= NEAR_DEG else "off, farther"


CATEGORIES = ("on the mask", "off, within 5 degrees", "off, farther", "no mask in the band-month")


def survival(raw, amin, tol):
    """For the raw prominent minima at or east of 40 E, per background, band and flag
    category, the count, the count on days with the background available, and the count
    with a prominent anomaly minimum within `tol` degrees on the same day."""
    index = {}
    for (y, b), (dates, per_bg) in amin.items():
        for i, d in enumerate(dates):
            index[(d, b)] = {bg: per_bg[bg][i] for bg in BACKGROUNDS}
    out = {bg: {} for bg in BACKGROUNDS}
    for c in raw:
        if c["prominence"] < F.PROMINENCE_CLASSES[F.PRIMARY_CLASS] or c["lon"] < EAST_OF:
            continue
        cat = category(c)
        for bg in BACKGROUNDS:
            d = out[bg].setdefault(c["band"], {}).setdefault(cat, {"count": 0, "available": 0, "survived": 0})
            d["count"] += 1
            mins = index[(c["date"], c["band"])][bg]
            if mins is None:
                continue
            d["available"] += 1
            d["survived"] += any(abs(m["lon"] - c["lon"]) <= tol for m in mins)
    return out


def ledger_survival(anchors, amin, tol=TOL_DEG):
    index = {}
    for (y, b), (dates, per_bg) in amin.items():
        for i, d in enumerate(dates):
            index[(d, b)] = {bg: per_bg[bg][i] for bg in BACKGROUNDS}
    out = {bg: {} for bg in BACKGROUNDS}
    for a in anchors:
        for bg in BACKGROUNDS:
            d = out[bg].setdefault(a["class"], {"entry_days": 0, "available": 0, "survived": 0})
            d["entry_days"] += 1
            mins = index[(a["date"], a["band"])][bg]
            if mins is None:
                continue
            d["available"] += 1
            d["survived"] += any(abs(m["lon"] - a["anchor_lon"]) <= tol for m in mins)
    return out


def recurrence(amin, masks, lons):
    """Per background and band, the prominent anomaly minima per available band-day in
    two-degree bins, and the leave-one-out band-day northerly fraction at each (median and
    tenths), from the flag's graded value."""
    edges = np.arange(20.0, 76.0 + 1e-9, BIN_DEG)
    out = {}
    for (y, b), (dates, per_bg) in amin.items():
        for bg in BACKGROUNDS:
            d = out.setdefault(bg, {}).setdefault(b, {"band_days": 0, "hist": np.zeros(len(edges) - 1, int), "graded": []})
            for i, mins in enumerate(per_bg[bg]):
                if mins is None:
                    continue
                d["band_days"] += 1
                m = int(dates[i][5:7])
                val = masks.band_value(y, m, b)
                for x in mins:
                    k = min(int(np.searchsorted(edges, x["lon"], side="right") - 1), len(edges) - 2)
                    d["hist"][k] += 1
                    d["graded"].append(float(val[F.index_of(lons, x["lon"], "anomaly minimum longitude")]))
    res = {"bin_edges_deg": edges.tolist(), "by": {}}
    for bg, bands in out.items():
        for b, d in bands.items():
            g = np.array(d["graded"]); g = g[np.isfinite(g)]
            res["by"].setdefault(bg, {})[b] = {"band_days": d["band_days"], "minima": int(d["hist"].sum()), "per_band_day": (d["hist"] / max(d["band_days"], 1)).tolist(),
                                               "graded_median": (float(np.median(g)) if g.size else None), "graded_tenths": F.tenths_histogram(g)}
    return res


def archive_text(selected, surv, surv2, led, rec, floor):
    pct = lambda d: "" if d["available"] == 0 else f"{d['survived'] / d['available']:.2f}"
    L = ["# background removal, the archive (section 2)", f"selected background {selected}, anomaly floor {floor:g} m/s; every background reported. A raw prominent minimum (prominence at least 2 m/s, at or east of 40 E) survives when a prominent anomaly minimum lies within the tolerance on the same day; the share is over the days with the background available", ""]
    for tol_name, sv in ((f"{TOL_DEG:g} degree", surv), (f"{TOL_SENS_DEG:g} degrees (sensitivity)", surv2)):
        L += [f"## raw prominent minima surviving within {tol_name}, by band and flag category (each share over its own background's available days)", "", "| band | category | count | available, " + ", ".join(BACKGROUNDS) + " | " + " | ".join(f"share {bg}" for bg in BACKGROUNDS) + " |", "| --- |" + " --- |" * (3 + len(BACKGROUNDS))]
        for b in sv[selected]:
            for cat in CATEGORIES:
                if cat not in sv[selected][b]:
                    continue
                d0 = sv[selected][b][cat]
                L.append(f"| {b} | {cat} | {d0['count']} | " + ", ".join(str(sv[bg][b][cat]["available"]) for bg in BACKGROUNDS) + " | " + " | ".join(pct(sv[bg][b][cat]) for bg in BACKGROUNDS) + " |")
        L.append("")
    L += ["## the ledger's anchored entry-days with a prominent anomaly minimum within 1 degree, by the entry's class (entry-weighted, descriptive)", "", "| class | entry-days | available, " + ", ".join(BACKGROUNDS) + " | " + " | ".join(f"share {bg}" for bg in BACKGROUNDS) + " |", "| --- |" + " --- |" * (2 + len(BACKGROUNDS))]
    for cls in sorted(led[selected]):
        d0 = led[selected][cls]
        L.append(f"| {cls} | {d0['entry_days']} | " + ", ".join(str(led[bg][cls]["available"]) for bg in BACKGROUNDS) + " | " + " | ".join(pct(led[bg][cls]) for bg in BACKGROUNDS) + " |")
    L += ["", "## prominent anomaly minima per available band-day, and the leave-one-out northerly fraction at them", "", "| background | band | band-days | minima | per band-day | graded median | graded tenths |", "| --- |" + " --- |" * 6]
    for bg in BACKGROUNDS:
        for b, d in rec["by"][bg].items():
            gm = "" if d["graded_median"] is None else f"{d['graded_median']:.2f}"
            L.append(f"| {bg} | {b} | {d['band_days']} | {d['minima']} | {d['minima'] / max(d['band_days'], 1):.2f} | {gm} | {', '.join(str(x) for x in d['graded_tenths'])} |")
    return "\n".join(L) + "\n"


def draw_recurrence(path, rec, raw_per_bin, selected):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    edges = np.array(rec["bin_edges_deg"]); mid = (edges[:-1] + edges[1:]) / 2
    bands = list(rec["by"][selected])
    fig, axes = plt.subplots(1, len(bands), figsize=(4.6 * len(bands) + 1, 4.6), sharey=True, constrained_layout=True)
    for ax, b in zip(np.atleast_1d(axes), bands):
        ax.plot(mid, raw_per_bin[b], color="k", lw=1.5, label="raw prominent minima")
        for bg, ls in zip(BACKGROUNDS, ("-", "-", "--", ":")):
            ax.plot(mid, rec["by"][bg][b]["per_band_day"], lw=1.6 if bg == selected else 1.0, ls=ls, label=f"anomaly minima, {bg}" + (" (selected)" if bg == selected else ""))
        for x in (40.0, 60.0):
            ax.axvline(x, color="k", lw=0.8, ls="--")
        ax.set_title(b, fontsize=11); ax.set_xlabel("longitude (degrees E)"); ax.grid(True, color="0.8", lw=0.4)
    np.atleast_1d(axes)[0].set_ylabel("prominent minima per band-day"); np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.suptitle(f"Prominent minima of the daily-mean 700 hPa band mean, raw and after each background is subtracted, {len(YEARS)} seasons (2005 closed), two-degree bins, per available band-day. Positions only, nothing linked across days.", fontsize=9, wrap=True)
    fig.savefig(path, dpi=100); plt.close(fig)


# ------------------------------------------------------------------ main

def refuse_existing(paths):
    for p in paths:
        if os.path.exists(p):
            raise SystemExit(f"REFUSED: {p} exists and a record is never overwritten")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("constructions", "archive"))
    ap.add_argument("--out-dir", required=True); ap.add_argument("--constructions", default=None); ap.add_argument("--records", default=None); ap.add_argument("--flag-ledger", default=None)
    ap.add_argument("--floor", choices=tuple(FLOOR_RULES), default=None, help="the anomaly floor of the constructions: declared (-2 m/s, section 2) or repaired (0 m/s, section 5)")
    a = ap.parse_args(argv)
    gen = {"script": "scripts/eastern_anomaly_minima.py", "sha256": sha256_of(os.path.abspath(__file__)), "declared": "EASTERN_BACKGROUND_REMOVAL_2026-10-19.md"}
    if a.mode == "constructions":
        if a.floor is None:
            raise SystemExit("REFUSED: --floor declared or repaired is required for the constructions")
        floor, stem = FLOOR_RULES[a.floor]
        pj, pm = os.path.join(a.out_dir, stem + ".json"), os.path.join(a.out_dir, stem + ".md")
        refuse_existing([pj, pm])
        results = run_constructions(floor); verdict, selected = apply_rule(results)
        with open(pj, "w", encoding="utf-8") as fh:
            json.dump({"generated_by": gen, "floor_rule": a.floor, "anomaly_floor": floor, "results": results, "verdict": verdict, "selected": selected}, fh, default=float)
        text = constructions_text(results, verdict, selected, floor)
        with open(pm, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(text)
        return 0
    if not (a.constructions and a.records and a.flag_ledger):
        raise SystemExit("REFUSED: --constructions, --records and --flag-ledger are required for the archive")
    with open(a.constructions, encoding="utf-8") as fh:
        cons = json.load(fh)
    selected = cons.get("selected")
    if not selected:
        raise SystemExit("REFUSED: no background passed the constructions, so the effort stops before the archive")
    if "anomaly_floor" not in cons:
        raise SystemExit("REFUSED: the constructions record names no anomaly floor, so the archive cannot use the floor its selection was made under")
    floor = float(cons["anomaly_floor"])
    paths = {k: os.path.join(a.out_dir, v) for k, v in (("json", "eval60_anom_archive.json"), ("tables", "eval60_anom_tables.md"), ("figure", "eval60_anom_recurrence.png"))}
    refuse_existing(paths.values())
    files = sorted(glob.glob(os.path.join(a.records, "eval60_regime_????_??.json.gz")))
    records = [SC.load_record(p) for p in files]
    if len(records) != len(YEARS) * len(MONTHS):
        raise SystemExit(f"REFUSED: {len(records)} month records, not {len(YEARS) * len(MONTHS)}")
    SC.check_archive_identity(files, records)
    lons = np.array(records[0]["lons"], float)
    amin = archive_minima(records, floor)
    masks = F.Masks(records)
    raw = F.flag_candidates(records, masks)
    surv = survival(raw, amin, TOL_DEG); surv2 = survival(raw, amin, TOL_SENS_DEG)
    with open(a.flag_ledger, encoding="utf-8") as fh:
        anchors = json.load(fh)["anchors"]
    led = ledger_survival(anchors, amin)
    rec = recurrence(amin, masks, lons)
    ct = F.candidate_tables(raw, records)
    raw_per_bin = {b: (np.array([c["count"] for c in ct["by_bin"][F.PRIMARY_CLASS][b]["all"]]) / ct["band_days"][b]["all"]).tolist() for b in records[0]["bands"]}
    draw_recurrence(paths["figure"], rec, raw_per_bin, selected)
    gen.update({"constructions": {"path": a.constructions, "sha256": sha256_of(a.constructions)}, "month_records": {os.path.basename(p): sha256_of(p) for p in files},
                "flag_ledger": {"path": a.flag_ledger, "sha256": sha256_of(a.flag_ledger)}})
    with open(paths["json"], "w", encoding="utf-8") as fh:
        json.dump({"generated_by": gen, "selected": selected, "anomaly_floor": floor, "survival_1deg": surv, "survival_2deg": surv2, "ledger": led, "recurrence": rec, "raw_per_band_day": raw_per_bin,
                   "figure": {"path": paths["figure"], "sha256": sha256_of(paths["figure"])}}, fh, default=float)
    text = archive_text(selected, surv, surv2, led, rec, floor)
    with open(paths["tables"], "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
