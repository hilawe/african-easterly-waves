#!/usr/bin/env python3
"""Westward histories across 40 E and the western record (`EASTERN_CROSSING_TEST_2026-10-19.md`,
section 2): the 700 hPa band means extended west to 10 W by the characterization's rule,
the minima histories followed across 40 E on the exact stored-precision path, constructions
of passages across 40 E and of a hand-off, and the eastern-origin westward histories tested
for coincidence with the paper's catalog and with QTrack, and for preceding the records'
easternmost tracks, each against a season-swapped chance level. A meeting is a coincidence
of positions and dates. Nothing is identified, connected or explained.

    python3 scripts/eastern_crossing_test.py render --out-dir DIR
    python3 scripts/eastern_crossing_test.py constructions --out-dir DIR
    python3 scripts/eastern_crossing_test.py archive --xband DIR --records DIR --catalog GLOB \\
        --qtrack DIR --constructions PATH --out-dir DIR
"""
import argparse
import datetime as dt
import glob
import gzip
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
import eastern_candidate_positions as CP  # noqa: E402
import eastern_standing_climatology as SC  # noqa: E402
import pilot_eastern_propagation as P  # noqa: E402

YEARS = SC.YEARS
MONTHS = SC.MONTHS
XLON = (-10.0, 75.0)
XLONS = np.arange(XLON[0], XLON[1] + 1e-9, 0.5)
EAST_ORIGIN = 45.0
REACH = (42.0, 40.0, 35.0, 30.0, 20.0, 10.0, 0.0)
MEET_EAST_LIMIT = 42.0
LON_TOL = 2.0
LAT_PAD = 2.5
PRECEDE_FIRST_LON = 30.0
COAST_PROXY = -15.0
PRESENT_X = (XLON[0] + 3.0, 71.0)
SPAN_EAST, SPAN_WEST = 45.0, 30.0
TOL_DEG = 1.0
PASS_RECOVERY = 0.80
CHANNEL_AT, SOUTHERLY_AT = 41.0, 33.0
SENSITIVITY = {"meet_lon_tol": 4.0, "precede_lon_tol": 5.0, "precede_back_days": 2}     # added after the archive was first read (section 5 of the record)


def sha256_of(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


# ------------------------------------------------------------------ the extended band means

def load_month_v700_sp(year, month, data_dir=P.DATA_DIR):
    """The six-hourly 700 hPa meridional wind and surface pressure of a calendar month over the
    characterization's latitudes and 10 W to 75 E, with the renderer's checks (held-out
    season refused, the month contiguous, the axes equal across files)."""
    import netCDF4
    if year == P.HELD_OUT:
        raise SystemExit("REFUSED: the held-out season is never opened")
    start, end = V.month_window(year, month)
    out, ref = {"year": year, "month": month}, None
    for name, var in (("v700", "v"), ("sp", "sp")):
        path = os.path.join(data_dir, f"era5_{name}_{year}_6h_region.nc")
        d = netCDF4.Dataset(path)
        try:
            t = np.asarray(d.variables["valid_time"][:], float)
            times = [dt.datetime(1970, 1, 1) + dt.timedelta(seconds=float(s)) for s in t]
            sel = np.flatnonzero([(start <= x <= end) for x in times])
            if sel.size == 0 or (sel[-1] - sel[0] + 1) != sel.size:
                raise SystemExit(f"REFUSED: {path} does not hold the month contiguously")
            lat = np.asarray(d.variables["latitude"][:], float); lon = np.asarray(d.variables["longitude"][:], float)
            rows = np.flatnonzero((lat >= V.BOX_LAT[0] - 1e-6) & (lat <= V.BOX_LAT[1] + 1e-6)); cols = np.flatnonzero((lon >= XLON[0] - 1e-6) & (lon <= XLON[1] + 1e-6))
            if rows.size == 0 or cols.size == 0 or (rows[-1] - rows[0] + 1) != rows.size or (cols[-1] - cols[0] + 1) != cols.size:
                raise SystemExit(f"REFUSED: {path} does not hold the box contiguously")
            a = d.variables[var]
            if a.ndim == 4:
                if abs(float(d.variables["pressure_level"][0]) - 700.0) > 1e-6:
                    raise SystemExit(f"REFUSED: {path} is not at 700 hPa")
                arr = np.asarray(a[sel[0]:sel[-1] + 1, 0, rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1], float)
            else:
                arr = np.asarray(a[sel[0]:sel[-1] + 1, rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1], float)
            if ref is None:
                ref = t[sel]; out["times"] = [times[i] for i in sel]; out["lat"] = lat[rows]; out["lon"] = lon[cols]
            elif not np.array_equal(t[sel], ref) or not np.array_equal(lat[rows], out["lat"]) or not np.array_equal(lon[cols], out["lon"]):
                raise SystemExit(f"REFUSED: {path} has a different time or space axis from the first file")
            out[name] = arr
            out.setdefault("files", {})[name] = {"path": path, "sha256": sha256_of(path)}
        finally:
            d.close()
    if not np.allclose(out["lon"], XLONS):
        raise SystemExit("REFUSED: the extended longitudes are not 10 W to 75 E at 0.5 degrees")
    return out


SCRIPT_SHA256_AT_LOAD = sha256_of(os.path.abspath(__file__))


def xband_record(M):
    """The extended month record: the band means at 700 hPa by the characterization's rule,
    stored to two decimals, with the digest of this script as it was when loaded."""
    rows = V.band_rows(M["lat"])
    means = V.band_means(M["v700"], M["sp"], 700, rows)
    return {"year": M["year"], "month": M["month"], "lons": M["lon"].tolist(), "times": [t.strftime("%Y-%m-%dT%H") for t in M["times"]],
            "bands": {b: list(v) for b, v in V.BANDS.items()},
            "band_means_700": {b: np.round(means[b]["series"], 2).tolist() for b in V.BANDS},
            "unavailable": {b: means[b]["unavailable"] for b in V.BANDS}, "files": M["files"],
            "generated_by": {"script": "scripts/eastern_crossing_test.py", "sha256_at_load": SCRIPT_SHA256_AT_LOAD, "declared": "EASTERN_CROSSING_TEST_2026-10-19.md"}}


def render_all(out_dir, years=YEARS):
    made = []
    for y in years:
        for m in MONTHS:
            path = os.path.join(out_dir, f"eval60_xband_{y}_{m:02d}.json.gz")
            if os.path.exists(path):
                continue
            rec = xband_record(load_month_v700_sp(y, m))
            with gzip.open(path, "wt", encoding="utf-8") as fh:
                json.dump(rec, fh)
            made.append((y, m))
    return made


def check_xband_identity(files, records):
    """The extended archive is the declared seasons times months once each, every file named
    for the record it holds, on one longitude grid and band set."""
    expected = sorted((y, m) for y in YEARS for m in MONTHS)
    actual = sorted((int(r["year"]), int(r["month"])) for r in records)
    if actual != expected:
        raise SystemExit(f"REFUSED: the extended archive is not the declared months once each: missing {sorted(set(expected) - set(actual))}, unexpected {sorted(set(actual) - set(expected))}")
    for path, r in zip(files, records):
        if os.path.basename(path) != f"eval60_xband_{r['year']}_{r['month']:02d}.json.gz":
            raise SystemExit(f"REFUSED: {path} holds the record of {r['year']}-{r['month']:02d}")
        if r["lons"] != records[0]["lons"] or r["bands"] != records[0]["bands"]:
            raise SystemExit(f"REFUSED: {r['year']}-{r['month']:02d} is not on the first record's grid and band set")


def check_against_climatology(xrec, crec):
    """The extended record's band means inside 20 to 75 E against the climatology record's,
    both stored to two decimals, equal with NaN equal."""
    xl = np.array(xrec["lons"]); cl = np.array(crec["lons"])
    cols = np.flatnonzero((xl >= cl[0] - 1e-9) & (xl <= cl[-1] + 1e-9))
    if not np.allclose(xl[cols], cl) or xrec["times"] != crec["times"]:
        return False
    return all(np.array_equal(np.array(xrec["band_means_700"][b], float)[:, cols], np.array(crec["band_means_700"][b], float), equal_nan=True) for b in crec["bands"])


# ------------------------------------------------------------------ gaps and minima on the extended grid

def season_hundredths_with_gaps(recs_by_month, band):
    """The dates, the daily sums in hundredths with gap columns set to zero, and the gap
    mask: a longitude with any missing sample in the season is a gap for the season."""
    dates, rows, gap = [], [], None
    for m in MONTHS:
        r = recs_by_month[m]
        S = np.array(r["band_means_700"][band], float)
        miss = ~np.isfinite(S).all(axis=0)
        gap = miss if gap is None else (gap | miss)
    for m in MONTHS:
        r = recs_by_month[m]
        S = np.array(r["band_means_700"][band], float)
        S[:, gap] = 0.0
        rr = dict(r, band_means_700={band: S.tolist()})
        dh = A.daily_hundredths(rr, band)
        for d in sorted(dh):
            dates.append(d); rows.append(dh[d])
    if len(dates) != A.SEASON_DAYS:
        raise SystemExit(f"REFUSED: {len(dates)} days in a season, not {A.SEASON_DAYS}")
    return dates, np.array(rows, dtype=np.int64), gap


def segment_minima(rows, scales, lons, gap, floor):
    """Prominent minima found separately in each run of complete longitudes, a gap acting as
    a box edge, so a minimum is never placed at a gap and no ridge spans one."""
    gap = np.zeros(len(lons), bool) if gap is None else np.asarray(gap, bool)
    runs, start = [], None
    for i, g in enumerate(list(gap) + [True]):
        if not g and start is None:
            start = i
        if g and start is not None:
            runs.append((start, i)); start = None
    out = []
    for i, (row, s) in enumerate(zip(rows, scales)):
        if row is None:
            out.append(None)
            continue
        day = []
        for a, b in runs:
            for c in CP.candidates_on(np.asarray(row[a:b], dtype=float), lons[a:b], floor=floor * s, context=f" day {i}"):
                if c["prominence"] >= A.PROMINENCE * s:
                    day.append({"lon": float(c["lon"]), "depth": float(c["depth"]) / s, "prominence": float(c["prominence"]) / s})
        out.append(sorted(day, key=lambda q: q["lon"]))
    return out


def rep_minima(rep, H, month_of, lons, gap=None):
    kind, floor = HI.REP_SPEC[rep]
    rows, scales = A.exact_anomaly(kind, H, month_of)
    return segment_minima(rows, scales, lons, gap, floor)


# ------------------------------------------------------------------ constructions

def path_positions(x0, t0, speed_of_x, t_end=A.SEASON_DAYS, dt_day=0.01):
    """{time: longitude} on the six-hourly samples for a transient that starts at x0 at t0 and
    moves west at speed_of_x(x) degrees per day, integrated in steps of dt_day."""
    out, x, t = {}, float(x0), float(t0)
    targets = [k / 4.0 for k in range(int(round(t0 * 4)), int(t_end * 4))]
    for tk in targets:
        while t + 1e-12 < tk:
            step = min(dt_day, tk - t); x -= speed_of_x(x) * step; t += step
        out[tk] = x
    return out


def speed_profile(top):
    return lambda x: 2.0 if x >= 45.0 else (top if x <= 35.0 else 2.0 + (top - 2.0) * (45.0 - x) / 10.0)


def synth_extended(transients, lons=XLONS):
    """Six-hourly samples on the extended grid with the fixed channel trough at 41 E (depth
    10, width 2) and the fixed southerly region at 33 E (+6 m/s, width 3), plus transients,
    each (path {time: lon}, depth_of(lon, time)), rounded to two decimals. Returns the daily
    sums in hundredths and each transient's daily-mean center."""
    t = np.arange(A.SEASON_DAYS * 4) / 4.0
    V_ = np.zeros((len(t), len(lons)))
    V_ += A.gauss(lons, CHANNEL_AT, A.STANDING_DEPTH)[None, :] + A.gauss(lons, SOUTHERLY_AT, -6.0, width=3.0)[None, :]
    centers = []
    for path, depth_of in transients:
        pos = np.full(len(t), np.nan)
        for k, tk in enumerate(t):
            if tk in path:
                dep = depth_of(path[tk], tk)
                if dep > 0:
                    pos[k] = path[tk]; V_[k] += A.gauss(lons, path[tk], dep)
        centers.append(pos.reshape(A.SEASON_DAYS, 4).mean(axis=1))
    V_ = np.round(V_, 2)
    return A.hundredths_from_samples(V_), centers


def construction_cases():
    cases = []
    for depth in (5.0, 10.0):
        cases.append((f"X1 constant 2 deg/day, depth {depth:g}", "passage", [(path_positions(72.0, 35.0, lambda x: 2.0), (lambda d: (lambda x, tt: d))(depth))], {"depth": depth}))
        cases.append((f"X2 rising to 6 deg/day, depth {depth:g}", "passage", [(path_positions(72.0, 35.0, speed_profile(6.0)), (lambda d: (lambda x, tt: d))(depth))], {"depth": depth}))
    cases.append(("X3 rising to 7 deg/day, depth 10", "limit", [(path_positions(72.0, 35.0, speed_profile(7.0)), lambda x, tt: 10.0)], {"depth": 10.0}))
    a_path = path_positions(72.0, 35.0, lambda x: 2.0)
    a_depth = lambda x, tt: 10.0 if x >= 46.0 else (10.0 * (x - 42.0) / 4.0 if x > 42.0 else 0.0)
    t_vanish = min(tk for tk, x in a_path.items() if x <= 42.0)
    b_path = path_positions(38.0, float(np.floor(t_vanish)) + 1.0, lambda x: 6.0)
    cases.append(("X4 hand-off at 42 and 38 E", "handoff", [(a_path, a_depth), (b_path, lambda x, tt: 10.0)], {"vanish_day": t_vanish}))
    c_path = path_positions(41.0, t_vanish, lambda x: 6.0)          # added after the declaration (section 4 of the record)
    cases.append(("X5 hand-off on consecutive days, 41 E", "handoff", [(a_path, a_depth), (c_path, lambda x, tt: 10.0)], {"vanish_day": t_vanish, "added": True}))
    return cases


def score_passage(hs, cls, center, mins):
    west = [h for h, c in zip(hs, cls) if c == "westward"]
    n = k = 0
    for d, c in enumerate(center):
        if mins[d] is None or not np.isfinite(c) or not (PRESENT_X[0] <= c <= PRESENT_X[1]):
            continue
        n += 1; k += any(abs(x - c) <= TOL_DEG for h in west for dd, x in zip(h["days"], h["lons"]) if dd == d)
    span = any(any(np.isfinite(center[d]) and center[d] >= SPAN_EAST and abs(x - center[d]) <= TOL_DEG for d, x in zip(h["days"], h["lons"]))
               and any(np.isfinite(center[d]) and center[d] <= SPAN_WEST and abs(x - center[d]) <= TOL_DEG for d, x in zip(h["days"], h["lons"])) for h in west)
    return {"present_days": n, "share": (k / n if n else None), "spanning": bool(span)}


def score_handoff(hs, cls, c_a, c_b):
    joined = any(any(np.isfinite(c_a[d]) and c_a[d] >= SPAN_EAST and abs(x - c_a[d]) <= TOL_DEG for d, x in zip(h["days"], h["lons"]))
                 and any(np.isfinite(c_b[d]) and c_b[d] <= SPAN_WEST and abs(x - c_b[d]) <= TOL_DEG for d, x in zip(h["days"], h["lons"]))
                 for h, c in zip(hs, cls) if c == "westward")
    return {"joined": bool(joined)}


def run_constructions():
    month_of = A.month_index(); out = {}
    for name, kind, trans, meta in construction_cases():
        H, centers = synth_extended(trans)
        res = {"kind": kind, "meta": meta}
        for rep in HI.REPS:
            mins = rep_minima(rep, H, month_of, XLONS)
            for linker in HI.LINKERS:
                hs = HI.link(mins, linker); cls = [HI.classify(h) for h in hs]
                res[HI.combo(rep, linker)] = score_handoff(hs, cls, centers[0], centers[1]) if kind == "handoff" else score_passage(hs, cls, centers[0], mins)
        out[name] = res
    return out


def apply_rule(results):
    verdict = {}
    for rep, linker in HI.ORDER:
        k = HI.combo(rep, linker)
        fails = [n for n, r in results.items() if r["kind"] == "passage" and (r[k]["share"] is None or r[k]["share"] < PASS_RECOVERY or not r[k]["spanning"])]
        verdict[k] = {"passes": not fails, "failing_cases": fails}
    return verdict


def constructions_text(results, verdict):
    keys = [HI.combo(r, l) for r, l in HI.ORDER]
    f = lambda x: "" if x is None else f"{x:.2f}"
    L = ["# crossing test, the constructions (section 2)", "fixed channel trough at 41 E and southerly region at 33 E in every case; a passage is recovered on a present day when a westward history has a member within 1 degree of its center, and spans when one westward history has such members at or east of 45 E and at or west of 30 E", "",
         "## the rule (X1 and X2 recovered on 0.80 of present days with a spanning history)", "", "| combination | passes | failing cases |", "| --- | --- | --- |"]
    for k in keys:
        L.append(f"| {k} | {'yes' if verdict[k]['passes'] else 'no'} | {', '.join(verdict[k]['failing_cases']) or 'none'} |")
    L += ["", "## passages, share recovered (spanning in brackets) and the hand-off (joined or not)", "", "| case | " + " | ".join(keys) + " |", "| --- |" + " --- |" * len(keys)]
    for name, r in results.items():
        if r["kind"] == "handoff":
            L.append(f"| {name} | " + " | ".join("joined" if r[k]["joined"] else "not joined" for k in keys) + " |")
        else:
            L.append(f"| {name} | " + " | ".join(f"{f(r[k]['share'])} [{'spans' if r[k]['spanning'] else 'no span'}]" for k in keys) + " |")
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ the western records

def catalog_points(paths):
    """The paper catalog's de-duplicated troughs, {date: [(track, lat, lon)]} and
    {track: (first datetime, first lat, first lon, westernmost lon)}."""
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
    from aew.data.aewc import load_aewc_troughs
    tr = load_aewc_troughs(paths, dedup=True)
    wid = tr.variables["wave_id"]
    times = tr.time.astype("datetime64[m]").astype(object)
    by_date, tracks = {}, {}
    order = np.argsort(tr.time, kind="stable")
    for i in order:
        k = ("catalog", int(wid[i])); t = times[i]; la, lo = float(tr.lat[i]), float(tr.lon[i])
        by_date.setdefault(t.strftime("%Y-%m-%d"), []).append((k, la, lo))
        if k not in tracks:
            tracks[k] = [t, la, lo, lo]
        tracks[k][3] = min(tracks[k][3], lo)
    return by_date, {k: tuple(v) for k, v in tracks.items()}


def decode_qtrack_times(values, units, calendar, year):
    """The file's time axis decoded by its own units and calendar (review fold, section 8 of
    the record), refused unless every value is present, the axis strictly increases and every
    time falls in the season the filename names."""
    import netCDF4
    if not units or np.ma.count_masked(values):
        raise SystemExit(f"REFUSED: a QTrack time axis for {year} has no units or a missing value")
    v = np.asarray(np.ma.getdata(values), float)
    if v.size and not np.all(np.diff(v) > 0):
        raise SystemExit(f"REFUSED: the QTrack time axis for {year} does not strictly increase")
    w = netCDF4.num2date(v, units, calendar or "standard", only_use_cftime_datetimes=False, only_use_python_datetimes=True)
    when = [dt.datetime(t.year, t.month, t.day, t.hour, t.minute, t.second) for t in np.atleast_1d(w)]
    if any(t.year != year for t in when):
        raise SystemExit(f"REFUSED: the QTrack time axis for {year} holds times outside that season")
    return when


def qtrack_points(paths):
    """QTrack's raw positions (AEW_lon, AEW_lat) on the file's time axis, in the same two
    structures, one track per system and year."""
    import netCDF4
    by_date, tracks = {}, {}
    for p in paths:
        d = netCDF4.Dataset(p)
        try:
            tv = d.variables["time"]
            when = decode_qtrack_times(tv[:], getattr(tv, "units", None), getattr(tv, "calendar", None), int(os.path.basename(p)[-7:-3]))
            lon = np.ma.filled(d.variables["AEW_lon"][:], np.nan).astype(float); lat = np.ma.filled(d.variables["AEW_lat"][:], np.nan).astype(float)
            sysn = np.asarray(d.variables["system"][:])
        finally:
            d.close()
        lon = np.where(lon > 180.0, lon - 360.0, lon)
        for s in range(lon.shape[0]):
            ok = np.flatnonzero(np.isfinite(lon[s]) & np.isfinite(lat[s]))
            if ok.size == 0:
                continue
            k = ("qtrack", os.path.basename(p), int(sysn[s]))
            for j in ok:
                by_date.setdefault(when[j].strftime("%Y-%m-%d"), []).append((k, float(lat[s, j]), float(lon[s, j])))
            tracks[k] = (when[ok[0]], float(lat[s, ok[0]]), float(lon[s, ok[0]]), float(np.nanmin(lon[s, ok])))
    return by_date, tracks


def check_western_coverage(paths, year_of, source):
    """One file per retained season for a western source, refused when any season is missing
    or duplicated (review fold, section 8 of the record)."""
    years = [year_of(p) for p in paths]
    missing = sorted(set(YEARS) - set(years)); extra = sorted({y for y in years if y not in YEARS})
    dup = sorted({y for y in years if years.count(y) > 1})
    if missing or extra or dup:
        raise SystemExit(f"REFUSED: the {source} files do not hold one file per retained season (missing {missing}, outside {extra}, duplicated {dup})")


def catalog_year(path):
    return int(os.path.basename(path).split("_")[3])


def qtrack_year(path):
    return int(os.path.basename(path)[-7:-3])


def western_paths(catalog_glob, qtrack_dir):
    """The catalog and QTrack files of the retained seasons, refused unless each source holds
    exactly one file per season."""
    cat_paths = sorted(p for p in glob.glob(catalog_glob) if catalog_year(p) in YEARS)
    q_paths = sorted(p for p in glob.glob(os.path.join(qtrack_dir, "ERA5_AEW_tracks_atlantic_????.nc")) if qtrack_year(p) in YEARS)
    check_western_coverage(cat_paths, catalog_year, "catalog")
    check_western_coverage(q_paths, qtrack_year, "QTrack")
    return cat_paths, q_paths


# ------------------------------------------------------------------ the archive

def season_histories(xrecords):
    by = {}
    for r in xrecords:
        by.setdefault(int(r["year"]), {})[int(r["month"])] = r
    month_of = A.month_index(); out, gaps = {}, {}
    for y in YEARS:
        for b in xrecords[0]["bands"]:
            dates, H, gap = season_hundredths_with_gaps(by[y], b)
            gaps[(y, b)] = [float(x) for x in XLONS[gap]]
            per = {}
            for rep in HI.REPS:
                mins = rep_minima(rep, H, month_of, XLONS, gap)
                for linker in HI.LINKERS:
                    hs = HI.link(mins, linker)
                    per[HI.combo(rep, linker)] = [(h, HI.classify(h)) for h in hs]
            out[(y, b)] = (dates, per)
    return out, gaps


def eastern_westward(hist, key):
    """The eastern-origin westward histories of one combination, as dicts with their
    season, band, dated members and first and last longitudes."""
    out = []
    for (y, b), (dates, per) in hist.items():
        for h, c in per[key]:
            if c == "westward" and h["lons"][0] >= EAST_ORIGIN:
                out.append({"year": y, "band": b, "members": [(dates[d], x) for d, x in zip(h["days"], h["lons"])], "first": h["lons"][0], "last": h["lons"][-1], "days": len(h["days"])})
    return out


def side_speed(members, east):
    sel = [(i, x) for i, (_, x) in enumerate(members) if (x > 40.0 if east else x <= 40.0)]
    return None if len(sel) < 2 else (sel[-1][1] - sel[0][1]) / (sel[-1][0] - sel[0][0])


def lat_window(band):
    lo, hi = V.BANDS[band]
    return lo - LAT_PAD, hi + LAT_PAD


def shift_date(date, year):
    return f"{year}{date[4:]}"


def meets(history, by_date, year=None, lon_tol=LON_TOL):
    """The record tracks with a position on the same calendar day within LON_TOL of a member
    at or west of MEET_EAST_LIMIT and inside the band's padded latitude window, with the
    history's dates placed in `year` when given."""
    lo, hi = lat_window(history["band"]); found = set()
    for date, x in history["members"]:
        if x > MEET_EAST_LIMIT:
            continue
        for k, la, lon in by_date.get(date if year is None else shift_date(date, year), []):
            if abs(lon - x) <= lon_tol and lo <= la <= hi:
                found.add(k)
    return found


def preceded(track, hist_index, year=None, lon_tol=LON_TOL, back_days=1):
    """Whether an eastern-origin westward history of a band whose padded window holds the
    track's first latitude has a member within LON_TOL at or east of the first longitude on
    the track's first day or the day before (in `year` when given)."""
    t0, la0, lo0 = track[0], track[1], track[2]
    yr = t0.year if year is None else year
    for back in range(back_days + 1):
        day = (t0 - dt.timedelta(days=back)).strftime("%m-%d")
        for b in V.BANDS:
            lo, hi = lat_window(b)
            if not (lo <= la0 <= hi):
                continue
            for x in hist_index.get((yr, b, day), []):
                if 0.0 <= x - lo0 <= lon_tol:
                    return True
    return False


def longitude_compatible(track, hist_index, same_season=True, lon_tol=LON_TOL):
    """Whether any member of a history in a band whose padded window holds the track's first
    latitude lies within lon_tol at or east of the first longitude on any date, in the track's
    own season or in any season. Added at the review fold (section 8 of the record) and
    descriptive only, it separates a comparison that geometry rules out from one that timing
    rules out."""
    t0, la0, lo0 = track[0], track[1], track[2]
    for (y, b, _), xs in hist_index.items():
        if same_season and y != t0.year:
            continue
        lo, hi = lat_window(b)
        if lo <= la0 <= hi and any(0.0 <= x - lo0 <= lon_tol for x in xs):
            return True
    return False


def eligible_tracks(tracks):
    out = {}
    for k, v in tracks.items():
        t0, la0, lo0 = v[0], v[1], v[2]
        if t0.year in YEARS and 6 <= t0.month <= 9 and lo0 >= PRECEDE_FIRST_LON and any(lat_window(b)[0] <= la0 <= lat_window(b)[1] for b in V.BANDS):
            out[k] = v
    return out


def archive_tables(hist, records):
    """Per combination and band, the reach of eastern-origin westward histories, side speeds,
    record meetings against the shuffled chance level, and per combination the record
    precedence against the same control."""
    keys = [HI.combo(r, l) for r, l in HI.ORDER]
    out = {}
    for key in keys:
        ew = eastern_westward(hist, key)
        res = {"bands": {}, "records": {}}
        for b in V.BANDS:
            w = [h for h in ew if h["band"] == b]
            crossing = [h for h in w if side_speed(h["members"], True) is not None and side_speed(h["members"], False) is not None]
            east_s = [side_speed(h["members"], True) for h in crossing]
            west_s = [side_speed(h["members"], False) for h in crossing]
            res["bands"][b] = {"eastern_origin_westward": len(w), "reach": {f"{r:g}": sum(1 for h in w if any(x <= r for _, x in h["members"])) for r in REACH},
                               "median_speed_east_of_40": (float(np.median(east_s)) if east_s else None), "median_speed_west_of_40": (float(np.median(west_s)) if west_s else None), "crossing_with_both_sides": len(west_s)}
        hist_index = {}
        for h in ew:
            for date, x in h["members"]:
                hist_index.setdefault((h["year"], h["band"], date[5:]), []).append(x)
        for rname, (by_date, tracks) in records.items():
            elig = [h for h in ew if any(x <= MEET_EAST_LIMIT for _, x in h["members"])]
            met = [meets(h, by_date) for h in elig]
            shuf = []
            for k_shift in range(1, len(YEARS)):
                cnt = 0
                for h in elig:
                    y2 = YEARS[(YEARS.index(h["year"]) + k_shift) % len(YEARS)]
                    cnt += bool(meets(h, by_date, y2))
                shuf.append(cnt)
            met_tracks = set().union(*met) if met else set()
            et = eligible_tracks(tracks)
            prec = [k for k, v in et.items() if preceded(v, hist_index)]
            pshuf = []
            for k_shift in range(1, len(YEARS)):
                pshuf.append(sum(1 for v in et.values() if preceded(v, hist_index, YEARS[(YEARS.index(v[0].year) + k_shift) % len(YEARS)])))
            sens_met = sum(1 for h in elig if meets(h, by_date, lon_tol=SENSITIVITY["meet_lon_tol"]))
            sens_met_shuf = [sum(1 for h in elig if meets(h, by_date, YEARS[(YEARS.index(h["year"]) + k) % len(YEARS)], lon_tol=SENSITIVITY["meet_lon_tol"])) for k in range(1, len(YEARS))]
            sens_prec = sum(1 for v in et.values() if preceded(v, hist_index, lon_tol=SENSITIVITY["precede_lon_tol"], back_days=SENSITIVITY["precede_back_days"]))
            sens_prec_shuf = [sum(1 for v in et.values() if preceded(v, hist_index, YEARS[(YEARS.index(v[0].year) + k) % len(YEARS)], lon_tol=SENSITIVITY["precede_lon_tol"], back_days=SENSITIVITY["precede_back_days"])) for k in range(1, len(YEARS))]
            firsts = np.array([[v[1], v[2]] for v in et.values()], float) if et else np.zeros((0, 2))
            res["records"][rname] = {"eligible_histories": len(elig), "meeting": sum(1 for m in met if m), "shuffled_mean": float(np.mean(shuf)) if shuf else None, "shuffled_max": int(max(shuf)) if shuf else None,
                                     "by_band": {b: {"eligible": sum(1 for h in elig if h["band"] == b), "meeting": sum(1 for h, m in zip(elig, met) if h["band"] == b and m)} for b in V.BANDS},
                                     "met_tracks": len(met_tracks), "met_tracks_reaching_15W": sum(1 for k in met_tracks if tracks[k][3] <= COAST_PROXY),
                                     "eastern_tracks": len(et), "preceded": len(prec), "preceded_shuffled_mean": float(np.mean(pshuf)) if pshuf else None, "preceded_shuffled_max": int(max(pshuf)) if pshuf else None,
                                     "preceded_reaching_15W": sum(1 for k in prec if tracks[k][3] <= COAST_PROXY),
                                     "sensitivity": {**SENSITIVITY, "meeting": sens_met, "meeting_shuffled_mean": float(np.mean(sens_met_shuf)), "meeting_shuffled_max": int(max(sens_met_shuf)),
                                                     "preceded": sens_prec, "preceded_shuffled_mean": float(np.mean(sens_prec_shuf)), "preceded_shuffled_max": int(max(sens_prec_shuf))},
                                     "shift_totals": {"meeting": shuf, "preceded": pshuf, "sensitivity_meeting": sens_met_shuf, "sensitivity_preceded": sens_prec_shuf},
                                     "longitude_compatible": {"same_season": sum(1 for v in et.values() if longitude_compatible(v, hist_index)), "any_season": sum(1 for v in et.values() if longitude_compatible(v, hist_index, same_season=False)),
                                                              "same_season_sensitivity": sum(1 for v in et.values() if longitude_compatible(v, hist_index, lon_tol=SENSITIVITY["precede_lon_tol"])),
                                                              "any_season_sensitivity": sum(1 for v in et.values() if longitude_compatible(v, hist_index, same_season=False, lon_tol=SENSITIVITY["precede_lon_tol"]))},
                                     "eastern_track_first_lat_q": ([float(np.percentile(firsts[:, 0], q)) for q in (25, 50, 75)] if len(firsts) else None),
                                     "eastern_track_first_lon_q": ([float(np.percentile(firsts[:, 1], q)) for q in (25, 50, 75)] if len(firsts) else None)}
        res["histories"] = [{k: v for k, v in h.items() if k != "members"} | {"members": h["members"]} for h in ew]
        out[key] = res
    return out


def archive_text(tabs, gaps, consistency):
    keys = [HI.combo(r, l) for r, l in HI.ORDER]
    f = lambda x, n=2: "" if x is None else f"{x:.{n}f}"
    L = ["# crossing test, the archive (section 2)", "eastern-origin westward histories start at or east of 45 E; a meeting is a record position on the same day within 2 degrees of longitude of a member at or west of 42 E and within 2.5 degrees of the band's latitudes; chance levels are the same test with the history's dates in each of the other 23 seasons", ""]
    ok = sum(1 for v in consistency.values() if v)
    L += [f"extended band means equal to the climatology's inside 20 to 75 E in {ok} of {len(consistency)} months", ""]
    gap_cols = {}
    for (y, b), g in gaps.items():
        gap_cols.setdefault(b, set()).update(g)
    L += ["gap longitudes over all seasons, by band: " + "; ".join(f"{b} {', '.join(f'{x:g}' for x in sorted(v)) or 'none'}" for b, v in gap_cols.items()), ""]
    L += ["## reach of eastern-origin westward histories, by band (a history reaches a longitude when any member lies at or west of it, corrected at the review fold)", "", "| combination | band | eastern-origin westward | reaching " + ", ".join(f"{r:g} E" for r in REACH) + " | crossing 40 E with two members each side | median speed east of 40 E | median speed west of 40 E |", "| --- |" + " --- |" * 6]
    for k in keys:
        for b, d in tabs[k]["bands"].items():
            L.append(f"| {k} | {b} | {d['eastern_origin_westward']} | " + ", ".join(str(d['reach'][f'{r:g}']) for r in REACH) + f" | {d['crossing_with_both_sides']} | {f(d['median_speed_east_of_40'])} | {f(d['median_speed_west_of_40'])} |")
    L += ["", "## meetings and precedence, against the shuffled chance level", "", "| combination | record | eligible histories | meeting | shuffled mean (max) | met tracks | met tracks reaching 15 W | eastern record tracks | preceded | shuffled mean (max) | preceded reaching 15 W |", "| --- |" + " --- |" * 10]
    for k in keys:
        for rname, d in tabs[k]["records"].items():
            L.append(f"| {k} | {rname} | {d['eligible_histories']} | {d['meeting']} | {f(d['shuffled_mean'])} ({d['shuffled_max']}) | {d['met_tracks']} | {d['met_tracks_reaching_15W']} | {d['eastern_tracks']} | {d['preceded']} | {f(d['preceded_shuffled_mean'])} ({d['preceded_shuffled_max']}) | {d['preceded_reaching_15W']} |")
    L += ["", "## added after the declaration, the sensitivity (meetings within 4 degrees, precedence within 5 degrees and up to 2 days before), and the eastern record tracks' first positions", "",
          "| combination | record | meeting | shuffled mean (max) | preceded | shuffled mean (max) | first latitude, quartiles | first longitude, quartiles |", "| --- |" + " --- |" * 7]
    for k in keys:
        for rname, d in tabs[k]["records"].items():
            sv = d["sensitivity"]; fl = d["eastern_track_first_lat_q"]; fo = d["eastern_track_first_lon_q"]
            L.append(f"| {k} | {rname} | {sv['meeting']} | {f(sv['meeting_shuffled_mean'])} ({sv['meeting_shuffled_max']}) | {sv['preceded']} | {f(sv['preceded_shuffled_mean'])} ({sv['preceded_shuffled_max']}) | {'' if fl is None else ', '.join(f'{x:.1f}' for x in fl)} | {'' if fo is None else ', '.join(f'{x:.1f}' for x in fo)} |")
    L += ["", "## added at the review fold, the eastern record tracks with a longitude-compatible history member on any date (descriptive, no chance level)", "",
          "| combination | record | eastern record tracks | same season, 2 degrees | any season, 2 degrees | same season, 5 degrees | any season, 5 degrees |", "| --- |" + " --- |" * 6]
    for k in keys:
        for rname, d in tabs[k]["records"].items():
            c = d["longitude_compatible"]
            L.append(f"| {k} | {rname} | {d['eastern_tracks']} | {c['same_season']} | {c['any_season']} | {c['same_season_sensitivity']} | {c['any_season_sensitivity']} |")
    L += ["", "## added at the review fold, the observed count against the 23 season shifts (shifts at or above the observed count, of 23)", "",
          "| combination | record | meeting | shifts at or above | preceded | shifts at or above | sensitivity meeting | shifts at or above | sensitivity preceded | shifts at or above |", "| --- |" + " --- |" * 9]
    for k in keys:
        for rname, d in tabs[k]["records"].items():
            st = d["shift_totals"]; obs = (d["meeting"], d["preceded"], d["sensitivity"]["meeting"], d["sensitivity"]["preceded"])
            cells = [f"{o} | {sum(1 for x in st[q] if x >= o)}" for o, q in zip(obs, ("meeting", "preceded", "sensitivity_meeting", "sensitivity_preceded"))]
            L.append(f"| {k} | {rname} | " + " | ".join(cells) + " |")
    return "\n".join(L) + "\n"


def draw_first_last(path, tabs, key, met_catalog, met_qtrack):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hs = tabs[key]["histories"]
    fig, axes = plt.subplots(1, len(V.BANDS), figsize=(4.4 * len(V.BANDS) + 1, 4.6), sharex=True, sharey=True, constrained_layout=True)
    for ax, b in zip(np.atleast_1d(axes), V.BANDS):
        w = [(i, h) for i, h in enumerate(hs) if h["band"] == b]
        for i, h in w:
            mk = "s" if (i in met_catalog and i in met_qtrack) else ("o" if i in met_catalog else ("^" if i in met_qtrack else "."))
            ax.plot(h["first"], h["last"], mk, color="k" if mk == "." else "tab:red", ms=6 if mk != "." else 5)
        ax.plot([-10, 75], [-10, 75], color="0.6", lw=0.8); ax.axhline(40.0, color="k", lw=0.7, ls="--")
        ax.set_title(f"{b} ({len(w)})", fontsize=10); ax.set_xlabel("first longitude (degrees E)"); ax.grid(True, color="0.85", lw=0.4)
    np.atleast_1d(axes)[0].set_ylabel("last longitude (degrees E)")
    fig.suptitle(f"Eastern-origin westward histories under {key} on 10 W to 75 E, first against last longitude. Red marks a meeting west of 42 E (circle catalog, triangle QTrack, square both). Dashed at 40 E. Meetings are coincidences of position and date, not identifications.", fontsize=9, wrap=True)
    fig.savefig(path, dpi=100); plt.close(fig)


# ------------------------------------------------------------------ main

def refuse_existing(paths):
    for p in paths:
        if os.path.exists(p):
            raise SystemExit(f"REFUSED: {p} exists and a record is never overwritten")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("render", "constructions", "archive"))
    ap.add_argument("--out-dir", required=True)
    for opt in ("--xband", "--records", "--catalog", "--qtrack", "--constructions"):
        ap.add_argument(opt, default=None)
    ap.add_argument("--tag", default="", help="suffix for a corrected archive written beside the first build, for example r1")
    a = ap.parse_args(argv)
    gen = {"script": "scripts/eastern_crossing_test.py", "sha256": sha256_of(os.path.abspath(__file__)), "declared": "EASTERN_CROSSING_TEST_2026-10-19.md",
           "histories_script_sha256": sha256_of(os.path.join(HERE, "eastern_minima_histories.py")), "anomaly_script_sha256": sha256_of(os.path.join(HERE, "eastern_anomaly_minima.py"))}
    if a.mode == "render":
        os.makedirs(a.out_dir, exist_ok=True)
        print(json.dumps({"rendered": len(render_all(a.out_dir))}))
        return 0
    if a.mode == "constructions":
        pj, pm = os.path.join(a.out_dir, "eval60_cross_constructions.json"), os.path.join(a.out_dir, "eval60_cross_constructions.md")
        refuse_existing([pj, pm])
        results = run_constructions(); verdict = apply_rule(results)
        with open(pj, "w", encoding="utf-8") as fh:
            json.dump({"generated_by": gen, "results": results, "verdict": verdict}, fh, default=float)
        text = constructions_text(results, verdict)
        with open(pm, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(text)
        return 0
    if not all((a.xband, a.records, a.catalog, a.qtrack, a.constructions)):
        raise SystemExit("REFUSED: --xband, --records, --catalog, --qtrack and --constructions are required for the archive")
    sfx = f"_{a.tag}" if a.tag else ""
    paths = {k: os.path.join(a.out_dir, v) for k, v in (("json", f"eval60_cross_archive{sfx}.json"), ("tables", f"eval60_cross_tables{sfx}.md"), ("figure", f"eval60_cross_first_last{sfx}.png"))}
    refuse_existing(paths.values())
    xfiles = sorted(glob.glob(os.path.join(a.xband, "eval60_xband_????_??.json.gz")))
    xrecs = [SC.load_record(p) for p in xfiles]
    check_xband_identity(xfiles, xrecs)
    consistency = {}
    for p, xr in zip(xfiles, xrecs):
        cpath = os.path.join(a.records, f"eval60_regime_{xr['year']}_{xr['month']:02d}.json.gz")
        consistency[f"{xr['year']}-{xr['month']:02d}"] = check_against_climatology(xr, SC.load_record(cpath))
    if not all(consistency.values()):
        raise SystemExit(f"REFUSED: the extended band means differ from the climatology's in {sorted(k for k, v in consistency.items() if not v)}")
    cat_paths, q_paths = western_paths(a.catalog, a.qtrack)
    hist, gaps = season_histories(xrecs)
    records = {"catalog": catalog_points(cat_paths), "qtrack": qtrack_points(q_paths)}
    tabs = archive_tables(hist, records)
    key = HI.combo(*HI.ORDER[0])
    met_c = {i for i, h in enumerate(tabs[key]["histories"]) if meets(h, records["catalog"][0])}
    met_q = {i for i, h in enumerate(tabs[key]["histories"]) if meets(h, records["qtrack"][0])}
    draw_first_last(paths["figure"], tabs, key, met_c, met_q)
    gen.update({"xband": {os.path.basename(p): sha256_of(p) for p in xfiles}, "catalog": {os.path.basename(p): sha256_of(p) for p in cat_paths},
                "qtrack": {os.path.basename(p): sha256_of(p) for p in q_paths}, "constructions": {"path": a.constructions, "sha256": sha256_of(a.constructions)}})
    with open(paths["json"], "w", encoding="utf-8") as fh:
        json.dump({"generated_by": gen, "consistency": consistency, "gaps": {f"{y}|{b}": g for (y, b), g in gaps.items()}, "tables": tabs,
                   "figure": {"path": paths["figure"], "combination": key, "sha256": sha256_of(paths["figure"])}}, fh, default=str)
    text = archive_text(tabs, gaps, consistency)
    with open(paths["tables"], "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
