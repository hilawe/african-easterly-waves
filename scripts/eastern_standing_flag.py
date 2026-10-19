#!/usr/bin/env python3
"""The standing-element flag for eastern positions (`EASTERN_STANDING_FLAG_2026-10-18.md`,
section 2): co-location of a position with the region where, in the other seasons of the
same calendar month, northerly flow at 700 hPa is present at least a stated fraction of
the time. Band-level positions (the daily candidates of all 96 months, the ledger's anchor
days) are tested against the band mask, built from the band-day northerly fraction of the
daily-mean band mean. The descriptive layer's row positions are tested against the cell
mask, built from the climatology's per-cell persistence fraction. Every decision is made
in the stored precision. A flag is co-location, never identity, and nothing is tracked,
linked or explained.

    python3 scripts/eastern_standing_flag.py --records DIR --layer-dir DIR --ledger PATH \\
        --candidates PATH --out-dir DIR
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import sys
from fractions import Fraction

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eastern_standing_climatology as SC  # noqa: E402
import eastern_t1_ledger as T  # noqa: E402
import eastern_candidate_positions as CP  # noqa: E402

YEARS = SC.YEARS
MONTHS = SC.MONTHS
PRIMARY = Fraction(1, 2)
FORMS = {"primary": (PRIMARY, True), "t0.4": (Fraction(2, 5), True), "t0.6": (Fraction(3, 5), True), "all_seasons": (PRIMARY, False)}
NORTHERLY_HUNDREDTHS = -1600          # four six-hourly samples summing to at or below this, a daily mean at or below -4 m/s
FLOOR_MS = -4.0
PROMINENCE_CLASSES = {"all": None, "p2": 2.0, "p4": 4.0}
PRIMARY_CLASS = "p2"
BIN_DEG = 2.0
EAST_OF = 40.0
LAT_BANDS = (("10 to 14 N", 10, 14), ("15 to 19 N", 15, 19), ("20 to 24 N", 20, 24), ("25 to 30 N", 25, 30))


def sha256_of(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


# ------------------------------------------------------------------ the band mask

def band_day_counts(record, band):
    """Per longitude, (northerly band-days, available band-days) over the month. A band-day
    is northerly when its four stored six-hourly band-mean samples (two decimals) sum to at
    or below -1600 hundredths, and unavailable when any sample is missing."""
    S = np.array(record["band_means_700"][band], float)
    times = record["times"]; days = [t[:10] for t in times]; hours = [t[11:13] for t in times]
    north = np.zeros(S.shape[1], int); avail = np.zeros(S.shape[1], int)
    for d in sorted(set(days)):
        idx = [i for i, t in enumerate(days) if t == d]
        if sorted(hours[i] for i in idx) != list(T.EXPECTED_HOURS):
            raise SystemExit(f"REFUSED: {d} does not hold exactly the four six-hourly samples")
        block = S[idx]
        ok = np.isfinite(block).all(axis=0)
        hund = np.rint(np.where(np.isfinite(block), block, 0.0) * 100.0).astype(np.int64).sum(axis=0)
        avail += ok
        north += ok & (hund <= NORTHERLY_HUNDREDTHS)
    return north, avail


def band_fractions(records):
    """{(year, month, band): (north, avail)} for every record and band."""
    return {(int(r["year"]), int(r["month"]), b): band_day_counts(r, b) for r in records for b in r["bands"]}


def band_mask(fr, year, month, band, threshold, leave_out=True, years=None):
    """Per longitude, whether the mean over the seasons (all but `year` when `leave_out`)
    that have an available band-day there of the band-day northerly fraction is at or above
    `threshold`, in exact rational arithmetic. A longitude with no contributing season is
    out of the mask."""
    years = YEARS if years is None else years
    seasons = [y for y in years if not (leave_out and y == year)]
    n = len(fr[(seasons[0], month, band)][0])
    out = np.zeros(n, bool)
    for i in range(n):
        total = Fraction(0); k = 0
        for y in seasons:
            nn, aa = fr[(y, month, band)]
            if aa[i] > 0:
                total += Fraction(int(nn[i]), int(aa[i])); k += 1
        out[i] = k > 0 and total >= threshold * k
    return out


def band_fraction(fr, year, month, band, leave_out=True, years=None):
    """Per longitude, the mean over the seasons (all but `year` when `leave_out`) that have
    an available band-day there of the band-day northerly fraction, as a float from the
    exact rational mean, NaN where no season contributes. The threshold-free value behind
    the band mask, added after the declaration (section 4 of the record says so)."""
    years = YEARS if years is None else years
    seasons = [y for y in years if not (leave_out and y == year)]
    n = len(fr[(seasons[0], month, band)][0])
    out = np.full(n, np.nan)
    for i in range(n):
        total = Fraction(0); k = 0
        for y in seasons:
            nn, aa = fr[(y, month, band)]
            if aa[i] > 0:
                total += Fraction(int(nn[i]), int(aa[i])); k += 1
        if k:
            out[i] = float(total / k)
    return out


def mask_distance(mask, lons, x):
    """The distance in degrees from longitude x to the nearest longitude in the mask, zero
    on the mask, None when the mask is empty."""
    on = np.asarray(lons, float)[np.asarray(mask, bool)]
    return None if on.size == 0 else float(np.min(np.abs(on - float(x))))


# ------------------------------------------------------------------ the cell mask

def cell_thousandths(records, month):
    """{year: the record's per-cell persistence fraction in stored thousandths (NaN kept)}."""
    return {int(r["year"]): np.rint(np.array(r["persistence"]["fraction"], float) * 1000.0) for r in records if int(r["month"]) == month}


def cell_mask(th, year, threshold, leave_out=True):
    """Per cell, whether the mean over the seasons (all but `year` when `leave_out`) of the
    stored persistence fraction is at or above `threshold`, decided on the thousandths: the
    sum of the contributing seasons' thousandths at or above 1000 times the threshold times
    their count, in integers."""
    stack = np.array([v for y, v in sorted(th.items()) if not (leave_out and y == year)])
    k = np.isfinite(stack).sum(axis=0)
    total = np.nansum(stack, axis=0).astype(np.int64)
    p, q = threshold.numerator, threshold.denominator
    return (k > 0) & (total * q >= p * 1000 * k)


# ------------------------------------------------------------------ the masks for every form, cached

class Masks:
    def __init__(self, records):
        self.records = records
        self.fr = band_fractions(records)
        self.th = {m: cell_thousandths(records, m) for m in MONTHS}
        self._band = {}; self._cell = {}; self._value = {}

    def band(self, year, month, band, form):
        key = (year, month, band, form)
        if key not in self._band:
            t, loo = FORMS[form]
            self._band[key] = band_mask(self.fr, year, month, band, t, loo)
        return self._band[key]

    def cell_value(self, year, month):
        """The leave-one-out mean of the stored persistence fraction per cell (NaN where no
        season contributes), the threshold-free value behind the cell mask, added after the
        declaration."""
        key = ("cell", year, month)
        if key not in self._value:
            stack = np.array([v for y, v in sorted(self.th[month].items()) if y != year])
            k = np.isfinite(stack).sum(axis=0)
            with np.errstate(invalid="ignore"):
                self._value[key] = np.where(k > 0, np.nansum(stack, axis=0) / np.maximum(k, 1) / 1000.0, np.nan)
        return self._value[key]

    def band_value(self, year, month, band):
        key = (year, month, band)
        if key not in self._value:
            self._value[key] = band_fraction(self.fr, year, month, band, leave_out=True)
        return self._value[key]

    def cell(self, year, month, form):
        key = (year, month, form)
        if key not in self._cell:
            t, loo = FORMS[form]
            self._cell[key] = cell_mask(self.th[month], year, t, loo)
        return self._cell[key]


def index_of(values, x, what):
    i = np.flatnonzero(np.asarray(values, float) == float(x))
    if len(i) != 1:
        raise SystemExit(f"REFUSED: {what} {x} is not on the grid")
    return int(i[0])


# ------------------------------------------------------------------ the candidates

def flag_candidates(records, masks):
    """Every candidate of every month record, with its flags in every form."""
    out = []
    for r in records:
        y, m = int(r["year"]), int(r["month"]); lons = np.array(r["lons"], float)
        for b in r["bands"]:
            dm = T.daily_means(r, b)
            for day in sorted(dm):
                for c in CP.candidates_on(dm[day], lons, context=f" ({y}-{m:02d}, {b}, {day})"):
                    i = index_of(lons, c["lon"], "candidate longitude")
                    out.append({"year": y, "month": m, "band": b, "date": day, "lon": float(c["lon"]), "depth": float(c["depth"]), "prominence": float(c["prominence"]),
                                "flags": {f: bool(masks.band(y, m, b, f)[i]) for f in FORMS},
                                "loo_fraction": float(masks.band_value(y, m, b)[i]), "mask_distance_deg": mask_distance(masks.band(y, m, b, "primary"), lons, c["lon"])})
    return out


def in_class(c, cls):
    p = PROMINENCE_CLASSES[cls]
    return p is None or c["prominence"] >= p


def candidate_tables(cands, records):
    """Counts and flagged counts by class, band and month, by class and band pooled, and by
    class, band and two-degree bin pooled and by month, with band-days."""
    bands = list(records[0]["bands"])
    band_days = {}
    for r in records:
        nd = len({t[:10] for t in r["times"]})
        for b in bands:
            band_days[(int(r["month"]), b)] = band_days.get((int(r["month"]), b), 0) + nd
    edges = np.arange(20.0, 76.0 + 1e-9, BIN_DEG)
    def blank():
        return {"count": 0, "flagged": {f: 0 for f in FORMS}, "east_count": 0, "east_flagged": {f: 0 for f in FORMS}}
    by_mb = {cls: {str(m): {b: blank() for b in bands} for m in MONTHS} for cls in PROMINENCE_CLASSES}
    pooled = {cls: {b: blank() for b in bands} for cls in PROMINENCE_CLASSES}
    by_bin = {cls: {b: {m: [{"count": 0, "flagged": {f: 0 for f in FORMS}} for _ in range(len(edges) - 1)] for m in ["all"] + [str(x) for x in MONTHS]} for b in bands} for cls in PROMINENCE_CLASSES}
    for c in cands:
        k = min(int(np.searchsorted(edges, c["lon"], side="right") - 1), len(edges) - 2)
        for cls in PROMINENCE_CLASSES:
            if not in_class(c, cls):
                continue
            for d in (by_mb[cls][str(c["month"])][c["band"]], pooled[cls][c["band"]]):
                d["count"] += 1
                for f in FORMS:
                    d["flagged"][f] += c["flags"][f]
                if c["lon"] >= EAST_OF:
                    d["east_count"] += 1
                    for f in FORMS:
                        d["east_flagged"][f] += c["flags"][f]
            for key in ("all", str(c["month"])):
                cell = by_bin[cls][c["band"]][key][k]
                cell["count"] += 1
                for f in FORMS:
                    cell["flagged"][f] += c["flags"][f]
    bd = {b: {"all": sum(band_days.get((m, b), 0) for m in MONTHS), **{str(m): band_days.get((m, b), 0) for m in MONTHS}} for b in bands}
    return {"by_month_band": by_mb, "pooled": pooled, "by_bin": by_bin, "bin_edges_deg": edges.tolist(), "band_days": bd, "graded": graded_tables(cands, bands)}


def tenths_histogram(values):
    """Counts in tenths, 0 to 0.1 up to 0.9 to 1 with 1 in the last, the bin decided on the
    value times ten rounded to nine places, so that a value of exactly 0.6 from an exact
    rational lands in 0.6 to 0.7 and not below a floating edge of 0.6000000000000001."""
    out = [0] * 10
    for v in values:
        out[min(int(np.floor(round(float(v) * 10.0, 9))), 9)] += 1
    return out


DISTANCE_CLASSES = (("0.5 to 2 degrees", 0.0, 2.0), ("2.5 to 5 degrees", 2.0, 5.0), ("more than 5 degrees", 5.0, np.inf))


def graded_tables(cands, bands):
    """Added after the declaration. For the prominent candidates by band, all longitudes and
    at or east of 40 E: quantiles and a histogram in tenths of the leave-one-out band-day
    northerly fraction at the candidate, and the off-mask candidates by their distance to
    the nearest primary-mask longitude of the same season, month and band."""
    out = {}
    for b in bands:
        for scope in ("all", "east"):
            cs = [c for c in cands if c["band"] == b and in_class(c, PRIMARY_CLASS) and (scope == "all" or c["lon"] >= EAST_OF)]
            v = np.array([c["loo_fraction"] for c in cs], float); v = v[np.isfinite(v)]
            hist = tenths_histogram(v)
            off = [c for c in cs if not c["flags"]["primary"]]
            dist = {name: sum(1 for c in off if c["mask_distance_deg"] is not None and lo < c["mask_distance_deg"] <= hi) for name, lo, hi in DISTANCE_CLASSES}
            dist["no mask in the band-month"] = sum(1 for c in off if c["mask_distance_deg"] is None)
            out.setdefault(b, {})[scope] = {"count": len(cs), "quantiles": {q: (float(np.quantile(v, q / 100)) if v.size else None) for q in (10, 25, 50, 75, 90)},
                                           "histogram_tenths": hist, "off_mask": len(off), "off_mask_by_distance": dist}
    return out


def reproduce_layer(cands, committed):
    """The ten characterization months' recomputed candidates against the committed
    candidate layer: equal when every band-day lists the same longitudes, depths and
    prominences."""
    have = {}
    for c in cands:
        have.setdefault((f"{c['year']}-{c['month']:02d}", c["band"], c["date"]), []).append(c)
    out = {}
    for mkey, bands in committed["layer"].items():
        same, n = True, 0
        for b, days in bands.items():
            for day, lst in days.items():
                mine = sorted(have.get((mkey, b, day), []), key=lambda q: q["lon"])
                ref = sorted(lst, key=lambda q: q["lon"])
                n += len(ref)
                if len(mine) != len(ref) or any(a["lon"] != r["lon"] or abs(a["depth"] - r["depth"]) > 1e-9 or abs(a["prominence"] - r["prominence"]) > 1e-9 for a, r in zip(mine, ref)):
                    same = False
        out[mkey] = {"equal": same, "candidates": n}
    return out


def band_mask_runs(masks, records, form="all_seasons"):
    """Per calendar month and band, the longitude runs of the band mask in the given form
    (the all-season form is one mask per month and band)."""
    lons = np.array(records[0]["lons"], float); out = {}
    y0 = YEARS[0]
    for m in MONTHS:
        for b in records[0]["bands"]:
            mk = masks.band(y0, m, b, form)
            runs, start = [], None
            for i, v in enumerate(list(mk) + [False]):
                if v and start is None:
                    start = i
                if not v and start is not None:
                    runs.append([float(lons[start]), float(lons[i - 1])]); start = None
            out.setdefault(str(m), {})[b] = runs
    return out


# ------------------------------------------------------------------ the descriptive layer's selected positions

def parse_minima(text):
    out = []
    for part in (text or "").split(";"):
        if not part:
            continue
        bits = part.split(":")
        out.append((float(bits[0]), float(bits[1])))
    return out


def flag_selected(layer_dir, interval_names, masks, lat, lon):
    """Every assigned row-step of the descriptive layer version 1, flagged in every form by
    the cell mask of its season and calendar month, with the count of the other minima in
    its window at or below the floor and how many of those lie off the primary mask."""
    rows_out = []
    for name in interval_names:
        with open(os.path.join(layer_dir, f"{name}.csv"), newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row["status"] != "assigned":
                    continue
                y, m = int(row["time"][:4]), int(row["time"][5:7])
                j = index_of(lat, row["row_lat"], "row latitude"); i = index_of(lon, row["selected_lon"], "selected longitude")
                flags = {f: bool(masks.cell(y, m, f)[j, i]) for f in FORMS}
                lo, hi, sel = float(row["window_lon_lo"]), float(row["window_lon_hi"]), float(row["selected_lon"])
                others = [(x, v) for x, v in parse_minima(row["all_minima"]) if lo <= x <= hi and x != sel and v <= FLOOR_MS]
                prim = masks.cell(y, m, "primary")
                off = [x for x, v in others if not prim[j, index_of(lon, x, "minimum longitude")]]
                sel_v = float(row["selected_v_ms"])
                off_deeper = [x for x, v in others if v <= sel_v and not prim[j, index_of(lon, x, "minimum longitude")]]
                rows_out.append({"interval": name, "time": row["time"], "row_lat": float(row["row_lat"]), "selected_lon": sel, "selected_v_ms": round(float(row["selected_v_ms"]), 3),
                                 "motion_label": row["motion_label"] or "unlabeled", "flags": flags, "other_window_minima": len(others), "other_window_minima_off_mask": len(off),
                                 "other_window_minima_off_mask_at_least_as_deep": len(off_deeper), "loo_cell_fraction": masks.cell_value(y, m)[j, i]})
    return rows_out


def selected_tables(sel):
    def blank():
        return {"assigned": 0, "flagged": {f: 0 for f in FORMS}, "flagged_with_off_mask_alternative": 0}
    by_int, by_lat, by_label = {}, {}, {}
    alt = {"flagged": 0, "flagged_with_off_mask_alternative": 0, "flagged_with_off_mask_alternative_at_least_as_deep": 0, "unflagged": 0}
    for s in sel:
        latb = next(n for n, lo, hi in LAT_BANDS if lo <= s["row_lat"] <= hi)
        for d in (by_int.setdefault(s["interval"], blank()), by_lat.setdefault(latb, blank()), by_label.setdefault(s["motion_label"], blank())):
            d["assigned"] += 1
            for f in FORMS:
                d["flagged"][f] += s["flags"][f]
            d["flagged_with_off_mask_alternative"] += bool(s["flags"]["primary"] and s["other_window_minima_off_mask"] > 0)
        if s["flags"]["primary"]:
            alt["flagged"] += 1; alt["flagged_with_off_mask_alternative"] += s["other_window_minima_off_mask"] > 0
            alt["flagged_with_off_mask_alternative_at_least_as_deep"] += s["other_window_minima_off_mask_at_least_as_deep"] > 0
        else:
            alt["unflagged"] += 1
    v = np.array([s["loo_cell_fraction"] for s in sel], float); v = v[np.isfinite(v)]
    graded = {"quantiles": {q: (float(np.quantile(v, q / 100)) if v.size else None) for q in (10, 25, 50, 75, 90)}, "histogram_tenths": tenths_histogram(v)}
    return {"by_interval": by_int, "by_latitude": by_lat, "by_motion_label": by_label, "alternatives": alt, "graded": graded}


# ------------------------------------------------------------------ the ledger's anchors

def flag_ledger(ledger, masks, lons):
    out = []
    for e in ledger:
        cls = (e.get("speeds") or {}).get("direction") or "no fit"
        for d in e["days"]:
            if not d["anchored"]:
                continue
            i = index_of(lons, d["anchor_lon"], "anchor longitude")
            out.append({"id": e["id"], "class": cls, "band": e["band"], "date": d["date"], "anchor_lon": float(d["anchor_lon"]), "edge": bool(d["edge"]),
                        "flags": {f: bool(masks.band(int(e["year"]), int(e["month"]), e["band"], f)[i]) for f in FORMS}})
    return out


def ledger_coordinates(rows):
    """The anchored entry-days against their distinct date-band-longitude coordinates, and
    every coordinate held by more than one entry with the entries and classes holding it
    (review of 2026-10-18, F4). The shares keep the entry weighting and deduplicate
    nothing."""
    by = {}
    for r in rows:
        by.setdefault((r["date"], r["band"], r["anchor_lon"]), []).append((r["id"], r["class"], r["flags"]["primary"]))
    shared = [{"date": k[0], "band": k[1], "anchor_lon": k[2], "entries": [{"id": i, "class": c, "primary": p} for i, c, p in v]} for k, v in sorted(by.items()) if len(v) > 1]
    return {"entry_days": len(rows), "distinct_coordinates": len(by), "shared": shared,
            "shared_across_classes": sum(1 for x in shared if len({e["class"] for e in x["entries"]}) > 1)}


def ledger_table(rows):
    out = {}
    for r in rows:
        d = out.setdefault(r["class"], {"entries": set(), "anchored_days": 0, "flagged": {f: 0 for f in FORMS}})
        d["entries"].add(r["id"]); d["anchored_days"] += 1
        for f in FORMS:
            d["flagged"][f] += r["flags"][f]
    return {k: {"entries": len(v["entries"]), "anchored_days": v["anchored_days"], "flagged": v["flagged"]} for k, v in out.items()}


def season_effect(ct, st, lt):
    """The largest absolute difference between the primary and the all-season shares at
    each aggregation scale, with where it occurs and the count behind it (review of
    2026-10-18, F1: the first account quoted one bound for every scale)."""
    out = {}
    def upd(scale, flagged, n, where):
        if n == 0:
            return
        diff = abs(flagged["primary"] - flagged["all_seasons"]) / n
        if scale not in out or diff > out[scale]["difference"]:
            out[scale] = {"difference": diff, "where": where, "count": n}
    edges = ct["bin_edges_deg"]
    for cls in PROMINENCE_CLASSES:
        for b, d in ct["pooled"][cls].items():
            upd("pooled by band", d["flagged"], d["count"], f"{cls}, {b}")
            upd("pooled by band", d["east_flagged"], d["east_count"], f"{cls}, {b}, at or east of 40 E")
        for m, bands in ct["by_month_band"][cls].items():
            for b, d in bands.items():
                upd("by month and band", d["flagged"], d["count"], f"{cls}, {SC.MONTH_NAMES[int(m)]}, {b}")
                upd("by month and band", d["east_flagged"], d["east_count"], f"{cls}, {SC.MONTH_NAMES[int(m)]}, {b}, at or east of 40 E")
        for b, months in ct["by_bin"][cls].items():
            for k, cell in enumerate(months["all"]):
                where = f"{cls}, {b}, {edges[k]:g} to {edges[k + 1]:g} E"
                upd("two-degree bins", cell["flagged"], cell["count"], where)
                if edges[k] >= EAST_OF:
                    upd("two-degree bins at or east of 40 E", cell["flagged"], cell["count"], where)
    for grp in ("by_interval", "by_latitude", "by_motion_label"):
        for k, d in st[grp].items():
            upd("selected positions", d["flagged"], d["assigned"], f"{grp.replace('by_', '')} {k}")
    for k, d in lt.items():
        upd("ledger classes", d["flagged"], d["anchored_days"], f"class {k}")
    return out


# ------------------------------------------------------------------ text and figure

def share(n, d):
    return "" if d == 0 else f"{n / d:.2f}"


def tables_text(ct, repro, runs, st, lt, n_records, lc=None, se=None):
    F = list(FORMS)
    L = ["# the standing-element flag, derived (section 2)",
         f"{n_records} month records; a flag is co-location with the mask of the object's season, month and band (band positions) or cell (row positions); primary is the leave-one-season-out mask at one half, t0.4 and t0.6 the threshold sensitivities, all_seasons the mask with the own season kept; a share is flagged over the stated count", ""]
    L += ["## the ten characterization months' candidates recomputed, against the committed layer", "", "| month | candidates | equal |", "| --- | --- | --- |"]
    for k, v in sorted(repro.items()):
        L.append(f"| {k} | {v['candidates']} | {'yes' if v['equal'] else 'NO'} |")
    L += ["", "## the band mask's longitude runs, all-season form at one half", "", "| month | band | runs (degrees E) |", "| --- | --- | --- |"]
    for m, bands in runs.items():
        for b, rr in bands.items():
            L.append(f"| {SC.MONTH_NAMES[int(m)]} | {b} | {', '.join(f'{a:g} to {z:g}' for a, z in rr) or 'none'} |")
    for cls in PROMINENCE_CLASSES:
        title = {"all": "all candidates", "p2": "prominent candidates (prominence at least 2 m/s, the primary population)", "p4": "candidates of prominence at least 4 m/s"}[cls]
        L += ["", f"## {title}, pooled over the four months, by band", "", "| band | count | " + " | ".join(f"share {f}" for f in F) + " | count at or east of 40 E | " + " | ".join(f"share {f}, east" for f in F) + " |", "| --- |" + " --- |" * (2 + 2 * len(F))]
        for b, d in ct["pooled"][cls].items():
            L.append(f"| {b} | {d['count']} | " + " | ".join(share(d["flagged"][f], d["count"]) for f in F) + f" | {d['east_count']} | " + " | ".join(share(d["east_flagged"][f], d["east_count"]) for f in F) + " |")
    L += ["", "## added after the declaration, the prominent candidates' leave-one-out band-day northerly fraction, and the off-mask ones by distance to the primary mask", "",
          "| band | scope | count | q10 | q25 | q50 | q75 | q90 | histogram in tenths | off-mask | 0.5 to 2 deg | 2.5 to 5 deg | more than 5 deg | no mask |", "| --- |" + " --- |" * 13]
    for b, scopes in ct["graded"].items():
        for sc, d in scopes.items():
            q = d["quantiles"]; dd = d["off_mask_by_distance"]
            L.append(f"| {b} | {'all longitudes' if sc == 'all' else 'at or east of 40 E'} | {d['count']} | " + " | ".join("" if q[k] is None else f"{q[k]:.2f}" for k in (10, 25, 50, 75, 90)) + f" | {', '.join(str(x) for x in d['histogram_tenths'])} | {d['off_mask']} | {dd['0.5 to 2 degrees']} | {dd['2.5 to 5 degrees']} | {dd['more than 5 degrees']} | {dd['no mask in the band-month']} |")
    L += ["", "## prominent candidates by calendar month and band", "", "| month | band | count | " + " | ".join(f"share {f}" for f in F) + " | count at or east of 40 E | share primary, east |", "| --- |" + " --- |" * (4 + len(F))]
    for m in MONTHS:
        for b, d in ct["by_month_band"][PRIMARY_CLASS][str(m)].items():
            L.append(f"| {SC.MONTH_NAMES[m]} | {b} | {d['count']} | " + " | ".join(share(d["flagged"][f], d["count"]) for f in F) + f" | {d['east_count']} | {share(d['east_flagged']['primary'], d['east_count'])} |")
    edges = ct["bin_edges_deg"]
    for cls in PROMINENCE_CLASSES:
        title = {"all": "all candidates", "p2": "prominent candidates", "p4": "candidates of prominence at least 4 m/s"}[cls]
        L += ["", f"## {title} by band and two-degree bin, pooled over the months (per band-day, all and off the primary mask, and the share flagged in each form)", ""]
        for b, months in ct["by_bin"][cls].items():
            nd = ct["band_days"][b]["all"]
            L += [f"{b}, {nd} band-days", "", "| bin (degrees E) | count | per band-day | off-mask per band-day | " + " | ".join(f"share {f}" for f in F) + " |", "| --- |" + " --- |" * (3 + len(F))]
            for k, cell in enumerate(months["all"]):
                if cell["count"] == 0:
                    continue
                L.append(f"| {edges[k]:g} to {edges[k + 1]:g} | {cell['count']} | {cell['count'] / nd:.3f} | {(cell['count'] - cell['flagged']['primary']) / nd:.3f} | " + " | ".join(share(cell["flagged"][f], cell["count"]) for f in F) + " |")
            L.append("")
    L += ["## the descriptive layer's selected positions (assigned row-steps), by interval", "", "| interval | assigned | " + " | ".join(f"share {f}" for f in F) + " | flagged (primary) | of which with an off-mask alternative |", "| --- |" + " --- |" * (3 + len(F))]
    for k, d in st["by_interval"].items():
        L.append(f"| {k} | {d['assigned']} | " + " | ".join(share(d["flagged"][f], d["assigned"]) for f in F) + f" | {d['flagged']['primary']} | {d['flagged_with_off_mask_alternative']} |")
    L += ["", "| latitude rows | assigned | " + " | ".join(f"share {f}" for f in F) + " | flagged (primary) | of which with an off-mask alternative |", "| --- |" + " --- |" * (3 + len(F))]
    for k, d in st["by_latitude"].items():
        L.append(f"| {k} | {d['assigned']} | " + " | ".join(share(d["flagged"][f], d["assigned"]) for f in F) + f" | {d['flagged']['primary']} | {d['flagged_with_off_mask_alternative']} |")
    L += ["", "| motion label | assigned | " + " | ".join(f"share {f}" for f in F) + " |", "| --- |" + " --- |" * (1 + len(F))]
    for k, d in sorted(st["by_motion_label"].items()):
        L.append(f"| {k} | {d['assigned']} | " + " | ".join(share(d["flagged"][f], d["assigned"]) for f in F) + " |")
    a = st["alternatives"]
    L += ["", f"flagged row-steps (primary) {a['flagged']}, of which with at least one other minimum at or below -4 m/s in the window off the mask {a['flagged_with_off_mask_alternative']}, and with one at least as deep as the selection {a['flagged_with_off_mask_alternative_at_least_as_deep']} (added after the declaration); unflagged {a['unflagged']}", ""]
    g = st["graded"]
    L += ["added after the declaration, the leave-one-out cell persistence fraction at the selected positions:", "", "| quantile 10 | 25 | 50 | 75 | 90 | histogram in tenths, 0 to 0.1 up to 0.9 to 1 |", "| --- | --- | --- | --- | --- | --- |",
          "| " + " | ".join(f"{g['quantiles'][q]:.2f}" for q in (10, 25, 50, 75, 90)) + " | " + ", ".join(str(x) for x in g["histogram_tenths"]) + " |", ""]
    L += ["## the ledger's anchored entry-days by the entry's fitted direction class (descriptive, the classes inherit the reader's intervals, entry-weighted)", "", "| class | entries | anchored entry-days | " + " | ".join(f"share {f}" for f in F) + " |", "| --- |" + " --- |" * (2 + len(F))]
    for k, d in sorted(lt.items()):
        L.append(f"| {k} | {d['entries']} | {d['anchored_days']} | " + " | ".join(share(d["flagged"][f], d["anchored_days"]) for f in F) + " |")
    if lc is not None:
        L += ["", f"{lc['entry_days']} anchored entry-days hold {lc['distinct_coordinates']} distinct date-band-longitude coordinates. {len(lc['shared'])} coordinates are held by more than one entry, {lc['shared_across_classes']} of them across classes:", ""]
        for x in lc["shared"]:
            L.append(f"- {x['date']}, {x['band']}, {x['anchor_lon']:g} E: " + ", ".join(f"{e['id']} {e['class']} (primary {'on' if e['primary'] else 'off'})" for e in x["entries"]))
    if se is not None:
        L += ["", "## leaving out the own season, the largest absolute difference between the primary and the all-season shares by aggregation scale", "", "| scale | largest difference | where | count |", "| --- | --- | --- | --- |"]
        for k, v in se.items():
            L.append(f"| {k} | {v['difference']:.3f} | {v['where']} | {v['count']} |")
    return "\n".join(L) + "\n"


def draw_figure(path, ct):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    edges = np.array(ct["bin_edges_deg"]); mid = (edges[:-1] + edges[1:]) / 2
    bands = list(ct["by_bin"][PRIMARY_CLASS])
    fig, axes = plt.subplots(2, len(bands), figsize=(4.6 * len(bands) + 1, 8), sharex=True, constrained_layout=True)
    for j, b in enumerate(bands):
        cells = ct["by_bin"][PRIMARY_CLASS][b]["all"]; nd = ct["band_days"][b]["all"]
        tot = np.array([c["count"] for c in cells]) / nd; off = np.array([c["count"] - c["flagged"]["primary"] for c in cells]) / nd
        ax = axes[0, j]
        ax.plot(mid, tot, color="k", lw=1.4, label="all prominent"); ax.plot(mid, off, color="tab:red", lw=1.4, label="off the standing mask")
        ax.set_title(b, fontsize=11); ax.grid(True, color="0.8", lw=0.4)
        ax = axes[1, j]
        for m in MONTHS:
            cm = ct["by_bin"][PRIMARY_CLASS][b][str(m)]
            sh = np.array([np.nan if c["count"] < 10 else c["flagged"]["primary"] / c["count"] for c in cm])
            ax.plot(mid, sh, lw=1.3, label=SC.MONTH_NAMES[m])
        ax.set_ylim(-0.02, 1.02); ax.grid(True, color="0.8", lw=0.4); ax.set_xlabel("longitude (degrees E)")
        for a in axes[:, j]:
            for x in (40.0, 60.0):
                a.axvline(x, color="k", lw=0.8, ls="--")
    axes[0, 0].set_ylabel("candidates per band-day"); axes[1, 0].set_ylabel("share on the standing mask")
    axes[0, 0].legend(fontsize=9); axes[1, 0].legend(fontsize=9)
    fig.suptitle(f"Prominent northerly minima of the daily-mean 700 hPa band mean (prominence at least 2 m/s), {len(YEARS)} seasons (2005 closed), two-degree bins. Top, per band-day pooled over June to September, all and off the leave-one-season-out standing mask at one half. Bottom, the share on the mask by calendar month, blank where a bin holds fewer than 10. Co-location only.", fontsize=9, wrap=True)
    fig.savefig(path, dpi=100); plt.close(fig)


# ------------------------------------------------------------------ main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--records", required=True); ap.add_argument("--layer-dir", required=True); ap.add_argument("--ledger", required=True)
    ap.add_argument("--candidates", required=True, help="the committed candidate layer, for the reproduction check"); ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    paths = {k: os.path.join(a.out_dir, v) for k, v in (("candidates", "eval60_flag_candidates.json"), ("selected", "eval60_flag_selected.csv"), ("ledger", "eval60_flag_ledger.json"),
                                                         ("tables", "eval60_flag_tables.md"), ("figure", "eval60_flag_shares.png"))}
    for p in paths.values():
        if os.path.exists(p):
            raise SystemExit(f"REFUSED: {p} exists and a record is never overwritten")
    files = sorted(glob.glob(os.path.join(a.records, "eval60_regime_????_??.json.gz")))
    records = [SC.load_record(p) for p in files]
    if len(records) != len(YEARS) * len(MONTHS):
        raise SystemExit(f"REFUSED: {len(records)} month records, not {len(YEARS) * len(MONTHS)}")
    SC.check_archive_identity(files, records)
    lat = np.array(records[0]["persistence"]["lat"], float); lon = np.array(records[0]["lons"], float)
    masks = Masks(records)
    cands = flag_candidates(records, masks)
    ct = candidate_tables(cands, records)
    with open(a.candidates, encoding="utf-8") as fh:
        committed = json.load(fh)
    repro = reproduce_layer(cands, committed)
    runs = band_mask_runs(masks, records)
    with open(os.path.join(a.layer_dir, "manifest.json"), encoding="utf-8") as fh:
        manifest = json.load(fh)
    names = [iv["name"] for iv in manifest["intervals"]]
    sel = flag_selected(a.layer_dir, names, masks, lat, lon)
    st = selected_tables(sel)
    with open(a.ledger, encoding="utf-8") as fh:
        ledger_doc = json.load(fh)
    led = flag_ledger(ledger_doc["ledger"], masks, lon)
    lt = ledger_table(led)
    lc = ledger_coordinates(led)
    se = season_effect(ct, st, lt)
    gen = {"script": "scripts/eastern_standing_flag.py", "sha256": sha256_of(os.path.abspath(__file__)), "declared": "EASTERN_STANDING_FLAG_2026-10-18.md",
           "month_records": {os.path.basename(p): sha256_of(p) for p in files}, "layer": {n: sha256_of(os.path.join(a.layer_dir, f"{n}.csv")) for n in names},
           "ledger": {"path": a.ledger, "sha256": sha256_of(a.ledger)}, "committed_candidates": {"path": a.candidates, "sha256": sha256_of(a.candidates)},
           "forms": {f: {"threshold": str(t), "leave_own_season_out": loo} for f, (t, loo) in FORMS.items()}}
    draw_figure(paths["figure"], ct)
    with open(paths["candidates"], "w", encoding="utf-8") as fh:
        json.dump({"generated_by": gen, "reproduction": repro, "band_mask_runs_all_seasons": runs, "tables": ct, "candidates_total": len(cands), "season_effect": se,
                   "figure": {"path": paths["figure"], "sha256": sha256_of(paths["figure"])}}, fh, default=float)
    with open(paths["selected"], "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["interval", "time", "row_lat", "selected_lon", "selected_v_ms", "motion_label"] + [f"flag_{f}" for f in FORMS] + ["other_window_minima_at_or_below_floor", "other_window_minima_off_mask", "other_window_minima_off_mask_at_least_as_deep", "loo_cell_fraction"])
        for s in sel:
            w.writerow([s["interval"], s["time"], f"{s['row_lat']:g}", f"{s['selected_lon']:g}", s["selected_v_ms"], s["motion_label"]] + [int(s["flags"][f]) for f in FORMS]
                       + [s["other_window_minima"], s["other_window_minima_off_mask"], s["other_window_minima_off_mask_at_least_as_deep"], ("" if not np.isfinite(s["loo_cell_fraction"]) else f"{s['loo_cell_fraction']:.4f}")])
    with open(paths["ledger"], "w", encoding="utf-8") as fh:
        json.dump({"generated_by": gen, "selected_tables": st, "ledger_table": lt, "ledger_coordinates": lc, "anchors": led}, fh, default=float)
    text = tables_text(ct, repro, runs, st, lt, len(records), lc, se)
    with open(paths["tables"], "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text[:2500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
