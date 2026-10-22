#!/usr/bin/env python3
"""The shape of the westward histories (`EASTERN_HISTORY_SHAPE_2026-10-21.md`): the 700 hPa
band-mean meridional wind, as its own-month anomaly, averaged around the daily members of the
westward histories, compared with the same average over constructions of known trough shape
detected by the same pipeline, and each shape then injected on the histories' tracks into the
lag-regression diagnostic as in the injection-recovery test. The average describes the field
around the members as a population. It identifies no feature and does not say what any one
history is.

    python3 scripts/eastern_history_shape.py constructions --inject PATH --records DIR --histories PATH --out-dir DIR
    python3 scripts/eastern_history_shape.py archive --inject PATH --records DIR --histories PATH --xband DIR \\
        --lagreg PATH --constructions PATH --out-dir DIR
"""

import argparse
import glob
import gzip
import hashlib
import json
import os
import sys
from decimal import Decimal

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
import eastern_anomaly_minima as A  # noqa: E402
import eastern_crossing_test as X  # noqa: E402
import eastern_injection_recovery as J  # noqa: E402
import eastern_minima_histories as HI  # noqa: E402
import eastern_wobble_climatology as W  # noqa: E402

L = J.L
YEARS = J.YEARS
XL = J.LONS
J0 = int(np.flatnonzero(np.isclose(XL, A.LONS[0]))[0])
NA = len(A.LONS)
WIDTH = 2.0
SHAPES = ("gauss", "m1", "m2.25", "dipole")
CONSTRUCTION_SHAPES = ("none",) + SHAPES
DX = np.arange(-30, 31) * 0.5
FLANK = (2.0, 8.0)
COMPOSITE_BANDS = ("15 to 19.5 N", "20 to 24.5 N")
SHAPE_SEED = 2026102201
SEPARATION_MIN = 0.15
SQRT_E = float(np.sqrt(np.e))
MEASURED = {"profile": None}   # the archive's primary average, set after it is computed (section 7)


def sha256_of(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


SCRIPT_SHA256_AT_LOAD = sha256_of(os.path.abspath(__file__))


# ------------------------------------------------------------------ the shapes

def shape(kind, x, c, d, w=WIDTH):
    """The meridional wind of a trough of depth d whose minimum lies at c. 'gauss' is the anomaly
    module's Gaussian. 'mK' is a trough of zero crossing one width from its minimum whose flanks
    beyond it are K times the Mexican-hat form, so their crest stands at about 0.446 K d. 'dipole'
    has its minimum of -d at c and a maximum of +d two widths east, the meridional wind of an
    isolated cyclonic vortex. 'none' is zero."""
    x = np.asarray(x, float)
    if kind == "gauss":
        return A.gauss(x, c, d, w)
    if kind == "none":
        return np.zeros_like(x)
    if kind == "dipole":
        z = (x - c) / w - 1.0
        return d * SQRT_E * z * np.exp(-z ** 2 / 2.0)
    if kind == "measured":
        if MEASURED["profile"] is None:
            raise SystemExit("REFUSED: the measured profile has not been set")
        P = np.asarray(MEASURED["profile"], float)
        return d * np.interp(x - c, DX, P / abs(P[DX == 0][0]), left=0.0, right=0.0)
    if kind.startswith("m"):
        k = float(kind[1:]); z = (x - c) / w
        core = -d * (1.0 - z ** 2) * np.exp(-z ** 2 / 2.0)
        return np.where(np.abs(z) <= 1.0, core, k * core)
    raise SystemExit(f"REFUSED: unknown shape {kind}")


def calibration_ratio(track, kind, month_of=None):
    """The injection-recovery calibration with the trough's shape as a parameter: a trough of
    J.CAL_DEPTH alone on the track from day J.CAL_START, no noise, the deepest minimum of the
    histories' representation within HI.TOL_DEG of its daily-mean position on each day, their
    mean over J.CAL_DEPTH, None unless at least half the days hold one."""
    month_of = A.month_index() if month_of is None else month_of
    t = np.arange(A.SEASON_DAYS * 4) / 4.0
    V = np.zeros((len(t), NA)); pos = np.full(len(t), np.nan)
    for k, tk in enumerate(t):
        if J.CAL_START <= tk < J.CAL_START + track["days"]:
            pos[k] = track["first_lon"] + track["speed"] * (tk - J.CAL_START)
            V[k] = shape(kind, A.LONS, pos[k], J.CAL_DEPTH)
    mins = HI.rep_minima(W.HISTORIES[0], A.hundredths_from_samples(np.round(V, 2)), month_of, A.LONS)
    centers = pos.reshape(A.SEASON_DAYS, 4).mean(axis=1)
    got = []
    for d in range(J.CAL_START, J.CAL_START + track["days"]):
        near = [m["depth"] for m in (mins[d] or []) if abs(m["lon"] - centers[d]) <= HI.TOL_DEG]
        if near:
            got.append(-min(near))
    if 2 * len(got) < track["days"]:
        return None
    return float(np.mean(got)) / J.CAL_DEPTH


def shape_tracks(hist, month_of=None, shapes=SHAPES):
    """{band: {year: [track]}} with each track's injected depth for every shape, refused unless
    every track has a ratio for every shape."""
    month_of = A.month_index() if month_of is None else month_of
    out = {}
    for (y, b), hs in hist.items():
        for h in hs:
            tr = J.track_of(h); tr["ratio"] = {}; tr["injected"] = {}
            for kind in shapes:
                r = calibration_ratio(tr, kind, month_of)
                if r is None:
                    raise SystemExit(f"REFUSED: the track {tr} has no calibration for the shape {kind}")
                tr["ratio"][kind] = r; tr["injected"][kind] = tr["recorded"] / r
            out.setdefault(b, {}).setdefault(y, []).append(tr)
    return out


def packet_field(tracks, rng, kind):
    """Troughs of the shape on the tracks, drawn exactly as the injection-recovery test draws P1
    (one start per track in order), at each track's injected depth for the shape."""
    t = J.times()[:, 0]; V = np.zeros((J.N_SAMPLES, len(XL)))
    for tr in tracks:
        t0 = int(rng.integers(0, A.SEASON_DAYS - tr["days"] + 1))
        if kind == "none":
            continue
        k0, k1 = t0 * L.OBS_PER_DAY, (t0 + tr["days"]) * L.OBS_PER_DAY
        d = tr["injected"][kind]
        V[k0:k1] += np.array([shape(kind, XL, tr["first_lon"] + tr["speed"] * (tk - t0), d) for tk in t[k0:k1]])
    return V


# ------------------------------------------------------------------ the composite

def b1_rows(S, month_of):
    """The own-month anomaly (each day minus its month's mean in the same season) of the daily
    means on the histories' grid, in m/s, from six-hourly samples on the extended grid."""
    H = A.hundredths_from_samples(S[:, J0:J0 + NA])
    rows, scales = A.exact_anomaly("B1", H, month_of)
    return H, [np.asarray(r, float) / sc for r, sc in zip(rows, scales)]


def composite(rows_by, hist_by, bands):
    """The anomaly summed over every member-day of the histories in the given bands at each
    relative longitude of DX, with the count of member-days contributing at each."""
    s = np.zeros(len(DX)); n = np.zeros(len(DX)); n_hist = 0
    for (y, b), hs in hist_by.items():
        if b not in bands:
            continue
        rows = rows_by[(y, b)]
        for h in hs:
            n_hist += 1
            for d, x in zip(h["days"], h["lons"]):
                j = int(np.flatnonzero(np.isclose(A.LONS, x))[0])
                for k, dx in enumerate(DX):
                    jj = j + int(round(dx / 0.5))
                    if 0 <= jj < NA:
                        s[k] += rows[d][jj]; n[k] += 1
    return {"sum": s.tolist(), "count": n.tolist(), "histories": n_hist}


def flank_readers(comp):
    """The profile and its center, and the largest values from 2 to 8 degrees west and east of
    the member, each over the center's magnitude."""
    s, n = np.asarray(comp["sum"]), np.asarray(comp["count"])
    if (n == 0).any():
        raise SystemExit("REFUSED: a relative longitude of the composite has no member-day")
    P = s / n; c = float(P[DX == 0][0])
    west = (DX >= -FLANK[1]) & (DX <= -FLANK[0]); east = (DX >= FLANK[0]) & (DX <= FLANK[1])
    return {"profile": P.tolist(), "center": c, "west": float(P[west].max()) / abs(c), "east": float(P[east].max()) / abs(c),
            "members": int(n[DX == 0][0]), "histories": comp["histories"]}


def detected(S, month_of):
    """The declared histories of a season's samples, and its anomaly rows."""
    H, rows = b1_rows(S, month_of)
    rep, linker = W.HISTORIES
    hs = [h for h in HI.link(HI.rep_minima(rep, H, month_of, A.LONS), linker) if HI.classify(h) == "westward"]
    return hs, rows


# ------------------------------------------------------------------ constructions

def construction(kind, p, tracks, matched_std, month_of):
    """One placement: matched noise of each band plus troughs of the shape on the band's tracks,
    starts drawn the same for every shape, the histories detected by the declared pipeline and
    averaged around their members."""
    rows_by, hist_by = {}, {}
    for bi, band in enumerate(L.BAND_LAT):
        bg = J.matched_background(matched_std[band], p)
        for i, y in enumerate(YEARS):
            rng = np.random.default_rng([SHAPE_SEED, bi, p, i])
            S = J.inject(bg[i], packet_field(tracks.get(band, {}).get(y, []), rng, kind))
            hist_by[(y, band)], rows_by[(y, band)] = detected(S, month_of)
    return {"primary": flank_readers(composite(rows_by, hist_by, COMPOSITE_BANDS)),
            "all_bands": flank_readers(composite(rows_by, hist_by, tuple(L.BAND_LAT)))}


def run_constructions(tracks, matched_std):
    month_of = A.month_index()
    return {k: [construction(k, p, tracks, matched_std, month_of) for p in range(J.PLACEMENTS)] for k in CONSTRUCTION_SHAPES}


def ranges(cons, reader="primary"):
    return {k: {f: (min(c[reader][f] for c in v), max(c[reader][f] for c in v), float(np.mean([c[reader][f] for c in v]))) for f in ("west", "east")}
            for k, v in cons.items()}


def apply_rule(cons):
    """Section 4: over the five placements of the primary composite, the Gaussian's largest west
    and east flanks lie below the m2.25 trough's smallest, the Gaussian's largest east flank below
    the dipole's smallest, and the dipole's east flank exceeds its west by at least
    SEPARATION_MIN in every placement, the margin taken in decimal on the stored values so that
    an exact stored margin of 0.15 passes (added on the review, section 8)."""
    r = ranges(cons); fails = []
    for f in ("west", "east"):
        if not r["gauss"][f][1] < r["m2.25"][f][0]:
            fails.append(f"gauss and m2.25 overlap on the {f} flank")
    if not r["gauss"]["east"][1] < r["dipole"]["east"][0]:
        fails.append("gauss and dipole overlap on the east flank")
    if any(Decimal(str(c["primary"]["east"])) - Decimal(str(c["primary"]["west"])) < Decimal(str(SEPARATION_MIN)) for c in cons["dipole"]):
        fails.append("a dipole placement is not east-heavy by the declared margin")
    return {"adequate": not fails, "failures": fails}


def check_constructions(path, tracks, matched_std, rerun=None):
    with open(path, encoding="utf-8") as fh:
        rec = json.load(fh)
    if rec["generated_by"]["sha256"] != SCRIPT_SHA256_AT_LOAD:
        raise SystemExit("REFUSED: the constructions were run by a different version of this script")
    fresh = json.loads(json.dumps((rerun or run_constructions)(tracks, matched_std)))
    if fresh != rec.get("results"):
        raise SystemExit("REFUSED: the constructions record differs from a fresh run of the same constructions")
    verdict = apply_rule(rec["results"])
    if not verdict["adequate"]:
        raise SystemExit(f"REFUSED: the constructions do not pass the rule of section 4 ({verdict['failures']})")
    if verdict != rec.get("verdict"):
        raise SystemExit("REFUSED: the stored summary disagrees with the rule applied to the stored cases")
    return rec


# ------------------------------------------------------------------ the archive

def nearest(archive_readers, cons):
    """The construction whose mean west and east flanks lie nearest the archive's, with every
    construction's distance and whether the archive lies inside its placement range on both."""
    r = ranges(cons); a = archive_readers
    d = {k: float(np.hypot(a["west"] - v["west"][2], a["east"] - v["east"][2])) for k, v in r.items()}
    inside = {k: bool(v["west"][0] <= a["west"] <= v["west"][1] and v["east"][0] <= a["east"] <= v["east"][1]) for k, v in r.items()}
    return {"nearest": min(d, key=d.get), "distance": d, "inside": inside}


P1_INDEX = [c["name"] for c in J.configurations()].index("P1")


def shape_p1(kind, backgrounds, tracks_by_year, band_i, lat, bg_i):
    """The injection-recovery test's P1, five placements with its exact seeds, with the trough's
    shape as a parameter."""
    reads = []
    for p in range(J.PLACEMENTS):
        seasons = []
        for i, y in enumerate(YEARS):
            rng = np.random.default_rng([J.INJ_SEED, P1_INDEX, bg_i, band_i, p, i])
            seasons.append(J.inject(backgrounds[p][i], packet_field(tracks_by_year.get(y, []), rng, kind)))
        r = J.read(seasons, lat)
        reads.append({"bases": r, "passes": J.passes(r)})
    return J.summarize_reads(reads)


def injections(by, tracks, inject_rec):
    """Every shape's P1 on both backgrounds in every band, compared with the injection-recovery
    record's read with nothing injected. The Gaussian must reproduce that record's P1."""
    out = {}
    for band_i, (band, lat) in enumerate(L.BAND_LAT.items()):
        e = inject_rec["bands"][band]
        arch = [J.hundredths(L.season_samples(by[y], band)) for y in YEARS]
        bgs = {"archive": [arch] * J.PLACEMENTS, "matched": [J.matched_background(e["matched_noise_std"], p) for p in range(J.PLACEMENTS)]}
        out[band] = {}
        for bg_i, (bg_name, bg) in enumerate(bgs.items()):
            g = e["backgrounds"][bg_name]
            rows = []
            for kind in SHAPES:
                s = json.loads(json.dumps(shape_p1(kind, bg, tracks.get(band, {}), band_i, lat, bg_i)))
                if kind == "gauss":
                    stored = next(r["summary"] for r in g["rows"] if r["config"]["name"] == "P1")
                    if s != stored:
                        raise SystemExit(f"REFUSED: the Gaussian troughs in {band} on the {bg_name} background do not reproduce the injection-recovery record's P1")
                rows.append({"config": {"part": "B", "name": kind}, "summary": s})
            ch = J.change_from_nothing({"nothing_injected": g["nothing_injected"], "rows": rows})
            out[band][bg_name] = {"rows": rows, "change": ch,
                                  "nothing_injected_slopes": {b: v["slope"] for b, v in g["nothing_injected"].items()}}
    return out


def check_sources(by_regime, by_xband):
    """The regime records, where the histories were found, equal the extended records on 20 to
    75 E in every month and band, so one field serves the histories, the composite and the
    injection."""
    for y in YEARS:
        for m in L.MONTHS:
            r, x = by_regime[y][m], by_xband[y][m]
            if list(r["times"]) != list(x["times"]) or not np.array_equal(np.asarray(r["lons"], float), A.LONS):
                raise SystemExit(f"REFUSED: {y}-{m:02d} regime and extended records differ in times or grid")
            for b in r["bands"]:
                if not np.array_equal(np.asarray(r["band_means_700"][b], float), np.asarray(x["band_means_700"][b], float)[:, J0:J0 + NA]):
                    raise SystemExit(f"REFUSED: {y}-{m:02d} {b} regime and extended records differ on 20 to 75 E")


def archive_composite(by_xband, hist, month_of):
    rows_by = {}
    for (y, b) in hist:
        S = L.season_samples(by_xband[y], b)
        rows_by[(y, b)] = b1_rows(S, month_of)[1]
    return {"primary": flank_readers(composite(rows_by, hist, COMPOSITE_BANDS)),
            "all_bands": flank_readers(composite(rows_by, hist, tuple(L.BAND_LAT)))}


def readings(res):
    """Section 2.5's generated sentences."""
    a = res["composite"]["primary"]; nr = res["nearest"]
    s = [f"the own-month anomaly averaged over the {a['members']} member-days of the {a['histories']} westward histories in 15 to 24.5 N has a center of "
         f"{a['center']:.2f} m/s and west and east flanks of {a['west']:.2f} and {a['east']:.2f} of the center's size, nearest the {nr['nearest']} construction, "
         + ("inside the placement range of " + ", ".join(k for k, v in nr["inside"].items() if v) if any(nr["inside"].values()) else "inside no construction's placement range")]
    for band, d in res["injections"].items():
        for bg, g in d.items():
            for kind in SHAPES:
                c = g["change"][kind]
                parts = [f"{k.replace('_', ' ')} at {', '.join(c[k])} E" for k in ("new_westward", "new_moving", "lost_westward", "lost_moving") if c[k]]
                s.append(f"{band}, {bg} background, {kind} troughs on the history tracks: " + ("; ".join(parts) if parts else "no change of westward or moving status"))
    return s


# ------------------------------------------------------------------ a reader added after the archive was read (section 7)

def measured_injection(by, hist, profile, inject_rec):
    """The archive's own primary average used as the trough's shape, centered on its member
    longitude and scaled so its center is -d, zero beyond 15 degrees, calibrated on every track
    as the other shapes are, and injected as P1 with the injection-recovery test's seeds on both
    backgrounds in every band. Added after the archive was read, because the measured average
    lay inside no construction's range."""
    MEASURED["profile"] = list(profile)
    tracks = shape_tracks(hist, shapes=("measured",))
    out = {}
    for band_i, (band, lat) in enumerate(L.BAND_LAT.items()):
        e = inject_rec["bands"][band]
        arch = [J.hundredths(L.season_samples(by[y], band)) for y in YEARS]
        bgs = {"archive": [arch] * J.PLACEMENTS, "matched": [J.matched_background(e["matched_noise_std"], p) for p in range(J.PLACEMENTS)]}
        out[band] = {}
        for bg_i, (bg_name, bg) in enumerate(bgs.items()):
            g = e["backgrounds"][bg_name]
            s = json.loads(json.dumps(shape_p1("measured", bg, tracks.get(band, {}), band_i, lat, bg_i)))
            rows = [{"config": {"part": "B", "name": "measured"}, "summary": s}]
            out[band][bg_name] = {"rows": rows, "change": J.change_from_nothing({"nothing_injected": g["nothing_injected"], "rows": rows})}
    ratios = [t["ratio"]["measured"] for d in tracks.values() for v in d.values() for t in v]
    return {"injections": out, "ratio_median": float(np.median(ratios)), "ratio_min": float(min(ratios))}


def measured_sentences(m):
    out = []
    for band, d in m["injections"].items():
        for bg, g in d.items():
            c = g["change"]["measured"]
            parts = [f"{k.replace('_', ' ')} at {', '.join(c[k])} E" for k in ("new_westward", "new_moving", "lost_westward", "lost_moving") if c[k]]
            out.append(f"{band}, {bg} background, troughs of the measured average on the history tracks: "
                       + ("; ".join(parts) if parts else "no change of westward or moving status")
                       + f", the median slope moving by at most {fmt(c['max_slope_change'])} degrees per day")
    return out


# ------------------------------------------------------------------ text and main

def fmt(v, f=".2f"):
    return "" if v is None else format(v, f)


def constructions_text(cons, verdict):
    r = ranges(cons); ra = ranges(cons, "all_bands")
    out = ["# history shape, the constructions (section 3)", "",
           "| shape | west flank over center, 15 to 24.5 N (min, max, mean) | east flank | histories per placement | west, all bands | east, all bands |",
           "| --- | --- | --- | --- | --- | --- |"]
    for k in CONSTRUCTION_SHAPES:
        n = ", ".join(str(c["primary"]["histories"]) for c in cons[k])
        out.append(f"| {k} | {r[k]['west'][0]:.2f}, {r[k]['west'][1]:.2f}, {r[k]['west'][2]:.2f} | {r[k]['east'][0]:.2f}, {r[k]['east'][1]:.2f}, {r[k]['east'][2]:.2f} | {n} | "
                   f"{ra[k]['west'][2]:.2f} | {ra[k]['east'][2]:.2f} |")
    out += ["", f"adequate under the rule of section 4, {'yes' if verdict['adequate'] else 'no'}"] + [f"- {x}" for x in verdict["failures"]]
    return "\n".join(out) + "\n"


def archive_text(res):
    a = res["composite"]; nr = res["nearest"]
    out = ["# history shape, the archive (section 7)", "",
           "| composite | member-days | histories | center (m/s) | west flank over center | east flank over center |", "| --- | --- | --- | --- | --- | --- |"]
    for k in ("primary", "all_bands"):
        out.append(f"| {k} | {a[k]['members']} | {a[k]['histories']} | {a[k]['center']:.2f} | {a[k]['west']:.2f} | {a[k]['east']:.2f} |")
    out += ["", "| relative longitude (deg) | " + " | ".join(f"{x:g}" for x in DX[::4]) + " |", "| --- | " + " | ".join("---" for _ in DX[::4]) + " |",
            "| archive, 15 to 24.5 N (m/s) | " + " | ".join(format(v, ".2f") for v in np.asarray(a["primary"]["profile"])[::4]) + " |", "",
            "| construction | distance from the archive | archive inside its range |", "| --- | --- | --- |"]
    for k, d in nr["distance"].items():
        out.append(f"| {k} | {d:.3f} | {'yes' if nr['inside'][k] else 'no'} |")
    out += ["", "| band | background | shape | new westward | new moving | lost westward | lost moving | largest median slope change | median slope by base 45 to 70 E |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for band, d in res["injections"].items():
        for bg, g in d.items():
            for row in g["rows"]:
                k = row["config"]["name"]; c = g["change"][k]
                sl = ", ".join(fmt(v["median_slope"]) for v in row["summary"]["bases"].values())
                out.append(f"| {band} | {bg} | {k} | {' '.join(c['new_westward'])} | {' '.join(c['new_moving'])} | {' '.join(c['lost_westward'])} | "
                           f"{' '.join(c['lost_moving'])} | {fmt(c['max_slope_change'])} | {sl} |")
            out.append(f"| {band} | {bg} | nothing injected | | | | | | {', '.join(fmt(v) for v in g['nothing_injected_slopes'].values())} |")
    out += [""] + [f"- {x}" for x in res["readings"]]
    if "post_hoc" in res:
        m = res["post_hoc"]
        out += ["", "## a reader added after the archive was read (section 7)", "",
                f"calibration ratio of the measured shape over the 104 tracks, median {m['ratio_median']:.3f}, smallest {m['ratio_min']:.3f}", "",
                "| band | background | new westward | new moving | lost westward | lost moving | largest median slope change | median slope by base 45 to 70 E |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for band, d in m["injections"].items():
            for bg, g in d.items():
                c = g["change"]["measured"]; sl = ", ".join(fmt(v["median_slope"]) for v in g["rows"][0]["summary"]["bases"].values())
                out.append(f"| {band} | {bg} | {' '.join(c['new_westward'])} | {' '.join(c['new_moving'])} | {' '.join(c['lost_westward'])} | "
                           f"{' '.join(c['lost_moving'])} | {fmt(c['max_slope_change'])} | {sl} |")
        out += [""] + [f"- {x}" for x in res["post_hoc_readings"]]
    return "\n".join(out) + "\n"


def refuse_existing(paths):
    for p in paths:
        if os.path.exists(p):
            raise SystemExit(f"REFUSED: {p} exists and a record is never overwritten")


def load_gz(files):
    out = []
    for p in files:
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            out.append(json.load(fh))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("constructions", "archive"))
    ap.add_argument("--out-dir", required=True)
    for opt in ("--inject", "--records", "--histories", "--xband", "--lagreg", "--constructions"):
        ap.add_argument(opt, default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    if not (a.inject and a.records and a.histories):
        raise SystemExit("REFUSED: --inject, --records and --histories are required")
    with open(a.inject, encoding="utf-8") as fh:
        inject_full = json.load(fh)
    if inject_full["generated_by"]["sha256"] != J.SCRIPT_SHA256_AT_LOAD:
        raise SystemExit("REFUSED: the injection-recovery record was made by a different version of its script")
    inject_rec = inject_full["results"]
    matched_std = {b: e["matched_noise_std"] for b, e in inject_rec["bands"].items()}
    hist, rfiles = J.load_histories(a.records, a.histories)
    tracks = shape_tracks(hist)
    gen = {"script": "scripts/eastern_history_shape.py", "sha256": SCRIPT_SHA256_AT_LOAD, "declared": "EASTERN_HISTORY_SHAPE_2026-10-21.md",
           "inject": {"path": a.inject, "sha256": sha256_of(a.inject)}, "records": {os.path.basename(p): sha256_of(p) for p in rfiles},
           "histories": {"path": a.histories, "sha256": sha256_of(a.histories)}}
    if a.mode == "constructions":
        pj, pm = (os.path.join(a.out_dir, f"eval60_shape_constructions{a.tag}.{e}") for e in ("json", "md"))
        refuse_existing([pj, pm])
        cons = json.loads(json.dumps(run_constructions(tracks, matched_std))); verdict = apply_rule(cons)
        with open(pj, "w", encoding="utf-8") as fh:
            json.dump({"generated_by": gen, "results": cons, "verdict": verdict}, fh)
        text = constructions_text(cons, verdict)
        with open(pm, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(text)
        return 0
    if not (a.xband and a.lagreg and a.constructions):
        raise SystemExit("REFUSED: --xband, --lagreg and --constructions are required for the archive")
    cons = check_constructions(a.constructions, tracks, matched_std)["results"]
    pj, pm = (os.path.join(a.out_dir, f"eval60_shape_{k}{a.tag}.{e}") for k, e in (("archive", "json"), ("tables", "md")))
    refuse_existing([pj, pm])
    xfiles = sorted(glob.glob(os.path.join(a.xband, "eval60_xband_????_??.json.gz")))
    xrecs = load_gz(xfiles)
    X.check_xband_identity(xfiles, xrecs)
    L.check_inputs(xrecs)
    by_x = {}
    for r in xrecs:
        by_x.setdefault(int(r["year"]), {})[int(r["month"])] = r
    by_r = {}
    for p in sorted(glob.glob(os.path.join(a.records, "eval60_regime_????_??.json.gz"))):
        r = J.SC.load_record(p); by_r.setdefault(int(r["year"]), {})[int(r["month"])] = r
    check_sources(by_r, by_x)
    with open(a.lagreg, encoding="utf-8") as fh:
        J.check_zero_injection(by_x, json.load(fh))
    month_of = A.month_index()
    res = {"composite": archive_composite(by_x, hist, month_of)}
    res["nearest"] = nearest(res["composite"]["primary"], cons)
    res["injections"] = injections(by_x, tracks, inject_rec)
    res["readings"] = readings(res)
    res["post_hoc"] = measured_injection(by_x, hist, res["composite"]["primary"]["profile"], inject_rec)
    res["post_hoc_readings"] = measured_sentences(res["post_hoc"])
    res["tracks"] = {b: {str(y): v for y, v in d.items()} for b, d in tracks.items()}
    gen.update({"xband": {os.path.basename(p): sha256_of(p) for p in xfiles}, "lagreg": {"path": a.lagreg, "sha256": sha256_of(a.lagreg)},
                "constructions": {"path": a.constructions, "sha256": sha256_of(a.constructions)}})
    with open(pj, "w", encoding="utf-8") as fh:
        json.dump({"generated_by": gen, "results": res}, fh)
    text = archive_text(res)
    with open(pm, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
