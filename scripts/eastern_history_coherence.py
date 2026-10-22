#!/usr/bin/env python3
"""The coherence of the westward histories (`EASTERN_HISTORY_COHERENCE_2026-10-22.md`): whether
the own-month anomaly at a history's first longitude persists after its minimum has moved on,
which a minimum migrating within a standing structure leaves behind and a traveling trough
does not, read against constructions of traveling troughs of the measured average shape on
the histories' tracks with a standing trough beneath them of increasing relative depth,
detected by the same pipeline, on the archive and on matched noise. The measure says whether
a persistent structure is left behind. It does not separate coherent travel from minima that
noise produces, it identifies no feature, and it does not say what any one history is.

    python3 scripts/eastern_history_coherence.py constructions --shape PATH --inject PATH --records DIR --histories PATH --xband DIR --out-dir DIR
    python3 scripts/eastern_history_coherence.py archive --shape PATH --inject PATH --records DIR --histories PATH --xband DIR \\
        --lagreg PATH --constructions PATH --out-dir DIR
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
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
import eastern_history_shape as H  # noqa: E402

J, L, A, HI, X = H.J, H.L, H.A, H.HI, H.X
YEARS = H.YEARS
XL = H.XL
BANDS = H.COMPOSITE_BANDS
PLATEAUS = (0.0, 0.5, 1.0, 2.0)
LAG_DAYS = (2, 3, 4)
CURVE_DAYS = tuple(range(0, 7))
MARGIN_DAYS = 2
COH_SEED = 2026102202
PLATEAU_WIDTH = 2.0


def sha256_of(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


SCRIPT_SHA256_AT_LOAD = sha256_of(os.path.abspath(__file__))


# ------------------------------------------------------------------ the constructions' fields

def plateau(x_a, x_b, xs):
    """A flat-bottomed trough of depth 1 spanning x_a to x_b, the mean of Gaussian troughs of the
    anomaly module's width centered every half degree between them, scaled so its minimum is -1."""
    lo, hi = min(x_a, x_b), max(x_a, x_b)
    P = np.mean([A.gauss(xs, c, 1.0, PLATEAU_WIDTH) for c in np.arange(lo, hi + 1e-9, 0.5)], axis=0)
    return P / abs(P.min())


def build(tracks, rng, p):
    """Troughs of the measured average shape on the tracks, drawn as the shape test's P1 draws
    them (one whole-day start per track in order), each over a standing plateau of p times its
    depth spanning its track's first and last positions and present from MARGIN_DAYS before
    its start to MARGIN_DAYS after its end. Returns the field and each trough's daily-mean
    position."""
    t = J.times()[:, 0]; V = np.zeros((J.N_SAMPLES, len(XL))); centers = []
    for tr in tracks:
        n = tr["days"]; t0 = int(rng.integers(0, A.SEASON_DAYS - n + 1)); d = tr["injected"]["measured"]
        k0, k1 = t0 * L.OBS_PER_DAY, (t0 + n) * L.OBS_PER_DAY
        pos = np.full(J.N_SAMPLES, np.nan); pos[k0:k1] = tr["first_lon"] + tr["speed"] * (t[k0:k1] - t0)
        V[k0:k1] += np.array([H.shape("measured", XL, x, d) for x in pos[k0:k1]])
        if p > 0:
            a0 = max(0, (t0 - MARGIN_DAYS) * L.OBS_PER_DAY); a1 = min(J.N_SAMPLES, (t0 + n + MARGIN_DAYS) * L.OBS_PER_DAY)
            V[a0:a1] += p * d * plateau(tr["first_lon"], tr["first_lon"] + tr["speed"] * (n - 1), XL)
        centers.append(pos.reshape(A.SEASON_DAYS, L.OBS_PER_DAY).mean(axis=1))
    return V, centers


# ------------------------------------------------------------------ the measure

def col(x):
    return int(np.flatnonzero(np.isclose(A.LONS, x))[0])


def ends(rows, h):
    """For a history whose first day has CURVE_DAYS[-1] days of season after it and whose last day
    has as many before it, the anomaly at the first longitude on the first day and on each later
    day of CURVE_DAYS, and at the last longitude on the last day and on each earlier day; None
    otherwise."""
    d0, d1 = h["days"][0], h["days"][-1]
    if d0 + CURVE_DAYS[-1] >= A.SEASON_DAYS or d1 - CURVE_DAYS[-1] < 0:
        return None
    j0, j1 = col(h["lons"][0]), col(h["lons"][-1])
    return {"departure": [float(rows[d0 + k][j0]) for k in CURVE_DAYS], "arrival": [float(rows[d1 - k][j1]) for k in CURVE_DAYS]}


def pooled(series):
    """For departure and arrival, the ratio of the summed anomalies to the summed anomalies on
    the reference day, at each day of CURVE_DAYS, and the ratio of the anomaly averaged over
    LAG_DAYS, summed, to the reference day's sum, which is the reader."""
    out = {"histories": len(series)}
    if not series:
        return out
    for key in ("departure", "arrival"):
        M = np.array([s[key] for s in series]); ref = M[:, 0].sum()
        if ref >= 0:
            raise SystemExit("REFUSED: the summed anomaly at the reference day is not negative")
        out[f"{key}_curve"] = [float(M[:, k].sum() / ref) for k in range(len(CURVE_DAYS))]
        out[key] = float(M[:, [CURVE_DAYS.index(k) for k in LAG_DAYS]].mean(axis=1).sum() / ref)
    return out


def measure(rows_by, hist_by, bands):
    series = []
    for (y, b), hs in hist_by.items():
        if b not in bands:
            continue
        for h in hs:
            s = ends(rows_by[(y, b)], h)
            if s is not None:
                series.append(s)
    return pooled(series)


# ------------------------------------------------------------------ constructions

def construction(p, bg_name, place, tracks, by, matched_std, month_of):
    """One placement of one plateau depth on one background: the detected histories that follow
    an injected trough, measured. On the archive, real histories that follow no injected trough
    are left out."""
    rows_by, hist_by = {}, {}
    for bi, band in enumerate(L.BAND_LAT):
        if band not in BANDS:
            continue
        bg = (J.matched_background(matched_std[band], place) if bg_name == "matched"
              else [J.hundredths(L.season_samples(by[y], band)) for y in YEARS])
        for i, y in enumerate(YEARS):
            rng = np.random.default_rng([COH_SEED, bi, place, i])
            V, centers = build(tracks.get(band, {}).get(y, []), rng, p)
            hs, rows = H.detected(J.inject(bg[i], V), month_of)
            hist_by[(y, band)] = [h for h in hs if any(HI.follows(h, c) for c in centers)]
            rows_by[(y, band)] = rows
    return measure(rows_by, hist_by, BANDS)


def noise_only(place, matched_std, month_of):
    rows_by, hist_by = {}, {}
    for band in BANDS:
        bg = J.matched_background(matched_std[band], place)
        for i, y in enumerate(YEARS):
            hist_by[(y, band)], rows_by[(y, band)] = H.detected(bg[i] / 100.0, month_of)
    return measure(rows_by, hist_by, BANDS)


def run_constructions(tracks, by, matched_std):
    month_of = A.month_index()
    out = {bg: {f"{p:g}": [construction(p, bg, k, tracks, by, matched_std, month_of) for k in range(J.PLACEMENTS)] for p in PLATEAUS}
           for bg in ("archive", "matched")}
    out["noise"] = [noise_only(k, matched_std, month_of) for k in range(J.PLACEMENTS)]
    return out


def ranges(cons, bg, key="departure"):
    return {p: (min(c[key] for c in v), max(c[key] for c in v), float(np.mean([c[key] for c in v]))) for p, v in cons[bg].items()}


def apply_rule(cons):
    """Section 4: on each background, the departure readers' placement ranges of the plateau
    depths 0, 1 and 2 are disjoint and ordered, the largest of the lower depth below the smallest
    of the higher."""
    fails = []
    for bg in ("archive", "matched"):
        r = ranges(cons, bg)
        for lo, hi in (("0", "1"), ("1", "2")):
            if not r[lo][1] < r[hi][0]:
                fails.append(f"{bg}: plateau {lo} and plateau {hi} overlap on departure")
    return {"adequate": not fails, "failures": fails}


def check_constructions(path, tracks, by, matched_std, rerun=None):
    with open(path, encoding="utf-8") as fh:
        rec = json.load(fh)
    if rec["generated_by"]["sha256"] != SCRIPT_SHA256_AT_LOAD:
        raise SystemExit("REFUSED: the constructions were run by a different version of this script")
    fresh = json.loads(json.dumps((rerun or run_constructions)(tracks, by, matched_std)))
    if fresh != rec.get("results"):
        raise SystemExit("REFUSED: the constructions record differs from a fresh run of the same constructions")
    verdict = apply_rule(rec["results"])
    if not verdict["adequate"]:
        raise SystemExit(f"REFUSED: the constructions do not pass the rule of section 4 ({verdict['failures']})")
    if verdict != rec.get("verdict"):
        raise SystemExit("REFUSED: the stored summary disagrees with the rule applied to the stored cases")
    return rec


# ------------------------------------------------------------------ the archive

def standing_depth(value, cons, bg="archive", key="departure"):
    """The plateau depth at which the constructions' mean reader equals the value, by linear
    interpolation between the declared depths, None outside their span, with the depths whose
    placement range contains the value."""
    r = ranges(cons, bg, key); ps = sorted(r, key=float)
    means = [r[p][2] for p in ps]
    inside = [p for p in ps if r[p][0] <= value <= r[p][1]]
    est = None
    for (pa, ma), (pb, mb) in zip(zip(ps, means), zip(ps[1:], means[1:])):
        if min(ma, mb) <= value <= max(ma, mb) and mb != ma:
            est = float(pa) + (value - ma) / (mb - ma) * (float(pb) - float(pa))
            break
    return {"estimate": est, "inside": inside, "means": dict(zip(ps, means))}


def archive_measure(by, hist, month_of):
    rows_by = {k: H.b1_rows(L.season_samples(by[k[0]], k[1]), month_of)[1] for k in hist}
    return {"primary": measure(rows_by, hist, BANDS), "all_bands": measure(rows_by, hist, tuple(L.BAND_LAT))}


def readings(res):
    a = res["measure"]["primary"]; s = res["standing"]
    noise = res["noise_range"]
    out = [f"the anomaly at the first longitude of the {a['histories']} westward histories in 15 to 24.5 N, averaged over 2 to 4 days after their first day, "
           f"is {a['departure']:.2f} of its value on the first day, "
           + (f"which the archive-background constructions place at a standing plateau of {s['departure']['estimate']:.2f} times the traveling trough's depth"
              if s["departure"]["estimate"] is not None else "outside the span of the archive-background constructions")
           + ("" if not s["departure"]["inside"] else f", inside the placement range of plateau depth {', '.join(s['departure']['inside'])}")
           + f", and noise alone gives {noise[0]:.2f} to {noise[1]:.2f}",
           f"the anomaly at their last longitude, averaged over 2 to 4 days before their last day, is {a['arrival']:.2f} of its value on the last day"]
    return out


# ------------------------------------------------------------------ text and main

def constructions_text(cons, verdict):
    out = ["# history coherence, the constructions (section 3)", "",
           "| background | plateau depth | departure (min, max, mean) | arrival (min, max, mean) | histories per placement |", "| --- | --- | --- | --- | --- |"]
    for bg in ("archive", "matched"):
        rd, ra = ranges(cons, bg), ranges(cons, bg, "arrival")
        for p in rd:
            n = ", ".join(str(c["histories"]) for c in cons[bg][p])
            out.append(f"| {bg} | {p} | {rd[p][0]:.2f}, {rd[p][1]:.2f}, {rd[p][2]:.2f} | {ra[p][0]:.2f}, {ra[p][1]:.2f}, {ra[p][2]:.2f} | {n} |")
    nd = [c["departure"] for c in cons["noise"]]; na = [c["arrival"] for c in cons["noise"]]
    out.append(f"| matched | noise alone | {min(nd):.2f}, {max(nd):.2f}, {np.mean(nd):.2f} | {min(na):.2f}, {max(na):.2f}, {np.mean(na):.2f} | "
               f"{', '.join(str(c['histories']) for c in cons['noise'])} |")
    out += ["", f"adequate under the rule of section 4, {'yes' if verdict['adequate'] else 'no'}"] + [f"- {x}" for x in verdict["failures"]]
    return "\n".join(out) + "\n"


def archive_text(res):
    out = ["# history coherence, the archive (section 7)", "",
           "| histories | count | departure | arrival | departure by day 0 to 6 | arrival by day 0 to 6 |", "| --- | --- | --- | --- | --- | --- |"]
    for k in ("primary", "all_bands"):
        a = res["measure"][k]
        out.append(f"| {k} | {a['histories']} | {a['departure']:.3f} | {a['arrival']:.3f} | {', '.join(format(v, '.2f') for v in a['departure_curve'])} | "
                   f"{', '.join(format(v, '.2f') for v in a['arrival_curve'])} |")
    out += ["", "| reader | standing plateau estimate on the archive background | inside the range of plateau depth | construction means |", "| --- | --- | --- | --- |"]
    for key in ("departure", "arrival"):
        s = res["standing"][key]
        out.append(f"| {key} | {'outside the span' if s['estimate'] is None else format(s['estimate'], '.2f')} | {', '.join(s['inside']) or 'none'} | "
                   f"{', '.join(f'{p}: {m:.2f}' for p, m in s['means'].items())} |")
    out += [""] + [f"- {x}" for x in res["readings"]]
    return "\n".join(out) + "\n"


def refuse_existing(paths):
    for p in paths:
        if os.path.exists(p):
            raise SystemExit(f"REFUSED: {p} exists and a record is never overwritten")


def load_inputs(a):
    """The shape record's measured average (made by the shape script as loaded), the injection
    record's matched noise levels (made by its script as loaded), the recomputed histories with
    the measured shape's calibration, and the extended records checked as in the lag-regression
    record and against the regime records."""
    with open(a.shape, encoding="utf-8") as fh:
        shape_rec = json.load(fh)
    if shape_rec["generated_by"]["sha256"] != H.SCRIPT_SHA256_AT_LOAD:
        raise SystemExit("REFUSED: the shape record was made by a different version of its script")
    with open(a.inject, encoding="utf-8") as fh:
        inject_rec = json.load(fh)
    if inject_rec["generated_by"]["sha256"] != J.SCRIPT_SHA256_AT_LOAD:
        raise SystemExit("REFUSED: the injection-recovery record was made by a different version of its script")
    H.MEASURED["profile"] = shape_rec["results"]["composite"]["primary"]["profile"]
    matched_std = {b: e["matched_noise_std"] for b, e in inject_rec["results"]["bands"].items()}
    hist, rfiles = J.load_histories(a.records, a.histories)
    tracks = H.shape_tracks(hist, shapes=("measured",))
    xfiles = sorted(glob.glob(os.path.join(a.xband, "eval60_xband_????_??.json.gz")))
    xrecs = H.load_gz(xfiles)
    X.check_xband_identity(xfiles, xrecs)
    L.check_inputs(xrecs)
    by = {}
    for r in xrecs:
        by.setdefault(int(r["year"]), {})[int(r["month"])] = r
    by_r = {}
    for p in rfiles:
        r = J.SC.load_record(p); by_r.setdefault(int(r["year"]), {})[int(r["month"])] = r
    H.check_sources(by_r, by)
    gen = {"script": "scripts/eastern_history_coherence.py", "sha256": SCRIPT_SHA256_AT_LOAD, "declared": "EASTERN_HISTORY_COHERENCE_2026-10-22.md",
           "shape": {"path": a.shape, "sha256": sha256_of(a.shape)}, "inject": {"path": a.inject, "sha256": sha256_of(a.inject)},
           "records": {os.path.basename(p): sha256_of(p) for p in rfiles}, "histories": {"path": a.histories, "sha256": sha256_of(a.histories)},
           "xband": {os.path.basename(p): sha256_of(p) for p in xfiles}}
    return hist, tracks, by, matched_std, gen


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("constructions", "archive"))
    ap.add_argument("--out-dir", required=True)
    for opt in ("--shape", "--inject", "--records", "--histories", "--xband", "--lagreg", "--constructions"):
        ap.add_argument(opt, default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    if not all((a.shape, a.inject, a.records, a.histories, a.xband)):
        raise SystemExit("REFUSED: --shape, --inject, --records, --histories and --xband are required")
    hist, tracks, by, matched_std, gen = load_inputs(a)
    if a.mode == "constructions":
        pj, pm = (os.path.join(a.out_dir, f"eval60_coherence_constructions{a.tag}.{e}") for e in ("json", "md"))
        refuse_existing([pj, pm])
        cons = json.loads(json.dumps(run_constructions(tracks, by, matched_std))); verdict = apply_rule(cons)
        with open(pj, "w", encoding="utf-8") as fh:
            json.dump({"generated_by": gen, "results": cons, "verdict": verdict}, fh)
        text = constructions_text(cons, verdict)
        with open(pm, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(text)
        return 0
    if not (a.lagreg and a.constructions):
        raise SystemExit("REFUSED: --lagreg and --constructions are required for the archive")
    with open(a.lagreg, encoding="utf-8") as fh:
        J.check_zero_injection(by, json.load(fh))
    cons = check_constructions(a.constructions, tracks, by, matched_std)["results"]
    pj, pm = (os.path.join(a.out_dir, f"eval60_coherence_{k}{a.tag}.{e}") for k, e in (("archive", "json"), ("tables", "md")))
    refuse_existing([pj, pm])
    res = {"measure": archive_measure(by, hist, A.month_index())}
    res["standing"] = {k: standing_depth(res["measure"]["primary"][k], cons, key=k) for k in ("departure", "arrival")}
    nd = [c["departure"] for c in cons["noise"]]
    res["noise_range"] = [min(nd), max(nd)]
    res["readings"] = readings(res)
    gen.update({"lagreg": {"path": a.lagreg, "sha256": sha256_of(a.lagreg)}, "constructions": {"path": a.constructions, "sha256": sha256_of(a.constructions)}})
    with open(pj, "w", encoding="utf-8") as fh:
        json.dump({"generated_by": gen, "results": res}, fh)
    text = archive_text(res)
    with open(pm, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
