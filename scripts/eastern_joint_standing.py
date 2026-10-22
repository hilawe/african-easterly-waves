#!/usr/bin/env python3
"""Standing southerly lobes against the westward histories' readings together
(`EASTERN_JOINT_STANDING_2026-10-22.md`): which layouts of a moving trough core with a southerly
lobe standing at a fixed offset from each history's first longitude, crossed over the lobe's
offset, width, strength and timing, the coherence departure, the lobe-position contrast and the
anomaly 6 degrees east of the first longitude on the first day leave consistent with the archive
when read together. The constructions use the lobe-position test's starts, so its traveling and
pinned constructions are reproduced exactly. The measure says which declared layouts stay
consistent with the archive's readings. It excludes no layout outside the family, identifies no
feature, and does not say what any one history is.

    python3 scripts/eastern_joint_standing.py constructions --shape PATH --inject PATH --records DIR --histories PATH --xband DIR \\
        --lobe-constructions PATH --out-dir DIR
    python3 scripts/eastern_joint_standing.py archive --shape PATH --inject PATH --records DIR --histories PATH --xband DIR \\
        --lagreg PATH --lobe-constructions PATH --coherence-archive PATH --lobe-archive PATH --constructions PATH --out-dir DIR
"""

import argparse
import hashlib
import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
import eastern_lobe_position as P  # noqa: E402

C, H, J, L, A, HI = P.C, P.H, P.J, P.L, P.A, P.HI
YEARS = P.YEARS
XL = P.XL
BANDS = P.BANDS
OFFSETS = (-4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0)
WIDTHS = (1.0, 2.0, 4.0)
STRENGTHS = (1.0, 1.5)
TIMINGS = ("steady", "relax", "long")
LONG_DAYS = 15
READERS = ("departure", "contrast", "first_day")
PROFILE_DAYS = (0, 1, 2, 3, 4)
NARROW_AT_START = ("o0_w1_s1_steady", "o0_w2_s1_steady")
REPRODUCED = {"travel": "travel", "o6_w2_s1_steady": "pinned", "o6_w2_s1_relax": "pinned_relax"}


def sha256_of(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


SCRIPT_SHA256_AT_LOAD = sha256_of(os.path.abspath(__file__))


# ------------------------------------------------------------------ the layouts

def layout_name(lay):
    if lay["kind"] == "travel":
        return "travel"
    return f"o{lay['offset']:g}_w{lay['width']:g}_s{lay['strength']:g}_{lay['timing']}"


def layouts():
    """Travel, then every standing lobe of the declared offsets, widths, strengths and timings."""
    out = [{"kind": "travel"}]
    for o, w, s, t in itertools.product(OFFSETS, WIDTHS, STRENGTHS, TIMINGS):
        out.append({"kind": "standing", "offset": o, "width": w, "strength": s, "timing": t})
    return out


def standing_rows(t0, n, timing):
    """The sample ranges and multipliers over which a standing lobe is present: 'steady' from
    MARGIN_DAYS before the start to MARGIN_DAYS after the end, 'relax' at twice the strength for
    RELAX_BEFORE_DAYS before the start and then as steady, 'long' from LONG_DAYS before the start
    to LONG_DAYS after the end."""
    ob = L.OBS_PER_DAY
    end = min(J.N_SAMPLES, (t0 + n + P.MARGIN_DAYS) * ob)
    if timing == "steady":
        return [(max(0, (t0 - P.MARGIN_DAYS) * ob), end, 1.0)]
    if timing == "relax":
        return [(max(0, (t0 - P.RELAX_BEFORE_DAYS) * ob), t0 * ob, 2.0), (t0 * ob, end, 1.0)]
    if timing == "long":
        return [(max(0, (t0 - LONG_DAYS) * ob), min(J.N_SAMPLES, (t0 + n + LONG_DAYS) * ob), 1.0)]
    raise SystemExit(f"REFUSED: unknown timing {timing}")


def build(tracks, rng, lay, parts):
    """On the tracks, drawn as the lobe-position test draws them (one whole-day start per track
    in order): 'travel', troughs of the measured shape; otherwise the measured shape's core moving
    on the track with a Gaussian southerly lobe of the layout's width and of its strength times
    the shape's lobe amplitude, standing at the layout's offset from the track's first longitude
    over the layout's timing. Returns the field and each trough's daily-mean position."""
    t = J.times()[:, 0]; V = np.zeros((J.N_SAMPLES, len(XL))); centers = []
    for tr in tracks:
        n = tr["days"]; t0 = int(rng.integers(0, A.SEASON_DAYS - n + 1)); d = tr["injected"]["measured"]
        k0, k1 = t0 * L.OBS_PER_DAY, (t0 + n) * L.OBS_PER_DAY
        pos = np.full(J.N_SAMPLES, np.nan); pos[k0:k1] = tr["first_lon"] + tr["speed"] * (t[k0:k1] - t0)
        if lay["kind"] == "travel":
            V[k0:k1] += np.array([H.shape("measured", XL, x, d) for x in pos[k0:k1]])
        elif lay["kind"] == "standing":
            V[k0:k1] += np.array([P.core_shape(XL, x, d, parts["core"]) for x in pos[k0:k1]])
            lobe = -A.gauss(XL, tr["first_lon"] + lay["offset"], lay["strength"] * parts["lobe"] * d, lay["width"])
            for a, b, m in standing_rows(t0, n, lay["timing"]):
                V[a:b] += m * lobe
        else:
            raise SystemExit(f"REFUSED: unknown layout {lay['kind']}")
        centers.append(pos.reshape(A.SEASON_DAYS, L.OBS_PER_DAY).mean(axis=1))
    return V, centers


# ------------------------------------------------------------------ the readers

def profile(b0_by, hist_by, bands):
    """For each day of PROFILE_DAYS, the other-season anomaly 6 degrees east of the history's
    member on that day, summed over the histories, over the summed magnitude of the anomaly at the
    member, skipping days whose eastern position lies east of 75 E. Day 0 is the first-day
    reader, where the member is the first longitude."""
    num = {k: 0.0 for k in PROFILE_DAYS}; den = {k: 0.0 for k in PROFILE_DAYS}; n = {k: 0 for k in PROFILE_DAYS}; off = {k: 0 for k in PROFILE_DAYS}
    for (y, b), hs in hist_by.items():
        if b not in bands:
            continue
        rows = b0_by[(y, b)]
        for h in hs:
            for k in PROFILE_DAYS:
                if k >= len(h["lons"]):
                    continue
                xk, d = h["lons"][k], h["days"][k]
                if xk + P.OFFSET > A.LONS[-1]:
                    off[k] += 1
                    continue
                num[k] += float(rows[d][P.col(xk + P.OFFSET)]); den[k] += abs(float(rows[d][P.col(xk)])); n[k] += 1
    if n[0] == 0 or den[0] <= 0:
        raise SystemExit("REFUSED: no first day enters the measure")
    return {"by_day": [num[k] / den[k] if den[k] > 0 else None for k in PROFILE_DAYS],
            "days": [n[k] for k in PROFILE_DAYS], "off_grid": [off[k] for k in PROFILE_DAYS]}


def measure(b1_by, b0_by, hist_by, bands):
    """The coherence departure on the own-month anomaly, the lobe-position contrast and the
    profile on the other-season anomaly, over the same histories."""
    coh = C.measure(b1_by, hist_by, bands)
    acc = P.new_acc(); n = 0
    for (y, b), hs in hist_by.items():
        if b not in bands:
            continue
        for h in hs:
            n += 1
            P.accumulate(b0_by[(y, b)], h, acc)
    lobe = P.reader(acc); prof = profile(b0_by, hist_by, bands)
    return {"histories": n, "departure": coh["departure"], "coherence_histories": coh["histories"], "contrast": lobe["contrast"],
            "carried": lobe["carried"], "pinned": lobe["pinned"], "samples": lobe["samples"], "first_day": prof["by_day"][0],
            "profile": prof["by_day"], "profile_days": prof["days"], "profile_off_grid": prof["off_grid"]}


# ------------------------------------------------------------------ constructions

def fields(lay, bg_name, place, tracks, by, matched_std, parts, month_of):
    """The own-month rows, the other-season rows and the histories of one placement: those that
    follow an injected trough, or every detected history for noise alone (lay None)."""
    b1_by, b0_by, hist_by = {}, {}, {}
    for bi, band in enumerate(L.BAND_LAT):
        if band not in BANDS:
            continue
        bg = (J.matched_background(matched_std[band], place) if bg_name == "matched"
              else [J.hundredths(L.season_samples(by[y], band)) for y in YEARS])
        Hb = {}
        for i, y in enumerate(YEARS):
            if lay is None:
                S = bg[i] / 100.0; centers = None
            else:
                rng = np.random.default_rng([P.LOBE_SEED, bi, place, i])
                V, centers = build(tracks.get(band, {}).get(y, []), rng, lay, parts)
                S = J.inject(bg[i], V)
            hs, rows = H.detected(S, month_of)
            hist_by[(y, band)] = hs if centers is None else [h for h in hs if any(HI.follows(h, c) for c in centers)]
            b1_by[(y, band)] = rows
            Hb[y] = P.daily_hundredths(S)
        b0 = P.b0_rows(Hb, month_of)
        for y in YEARS:
            b0_by[(y, band)] = b0[y]
    return b1_by, b0_by, hist_by


def construction(lay, bg_name, place, tracks, by, matched_std, parts, month_of):
    return measure(*fields(lay, bg_name, place, tracks, by, matched_std, parts, month_of), BANDS)


def run_constructions(tracks, by, matched_std, parts):
    month_of = A.month_index()
    out = {bg: {layout_name(lay): [construction(lay, bg, p, tracks, by, matched_std, parts, month_of) for p in range(J.PLACEMENTS)]
                for lay in layouts()} for bg in ("archive", "matched")}
    out["noise"] = [construction(None, "matched", p, tracks, by, matched_std, parts, month_of) for p in range(J.PLACEMENTS)]
    return out


def ranges(cons, bg, reader):
    return {k: (min(c[reader] for c in v), max(c[reader] for c in v), float(np.mean([c[reader] for c in v]))) for k, v in cons[bg].items()}


def check_reproduced(cons, lobe_cons):
    """The traveling and pinned constructions reproduce the lobe-position record's contrasts,
    carried and pinned readers, and history-days exactly."""
    for bg in ("archive", "matched"):
        for name, kind in REPRODUCED.items():
            mine = [{k: c[k] for k in ("contrast", "carried", "pinned", "samples")} for c in cons[bg][name]]
            theirs = [{k: c[k] for k in ("contrast", "carried", "pinned", "samples")} for c in lobe_cons[bg][kind]]
            if mine != theirs:
                raise SystemExit(f"REFUSED: {bg} {name} does not reproduce the lobe-position record's {kind}")
    mine = [{k: c[k] for k in ("contrast", "carried", "pinned", "samples")} for c in cons["noise"]]
    theirs = [{k: c[k] for k in ("contrast", "carried", "pinned", "samples")} for c in lobe_cons["noise"]]
    if mine != theirs:
        raise SystemExit("REFUSED: noise alone does not reproduce the lobe-position record")


def apply_rule(cons):
    """Section 4: on each background, the smallest first-day reading of travel lies above the
    largest of each narrow lobe standing at the first longitude (widths 1 and 2, strength 1,
    steady)."""
    fails = []
    for bg in ("archive", "matched"):
        r = ranges(cons, bg, "first_day")
        for k in NARROW_AT_START:
            if not r[k][1] < r["travel"][0]:
                fails.append(f"{bg}: {k} overlaps travel on the first day")
    return {"adequate": not fails, "failures": fails}


def check_constructions(path, tracks, by, matched_std, parts, rerun=None):
    with open(path, encoding="utf-8") as fh:
        rec = json.load(fh)
    if rec["generated_by"]["sha256"] != SCRIPT_SHA256_AT_LOAD:
        raise SystemExit("REFUSED: the constructions were run by a different version of this script")
    fresh = json.loads(json.dumps((rerun or run_constructions)(tracks, by, matched_std, parts)))
    if fresh != rec.get("results"):
        raise SystemExit("REFUSED: the constructions record differs from a fresh run of the same constructions")
    verdict = apply_rule(rec["results"])
    if not verdict["adequate"]:
        raise SystemExit(f"REFUSED: the constructions do not pass the rule of section 4 ({verdict['failures']})")
    if verdict != rec.get("verdict"):
        raise SystemExit("REFUSED: the stored summary disagrees with the rule applied to the stored cases")
    return rec


# ------------------------------------------------------------------ the archive

def archive_measure(by, hist, month_of):
    b1_by = {(y, b): H.b1_rows(L.season_samples(by[y], b), month_of)[1] for y in YEARS for b in L.BAND_LAT}
    b0_by = {}
    for b in L.BAND_LAT:
        rows = P.b0_rows({y: P.daily_hundredths(L.season_samples(by[y], b)) for y in YEARS}, month_of)
        for y in YEARS:
            b0_by[(y, b)] = rows[y]
    hist_by = {k: v for k, v in hist.items()}
    return measure(b1_by, b0_by, hist_by, BANDS)


def check_records(m, coherence_rec, lobe_rec):
    """The archive's departure and contrast equal the coherence and lobe-position records'."""
    if m["departure"] != coherence_rec["results"]["measure"]["primary"]["departure"]:
        raise SystemExit("REFUSED: the archive departure differs from the coherence record")
    lp = lobe_rec["results"]["measure"]["primary"]
    if (m["contrast"], m["carried"], m["pinned"], m["samples"]) != (lp["contrast"], lp["carried"], lp["pinned"], lp["samples"]):
        raise SystemExit("REFUSED: the archive contrast differs from the lobe-position record")


def classify(m, cons, bg="archive"):
    """For every layout, whether each reader's archive value lies inside its placement range on
    the background, the signed distance outside when not, and whether all three lie inside."""
    r = {k: ranges(cons, bg, k) for k in READERS}
    out = {}
    for name in cons[bg]:
        row = {}
        for k in READERS:
            lo, hi, _ = r[k][name]; v = m[k]
            row[k] = 0.0 if lo <= v <= hi else (v - hi if v > hi else v - lo)
        row["consistent"] = all(row[k] == 0.0 for k in READERS)
        out[name] = row
    return out


def summary(m, cons):
    cls = {bg: classify(m, cons, bg) for bg in ("archive", "matched")}
    standing = [n for n in cons["archive"] if n != "travel"]
    noise = {k: [min(c[k] for c in cons["noise"]), max(c[k] for c in cons["noise"])] for k in READERS}
    return {"classes": cls, "noise_range": noise,
            "consistent": {bg: [n for n in standing if cls[bg][n]["consistent"]] for bg in cls},
            "travel_consistent": {bg: cls[bg]["travel"]["consistent"] for bg in cls},
            "excluded_by": {bg: {k: sum(1 for n in standing if cls[bg][n][k] != 0.0) for k in READERS} for bg in cls},
            "standing_layouts": len(standing)}


def readings(m, s):
    ca = s["consistent"]["archive"]; n = s["standing_layouts"]; nr = s["noise_range"]["first_day"]
    out = [f"on the first day of the {m['histories']} westward histories in 15 to 24.5 N, the other-season anomaly 6 degrees east of the first "
           f"longitude is {m['first_day']:.2f} of the magnitude at the first longitude, and noise alone gives {nr[0]:.2f} to {nr[1]:.2f}",
           f"travel {'contains' if s['travel_consistent']['archive'] else 'does not contain'} the archive's three readings in its archive-background ranges",
           f"{len(ca)} of the {n} standing layouts contain the archive's three readings in their archive-background ranges"
           + (f" ({', '.join(ca)})" if ca else "")]
    return out


# ------------------------------------------------------------------ text and main

def fmt(v):
    return "none" if v is None else f"{v:.2f}"


def constructions_text(cons, verdict, reproduced):
    out = ["# standing lobes read together, the constructions (section 3)", "",
           "| background | layout | departure (min, max) | contrast (min, max) | first day (min, max) | histories per placement |",
           "| --- | --- | --- | --- | --- | --- |"]
    for bg in ("archive", "matched"):
        r = {k: ranges(cons, bg, k) for k in READERS}
        for name, v in cons[bg].items():
            out.append(f"| {bg} | {name} | " + " | ".join(f"{r[k][name][0]:.2f}, {r[k][name][1]:.2f}" for k in READERS)
                       + f" | {', '.join(str(c['histories']) for c in v)} |")
    nz = cons["noise"]
    out.append("| matched | noise alone | " + " | ".join(f"{min(c[k] for c in nz):.2f}, {max(c[k] for c in nz):.2f}" for k in READERS)
               + f" | {', '.join(str(c['histories']) for c in nz)} |")
    out += ["", f"the traveling and pinned constructions and noise alone reproduce the lobe-position record, {'yes' if reproduced else 'no'}",
            f"adequate under the rule of section 4, {'yes' if verdict['adequate'] else 'no'}"] + [f"- {x}" for x in verdict["failures"]]
    return "\n".join(out) + "\n"


def archive_text(m, s, cons):
    out = ["# standing lobes read together, the archive (section 7)", "",
           "| histories | departure | contrast | first day | profile by day 0 to 4 | days per day | off the grid per day |",
           "| --- | --- | --- | --- | --- | --- | --- |",
           f"| {m['histories']} | {m['departure']:.3f} | {m['contrast']:.3f} | {m['first_day']:.3f} | {', '.join(fmt(v) for v in m['profile'])} | "
           f"{', '.join(map(str, m['profile_days']))} | {', '.join(map(str, m['profile_off_grid']))} |", "",
           "| background | layouts excluded by departure | by contrast | by first day | consistent on all three | travel consistent |",
           "| --- | --- | --- | --- | --- | --- |"]
    for bg in ("archive", "matched"):
        e = s["excluded_by"][bg]
        out.append(f"| {bg} | {e['departure']} | {e['contrast']} | {e['first_day']} | {len(s['consistent'][bg])} of {s['standing_layouts']} | "
                   f"{'yes' if s['travel_consistent'][bg] else 'no'} |")
    out += ["", "| background | layout | departure outside by | contrast outside by | first day outside by |", "| --- | --- | --- | --- | --- |"]
    for bg in ("archive", "matched"):
        for name in ["travel"] + s["consistent"][bg]:
            c = s["classes"][bg][name]
            out.append(f"| {bg} | {name} | " + " | ".join(f"{c[k]:.3f}" for k in READERS) + " |")
    nr = s["noise_range"]
    out += ["", "noise alone on the matched background, " + ", ".join(f"{k} {nr[k][0]:.2f} to {nr[k][1]:.2f}" for k in READERS), ""]
    out += [f"- {x}" for x in readings(m, s)]
    return "\n".join(out) + "\n"


def load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("constructions", "archive"))
    ap.add_argument("--out-dir", required=True)
    for opt in ("--shape", "--inject", "--records", "--histories", "--xband", "--lagreg", "--lobe-constructions", "--coherence-archive",
                "--lobe-archive", "--constructions"):
        ap.add_argument(opt, default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    if not all((a.shape, a.inject, a.records, a.histories, a.xband, a.lobe_constructions)):
        raise SystemExit("REFUSED: --shape, --inject, --records, --histories, --xband and --lobe-constructions are required")
    hist, tracks, by, matched_std, gen = C.load_inputs(a)
    lobe_cons_rec = load_json(a.lobe_constructions)
    if lobe_cons_rec["generated_by"]["sha256"] != P.SCRIPT_SHA256_AT_LOAD:
        raise SystemExit("REFUSED: the lobe-position constructions were made by a different version of its script")
    gen.update({"script": "scripts/eastern_joint_standing.py", "sha256": SCRIPT_SHA256_AT_LOAD, "declared": "EASTERN_JOINT_STANDING_2026-10-22.md",
                "coherence_script_sha256": C.SCRIPT_SHA256_AT_LOAD, "lobe_script_sha256": P.SCRIPT_SHA256_AT_LOAD,
                "lobe_constructions": {"path": a.lobe_constructions, "sha256": sha256_of(a.lobe_constructions)}})
    parts = P.measured_parts()
    if a.mode == "constructions":
        pj, pm = (os.path.join(a.out_dir, f"eval60_joint_constructions{a.tag}.{e}") for e in ("json", "md"))
        C.refuse_existing([pj, pm])
        cons = json.loads(json.dumps(run_constructions(tracks, by, matched_std, parts)))
        check_reproduced(cons, lobe_cons_rec["results"])
        verdict = apply_rule(cons)
        with open(pj, "w", encoding="utf-8") as fh:
            json.dump({"generated_by": gen, "results": cons, "verdict": verdict}, fh)
        text = constructions_text(cons, verdict, True)
        with open(pm, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(text)
        return 0
    if not (a.lagreg and a.coherence_archive and a.lobe_archive and a.constructions):
        raise SystemExit("REFUSED: --lagreg, --coherence-archive, --lobe-archive and --constructions are required for the archive")
    J.check_zero_injection(by, load_json(a.lagreg))
    coh_rec, lobe_rec = load_json(a.coherence_archive), load_json(a.lobe_archive)
    if coh_rec["generated_by"]["sha256"] != C.SCRIPT_SHA256_AT_LOAD or lobe_rec["generated_by"]["sha256"] != P.SCRIPT_SHA256_AT_LOAD:
        raise SystemExit("REFUSED: the coherence or lobe-position archive record was made by a different version of its script")
    cons = check_constructions(a.constructions, tracks, by, matched_std, parts)["results"]
    check_reproduced(cons, lobe_cons_rec["results"])
    pj, pm = (os.path.join(a.out_dir, f"eval60_joint_{k}{a.tag}.{e}") for k, e in (("archive", "json"), ("tables", "md")))
    C.refuse_existing([pj, pm])
    m = archive_measure(by, hist, A.month_index())
    check_records(m, coh_rec, lobe_rec)
    s = summary(m, cons)
    res = {"measure": m, "summary": s, "readings": readings(m, s), "lobe_amplitude": parts["lobe"]}
    gen.update({"lagreg": {"path": a.lagreg, "sha256": sha256_of(a.lagreg)}, "constructions": {"path": a.constructions, "sha256": sha256_of(a.constructions)},
                "coherence_archive": {"path": a.coherence_archive, "sha256": sha256_of(a.coherence_archive)},
                "lobe_archive": {"path": a.lobe_archive, "sha256": sha256_of(a.lobe_archive)}})
    with open(pj, "w", encoding="utf-8") as fh:
        json.dump({"generated_by": gen, "results": res}, fh)
    text = archive_text(m, s, cons)
    with open(pm, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
