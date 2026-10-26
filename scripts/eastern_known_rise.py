#!/usr/bin/env python3
"""A known rise for the westward histories' fixed-cohort daily profile
(`EASTERN_KNOWN_RISE_2026-10-24.md`): the pre-start test's growing traveler, whose core moves at
constant depth on the histories' tracks while the rest of the measured shape grows from zero,
read by the matched-profile record's own measure after detection, alignment to the first
detected member, cohort selection and normalization. On a background with no noise ('clean')
it shows whether the growth survives the pipeline as a rise of the archive's size, with each
cohort history's lag behind its injected start. On the matched noise of the noise comparison
('noise'), with a frozen variant, it says whether the comparison can tell such a rise from that
noise. The control is a constructed stand-in, its results are stated for it alone, and nothing
here says whether the archive's histories travel.

    python3 scripts/eastern_known_rise.py clean --shape PATH --inject PATH --records DIR --histories PATH --xband DIR \\
        --matched-archive PATH --grow-days G --lobe-mult M --out-dir DIR
    python3 scripts/eastern_known_rise.py noise --shape PATH --inject PATH --records DIR --histories PATH --xband DIR \\
        --matched-archive PATH --prestart-constructions PATH --noise-archive PATH --clean PATH --out-dir DIR
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
import eastern_matched_noise as N  # noqa: E402

PS, M = N.PS, N.M
G = PS.G
P, C, H, J, L, A, HI = G.P, G.C, G.H, G.J, G.L, G.A, G.HI
YEARS, BANDS, XL = G.YEARS, G.BANDS, G.XL
BLOCKS, PLACEMENTS, RISES = N.BLOCKS, N.PLACEMENTS, N.RISES
STORED_CASE = "travel_lobe_zero"


def sha256_of(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


SCRIPT_SHA256_AT_LOAD = sha256_of(os.path.abspath(__file__))


def build(tracks, rng, grow_days, lobe_mult, parts):
    """The pre-start test's 'lobe_zero' traveler with its growth time and its grown lobe as
    parameters: on each track, drawn as the pre-start test draws it, the measured shape's core at
    constant depth plus lobe_mult times the rest of the shape scaled by min(age / grow_days, 1).
    Returns the field, each trough's daily-mean position and each trough's first day."""
    t = J.times()[:, 0]; V = np.zeros((J.N_SAMPLES, len(XL))); centers, starts = [], []
    for tr in tracks:
        n = tr["days"]; t0 = int(rng.integers(0, A.SEASON_DAYS - n + 1)); d = tr["injected"]["measured"]
        k0, k1 = t0 * L.OBS_PER_DAY, (t0 + n) * L.OBS_PER_DAY
        pos = np.full(J.N_SAMPLES, np.nan); pos[k0:k1] = tr["first_lon"] + tr["speed"] * (t[k0:k1] - t0)
        for k in range(k0, k1):
            f = min((t[k] - t0) / grow_days, 1.0); full = H.shape("measured", XL, pos[k], d)
            core = P.core_shape(XL, pos[k], d, parts["core"])
            V[k] += core + lobe_mult * f * (full - core)
        centers.append(pos.reshape(A.SEASON_DAYS, L.OBS_PER_DAY).mean(axis=1)); starts.append(t0)
    return V, centers, starts


def zero_background(std, place):
    """A background with no noise, in integer hundredths per season."""
    return [np.zeros((J.N_SAMPLES, len(J.LONS))) for _ in YEARS]


def fields(bg_fn, place, tracks, grow_days, lobe_mult, matched_std, parts, month_of):
    """The pre-start test's fields for one placement with this traveler, on the background bg_fn
    gives: the other-season rows, the histories that follow an injected trough, and each kept
    history's lag in days behind the start of the first trough it follows."""
    b0_by, hist_by, lags = {}, {}, {}
    for bi, band in enumerate(L.BAND_LAT):
        if band not in BANDS:
            continue
        bg = bg_fn(matched_std[band], place)
        Hb = {}
        for i, y in enumerate(YEARS):
            rng = np.random.default_rng([P.LOBE_SEED, bi, place, i])
            V, centers, starts = build(tracks.get(band, {}).get(y, []), rng, grow_days, lobe_mult, parts)
            S = J.inject(bg[i], V)
            hs, _ = H.detected(S, month_of)
            kept = []
            for h in hs:
                followed = [s for c, s in zip(centers, starts) if HI.follows(h, c)]
                if followed:
                    kept.append(h); lags[id(h)] = h["days"][0] - followed[0]
            hist_by[(y, band)] = kept
            Hb[y] = P.daily_hundredths(S)
        b0 = P.b0_rows(Hb, month_of)
        for y in YEARS:
            b0_by[(y, band)] = b0[y]
    return b0_by, hist_by, lags


def placement(bg_fn, place, tracks, grow_days, lobe_mult, matched_std, parts, month_of):
    """One placement read by the matched-profile record's measure, with the lags of the cohort's
    histories."""
    b0, hist, lags = fields(bg_fn, place, tracks, grow_days, lobe_mult, matched_std, parts, month_of)
    r = M.measure(b0, hist)
    cohort, _ = M.matched(hist, BANDS)
    cl = [lags[id(h)] for hs in cohort.values() for h in hs]
    return {"placement": place, "changing": r["changing"], "matched": r["matched"], "cohort_histories": len(r["cohort"]),
            "additive_max_gap_ms": r["additive_max_gap_ms"], "rise": N.rises(r["matched"]["by_day"]),
            "cohort_lag_days": {str(k): cl.count(k) for k in sorted(set(cl))}}


def run(bg_fn, tracks, grow_days, lobe_mult, matched_std, parts):
    month_of = A.month_index()
    return [placement(bg_fn, p, tracks, grow_days, lobe_mult, matched_std, parts, month_of) for p in PLACEMENTS]


def block_summary(entries):
    """For each block, the range of each rise and of the cohort sizes and the pooled cohort
    curve, by the noise comparison's own functions."""
    return N.summary({"noise": entries, "travel": entries})["noise"]


def adequate(archive_rise, s):
    """Section 2.3: the variant produces a known rise of the archive's size on the clean
    background when its pooled cohort rise to each day reaches the archive's in both blocks."""
    reach = {f"day {k}, block {b}": N.pooled_rise(s[b]["pooled"]["by_day"], k) >= archive_rise[k] for k in map(str, RISES) for b in BLOCKS}
    return {"adequate": all(reach.values()), "reaches": reach}


def check_reproduced(entries, prestart_rec):
    """With the pre-start test's own growth (2 days, lobe once), every noise placement's curve over
    all its histories equals the pre-start record's for its growing traveler."""
    theirs = prestart_rec["results"]["matched"][STORED_CASE]
    if len(theirs) != len(PLACEMENTS) or len(entries) != len(PLACEMENTS):
        raise SystemExit("REFUSED: the growing traveler does not hold one entry per placement")
    for i, p in enumerate(PLACEMENTS):
        c, t = entries[i]["changing"], theirs[i]
        if entries[i]["placement"] != p or (c["by_day"], c["days"], c["off_grid"], c["histories"]) != (
                t["profile"], t["profile_days"], t["profile_off_grid"], t["histories"]):
            raise SystemExit(f"REFUSED: placement {p} does not reproduce the pre-start record's growing traveler")


def noise_reading(archive_rise, noise_rec, entries):
    """The noise comparison's own verdict and readings, with this traveler in place of the
    constant one and noise alone as stored in the noise record."""
    noise = noise_rec["results"]["placements"]["noise"]
    res = {"noise": noise, "travel": entries}
    s = N.summary(res)
    if s["noise"] != noise_rec["results"]["summary"]["noise"]:
        raise SystemExit("REFUSED: the stored noise placements do not give the noise record's summary")
    v = N.verdict(archive_rise, s, res)
    return {"summary": s["travel"], "verdict": v, "readings": readings(v, N.readings(v)[0])}


def readings(v, archive_line):
    """Section 3.3's readings for the growing traveler as amended in section 10, chosen by the
    verdict alone, beside the noise comparison's own reading of the archive. Not separating is
    failure of complete separation under the strict rule, never a statement that the comparison
    distinguishes nothing."""
    if v["traveler"] == "separates":
        trav = (f"the growing traveler separates completely from this noise, its smallest rise above noise's largest in all {v['archive_checks']} "
                f"block-and-day checks, while the archive's rises pass {v['archive_checks_above']} of those checks")
    elif v["traveler_rise"] == "reaches the archive's":
        trav = ("the growing traveler's pooled rise on this noise reaches the archive's, but it fails complete separation from this noise, "
                "its smallest rise not lying above noise's largest in every block-and-day check")
    else:
        trav = ("on this noise the growing traveler's pooled rise does not reach the archive's, although on the clean background it does, so "
                "its not separating does not establish insensitivity to a rise of the archive's size, and the noise attenuates the "
                "observable this construction carries")
    out = [archive_line, trav]
    if v["archive"] != "above" and v["traveler"] != "separates" and v["traveler_rise"] == "reaches the archive's":
        out.append("neither the archive's rise nor a constructed rise of its size separates completely from this noise, and this exploratory "
                   "comparison does not calibrate the archive's physical interpretation")
    out.append("the readings hold for this noise model, this constructed traveler and these 20 placements, count no histories one by one, "
               "and leave travel unresolved")
    return out


def fmt(v, f=".3f"):
    return "none" if v is None else format(v, f)


def table(entries, s, archive, title):
    out = [f"# {title}", "", "| placement | block | histories | cohort | day 0 | day 1 | day 2 | day 3 | day 4 | rise to day 3 | rise to day 4 | cohort lag, days (count) |",
           "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    out.append(f"| archive | | {archive['changing_histories']} | {archive['cohort_histories']} | " + " | ".join(fmt(x) for x in archive["by_day"])
               + f" | {fmt(archive['rise']['3'])} | {fmt(archive['rise']['4'])} | |")
    for e in entries:
        b = next(k for k, ps in BLOCKS.items() if e["placement"] in ps)
        lag = ", ".join(f"{k} ({n})" for k, n in e["cohort_lag_days"].items())
        out.append(f"| {e['placement']} | {b} | {e['changing']['histories']} | {e['cohort_histories']} | " + " | ".join(fmt(x) for x in e["matched"]["by_day"])
                   + f" | {fmt(e['rise']['3'])} | {fmt(e['rise']['4'])} | {lag} |")
    out += ["", "| block | cohort sizes | rise to day 3 (min, max) | rise to day 4 (min, max) | pooled cohort curve, days 0 to 4 | pooled rises to days 3 and 4 |",
            "| --- | --- | --- | --- | --- | --- |"]
    for b in BLOCKS:
        x = s[b]
        out.append(f"| {b} | {x['cohort_histories'][0]} to {x['cohort_histories'][1]} | {fmt(x['rise']['3'][0])}, {fmt(x['rise']['3'][1])} | "
                   f"{fmt(x['rise']['4'][0])}, {fmt(x['rise']['4'][1])} | " + ", ".join(fmt(y) for y in x["pooled"]["by_day"])
                   + f" over {x['pooled']['histories']} | {fmt(N.pooled_rise(x['pooled']['by_day'], '3'))}, {fmt(N.pooled_rise(x['pooled']['by_day'], '4'))} |")
    return out


def load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write(pj, pm, gen, results, lines):
    with open(pj, "w", encoding="utf-8") as fh:
        json.dump({"generated_by": gen, "results": results}, fh)
    t = "\n".join(lines) + "\n"
    with open(pm, "w", encoding="utf-8") as fh:
        fh.write(t)
    print(t)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("clean", "noise"))
    for opt in ("--shape", "--inject", "--records", "--histories", "--xband", "--matched-archive", "--out-dir"):
        ap.add_argument(opt, required=True)
    for opt in ("--prestart-constructions", "--noise-archive", "--clean"):
        ap.add_argument(opt, default=None)
    ap.add_argument("--grow-days", type=float, default=PS.GROW_DAYS)
    ap.add_argument("--lobe-mult", type=float, default=1.0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    if a.grow_days <= 0 or a.lobe_mult <= 0:
        raise SystemExit("REFUSED: the growth time and the lobe multiplier must be positive")
    matched_rec = load_json(a.matched_archive)
    if matched_rec["generated_by"]["sha256"] != M.SCRIPT_SHA256_AT_LOAD:
        raise SystemExit("REFUSED: the matched-profile record was made by a different version of its script")
    hist, tracks, by, matched_std, gen = C.load_inputs(a)
    M.check_same_inputs(gen, matched_rec)
    archive = N.archive_values(matched_rec)
    parts = P.measured_parts()
    gen.update({"script": "scripts/eastern_known_rise.py", "sha256": SCRIPT_SHA256_AT_LOAD, "declared": "EASTERN_KNOWN_RISE_2026-10-24.md",
                "matched_archive": {"path": a.matched_archive, "sha256": sha256_of(a.matched_archive)}})
    if a.mode == "clean":
        name = f"g{a.grow_days:g}_m{a.lobe_mult:g}"
        pj, pm = (os.path.join(a.out_dir, f"eval60_known_rise_clean_{name}{a.tag}.{e}") for e in ("json", "md"))
        C.refuse_existing([pj, pm])
        entries = run(zero_background, tracks, a.grow_days, a.lobe_mult, matched_std, parts)
        s = block_summary(entries); ad = adequate(archive["rise"], s)
        gen.update({"grow_days": a.grow_days, "lobe_mult": a.lobe_mult})
        lines = table(entries, s, archive, f"known rise on a clean background, growth over {a.grow_days:g} days, lobe times {a.lobe_mult:g}")
        lines += ["", f"Adequate: {ad['adequate']} ({', '.join(f'{k} {v}' for k, v in ad['reaches'].items())})."]
        write(pj, pm, gen, {"placements": entries, "summary": s, "adequacy": ad, "archive": archive}, lines)
        return 0
    if not (a.prestart_constructions and a.noise_archive and a.clean):
        raise SystemExit("REFUSED: --prestart-constructions, --noise-archive and --clean are required for the noise run")
    prestart_rec, noise_rec, clean_rec = load_json(a.prestart_constructions), load_json(a.noise_archive), load_json(a.clean)
    if prestart_rec["generated_by"]["sha256"] != PS.SCRIPT_SHA256_AT_LOAD:
        raise SystemExit("REFUSED: the pre-start constructions were made by a different version of their script")
    if noise_rec["generated_by"]["sha256"] != N.SCRIPT_SHA256_AT_LOAD:
        raise SystemExit("REFUSED: the noise record was made by a different version of its script")
    if clean_rec["generated_by"]["sha256"] != SCRIPT_SHA256_AT_LOAD:
        raise SystemExit("REFUSED: the clean record was made by a different version of this script")
    for rec in (prestart_rec, noise_rec, clean_rec):
        M.check_same_inputs(gen, rec)
    if not clean_rec["results"]["adequacy"]["adequate"]:
        raise SystemExit("REFUSED: the frozen variant does not produce a known rise of the archive's size on the clean background")
    gd, lm = clean_rec["generated_by"]["grow_days"], clean_rec["generated_by"]["lobe_mult"]
    pj, pm = (os.path.join(a.out_dir, f"eval60_known_rise_noise{a.tag}.{e}") for e in ("json", "md"))
    C.refuse_existing([pj, pm])
    stored = run(J.matched_background, tracks, PS.GROW_DAYS, 1.0, matched_std, parts)
    check_reproduced(stored, prestart_rec)
    entries = stored if (gd, lm) == (PS.GROW_DAYS, 1.0) else run(J.matched_background, tracks, gd, lm, matched_std, parts)
    nr = noise_reading(archive["rise"], noise_rec, entries)
    gen.update({"grow_days": gd, "lobe_mult": lm, "prestart_constructions": {"path": a.prestart_constructions, "sha256": sha256_of(a.prestart_constructions)},
                "noise_archive": {"path": a.noise_archive, "sha256": sha256_of(a.noise_archive)}, "clean": {"path": a.clean, "sha256": sha256_of(a.clean)}})
    lines = table(entries, nr["summary"], archive, f"known rise on matched noise, growth over {gd:g} days, lobe times {lm:g}")
    lines += ["", f"Archive: {nr['verdict']['archive']}. Traveler: {nr['verdict']['traveler']}, pooled rise {nr['verdict']['traveler_rise']}.", ""]
    lines += [f"- {x}" for x in nr["readings"]]
    write(pj, pm, gen, {"placements": entries, "stored_case_placements": stored if entries is not stored else None, **nr, "archive": archive}, lines)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
