#!/usr/bin/env python3
"""Diagnostics of the known rise (`EASTERN_KNOWN_RISE_2026-10-24.md`, section 9): at the multiple
1.3, the cohort rise with the histories held fixed and the field varied, the clean and noisy
histories read on the clean and noisy other-season fields from the same injected troughs and
starts, and the count of cohort histories that follow more than one injected trough; at the
multiple 2.5, the cohort over every detected westward history on the same noisy fields, and the
outcomes of the frozen control against noise alone in twenty further placements, with further
control placements against the original noise. The diagnostics describe this construction on
this noise model and estimate no calibrated probability.

    python3 scripts/eastern_known_rise_diagnostics.py --shape PATH --inject PATH --records DIR --histories PATH --xband DIR \\
        --matched-archive PATH --noise-archive PATH --map PATH --out-dir DIR
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
import eastern_known_rise as K  # noqa: E402

N, M, PS = K.N, K.M, K.PS
P, H, J, L, A, HI = K.P, K.H, K.J, K.L, K.A, K.HI
CONTROL, MATCHED = 1.3, 2.5
FRESH = {"A": tuple(range(20, 30)), "B": tuple(range(30, 40))}
DAYS = tuple(map(str, K.RISES))


def sha256_of(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


SCRIPT_SHA256_AT_LOAD = sha256_of(os.path.abspath(__file__))


def fields(bg_fn, place, tracks, mult, matched_std, parts, month_of):
    """The known-rise script's fields for one placement, with every detected westward history kept
    beside the following ones and, for each following history, the number of troughs it follows."""
    b0_by, follow_by, all_by, n_followed = {}, {}, {}, {}
    for bi, band in enumerate(L.BAND_LAT):
        if band not in K.BANDS:
            continue
        bg = bg_fn(matched_std[band], place)
        Hb = {}
        for i, y in enumerate(K.YEARS):
            rng = np.random.default_rng([P.LOBE_SEED, bi, place, i])
            V, centers, starts = K.build(tracks.get(band, {}).get(y, []), rng, PS.GROW_DAYS, mult, parts)
            S = J.inject(bg[i], V)
            hs, _ = H.detected(S, month_of)
            follow = []
            for h in hs:
                n = sum(1 for c in centers if HI.follows(h, c))
                if n:
                    follow.append(h); n_followed[id(h)] = n
            follow_by[(y, band)], all_by[(y, band)] = follow, hs
            Hb[y] = P.daily_hundredths(S)
        b0 = P.b0_rows(Hb, month_of)
        for y in K.YEARS:
            b0_by[(y, band)] = b0[y]
    return b0_by, follow_by, all_by, n_followed


def cohort_entry(place, b0, hist):
    """One placement's cohort reading, in the shape the noise comparison's summary reads."""
    r = M.measure(b0, hist)
    return {"placement": place, "cohort_histories": len(r["cohort"]), "rise": N.rises(r["matched"]["by_day"]), "matched": r["matched"]}


def pooled_rises(entries):
    s = K.block_summary(entries)
    return {b: {k: N.pooled_rise(s[b]["pooled"]["by_day"], k) for k in DAYS} for b in K.BLOCKS}


def outcomes(entries, maxima):
    per = [N.exceedance(e["rise"], maxima)[0] for e in entries]
    return {o: per.count(o) for o in ("above", "mixed", "not separated")}


def check_follow(entries_fields, stored):
    """The following histories' readings and the known-rise record's placements both hold every
    declared placement in order, and their readings are equal exactly."""
    if [e["placement"] for e in entries_fields] != list(K.PLACEMENTS) or [s["placement"] for s in stored] != list(K.PLACEMENTS):
        raise SystemExit("REFUSED: the readings or the record do not hold every declared placement in order")
    for e, s in zip(entries_fields, stored):
        if (e["rise"], e["cohort_histories"], e["matched"]["by_day"]) != (s["rise"], s["cohort_histories"], s["matched"]["by_day"]):
            raise SystemExit(f"REFUSED: placement {e['placement']} does not reproduce the known-rise record")


def fixed_history(tracks, matched_std, parts, month_of, stored_noise, stored_clean):
    """At the control's multiple, each placement's clean and noisy following histories read on the
    clean and noisy fields, and the cohort histories that follow more than one trough."""
    cells = {k: [] for k in ("noisy_on_clean", "noisy_on_noisy", "clean_on_noisy", "clean_on_clean")}
    multi = 0
    for p in K.PLACEMENTS:
        b0c, fc, _, _ = fields(K.zero_background, p, tracks, CONTROL, matched_std, parts, month_of)
        b0n, fn, _, nf = fields(J.matched_background, p, tracks, CONTROL, matched_std, parts, month_of)
        for name, (b0, hist) in (("noisy_on_clean", (b0c, fn)), ("noisy_on_noisy", (b0n, fn)), ("clean_on_noisy", (b0n, fc)), ("clean_on_clean", (b0c, fc))):
            cells[name].append(cohort_entry(p, b0, hist))
        coh, _ = M.matched(fn, K.BANDS)
        multi += sum(1 for hs in coh.values() for h in hs if nf[id(h)] > 1)
    check_follow(cells["noisy_on_noisy"], stored_noise); check_follow(cells["clean_on_clean"], stored_clean)
    return {"pooled_rise": {k: pooled_rises(v) for k, v in cells.items()}, "cohort_histories_following_several_troughs": multi}


def population(tracks, matched_std, parts, month_of, stored, maxima):
    """At the noise-matched multiple, the cohort over every detected westward history on the same
    noisy fields beside the following histories, with their outcomes under the archive's rule."""
    follow, every = [], []
    for p in K.PLACEMENTS:
        b0, fh, ah, _ = fields(J.matched_background, p, tracks, MATCHED, matched_std, parts, month_of)
        follow.append(cohort_entry(p, b0, fh)); every.append(cohort_entry(p, b0, ah))
    check_follow(follow, stored)
    return {name: {"pooled_rise": pooled_rises(es), "outcomes": outcomes(es, maxima)} for name, es in (("following", follow), ("every_detected", every))}


def fresh(tracks, matched_std, parts, month_of, stored, maxima):
    """Noise alone in twenty further placements, two blocks of ten, as a second reference for the
    frozen placements at the noise-matched multiple, and twenty further placements of that
    construction against the original reference."""
    noise = {p: N.placement(None, p, tracks, {}, matched_std, parts, month_of) for b in FRESH for p in FRESH[b]}
    fresh_max = {(k, b): max(noise[p]["rise"][k] for p in FRESH[b]) for k in DAYS for b in FRESH}
    ctl = []
    for b in FRESH:
        for p in FRESH[b]:
            b0, fh, _, _ = fields(J.matched_background, p, tracks, MATCHED, matched_std, parts, month_of)
            ctl.append(cohort_entry(p, b0, fh))
    return {"fresh_noise_maxima": {f"day {k}, block {b}": v for (k, b), v in fresh_max.items()},
            "original_control_on_fresh_noise": outcomes(stored, fresh_max), "fresh_control_on_original_noise": outcomes(ctl, maxima)}


def individually_reaching(stored, archive_rise, maxima):
    """At the noise-matched multiple, the placements whose own rises meet both of the archive's,
    with their outcomes."""
    reach = [e for e in stored if all(e["rise"][k] >= archive_rise[k] for k in DAYS)]
    return {"placements": len(reach), "outcomes": outcomes(reach, maxima)}


def fmt(v, f=".3f"):
    return format(v, f)


def text(res):
    out = ["# known-rise diagnostics", "", "| reading at multiple 1.3 | rise to day 3, A and B | rise to day 4, A and B |", "| --- | --- | --- |"]
    for k, v in res["fixed_history"]["pooled_rise"].items():
        out.append(f"| {k.replace('_', ' ')} | {fmt(v['A']['3'])}, {fmt(v['B']['3'])} | {fmt(v['A']['4'])}, {fmt(v['B']['4'])} |")
    out += ["", f"Cohort histories following more than one trough at 1.3 on noise: {res['fixed_history']['cohort_histories_following_several_troughs']}.", "",
            "| population at multiple 2.5 | rise to day 3, A and B | rise to day 4, A and B | ABOVE | MIXED | NOT SEPARATED |", "| --- | --- | --- | --- | --- | --- |"]
    for k, v in res["population"].items():
        p, o = v["pooled_rise"], v["outcomes"]
        out.append(f"| {k.replace('_', ' ')} | {fmt(p['A']['3'])}, {fmt(p['B']['3'])} | {fmt(p['A']['4'])}, {fmt(p['B']['4'])} | {o['above']} | {o['mixed']} | {o['not separated']} |")
    f, ir = res["fresh"], res["individually_reaching"]
    out += ["", "| comparison at multiple 2.5 | ABOVE | MIXED | NOT SEPARATED |", "| --- | --- | --- | --- |"]
    for k, o in (("placements whose own rises meet the archive's, original noise", ir["outcomes"]), ("original placements, further noise", f["original_control_on_fresh_noise"]),
                 ("further placements, original noise", f["fresh_control_on_original_noise"])):
        out.append(f"| {k} | {o['above']} | {o['mixed']} | {o['not separated']} |")
    return "\n".join(out) + "\n"


def load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for opt in ("--shape", "--inject", "--records", "--histories", "--xband", "--matched-archive", "--noise-archive", "--map", "--out-dir"):
        ap.add_argument(opt, required=True)
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    pj, pm = (os.path.join(a.out_dir, f"eval60_known_rise_diagnostics{a.tag}.{e}") for e in ("json", "md"))
    K.C.refuse_existing([pj, pm])
    matched_rec, noise_rec, map_rec = load_json(a.matched_archive), load_json(a.noise_archive), load_json(a.map)
    if noise_rec["generated_by"]["sha256"] != N.SCRIPT_SHA256_AT_LOAD or matched_rec["generated_by"]["sha256"] != M.SCRIPT_SHA256_AT_LOAD:
        raise SystemExit("REFUSED: the noise or matched-profile record was made by a different version of its script")
    hist, tracks, by, matched_std, gen = K.C.load_inputs(a)
    for rec in (matched_rec, noise_rec):
        M.check_same_inputs(gen, rec)
    runs = {}
    for path in map_rec["generated_by"]["runs"]:
        rec = load_json(path)
        if sha256_of(path) != map_rec["generated_by"]["runs"][path] or rec["generated_by"]["sha256"] != K.SCRIPT_SHA256_AT_LOAD:
            raise SystemExit(f"REFUSED: {path} is not the run the map read, or was made by another version of the known-rise script")
        runs[rec["generated_by"]["lobe_mult"]] = rec
    if CONTROL not in runs or MATCHED not in runs or map_rec["results"]["noise_matched"] != MATCHED:
        raise SystemExit("REFUSED: the map does not hold the control and the noise-matched multiple named here")
    clean_path = runs[CONTROL]["generated_by"]["clean"]["path"]
    if sha256_of(clean_path) != runs[CONTROL]["generated_by"]["clean"]["sha256"]:
        raise SystemExit("REFUSED: the control's clean record has changed since its noise run")
    archive_rise = N.archive_values(matched_rec)["rise"]; maxima = N.noise_maxima(noise_rec["results"]["summary"])
    parts, month_of = P.measured_parts(), A.month_index()
    res = {"fixed_history": fixed_history(tracks, matched_std, parts, month_of, runs[CONTROL]["results"]["placements"], load_json(clean_path)["results"]["placements"]),
           "population": population(tracks, matched_std, parts, month_of, runs[MATCHED]["results"]["placements"], maxima),
           "fresh": fresh(tracks, matched_std, parts, month_of, runs[MATCHED]["results"]["placements"], maxima),
           "individually_reaching": individually_reaching(runs[MATCHED]["results"]["placements"], archive_rise, maxima)}
    gen.update({"script": "scripts/eastern_known_rise_diagnostics.py", "sha256": SCRIPT_SHA256_AT_LOAD, "declared": "EASTERN_KNOWN_RISE_2026-10-24.md",
                "map": {"path": a.map, "sha256": sha256_of(a.map)}, "noise_archive": {"path": a.noise_archive, "sha256": sha256_of(a.noise_archive)},
                "matched_archive": {"path": a.matched_archive, "sha256": sha256_of(a.matched_archive)}})
    with open(pj, "w", encoding="utf-8") as fh:
        json.dump({"generated_by": gen, "results": res}, fh)
    t = text(res)
    with open(pm, "w", encoding="utf-8") as fh:
        fh.write(t)
    print(t)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
