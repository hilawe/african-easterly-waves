#!/usr/bin/env python3
"""The westward histories' fixed-cohort daily profile against matched noise
(`EASTERN_MATCHED_NOISE_2026-10-23.md`): the matched-profile record's cohort curve read on the
pre-start test's matched noise alone and on its constant traveler of the measured shape over
the same noise, in its two independent blocks of ten placements, with each placement's histories
detected as on the archive and its cohort formed by the same rule. Every placement's curve over
all its histories must equal the pre-start record. The archive's rise from day 0 to days 3 and 4
is compared with each block's range under a rule fixed before the run, with the number of noise
placements each rise exceeds. The traveler's result bears on the comparison's sensitivity only
when its own pooled rise reaches the archive's. The measure is stated for
this noise model and this traveler, identifies no feature, and does not say whether the
histories travel.

    python3 scripts/eastern_matched_noise.py --shape PATH --inject PATH --records DIR --histories PATH --xband DIR \\
        --prestart-constructions PATH --matched-archive PATH --out-dir DIR
"""

import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
import eastern_matched_profile as M  # noqa: E402
import eastern_prestart as PS  # noqa: E402

G = PS.G
P, C, A = G.P, G.C, G.A
BLOCKS = PS.BLOCKS
PLACEMENTS = PS.PLACEMENTS
DAYS = G.PROFILE_DAYS
RISES = (3, 4)
CONTROL = {"kind": "travel", "growth": "constant"}
SAMPLES = (("noise", None), ("travel", CONTROL))


def sha256_of(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


SCRIPT_SHA256_AT_LOAD = sha256_of(os.path.abspath(__file__))


def rises(by_day):
    """The cohort ratio on each day of RISES less its ratio on day 0."""
    return {str(k): by_day[k] - by_day[0] for k in RISES}


def placement(case, place, tracks, by, matched_std, parts, month_of):
    """One placement on matched noise, with nothing injected (case None) or the traveler, read by
    the matched-profile record's own measure over the histories the pre-start test keeps."""
    _, b0, hist = PS.fields(case, "matched", place, tracks, by, matched_std, parts, month_of)
    r = M.measure(b0, hist)
    return {"placement": place, "changing": r["changing"], "matched": r["matched"], "cohort_histories": len(r["cohort"]),
            "additive_max_gap_ms": r["additive_max_gap_ms"], "rise": rises(r["matched"]["by_day"])}


def run(tracks, by, matched_std, parts):
    month_of = A.month_index()
    return {name: [placement(case, p, tracks, by, matched_std, parts, month_of) for p in PLACEMENTS] for name, case in SAMPLES}


def stored(prestart_rec, name):
    return prestart_rec["results"]["noise"] if name == "noise" else prestart_rec["results"]["matched"]["travel"]


def check_reproduced(res, prestart_rec):
    """Every placement's curve over all its histories, with its counts, equals the pre-start
    record's, for noise alone and for the traveler."""
    for name, _ in SAMPLES:
        theirs = stored(prestart_rec, name)
        if len(theirs) != len(PLACEMENTS) or len(res[name]) != len(PLACEMENTS):
            raise SystemExit(f"REFUSED: {name} does not hold one entry per placement")
        for i, p in enumerate(PLACEMENTS):
            c, t = res[name][i]["changing"], theirs[i]
            if res[name][i]["placement"] != p or (c["by_day"], c["days"], c["off_grid"], c["histories"]) != (
                    t["profile"], t["profile_days"], t["profile_off_grid"], t["histories"]):
                raise SystemExit(f"REFUSED: {name} placement {p} does not reproduce the pre-start record")


def block_entries(res, name, block):
    return [res[name][PLACEMENTS.index(p)] for p in BLOCKS[block]]


def pooled(entries):
    """The cohort curve over every cohort of a block, from the kept sums."""
    num = [sum(e["matched"]["numerator"][i] for e in entries) for i in range(len(DAYS))]
    den = [sum(e["matched"]["denominator"][i] for e in entries) for i in range(len(DAYS))]
    return {"by_day": [n / d for n, d in zip(num, den)], "histories": sum(e["cohort_histories"] for e in entries)}


def summary(res):
    """For each sample and block, the range of each rise and of the cohort sizes, and the pooled
    cohort curve."""
    out = {}
    for name, _ in SAMPLES:
        out[name] = {}
        for block in BLOCKS:
            es = block_entries(res, name, block)
            out[name][block] = {"rise": {k: [min(e["rise"][k] for e in es), max(e["rise"][k] for e in es)] for k in map(str, RISES)},
                                "cohort_histories": [min(e["cohort_histories"] for e in es), max(e["cohort_histories"] for e in es)],
                                "pooled": pooled(es)}
    return out


def pooled_rise(by_day, k):
    return by_day[int(k)] - by_day[0]


def noise_maxima(s):
    """Noise alone's largest rise to each day in each block of a summary."""
    return {(k, b): s["noise"][b]["rise"][k][1] for k in map(str, RISES) for b in BLOCKS}


def exceedance(rise, maxima):
    """Section 2.4's rule for one pair of rises: each rise to each day against noise's largest in
    each block, 'above' when every check passes strictly, 'not separated' when none does, and
    'mixed' otherwise, with the checks."""
    above = {(k, b): rise[k] > maxima[(k, b)] for k in map(str, RISES) for b in BLOCKS}
    return ("above" if all(above.values()) else "not separated" if not any(above.values()) else "mixed"), above


def verdict(archive_rise, s, res):
    """Section 2.4. The archive's rises against noise alone: 'above' when each lies above the
    noise's largest in both blocks, 'not separated' when none does in either block, and
    'mixed' otherwise. The traveler 'separates' when its smallest rise lies above the noise's
    largest for each rise in both blocks. Added by section 7 as description: how many of the
    20 noise placements each archive rise exceeds, and whether the traveler's pooled rises
    reach the archive's in both blocks, without which its not separating says nothing about
    the comparison's sensitivity to a rise of the archive's size."""
    archive, above = exceedance(archive_rise, noise_maxima(s))
    sep = {(k, b): s["travel"][b]["rise"][k][0] > s["noise"][b]["rise"][k][1] for k in map(str, RISES) for b in BLOCKS}
    reach = {(k, b): pooled_rise(s["travel"][b]["pooled"]["by_day"], k) >= archive_rise[k] for k in map(str, RISES) for b in BLOCKS}
    return {"archive": archive, "archive_above": {f"day {k}, block {b}": v for (k, b), v in above.items()},
            "archive_checks_above": sum(above.values()), "archive_checks": len(above),
            "archive_exceeds": {k: sum(e["rise"][k] < archive_rise[k] for e in res["noise"]) for k in map(str, RISES)},
            "noise_placements": len(res["noise"]),
            "traveler": "separates" if all(sep.values()) else "does not separate",
            "traveler_above": {f"day {k}, block {b}": v for (k, b), v in sep.items()},
            "traveler_rise": "reaches the archive's" if all(reach.values()) else "below the archive's"}


def readings(v):
    """Section 2.4's readings as amended in sections 7 and 8, chosen by the verdict alone. Not
    separating is failure of complete separation under the strict rule, never a statement that
    the comparison distinguishes nothing, and a traveler whose pooled rise falls short of the
    archive's says nothing about sensitivity to a rise of the archive's size."""
    n, ex = v["noise_placements"], v["archive_exceeds"]
    counts = f"its rise to day 3 exceeds {ex['3']} of the {n} noise placements and its rise to day 4 exceeds {ex['4']}"
    if v["archive"] == "above":
        arch = f"the declared rule gives ABOVE, since the archive's fixed-cohort rises lie above every placement of this noise model in both blocks, and {counts}"
    elif v["archive"] == "not separated":
        arch = f"the declared rule gives NOT SEPARATED, since neither of the archive's fixed-cohort rises lies above this noise model's largest in either block, and {counts}"
    else:
        arch = (f"the declared rule gives MIXED, since the archive passes {v['archive_checks_above']} of its {v['archive_checks']} block-and-day "
                f"exceedance checks, and {counts}")
    if v["traveler"] == "separates":
        trav = "the constant traveler of the measured shape separates completely from this noise, its smallest rises above noise's largest in every block-and-day check"
    elif v["traveler_rise"] == "below the archive's":
        trav = ("the constant traveler of the measured shape does not establish sensitivity to a known rise of the archive's size, because its "
                "pooled rise does not reach the archive's, so that sensitivity is untested")
    else:
        trav = ("the constant traveler's pooled rise reaches the archive's, but it fails complete separation from this noise, its smallest "
                "rise not lying above noise's largest in every block-and-day check")
    out = [arch, trav]
    if v["archive"] != "above" and v["traveler"] != "separates" and v["traveler_rise"] == "reaches the archive's":
        out.append("neither the archive's rise nor a traveler's rise of its size separates completely from this noise, and this exploratory "
                   "comparison does not calibrate the archive's physical interpretation")
    out.append("the counts are not calibrated confidence levels, and the readings hold for this noise model, this traveler and these 20 placements, "
               "count no histories one by one, and leave travel unresolved")
    return out


def fmt(v, f=".3f"):
    return "none" if v is None else format(v, f)


def text(res, s, v, archive):
    out = ["# fixed-cohort daily profile against matched noise", "",
           "| sample | placement | block | histories | cohort | day 0 | day 1 | day 2 | day 3 | day 4 | rise to day 3 | rise to day 4 |",
           "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    out.append(f"| archive | | | {archive['changing_histories']} | {archive['cohort_histories']} | " + " | ".join(fmt(x) for x in archive["by_day"])
               + f" | {fmt(archive['rise']['3'])} | {fmt(archive['rise']['4'])} |")
    for name, _ in SAMPLES:
        for e in res[name]:
            b = next(k for k, ps in BLOCKS.items() if e["placement"] in ps)
            out.append(f"| {name} | {e['placement']} | {b} | {e['changing']['histories']} | {e['cohort_histories']} | "
                       + " | ".join(fmt(x) for x in e["matched"]["by_day"]) + f" | {fmt(e['rise']['3'])} | {fmt(e['rise']['4'])} |")
    out += ["", "| sample | block | cohort sizes | rise to day 3 (min, max) | rise to day 4 (min, max) | pooled cohort curve, days 0 to 4 |",
            "| --- | --- | --- | --- | --- | --- |"]
    for name, _ in SAMPLES:
        for b in BLOCKS:
            x = s[name][b]
            out.append(f"| {name} | {b} | {x['cohort_histories'][0]} to {x['cohort_histories'][1]} | {fmt(x['rise']['3'][0])}, {fmt(x['rise']['3'][1])} | "
                       f"{fmt(x['rise']['4'][0])}, {fmt(x['rise']['4'][1])} | " + ", ".join(fmt(y) for y in x["pooled"]["by_day"])
                       + f" over {x['pooled']['histories']} |")
    out += ["", f"Archive: {v['archive']}. Traveler: {v['traveler']}, pooled rise {v['traveler_rise']}.", ""] + [f"- {x}" for x in readings(v)]
    return "\n".join(out) + "\n"


def archive_values(matched_rec):
    m = matched_rec["results"]
    return {"by_day": m["matched"]["by_day"], "cohort_histories": len(m["cohort"]), "changing_histories": m["changing"]["histories"],
            "rise": rises(m["matched"]["by_day"])}


def load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for opt in ("--shape", "--inject", "--records", "--histories", "--xband", "--prestart-constructions", "--matched-archive", "--out-dir"):
        ap.add_argument(opt, required=True)
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    pj, pm = (os.path.join(a.out_dir, f"eval60_matched_noise{a.tag}.{e}") for e in ("json", "md"))
    C.refuse_existing([pj, pm])
    prestart_rec, matched_rec = load_json(a.prestart_constructions), load_json(a.matched_archive)
    if prestart_rec["generated_by"]["sha256"] != PS.SCRIPT_SHA256_AT_LOAD:
        raise SystemExit("REFUSED: the pre-start constructions were made by a different version of their script")
    if matched_rec["generated_by"]["sha256"] != M.SCRIPT_SHA256_AT_LOAD:
        raise SystemExit("REFUSED: the matched-profile record was made by a different version of its script")
    hist, tracks, by, matched_std, gen = C.load_inputs(a)
    M.check_same_inputs(gen, prestart_rec)
    M.check_same_inputs(gen, matched_rec)
    res = run(tracks, by, matched_std, P.measured_parts())
    check_reproduced(res, prestart_rec)
    archive = archive_values(matched_rec)
    s = summary(res); v = verdict(archive["rise"], s, res)
    gen.update({"script": "scripts/eastern_matched_noise.py", "sha256": SCRIPT_SHA256_AT_LOAD, "declared": "EASTERN_MATCHED_NOISE_2026-10-23.md",
                "prestart_script_sha256": PS.SCRIPT_SHA256_AT_LOAD, "matched_script_sha256": M.SCRIPT_SHA256_AT_LOAD,
                "prestart_constructions": {"path": a.prestart_constructions, "sha256": sha256_of(a.prestart_constructions)},
                "matched_archive": {"path": a.matched_archive, "sha256": sha256_of(a.matched_archive)}})
    with open(pj, "w", encoding="utf-8") as fh:
        json.dump({"generated_by": gen, "results": {"placements": res, "archive": archive, "summary": s, "verdict": v, "readings": readings(v)}}, fh)
    t = text(res, s, v, archive)
    with open(pm, "w", encoding="utf-8") as fh:
        fh.write(t)
    print(t)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
