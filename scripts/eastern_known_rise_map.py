#!/usr/bin/env python3
"""The detectability map of the known rise (`EASTERN_KNOWN_RISE_2026-10-24.md`, section 7): for
each multiple of the growing traveler's lobe read on the noise comparison's matched noise, the
pooled cohort rise on the noise and the outcome each of its 20 placements receives under the
rule the noise comparison applied to the archive, with the smallest multiple whose pooled rise
on the noise reaches the archive's. The counts describe this construction on this noise model,
are not calibrated probabilities, and say nothing about whether the archive's histories travel.

    python3 scripts/eastern_known_rise_map.py --noise-archive PATH --matched-archive PATH --runs PATH [PATH ...] --out-dir DIR
"""

import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
import eastern_known_rise as K  # noqa: E402

N, M, PS = K.N, K.M, K.PS
DAYS = tuple(map(str, K.RISES))
OUTCOMES = ("above", "mixed", "not separated")


def sha256_of(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


SCRIPT_SHA256_AT_LOAD = sha256_of(os.path.abspath(__file__))


def outcome(rise, maxima):
    """The noise comparison's own rule for one pair of rises, with the number of checks passed."""
    o, above = N.exceedance(rise, maxima)
    return o, sum(above.values())


def row(multiple, entries, maxima, archive_rise):
    """One multiple: the pooled rises on the noise by the noise comparison's summary, whether they
    reach the archive's in both blocks on both days, and its placements' outcomes."""
    s = K.block_summary(entries)
    pooled = {b: {k: N.pooled_rise(s[b]["pooled"]["by_day"], k) for k in DAYS} for b in K.BLOCKS}
    per = [outcome(e["rise"], maxima)[0] for e in entries]
    return {"multiple": multiple, "placements": len(entries), "pooled_rise": pooled,
            "reaches_archive": all(pooled[b][k] >= archive_rise[k] for b in K.BLOCKS for k in DAYS),
            "outcomes": {o: per.count(o) for o in OUTCOMES}, "by_placement": per}


def build_map(runs, maxima, archive_rise):
    """Every multiple's row, the construction as built taken once from the runs, which must agree
    on it, and refused when a multiple appears twice."""
    rows, built = {}, None
    for rec in runs:
        m = rec["generated_by"]["lobe_mult"]
        if rec["generated_by"]["grow_days"] != PS.GROW_DAYS:
            raise SystemExit(f"REFUSED: the run at multiple {m:g} does not grow over {PS.GROW_DAYS:g} days")
        if m in rows:
            raise SystemExit(f"REFUSED: two runs at multiple {m:g}")
        rows[m] = row(m, rec["results"]["placements"], maxima, archive_rise)
        st = rec["results"]["stored_case_placements"]
        if st is not None:
            if built is not None and st != built:
                raise SystemExit("REFUSED: the runs disagree on the construction as built")
            built = st
    if built is not None and 1.0 not in rows:
        rows[1.0] = row(1.0, built, maxima, archive_rise)
    out = [rows[m] for m in sorted(rows)]
    matched = next((r for r in out if r["reaches_archive"]), None)
    return {"rows": out, "noise_matched": None if matched is None else matched["multiple"]}


def readings(mp, archive_outcome, archive_passed):
    out = [f"under the noise comparison's rule the archive is {archive_outcome.upper()}, passing {archive_passed} of its 4 block-and-day checks"]
    if mp["noise_matched"] is None:
        out.append(f"no multiple up to {mp['rows'][-1]['multiple']:g} gives a pooled rise on this noise that reaches the archive's in both blocks on both days")
    else:
        r = next(x for x in mp["rows"] if x["multiple"] == mp["noise_matched"]); o = r["outcomes"]
        out.append(f"the smallest multiple whose pooled rise on this noise reaches the archive's is {r['multiple']:g}, and of its {r['placements']} "
                   f"placements {o['above']} come out ABOVE, {o['mixed']} MIXED and {o['not separated']} NOT SEPARATED under the same rule")
    out.append("the counts describe this constructed traveler on this noise model, are not calibrated probabilities, and leave travel unresolved")
    return out


def fmt(v, f=".3f"):
    return "none" if v is None else format(v, f)


def text(mp, rd):
    out = ["# detectability map of the known rise on matched noise", "",
           "| multiple | pooled rise to day 3, A and B | pooled rise to day 4, A and B | reaches the archive's | ABOVE | MIXED | NOT SEPARATED |",
           "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in mp["rows"]:
        p, o = r["pooled_rise"], r["outcomes"]
        out.append(f"| {r['multiple']:g} | {fmt(p['A']['3'])}, {fmt(p['B']['3'])} | {fmt(p['A']['4'])}, {fmt(p['B']['4'])} | "
                   f"{'yes' if r['reaches_archive'] else 'no'} | {o['above']} | {o['mixed']} | {o['not separated']} |")
    return "\n".join(out + [""] + [f"- {x}" for x in rd]) + "\n"


def load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--noise-archive", required=True)
    ap.add_argument("--matched-archive", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    pj, pm = (os.path.join(a.out_dir, f"eval60_known_rise_map{a.tag}.{e}") for e in ("json", "md"))
    K.C.refuse_existing([pj, pm])
    noise_rec, matched_rec = load_json(a.noise_archive), load_json(a.matched_archive)
    if noise_rec["generated_by"]["sha256"] != N.SCRIPT_SHA256_AT_LOAD or matched_rec["generated_by"]["sha256"] != M.SCRIPT_SHA256_AT_LOAD:
        raise SystemExit("REFUSED: the noise or matched-profile record was made by a different version of its script")
    runs = [load_json(p) for p in a.runs]
    for p, rec in zip(a.runs, runs):
        g = rec["generated_by"]
        if g["sha256"] != K.SCRIPT_SHA256_AT_LOAD:
            raise SystemExit(f"REFUSED: {p} was made by a different version of the known-rise script")
        if g["noise_archive"]["sha256"] != sha256_of(a.noise_archive) or g["matched_archive"]["sha256"] != sha256_of(a.matched_archive):
            raise SystemExit(f"REFUSED: {p} was read against another noise or matched-profile record")
        M.check_same_inputs(g, noise_rec)
    archive_rise = N.archive_values(matched_rec)["rise"]
    maxima = N.noise_maxima(noise_rec["results"]["summary"])
    a_out, a_passed = outcome(archive_rise, maxima)
    if a_out != noise_rec["results"]["verdict"]["archive"]:
        raise SystemExit("REFUSED: the archive's outcome differs from the noise record's verdict")
    mp = build_map(runs, maxima, archive_rise)
    rd = readings(mp, a_out, a_passed)
    gen = {"script": "scripts/eastern_known_rise_map.py", "sha256": SCRIPT_SHA256_AT_LOAD, "declared": "EASTERN_KNOWN_RISE_2026-10-24.md",
           "noise_archive": {"path": a.noise_archive, "sha256": sha256_of(a.noise_archive)},
           "matched_archive": {"path": a.matched_archive, "sha256": sha256_of(a.matched_archive)},
           "runs": {p: sha256_of(p) for p in a.runs}}
    with open(pj, "w", encoding="utf-8") as fh:
        json.dump({"generated_by": gen, "results": {"archive": {"rise": archive_rise, "outcome": a_out, "checks_passed": a_passed},
                                                     "noise_maxima": {f"day {k}, block {b}": v for (k, b), v in maxima.items()}, **mp, "readings": rd}}, fh)
    t = text(mp, rd)
    with open(pm, "w", encoding="utf-8") as fh:
        fh.write(t)
    print(t)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
