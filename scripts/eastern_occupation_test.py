#!/usr/bin/env python3
"""The occupation test (`EASTERN_OCCUPATION_TEST_2026-10-20.md`): while a westward history rides
a strip that moves west with it inside S2's window, does the position S2 held just before the
event still hold qualifying northerly flow on another strip? The two hypotheses it is built to
discriminate are a shift, one feature that moves and leaves its earlier position vacated, and a
passage, two coexisting features, the earlier position still held while the history moves away.
"Held" and "vacated" are the rule's outputs, and nothing here establishes a feature's identity.

    python3 scripts/eastern_occupation_test.py constructions --out-dir DIR
    python3 scripts/eastern_occupation_test.py archive --records DIR --histories PATH \\
        --constructions PATH --out-dir DIR
"""

import argparse
import glob
import hashlib
import json
import os
import sys
from fractions import Fraction

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eastern_anomaly_minima as A  # noqa: E402
import eastern_minima_histories as HI  # noqa: E402
import eastern_standing_climatology as SC  # noqa: E402
import eastern_strip_following as S  # noqa: E402
import eastern_wobble_climatology as W  # noqa: E402

YEARS = SC.YEARS
MONTHS = SC.MONTHS
LONS = A.LONS
ALL = np.arange(len(LONS))
S2_BANDS = S.S2_BANDS
S2_COLS = S.S2_COLS
S2_WINDOW = W.ELEMENTS["S2"]["window"]
PRE_DAYS = 5
PRE_MIN_PRESENT = 3
TOL = Fraction(3, 2)
SENSITIVITY_THRESHOLD = -3.0
CLASSES = ("held_continuous", "held_new", "vacated", "not_departed", "no_reference")


def sha256_of(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


SCRIPT_SHA256_AT_LOAD = sha256_of(os.path.abspath(__file__))


# ------------------------------------------------------------------ the observable

def pre_position(H, d0):
    """The exact median of S2's core (the wobble record's repaired measure) over the PRE_DAYS
    days before day d0, or None when fewer than PRE_MIN_PRESENT of them have S2 present."""
    cores = []
    for d in range(max(0, d0 - PRE_DAYS), d0):
        m = W.day_measure(H[d], S2_COLS)
        if m is not None:
            cores.append(m["core"])
    if len(cores) < PRE_MIN_PRESENT:
        return None
    return W.exact_median(cores)


def run_distance(a, b, pos):
    """The exact distance from `pos` to the longitude span of the run of columns a to b."""
    lo, hi = Fraction(LONS[a]).limit_denominator(2), Fraction(LONS[b]).limit_denominator(2)
    return lo - pos if pos < lo else (pos - hi if pos > hi else Fraction(0))


def occupants(by_day, d, pos, tol=TOL):
    """The strips whose run on day d has a cell within `tol` of `pos`."""
    return {sid for a, b, sid in by_day[d] if run_distance(a, b, pos) <= tol}


def cells_near(h_row, pos, threshold, tol=TOL):
    """True when a cell within `tol` of `pos` is at or below `threshold`."""
    thr = int(round(threshold * 400))
    h = np.asarray(h_row, dtype=np.int64)
    return any(h[c] <= thr for c in ALL if abs(Fraction(LONS[c]).limit_denominator(2) - pos) <= tol)


HISTORY_READINGS = ("held", "vacated", "tied", "not_departed", "no_reference")


def history_occupation(path, mc, by_day, H):
    """For a history with at least one member inside S2's window in a `with` segment, None
    otherwise: the class of each member and one reading. The pre-event position is S2's core
    over the PRE_DAYS days before the history's first member inside the window. A member is
    departed when that position lies farther than TOL outside its own run, or when it lies on
    no run, so its own run is never among the occupants. A departed member is held when another
    run has a cell within TOL of the position,
    continuously when that run's strip also held the position on the day before the history's
    first in-window day, and vacated otherwise. The history reads held or vacated by the
    majority of its departed members, tied on equal counts, and not_departed when none
    departed. At each departed member, whether a cell within TOL of the position holds flow at
    the sensitivity threshold is counted alongside."""
    lo, hi = S2_WINDOW
    inside = [j for j, m in enumerate(path) if lo <= m["lon"] <= hi]
    if not any(mc[j].startswith("with_") for j in inside):
        return None
    d_in = path[inside[0]]["day"]
    pos = pre_position(H, d_in)
    if pos is None:
        return {"members": [("no_reference", lo <= m["lon"] <= hi) for m in path], "reading": "no_reference", "sensitivity": (0, 0), "sensitivity_reading": "no_reference"}
    before = occupants(by_day, d_in - 1, pos) if d_in >= 1 else set()
    members, sens_dep, sens_held = [], 0, 0
    for m in path:
        d, own = m["day"], m["strip"]
        if own is not None:
            a, b = next((a, b) for a, b, sid in by_day[d] if sid == own)
            if run_distance(a, b, pos) <= TOL:
                members.append(("not_departed", lo <= m["lon"] <= hi))
                continue
        occ = occupants(by_day, d, pos)
        c = "vacated" if not occ else ("held_continuous" if occ & before else "held_new")
        members.append((c, lo <= m["lon"] <= hi))
        sens_dep += 1
        sens_held += int(cells_near(H[d], pos, SENSITIVITY_THRESHOLD))
    held = sum(1 for c, _ in members if c.startswith("held"))
    vac = sum(1 for c, _ in members if c == "vacated")
    reading = "not_departed" if held + vac == 0 else ("held" if held > vac else ("vacated" if vac > held else "tied"))
    sens_vac = sens_dep - sens_held
    sens_reading = "not_departed" if sens_dep == 0 else ("held" if sens_held > sens_vac else ("vacated" if sens_vac > sens_held else "tied"))
    return {"members": members, "reading": reading, "sensitivity": (sens_dep, sens_held), "sensitivity_reading": sens_reading}


def occupation_counts(histories, seasons, fields, bands=S2_BANDS):
    """Over the histories in `bands` with an in-window `with` member: histories by reading,
    member-days by class over all their members and over those inside S2's window, and the
    departed member-days held at the sensitivity threshold, with the histories read again by the
    majority of their departed members holding flow at that threshold and the table of each
    history's two readings (readers added after the archive was read, section 7, the table on
    the review, section 8). `seasons` maps (year, band) to
    (strips, by_day), `fields` to the daily sums."""
    per = {r: 0 for r in HISTORY_READINGS}
    per_sens = {r: 0 for r in HISTORY_READINGS}
    trans = {r: {q: 0 for q in HISTORY_READINGS} for r in HISTORY_READINGS}
    mem_all = {c: 0 for c in CLASSES}
    mem_in = {c: 0 for c in CLASSES}
    sens = {"departed": 0, "held": 0}
    for (y, b), hs in histories.items():
        if b not in bands:
            continue
        strips, by_day = seasons[(y, b)]
        H = fields[(y, b)]
        for h in hs:
            path = S.history_path(h, strips, by_day)
            o = history_occupation(path, S.member_classes(path, strips), by_day, H)
            if o is None:
                continue
            per[o["reading"]] += 1
            per_sens[o["sensitivity_reading"]] += 1
            trans[o["reading"]][o["sensitivity_reading"]] += 1
            for c, inside in o["members"]:
                mem_all[c] += 1
                mem_in[c] += inside
            sens["departed"] += o["sensitivity"][0]; sens["held"] += o["sensitivity"][1]
    dep = mem_all["held_continuous"] + mem_all["held_new"] + mem_all["vacated"]
    return {"histories": sum(per.values()), "history_reading": per, "member_days": mem_all, "member_days_in_window": mem_in,
            "departed_member_days": dep, "held_member_days": mem_all["held_continuous"] + mem_all["held_new"], "sensitivity": sens,
            "sensitivity_history_reading": per_sens, "reading_transitions": trans}


# ------------------------------------------------------------------ constructions

CASES = ("O1", "O2", "O3", "O4", "O5", "O6")
CASE_NOTES = {
    "O1": "one feature of depth 10 wobbling 3 degrees about 50 E with a 14-day period (the strip measure's K1)",
    "O2": "one feature of depth 10 at 54 E shifting to 44 E at 2 degrees per day over days 40 to 45 (the strip measure's K6)",
    "O3": "a feature of depth 10 at 50 E and a transient of depth 10 from 72 E at 2 degrees per day from day 35 (the strip measure's K3)",
    "O4": "a feature of depth 12 at 48 E and a transient of depth 10 from 72 E pausing at 55 E on days 33 to 43",
    "O5": "a feature of depth 5 at 50 E and a transient of depth 12 from 72 E at 2 degrees per day from day 35",
    "O6": "a feature of depth 3 at 50 E, below the threshold alone, and a transient of depth 10 from 72 E at 2 degrees per day from day 35",
}


def construction_season(case, i):
    """One season of an occupation construction: the daily sums and the daily-mean center of
    the moving feature, by which the histories that follow it are selected."""
    const = lambda v: (lambda t: v)
    if case in ("O1", "O2", "O3"):
        return S.construction_season({"O1": "K1", "O2": "K6", "O3": "K3"}[case], i)
    if case == "O4":
        def pause(t):
            if t < 25.0:
                return None
            if t < 33.5:
                return 72.0 - 2.0 * (t - 25.0)
            if t < 43.5:
                return 55.0
            x = 55.0 - 2.0 * (t - 43.5)
            return x if x >= 20.0 else None
        H, pos = S.synth([(const(48.0), const(12.0)), (pause, const(10.0))], HI.NOISE_SEEDS[i])
        return H, pos[-1]
    if case in ("O5", "O6"):
        depth = 5.0 if case == "O5" else 3.0
        moving = (lambda t: (72.0 - 2.0 * (t - 35.0)) if t >= 35.0 and 72.0 - 2.0 * (t - 35.0) >= 20.0 else None,
                  const(12.0 if case == "O5" else 10.0))
        H, pos = S.synth([(const(50.0), const(depth)), moving], HI.NOISE_SEEDS[i])
        return H, pos[-1]
    raise SystemExit(f"REFUSED: unknown construction {case}")


def score_case(case):
    month_of = A.month_index()
    band = "15 to 19.5 N"
    hist, seasons, fields = {}, {}, {}
    for i, y in enumerate(YEARS):
        H, tr = construction_season(case, i)
        seasons[(y, band)] = S.link_strips(H)
        fields[(y, band)] = H
        hs = [h for h in HI.link(HI.rep_minima("RB9", H, month_of, LONS), "LN") if HI.classify(h) == "westward"]
        if tr is not None:
            hs = [h for h in hs if HI.follows(h, tr)]
        hist[(y, band)] = hs
    return occupation_counts(hist, seasons, fields, bands=(band,))


# ------------------------------------------------------------------ the rule, set on the constructions before the archive

RULE = {
    "O2": {"held_histories_at_most": Fraction(3, 10), "held_member_days_at_most": Fraction(1, 4)},
    "O3": {"held_histories_at_least": Fraction(8, 10), "held_member_days_at_least": Fraction(85, 100)},
    "O4": {"held_histories_at_least": Fraction(7, 10), "held_member_days_at_least": Fraction(7, 10)},
}


def held_shares(r):
    """Exact shares: histories reading held among those reading held or vacated, and departed
    member-days held. None where the denominator is zero."""
    hr = r["history_reading"]
    n_h = hr["held"] + hr["vacated"]
    return {"held_histories": Fraction(hr["held"], n_h) if n_h else None,
            "held_member_days": Fraction(r["held_member_days"], r["departed_member_days"]) if r["departed_member_days"] else None}


def apply_rule(results):
    fails = []
    for case, bars in RULE.items():
        sh = held_shares(results[case])
        for bar, v in bars.items():
            key = bar.rsplit("_at_", 1)[0]
            got = sh[key]
            if got is None:
                fails.append(f"{case} {bar} unavailable")
            elif not (got >= v if bar.endswith("_at_least") else got <= v):
                fails.append(f"{case} {bar} {v} not met ({float(got):.4})")
    return {"adequate": not fails, "failures": fails}


def run_constructions():
    return {c: score_case(c) for c in CASES}


def check_constructions(path, rerun=None):
    """The archive runs only on constructions run by this exact script whose stored results equal
    a fresh run and pass the rule recomputed from them."""
    with open(path, encoding="utf-8") as fh:
        rec = json.load(fh)
    if rec["generated_by"]["sha256"] != SCRIPT_SHA256_AT_LOAD:
        raise SystemExit("REFUSED: the constructions were run by a different version of this script")
    fresh = json.loads(json.dumps({"results": (rerun or run_constructions)()}, default=float))["results"]
    if fresh != rec.get("results"):
        raise SystemExit("REFUSED: the constructions record differs from a fresh run of the same constructions")
    verdict = apply_rule(rec["results"])
    if not verdict["adequate"]:
        raise SystemExit(f"REFUSED: the constructions do not pass the rule of section 4 ({verdict['failures']})")
    if verdict != rec.get("verdict"):
        raise SystemExit("REFUSED: the stored summary disagrees with the rule applied to the stored cases")
    return rec


# ------------------------------------------------------------------ the archive

READINGS = {"held": ("among the {n} histories with an in-window carried member that read held or vacated (of {total}), held ones outnumber vacated ones, {held} to {vacated}, "
                     "S2's pre-event position keeping qualifying flow on another run while their members were away from it, which is what the passage hypothesis predicts"),
            "vacated": ("among the {n} histories with an in-window carried member that read held or vacated (of {total}), vacated ones outnumber held ones, {vacated} to {held}, "
                        "S2's pre-event position holding no qualifying flow on another run while their members were away from it, which is what the shift hypothesis predicts")}


def reading(r):
    """Section 2.5's fixed reading over the pooled histories that read held or vacated, naming
    that population and its counts, since tied and not-departed histories are outside it."""
    hr = r["history_reading"]
    if hr["held"] + hr["vacated"] == 0 or hr["held"] == hr["vacated"]:
        return None
    key = "held" if hr["held"] > hr["vacated"] else "vacated"
    return READINGS[key].format(n=hr["held"] + hr["vacated"], total=r["histories"], held=hr["held"], vacated=hr["vacated"])


def archive_results(by, histories):
    seasons, fields = {}, {}
    for y in YEARS:
        for b in S2_BANDS:
            H = A.season_hundredths(by[y], b)[1]
            seasons[(y, b)] = S.link_strips(H)
            fields[(y, b)] = H
    res = {b: occupation_counts({k: v for k, v in histories.items() if k[1] == b}, seasons, fields, bands=(b,)) for b in S2_BANDS}
    res["pooled"] = occupation_counts(histories, seasons, fields)
    return {"occupation": res, "reading": reading(res["pooled"])}


# ------------------------------------------------------------------ text and main

HEAD = ("| case or band | histories | held | vacated | tied | not departed | no reference | departed member-days | held | of which continuously | held at -3 m/s | "
        "histories at -3 m/s, held, vacated, tied |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")


def row(name, r):
    hr, m = r["history_reading"], r["member_days"]
    return (f"| {name} | {r['histories']} | {hr['held']} | {hr['vacated']} | {hr['tied']} | {hr['not_departed']} | {hr['no_reference']} | "
            f"{r['departed_member_days']} | {r['held_member_days']} | {m['held_continuous']} | {r['sensitivity']['held']} of {r['sensitivity']['departed']} | "
            f"{r['sensitivity_history_reading']['held']}, {r['sensitivity_history_reading']['vacated']}, {r['sensitivity_history_reading']['tied']} |")


def constructions_text(results, verdict):
    L = ["# occupation test, the constructions (section 3)", "", HEAD] + [row(c, results[c]) for c in CASES]
    L += ["", f"adequate under the rule of section 4, {'yes' if verdict['adequate'] else 'no'}"] + [f"- {x}" for x in verdict["failures"]]
    return "\n".join(L) + "\n"


def archive_text(res):
    L = ["# occupation test, the archive (section 5)", "", HEAD] + [row(b, res["occupation"][b]) for b in S2_BANDS + ("pooled",)]
    L += ["", f"reading under section 2.5, {res['reading'] or 'none, the result reported by its counts'}"]
    t = res["occupation"]["pooled"]["reading_transitions"]
    L += ["", "## pooled histories, the reading at -4 m/s against the reading at -3 m/s", "",
          "| at -4 m/s | held at -3 | vacated at -3 | tied at -3 | not departed at -3 | no reference at -3 |", "| --- | --- | --- | --- | --- | --- |"]
    L += [f"| {a} | " + " | ".join(str(t[a][b]) for b in HISTORY_READINGS) + " |" for a in HISTORY_READINGS]
    return "\n".join(L) + "\n"


def refuse_existing(paths):
    for p in paths:
        if os.path.exists(p):
            raise SystemExit(f"REFUSED: {p} exists and a record is never overwritten")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("constructions", "archive"))
    ap.add_argument("--out-dir", required=True)
    for opt in ("--records", "--histories", "--constructions"):
        ap.add_argument(opt, default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    gen = {"script": "scripts/eastern_occupation_test.py", "sha256": SCRIPT_SHA256_AT_LOAD, "declared": "EASTERN_OCCUPATION_TEST_2026-10-20.md",
           "strip_script_sha256": sha256_of(os.path.join(HERE, "eastern_strip_following.py")), "wobble_script_sha256": sha256_of(os.path.join(HERE, "eastern_wobble_climatology.py")),
           "histories_script_sha256": sha256_of(os.path.join(HERE, "eastern_minima_histories.py")), "anomaly_script_sha256": sha256_of(os.path.join(HERE, "eastern_anomaly_minima.py"))}
    if a.mode == "constructions":
        pj, pm = (os.path.join(a.out_dir, f"eval60_occupation_constructions{a.tag}.{e}") for e in ("json", "md"))
        refuse_existing([pj, pm])
        results = json.loads(json.dumps(run_constructions(), default=float)); verdict = apply_rule(results)
        with open(pj, "w", encoding="utf-8") as fh:
            json.dump({"generated_by": gen, "results": results, "verdict": verdict}, fh)
        text = constructions_text(results, verdict)
        with open(pm, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(text)
        return 0
    if not all((a.records, a.histories, a.constructions)):
        raise SystemExit("REFUSED: --records, --histories and --constructions are required for the archive")
    check_constructions(a.constructions)
    pj, pm = (os.path.join(a.out_dir, f"eval60_occupation_archive{a.tag}.{e}") for e in ("json", "md"))
    refuse_existing([pj, pm])
    files = sorted(glob.glob(os.path.join(a.records, "eval60_regime_????_??.json.gz")))
    recs = [SC.load_record(p) for p in files]
    SC.check_archive_identity(files, recs)
    by = {}
    for r in recs:
        by.setdefault(int(r["year"]), {})[int(r["month"])] = r
    with open(a.histories, encoding="utf-8") as fh:
        expected = W.stored_history_count(json.load(fh))
    res = archive_results(by, W.recompute_histories(by, expected))
    gen.update({"records": {os.path.basename(p): sha256_of(p) for p in files}, "histories": {"path": a.histories, "sha256": sha256_of(a.histories)},
                "constructions": {"path": a.constructions, "sha256": sha256_of(a.constructions)}})
    with open(pj, "w", encoding="utf-8") as fh:
        json.dump({"generated_by": gen, "results": res}, fh, default=float)
    text = archive_text(res)
    with open(pm, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
