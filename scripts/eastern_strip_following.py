#!/usr/bin/env python3
"""The strip-following measure (`EASTERN_STRIP_FOLLOWING_2026-10-20.md`): the connected runs of
daily band means at or below -4 m/s over the whole band, 20 to 75 E, followed from one day to
the next by overlap first and by the nearest centroid within 6 degrees second, with splits and
merges recorded, then two questions. When S2's strongest position changes between consecutive
days, is it the same strip or another one, and are the largest steps the changes of strip? On
what strip does each member of a westward history lie, one that itself moves west by the
histories' criterion, another strip, or none? Strips are positions of qualifying flow, and
nothing here establishes a physical feature's identity.

    python3 scripts/eastern_strip_following.py constructions --out-dir DIR
    python3 scripts/eastern_strip_following.py archive --records DIR --histories PATH \\
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
import eastern_wobble_climatology as W  # noqa: E402

YEARS = SC.YEARS
MONTHS = SC.MONTHS
LONS = A.LONS
ALL = np.arange(len(LONS))
THRESHOLD = W.THRESHOLD
FALLBACK_DEG = Fraction(int(HI.S_MAX))
S2 = "S2"
S2_BANDS = W.ELEMENTS[S2]["bands"]
S2_COLS = W.window_cols(W.ELEMENTS[S2]["window"])


def sha256_of(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


SCRIPT_SHA256_AT_LOAD = sha256_of(os.path.abspath(__file__))


# ------------------------------------------------------------------ strips

def day_runs(h_row, threshold=THRESHOLD):
    """The day's qualifying runs over the whole band, each with its first and last columns, its
    exact centroid weighted by the negated sums, and its strongest column (westernmost on a tie)."""
    h = np.asarray(h_row, dtype=np.int64)
    out = []
    for a, b in W.runs(h, ALL, threshold):
        seg = h[a:b + 1]
        w = [int(-v) for v in seg]
        x2 = [int(round(2 * LONS[c])) for c in range(a, b + 1)]
        out.append({"first": a, "last": b, "centroid": Fraction(sum(p * q for p, q in zip(x2, w)), 2 * sum(w)),
                    "strongest": a + int(np.argmin(seg))})
    return out


def link_strips(H, threshold=THRESHOLD, fallback=FALLBACK_DEG):
    """The strips of one band over one season. A run on day d + 1 continues the strip of a run on
    day d when the two share a column, pairs taken in order of the larger overlap, then the
    nearer centroids, then the western runs. Runs left unpaired on both days are then paired by
    the nearest centroids within `fallback` degrees. Pairing is one to one. A run that overlaps
    a continued strip without continuing it starts a strip born by a split, and a run whose
    strip is not continued although it overlaps a run of the next day ends by a merge. Returns
    the strips and, per day, the strip of each run."""
    strips, prev, by_day = [], [], []
    for d, row in enumerate(H):
        runs = day_runs(row, threshold)
        ids, kinds = [None] * len(runs), [None] * len(runs)
        overlaps = []
        for pi, (_, a) in enumerate(prev):
            for bi, b in enumerate(runs):
                o = min(a["last"], b["last"]) - max(a["first"], b["first"]) + 1
                if o > 0:
                    overlaps.append((-o, abs(a["centroid"] - b["centroid"]), a["first"], b["first"], pi, bi))
        overlaps.sort()
        taken_p, taken_b = set(), set()
        for *_, pi, bi in overlaps:
            if pi in taken_p or bi in taken_b:
                continue
            taken_p.add(pi); taken_b.add(bi); ids[bi] = prev[pi][0]; kinds[bi] = "overlap"
        near = []
        for pi, (_, a) in enumerate(prev):
            if pi in taken_p:
                continue
            for bi, b in enumerate(runs):
                if bi in taken_b:
                    continue
                dist = abs(a["centroid"] - b["centroid"])
                if dist <= fallback:
                    near.append((dist, a["first"], b["first"], pi, bi))
        near.sort()
        for *_, pi, bi in near:
            if pi in taken_p or bi in taken_b:
                continue
            taken_p.add(pi); taken_b.add(bi); ids[bi] = prev[pi][0]; kinds[bi] = "fallback"
        for pi, (sid, _) in enumerate(prev):
            if pi not in taken_p:
                into = [ids[bi] for (_, _, _, _, p, bi) in overlaps if p == pi and ids[bi] is not None]
                strips[sid]["ended"] = "merge" if into else "end"
                if into:
                    strips[sid]["merged_into"] = into[0]
                    strips[into[0]]["events"].setdefault(d, []).append(("absorbed", sid))
        for bi, b in enumerate(runs):
            if ids[bi] is None:
                parent = [prev[p][0] for (_, _, _, _, p, q) in overlaps if q == bi]
                strips.append({"days": [], "runs": [], "centroids": [], "strongest": [], "links": [], "events": {},
                               "born": "split" if parent else "new", "ended": "open"})
                if parent:
                    strips[-1]["split_from"] = parent[0]
                    strips[parent[0]]["events"].setdefault(d, []).append(("split_off", len(strips) - 1))
                ids[bi] = len(strips) - 1
            s = strips[ids[bi]]
            s["days"].append(d); s["runs"].append((b["first"], b["last"])); s["centroids"].append(b["centroid"])
            s["strongest"].append(b["strongest"]); s["links"].append(kinds[bi])
        by_day.append([(b["first"], b["last"], ids[bi]) for bi, b in enumerate(runs)])
        prev = [(ids[bi], b) for bi, b in enumerate(runs)]
    for s in strips:
        if s["ended"] == "open":
            s["ended"] = "season"
    return strips, by_day


def strip_at(by_day, d, col):
    """The strip whose run holds column `col` on day `d`, or None."""
    for a, b, sid in by_day[d]:
        if a <= col <= b:
            return sid
    return None


# ------------------------------------------------------------------ synthetic seasons

CONSTRUCTION_NOISE_STD = W.CONSTRUCTION_NOISE_STD


def synth(features, noise_seed=None, noise_std=CONSTRUCTION_NOISE_STD):
    """Daily integer sums of six-hourly samples of Gaussian troughs of the anomaly module's width,
    each feature a pair of functions of time in days giving its center (None when absent) and
    its depth, plus seeded noise, on the exact path. Returns the sums and the daily-mean center
    of every feature (NaN on a day it is absent at any sample)."""
    t = np.arange(A.SEASON_DAYS * 4) / 4.0
    V = np.zeros((len(t), len(LONS)))
    pos = np.full((len(features), len(t)), np.nan)
    for k, tk in enumerate(t):
        for f, (center, depth) in enumerate(features):
            c = center(tk)
            if c is not None:
                pos[f, k] = c
                V[k] += A.gauss(LONS, c, depth(tk))
    if noise_seed is not None:
        V = V + A.noise_field(noise_std, seed=noise_seed, lons=LONS)
    return A.hundredths_from_samples(np.round(V, 2)), pos.reshape(len(features), A.SEASON_DAYS, 4).mean(axis=2)


# ------------------------------------------------------------------ question 1, continuation or change of strip

def col_of(lon):
    return int(np.flatnonzero(np.isclose(LONS, lon))[0])


def s2_steps(H, by_day):
    """For each pair of consecutive days with S2 present in its window, the change of S2's core
    (the wobble record's repaired measure) and whether the window's strongest cell lies on the
    same strip on both days."""
    out, prev = [], None
    for d, row in enumerate(H):
        m = W.day_measure(row, S2_COLS)
        if m is None:
            prev = None
            continue
        sid = strip_at(by_day, d, col_of(m["strongest"]))
        if sid is None:
            raise SystemExit(f"REFUSED: S2's strongest cell on day {d} lies on no strip")
        if prev is not None:
            out.append({"day": d, "step": abs(m["core"] - prev[0]), "continuation": sid == prev[1]})
        prev = (m["core"], sid)
    return out


def pct(values):
    v = [float(x) for x in values]
    return [float(np.percentile(v, q)) for q in (25, 50, 75, 90)] if v else None


def switch_summary(steps):
    """Counts of continuations and changes of strip, the step quartiles and 90th percentile of
    each, and among the largest tenth of steps (the ceil(n/10) largest and every step tied with
    the smallest of them), how many are changes of strip."""
    n = len(steps)
    cont = [s["step"] for s in steps if s["continuation"]]
    sw = [s["step"] for s in steps if not s["continuation"]]
    out = {"pairs": n, "continuations": len(cont), "changes": len(sw), "continuation_step_q": pct(cont), "change_step_q": pct(sw)}
    if n:
        cutoff = sorted((s["step"] for s in steps), reverse=True)[-(-n // 10) - 1]
        large = [s for s in steps if s["step"] >= cutoff]
        out.update(large_cutoff=float(cutoff), large=len(large), large_changes=sum(1 for s in large if not s["continuation"]))
    else:
        out.update(large_cutoff=None, large=0, large_changes=0)
    return out


# ------------------------------------------------------------------ question 2, the strips under a history

CLASSES = ("with_resident", "with_new", "through", "within", "undetermined", "none")
RESIDENT_DAYS = 5
TRANSITIONS = ("same", "merge", "split", "jump", "on", "off")


def history_path(h, strips, by_day):
    """The strip under each member of a history, None when the member lies on no strip, and the
    transition into it from the previous member: the same strip, a merge (the previous strip
    ended the day before by merging into this one), a split (this strip was born today by
    splitting from the previous one), a jump to an unrelated strip, or a step on or off strips."""
    path = []
    for k, (d, x) in enumerate(zip(h["days"], h["lons"])):
        sid = strip_at(by_day, d, col_of(x))
        if k == 0:
            t = "start"
        else:
            p = path[-1]["strip"]
            if sid is None:
                t = "off"
            elif p is None:
                t = "on"
            elif sid == p:
                t = "same"
            elif strips[p].get("merged_into") == sid and strips[p]["days"][-1] == d - 1:
                t = "merge"
            elif strips[sid].get("split_from") == p and strips[sid]["days"][0] == d:
                t = "split"
            else:
                t = "jump"
        path.append({"day": d, "lon": x, "strip": sid, "into": t})
    return path


def segments(path, strips):
    """Maximal runs of consecutive members on one strip, a new run starting on a day the strip
    absorbs another strip or another strip splits from it, and runs of members on no strip."""
    out, k = [], 0
    while k < len(path):
        sid, e = path[k]["strip"], k
        while e + 1 < len(path) and path[e + 1]["strip"] == sid and (sid is None or (path[e + 1]["into"] == "same" and path[e + 1]["day"] not in strips[sid]["events"])):
            e += 1
        out.append((k, e))
        k = e + 1
    return out


def resident(strip, day):
    """True when the strip existed on each of the RESIDENT_DAYS days before `day` with its
    centroid inside S2's window."""
    lo, hi = W.ELEMENTS[S2]["window"]
    for d in range(day - RESIDENT_DAYS, day):
        if d not in strip["days"] or not lo <= strip["centroids"][strip["days"].index(d)] <= hi:
            return False
    return True


def segment_class(path, k, e, strips):
    """'with' when the strip's centroid moves west over the segment by at least half the
    member's westward displacement, 'with_resident' when the strip was resident in S2's window
    on the five days before the segment and 'with_new' when it was not, otherwise 'through' when the segment
    begins or ends at a merge or a split (of the history's strips or of this strip), and
    'within' when it does not. 'undetermined' for a segment whose members do not move west,
    which includes a segment of one member, 'none' off strips. Decided on exact positions."""
    sid = path[k]["strip"]
    if sid is None:
        return "none"
    s = strips[sid]
    dm = Fraction(path[e]["lon"]).limit_denominator(2) - Fraction(path[k]["lon"]).limit_denominator(2)
    if dm >= 0:
        return "undetermined"
    ds = s["centroids"][s["days"].index(path[e]["day"])] - s["centroids"][s["days"].index(path[k]["day"])]
    if 2 * ds <= dm:
        return "with_resident" if resident(s, path[k]["day"]) else "with_new"
    starts = path[k]["into"] in ("merge", "split") or path[k]["day"] in s["events"]
    ends = e + 1 < len(path) and (path[e + 1]["into"] in ("merge", "split") or (path[e + 1]["strip"] == sid and path[e + 1]["day"] in s["events"]))
    return "through" if (starts or ends) else "within"


def member_classes(path, strips):
    out = [None] * len(path)
    for k, e in segments(path, strips):
        c = segment_class(path, k, e, strips)
        for j in range(k, e + 1):
            out[j] = c
    return out


def history_strips(histories, seasons, s2_only=False):
    """Member-days of the histories by class and the transitions along their paths, and per
    history its class over the counted members: 'through' when any counted member is, otherwise
    the class shared by every counted member, otherwise 'mixed'. With `s2_only`, only members
    inside S2's window in S2's bands count, and only histories with such members. `seasons`
    maps (year, band) to (strips, by_day)."""
    lo, hi = W.ELEMENTS[S2]["window"]
    counts = {c: 0 for c in CLASSES}
    trans = {t: 0 for t in TRANSITIONS}
    per = {c: 0 for c in CLASSES + ("mixed",)}
    for (y, b), hs in histories.items():
        if s2_only and b not in S2_BANDS:
            continue
        strips, by_day = seasons[(y, b)]
        for h in hs:
            path = history_path(h, strips, by_day)
            mc = member_classes(path, strips)
            keep = [k for k, m in enumerate(path) if not s2_only or lo <= m["lon"] <= hi]
            if not keep:
                continue
            for k in keep:
                counts[mc[k]] += 1
            for m in path[1:]:
                trans[m["into"]] += 1
            got = {mc[k] for k in keep}
            per["through" if "through" in got else (got.pop() if len(got) == 1 else "mixed")] += 1
    return {"member_days": sum(counts.values()), "by_class": counts, "transitions": trans, "histories": sum(per.values()), "history_class": per}


# ------------------------------------------------------------------ constructions

CASES = ("K1", "K2", "K3", "K4", "K5", "K6")
PHASE_SEED = W.PHASE_SEED


def construction_season(case, i):
    """One season of a construction: the daily sums, and the daily-mean center of the moving
    feature (None for the cases without one)."""
    const = lambda v: (lambda t: v)
    if case == "K1":
        ph = float(np.random.default_rng(PHASE_SEED + i).uniform(0.0, 2.0 * np.pi))
        H, _ = synth([(lambda t: 50.0 + 3.0 * np.sin(2.0 * np.pi * t / 14.0 + ph), const(10.0))], HI.NOISE_SEEDS[i])
        return H, None
    if case == "K2":
        a = lambda t: 10.0 if int(t // 7) % 2 == 0 else 7.0
        b = lambda t: 7.0 if int(t // 7) % 2 == 0 else 10.0
        H, _ = synth([(const(47.0), a), (const(54.0), b)], HI.NOISE_SEEDS[i])
        return H, None
    if case in ("K3", "K5"):
        start, x0, speed = (35.0, 72.0, 2.0) if case == "K3" else (35.0, 75.0, 5.0)
        moving = (lambda t: (x0 - speed * (t - start)) if t >= start and x0 - speed * (t - start) >= 20.0 else None, const(10.0))
        feats = [(const(50.0), const(10.0)), moving] if case == "K3" else [moving]
        H, pos = synth(feats, HI.NOISE_SEEDS[i])
        return H, pos[-1]
    if case == "K6":
        shift = lambda t: 54.0 - 2.0 * (t - 40.0) if 40.0 <= t < 45.0 else (44.0 if t >= 45.0 else 54.0)
        H, pos = synth([(shift, const(10.0))], HI.NOISE_SEEDS[i])
        return H, pos[0]
    if case == "K4":
        H = np.zeros((A.SEASON_DAYS, len(LONS)), dtype=np.int64)
        center = W.LAG_CENTERS[i]
        H[:, col_of(center)] = W.LAG_FIXED
        tr = np.full(A.SEASON_DAYS, np.nan)
        for day, x in enumerate([center + o for o in W.LAG_OFFSETS], 50):
            H[day, col_of(x)] = W.LAG_CASES["C1"]; tr[day] = x
        return H, tr
    raise SystemExit(f"REFUSED: unknown construction {case}")


def season_strips(H):
    return link_strips(H)


def score_case(case, n_seasons=None):
    """Both questions for one construction over its seasons. Histories are the westward
    histories of the nine-day anomaly with the nearest linker; for the cases with a moving
    feature, only those following it (at least half the members within 1 degree of its
    daily-mean center) count."""
    month_of = A.month_index()
    steps, hist, seasons = [], {}, {}
    for i, y in enumerate(YEARS[:n_seasons]):
        H, tr = construction_season(case, i)
        strips, by_day = season_strips(H)
        seasons[(y, "15 to 19.5 N")] = (strips, by_day)
        steps += s2_steps(H, by_day)
        hs = [h for h in HI.link(HI.rep_minima("RB9", H, month_of, LONS), "LN") if HI.classify(h) == "westward"]
        if tr is not None:
            hs = [h for h in hs if HI.follows(h, tr)]
        hist[(y, "15 to 19.5 N")] = hs
    return {"switch": switch_summary(steps), "histories_all": history_strips(hist, seasons),
            "histories_s2": history_strips(hist, seasons, s2_only=True), "seasons": len(seasons)}


# ------------------------------------------------------------------ the rule, set on the constructions before the archive

RULE = {
    "K1": {"continuation_at_least": Fraction(95, 100), "resident_at_least": Fraction(5, 10), "new_or_through_at_most": Fraction(45, 100)},
    "K2": {"large_changes_at_least": Fraction(9, 10)},
    "K3": {"new_or_through_at_least": Fraction(6, 10), "through_at_least": Fraction(3, 10), "resident_at_most": Fraction(3, 10)},
    "K4": {"none_at_least": Fraction(1)},
    "K6": {"resident_at_least": Fraction(6, 10), "through_at_most": Fraction(1, 10)},
}

def shares(q):
    """Exact shares of the counted member-days, None when there are none."""
    n, c = q["member_days"], q["by_class"]
    if not n:
        return None
    return {"resident": Fraction(c["with_resident"], n), "new_or_through": Fraction(c["with_new"] + c["through"], n),
            "through": Fraction(c["through"], n), "none": Fraction(c["none"], n)}


def apply_rule(results):
    """The section 4 rule over the stored construction results, decided on exact counts."""
    fails = []
    for case, bars in RULE.items():
        r = results[case]
        sw = r["switch"]
        sh = shares(r["histories_s2"])
        for bar, v in bars.items():
            if bar == "continuation_at_least":
                got = Fraction(sw["continuations"], sw["pairs"]) if sw["pairs"] else None
            elif bar == "large_changes_at_least":
                got = Fraction(sw["large_changes"], sw["large"]) if sw["large"] else None
            else:
                key = bar.rsplit("_at_", 1)[0]
                got = None if sh is None else sh[key]
            ok = got is not None and (got >= v if bar.endswith("_at_least") else got <= v)
            if not ok:
                fails.append(f"{case} {bar} {v} not met ({None if got is None else float(got):.4})" if got is not None else f"{case} {bar} unavailable")
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

def archive_results(by, histories):
    """Both questions over the 24 seasons: per S2 band and pooled for the first, and the members
    of the westward histories inside S2's window, per S2 band and pooled, with all members of
    every history beside them, for the second."""
    seasons, steps = {}, {b: [] for b in S2_BANDS}
    bands = list(by[YEARS[0]][MONTHS[0]]["bands"])
    strip_counts = {}
    for y in YEARS:
        for b in bands:
            H = A.season_hundredths(by[y], b)[1]
            strips, by_day = link_strips(H)
            seasons[(y, b)] = (strips, by_day)
            sc = strip_counts.setdefault(b, {"strips": 0, "born_split": 0, "ended_merge": 0, "fallback_links": 0})
            sc["strips"] += len(strips); sc["born_split"] += sum(s["born"] == "split" for s in strips)
            sc["ended_merge"] += sum(s["ended"] == "merge" for s in strips); sc["fallback_links"] += sum(k == "fallback" for s in strips for k in s["links"])
            if b in S2_BANDS:
                steps[b] += s2_steps(H, by_day)
    q1 = {b: switch_summary(v) for b, v in steps.items()}
    q1["pooled"] = switch_summary([s for v in steps.values() for s in v])
    q2 = {"pooled": history_strips(histories, seasons, s2_only=True), "all_members": history_strips(histories, seasons)}
    for b in S2_BANDS:
        q2[b] = history_strips({k: v for k, v in histories.items() if k[1] == b}, seasons, s2_only=True)
    return {"strips": strip_counts, "q1": q1, "q2": q2, "reading": readings(q1["pooled"], q2["pooled"])}


def readings(q1, q2):
    """The fixed readings of section 4."""
    r1 = None
    if q1["large"]:
        r1 = ("the largest steps of S2's core are mostly changes of strip" if 2 * q1["large_changes"] >= q1["large"]
              else "the largest steps of S2's core are mostly continuations of one strip")
    sh = shares(q2)
    r2 = None
    if sh is not None:
        if 2 * sh["resident"] >= 1:
            r2 = "most in-window member-days of the westward histories lie on a strip resident in S2's window that moves with them"
        elif 2 * sh["new_or_through"] >= 1:
            r2 = "most in-window member-days of the westward histories lie on strips new to S2's window or crossing a merge or a split"
    return {"q1": r1, "q2": r2}


# ------------------------------------------------------------------ text and main

def fmt(q, n=2):
    return "" if q is None else ", ".join(f"{x:.{n}f}" for x in q)


def q1_row(name, s):
    return (f"| {name} | {s['pairs']} | {s['continuations']} | {s['changes']} | {fmt(s['continuation_step_q'])} | {fmt(s['change_step_q'])} | "
            f"{'' if s['large_cutoff'] is None else format(s['large_cutoff'], '.2f')} | {s['large_changes']} of {s['large']} |")


def q2_row(name, q):
    c, h = q["by_class"], q["history_class"]
    return (f"| {name} | {q['member_days']} | {c['with_resident']} | {c['with_new']} | {c['through']} | {c['within']} | {c['undetermined']} | {c['none']} | "
            f"{h['with_resident']}, {h['with_new']}, {h['through']}, {h['within']}, {h['undetermined']}, {h['none']}, {h['mixed']} of {q['histories']} |")


Q1_HEAD = ("| case or band | pairs of present days | same strip | change of strip | step on the same strip, quartiles and 90th percentile (deg) | "
           "step at a change, quartiles and 90th percentile (deg) | largest-tenth cutoff (deg) | changes among the largest tenth |\n| --- | --- | --- | --- | --- | --- | --- | --- |")
Q2_HEAD = ("| case or band | member-days | with, resident | with, new | through | within | undetermined | none | "
           "histories by class (resident, new, through, within, undetermined, none, mixed) |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- |")


def constructions_text(results, verdict):
    L = ["# strip following, the constructions (section 3)", "", "## question 1, S2's strongest cell from day to day", "", Q1_HEAD]
    L += [q1_row(c, results[c]["switch"]) for c in CASES]
    L += ["", "## question 2, the members of the westward histories inside S2's window", "", Q2_HEAD]
    L += [q2_row(c, results[c]["histories_s2"]) for c in CASES]
    L += ["", f"adequate under the rule of section 4, {'yes' if verdict['adequate'] else 'no'}"] + [f"- {x}" for x in verdict["failures"]]
    return "\n".join(L) + "\n"


def archive_text(res):
    L = ["# strip following, the archive (section 5)", "", "## strips per band over the 24 seasons", "",
         "| band | strips | born by a split | ended by a merge | links by the 6-degree fallback |", "| --- | --- | --- | --- | --- |"]
    L += [f"| {b} | {v['strips']} | {v['born_split']} | {v['ended_merge']} | {v['fallback_links']} |" for b, v in res["strips"].items()]
    L += ["", "## question 1, S2's strongest cell from day to day", "", Q1_HEAD]
    L += [q1_row(b, res["q1"][b]) for b in S2_BANDS + ("pooled",)]
    L += ["", "## question 2, the members of the westward histories", "", Q2_HEAD]
    L += [q2_row(f"inside S2's window, {b}", res["q2"][b]) for b in S2_BANDS]
    L += [q2_row("inside S2's window, pooled", res["q2"]["pooled"]), q2_row("every member, every band", res["q2"]["all_members"])]
    L += ["", f"reading of question 1 under section 4, {res['reading']['q1'] or 'none'}",
          f"reading of question 2 under section 4, {res['reading']['q2'] or 'none, the result reported by its counts'}"]
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
    gen = {"script": "scripts/eastern_strip_following.py", "sha256": SCRIPT_SHA256_AT_LOAD, "declared": "EASTERN_STRIP_FOLLOWING_2026-10-20.md",
           "wobble_script_sha256": sha256_of(os.path.join(HERE, "eastern_wobble_climatology.py")),
           "histories_script_sha256": sha256_of(os.path.join(HERE, "eastern_minima_histories.py")), "anomaly_script_sha256": sha256_of(os.path.join(HERE, "eastern_anomaly_minima.py"))}
    if a.mode == "constructions":
        pj, pm = (os.path.join(a.out_dir, f"eval60_strips_constructions{a.tag}.{e}") for e in ("json", "md"))
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
    pj, pm = (os.path.join(a.out_dir, f"eval60_strips_archive{a.tag}.{e}") for e in ("json", "md"))
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
