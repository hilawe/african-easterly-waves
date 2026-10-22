#!/usr/bin/env python3
"""The anomaly at the westward histories' first longitude before their first day
(`EASTERN_PRESTART_2026-10-22.md`): whether southerly anomaly is already present where a history
begins, which a lobe standing near the start gives and a trough born there or growing into view
does not, read against travelers of the measured shape at constant amplitude and growing over
their first two days, and lobes standing at or 2 degrees west of the first longitude, in two
independent blocks of ten placements on the archive and on matched noise. The pre-start reading
is read with the joint test's three readings, and a verdict counts only when both blocks agree.
The measure says which declared constructions stay consistent with the archive. It excludes no
construction outside the family, identifies no feature, and does not say what any one history
is.

    python3 scripts/eastern_prestart.py constructions --shape PATH --inject PATH --records DIR --histories PATH --xband DIR \\
        --joint-constructions PATH --out-dir DIR
    python3 scripts/eastern_prestart.py archive --shape PATH --inject PATH --records DIR --histories PATH --xband DIR \\
        --lagreg PATH --joint-constructions PATH --joint-archive PATH --constructions PATH --out-dir DIR
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
import eastern_joint_standing as G  # noqa: E402

P, C, H, J, L, A, HI = G.P, G.C, G.H, G.J, G.L, G.A, G.HI
YEARS = G.YEARS
XL = G.XL
BANDS = G.BANDS
GROWTHS = ("constant", "whole_quarter", "whole_zero", "lobe_zero")
GROW_DAYS = 2.0
OFFSETS = (-2.0, 0.0)
WIDTHS = (1.0, 2.0, 4.0)
STRENGTHS = (0.5, 1.0, 1.5)
PRE_DAYS = (-2, -1)
PROFILE_DAYS = (-3, -2, -1, 0)
BLOCKS = {"A": tuple(range(0, 10)), "B": tuple(range(10, 20))}
PLACEMENTS = BLOCKS["A"] + BLOCKS["B"]
JOINT_PLACEMENTS = tuple(range(5))
DECIDING = ("o-2_w4_s1_steady", "o-2_w4_s1_relax", "o0_w2_s1_steady", "o0_w2_s1.5_relax")
READERS = ("departure", "contrast", "first_day", "prestart")


def sha256_of(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


SCRIPT_SHA256_AT_LOAD = sha256_of(os.path.abspath(__file__))


# ------------------------------------------------------------------ the cases

def case_name(case):
    if case["kind"] == "travel":
        return "travel" if case["growth"] == "constant" else f"travel_{case['growth']}"
    return G.layout_name(case)


def cases():
    """The four travelers, then every standing lobe at the declared offsets, widths, strengths
    and the joint test's timings."""
    out = [{"kind": "travel", "growth": g} for g in GROWTHS]
    for o, w, s, t in itertools.product(OFFSETS, WIDTHS, STRENGTHS, G.TIMINGS):
        out.append({"kind": "standing", "offset": o, "width": w, "strength": s, "timing": t})
    return out


def growth_factor(growth, age):
    """The traveler's amplitude at an age in days: constant at 1, 'whole_quarter' from a quarter to
    1 over GROW_DAYS, 'whole_zero' and 'lobe_zero' from 0 to 1 over GROW_DAYS."""
    if growth == "constant":
        return 1.0
    if growth == "whole_quarter":
        return 0.25 + 0.75 * min(age / GROW_DAYS, 1.0)
    if growth in ("whole_zero", "lobe_zero"):
        return min(age / GROW_DAYS, 1.0)
    raise SystemExit(f"REFUSED: unknown growth {growth}")


def build(tracks, rng, case, parts):
    """The joint test's constructions for the constant traveler and the standing lobes. A growing
    traveler is drawn the same way, with its whole trough ('whole_quarter', 'whole_zero') or only
    the part of the measured shape above its core ('lobe_zero') scaled by its growth factor at
    each sample's age."""
    if case["kind"] == "standing" or case["growth"] == "constant":
        return G.build(tracks, rng, {"kind": "travel"} if case["kind"] == "travel" else case, parts)
    t = J.times()[:, 0]; V = np.zeros((J.N_SAMPLES, len(XL))); centers = []
    for tr in tracks:
        n = tr["days"]; t0 = int(rng.integers(0, A.SEASON_DAYS - n + 1)); d = tr["injected"]["measured"]
        k0, k1 = t0 * L.OBS_PER_DAY, (t0 + n) * L.OBS_PER_DAY
        pos = np.full(J.N_SAMPLES, np.nan); pos[k0:k1] = tr["first_lon"] + tr["speed"] * (t[k0:k1] - t0)
        for k in range(k0, k1):
            f = growth_factor(case["growth"], t[k] - t0); full = H.shape("measured", XL, pos[k], d)
            if case["growth"] == "lobe_zero":
                core = P.core_shape(XL, pos[k], d, parts["core"])
                V[k] += core + f * (full - core)
            else:
                V[k] += f * full
        centers.append(pos.reshape(A.SEASON_DAYS, L.OBS_PER_DAY).mean(axis=1))
    return V, centers


# ------------------------------------------------------------------ the pre-start reader

def prestart(b0_by, hist_by, bands):
    """The other-season anomaly at each history's first longitude, averaged over PRE_DAYS before
    its first day, summed, over the summed magnitude of the anomaly there on the first day, for
    histories whose first day has PROFILE_DAYS of season before it, with the same ratio on each
    day of PROFILE_DAYS, the sums and the counts."""
    num = {k: 0.0 for k in PROFILE_DAYS}; den = 0.0; n = 0; edge = 0
    for (y, b), hs in hist_by.items():
        if b not in bands:
            continue
        rows = b0_by[(y, b)]
        for h in hs:
            d0, j = h["days"][0], P.col(h["lons"][0])
            if d0 + PROFILE_DAYS[0] < 0:
                edge += 1
                continue
            for k in PROFILE_DAYS:
                num[k] += float(rows[d0 + k][j])
            den += abs(float(rows[d0][j])); n += 1
    if n == 0 or den <= 0:
        raise SystemExit("REFUSED: no history enters the pre-start reader")
    pre = sum(num[k] for k in PRE_DAYS) / len(PRE_DAYS)
    return {"prestart": pre / den, "prestart_numerator": pre, "prestart_denominator": den, "prestart_histories": n, "season_edge": edge,
            "prestart_profile": [num[k] / den for k in PROFILE_DAYS]}


def measure(b1_by, b0_by, hist_by, bands):
    m = G.measure(b1_by, b0_by, hist_by, bands)
    m.update(prestart(b0_by, hist_by, bands))
    return m


# ------------------------------------------------------------------ constructions

def fields(case, bg_name, place, tracks, by, matched_std, parts, month_of):
    """The joint test's fields for one placement, with this script's constructions."""
    b1_by, b0_by, hist_by = {}, {}, {}
    for bi, band in enumerate(L.BAND_LAT):
        if band not in BANDS:
            continue
        bg = (J.matched_background(matched_std[band], place) if bg_name == "matched"
              else [J.hundredths(L.season_samples(by[y], band)) for y in YEARS])
        Hb = {}
        for i, y in enumerate(YEARS):
            if case is None:
                S = bg[i] / 100.0; centers = None
            else:
                rng = np.random.default_rng([P.LOBE_SEED, bi, place, i])
                V, centers = build(tracks.get(band, {}).get(y, []), rng, case, parts)
                S = J.inject(bg[i], V)
            hs, rows = H.detected(S, month_of)
            hist_by[(y, band)] = hs if centers is None else [h for h in hs if any(HI.follows(h, c) for c in centers)]
            b1_by[(y, band)] = rows
            Hb[y] = P.daily_hundredths(S)
        b0 = P.b0_rows(Hb, month_of)
        for y in YEARS:
            b0_by[(y, band)] = b0[y]
    return b1_by, b0_by, hist_by


def construction(case, bg_name, place, tracks, by, matched_std, parts, month_of):
    return measure(*fields(case, bg_name, place, tracks, by, matched_std, parts, month_of), BANDS)


def run_constructions(tracks, by, matched_std, parts):
    month_of = A.month_index()
    out = {bg: {case_name(c): [construction(c, bg, p, tracks, by, matched_std, parts, month_of) for p in PLACEMENTS] for c in cases()}
           for bg in ("archive", "matched")}
    out["noise"] = [construction(None, "matched", p, tracks, by, matched_std, parts, month_of) for p in PLACEMENTS]
    return out


def block_ranges(cons, bg, reader, block):
    idx = [PLACEMENTS.index(p) for p in BLOCKS[block]]
    return {k: (min(v[i][reader] for i in idx), max(v[i][reader] for i in idx)) for k, v in cons[bg].items()}


def noise_range(cons, reader, block):
    idx = [PLACEMENTS.index(p) for p in BLOCKS[block]]
    return (min(cons["noise"][i][reader] for i in idx), max(cons["noise"][i][reader] for i in idx))


def check_reproduced(cons, joint_cons):
    """In the joint test's placements, the constant traveler, every standing lobe the joint record
    holds and noise alone reproduce its readers exactly."""
    for bg in ("archive", "matched"):
        for name, v in cons[bg].items():
            if name not in joint_cons[bg]:
                continue
            for p in JOINT_PLACEMENTS:
                mine, theirs = v[PLACEMENTS.index(p)], joint_cons[bg][name][p]
                if {k: mine[k] for k in theirs} != theirs:
                    raise SystemExit(f"REFUSED: {bg} {name} placement {p} does not reproduce the joint record")
    for p in JOINT_PLACEMENTS:
        mine, theirs = cons["noise"][PLACEMENTS.index(p)], joint_cons["noise"][p]
        if {k: mine[k] for k in theirs} != theirs:
            raise SystemExit(f"REFUSED: noise alone placement {p} does not reproduce the joint record")


def apply_rule(cons):
    """Section 4: on each background and in each block, the largest pre-start reading of the four
    travelers lies below the smallest of each layout in DECIDING."""
    fails = []
    for bg in ("archive", "matched"):
        for block in BLOCKS:
            r = block_ranges(cons, bg, "prestart", block)
            top = max(r[case_name({"kind": "travel", "growth": g})][1] for g in GROWTHS)
            for k in DECIDING:
                if not top < r[k][0]:
                    fails.append(f"{bg} block {block}: {k} overlaps the travelers")
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
    return measure(b1_by, b0_by, dict(hist), BANDS)


def check_joint(m, joint_rec):
    """The archive's departure, contrast and first-day reading equal the joint record's."""
    jm = joint_rec["results"]["measure"]
    for k in ("departure", "contrast", "first_day", "samples", "histories"):
        if m[k] != jm[k]:
            raise SystemExit(f"REFUSED: the archive {k} differs from the joint record")


def classify(m, cons, bg="archive"):
    """For every case and block, whether each reader's archive value lies inside the block's range
    (bounds included) and the signed distance outside when not, and the case's verdict:
    'consistent' when all four lie inside in both blocks, 'excluded' when some reader lies outside
    in both, 'split' otherwise."""
    r = {(k, b): block_ranges(cons, bg, k, b) for k in READERS for b in BLOCKS}
    out = {}
    for name in cons[bg]:
        row = {}
        for b in BLOCKS:
            row[b] = {}
            for k in READERS:
                lo, hi = r[(k, b)][name]; v = m[k]
                row[b][k] = 0.0 if lo <= v <= hi else (v - hi if v > hi else v - lo)
        inside_both = all(row[b][k] == 0.0 for b in BLOCKS for k in READERS)
        outside_both = any(all(row[b][k] != 0.0 for b in BLOCKS) for k in READERS)
        row["verdict"] = "consistent" if inside_both else ("excluded" if outside_both else "split")
        out[name] = row
    return out


def summary(m, cons):
    cls = {bg: classify(m, cons, bg) for bg in ("archive", "matched")}
    travelers = [case_name({"kind": "travel", "growth": g}) for g in GROWTHS]
    a = cls["archive"]
    tv = {t: a[t]["verdict"] for t in travelers}; dv = {k: a[k]["verdict"] for k in DECIDING}
    if any(v == "consistent" for v in tv.values()) and all(v == "excluded" for v in dv.values()):
        decision = "standing excluded"
    elif all(v == "excluded" for v in tv.values()) and any(v == "consistent" for v in dv.values()):
        decision = "travelers excluded"
    else:
        decision = "unresolved"
    noise = {b: noise_range(cons, "prestart", b) for b in BLOCKS}
    standing = [n for n in cons["archive"] if n not in travelers]
    return {"classes": cls, "travelers": tv, "deciding": dv, "decision": decision, "noise_prestart": noise,
            "noise_contains": {b: noise[b][0] <= m["prestart"] <= noise[b][1] for b in BLOCKS},
            "standing_consistent": [n for n in standing if a[n]["verdict"] == "consistent"],
            "standing_split": [n for n in standing if a[n]["verdict"] == "split"], "standing_cases": len(standing)}


def readings(m, s):
    nz = s["noise_prestart"]
    out = [f"over 2 and 1 days before the first day of {m['prestart_histories']} westward histories in 15 to 24.5 N, the other-season anomaly at the "
           f"first longitude is {m['prestart']:.2f} of the magnitude there on the first day, and noise alone gives "
           f"{nz['A'][0]:.2f} to {nz['A'][1]:.2f} in block A and {nz['B'][0]:.2f} to {nz['B'][1]:.2f} in block B",
           "on the archive background the travelers are " + ", ".join(f"{k} {v}" for k, v in s["travelers"].items())
           + ", and the joint test's consistent standing lobes are " + ", ".join(f"{k} {v}" for k, v in s["deciding"].items()),
           {"standing excluded": "the readings exclude the four standing lobes that the joint test left, and at least one traveler stays consistent",
            "travelers excluded": "the readings exclude every traveler, and at least one of the joint test's standing lobes stays consistent",
            "unresolved": "the readings do not decide between the travelers and the joint test's standing lobes"}[s["decision"]],
           f"{len(s['standing_consistent'])} of the {s['standing_cases']} standing lobes are consistent in both blocks"
           + (f" ({', '.join(s['standing_consistent'])})" if s["standing_consistent"] else "")]
    return out


# ------------------------------------------------------------------ text and main

def fmt(v):
    return "none" if v is None else f"{v:.2f}"


def constructions_text(cons, verdict):
    out = ["# pre-start reading, the constructions (section 3)", "",
           "| background | case | pre-start block A (min, max) | block B (min, max) | first day A | first day B | histories per placement, A then B |",
           "| --- | --- | --- | --- | --- | --- | --- |"]
    for bg in ("archive", "matched"):
        pa, pb = block_ranges(cons, bg, "prestart", "A"), block_ranges(cons, bg, "prestart", "B")
        fa, fb = block_ranges(cons, bg, "first_day", "A"), block_ranges(cons, bg, "first_day", "B")
        for name, v in cons[bg].items():
            out.append(f"| {bg} | {name} | {pa[name][0]:.2f}, {pa[name][1]:.2f} | {pb[name][0]:.2f}, {pb[name][1]:.2f} | {fa[name][0]:.2f}, {fa[name][1]:.2f} | "
                       f"{fb[name][0]:.2f}, {fb[name][1]:.2f} | {', '.join(str(c['prestart_histories']) for c in v)} |")
    na, nb = noise_range(cons, "prestart", "A"), noise_range(cons, "prestart", "B")
    out.append(f"| matched | noise alone | {na[0]:.2f}, {na[1]:.2f} | {nb[0]:.2f}, {nb[1]:.2f} | | | {', '.join(str(c['prestart_histories']) for c in cons['noise'])} |")
    out += ["", f"adequate under the rule of section 4, {'yes' if verdict['adequate'] else 'no'}"] + [f"- {x}" for x in verdict["failures"]]
    return "\n".join(out) + "\n"


def archive_text(m, s):
    out = ["# pre-start reading, the archive (section 7)", "",
           "| histories | pre-start | numerator | denominator | season edge | profile, days -3 to 0 | departure | contrast | first day |",
           "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
           f"| {m['prestart_histories']} | {m['prestart']:.3f} | {m['prestart_numerator']:.3f} | {m['prestart_denominator']:.3f} | {m['season_edge']} | "
           f"{', '.join(fmt(v) for v in m['prestart_profile'])} | {m['departure']:.3f} | {m['contrast']:.3f} | {m['first_day']:.3f} |", "",
           "| background | case | verdict | pre-start outside by, A and B | departure, A and B | contrast, A and B | first day, A and B |",
           "| --- | --- | --- | --- | --- | --- | --- |"]
    travelers = list(s["travelers"])
    for bg in ("archive", "matched"):
        c = s["classes"][bg]
        for name in travelers + list(DECIDING) + [n for n in s["standing_consistent"] if n not in DECIDING]:
            row = c[name]
            out.append(f"| {bg} | {name} | {row['verdict']} | " + " | ".join(f"{row['A'][k]:.3f}, {row['B'][k]:.3f}" for k in ("prestart", "departure", "contrast", "first_day")) + " |")
    out += ["", f"decision on the archive background, {s['decision']}",
            "noise alone contains the archive's pre-start reading, " + ", ".join(f"block {b} {'yes' if v else 'no'}" for b, v in s["noise_contains"].items()),
            f"standing lobes split between the blocks, {', '.join(s['standing_split']) or 'none'}", ""]
    out += [f"- {x}" for x in readings(m, s)]
    return "\n".join(out) + "\n"


def load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("constructions", "archive"))
    ap.add_argument("--out-dir", required=True)
    for opt in ("--shape", "--inject", "--records", "--histories", "--xband", "--lagreg", "--joint-constructions", "--joint-archive", "--constructions"):
        ap.add_argument(opt, default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    if not all((a.shape, a.inject, a.records, a.histories, a.xband, a.joint_constructions)):
        raise SystemExit("REFUSED: --shape, --inject, --records, --histories, --xband and --joint-constructions are required")
    hist, tracks, by, matched_std, gen = C.load_inputs(a)
    joint_cons_rec = load_json(a.joint_constructions)
    if joint_cons_rec["generated_by"]["sha256"] != G.SCRIPT_SHA256_AT_LOAD:
        raise SystemExit("REFUSED: the joint constructions were made by a different version of its script")
    gen.update({"script": "scripts/eastern_prestart.py", "sha256": SCRIPT_SHA256_AT_LOAD, "declared": "EASTERN_PRESTART_2026-10-22.md",
                "joint_script_sha256": G.SCRIPT_SHA256_AT_LOAD, "lobe_script_sha256": P.SCRIPT_SHA256_AT_LOAD, "coherence_script_sha256": C.SCRIPT_SHA256_AT_LOAD,
                "joint_constructions": {"path": a.joint_constructions, "sha256": sha256_of(a.joint_constructions)}})
    parts = P.measured_parts()
    if a.mode == "constructions":
        pj, pm = (os.path.join(a.out_dir, f"eval60_prestart_constructions{a.tag}.{e}") for e in ("json", "md"))
        C.refuse_existing([pj, pm])
        cons = json.loads(json.dumps(run_constructions(tracks, by, matched_std, parts)))
        check_reproduced(cons, joint_cons_rec["results"])
        verdict = apply_rule(cons)
        with open(pj, "w", encoding="utf-8") as fh:
            json.dump({"generated_by": gen, "results": cons, "verdict": verdict}, fh)
        text = constructions_text(cons, verdict)
        with open(pm, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(text)
        return 0
    if not (a.lagreg and a.joint_archive and a.constructions):
        raise SystemExit("REFUSED: --lagreg, --joint-archive and --constructions are required for the archive")
    J.check_zero_injection(by, load_json(a.lagreg))
    joint_rec = load_json(a.joint_archive)
    if joint_rec["generated_by"]["sha256"] != G.SCRIPT_SHA256_AT_LOAD:
        raise SystemExit("REFUSED: the joint archive record was made by a different version of its script")
    cons = check_constructions(a.constructions, tracks, by, matched_std, parts)["results"]
    check_reproduced(cons, joint_cons_rec["results"])
    pj, pm = (os.path.join(a.out_dir, f"eval60_prestart_{k}{a.tag}.{e}") for k, e in (("archive", "json"), ("tables", "md")))
    C.refuse_existing([pj, pm])
    m = archive_measure(by, hist, A.month_index())
    check_joint(m, joint_rec)
    s = summary(m, cons)
    res = {"measure": m, "summary": s, "readings": readings(m, s)}
    gen.update({"lagreg": {"path": a.lagreg, "sha256": sha256_of(a.lagreg)}, "constructions": {"path": a.constructions, "sha256": sha256_of(a.constructions)},
                "joint_archive": {"path": a.joint_archive, "sha256": sha256_of(a.joint_archive)}})
    with open(pj, "w", encoding="utf-8") as fh:
        json.dump({"generated_by": gen, "results": res}, fh)
    text = archive_text(m, s)
    with open(pm, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
