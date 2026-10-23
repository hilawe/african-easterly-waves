#!/usr/bin/env python3
"""The westward histories' daily profile on a matched sample (`EASTERN_MATCHED_PROFILE_2026-10-23.md`):
the joint record's exploratory daily profile, the other-season anomaly 6 degrees east of each day's
member over the summed magnitude at the members on days 0 to 4, read once over every history in
15 to 24.5 N, whose sample changes from day to day, and once over the cohort of histories that
enter it on all five days, with each day's numerator and denominator kept so that the
history-days outside the cohort can be read from the difference. Both curves are read by the
joint record's own reader, and the curve over every history must equal the joint record. The
histories outside the cohort are also accumulated directly, and the run is refused unless the
changing sample's sums are the cohort's plus theirs. The measure describes pooled curves,
carries no noise reference, and identifies no feature.

    python3 scripts/eastern_matched_profile.py --shape PATH --inject PATH --records DIR --histories PATH --xband DIR \\
        --joint-archive PATH --out-dir DIR
"""

import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
import eastern_joint_standing as G  # noqa: E402

P, C, L, A = G.P, G.C, G.L, G.A
YEARS = G.YEARS
BANDS = G.BANDS
DAYS = G.PROFILE_DAYS
INPUTS = ("shape", "inject", "records", "histories", "xband")
ADDITIVE_TOL = 1e-9


def sha256_of(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


SCRIPT_SHA256_AT_LOAD = sha256_of(os.path.abspath(__file__))


def history_id(y, b, k):
    """The subset's identifier of the k-th history (from 1) of a season and band."""
    return f"{y}-{b.split(' ')[0]}N-{k:02d}"


def eligible_days(h):
    """The days of DAYS on which the joint reader reads the history: a member exists and its
    position 6 degrees east lies on the grid."""
    return [k for k in DAYS if k < len(h["lons"]) and h["lons"][k] + P.OFFSET <= A.LONS[-1]]


def matched(hist_by, bands):
    """The histories of the given bands the profile reads on every day of DAYS, by season and
    band, and their identifiers."""
    out, ids = {}, []
    for (y, b), hs in hist_by.items():
        keep = [(k, h) for k, h in enumerate(hs, 1) if b in bands and eligible_days(h) == list(DAYS)]
        out[(y, b)] = [h for _, h in keep]
        ids += [history_id(y, b, k) for k, _ in keep]
    return out, ids


def sums(b0_by, hist_by, bands):
    """Each day's numerator, denominator, count and off-grid count, accumulated in the joint
    reader's order."""
    num = [0.0] * len(DAYS); den = [0.0] * len(DAYS); n = [0] * len(DAYS); off = [0] * len(DAYS); hist_n = 0
    for (y, b), hs in hist_by.items():
        if b not in bands:
            continue
        rows = b0_by[(y, b)]
        for h in hs:
            hist_n += 1
            for i, k in enumerate(DAYS):
                if k >= len(h["lons"]):
                    continue
                xk, d = h["lons"][k], h["days"][k]
                if xk + P.OFFSET > A.LONS[-1]:
                    off[i] += 1
                    continue
                num[i] += float(rows[d][P.col(xk + P.OFFSET)]); den[i] += abs(float(rows[d][P.col(xk)])); n[i] += 1
    return {"numerator": num, "denominator": den, "days": n, "off_grid": off, "histories": hist_n}


def curve(b0_by, hist_by, bands):
    """The profile of a sample with its sums, refused unless the sums' ratios, counts and
    off-grid counts equal the joint reader's on the same sample."""
    s = sums(b0_by, hist_by, bands)
    pr = G.profile(b0_by, hist_by, bands)
    ratio = [nu / de if de > 0 else None for nu, de in zip(s["numerator"], s["denominator"])]
    if (ratio, s["days"], s["off_grid"]) != (pr["by_day"], pr["days"], pr["off_grid"]):
        raise SystemExit("REFUSED: the kept sums do not reproduce the joint reader's profile")
    return dict(s, by_day=ratio)


def complement(hist_by, cohort_by):
    """The histories of each season and band that are not in the cohort."""
    return {key: [h for h in hs if not any(h is c for c in cohort_by.get(key, []))] for key, hs in hist_by.items()}


def check_additive(changing, match, rest):
    """The changing sample's counts equal the cohort's plus those of the histories outside it,
    and its sums equal theirs to within summation order, or the run is refused. The joint
    reader returns ratios only, so this is what binds the kept sums of the two samples to each
    other; their absolute size is bound by the tests."""
    if changing["histories"] != match["histories"] + rest["histories"] or any(
            changing[k][i] != match[k][i] + rest[k][i] for k in ("days", "off_grid") for i in range(len(DAYS))):
        raise SystemExit("REFUSED: the changing sample's counts are not the cohort's plus those outside it")
    worst = 0.0
    for k in ("numerator", "denominator"):
        for i in range(len(DAYS)):
            gap = abs(changing[k][i] - match[k][i] - rest[k][i])
            if gap > ADDITIVE_TOL * max(1.0, abs(changing[k][i])):
                raise SystemExit(f"REFUSED: the changing sample's {k} on day {DAYS[i]} is not the cohort's plus the outside's")
            worst = max(worst, gap)
    return worst


def outside(changing, match):
    """The history-days of the changing sample outside the cohort, read from the difference of
    the kept sums, refused unless the cohort lies inside the changing sample on every day."""
    out = {"numerator": [], "denominator": [], "days": [], "by_day": []}
    for i in range(len(DAYS)):
        n = changing["days"][i] - match["days"][i]
        if n < 0:
            raise SystemExit("REFUSED: the cohort has more history-days than the changing sample")
        nu = changing["numerator"][i] - match["numerator"][i]; de = changing["denominator"][i] - match["denominator"][i]
        out["numerator"].append(nu); out["denominator"].append(de); out["days"].append(n)
        out["by_day"].append(nu / de if n > 0 and de > 0 else None)
    return out


def b0_rows(by):
    month_of = A.month_index(); out = {}
    for b in L.BAND_LAT:
        rows = P.b0_rows({y: P.daily_hundredths(L.season_samples(by[y], b)) for y in YEARS}, month_of)
        for y in YEARS:
            out[(y, b)] = rows[y]
    return out


def check_same_inputs(gen, joint_rec):
    """The inputs read now are the joint record's, file by file."""
    jg = joint_rec["generated_by"]
    for k in INPUTS:
        mine, theirs = gen[k], jg.get(k)
        if k in ("records", "xband"):
            same = mine == theirs
        else:
            same = isinstance(theirs, dict) and mine["sha256"] == theirs.get("sha256")
        if not same:
            raise SystemExit(f"REFUSED: the {k} input differs from the joint record's")


def check_joint(changing, joint_rec):
    """The curve over every history equals the joint record's profile."""
    jm = joint_rec["results"]["measure"]
    if (changing["by_day"], changing["days"], changing["off_grid"], changing["histories"]) != (
            jm["profile"], jm["profile_days"], jm["profile_off_grid"], jm["histories"]):
        raise SystemExit("REFUSED: the curve over every history differs from the joint record")


def measure(b0, hist):
    changing = curve(b0, hist, BANDS)
    cohort, ids = matched(hist, BANDS)
    match = curve(b0, cohort, BANDS)
    if match["histories"] != len(ids) or match["days"] != [len(ids)] * len(DAYS):
        raise SystemExit("REFUSED: a cohort history is not read on every day")
    gap = check_additive(changing, match, sums(b0, complement(hist, cohort), BANDS))
    return {"changing": changing, "matched": match, "outside": outside(changing, match), "cohort": ids, "additive_max_gap_ms": gap,
            "difference": [m - c if m is not None and c is not None else None for m, c in zip(match["by_day"], changing["by_day"])]}


def fmt(v, f=".3f"):
    return "none" if v is None else format(v, f)


def text(res):
    out = ["# matched daily profile, the archive", "",
           "| sample | histories | day | ratio | numerator (m/s) | denominator (m/s) | history-days | off the grid |",
           "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name in ("changing", "matched"):
        c = res[name]
        for i, k in enumerate(DAYS):
            out.append(f"| {name} | {c['histories']} | {k} | {fmt(c['by_day'][i])} | {fmt(c['numerator'][i])} | {fmt(c['denominator'][i])} | "
                       f"{c['days'][i]} | {c['off_grid'][i]} |")
    o = res["outside"]
    for i, k in enumerate(DAYS):
        out.append(f"| outside the cohort | | {k} | {fmt(o['by_day'][i])} | {fmt(o['numerator'][i])} | {fmt(o['denominator'][i])} | {o['days'][i]} | |")
    out += ["", "| day | matched minus changing |", "| --- | --- |"] + [f"| {k} | {fmt(res['difference'][i])} |" for i, k in enumerate(DAYS)]
    out += ["", f"The cohort is {len(res['cohort'])} histories: " + ", ".join(res["cohort"]) + "."]
    out += ["", f"Accumulated directly, the histories outside the cohort add to the cohort's sums to give the changing sample's within "
                f"{res['additive_max_gap_ms']:.1e} m/s."]
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for opt in ("--shape", "--inject", "--records", "--histories", "--xband", "--joint-archive", "--out-dir"):
        ap.add_argument(opt, required=True)
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    pj, pm = (os.path.join(a.out_dir, f"eval60_matched_profile{a.tag}.{e}") for e in ("json", "md"))
    C.refuse_existing([pj, pm])
    with open(a.joint_archive, encoding="utf-8") as fh:
        joint_rec = json.load(fh)
    if joint_rec["generated_by"]["sha256"] != G.SCRIPT_SHA256_AT_LOAD:
        raise SystemExit("REFUSED: the joint archive record was made by a different version of its script")
    hist, _, by, _, gen = C.load_inputs(a)
    check_same_inputs(gen, joint_rec)
    res = measure(b0_rows(by), hist)
    check_joint(res["changing"], joint_rec)
    gen.update({"script": "scripts/eastern_matched_profile.py", "sha256": SCRIPT_SHA256_AT_LOAD, "declared": "EASTERN_MATCHED_PROFILE_2026-10-23.md",
                "joint_script_sha256": G.SCRIPT_SHA256_AT_LOAD, "joint_archive": {"path": a.joint_archive, "sha256": sha256_of(a.joint_archive)}})
    with open(pj, "w", encoding="utf-8") as fh:
        json.dump({"generated_by": gen, "results": res}, fh)
    t = text(res)
    with open(pm, "w", encoding="utf-8") as fh:
        fh.write(t)
    print(t)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
