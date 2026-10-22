#!/usr/bin/env python3
"""The position of the westward histories' southerly lobe (`EASTERN_LOBE_POSITION_2026-10-22.md`):
whether the southerly anomaly east of a history's minimum moves with the minimum or stays where
it was when the history began. On days 2 and 3 of each history the anomaly 6 degrees east of the
minimum (where a carried lobe sits) is compared with the anomaly 6 degrees east of the history's
first longitude (where a pinned lobe stays), pooled over the histories and scaled by the
minimum's depth, read against constructions of carried and pinned lobes on the histories' tracks
detected by the same pipeline, on the archive and on matched noise. The anomaly is taken against
the same month of the other 23 seasons, so a standing structure is not cancelled by its own
month's mean. The measure says where the lobe sits on average. It identifies no feature and does
not say what any one history is.

    python3 scripts/eastern_lobe_position.py constructions --shape PATH --inject PATH --records DIR --histories PATH --xband DIR --out-dir DIR
    python3 scripts/eastern_lobe_position.py archive --shape PATH --inject PATH --records DIR --histories PATH --xband DIR \\
        --lagreg PATH --constructions PATH --out-dir DIR
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
import eastern_history_coherence as C  # noqa: E402

H, J, L, A, HI = C.H, C.J, C.L, C.A, C.HI
YEARS = C.YEARS
XL = C.XL
BANDS = C.BANDS
KINDS = ("travel", "pinned", "pinned_relax")
OFFSET = 6.0
CP_DAYS = (2, 3)
LOBE_SEED = 2026102203
RELAX_BEFORE_DAYS = 10
MARGIN_DAYS = 2


def sha256_of(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


SCRIPT_SHA256_AT_LOAD = sha256_of(os.path.abspath(__file__))


# ------------------------------------------------------------------ the constructions' fields

def measured_parts():
    """The measured average's center magnitude, its negative part (the core) and its lobe
    amplitude, the largest value from 2 to 8 degrees east over the center's magnitude."""
    P = np.asarray(H.MEASURED["profile"], float)
    c = abs(P[H.DX == 0][0])
    east = (H.DX >= 2.0) & (H.DX <= 8.0)
    return {"core": np.minimum(P, 0.0) / c, "lobe": float(P[east].max()) / c}


def core_shape(x, center, d, core):
    return d * np.interp(np.asarray(x, float) - center, H.DX, core, left=0.0, right=0.0)


def build(tracks, rng, kind, parts):
    """On the tracks, drawn as the coherence test draws them (one whole-day start per track in
    order): 'travel', troughs of the measured shape, whose lobe moves with them; 'pinned', the
    measured shape's core moving on the track with a Gaussian southerly lobe of the shape's lobe
    amplitude standing 6 degrees east of the first longitude from MARGIN_DAYS before the start to
    MARGIN_DAYS after the end; 'pinned_relax', the same with the standing lobe at twice that
    amplitude for RELAX_BEFORE_DAYS before the start. Returns the field and each trough's
    daily-mean position."""
    t = J.times()[:, 0]; V = np.zeros((J.N_SAMPLES, len(XL))); centers = []
    for tr in tracks:
        n = tr["days"]; t0 = int(rng.integers(0, A.SEASON_DAYS - n + 1)); d = tr["injected"]["measured"]
        k0, k1 = t0 * L.OBS_PER_DAY, (t0 + n) * L.OBS_PER_DAY
        pos = np.full(J.N_SAMPLES, np.nan); pos[k0:k1] = tr["first_lon"] + tr["speed"] * (t[k0:k1] - t0)
        if kind == "travel":
            V[k0:k1] += np.array([H.shape("measured", XL, x, d) for x in pos[k0:k1]])
        elif kind in ("pinned", "pinned_relax"):
            V[k0:k1] += np.array([core_shape(XL, x, d, parts["core"]) for x in pos[k0:k1]])
            lobe = -A.gauss(XL, tr["first_lon"] + OFFSET, parts["lobe"] * d, 2.0)
            b = min(J.N_SAMPLES, (t0 + n + MARGIN_DAYS) * L.OBS_PER_DAY)
            if kind == "pinned":
                V[max(0, (t0 - MARGIN_DAYS) * L.OBS_PER_DAY):b] += lobe
            else:
                V[max(0, (t0 - RELAX_BEFORE_DAYS) * L.OBS_PER_DAY):k0] += 2.0 * lobe
                V[k0:b] += lobe
        else:
            raise SystemExit(f"REFUSED: unknown construction {kind}")
        centers.append(pos.reshape(A.SEASON_DAYS, L.OBS_PER_DAY).mean(axis=1))
    return V, centers


# ------------------------------------------------------------------ the anomaly and the measure

def daily_hundredths(S):
    return np.asarray(A.hundredths_from_samples(np.asarray(S)[:, H.J0:H.J0 + H.NA]), dtype=np.int64)


def b0_rows(H_by_year, month_of):
    """Per season, the anomaly of the daily means against the same month's mean over the other
    seasons, in m/s, on the exact integer path."""
    sums = {y: {m: H_by_year[y][month_of == m].sum(axis=0) for m in L.MONTHS} for y in H_by_year}
    tot = {m: sum(sums[y][m] for y in H_by_year) for m in L.MONTHS}
    out = {}
    for y, Hy in H_by_year.items():
        rows, sc = A.exact_anomaly("B0", Hy, month_of, {m: (tot[m] - sums[y][m], len(H_by_year) - 1) for m in L.MONTHS})
        out[y] = [np.asarray(r, float) / s for r, s in zip(rows, sc)]
    return out


def col(x):
    return int(np.flatnonzero(np.isclose(A.LONS, x))[0])


def accumulate(rows, h, acc):
    """For each day of CP_DAYS within the history, the anomaly OFFSET degrees east of the member
    (carried) and OFFSET degrees east of the first longitude (pinned), and the magnitude at the
    member, skipped when either eastern position lies off the grid."""
    x0 = h["lons"][0]
    for k in CP_DAYS:
        if k >= len(h["lons"]):
            continue
        xk, d = h["lons"][k], h["days"][k]
        if x0 + OFFSET > A.LONS[-1] or xk + OFFSET > A.LONS[-1]:
            acc["off_grid"] += 1
            continue
        acc["carried"] += float(rows[d][col(xk + OFFSET)]); acc["pinned"] += float(rows[d][col(x0 + OFFSET)])
        acc["center"] += abs(float(rows[d][col(xk)])); acc["samples"] += 1


def new_acc():
    return {"carried": 0.0, "pinned": 0.0, "center": 0.0, "samples": 0, "off_grid": 0}


def reader(acc):
    """The carried and pinned sums over the summed magnitude at the members, and their
    difference, the contrast."""
    if acc["samples"] == 0 or acc["center"] <= 0:
        raise SystemExit("REFUSED: no history-day enters the measure")
    return {"samples": acc["samples"], "off_grid": acc["off_grid"], "carried": acc["carried"] / acc["center"],
            "pinned": acc["pinned"] / acc["center"], "contrast": (acc["carried"] - acc["pinned"]) / acc["center"]}


def measure(rows_by_year, hist_by, bands):
    acc = new_acc(); hist_n = 0
    for (y, b), hs in hist_by.items():
        if b not in bands:
            continue
        for h in hs:
            hist_n += 1
            accumulate(rows_by_year[b][y], h, acc)
    r = reader(acc); r["histories"] = hist_n
    return r


# ------------------------------------------------------------------ constructions

def construction(kind, bg_name, place, tracks, by, matched_std, parts, month_of):
    """One placement: the detected histories that follow an injected trough, measured on the
    other-season anomaly of the constructed field, or every detected history for noise alone."""
    rows_by, hist_by = {}, {}
    for bi, band in enumerate(L.BAND_LAT):
        if band not in BANDS:
            continue
        bg = (J.matched_background(matched_std[band], place) if bg_name == "matched"
              else [J.hundredths(L.season_samples(by[y], band)) for y in YEARS])
        Hb = {}
        for i, y in enumerate(YEARS):
            if kind == "none":
                S = bg[i] / 100.0; centers = None
            else:
                rng = np.random.default_rng([LOBE_SEED, bi, place, i])
                V, centers = build(tracks.get(band, {}).get(y, []), rng, kind, parts)
                S = J.inject(bg[i], V)
            hs = H.detected(S, month_of)[0]
            hist_by[(y, band)] = hs if centers is None else [h for h in hs if any(HI.follows(h, c) for c in centers)]
            Hb[y] = daily_hundredths(S)
        rows_by[band] = b0_rows(Hb, month_of)
    return measure(rows_by, hist_by, BANDS)


def run_constructions(tracks, by, matched_std, parts):
    month_of = A.month_index()
    out = {bg: {k: [construction(k, bg, p, tracks, by, matched_std, parts, month_of) for p in range(J.PLACEMENTS)] for k in KINDS}
           for bg in ("archive", "matched")}
    out["noise"] = [construction("none", "matched", p, tracks, by, matched_std, parts, month_of) for p in range(J.PLACEMENTS)]
    return out


def ranges(cons, bg):
    return {k: (min(c["contrast"] for c in v), max(c["contrast"] for c in v), float(np.mean([c["contrast"] for c in v]))) for k, v in cons[bg].items()}


def apply_rule(cons):
    """Section 4: on each background, the largest contrast of the pinned and of the relaxing pinned
    constructions lies below the smallest of the travelers."""
    fails = []
    for bg in ("archive", "matched"):
        r = ranges(cons, bg)
        for k in ("pinned", "pinned_relax"):
            if not r[k][1] < r["travel"][0]:
                fails.append(f"{bg}: {k} overlaps travel")
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
    rows_by = {b: b0_rows({y: daily_hundredths(L.season_samples(by[y], b)) for y in YEARS}, month_of) for b in L.BAND_LAT}
    return {"primary": measure(rows_by, hist, BANDS), "all_bands": measure(rows_by, hist, tuple(L.BAND_LAT))}


def position(value, cons):
    """The constructions, on each background, whose placement range contains the value, and the
    noise-alone range."""
    inside = {bg: [k for k, (lo, hi, _) in ranges(cons, bg).items() if lo <= value <= hi] for bg in ("archive", "matched")}
    noise = [c["contrast"] for c in cons["noise"]]
    return {"inside": inside, "noise_range": [min(noise), max(noise)], "means": {bg: {k: v[2] for k, v in ranges(cons, bg).items()} for bg in ("archive", "matched")}}


def readings(res):
    a = res["measure"]["primary"]; p = res["position"]
    ins = p["inside"]["archive"]
    s = (f"on days 2 and 3 of the {a['histories']} westward histories in 15 to 24.5 N ({a['samples']} history-days), the other-season anomaly "
         f"6 degrees east of the minimum is {a['carried']:.2f} and 6 degrees east of the first longitude {a['pinned']:.2f} of the minimum's depth, "
         f"a contrast of {a['contrast']:.2f}, "
         + (f"inside the archive-background range of the {', '.join(ins)} construction" if ins else "inside no archive-background construction's range")
         + f", and noise alone gives {p['noise_range'][0]:.2f} to {p['noise_range'][1]:.2f}")
    return [s]


# ------------------------------------------------------------------ text and main

def constructions_text(cons, verdict):
    out = ["# lobe position, the constructions (section 3)", "",
           "| background | construction | contrast (min, max, mean) | carried, pinned (means) | history-days per placement |", "| --- | --- | --- | --- | --- |"]
    for bg in ("archive", "matched"):
        r = ranges(cons, bg)
        for k in KINDS:
            v = cons[bg][k]
            out.append(f"| {bg} | {k} | {r[k][0]:.2f}, {r[k][1]:.2f}, {r[k][2]:.2f} | {np.mean([c['carried'] for c in v]):.2f}, {np.mean([c['pinned'] for c in v]):.2f} | "
                       f"{', '.join(str(c['samples']) for c in v)} |")
    nc = [c["contrast"] for c in cons["noise"]]
    out.append(f"| matched | noise alone | {min(nc):.2f}, {max(nc):.2f}, {np.mean(nc):.2f} | {np.mean([c['carried'] for c in cons['noise']]):.2f}, "
               f"{np.mean([c['pinned'] for c in cons['noise']]):.2f} | {', '.join(str(c['samples']) for c in cons['noise'])} |")
    out += ["", f"adequate under the rule of section 4, {'yes' if verdict['adequate'] else 'no'}"] + [f"- {x}" for x in verdict["failures"]]
    return "\n".join(out) + "\n"


def archive_text(res):
    out = ["# lobe position, the archive (section 7)", "",
           "| histories | count | history-days | off the grid | carried | pinned | contrast |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for k in ("primary", "all_bands"):
        a = res["measure"][k]
        out.append(f"| {k} | {a['histories']} | {a['samples']} | {a['off_grid']} | {a['carried']:.3f} | {a['pinned']:.3f} | {a['contrast']:.3f} |")
    p = res["position"]
    out += ["", "| background | constructions whose range contains the contrast | construction means |", "| --- | --- | --- |"]
    for bg in ("archive", "matched"):
        out.append(f"| {bg} | {', '.join(p['inside'][bg]) or 'none'} | {', '.join(f'{k}: {m:.2f}' for k, m in p['means'][bg].items())} |")
    out += ["", f"noise alone on the matched background, {p['noise_range'][0]:.2f} to {p['noise_range'][1]:.2f}", ""] + [f"- {x}" for x in res["readings"]]
    return "\n".join(out) + "\n"


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
    hist, tracks, by, matched_std, gen = C.load_inputs(a)
    gen.update({"script": "scripts/eastern_lobe_position.py", "sha256": SCRIPT_SHA256_AT_LOAD, "declared": "EASTERN_LOBE_POSITION_2026-10-22.md",
                "coherence_script_sha256": C.SCRIPT_SHA256_AT_LOAD})
    parts = measured_parts()
    if a.mode == "constructions":
        pj, pm = (os.path.join(a.out_dir, f"eval60_lobe_constructions{a.tag}.{e}") for e in ("json", "md"))
        C.refuse_existing([pj, pm])
        cons = json.loads(json.dumps(run_constructions(tracks, by, matched_std, parts))); verdict = apply_rule(cons)
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
    cons = check_constructions(a.constructions, tracks, by, matched_std, parts)["results"]
    pj, pm = (os.path.join(a.out_dir, f"eval60_lobe_{k}{a.tag}.{e}") for k, e in (("archive", "json"), ("tables", "md")))
    C.refuse_existing([pj, pm])
    res = {"measure": archive_measure(by, hist, A.month_index()), "lobe_amplitude": parts["lobe"]}
    res["position"] = position(res["measure"]["primary"]["contrast"], cons)
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
