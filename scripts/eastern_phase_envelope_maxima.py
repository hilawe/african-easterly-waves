#!/usr/bin/env python3
"""The drift of regressed wind maxima at the published bases of Diaz and Aiyyer (2013)
(`EASTERN_PHASE_ENVELOPE_MAXIMA_2026-10-27.md`). Real-valued lag regressions of the 8 to 15 N
band-mean meridional wind on the 45 W to 35 E grid, at 850 hPa with the base at 15 W and at
700 hPa with the base at 30 W on the 17 shared seasons, are read by a declared rule that
follows each trough and ridge across lags and takes its peak, with lags to 4 and to 7 days
and with the band-mean and the 10 N point references. An alternative rule takes the strongest
feature at each lag. Before any ERA5 map is read, the rule must pass an adequacy rule on six
of the phase and envelope effort's constructions, moved to the published bases. Development
change 1 compares values along each chain after scaling every lag by its largest absolute
value, because the fading of the map with lag pulls each feature's peak toward lag 0 and gives
a drift toward the east where the envelope stands or moves west (section 8). Changes 2 and 3
treat the ties that scaling creates (section 12). The rule fails the adequacy rule after the
three changes the declaration allows. Under the amendment of section 13 the reading path
reads the ERA5 maps with reader A, the rule as declared, and reader B, change 1 with a
two-peak minimum, and refuses unless reader B passes the adequacy rule. The rule as declared
and after each change are kept as construction readers so that the record carries the
development. The frozen
centroid readings of the same maps are reproduced from their stored records and set beside
the maxima readings. The readings say where the regressed maxima lie at each lag and how fast
that position drifts, not why.

    python3 scripts/eastern_phase_envelope_maxima.py constructions --out-dir DIR
    python3 scripts/eastern_phase_envelope_maxima.py read --constructions PATH --earlier PATH --matched PATH --reference PATH --out-dir DIR
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eastern_anomaly_minima as A  # noqa: E402
import eastern_lag_regression as LR  # noqa: E402
import eastern_minima_histories as HI  # noqa: E402
import eastern_phase_envelope as PE  # noqa: E402
import eastern_phase_envelope_matched as M  # noqa: E402
import eastern_phase_envelope_probes as PR  # noqa: E402

SCRIPT_SHA256_AT_LOAD = PE.sha256_of(os.path.abspath(__file__))
MAX_LAG_DAYS = 7
LAGS = np.arange(-MAX_LAG_DAYS * PE.OBS_PER_DAY, MAX_LAG_DAYS * PE.OBS_PER_DAY + 1)
RANGES = (4, 7)                      # lags to 4 days (the centroid's) and to 7 (the paper's)
THRESH = 0.3                         # m/s, the paper's first contour
LINK_DEG = 5.0                       # a chain continues within this distance per six-hourly step
MIN_COUNT = 3
BAND_REF, POINT_REF = "band mean, 8 to 15 N", "point, 10 N"
REFERENCES = (BAND_REF, POINT_REF)
RULES = ("maxima", "strongest")
CASES = ("ND7", "D07", "DE7", "U1", "U2", "N")
CONSTRUCTED = {"ND7": "westward", "D07": "standing", "DE7": "eastward", "U1": "standing", "U2": "eastward", "N": "standing"}
N_SEASONS = len(M.SHARED_YEARS)
PUBLISHED = {lv: M.published_deg_per_day(v["maxima_ms"]) for lv, v in M.PUBLISHED.items()}


# ------------------------------------------------------------------ the maps

def block(S, ref):
    """The season's field and reference band-passed together by the lag regression's filter,
    cut to the finite block and trimmed by PE.TRIM samples at each end (the samples the frozen
    analytic signal keeps), as (field, reference)."""
    F = LR.filtered(np.column_stack([np.asarray(S, float), np.asarray(ref, float)]))
    e0, e1 = LR.valid_block(F)
    B = F[e0:e1][PE.TRIM:e1 - e0 - PE.TRIM]
    if B.shape[0] <= 2 * len(LAGS):
        raise SystemExit("REFUSED: a season's block is too short for the lags")
    return B[:, :-1], B[:, -1]


def real_sums(X, r):
    """The sums one season contributes at each lag, over the times t with t and t + lag both
    in the block: count, sum and sum of squares of the reference, sum of the field at every
    longitude, and the sum of field(t + lag) times reference(t)."""
    T, nx = X.shape
    nl = len(LAGS)
    n = np.zeros(nl); sa = np.zeros(nl); saa = np.zeros(nl); sb = np.zeros((nl, nx)); sab = np.zeros((nl, nx))
    for li, L in enumerate(LAGS):
        L = int(L)
        t0, t1 = max(0, -L), min(T, T - L)
        a = r[t0:t1]; B = X[t0 + L:t1 + L]
        n[li] = t1 - t0; sa[li] = a.sum(); saa[li] = float((a * a).sum())
        sb[li] = B.sum(axis=0); sab[li] = (B * a[:, None]).sum(axis=0)
    return {"n": n, "sa": sa, "saa": saa, "sb": sb, "sab": sab}


def regression(st, w=None):
    """The pooled map R (lags by longitudes): the covariance about the pooled means divided by
    the reference's standard deviation over the same pairs, from stacked per-season sums
    weighted by `w` (season counts of a resample, ones by default)."""
    w = np.ones(st["n"].shape[0]) if w is None else np.asarray(w, float)
    n = w @ st["n"]; sa = w @ st["sa"]; saa = w @ st["saa"]
    sb = np.tensordot(w, st["sb"], axes=1); sab = np.tensordot(w, st["sab"], axes=1)
    ma = sa / n; mb = sb / n[:, None]
    C = sab / n[:, None] - mb * ma[:, None]
    va = saa / n - ma ** 2
    return C / np.sqrt(va)[:, None]


def season_stack(seasons, refs):
    return PE.stack([real_sums(*block(S, r)) for S, r in zip(seasons, refs)])


# ------------------------------------------------------------------ the two rules

def row_extrema(row, sign, thresh=THRESH):
    """Interior columns where sign times the row exceeds both neighbors and reaches `thresh`."""
    v = sign * np.asarray(row, float)
    return np.flatnonzero((v[1:-1] > v[:-2]) & (v[1:-1] > v[2:]) & (v[1:-1] >= thresh)) + 1


def chains(R, lons, sign, link_deg=LINK_DEG, thresh=THRESH):
    """Section 2, step 2. Extrema of one sign linked from each lag to the next, each one
    continuing the chain whose last point lies at the previous lag within `link_deg`, nearest
    pairs first and one to one; an extremum that continues no chain starts one. Chains are
    lists of (lag index, column)."""
    done, live = [], []
    for li in range(R.shape[0]):
        ext = [int(j) for j in row_extrema(R[li], sign, thresh)]
        pairs = sorted((abs(lons[j] - lons[c[-1][1]]), ci, j) for ci, c in enumerate(live) for j in ext
                       if abs(lons[j] - lons[c[-1][1]]) <= link_deg + 1e-9)
        used_c, used_j = set(), set()
        for _, ci, j in pairs:
            if ci in used_c or j in used_j:
                continue
            live[ci].append((li, j)); used_c.add(ci); used_j.add(j)
        done += [c for ci, c in enumerate(live) if ci not in used_c]
        live = [c for ci, c in enumerate(live) if ci in used_c] + [[(li, j)] for j in ext if j not in used_j]
    return done + live


def chain_maxima(R, lons, lags, version=3):
    """Section 2, step 3. The peak of every chain whose largest value lies inside it (which
    needs at least three points, so no separate length check is made), as sorted (lag in
    days, longitude, value), in the version of the rule given (section 12). Version 0 is the
    rule as declared. From version 1 on, the value compared along a chain is R over the largest
    absolute R at its lag, so that the fading of the map with lag does not pull each peak
    toward lag 0 (development change 1). That makes a chain's value exactly 1 at every lag
    where it holds the row's largest value, and versions 0 and 1 take the first of such tied
    points. Version 2 takes the middle one, the lower of two (development change 2), and
    version 3 also refuses a tied run that includes the chain's first or last point, since
    its middle may then lie outside the chain (development change 3)."""
    scaled = version >= 1
    rowmax = np.abs(R).max(axis=1)
    out = []
    for sign in (1, -1):
        for c in chains(R, lons, sign):
            v = np.array([sign * R[li, j] / (rowmax[li] if scaled else 1.0) for li, j in c])
            top = np.flatnonzero(v == v.max())
            if version >= 3 and (top[0] == 0 or top[-1] == len(c) - 1):
                continue
            k = int(top[(len(top) - 1) // 2]) if version >= 2 else int(top[0])
            if 0 < k < len(c) - 1:
                li, j = c[k]
                out.append((float(lags[li] * PE.DT_DAYS), float(lons[j]), float(R[li, j])))
    return sorted(out)


def strongest_per_lag(R, lons, lags):
    """The alternative rule's points: at each lag the column of largest absolute value, kept
    when it reaches THRESH and is not an end column, as (lag in days, longitude, value)."""
    out = []
    for li in range(R.shape[0]):
        j = int(np.argmax(np.abs(R[li])))
        if 0 < j < len(lons) - 1 and abs(R[li, j]) >= THRESH:
            out.append((float(lags[li] * PE.DT_DAYS), float(lons[j]), float(R[li, j])))
    return out


def drift(points, min_count=MIN_COUNT):
    """Section 2, step 4. The least-squares slope of longitude against lag in days, defined for
    at least `min_count` points at no fewer than two distinct lags."""
    if len(points) < min_count or len({p[0] for p in points}) < 2:
        return None
    return float(np.polyfit([p[0] for p in points], [p[1] for p in points], 1)[0])


POINTS = {"maxima": chain_maxima, "strongest": strongest_per_lag,
          "maxima as declared": lambda R, lons, lags: chain_maxima(R, lons, lags, version=0),
          "maxima after change 1": lambda R, lons, lags: chain_maxima(R, lons, lags, version=1),
          "maxima after change 2": lambda R, lons, lags: chain_maxima(R, lons, lags, version=2)}
EARLIER = ("maxima as declared", "maxima after change 1", "maxima after change 2")
READER_A, READER_B = "maxima as declared", "maxima after change 1, two peaks"      # section 13
POINTS[READER_B] = POINTS["maxima after change 1"]
MIN_COUNTS = {READER_B: 2}
READ_RULES = (READER_A, READER_B)
CONSTRUCTION_RULES = RULES + EARLIER + (READER_B,)


def read_map(R, lons, rule, days):
    """One rule's reading of one map within lags to `days`."""
    sel = np.abs(LAGS) <= days * PE.OBS_PER_DAY
    pts = POINTS[rule](R[sel], lons, LAGS[sel])
    return {"drift": drift(pts, MIN_COUNTS.get(rule, MIN_COUNT)), "points": [list(p) for p in pts],
            "pos_lag_days": max(p[0] for p in pts) if pts else 0.0}


def read_all(st, lons, W, rules=RULES):
    """Each rule at both lag ranges, with its interval from the resamples W and the effort's
    class, the largest counted lag standing for the followed positive lag."""
    R = regression(st)
    out = {rule: {f"{d}": read_map(R, lons, rule, d) for d in RANGES} for rule in rules}
    boot = {rule: {f"{d}": [] for d in RANGES} for rule in rules}
    for w in W:
        Rb = regression(st, w)
        for rule in rules:
            for d in RANGES:
                boot[rule][f"{d}"].append(read_map(Rb, lons, rule, d)["drift"])
    for rule in rules:
        for d in RANGES:
            r = out[rule][f"{d}"]
            lo, hi, und = PE.interval(boot[rule][f"{d}"])
            r.update({"interval": [lo, hi], "undefined": und,
                      "class": PE.classify(r["drift"], lo, hi, und, r["pos_lag_days"])})
    return out


def centroid_summary(r):
    keep = ("envelope_class", "envelope_speed", "envelope_interval", "envelope_undefined", "phase_class", "phase_speed", "lags", "positions")
    return {k: r[k] for k in keep}


# ------------------------------------------------------------------ constructions (section 3)

def noise(i, lons=M.MLONS):
    return A.noise_field(LR.CONSTRUCTION_NOISE_STD, seed=HI.NOISE_SEEDS[i], n_steps=A.SEASON_DAYS * PE.OBS_PER_DAY, lons=lons)


def construction_signal(case, i, base, amp, lons=M.MLONS):
    """The effort's signal for one season, evaluated on a longitude shifted so that its birth
    range (or U1's region) is centered on `base`; zero for N."""
    if case == "N":
        return np.zeros((A.SEASON_DAYS * PE.OBS_PER_DAY, len(lons)))
    p = PE.CASES[case]
    center = PE.U1["center"] if p["kind"] == "fixed" else 0.5 * sum(PE.CENTER_RANGE)
    return PE.signal(case, i, amp[case], lons=lons + (center - base))


def construction_season(case, i, base, amp, with_noise=True, lons=M.MLONS):
    V = construction_signal(case, i, base, amp, lons)
    return np.round(V + noise(i, lons) if with_noise else V, 2)


def construction_results(case, base, amp, n_boot=PE.N_BOOT, lons=M.MLONS):
    col = int(np.flatnonzero(np.isclose(lons, base))[0])
    W = PE.boot_weights(N_SEASONS, n_boot)
    seasons = [construction_season(case, i, base, amp) for i in range(N_SEASONS)]
    out = {"readers": read_all(season_stack(seasons, [S[:, col] for S in seasons]), lons, W, CONSTRUCTION_RULES),
           "centroid": centroid_summary(M.results(seasons, (base,), lons, n_boot)[f"{base:g}"])}
    if case != "N":
        clean = [construction_season(case, i, base, amp, with_noise=False) for i in range(N_SEASONS)]
        R = regression(season_stack(clean, [S[:, col] for S in clean]))
        Zs = [PE.analytic(S) for S in clean]
        C, E, N = PE.maps(PE.stack([PE.season_sums(Z, col) for Z in Zs]))
        out["signal_alone"] = {"readers": {rule: {f"{d}": read_map(R, lons, rule, d)["drift"] for d in RANGES} for rule in CONSTRUCTION_RULES},
                               "centroid": PE.readings(C, N, col, None, lons)["envelope_speed"]}
    return out


def run_constructions(n_boot=PE.N_BOOT, cases=CASES):
    cal = PE.amplitudes()
    out = {c: {f"{b:g}": construction_results(c, b, cal["amplitude"], n_boot) for _, b in M.PRIMARY} for c in cases}
    return {"calibration": cal, "cases": out}


def adequacy(results, rule="maxima"):
    """Section 3's rule for one reader: DE7 eastward at both bases and lag ranges, ND7 westward
    at 15 W at both, and no resolved class other than the constructed one anywhere."""
    fails = []
    cases = results["cases"]
    for d in RANGES:
        for _, b in M.PRIMARY:
            if cases["DE7"][f"{b:g}"]["readers"][rule][f"{d}"]["class"] != "eastward":
                fails.append(f"DE7 {b:g} E lags to {d} days not eastward")
        if cases["ND7"]["-15"]["readers"][rule][f"{d}"]["class"] != "westward":
            fails.append(f"ND7 -15 E lags to {d} days not westward")
    for c, by in cases.items():
        for b, v in by.items():
            for d in RANGES:
                k = v["readers"][rule][f"{d}"]["class"]
                if k not in (CONSTRUCTED[c], "unresolved"):
                    fails.append(f"{c} {b} E lags to {d} days reads {k}, constructed {CONSTRUCTED[c]}")
    return {"adequate": not fails, "failures": fails}


def coverage(cases, readers, what):
    """Refuse a constructions record that lacks a declared case, a published base, a lag range
    of one of `readers`, or reader A's signal-alone reading where `construction_results` makes
    one (every case but N), so that gate 3 scores all of section 3's evidence and not only the
    part a record still holds."""
    bases, ranges = {f"{b:g}" for _, b in M.PRIMARY}, {f"{d}" for d in RANGES}
    if set(cases) != set(CASES):
        raise SystemExit(f"REFUSED: the {what} holds the cases {sorted(cases)}, not the declared {sorted(CASES)}")
    for c in CASES:
        if set(cases[c]) != bases:
            raise SystemExit(f"REFUSED: the {what} holds {c} at the bases {sorted(cases[c])}, not {sorted(bases)}")
        for b in sorted(bases):
            v = cases[c][b]
            if ("signal_alone" in v) != (c != "N"):
                raise SystemExit(f"REFUSED: the {what} has the wrong signal-alone readings for {c} at {b} E")
            for rule in readers:
                if set(v["readers"].get(rule, {})) != ranges:
                    raise SystemExit(f"REFUSED: the {what} lacks {rule} at both lag ranges for {c} at {b} E")
            if c != "N" and set(v["signal_alone"].get("readers", {}).get(READER_A, {})) != ranges:
                raise SystemExit(f"REFUSED: the {what} lacks reader A's signal-alone readings for {c} at {b} E")


def check_constructions(path, earlier):
    """Gate 3 of section 13. The constructions record was written by this script and holds every
    declared case, base and lag range, its stored verdicts agree with the adequacy rule applied
    to its readings, reader A's readings equal those of the earlier record `earlier` over the
    whole declared set, and reader B passes the adequacy rule."""
    with open(path, encoding="utf-8") as fh:
        rec = json.load(fh)
    with open(earlier, encoding="utf-8") as fh:
        old = json.load(fh)["results"]["cases"]
    if rec["generated_by"]["sha256"] != SCRIPT_SHA256_AT_LOAD:
        raise SystemExit("REFUSED: the constructions were run by a different version of this script")
    coverage(rec["results"]["cases"], ("maxima", READER_A, READER_B), "constructions record")
    coverage(old, (READER_A,), "earlier record")
    if adequacy(rec["results"]) != rec.get("verdict") or adequacy(rec["results"], READER_B) != rec.get("amendment_verdict"):
        raise SystemExit("REFUSED: a stored verdict disagrees with the rule applied to the stored readings")
    for c in CASES:
        for b in sorted(old[c]):
            v, was = rec["results"]["cases"][c][b], old[c][b]
            same = v["readers"][READER_A] == was["readers"][READER_A]
            if c != "N":
                same = same and v["signal_alone"]["readers"][READER_A] == was["signal_alone"]["readers"][READER_A]
            if not same:
                raise SystemExit(f"REFUSED: reader A's readings of {c} at {b} E differ from the earlier record")
    if not rec["amendment_verdict"]["adequate"]:
        raise SystemExit(f"REFUSED: reader B is not adequate on the constructions ({rec['amendment_verdict']['failures']})")
    return rec


# ------------------------------------------------------------------ the ERA5 reading

def era5_inputs(level, base):
    seasons = [M.band_means(y, level, M.MATCHED_BAND) for y in M.SHARED_YEARS]
    col = int(np.flatnonzero(np.isclose(M.MLONS, base))[0])
    return seasons, {BAND_REF: [S[:, col] for S in seasons], POINT_REF: [PR.point_series(y, level, base) for y in M.SHARED_YEARS]}


def check_centroid(level, base, seasons, refs, matched, reference, n_boot=PE.N_BOOT):
    """Gate 2. The centroid readings of the same maps reproduce the stored ones."""
    got = {BAND_REF: json.loads(json.dumps(M.results(seasons, (base,), M.MLONS, n_boot)))[f"{base:g}"],
           POINT_REF: json.loads(json.dumps(PR.reference_results(seasons, refs[POINT_REF], base, n_boot=n_boot)))}
    want = {BAND_REF: matched["results"]["primary and transect"][f"{level}"][f"{base:g}"],
            POINT_REF: reference["results"][f"{level} hPa, {base:g} E"]["point_reference"]}
    for ref in REFERENCES:
        diffs = M._same(got[ref], want[ref])
        if diffs:
            raise SystemExit(f"REFUSED: the centroid reading at {level} hPa, {base:g} E with the {ref} reference does not reproduce the stored one, "
                             + "; ".join(diffs[:5]))
    return {ref: centroid_summary(got[ref]) for ref in REFERENCES}


def outcome(a_classes, b_classes):
    """Section 13's row from the eight classes of reader A and of reader B."""
    n = 2 * len(REFERENCES) * len(RANGES)
    if len(a_classes) != n or len(b_classes) != n:
        raise SystemExit("REFUSED: the outcome needs all eight readings of both readers")
    a_east = sum(c == "eastward" for c in a_classes)
    b_east = sum(c == "eastward" for c in b_classes)
    if a_east == n and b_east == 0:
        return "A eastward in all eight, B eastward in none"
    if a_east == n and b_east == n:
        return "A eastward in all eight, B eastward in all eight"
    if a_east == 0:
        return "A eastward in none"
    return "any other combination"


def summary(res):
    keys = [(lv, ref, d) for lv, _ in M.PRIMARY for ref in REFERENCES for d in RANGES]
    classes = {rule: {k: res[f"{k[0]}"][k[1]]["readers"][rule][f"{k[2]}"]["class"] for k in keys} for rule in READ_RULES}
    return {"classes": {rule: {f"{lv} hPa, {ref}, lags to {d} days": c for (lv, ref, d), c in v.items()} for rule, v in classes.items()},
            "outcome": outcome(list(classes[READER_A].values()), list(classes[READER_B].values())),
            "published_deg_per_day": {f"{lv} hPa": v for lv, v in PUBLISHED.items()}}


# ------------------------------------------------------------------ text, figure and main

def _f(v):
    return "none" if v is None else f"{v:+.2f}"


def _reading(r):
    pts = r["points"]
    span = f"{min(p[0] for p in pts):+.2f} to {max(p[0] for p in pts):+.2f}" if pts else "none"
    lons = f"{min(p[1] for p in pts):+.1f} to {max(p[1] for p in pts):+.1f}" if pts else "none"
    return f"{r['class']} | {_f(r['drift'])} ({_f(r['interval'][0])} to {_f(r['interval'][1])}) | {r['undefined']:.3f} | {len(pts)} | {span} | {lons}"


HEAD = ("| configuration | reference | rule | lags to (days) | class | drift (interval) | undefined | points | lag span (days) | longitudes (deg E) |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")


def constructions_text(rec):
    L = ["# the maxima rule on six constructions at the published bases (sections 3, 12 and 13)", ""]
    for name, key in (("The maxima rule after change 3", "verdict"), ("Reader B of section 13", "amendment_verdict")):
        v = rec.get(key)
        if v is not None:
            L += [f"{name} adequate: {'yes' if v['adequate'] else 'no'}." + ("" if v["adequate"] else " " + "; ".join(v["failures"])), ""]
    L += ["| case | base (deg E) | constructed | rule | lags to (days) | class | drift (interval) | undefined | points | lag span (days) | longitudes (deg E) | signal alone |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for c, by in rec["results"]["cases"].items():
        for b, v in by.items():
            for rule in [r for r in CONSTRUCTION_RULES if r in v["readers"]]:
                for d in RANGES:
                    clean = v.get("signal_alone", {}).get("readers", {}).get(rule, {}).get(f"{d}")
                    L.append(f"| {c} | {b} | {CONSTRUCTED[c]} | {rule} | {d} | {_reading(v['readers'][rule][f'{d}'])} | {_f(clean)} |")
    L += ["", "| case | base (deg E) | centroid class | centroid speed (interval) | signal alone |", "| --- | --- | --- | --- | --- |"]
    for c, by in rec["results"]["cases"].items():
        for b, v in by.items():
            r = v["centroid"]
            L.append(f"| {c} | {b} | {r['envelope_class']} | {_f(r['envelope_speed'])} ({_f(r['envelope_interval'][0])} to {_f(r['envelope_interval'][1])}) | "
                     f"{_f(v.get('signal_alone', {}).get('centroid'))} |")
    return "\n".join(L) + "\n"


def reading_text(res, summ):
    L = ["# the drift of regressed wind maxima at the published bases of Diaz and Aiyyer (2013), read under the amendment of section 13", "",
         f"The outcome row of section 13 is {summ['outcome']}.", ""]
    for rule, label in ((READER_A, "Reader A, the rule as declared"), (READER_B, "Reader B, change 1 with a two-peak minimum")):
        L += [f"{label}, gives " + ", ".join(f"{v} ({k})" for k, v in summ["classes"][rule].items()) + ".", ""]
    L += ["The paper's figures give regressed wind maxima moving at "
          + " and ".join(f"{v:+.2f} deg/day at {k}" for k, v in summ["published_deg_per_day"].items()) + ".", "", HEAD]
    for lv, b in M.PRIMARY:
        for ref in REFERENCES:
            for rule in READ_RULES:
                for d in RANGES:
                    L.append(f"| {lv} hPa, {b:g} E | {ref} | {rule} | {d} | {_reading(res[f'{lv}'][ref]['readers'][rule][f'{d}'])} |")
    L += ["", "| configuration | reference | centroid class (lags to 4 days) | centroid speed (interval) |", "| --- | --- | --- | --- |"]
    for lv, b in M.PRIMARY:
        for ref in REFERENCES:
            r = res[f"{lv}"][ref]["centroid"]
            L.append(f"| {lv} hPa, {b:g} E | {ref} | {r['envelope_class']} | {_f(r['envelope_speed'])} ({_f(r['envelope_interval'][0])} to {_f(r['envelope_interval'][1])}) |")
    return "\n".join(L) + "\n"


TITLE = ("Regressed meridional wind at the published bases, {band}, {n} seasons, read under the amendment of section 13",
         "Black circles and lines are reader A's maxima at lags to 7 days and its drifts, green crosses and lines reader B's.",
         "Solid lines are fitted to 7 days, dashed to 4. Magenta points are the centroid's positions at lags to 4 days.",
         "Gray contours mark plus and minus {thresh} m/s.")


def figure(maps, res):
    """The regression maps to 7 days with both readers' counted maxima at lags to 7 days and
    their fitted drifts at both lag ranges, and the centroid's followed positions, under a
    title that names the amendment and fits the canvas."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(M.PRIMARY), len(REFERENCES), figsize=(10.0, 8.0), sharex=True, sharey=True, constrained_layout=True)
    lagd = LAGS * PE.DT_DAYS
    for r_i, (lv, b) in enumerate(M.PRIMARY):
        for c_i, ref in enumerate(REFERENCES):
            ax = axes[r_i, c_i]
            R = maps[(lv, ref)]
            levels = np.arange(-3.15, 3.2, 0.3)
            cs = ax.contourf(M.MLONS, lagd, R, levels=levels, cmap="RdBu_r", extend="both")
            ax.contour(M.MLONS, lagd, R, levels=[-THRESH, THRESH], colors="0.4", linewidths=0.5)
            v = res[f"{lv}"][ref]
            for rule, color, mark in ((READER_A, "k", "o"), (READER_B, "g", "x")):
                for d, style in ((7, "-"), (4, "--")):
                    rr = v["readers"][rule][f"{d}"]
                    pts = np.array(rr["points"]) if rr["points"] else np.zeros((0, 3))
                    if d == 7 and len(pts):
                        ax.plot(pts[:, 1], pts[:, 0], mark, mfc="none", mec=color, color=color, ms=5)
                    if rr["drift"] is not None:
                        t = np.array([-d, d], float)
                        x0 = float(np.mean(pts[:, 1]) - rr["drift"] * np.mean(pts[:, 0]))
                        ax.plot(x0 + rr["drift"] * t, t, style, color=color, lw=1.2)
            cr = v["centroid"]
            ax.plot(cr["positions"], np.array(cr["lags"]) * PE.DT_DAYS, ".", color="m", ms=3)
            ax.set_title(f"{lv} hPa, base {b:g} E, {ref} reference", fontsize=9)
            ax.set_xlim(M.MLONS[0], M.MLONS[-1]); ax.set_ylim(-MAX_LAG_DAYS, MAX_LAG_DAYS)
            if c_i == 0:
                ax.set_ylabel("lag (days)")
            if r_i == len(M.PRIMARY) - 1:
                ax.set_xlabel("longitude (deg E)")
    fig.colorbar(cs, ax=axes, shrink=0.6, label="regressed meridional wind (m/s per standard deviation of the reference)")
    fig.suptitle("\n".join(TITLE).format(band=M.MATCHED_BAND, n=N_SEASONS, thresh=THRESH), fontsize=8)
    return fig


def draw(path, maps, res):
    import matplotlib.pyplot as plt
    fig = figure(maps, res)
    fig.savefig(path, dpi=90); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("constructions", "read"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--constructions", default=None)
    ap.add_argument("--matched", default=None)
    ap.add_argument("--reference", default=None)
    ap.add_argument("--earlier", default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    M.check_frozen()
    gen = {"script": "scripts/eastern_phase_envelope_maxima.py", "sha256": SCRIPT_SHA256_AT_LOAD,
           "declared": "EASTERN_PHASE_ENVELOPE_MAXIMA_2026-10-27.md", "frozen_sha256": PE.SCRIPT_SHA256_AT_LOAD,
           "matched_sha256": M.SCRIPT_SHA256_AT_LOAD, "probes_sha256": PR.SCRIPT_SHA256_AT_LOAD,
           "rule": {"thresh_ms": THRESH, "link_deg": LINK_DEG, "min_count": MIN_COUNT, "ranges_days": list(RANGES),
                    "version": 3, "earlier_versions": list(EARLIER), "amendment_readers": list(READ_RULES),
                    "min_counts": MIN_COUNTS}}
    if a.mode == "constructions":
        pj, pm = (os.path.join(a.out_dir, f"eval60_phase_env_maxima_constructions{a.tag}.{e}") for e in ("json", "md"))
        PE.refuse_existing([pj, pm])
        res = json.loads(json.dumps(run_constructions()))
        rec = {"generated_by": gen, "results": res, "verdict": adequacy(res), "amendment_verdict": adequacy(res, READER_B)}
        with open(pj, "w", encoding="utf-8") as fh:
            json.dump(rec, fh)
        text = constructions_text(rec)
        with open(pm, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(text)
        return 0
    if not (a.constructions and a.earlier and a.matched and a.reference):
        raise SystemExit("REFUSED: --constructions, --earlier, --matched and --reference are required for the reading")
    paths = {k: os.path.join(a.out_dir, f"eval60_phase_env_maxima{s}{a.tag}.{e}") for k, s, e in
             (("record", "", "json"), ("tables", "_tables", "md"), ("maps", "_maps", "png"))}
    PE.refuse_existing(paths.values())
    check_constructions(a.constructions, a.earlier)
    with open(a.matched, encoding="utf-8") as fh:
        matched = json.load(fh)
    with open(a.reference, encoding="utf-8") as fh:
        reference = json.load(fh)
    W = PE.boot_weights(N_SEASONS)
    res, maps = {}, {}
    for lv, b in M.PRIMARY:
        seasons, refs = era5_inputs(lv, b)
        cent = check_centroid(lv, b, seasons, refs, matched, reference)
        res[f"{lv}"] = {}
        for ref in REFERENCES:
            st = season_stack(seasons, refs[ref])
            maps[(lv, ref)] = regression(st)
            res[f"{lv}"][ref] = {"readers": read_all(st, M.MLONS, W, READ_RULES), "centroid": cent[ref]}
        print(f"gates passed and read at {lv} hPa, {b:g} E")
    res = json.loads(json.dumps(res))
    summ = summary(res)
    draw(paths["maps"], maps, res)
    gen.update({"amendment": "section 13, a departure from section 3 approved by the authors",
                "constructions": {"path": a.constructions, "sha256": PE.sha256_of(a.constructions)},
                "earlier": {"path": a.earlier, "sha256": PE.sha256_of(a.earlier)},
                "matched": {"path": a.matched, "sha256": PE.sha256_of(a.matched)},
                "reference": {"path": a.reference, "sha256": PE.sha256_of(a.reference)},
                "inputs": {os.path.basename(p): PE.sha256_of(p) for p in M.input_files(M.SHARED_YEARS)},
                "gates": "1 frozen digest, 2 centroid against the stored readings, 3 constructions record current with reader A unchanged and reader B adequate, "
                         "4 complete seasons, 5 readers A and B only"})
    with open(paths["record"], "w", encoding="utf-8") as fh:
        json.dump({"generated_by": gen, "results": res, "summary": summ,
                   "figure": {"path": paths["maps"], "sha256": PE.sha256_of(paths["maps"])}}, fh)
    text = reading_text(res, summ)
    with open(paths["tables"], "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
