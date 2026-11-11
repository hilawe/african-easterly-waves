#!/usr/bin/env python3
"""The break reading of the continuity effort, declared in section 27 of
`CONTINUITY_ACROSS_AFRICA_2026-11-09.md` before it was run.

On the development seasons 1990 and 2006, for the released configuration and for VCLIP, the reading
finds where single-history accompaniment of the reference tracks breaks and what stands behind each
break. The histories are the stored tracks the lifecycle records retain, and a season is refused
unless they reproduce the lifecycle census and the placement development records.

A TRANSITION is counted at every reference observation a stored history accompanies, except the
last one read. Each history beside the reference there (a HOLDER) is followed to its own next
observation, 6 or 12 hours on: BESIDE within 500 km of the reference at that time, AWAY when
farther, ENDED with no later observation, PAST THE WINDOW when the next observation falls after the
first Atlantic-side observation. The transition is HELD when any holder is beside, UNRESOLVED when
none is and one is past the window, and a BREAK otherwise.

Four parts are read. The first compares the break share of the COASTAL transitions (leaving an
observation 24 hours or less before the first Atlantic-side observation) with the EARLIER ones, by
reference and with an exact sign test. The second gives where each primary history of the
placement records stops. The third classes each break of a coast-reaching reference by what stands
at the reference's next observation, with section 8's classes under that configuration's final
candidates and section 10's field reading. The fourth replays each season with the replay
tooling's matching read taken before the production call, and classes the final candidate nearest
the reference at the main holder's 6-hour and 12-hour attempts.

Everything is relative to QTrack and to two development seasons. A candidate within 500 km of a
reference is not shown to be the disturbance QTrack follows. Nothing here changes a tracker
setting or an earlier record.

    python3 scripts/continuity_break_reading.py season --year 1990 --out-dir DIR
    python3 scripts/continuity_break_reading.py combine --out-dir DIR
"""

import argparse
import gzip
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
import continuity_census as C  # noqa: E402
import continuity_evaluation as E  # noqa: E402
import continuity_lifecycle as L  # noqa: E402
import continuity_placement_check as P  # noqa: E402
import continuity_readings as RD  # noqa: E402
import continuity_recovery as REC  # noqa: E402
import continuity_stage_sample as S  # noqa: E402

with open(os.path.abspath(__file__), "rb") as _fh:
    SCRIPT_SHA256_AT_LOAD = hashlib.sha256(_fh.read()).hexdigest()

CONFIGS = P.CONFIGS
YEARS = S.YEARS
ARTIFACTS = L.JOIN_DIR
STEM = "continuity_break_reading"
RADIUS_KM = C.RADIUS_KM
MARGINAL_KM = 750.0                                  # an away holder this near is recorded as marginal
COASTAL_HOURS = 24                                   # a transition leaving an observation this long or less before the coast is coastal
MIN_SCORED = 5                                       # the fewest references for which the sign test can reach 0.05
SEGMENTS = ("coastal", "earlier")
BANDS = ("east_of_10E", "0_to_10E", "10W_to_0", "west_of_10W")
TRANSITIONS = ("held", "break", "unresolved")
STOPS = ("at_the_first_atlantic_side_observation", "within_24_hours", "30_to_48_hours", "earlier")
SHORTFALLS = ("start", "end", "majority", "no_long_gap")
NEXT = ("handed_off_fresh", "handed_off_running", "candidate_not_stored", "placed_away", "signal_without_axis", "masked_by_westerlies", "below_threshold")
NOT_STORED = ("held_by_a_finished_history", "held_by_pruned_histories_only", "held_by_a_history_the_final_filter_removed", "not_held")
ATTEMPTS = ("no_candidate_near", "outside_the_search_area", "over_the_speed_limit", "another_candidate_nearer", "linked")
SUPPORT = ("supported", "partial", "unsupported")
OUTCOMES = ("concentrated near the coast", "concentrated earlier", "cannot be scored", "not shown")


# --- transitions along one reference --------------------------------------------------------------

def holder_step(h, q, k, n, radius_km=RADIUS_KM, marginal_km=MARGINAL_KM):
    """What one history beside observation `k` of a reference read over `n` observations does at
    its own next observation."""
    key = C.time_key(q["time"][k])
    j = next(i for i, d in enumerate(h["time"]) if C.time_key(d) == key)
    out = {"step": int(h["step"][j])}
    if j + 1 >= len(h["time"]):
        return {**out, "outcome": "ended"}
    k2 = k + C.time_key(h["time"][j + 1]) - key
    out.update({"to": k2, "next_step": int(h["step"][j + 1])})
    if k2 > n - 1:
        return {**out, "outcome": "past_the_window"}
    if C.time_key(q["time"][k2]) != C.time_key(h["time"][j + 1]):
        raise SystemExit(f"REFUSED: reference {q['id']} is not six-hourly over the observations read")
    d = float(C.km(h["lat"][j + 1], h["lon"][j + 1], q["lat"][k2], q["lon"][k2]))
    if d <= radius_km:
        return {**out, "outcome": "beside"}
    raw = float(C.km(h["raw_lat"][j + 1], h["raw_lon"][j + 1], q["lat"][k2], q["lon"][k2]))
    return {**out, "outcome": "away", "km": round(d, 1), "marginal": bool(d <= marginal_km), "by_smoothing_alone": bool(raw <= radius_km)}


def segment(hours_to_coast):
    return "coastal" if hours_to_coast <= COASTAL_HOURS else "earlier"


def band(lon):
    return BANDS[0] if lon >= 10.0 else (BANDS[1] if lon >= 0.0 else (BANDS[2] if lon >= -10.0 else BANDS[3]))


def transitions(row, q, by_id):
    """One row per accompanied observation of a reference except the last one read, from a row of
    `continuity_evaluation.reference_measures`."""
    n, coast = row["steps"], row["outcome"] == "atlantic_side"
    at = {}
    for hid, idx in row["by_history"].items():
        for k in idx:
            at.setdefault(k, []).append(hid)
    out = []
    for k in sorted(at):
        if k >= n - 1:
            continue
        holders = {hid: holder_step(by_id[hid], q, k, n) for hid in sorted(at[k])}
        kinds = {v["outcome"] for v in holders.values()}
        cls = "held" if "beside" in kinds else ("unresolved" if "past_the_window" in kinds else "break")
        hours = int(round(24.0 * (float(q["time"][n - 1]) - float(q["time"][k])))) if coast else None
        t = {"k": k, "lon": round(float(q["lon"][k]), 3), "band": band(float(q["lon"][k])), "hours_to_coast": hours,
             "segment": segment(hours) if coast else None, "class": cls, "holders": holders}
        if cls == "break":
            away = [v for v in holders.values() if v["outcome"] == "away"]
            t["main_holder"] = min(holders, key=lambda h: (-len(row["by_history"][h]), by_id[h]["step"][0], h))
            t["last_index"] = n - 1
            t["holders_did"] = "continued_away" if away else "ended"
            if away:
                t.update({"away_km": min(v["km"] for v in away), "marginal": any(v["marginal"] for v in away),
                          "by_smoothing_alone": any(v["by_smoothing_alone"] for v in away)})
        out.append(t)
    return out


def primary_stop(prow):
    """Where the primary history of a coast-reaching reference of the placement records stops."""
    p = prow["primary"]
    if p is None:
        return None
    hours = 6 * (prow["steps"] - 1 - p["last"])
    stop = STOPS[0] if hours == 0 else (STOPS[1] if hours <= 24 else (STOPS[2] if hours <= 48 else STOPS[3]))
    return {"history": p["history"], "last": p["last"], "hours_before": hours, "stop": stop, **{k: bool(p[k]) for k in SHORTFALLS}}


def reference_reading(base, place_rows, q, by_id):
    """One row per reference of one configuration, refusing unless the lifecycle rows give the
    reference rows the placement record holds."""
    mine = [(r["id"], r["outcome"], r["steps"], r["accompanied"], r["longest"]) for r in base]
    theirs = [(p["id"], p["outcome"], p["steps"], p["accompanied"], p["best_count"]) for p in place_rows]
    if mine != theirs:
        raise SystemExit("REFUSED: the retained histories do not give the reference rows of the placement record")
    out = []
    for r, p in zip(base, place_rows):
        coast = r["outcome"] == "atlantic_side"
        out.append({"id": r["id"], "coast_reaching": coast, "steps": r["steps"], "accompanied": r["accompanied"], "sustained": bool(p["sustained"]),
                    "primary": primary_stop(p) if coast else None, "transitions": transitions(r, q[r["id"]], by_id)})
    return out


# --- the concentration reading ----------------------------------------------------------------------

def counts_of(ts):
    return {c: sum(1 for t in ts if t["class"] == c) for c in TRANSITIONS}


def per_reference(refs):
    """For each coast-reaching reference, [breaks, counted transitions] in each segment."""
    out = []
    for r in refs:
        if not r["coast_reaching"]:
            continue
        row = {"id": r["id"]}
        for s in SEGMENTS:
            c = counts_of([t for t in r["transitions"] if t["segment"] == s])
            row[s] = [c["break"], c["held"] + c["break"]]
        out.append(row)
    return out


def concentration(rows):
    """Section 27's reading over the references of both seasons, decided in whole numbers. A
    reference is scored when it has counted transitions in both segments, and the sign test is
    read over the scored references whose two break shares differ."""
    plus = minus = ties = one_segment = 0
    for r in rows:
        (bc, nc), (be, ne) = r["coastal"], r["earlier"]
        if nc == 0 or ne == 0:
            one_segment += 1
        elif bc * ne > be * nc:
            plus += 1
        elif bc * ne < be * nc:
            minus += 1
        else:
            ties += 1
    bc, nc = sum(r["coastal"][0] for r in rows), sum(r["coastal"][1] for r in rows)
    be, ne = sum(r["earlier"][0] for r in rows), sum(r["earlier"][1] for r in rows)
    if plus + minus + ties < MIN_SCORED:
        outcome = OUTCOMES[2]
    elif plus > minus and P.at_most_alpha(P.sign_test(plus, minus)) and bc * ne > be * nc:
        outcome = OUTCOMES[0]
    elif minus > plus and P.at_most_alpha(P.sign_test(minus, plus)) and bc * ne < be * nc:
        outcome = OUTCOMES[1]
    else:
        outcome = OUTCOMES[3]
    hi, lo = max(plus, minus), min(plus, minus)
    num, den = P.sign_test(hi, lo)
    return {"references": len(rows), "scored": plus + minus + ties, "higher_share_coastal": plus, "higher_share_earlier": minus, "equal_shares": ties,
            "transitions_in_one_segment_only": one_segment, "sign_test": [num, den], "sign_test_value": round(num / den, 4),
            "pooled": {"coastal": [bc, nc], "earlier": [be, ne]}, "outcome": outcome}


# --- what stands at the reference's next observation -----------------------------------------------

def check_finals(waves, held, step, name):
    """The final candidates recomputed at a step must be the candidate positions the retained
    histories hold there: every retained position is a candidate's, and every candidate with a
    trough point whose position lies in the retained box is held."""
    cand = {(round(float(w["lat_mean"]), 9), round(float(w["lon_mean"]), 9)) for w in waves}
    if set(held) - cand:
        raise SystemExit(f"REFUSED: at step {step} a retained {name} history holds a position that is no final candidate of the recomputed detection")
    for w in waves:
        pos = (round(float(w["lat_mean"]), 9), round(float(w["lon_mean"]), 9))
        if np.asarray(w["lat_wave"]).size and L.in_box({"lat": [pos[0]], "lon": [pos[1]]}) and pos not in held:
            raise SystemExit(f"REFUSED: at step {step} a final candidate of the recomputed {name} detection is held by no retained history")


def not_stored_class(fates):
    if "finished" in fates:
        return NOT_STORED[0]
    if "pruned_in_loop" in fates:
        return NOT_STORED[1]
    return NOT_STORED[2] if fates else NOT_STORED[3]


def next_state(position, waves, held, receiver, step, lat, lon, radius_km=RADIUS_KM):
    """The class of a break from `continuity_stage_sample.read_position`'s reading at the
    reference's next observation. `receiver` is the nearest stored history there, `held` the
    retained histories by candidate position at that step."""
    cls = position["class"]
    if cls == "tracked":
        return {"next": NEXT[0] if receiver["step"][0] == step else NEXT[1], "receiver": receiver["id"]}
    if cls != "candidate_not_tracked":
        return {"next": cls}
    d = [float(C.km(lat, lon, w["lat_mean"], w["lon_mean"])) for w in waves]
    i = int(np.argmin(d))
    if d[i] > radius_km:
        raise SystemExit(f"REFUSED: at step {step} a candidate-not-stored class has no final candidate within the radius")
    fates = sorted(h["fate"] for h in held.get((round(float(waves[i]["lat_mean"]), 9), round(float(waves[i]["lon_mean"]), 9)), ()))
    return {"next": NEXT[2], "holder_fates": fates, "not_stored": not_stored_class(fates)}


# --- the association attempts of a break's main holder ----------------------------------------------

def attempts_of(t):
    """The attempts to read for a break: the main holder's 6-hour attempt at the reference's next
    observation and, when the holder has no observation there and the observation after it is
    inside the window, its 12-hour attempt. Each is (pass, step, reference index)."""
    h = t["holders"][t["main_holder"]]
    out = [("6hr", h["step"] + 1, t["k"] + 1)]
    if h.get("next_step") != h["step"] + 1 and t["k"] + 2 <= t["last_index"]:
        out.append(("12hr", h["step"] + 2, t["k"] + 2))
    return out


def attempt_read(match, live, waves, pss, lat, lon, radius_km=RADIUS_KM):
    """The class of the final candidate nearest a reference position at one association attempt,
    from the replay tooling's matching read and live snapshot of the holder."""
    from aew.v1port import association as ASSOC
    from aew.v1port.geometry import points_in_polygon
    if match is None or live is None or match["pass"] != pss:
        raise SystemExit(f"REFUSED: a holder was not due in the {pss} pass at the step its attempt was to be read")
    d = [float(C.km(lat, lon, w["lat_mean"], w["lon_mean"])) for w in waves]
    chosen = match["chosen"]
    out = {"pass": pss, "chosen": None if chosen is None else [float(waves[chosen]["lat_mean"]), float(waves[chosen]["lon_mean"])],
           "candidates_inside": len(match["candidates_inside_polygon"])}
    if chosen is not None:
        out["chosen_km_from_reference"] = round(d[chosen], 1)
    i = int(np.argmin(d)) if d else None
    if i is None or d[i] > radius_km:
        return {**out, "class": ATTEMPTS[0]}
    w, est = waves[i], live["est_" + pss]
    known = est[0] is not None and est[1] is not None
    out.update({"candidate_km": round(d[i], 1), "prediction_km": round(float(C.km(est[0], est[1], w["lat_mean"], w["lon_mean"])), 1) if known else None})
    if i not in match["candidates_inside_polygon"]:
        poly = np.asarray(match["polygon_as_read"], float)
        minimum = ASSOC.POLY_AREA_6HR if pss == "6hr" else ASSOC.POLY_AREA_12HR
        area = float(ASSOC._polygon_area(poly[:, 0], poly[:, 1]))
        plon, plat = ASSOC._inflate(poly[:, 0].copy(), poly[:, 1].copy(), minimum)
        return {**out, "class": ATTEMPTS[1], "polygon_area": round(area, 2) if np.isfinite(area) else None, "polygon_at_or_below_minimum": bool(area <= minimum),
                "inside_if_inflated": bool(points_in_polygon(plon, plat, w["lon_wave"], w["lat_wave"]).any())}
    if match["implied_speed_ms"][str(i)] >= ASSOC.MAX_SPEED_MS:
        return {**out, "class": ATTEMPTS[2]}
    return {**out, "class": ATTEMPTS[3] if chosen != i else ATTEMPTS[4]}


def traced_loop(case, grids, ct, ft, flags, options, wanted, loop=L.lifecycle_loop):
    """The registered production loop with the matching read taken before the production call
    at the wanted steps. `wanted` maps a step to [(key, holder birth, pass, lat, lon)]. The
    production function is put back whatever happens."""
    import pilot_association_replay as AR
    from aew.v1port import association as ASSOC
    original, seen, out = ASSOC.associate_step, {}, {}

    def hook(**kw):
        seen["registry"] = kw["registry"]
        return kw["tracks"], kw["states"]

    def reading(tracks, states, waves, step, u_median, v_median, **kw):
        if step in wanted:
            births = {i: r["birth"] for i, r in seen.get("registry", {}).items()}
            match = {m["birth"]: m for m in AR.read_matching(tracks, states, waves, step, bool(kw.get("exclusive", False)), births)} if waves else {}
            live = {r["birth"]: r for r in AR.live_snapshot(tracks, states, step, births)}
            for key, birth, pss, lat, lon in wanted[step]:
                out[key] = attempt_read(match.get(birth), live.get(birth), waves, pss, lat, lon)
        return original(tracks, states, waves, step, u_median, v_median, **kw)

    ASSOC.associate_step = reading
    try:
        finished, rows, _checks = loop(case, grids, ct, ft, flags, options, hook=hook)
    finally:
        ASSOC.associate_step = original
    return finished, rows, out


def check_attempt(a, holder, step):
    """An attempt's read must be what the retained history did at that step."""
    did = next(([float(la), float(lo)] for s, la, lo in zip(holder["step"], holder["raw_lat"], holder["raw_lon"]) if s == step), None)
    if a["chosen"] != did:
        raise SystemExit(f"REFUSED: at step {step} the matching read and the retained history {holder['id']} disagree on what the holder took")


# --- one season ---------------------------------------------------------------------------------------

def load_records(year, artifacts=ARTIFACTS):
    with gzip.open(os.path.join(artifacts, f"continuity_lifecycle_{year}.json.gz"), "rt", encoding="utf-8") as fh:
        life = json.load(fh)
    if life["generated_by"]["sha256"] != L.SCRIPT_SHA256_AT_LOAD:
        raise SystemExit(f"REFUSED: the {year} lifecycle record was not written by the current lifecycle script")
    with gzip.open(os.path.join(artifacts, f"{P.STEM['development']}_{year}.json.gz"), "rt", encoding="utf-8") as fh:
        place = json.load(fh)
    g = place["generated_by"]
    if g["sha256"] != P.SCRIPT_SHA256_AT_LOAD or g["lifecycle_sha256"] != L.SCRIPT_SHA256_AT_LOAD:
        raise SystemExit(f"REFUSED: the {year} placement development record was not written by the current scripts")
    if place["artifact"]["sha256"] != life["artifact"]["sha256"] or C.sha256_of(place["artifact"]["path"]) != place["artifact"]["sha256"]:
        raise SystemExit(f"REFUSED: the {year} records do not name one coast-crossing artifact, or it has changed")
    return life, place


def breaks_of(refs):
    return [(r, t) for r in refs if r["coast_reaching"] for t in r["transitions"] if t["class"] == "break"]


def _finite(x):
    return x if np.isfinite(x) else None


def run_season(year, artifacts=ARTIFACTS, loader=REC.load_season, loop=L.lifecycle_loop):
    if year not in YEARS:
        raise SystemExit(f"REFUSED: {year} is not a development season of section 8")
    import pilot_alledge_replay as A
    import pilot_event_continuity as EC
    import pilot_margin_replay as MR
    import qtrack_pairing_pilot as M
    life, place = load_records(year, artifacts)
    art = json.load(open(place["artifact"]["path"], encoding="utf-8"))
    if C.sha256_of(art["inputs"]["qtrack"]["path"]) != art["inputs"]["qtrack"]["sha256"]:
        raise SystemExit(f"REFUSED: {art['inputs']['qtrack']['path']} is not the file the {year} artifact names")
    q_tracks = M.load_qtrack(art["inputs"]["qtrack"]["path"])[0]
    q, q_rows = {t["id"]: t for t in q_tracks}, art["sides"]["qtrack"]["tracks"]
    run, case, ct, ft, flags, grids = loader(year)
    if case["sha256"] != life["case_sha256"]:
        raise SystemExit(f"REFUSED: the {year} case is not the one the lifecycle census replayed")
    absorb = bool(flags["absorb"])
    allc = (np.ones(case["lat_c"].size, bool), np.ones(case["lon_c"].size, bool))
    allf = (np.ones(case["latgrid"].shape[0], bool), np.ones(case["longrid"].shape[1], bool))
    # the histories and the breaks of each configuration, from the retained records alone
    cfg, needs = {}, {}
    for name in CONFIGS:
        rows = life["configs"][name]["rows"]
        sets = L.history_sets(rows)
        by_id = {h["id"]: h for h in sets["STORED"]}
        base = E.reference_measures(sets["STORED"], q, q_rows)
        if E.pool_reference(base) != life["configs"][name]["census"]["STORED"]["reference"]:
            raise SystemExit(f"REFUSED: the {year} {name} rows do not reproduce the lifecycle census")
        held = {}
        for h in sets["ALL"]:
            for s, la, lo in zip(h["step"], h["raw_lat"], h["raw_lon"]):
                held.setdefault(s, {}).setdefault((round(la, 9), round(lo, 9)), []).append(h)
        cfg[name] = {"rows": rows, "by_id": by_id, "refs": reference_reading(base, place["configs"][name]["rows"], q, by_id), "held": held,
                     "nearest": {r["id"]: r["nearest"] for r in base}, "index": C.index_by_time(sets["STORED"]), "wanted": {}, "pending": []}
        for r, t in breaks_of(cfg[name]["refs"]):
            step = t["holders"][t["main_holder"]]["step"] + 1
            if C.time_key(case["time"][step]) != C.time_key(q[r["id"]]["time"][t["k"] + 1]):
                raise SystemExit(f"REFUSED: step {step} of the {year} case is not the time of reference {r['id']}'s next observation")
            needs.setdefault(step, []).append((name, r, t))
    # the detector and the fields at each step a break's next observation falls on
    staged_steps = {name: 0 for name in CONFIGS}
    for step in sorted(needs):
        tt = float(case["time"][step])
        fields = A.prepared_fields(case, step, *allc, *allf)
        stages = {}
        for name in sorted({x[0] for x in needs[step]}):
            options = CONFIGS[name]
            if options:
                st = EC.stages_from_prepared(fields, tt, ct, ft, grids, absorb=absorb, **options)
                if MR.signature(st["final"]) != MR.signature(A.detect_from_prepared(fields, tt, ct, ft, grids, absorb=absorb, **options)):
                    raise SystemExit(f"REFUSED: at step {step} the staged {name} detection does not reproduce the replay's detector")
            else:
                st = S.staged(case, step, ct, ft, grids, absorb)
            check_finals(st["final"], cfg[name]["held"].get(step, {}), step, f"{year} {name}")
            stages[name] = st
            staged_steps[name] += 1
        for name, r, t in needs[step]:
            c, qt, k1 = cfg[name], q[r["id"]], t["k"] + 1
            lat, lon = float(qt["lat"][k1]), float(qt["lon"][k1])
            hit = c["index"].get(C.time_key(qt["time"][k1]))
            stored_km = S.nearest_km(lat, lon, hit[0], hit[1]) if hit is not None else float("inf")
            pos = S.read_position(stages[name], grids, ct, lat, lon, stored_km)
            receiver = c["by_id"][c["nearest"][r["id"]][k1]] if pos["class"] == "tracked" else None
            cr = RD.trough_crossing(fields["fine_meridional_wind_smoothed"], grids["latgrid_f"], grids["longrid_f"], lat, lon)
            mx = RD.local_maximum(fields["fine_curvature"], grids["latgrid_f"], grids["longrid_f"], ft, lat, lon)
            t.update({"next_step": step, "next_position": [round(lat, 3), round(lon, 3)], "axis_km": _finite(pos["axis_km"]), "final_km": _finite(pos["final_km"]),
                      "support": RD.support(cr["present"], mx), **next_state(pos, stages[name]["final"], c["held"].get(step, {}), receiver, step, lat, lon)})
            for pss, s, k in attempts_of(t):
                key = (r["id"], t["k"], pss)
                c["wanted"].setdefault(s, []).append((key, t["main_holder"], pss, float(qt["lat"][k]), float(qt["lon"][k])))
                c["pending"].append((key, t, s))
    # the association at each break, read before the production call in an unchanged replay
    out = {}
    for name, options in CONFIGS.items():
        c = cfg[name]
        finished, traced, reads = traced_loop(case, grids, ct, ft, flags, options, c["wanted"], loop=loop)
        if name == "released" and not MR.archive_gate(finished, run)["passed"]:
            raise SystemExit(f"REFUSED: the {year} released replay does not reproduce the stored campaign tracks")
        if [x for x in traced if L.in_box(x)] != c["rows"]:
            raise SystemExit(f"REFUSED: the {year} {name} traced replay's histories are not those of the lifecycle census")
        for key, t, s in c["pending"]:
            if key not in reads:
                raise SystemExit(f"REFUSED: the {year} {name} replay did not read the attempt {key}")
            check_attempt(reads[key], c["by_id"][t["main_holder"]], s)
            t.setdefault("attempts", []).append(reads[key])
        out[name] = {"references": c["refs"], "steps_staged": staged_steps[name], "attempts_read": len(c["pending"])}
    return {"year": year, "case_sha256": case["sha256"], "artifact": place["artifact"],
            "lifecycle_record_sha256": C.sha256_of(os.path.join(artifacts, f"continuity_lifecycle_{year}.json.gz")),
            "placement_record_sha256": C.sha256_of(os.path.join(artifacts, f"{P.STEM['development']}_{year}.json.gz")), "configs": out}


# --- pooling and tables ---------------------------------------------------------------------------------

def _share(pair):
    return f"{pair[0]} of {pair[1]}" + (f" ({100.0 * pair[0] / pair[1]:.1f} %)" if pair[1] else "")


def _median(values):
    return None if not values else round(float(np.median(values)), 1)


def tally(refs):
    """Every table of one configuration from its reference rows, one season or several."""
    coast = [r for r in refs if r["coast_reaching"]]
    inland = [r for r in refs if not r["coast_reaching"]]
    ts = [t for r in coast for t in r["transitions"]]
    out = {"references": len(refs), "coast_reaching": len(coast),
           "transitions": {s: counts_of([t for t in ts if t["segment"] == s]) for s in SEGMENTS},
           "bands": {"coast_reaching": {b: counts_of([t for t in ts if t["band"] == b]) for b in BANDS},
                     "ending_inside_africa": {b: counts_of([t for r in inland for t in r["transitions"] if t["band"] == b]) for b in BANDS}},
           "per_reference": per_reference(refs)}
    with_primary = [r for r in coast if r["primary"] is not None and not r["sustained"]]
    out["primary"] = {"sustained": sum(1 for r in coast if r["sustained"]), "no_history": sum(1 for r in coast if r["primary"] is None), "others": len(with_primary),
                      "lacking": {k: sum(1 for r in with_primary if not r["primary"][k]) for k in SHORTFALLS},
                      "stops": {s: sum(1 for r in coast if r["primary"] is not None and r["primary"]["stop"] == s) for s in STOPS},
                      "stops_lacking_the_end": {s: sum(1 for r in with_primary if not r["primary"]["end"] and r["primary"]["stop"] == s) for s in STOPS}}
    br = [t for t in ts if t["class"] == "break"]
    by_seg = {s: [t for t in br if t["segment"] == s] for s in SEGMENTS}
    out["breaks"] = {"count": len(br), "references_with_a_break": sum(1 for r in coast if any(t["class"] == "break" for t in r["transitions"])),
                     "next": {s: {c: sum(1 for t in by_seg[s] if t["next"] == c) for c in NEXT} for s in SEGMENTS},
                     "holders_did": {s: {c: sum(1 for t in by_seg[s] if t["holders_did"] == c) for c in ("ended", "continued_away")} for s in SEGMENTS},
                     "holders_did_by_next": {c: {d: sum(1 for t in br if t["next"] == c and t["holders_did"] == d) for d in ("ended", "continued_away")} for c in NEXT},
                     "away_marginal": sum(1 for t in br if t.get("marginal")), "away_by_smoothing_alone": sum(1 for t in br if t.get("by_smoothing_alone")),
                     "not_stored": {c: sum(1 for t in br if t.get("not_stored") == c) for c in NOT_STORED},
                     "support": {c: sum(1 for t in br if t["support"] == c) for c in SUPPORT},
                     "support_by_next": {c: {u: sum(1 for t in br if t["next"] == c and t["support"] == u) for u in SUPPORT} for c in NEXT}}
    first = [(t, t["attempts"][0]) for t in br]
    second = [(t, t["attempts"][1]) for t in br if len(t["attempts"]) > 1]
    outside = [a for _, a in first if a["class"] == ATTEMPTS[1]]
    out["attempts"] = {"first": {c: sum(1 for _, a in first if a["class"] == c) for c in ATTEMPTS},
                       "first_by_next": {n: {c: sum(1 for t, a in first if t["next"] == n and a["class"] == c) for c in ATTEMPTS} for n in NEXT},
                       "first_by_segment": {s: {c: sum(1 for t, a in first if t["segment"] == s and a["class"] == c) for c in ATTEMPTS} for s in SEGMENTS},
                       "second_read": len(second), "second": {c: sum(1 for _, a in second if a["class"] == c) for c in ATTEMPTS},
                       "first_outside": {"count": len(outside), "inside_if_inflated": sum(1 for a in outside if a["inside_if_inflated"]),
                                         "polygon_at_or_below_minimum": sum(1 for a in outside if a["polygon_at_or_below_minimum"]),
                                         "median_prediction_km": _median([a["prediction_km"] for a in outside]),
                                         "median_polygon_area": _median([a["polygon_area"] for a in outside if a["polygon_area"] is not None])},
                       "candidate_near_at_either_attempt": sum(1 for t in br if any(a["class"] != ATTEMPTS[0] for a in t["attempts"]))}
    out["concentration"] = concentration(out["per_reference"])
    return out


def tables_text(rec):
    T = ["# Break reading of the continuity effort (declared section 27)", "",
         f"{', '.join(map(str, YEARS))} pooled, the coast-reaching reference tracks read to their first Atlantic-side observation. A transition is counted at every reference "
         "observation a stored history accompanies except the last, and is held when a history beside the reference there is beside it at that history's next observation. "
         "Everything is relative to QTrack.", ""]
    for name, p in rec["pooled"].items():
        c, pr, b, a = p["concentration"], p["primary"], p["breaks"], p["attempts"]
        T += [f"## {name}", "", "| segment | held | break | unresolved | break share of held and break |", "| --- | --- | --- | --- | --- |"]
        for s in SEGMENTS:
            x = p["transitions"][s]
            T.append(f"| {s} | {x['held']} | {x['break']} | {x['unresolved']} | {_share([x['break'], x['held'] + x['break']])} |")
        T += ["", f"By reference ({c['references']} coast-reaching, {c['scored']} with counted transitions in both segments): higher break share in the coastal segment "
                  f"{c['higher_share_coastal']}, in the earlier segment {c['higher_share_earlier']}, equal {c['equal_shares']}, transitions in one segment only "
                  f"{c['transitions_in_one_segment_only']}. Sign test over the {c['higher_share_coastal'] + c['higher_share_earlier']} references whose shares differ "
                  f"{c['sign_test'][0]}/{c['sign_test'][1]} = {c['sign_test_value']:.4f}. Outcome: {c['outcome'].upper()}.", "",
              "| longitude of the observation left | held | break | break share | held, references ending inside Africa | break, references ending inside Africa | break share |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
        for bd in BANDS:
            x, y = p["bands"]["coast_reaching"][bd], p["bands"]["ending_inside_africa"][bd]
            T.append(f"| {bd.replace('_', ' ')} | {x['held']} | {x['break']} | {_share([x['break'], x['held'] + x['break']])} | {y['held']} | {y['break']} | "
                     f"{_share([y['break'], y['held'] + y['break']])} |")
        T += ["", f"Primary histories of the {p['coast_reaching']} coast-reaching references: sustained carriage {pr['sustained']}, no history {pr['no_history']}, others {pr['others']}. "
                  "Of the others the primary lacks " + ", ".join(f"{k.replace('_', ' ') if k != 'no_long_gap' else 'continuity (a long gap)'} {v}" for k, v in pr["lacking"].items()) + ".", "",
              "| last accompanied observation of the primary | all primaries | primaries lacking the end, without sustained carriage |", "| --- | --- | --- |"]
        for s in STOPS:
            T.append(f"| {s.replace('_', ' ')} | {pr['stops'][s]} | {pr['stops_lacking_the_end'][s]} |")
        T += ["", f"Breaks: {b['count']} on {b['references_with_a_break']} references.", "",
              "| at the reference's next observation | coastal | earlier | holders ended | a holder continued away | supported | partial | unsupported |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for n in NEXT:
            T.append(f"| {n.replace('_', ' ')} | {b['next']['coastal'][n]} | {b['next']['earlier'][n]} | {b['holders_did_by_next'][n]['ended']} | "
                     f"{b['holders_did_by_next'][n]['continued_away']} | " + " | ".join(str(b["support_by_next"][n][u]) for u in SUPPORT) + " |")
        T += ["", f"Away holders: marginal (within {MARGINAL_KM:g} km) on {b['away_marginal']} breaks, away by smoothing alone on {b['away_by_smoothing_alone']}. "
                  "Candidate not stored: " + ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in b["not_stored"].items()) + ".", "",
              "| main holder's 6-hour attempt, nearest final candidate within 500 km | " + " | ".join(n.replace("_", " ") for n in NEXT) + " | all | coastal | earlier |",
              "| --- |" + " --- |" * (len(NEXT) + 3)]
        for k in ATTEMPTS:
            T.append(f"| {k.replace('_', ' ')} | " + " | ".join(str(a["first_by_next"][n][k]) for n in NEXT) + f" | {a['first'][k]} | {a['first_by_segment']['coastal'][k]} | "
                     f"{a['first_by_segment']['earlier'][k]} |")
        o = a["first_outside"]
        T += ["", f"Outside the search area at the 6-hour attempt: {o['count']}, of which a trough point lies inside the polygon inflated to the pass's minimum area for "
                  f"{o['inside_if_inflated']}, the polygon as read at or below that minimum for {o['polygon_at_or_below_minimum']}, median distance from the prediction "
                  f"{o['median_prediction_km']} km, median polygon area {o['median_polygon_area']} square degrees. 12-hour attempts read: {a['second_read']}, "
                  + ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in a["second"].items()) + f". Breaks with a final candidate within 500 km at either attempt: "
                  f"{a['candidate_near_at_either_attempt']} of {b['count']}.", ""]
    T += ["## Per season", ""]
    for y, cfgs in rec["seasons"].items():
        for name, p in cfgs.items():
            x = p["transitions"]
            T.append(f"- {y} {name}: coastal " + _share([x["coastal"]["break"], x["coastal"]["held"] + x["coastal"]["break"]]) + ", earlier "
                     + _share([x["earlier"]["break"], x["earlier"]["held"] + x["earlier"]["break"]]) + f"; breaks {p['breaks']['count']}; "
                     + "next " + ", ".join(f"{n.replace('_', ' ')} {p['breaks']['next']['coastal'][n] + p['breaks']['next']['earlier'][n]}" for n in NEXT))
    T.append("")
    return "\n".join(T)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("season", "combine"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--year", type=int)
    a = ap.parse_args(argv)
    stamp = {"script": "scripts/continuity_break_reading.py", "sha256": SCRIPT_SHA256_AT_LOAD, "lifecycle_sha256": L.SCRIPT_SHA256_AT_LOAD,
             "placement_sha256": P.SCRIPT_SHA256_AT_LOAD, "declared": "CONTINUITY_ACROSS_AFRICA_2026-11-09.md section 27"}
    if a.mode == "season":
        p = os.path.join(a.out_dir, f"{STEM}_{a.year}.json.gz")
        if os.path.exists(p):
            raise SystemExit(f"REFUSED: {p} exists and records are never overwritten")
        rec = {"generated_by": stamp, **run_season(a.year)}
        with gzip.open(p, "xt", encoding="utf-8") as fh:
            json.dump(rec, fh)
        print(json.dumps({n: {"steps_staged": c["steps_staged"], "attempts_read": c["attempts_read"]} for n, c in rec["configs"].items()}))
        return 0
    pj, pm = (os.path.join(a.out_dir, f"{STEM}.{e}") for e in ("json", "md"))
    for p in (pj, pm):
        if os.path.exists(p):
            raise SystemExit(f"REFUSED: {p} exists and records are never overwritten")
    seasons, digests = [], {}
    for y in YEARS:
        path = os.path.join(a.out_dir, f"{STEM}_{y}.json.gz")
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            r = json.load(fh)
        if r["generated_by"]["sha256"] != SCRIPT_SHA256_AT_LOAD or r["year"] != y:
            raise SystemExit(f"REFUSED: the {y} season record was not written by the current script")
        seasons.append(r); digests[str(y)] = C.sha256_of(path)
    rec = {"generated_by": {**stamp, "seasons": digests},
           "seasons": {str(s["year"]): {n: tally(c["references"]) for n, c in s["configs"].items()} for s in seasons},
           "pooled": {n: tally([r for s in seasons for r in s["configs"][n]["references"]]) for n in CONFIGS}}
    with open(pj, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    text = tables_text(rec)
    with open(pm, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
