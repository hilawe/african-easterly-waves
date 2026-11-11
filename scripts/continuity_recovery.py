#!/usr/bin/env python3
"""The recovery rule of the continuity effort, declared in section 20 of
`CONTINUITY_ACROSS_AFRICA_2026-11-09.md` before any season was replayed with it, and its one
narrowing with the repaired join reading, declared in section 22 before that replay.

RECOVER gives a track whose 6-hour and 12-hour attempts are spent two later attempts, at 18 and at
24 hours after its last observation. A later attempt is built as production builds its own, from
the track's last trough advected by the median wind it had at that observation, with the minimum
search area extended by production's own increment (125 square degrees at 18 hours and 150 at 24,
after 75 and 100). It may take only a candidate no production attempt claimed at that step, one
that would otherwise seed a new track, and each such candidate serves one later attempt. Tracks are
taken in list order, the 18-hour attempts before the 24-hour ones, each choosing the candidate
nearest its prediction. A short track is held one day instead of half a day so that it reaches its
24-hour attempt. Nothing else in the loop differs from production.

The rule runs as a hook of `continuity_lifecycle.registered_loop`, so every history is registered
and every link carries the prediction it was matched under, a join included. A season replays the
released configuration twice. The CONTROL runs the hook with no later attempt and must reproduce
the stored campaign tracks and the histories of the lifecycle census. RECOVER is then scored by
the same census on the fixed reference population, and each JOIN (a link made by a later attempt)
is read against QTrack as any link is. `combine` applies section 20's rule.

NEAR (section 22) is RECOVER with one more condition on a later attempt, that the candidate lie
within 500 km of the prediction, so that no join is a jump link. Its joins are also read FROM THE
EARLIER SIDE. Among the links whose earlier observation has a QTrack system within 500 km, a link
is kept when one of those systems is beside the later observation too, left when none is and one
still has a position, ended when all have ended and undetermined when one is alive without a
position, and a left link is supported when it was claimed 500 km or more from its prediction
while a system it left lies within 500 km of that prediction. The released configuration's own
12-hour links, read the same way, are the baseline. RECOVER's records were written by this script
at digest 1175a9be, before NEAR was added, and its path through the hook is unchanged.

    python3 scripts/continuity_recovery.py season --year 1990 --out-dir DIR [--rule near]
    python3 scripts/continuity_recovery.py combine --out-dir DIR [--rule near]
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
import continuity_join_test as J  # noqa: E402
import continuity_lifecycle as L  # noqa: E402
import continuity_stage_sample as S  # noqa: E402
import continuity_variant_replay as R  # noqa: E402

with open(os.path.abspath(__file__), "rb") as _fh:
    SCRIPT_SHA256_AT_LOAD = hashlib.sha256(_fh.read()).hexdigest()

HOURS = (18, 24)
AREAS = {18: 125.0, 24: 150.0}                       # production's 75 and 100 square degrees at 6 and 12 hours, extended by its own step of 25
RECENT_DAYS = 1.0                                    # a short track is held until its 24-hour attempt
LIFECYCLE_DIR = L.JOIN_DIR
BENEFIT_CARRIED = 2
BENEFIT_ONE_IN = 20                                  # or 5 % of the reference observations on the longest single-history accompaniment
LOSS_ONE_IN = 100                                    # the accompanied observations may not fall by more than 1 %
MIN_ASSESSED = 20                                    # fewer assessed joins than this and the join conditions are unscored
CONSISTENT_AT_LEAST = (4, 5)                         # of the assessed joins
FLAGGED_AT_MOST = (1, 20)
SUPPORTED_AT_MOST = (1, 40)
JOIN_STATUS = E.LINK_STATUS
RULES = {"recover": {}, "near": {"max_km": C.RADIUS_KM}}
DECLARED = {"recover": "CONTINUITY_ACROSS_AFRICA_2026-11-09.md section 20", "near": "CONTINUITY_ACROSS_AFRICA_2026-11-09.md section 22"}
STEM = {"recover": "continuity_recovery", "near": "continuity_recovery_near"}
EARLIER = ("kept", "left", "ended", "undetermined")
MARGIN = (5, 4)                                      # a share may reach 1.25 times its baseline


class Recovery:
    """The later attempts, as a hook of `continuity_lifecycle.registered_loop`. With no hours it
    changes nothing, which is the control."""

    def __init__(self, hours=HOURS, areas=None, max_km=None):
        self.hours = tuple(hours)
        self.areas = dict(AREAS if areas is None else areas)
        self.max_km = max_km                         # NEAR's condition, None for RECOVER
        self.winds = {}                              # a track's median winds at its last observation
        self.joins = []

    def __call__(self, step, time, waves, winds, tracks, states, registry, order, seeded):
        from aew.v1port.association import MAX_SPEED_MS, _extend, _fallback_polygon, _hull_polygon, _inflate, predict_displacement
        from aew.v1port.geometry import great_circle_distance, points_in_polygon
        um, vm = winds
        wave_of = {id(w["region"]): w for w in waves}
        for tr in tracks:
            if tr["step"][-1] == step:
                w = wave_of[id(tr["wave_points"][-1])]
                self.winds[id(tr)] = (float(um(w)), float(vm(w)))
        free = {id(t): t for t in seeded}
        slot = {id(t): k for k, t in enumerate(tracks)}
        taken, gone = {}, set()
        for hours in self.hours:
            lag = hours // 6
            for k, tr in enumerate(tracks):
                if not free:
                    break
                if tr["step"][-1] != step - lag:
                    continue
                u, v = self.winds[id(tr)]
                lat0, lon0 = float(tr["meanlat"][-1]), float(tr["meanlon"][-1])
                dlat, dlon = predict_displacement(u, v, lat0, float(hours))
                hull = _hull_polygon(tr["lon_wave"][-1], tr["lat_wave"][-1])
                plons, plats = _fallback_polygon(tr, dlat, dlon) if hull is None else (hull[0] + dlon, hull[1] + dlat)
                plons, plats = _inflate(plons, plats, self.areas[hours])
                best = None
                for sid, s in free.items():
                    w = wave_of[id(s["wave_points"][0])]
                    if not points_in_polygon(plons, plats, w["lon_wave"], w["lat_wave"]).any():
                        continue
                    travelled_km = float(great_circle_distance(w["lat_mean"], w["lon_mean"], lat0, lon0, "km"))
                    if not (travelled_km * 1000.0) / (hours * 3600.0) < MAX_SPEED_MS:
                        continue
                    if self.max_km is not None and not float(C.km(round(lat0 + dlat, 3), round(lon0 + dlon, 3), w["lat_mean"], w["lon_mean"])) < self.max_km:
                        continue                                 # measured from the prediction as it is recorded, as a jump link is
                    d = float(great_circle_distance(lat0 + dlat, lon0 + dlon, w["lat_mean"], w["lon_mean"], "km"))
                    if best is None or d < best[0]:
                        best = (d, sid)
                if best is None:
                    continue
                d, sid = best
                s = free.pop(sid)
                _extend(tr, wave_of[id(s["wave_points"][0])], step)
                registry[id(tr)]["est"].append([round(lat0 + dlat, 3), round(lon0 + dlon, 3), hours])
                row = registry.pop(sid)
                for i in range(len(order) - 1, -1, -1):          # the seed was registered this step, so it sits near the end
                    if order[i] is row:
                        del order[i]
                        break
                self.winds[id(tr)] = self.winds.pop(sid)
                self.joins.append({"track": registry[id(tr)]["birth"], "seed": row["birth"], "step": int(step), "hours": int(hours), "km_from_prediction": round(d, 1)})
                taken[slot[sid]] = tr
                gone.add(k)
        if not taken:
            return tracks, states
        new_tracks, new_states = [], []
        for k, (tr, st) in enumerate(zip(tracks, states)):
            if k in gone:
                continue
            new_tracks.append(taken.get(k, tr))                  # the older track takes the seed's place and the seed's state
            new_states.append(st)
        return new_tracks, new_states


def join_marks(history, j, q_index, q_spans):
    """One join read against QTrack: its status as a link, and whether it is a SUPPORTED FALSE JOIN,
    a switch claimed 500 km or more from its prediction while a system left behind lies within
    500 km of that prediction."""
    status = E.link_status(history, j, q_index, q_spans)
    supported = False
    if status == "switch":
        lost = sorted({tid for tid, _ in C.companions(history["lat"][j - 1], history["lon"][j - 1], history["time"][j - 1], q_index)})
        supported = E.link_marks({"from": j - 1, "to": j, "lost": lost}, history, q_spans)["supported"]
    return status, supported


def join_census(histories, q_index, q_spans, hours=HOURS):
    """The joins inside the given histories that lie in the sector, by status."""
    out = {"joins": 0, **{k: 0 for k in JOIN_STATUS}, "supported_false_joins": 0, "jump_joins": 0, "by_hours": {str(h): 0 for h in hours}}
    for h in R.in_sector_tracks(histories):
        for j in range(1, len(h["est"]) + 1):
            hrs = int(h["est"][j - 1][2])
            if hrs not in hours:
                continue
            status, supported = join_marks(h, j, q_index, q_spans)
            out["joins"] += 1
            out[status] += 1
            out["supported_false_joins"] += int(supported)
            out["jump_joins"] += int(E.jump(h, j))
            out["by_hours"][str(hrs)] += 1
    return out


def earlier_side(history, j, q_index, q_spans):
    """The link into observation j read from its earlier side: None when no QTrack system is
    beside the earlier observation, otherwise one of `EARLIER` and whether a left link is supported."""
    a = {tid for tid, _ in C.companions(history["lat"][j - 1], history["lon"][j - 1], history["time"][j - 1], q_index)}
    if not a:
        return None, False
    b = {tid for tid, _ in C.companions(history["lat"][j], history["lon"][j], history["time"][j], q_index)}
    if a & b:
        return "kept", False
    d = float(history["time"][j])
    present = sorted(x for x in a if C.time_key(d) in q_spans[x]["positions"])
    if present:
        return "left", E.link_marks({"from": j - 1, "to": j, "lost": present}, history, q_spans)["supported"]
    return ("undetermined" if any(q_spans[x]["last"] >= d - J.TOL for x in a) else "ended"), False


def earlier_side_census(histories, q_index, q_spans, hours):
    """The links of the given prediction hours inside the histories in the sector, read from
    their earlier side."""
    out = {"all": 0, "jumps": 0, "links": 0, **{k: 0 for k in EARLIER}, "supported": 0}
    for h in R.in_sector_tracks(histories):
        for j in range(1, len(h["est"]) + 1):
            if int(h["est"][j - 1][2]) not in hours:
                continue
            out["all"] += 1
            out["jumps"] += int(E.jump(h, j))
            status, supported = earlier_side(h, j, q_index, q_spans)
            if status is not None:
                out["links"] += 1
                out[status] += 1
                out["supported"] += int(supported)
    return out


def far_joins(rows, max_km, hours=HOURS):
    """The joins, over every registered history, claimed `max_km` or more from their recorded prediction."""
    return sum(1 for r in rows for j, e in enumerate(r["est"], start=1)
               if int(e[2]) in hours and not float(C.km(e[0], e[1], r["lat"][j], r["lon"][j])) < max_km)


def check_near(year, rows, max_km):
    """Under NEAR no join, in any registered history, may lie `max_km` or more from its prediction."""
    if max_km is not None and far_joins(rows, max_km):
        raise SystemExit(f"REFUSED: the {year} NEAR replay made a join {max_km:g} km or more from its prediction")


def earlier_side_references(lifecycle_dir, years=S.YEARS):
    """Production's own links and RECOVER's joins read from the earlier side, from the histories
    the lifecycle census and the RECOVER replay retained."""
    import qtrack_pairing_pilot as M
    out, inputs = {}, {}
    for y in years:
        life_path = os.path.join(lifecycle_dir, f"continuity_lifecycle_{y}.json.gz")
        with gzip.open(life_path, "rt", encoding="utf-8") as fh:
            life = json.load(fh)
        g = life["generated_by"]
        if g["sha256"] != L.SCRIPT_SHA256_AT_LOAD or g["evaluation_sha256"] != E.SCRIPT_SHA256_AT_LOAD:
            raise SystemExit(f"REFUSED: the {y} lifecycle record was not written by the current scripts")
        if C.sha256_of(life["artifact"]["path"]) != life["artifact"]["sha256"]:
            raise SystemExit(f"REFUSED: {life['artifact']['path']} is not the file the {y} lifecycle record names")
        art = json.load(open(life["artifact"]["path"], encoding="utf-8"))
        if C.sha256_of(art["inputs"]["qtrack"]["path"]) != art["inputs"]["qtrack"]["sha256"]:
            raise SystemExit(f"REFUSED: {art['inputs']['qtrack']['path']} is not the file the {y} artifact names")
        q_tracks = M.load_qtrack(art["inputs"]["qtrack"]["path"])[0]
        q_index, q_spans = C.index_by_time(q_tracks), J.spans(q_tracks)
        rec_path = os.path.join(lifecycle_dir, f"{STEM['recover']}_{y}.json.gz")
        with gzip.open(rec_path, "rt", encoding="utf-8") as fh:
            rec = json.load(fh)
        if rec["generated_by"]["declared"] != DECLARED["recover"] or rec["lifecycle"]["sha256"] != C.sha256_of(life_path):
            raise SystemExit(f"REFUSED: the {y} RECOVER record is not section 20's replay over this lifecycle record")
        stored = L.history_sets(life["configs"]["released"]["rows"])["STORED"]
        recovered = L.history_sets(rec["recover"]["rows"])["STORED"]
        parts = {"released_6_hour_links": earlier_side_census(stored, q_index, q_spans, (6,)),
                 "released_12_hour_links": earlier_side_census(stored, q_index, q_spans, (12,)),
                 "recover_joins": earlier_side_census(recovered, q_index, q_spans, HOURS)}
        counted = rec["recover"]["joins_in_finished_sector_tracks"]
        if (parts["recover_joins"]["all"], parts["recover_joins"]["jumps"]) != (counted["joins"], counted["jump_joins"]):
            raise SystemExit(f"REFUSED: the {y} RECOVER histories do not hold the joins its record counted")
        for k, v in parts.items():
            out[k] = L._add(out[k], v) if k in out else v
        inputs[str(y)] = {"lifecycle": C.sha256_of(life_path), "recover": C.sha256_of(rec_path)}
    return out, inputs


def load_season(year, campaign=S.CAMPAIGN, evidence=S.EVIDENCE):
    """The campaign case and its thresholds, as the earlier stages load them."""
    import pilot_margin_replay as MR
    import pilot_stage_diagnostic as SD
    import pilot_track_crosswalk as P
    run_dir = os.path.join(campaign, f"era5_{year}")
    run = P.load_run(run_dir, year=year)
    ev = json.load(open(os.path.join(evidence, f"era5_{year}", f"tracking_era5_{year}.json"), encoding="utf-8"))
    if run["tracks_sha256"] != ev["dataset_specific"]["tracks_sha256"]:
        raise SystemExit(f"REFUSED: the {year} campaign run is not the one the evidence records")
    case = SD.load_case(os.path.join(run_dir, "tracker_case.mat"), ev)
    ds = ev["dataset_specific"]
    return run, case, float(ds["coarse_threshold"]), float(ds["fine_threshold"]), ds["tracker_flags"], MR.control_grids(case)


def check_control(year, finished, rows, run, lifecycle):
    """The hook with no later attempt must be production: the stored campaign tracks, and the
    histories the lifecycle census registered."""
    import pilot_margin_replay as MR
    if not MR.archive_gate(finished, run)["passed"]:
        raise SystemExit(f"REFUSED: the {year} control does not reproduce the stored campaign tracks")
    if [r for r in rows if L.in_box(r)] != lifecycle["configs"]["released"]["rows"]:
        raise SystemExit(f"REFUSED: the {year} control's histories are not those of the lifecycle census")


def run_season(year, lifecycle_dir=LIFECYCLE_DIR, loop=L.lifecycle_loop, loader=load_season, rule="recover"):
    if year not in S.YEARS:
        raise SystemExit(f"REFUSED: {year} is not a development season of section 8")
    import coast_crossing_measurement as CC
    import qtrack_pairing_pilot as M
    import season_metrics as SM
    life_path = os.path.join(lifecycle_dir, f"continuity_lifecycle_{year}.json.gz")
    with gzip.open(life_path, "rt", encoding="utf-8") as fh:
        life = json.load(fh)
    g = life["generated_by"]
    if g["sha256"] != L.SCRIPT_SHA256_AT_LOAD or g["evaluation_sha256"] != E.SCRIPT_SHA256_AT_LOAD:
        raise SystemExit(f"REFUSED: the {year} lifecycle record was not written by the current scripts")
    for key in ("artifact", "sample", "join_record"):
        if C.sha256_of(life[key]["path"]) != life[key]["sha256"]:
            raise SystemExit(f"REFUSED: {life[key]['path']} is not the file the {year} lifecycle record names")
    run, case, ct, ft, flags, grids = loader(year)
    if case["sha256"] != life["case_sha256"]:
        raise SystemExit(f"REFUSED: the {year} case is not the one the lifecycle census read")
    art = json.load(open(life["artifact"]["path"], encoding="utf-8"))
    q_tracks = M.load_qtrack(art["inputs"]["qtrack"]["path"])[0]
    q = {t["id"]: t for t in q_tracks}
    q_index, q_spans, q_rows = C.index_by_time(q_tracks), J.spans(q_tracks), art["sides"]["qtrack"]["tracks"]
    regions, _ = SM.load_regions(art["inputs"]["region_polygons"]["dir"])
    sample = json.load(open(life["sample"]["path"], encoding="utf-8"))
    sample_season = next(s for s in sample["seasons"] if s["year"] == year)
    times = [float(x) for x in np.asarray(case["time"], float).ravel()]
    finished0, rows0, checks0 = loop(case, grids, ct, ft, flags, {}, hook=Recovery(hours=()))
    check_control(year, finished0, rows0, run, life)
    hook = Recovery(**RULES[rule])
    finished, rows, checks = loop(case, grids, ct, ft, flags, {}, hook=hook, recent_days=RECENT_DAYS)
    boxed = [r for r in rows if L.in_box(r)]
    census, sets = L.season_census(boxed, q_tracks, q_rows, sample_season, None, times)
    L.check_stored(f"{year} RECOVER", sets, boxed, finished)
    tracks = R.as_tracks(finished)
    membership = CC.groups_ours(tracks)
    side = CC.measure(tracks, year, regions, M._day(year, 1, 1), membership)
    counts = {"tracks": len(tracks), **R.correspondence(tracks, side["tracks"], q, q_rows), **R.side_effects(tracks, q_index, membership)}
    ref = census["STORED"]["reference"]
    if (ref["accompanied"], ref["steps"], ref["classes"]) != (counts["accompanied_steps"], counts["qtrack_steps"], counts["qtrack_classes"]):
        raise SystemExit(f"REFUSED: the {year} RECOVER census and the whole-domain correspondence disagree")
    with gzip.open(life["join_record"]["path"], "rt", encoding="utf-8") as fh:
        released_counts = json.load(fh)["configs"]["released"]["counts"]
    by_hours = {str(h): sum(1 for x in hook.joins if x["hours"] == h) for h in HOURS}
    if len(rows0) - len(rows) != len(hook.joins):
        raise SystemExit(f"REFUSED: the {year} RECOVER replay lost or gained histories beyond its joins")
    in_tracks = join_census(sets["STORED"], q_index, q_spans)
    check_near(year, rows, hook.max_km)
    return {"year": year, "rule": rule, "case_sha256": case["sha256"], "lifecycle": {"path": life_path, "sha256": C.sha256_of(life_path)},
            "control": {"reproduces_stored_tracks_and_lifecycle_histories": True, "histories": len(rows0), "prune_checks": checks0},
            "released_counts": released_counts,
            "recover": {"fates": {k: sum(1 for r in rows if r["fate"] == k) for k in L.FATES}, "prune_checks": checks, "counts": counts, "census": census,
                        "joins_made": {"all": len(hook.joins), "by_hours": by_hours},
                        "joins_in_finished_sector_tracks": in_tracks,
                        "joins_from_the_earlier_side": earlier_side_census(sets["STORED"], q_index, q_spans, HOURS),
                        "joins": hook.joins, "rows": boxed}}


COUNT_KEYS = ("tracks", "tracks_in_sector", "multi_system_tracks", "stationary_tracks", "duplicate_groups", "aewc_cohort", "aewc_atlantic_side")


def pool(seasons, lifecycle_pooled):
    rc = [s["recover"] for s in seasons]
    census = rc[0]["census"]
    for c in rc[1:]:
        census = {k: (L._add(census[k], c["census"][k]) if k != "keep_cost" else census[k]) for k in census}
    census.pop("keep_cost", None)
    joins = rc[0]["joins_in_finished_sector_tracks"]
    for c in rc[1:]:
        joins = L._add(joins, c["joins_in_finished_sector_tracks"])
    made = rc[0]["joins_made"]
    for c in rc[1:]:
        made = L._add(made, c["joins_made"])
    early = rc[0].get("joins_from_the_earlier_side")
    for c in rc[1:]:
        early = L._add(early, c["joins_from_the_earlier_side"]) if early is not None else None
    return {"released": {"reference": lifecycle_pooled["released"]["STORED"]["reference"], "flags": lifecycle_pooled["released"]["STORED"]["flags"],
                         "costs": lifecycle_pooled["released"]["STORED"]["costs"], "link_status": lifecycle_pooled["released"]["STORED"]["link_status"],
                         "counts": {k: sum(s["released_counts"][k] for s in seasons) for k in COUNT_KEYS}},
            "recover": {"reference": census["STORED"]["reference"], "flags": census["STORED"]["flags"], "costs": census["STORED"]["costs"],
                        "link_status": census["STORED"]["link_status"], "counts": {k: sum(c["counts"][k] for c in rc) for k in COUNT_KEYS},
                        "fates": {k: sum(c["fates"][k] for c in rc) for k in L.FATES}, "joins_made": made, "joins_in_finished_sector_tracks": joins,
                        **({"joins_from_the_earlier_side": early} if early is not None else {}),
                        "sample": census["sample"], "breaks": census["breaks"]}}


def decide(pooled):
    """Section 20's rule, every comparison in whole numbers."""
    base, rec = pooled["released"]["reference"], pooled["recover"]["reference"]
    if base["steps"] != rec["steps"] or base["tracks"] != rec["tracks"]:
        raise SystemExit("REFUSED: the two configurations were scored over different reference observations")
    steps = base["steps"]
    j = pooled["recover"]["joins_in_finished_sector_tracks"]
    assessed = j["consistent"] + j["switch"] + j["relay"] + j["undetermined"]
    if assessed + j["unassessed"] != j["joins"]:
        raise SystemExit("REFUSED: the join statuses do not sum to the joins")
    flagged = j["switch"] + j["undetermined"]
    carried_rise = rec["classes"]["carried"] - base["classes"]["carried"]
    longest_rise = rec["longest_single_history"] - base["longest_single_history"]
    accompanied_change = rec["accompanied"] - base["accompanied"]
    scored = assessed >= MIN_ASSESSED
    checks = {"benefit": bool(carried_rise >= BENEFIT_CARRIED or longest_rise * BENEFIT_ONE_IN >= steps),
              "no_loss_of_accompaniment": bool(accompanied_change * LOSS_ONE_IN >= -steps),
              "joins_scored": bool(scored),
              "joins_consistent": bool(scored and j["consistent"] * CONSISTENT_AT_LEAST[1] >= CONSISTENT_AT_LEAST[0] * assessed),
              "joins_not_flagged": bool(scored and flagged * FLAGGED_AT_MOST[1] <= FLAGGED_AT_MOST[0] * assessed),
              "joins_not_supported_false": bool(scored and j["supported_false_joins"] * SUPPORTED_AT_MOST[1] <= SUPPORTED_AT_MOST[0] * assessed)}
    return {"carried_rise": carried_rise, "longest_rise": longest_rise, "accompanied_change": accompanied_change, "assessed_joins": assessed,
            "flagged_joins": flagged, "checks": checks, "carried_to_freezing": all(checks.values())}


def decide_near(pooled):
    """Section 22's rule: section 20's six conditions and four on the joins read from the earlier
    side, against the released configuration's 12-hour links, in whole numbers."""
    d = decide(pooled)
    e, b = pooled["earlier_side"]["near_joins"], pooled["earlier_side"]["released_12_hour_links"]
    for x in (e, b):
        if sum(x[k] for k in EARLIER) != x["links"]:
            raise SystemExit("REFUSED: the earlier-side statuses do not sum to their links")
    scored = e["links"] >= MIN_ASSESSED and b["links"] > 0
    flagged, base = e["left"] + e["undetermined"], b["left"] + b["undetermined"]
    more = {"earlier_side_scored": bool(scored),
            "left_not_above_production": bool(scored and flagged * b["links"] * MARGIN[1] <= MARGIN[0] * base * e["links"]),
            "supported_not_above_production": bool(scored and e["supported"] * b["links"] * MARGIN[1] <= MARGIN[0] * b["supported"] * e["links"]),
            "no_join_is_a_jump": bool(e["jumps"] == 0 and pooled["recover"]["joins_in_finished_sector_tracks"]["jump_joins"] == 0)}
    checks = {**d["checks"], **more}
    return {**d, "earlier_side_joins": e["links"], "left_or_undetermined_joins": flagged, "checks": checks, "carried_to_freezing": all(checks.values())}


def earlier_side_table(sides):
    names = {"released_6_hour_links": "released, 6-hour links", "released_12_hour_links": "released, 12-hour links", "recover_joins": "RECOVER, joins", "near_joins": "NEAR, joins"}
    out = ["", "| links in finished tracks in the sector, read from the earlier side | all | claimed 500 km or more from the prediction | with a QTrack system beside the earlier side | "
               "kept | left | ended | undetermined | supported |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for k, label in names.items():
        if k in sides:
            v = sides[k]
            out.append(f"| {label} | {v['all']} | {v['jumps']} | {v['links']} | {v['kept']} | {v['left']} | {v['ended']} | {v['undetermined']} | {v['supported']} |")
    return out


def tables_text(rec, rule="recover"):
    P_, D = rec["pooled"], rec["decision"]
    label = rule.upper()
    L_ = [f"# Recovery at 18 and 24 hours, {label} (declared {DECLARED[rule].split()[-2]} {DECLARED[rule].split()[-1]})", "",
          "1990 and 2006 pooled, finished tracks of each configuration. The reference population is the QTrack cohort tracks starting at 10 to 30 E.", "",
          "| configuration | reference observations accompanied | carried | relayed | picked up late | dropped | unaccompanied at both | longest single-history accompaniment | "
          "reference tracks accompanied | histories per accompanied reference track |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name in ("released", "recover"):
        r = P_[name]["reference"]; c = r["classes"]
        per = r["histories_on_accompanied"] / r["tracks_accompanied"] if r["tracks_accompanied"] else 0.0
        L_.append(f"| {name if name == 'released' else label} | {r['accompanied']} of {r['steps']} ({100.0 * r['accompanied'] / r['steps']:.1f} %) | {c['carried']} | {c['relayed']} | {c['picked_up_late']} | "
                  f"{c['dropped']} | {c['unaccompanied_at_both']} | {r['longest_single_history']} | {r['tracks_accompanied']} of {r['tracks']} | {per:.2f} |")
    L_ += ["", "| configuration | tracks | in sector | AEWC v2 cohort reaching the Atlantic side | stationary | duplicate groups | flagged tracks | switch events | relay events | "
               "exposure pairs | QTrack systems left behind | reference tracks with a flagged history | supported false-link candidates | links | jump links |",
           "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name in ("released", "recover"):
        c, f, k = P_[name]["counts"], P_[name]["flags"], P_[name]["costs"]
        L_.append(f"| {name if name == 'released' else label} | {c['tracks']} | {c['tracks_in_sector']} | {c['aewc_atlantic_side']} of {c['aewc_cohort']} | {c['stationary_tracks']} | {c['duplicate_groups']} | "
                  f"{f['flagged_tracks']} | {f['events']['switch']} | {f['events']['relay']} | {f['exposure_pairs']} | {f['systems_left_behind']} | "
                  f"{f['reference_tracks_with_a_flagged_history']} | {f['supported_false_link_candidates']} | {k['links']} | {k['jump_links']} |")
    L_ += ["", "| links in the sector, by hours of the prediction | consistent | switch | relay | undetermined | unassessed |", "| --- | --- | --- | --- | --- | --- |"]
    for name in ("released", "recover"):
        for hrs, v in P_[name]["link_status"].items():
            L_.append(f"| {name if name == 'released' else label}, {hrs} | {v['consistent']} | {v['switch']} | {v['relay']} | {v['undetermined']} | {v['unassessed']} |")
    j, m = P_["recover"]["joins_in_finished_sector_tracks"], P_["recover"]["joins_made"]
    L_ += ["", f"Joins made over the whole domain: {m['all']} ({', '.join(f'{n} at {h} hours' for h, n in m['by_hours'].items())}). In finished tracks in the sector: {j['joins']} "
               f"({', '.join(f'{n} at {h} hours' for h, n in j['by_hours'].items())}), of which {j['jump_joins']} were claimed 500 km or more from their prediction. "
               f"Read against QTrack: consistent {j['consistent']}, switch {j['switch']}, relay {j['relay']}, undetermined {j['undetermined']}, unassessed {j['unassessed']}, "
               f"supported false joins {j['supported_false_joins']}.",
           f"Histories by fate under {label}, whole domain: " + ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in P_["recover"]["fates"].items()) + ".", "",
           "Per season (accompanied, carried, longest; joins in finished sector tracks as consistent, switch, relay, undetermined, unassessed):", ""]
    for y, s in rec["seasons"].items():
        r, jj = s["reference"], s["joins_in_finished_sector_tracks"]
        L_.append(f"- {y}: {r['accompanied']} of {r['steps']}, {r['classes']['carried']}, {r['longest_single_history']}; "
                  f"{jj['consistent']}, {jj['switch']}, {jj['relay']}, {jj['undetermined']}, {jj['unassessed']}")
    if "earlier_side" in P_:
        L_ += earlier_side_table(P_["earlier_side"])
    L_ += ["", f"Carried rise {D['carried_rise']}, longest single-history rise {D['longest_rise']}, change in accompanied observations {D['accompanied_change']}, "
               f"assessed joins {D['assessed_joins']}, flagged joins {D['flagged_joins']}.",
           "Checks: " + ", ".join(f"{k.replace('_', ' ')} {'met' if v else 'not met'}" for k, v in D["checks"].items()) + ".",
           f"Carried to freezing: {'yes' if D['carried_to_freezing'] else 'no'}.", ""]
    return "\n".join(L_)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("season", "combine"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--lifecycle-dir", default=LIFECYCLE_DIR)
    ap.add_argument("--rule", choices=tuple(RULES), default="recover")
    ap.add_argument("--year", type=int)
    a = ap.parse_args(argv)
    stem = STEM[a.rule]
    stamp = {"script": "scripts/continuity_recovery.py", "sha256": SCRIPT_SHA256_AT_LOAD, "lifecycle_sha256": L.SCRIPT_SHA256_AT_LOAD, "evaluation_sha256": E.SCRIPT_SHA256_AT_LOAD}
    if a.mode == "season":
        p = os.path.join(a.out_dir, f"{stem}_{a.year}.json.gz")
        if os.path.exists(p):
            raise SystemExit(f"REFUSED: {p} exists and records are never overwritten")
        rec = {"generated_by": {**stamp, "declared": DECLARED[a.rule], "rule": a.rule, "hours": list(HOURS), "areas": {str(k): v for k, v in AREAS.items()},
                                "recent_days": RECENT_DAYS, **RULES[a.rule]}, **run_season(a.year, lifecycle_dir=a.lifecycle_dir, rule=a.rule)}
        with gzip.open(p, "xt", encoding="utf-8") as fh:
            json.dump(rec, fh)
        print(json.dumps({k: v for k, v in rec["recover"].items() if k not in ("rows", "joins")}, indent=1))
        return 0
    pj, pm = (os.path.join(a.out_dir, f"{stem}.{e}") for e in ("json", "md"))
    for p in (pj, pm):
        if os.path.exists(p):
            raise SystemExit(f"REFUSED: {p} exists and records are never overwritten")
    life_path = os.path.join(a.lifecycle_dir, "continuity_lifecycle.json")
    life = json.load(open(life_path, encoding="utf-8"))
    if life["generated_by"]["sha256"] != L.SCRIPT_SHA256_AT_LOAD or life["generated_by"]["evaluation_sha256"] != E.SCRIPT_SHA256_AT_LOAD:
        raise SystemExit("REFUSED: the combined lifecycle record was not written by the current scripts")
    seasons, digests = [], {}
    for y in S.YEARS:
        path = os.path.join(a.out_dir, f"{stem}_{y}.json.gz")
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            r = json.load(fh)
        if any(r["generated_by"][k] != stamp[k] for k in ("sha256", "lifecycle_sha256", "evaluation_sha256")) or r["generated_by"].get("rule") != a.rule:
            raise SystemExit(f"REFUSED: the {y} season record was not written by the current scripts under this rule")
        if r["lifecycle"]["sha256"] != life["generated_by"]["seasons"][str(y)]:
            raise SystemExit(f"REFUSED: the {y} season record and the combined lifecycle record read different lifecycle records")
        seasons.append(r); digests[str(y)] = C.sha256_of(path)
    pooled = pool(seasons, life["pooled"])
    generated = {**stamp, "rule": a.rule, "declared": DECLARED[a.rule], "seasons": digests, "lifecycle": {"path": life_path, "sha256": C.sha256_of(life_path)}}
    if a.rule == "near":
        references, inputs = earlier_side_references(a.lifecycle_dir)
        pooled["earlier_side"] = {**references, "near_joins": pooled["recover"]["joins_from_the_earlier_side"]}
        generated["earlier_side_inputs"] = inputs
    rec = {"generated_by": generated,
           "seasons": {str(s["year"]): {"reference": s["recover"]["census"]["STORED"]["reference"], "counts": s["recover"]["counts"], "fates": s["recover"]["fates"],
                                        "joins_made": s["recover"]["joins_made"], "joins_in_finished_sector_tracks": s["recover"]["joins_in_finished_sector_tracks"],
                                        "control": s["control"]} for s in seasons},
           "pooled": pooled, "decision": decide_near(pooled) if a.rule == "near" else decide(pooled)}
    with open(pj, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    text = tables_text(rec, a.rule)
    with open(pm, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
