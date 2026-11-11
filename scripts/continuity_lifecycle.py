#!/usr/bin/env python3
"""The lifecycle census of the continuity effort, declared in section 18 of
`CONTINUITY_ACROSS_AFRICA_2026-11-09.md` before any season was replayed for it.

One development season is replayed through the production loop under the released configuration
and under VCLIP with EVERY HISTORY REGISTERED, from its seed to its fate (finished, pruned in the
loop, removed by the final speed filter, or left as a single observation), and with the prediction
each link was matched under. The replay goes through the false-join test's own bindings, so the
released loop must reproduce the stored campaign tracks, the coast-crossing rows and the census
totals, both configurations must reproduce section 14's counts, and the finished tracks must equal
the tracks section 16 retained. The loop refuses if a track is pruned with an association attempt
still due, which is what makes retention a change to the output alone.

Three sets of histories are scored. STORED is the finished tracks. KEEP adds the pruned histories
that pass production's final filter, smoothed as stored tracks are. ALL is every history the
association built, at its unsmoothed positions. The census reports, on the fixed reference
population and on the stage sample, what each set accompanies, how histories break along the
reference tracks, an optimistic limit on what chaining histories across gaps of 18 hours to a
horizon could deliver, and the repaired evaluation's flags. `combine` pools the seasons and reads
section 18's two provisional marks. Nothing here changes a tracker setting or an earlier record.

    python3 scripts/continuity_lifecycle.py season --year 1990 --out-dir DIR
    python3 scripts/continuity_lifecycle.py combine --out-dir DIR
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
import continuity_merge_variants as V  # noqa: E402
import continuity_readings as RD  # noqa: E402
import continuity_stage_sample as S  # noqa: E402
import continuity_variant_replay as R  # noqa: E402

with open(os.path.abspath(__file__), "rb") as _fh:
    SCRIPT_SHA256_AT_LOAD = hashlib.sha256(_fh.read()).hexdigest()

CONFIGS = J.CONFIGS
JOIN_DIR = J.STAGE_DIR
MARGIN_DEG = 6.0                                     # 500 km lies inside 6 degrees at every latitude of the sector
SETS = ("STORED", "KEEP", "ALL")
FATES = ("finished", "pruned_in_loop", "removed_by_final_speed_filter", "removed_as_single_observation")
BREAKS = ("concurrent", "within_horizon", "18_to_24_hours", "30_to_48_hours", "over_48_hours")
ENDS = ("candidate_present", "placed_away", "beyond_reference")
HORIZONS = (18, 24, 36, 48)
FIRST_NEW_HOURS = 18.0                               # the first gap the production association does not attempt
MAX_SPEED_MS = 25.0
BAR_HORIZON = 24
BAR_CARRIED = 2
BAR_ONE_IN = 20                                      # a rise of one reference observation in twenty, 5 %, decided in whole numbers
KEEP_ONE_IN = 20
PREDICTION_KEYS = ("est_step_6hr", "est_lat_6hr", "est_lon_6hr", "est_step_12hr", "est_lat_12hr", "est_lon_12hr")


def registered_loop(times, inputs_at, exclusive, hook=None, recent_days=None):
    """The production association, prune and final filter over `times`, with every history
    registered. `inputs_at(step)` returns that step's candidates and the two median-wind callables.
    Returns production's finished tracks, one row per history ever seeded, and the prune checks.

    `hook`, when given, is called after each step's association and registration and before the
    prune, with the step's state by keyword, and returns the track and state lists to go on with.
    `recent_days` replaces the prune's half-day clause. With neither, the loop is production's."""
    from aew.v1port.association import NO_PREDICTION, RECENT_DAYS, associate_step, finalize_tracks, prune_stale_tracks
    recent = RECENT_DAYS if recent_days is None else recent_days
    tracks, states, registry, order = [], [], {}, []
    spent = stranded = 0
    for step in range(len(times)):
        t = float(times[step])
        waves, um, vm = inputs_at(step)
        before = {id(tr): (len(tr["time"]), {k: st.get(k) for k in PREDICTION_KEYS}) for tr, st in zip(tracks, states)}
        tracks, states = associate_step(tracks, states, waves, step, um, vm, exclusive=exclusive)
        seeded = []
        for tr in tracks:
            i = id(tr)
            if i not in registry:
                cand = next((k for k, w in enumerate(waves) if w.get("region") is not None and tr["wave_points"][0] is w["region"]), None)
                if cand is None:
                    raise SystemExit(f"REFUSED: at step {step} a seeded track cannot be traced to its candidate")
                registry[i] = {"birth": f"{step}:{cand}", "track": tr, "est": [], "fate": "live"}
                order.append(registry[i])
                seeded.append(tr)
            elif len(tr["time"]) > before[i][0]:
                n0, pred = before[i]
                key = "6hr" if pred["est_step_6hr"] == step else "12hr"
                if len(tr["time"]) != n0 + 1 or pred["est_step_" + key] != step:
                    raise SystemExit(f"REFUSED: at step {step} an extension does not follow from a prediction due at that step")
                registry[i]["est"].append([round(float(pred["est_lat_" + key]), 3), round(float(pred["est_lon_" + key]), 3), 6 if key == "6hr" else 12])
        if hook is not None:
            tracks, states = hook(step=step, time=t, waves=waves, winds=(um, vm), tracks=tracks, states=states, registry=registry, order=order, seeded=seeded)
        held = {id(tr): st for tr, st in zip(tracks, states)}
        tracks, states = prune_stale_tracks(tracks, states, step, t, waves, recent_days=recent)
        kept = {id(tr) for tr in tracks}
        for i, st in held.items():
            if i in kept:
                continue
            due = [st.get("est_step_6hr"), st.get("est_step_12hr")]
            if any(d is not None and d != NO_PREDICTION and d > step for d in due):
                raise SystemExit(f"REFUSED: at step {step} a track was pruned with an association attempt still due")
            if all(d == NO_PREDICTION for d in due):
                spent += 1
            else:
                stranded += 1
            row = registry[i]
            row["fate"] = "pruned_in_loop"
            row["track"]["lat_wave"], row["track"]["lon_wave"], row["track"]["wave_points"] = [], [], []
    finished = finalize_tracks(tracks, total_steps=len(times))
    live = {id(tr["time"]): tr for tr in tracks}
    for k, f in enumerate(finished):
        row = registry[id(live[id(f["time"])])]
        row["fate"], row["finished_index"] = "finished", k
    for tr in tracks:
        row = registry[id(tr)]
        if row["fate"] == "live":
            row["fate"] = "removed_by_final_speed_filter" if len(tr["time"]) >= 2 else "removed_as_single_observation"
    rows = []
    for r in order:
        if r["fate"] == "live":
            raise SystemExit(f"REFUSED: the history {r['birth']} left the live list without a fate")
        tr = r["track"]
        row = {"birth": r["birth"], "fate": r["fate"], "step": [int(s) for s in tr["step"]], "time": [float(x) for x in tr["time"]],
               "lat": [float(x) for x in tr["meanlat"]], "lon": [float(x) for x in tr["meanlon"]], "est": r["est"]}
        if "finished_index" in r:
            row["finished_index"] = r["finished_index"]
        rows.append(row)
    return finished, rows, {"pruned_with_both_attempts_spent": spent, "pruned_with_an_attempt_stranded": stranded}


def lifecycle_loop(case, grids, ct, ft, flags, options, hook=None, recent_days=None):
    """`registered_loop` on a campaign case with the coarse merge's `options`, the finished tracks
    returned as `continuity_variant_replay.season_loop` returns them."""
    import pilot_alledge_replay as A
    from aew.v1port import pipeline as PL
    absorb = bool(flags["absorb"])
    allc = (np.ones(case["lat_c"].size, bool), np.ones(case["lon_c"].size, bool))
    allf = (np.ones(case["latgrid"].shape[0], bool), np.ones(case["longrid"].shape[1], bool))
    times = np.asarray(case["time"], float).ravel()

    def inputs_at(step):
        fields = A.prepared_fields(case, step, *allc, *allf)
        waves = A.detect_from_prepared(fields, float(times[step]), ct, ft, grids, absorb=absorb, **options)
        return waves, PL._median_over(fields["fine_zonal_wind_smoothed"]), PL._median_over(fields["fine_meridional_wind_smoothed"])

    finished, rows, checks = registered_loop(times, inputs_at, bool(flags["exclusive"]), hook=hook, recent_days=recent_days)
    out = [{"time": np.asarray(f["time"], float), "lat": np.asarray(f["meanlat"], float), "lon": np.asarray(f["meanlon"], float)} for f in finished]
    return out, rows, checks


def in_box(row, sector=V.SECTOR, margin=MARGIN_DEG):
    """True when a history has an unsmoothed position in the sector widened by `margin`."""
    return any(sector[0] - margin <= lo <= sector[1] + margin and sector[2] - margin <= la <= sector[3] + margin
               for la, lo in zip(row["lat"], row["lon"]))


def covered(lat, lon, sector=V.SECTOR, margin=MARGIN_DEG, radius_km=C.RADIUS_KM):
    """True when every point within `radius_km` of a position lies inside the retained box, so no
    history that could accompany an observation there was left out of the record."""
    dlat = radius_km / 111.19
    dlon = dlat / max(np.cos(np.deg2rad(min(abs(lat) + dlat, 89.0))), 1e-6)
    return bool(sector[2] - margin <= lat - dlat and lat + dlat <= sector[3] + margin
                and sector[0] - margin <= lon - dlon and lon + dlon <= sector[1] + margin)


def as_history(row, lat, lon):
    return {"id": row["birth"], "time": np.asarray(row["time"], float), "lat": np.asarray(lat, float), "lon": np.asarray(lon, float),
            "step": row["step"], "raw_lat": row["lat"], "raw_lon": row["lon"], "est": row["est"], "fate": row["fate"]}


def finalized(rows):
    """The rows that pass production's final filter, at their smoothed positions."""
    from aew.v1port.association import finalize_tracks
    proxies = [{"time": r["time"], "meanlat": r["lat"], "meanlon": r["lon"], "row": r} for r in rows]
    return [as_history(f["row"], f["meanlat"], f["meanlon"]) for f in finalize_tracks(proxies, total_steps=None)]


def history_sets(rows):
    stored = finalized([r for r in rows if r["fate"] == "finished"])
    if len(stored) != sum(1 for r in rows if r["fate"] == "finished"):
        raise SystemExit("REFUSED: a finished history does not pass the final filter a second time")
    added = finalized([r for r in rows if r["fate"] == "pruned_in_loop"])
    return {"STORED": stored, "KEEP": stored + added, "ALL": [as_history(r, r["lat"], r["lon"]) for r in rows]}


def break_class(a, b):
    """The class of a break from history `a` to history `b` beside one reference track."""
    gap = (b["step"][0] - a["step"][-1]) * 6
    if gap <= 0:
        return "concurrent"
    if gap <= 12:
        return "within_horizon"
    if gap <= 24:
        return "18_to_24_hours"
    return "30_to_48_hours" if gap <= 48 else "over_48_hours"


def breaks(ref_rows, by_id):
    out = {k: 0 for k in BREAKS}
    for r in ref_rows:
        seq = [n for n in r["nearest"] if n is not None]
        for a, b in zip(seq, seq[1:]):
            if a != b:
                out[break_class(by_id[a], by_id[b])] += 1
    return out


def in_order(histories):
    return sorted(histories, key=lambda h: (h["step"][0], h["id"]))


def successors(histories, horizon_hours):
    """For each history, the ones that begin `FIRST_NEW_HOURS` to `horizon_hours` after it ends
    at an implied speed under the production maximum."""
    by_first = {}
    for h in histories:
        by_first.setdefault(h["step"][0], []).append(h)
    after = {}
    for a in histories:
        after[a["id"]] = []
        for s in range(a["step"][-1] + int(FIRST_NEW_HOURS // 6), a["step"][-1] + int(horizon_hours // 6) + 1):   # whole steps, so the bounds are exact
            gap = (s - a["step"][-1]) * 6.0
            for b in by_first.get(s, []):
                km = float(C.km(a["raw_lat"][-1], a["raw_lon"][-1], b["raw_lat"][0], b["raw_lon"][0]))
                if km * 1000.0 / (gap * 3600.0) < MAX_SPEED_MS:
                    after[a["id"]].append(b["id"])
    return after


def best_chain(row, order, after):
    """The longest accompaniment of one reference track by a chain of the histories in `order`
    and, for a track on the Atlantic side, whether a chain accompanies it in its first 48 hours
    and at that observation."""
    best, reach = {}, {}
    event = set(row["event"])
    for h in reversed(order):                             # a successor always starts later, so it is already scored
        i = h["id"]
        best[i] = len(row["by_history"].get(i, ())) + max((best[b] for b in after[i]), default=0)
        reach[i] = i in event or any(reach[b] for b in after[i])
    carried = any(reach.get(i, False) for i in row["start"]) if row["outcome"] == "atlantic_side" else None
    return max(best.values(), default=0), carried


def chained(row, by_id, horizon_hours):
    """Section 18's limit for one reference track, chains made only of the histories beside it."""
    order = in_order([by_id[h] for h in row["by_history"]])
    return best_chain(row, order, successors(order, horizon_hours))


def chain_limits(ref_rows, by_id, horizons=HORIZONS):
    out = {}
    for h in (0,) + tuple(horizons):
        res = [chained(r, by_id, h) for r in ref_rows]
        out[str(h)] = {"longest_single_history": sum(x[0] for x in res), "carried": sum(1 for x in res if x[1])}
    return out


def open_chain_limits(ref_rows, histories, horizons=HORIZONS):
    """The same limit when ANY retained history may sit inside a chain, including one that never
    comes within 500 km of the reference track. It bounds what joining whole histories under the
    gap and speed conditions can deliver, which section 18's limit does not. Beside each horizon
    it gives how many histories have a successor and how many successor links there are."""
    order = in_order(histories)
    out = {}
    for h in (0,) + tuple(horizons):
        after = successors(order, h)
        res = [best_chain(r, order, after) for r in ref_rows]
        out[str(h)] = {"longest_single_history": sum(x[0] for x in res), "carried": sum(1 for x in res if x[1]),
                       "histories_with_a_successor": sum(1 for v in after.values() if v), "successor_links": sum(len(v) for v in after.values())}
    return out


def ever_accompanied(history, q_index):
    return any(C.companions(la, lo, d, q_index) for la, lo, d in zip(history["lat"], history["lon"], history["time"]))


def keep_cost(sets, q_index, sector=V.SECTOR):
    """The histories KEEP adds in the sector beside the stored tracks there."""
    stored = R.in_sector_tracks(sets["STORED"], sector)
    added = R.in_sector_tracks([h for h in sets["KEEP"] if h["fate"] == "pruned_in_loop"], sector)
    held = {}
    for h in sets["ALL"]:
        for s, la, lo in zip(h["step"], h["raw_lat"], h["raw_lon"]):
            held.setdefault((s, round(la, 9), round(lo, 9)), set()).add(h["id"])
    shared = sum(1 for h in added if any(len(held[(s, round(la, 9), round(lo, 9))]) > 1 for s, la, lo in zip(h["step"], h["raw_lat"], h["raw_lon"])))
    lengths = {}
    for h in added:
        lengths[str(len(h["step"]))] = lengths.get(str(len(h["step"])), 0) + 1
    return {"stored_in_sector": len(stored), "stored_ever_accompanied": sum(1 for h in stored if ever_accompanied(h, q_index)),
            "added_in_sector": len(added), "added_ever_accompanied": sum(1 for h in added if ever_accompanied(h, q_index)),
            "added_sharing_an_observation": shared, "added_by_observations": dict(sorted(lengths.items(), key=lambda x: int(x[0])))}


def sample_measures(sets, sample_season, time_of_step, bind_stored):
    """Of the stage sample's observations, by its class, how many each set accompanies. With
    `bind_stored`, STORED must accompany exactly the observations the sample classed as tracked."""
    obs = [(tr["qtrack_id"], st) for tr in sample_season["tracks"] for st in tr["steps"]]
    out = {"observations": {c: sum(1 for _, st in obs if st["class"] == c) for c in ("tracked", "candidate_not_tracked", "placed_away")}}
    for name in SETS:
        idx = C.index_by_time(sets[name])
        hit = {c: 0 for c in out["observations"]}
        for qid, st in obs:
            got = bool(C.companions(st["lat"], st["lon"], time_of_step[st["step"]], idx))
            if bind_stored and name == "STORED" and got != (st["class"] == "tracked"):
                raise SystemExit(f"REFUSED: the stored tracks do not accompany QTrack {qid} at step {st['step']} as the stage sample records")
            hit[st["class"]] += int(got)
        out[name] = hit
    return out


def holder_ends(rows, sample_season, readings_season):
    """For each pruned history the association reading named, its length and what the stage
    sample holds at the two attempts after its last observation."""
    by_birth = {r["birth"]: r for r in rows}
    classes = {(tr["qtrack_id"], st["step"]): st["class"] for tr in sample_season["tracks"] for st in tr["steps"]}
    pairs = {}
    for a in readings_season["association"]:
        if a["fate"] != "pruned_in_loop":
            continue
        row = by_birth.get(a["holder"])
        if row is None or row["fate"] != "pruned_in_loop" or len(row["step"]) != a["n_obs"]:
            raise SystemExit(f"REFUSED: the pruned history {a['holder']} of the association reading is not reproduced by the replay")
        pairs.setdefault((a["qtrack_id"], a["holder"]), row)
    out = []
    for (qid, holder), row in sorted(pairs.items()):
        last = row["step"][-1]
        after = [classes.get((qid, last + k)) for k in (1, 2)]
        if any(c in ("candidate_not_tracked", "tracked") for c in after):
            end = "candidate_present"
        elif "placed_away" in after:
            end = "placed_away"
        else:
            end = "beyond_reference"
        out.append({"qtrack_id": qid, "holder": holder, "n_obs": len(row["step"]), "span_steps": last - row["step"][0] + 1, "end": end})
    return out


def season_census(rows, q_tracks, q_rows, sample_season, readings_season, time_of_step):
    """Every measure of section 18's first stage for one configuration of one season."""
    q = {t["id"]: t for t in q_tracks}
    q_index, q_spans = C.index_by_time(q_tracks), J.spans(q_tracks)
    sets = history_sets(rows)
    by_id = {h["id"]: h for h in sets["ALL"]}
    out, ref = {}, {}
    for name in SETS:
        ref[name] = E.reference_measures(sets[name], q, q_rows)
        out[name] = {"histories": len(sets[name]), "reference": E.pool_reference(ref[name])}
        if name != "ALL":
            out[name]["flags"], out[name]["costs"] = E.flag_measures(sets[name], q_index, q_spans, ref[name])
            out[name]["link_status"] = E.link_census(sets[name], q_index, q_spans)
    for r in ref["ALL"]:
        t = q[r["id"]]
        for k in range(r["steps"]):
            if np.isfinite(t["lat"][k]) and np.isfinite(t["lon"][k]) and not covered(float(t["lat"][k]), float(t["lon"][k])):
                raise SystemExit(f"REFUSED: reference track {r['id']} has a used observation whose surroundings the retained box does not cover")
    out["breaks"] = breaks(ref["ALL"], by_id)
    out["chain_limits"] = chain_limits(ref["ALL"], by_id)
    out["chain_limits_through_any_history"] = open_chain_limits(ref["ALL"], sets["ALL"])
    out["keep_cost"] = keep_cost(sets, q_index)
    out["sample"] = sample_measures(sets, sample_season, time_of_step, bind_stored=readings_season is not None)
    if readings_season is not None:
        out["holders"] = holder_ends(rows, sample_season, readings_season)
    return out, sets


def bind(name, census, jconf):
    """The census's stored set must be the tracks the false-join test classed."""
    ref, flags, costs, counts = census["STORED"]["reference"], census["STORED"]["flags"], census["STORED"]["costs"], jconf["counts"]
    if (ref["accompanied"], ref["steps"], ref["classes"]) != (counts["accompanied_steps"], counts["qtrack_steps"], counts["qtrack_classes"]):
        raise SystemExit(f"REFUSED: the {name} stored set's reference measures differ from the false-join test's correspondence")
    if (costs["tracks_in_sector"], costs["stationary_tracks"]) != (counts["tracks_in_sector"], counts["stationary_tracks"]):
        raise SystemExit(f"REFUSED: the {name} stored set's sector counts differ from the false-join test's")
    if {k: flags["events"][k] for k in jconf["events"]} != jconf["events"]:
        raise SystemExit(f"REFUSED: the {name} stored set's transitions differ from the false-join test's")
    if flags["events"]["undetermined"] == 0 and {k: flags["display_classes"][k] for k in J.CLASS_ORDER} != jconf["classes"]:
        raise SystemExit(f"REFUSED: the {name} stored set's classes differ from the false-join test's")


def check_retained(year, jrec, retained):
    """The replay's finished tracks, classes and counts must be those section 16 retained."""
    for name in CONFIGS:
        a, b = jrec["configs"][name], retained["configs"][name]
        if a["tracks"] != b["tracks"] or a["classes"] != b["classes"] or a["counts"] != b["counts"]:
            raise SystemExit(f"REFUSED: the {year} {name} replay does not reproduce the tracks the false-join test retained")


def check_stored(name, sets, rows, finished):
    """Each stored history rebuilt from its registered row must equal the track the loop finished."""
    index = {r["birth"]: r["finished_index"] for r in rows if r["fate"] == "finished"}
    for h in sets["STORED"]:
        f = finished[index[h["id"]]]
        if not (np.array_equal(h["time"], f["time"]) and np.array_equal(h["lat"], f["lat"]) and np.array_equal(h["lon"], f["lon"])):
            raise SystemExit(f"REFUSED: the {name} history {h['id']} is not stored as the replay finished it")


def run_season(year, join_dir=JOIN_DIR, sample_path=RD.SAMPLE, readings_path=None):
    if year not in S.YEARS:
        raise SystemExit(f"REFUSED: {year} is not a development season of section 8")
    import qtrack_pairing_pilot as M
    readings_path = readings_path or os.path.join(join_dir, "continuity_readings.json")
    join_path = os.path.join(join_dir, f"continuity_join_test_{year}.json.gz")
    with gzip.open(join_path, "rt", encoding="utf-8") as fh:
        retained = json.load(fh)
    if retained["generated_by"]["sha256"] != J.SCRIPT_SHA256_AT_LOAD:
        raise SystemExit(f"REFUSED: the {year} false-join record was not written by the current false-join script")
    sample = json.load(open(sample_path, encoding="utf-8"))
    if sample["generated_by"]["sha256"] != S.SCRIPT_SHA256_AT_LOAD:
        raise SystemExit("REFUSED: the stage sample was not written by the current stage-sample script")
    readings = json.load(open(readings_path, encoding="utf-8"))
    if readings["generated_by"]["sha256"] != RD.SCRIPT_SHA256_AT_LOAD or readings["generated_by"]["sample"]["sha256"] != C.sha256_of(sample_path):
        raise SystemExit("REFUSED: the readings record was not written by the current readings script from this stage sample")
    sample_season = next(s for s in sample["seasons"] if s["year"] == year)
    readings_season = next(s for s in readings["seasons"] if s["year"] == year)
    captured = {}

    def loop(case, grids, ct, ft, flags, options):
        finished, rows, checks = lifecycle_loop(case, grids, ct, ft, flags, options)
        captured[json.dumps(options, sort_keys=True)] = (finished, rows, checks, [float(x) for x in np.asarray(case["time"], float).ravel()])
        return finished

    jrec = J.run_season(year, stage_dir=join_dir, loop=loop)
    check_retained(year, jrec, retained)
    art = json.load(open(jrec["artifact"]["path"], encoding="utf-8"))
    q_tracks = M.load_qtrack(art["inputs"]["qtrack"]["path"])[0]
    q_rows = art["sides"]["qtrack"]["tracks"]
    out = {}
    for name, options in CONFIGS.items():
        finished, rows, checks, times = captured[json.dumps(options, sort_keys=True)]
        if checks["pruned_with_both_attempts_spent"] + checks["pruned_with_an_attempt_stranded"] != sum(1 for r in rows if r["fate"] == "pruned_in_loop"):
            raise SystemExit(f"REFUSED: the {year} {name} prune checks do not cover every pruned history")
        boxed = [r for r in rows if in_box(r)]
        census, sets = season_census(boxed, q_tracks, q_rows, sample_season, readings_season if name == "released" else None, times)
        check_stored(f"{year} {name}", sets, boxed, finished)
        bind(f"{year} {name}", census, jrec["configs"][name])
        out[name] = {"fates": {k: sum(1 for r in rows if r["fate"] == k) for k in FATES}, "prune_checks": checks,
                     "histories_retained_in_record": len(boxed), "census": census, "rows": boxed}
    return {"year": year, "case_sha256": jrec["case_sha256"], "join_record": {"path": join_path, "sha256": C.sha256_of(join_path)},
            "sample": {"path": sample_path, "sha256": C.sha256_of(sample_path)}, "readings": {"path": readings_path, "sha256": C.sha256_of(readings_path)},
            "artifact": jrec["artifact"], "configs": out}


def _add(a, b):
    """The sum of two nested records of counts, a key missing on one side counting as zero."""
    if isinstance(a, dict):
        return {k: (_add(a[k], b[k]) if k in a and k in b else a.get(k, b.get(k))) for k in list(a) + [k for k in b if k not in a]}
    return a + b


def pool(seasons):
    pooled = {}
    for name in CONFIGS:
        cs = [s["configs"][name]["census"] for s in seasons]
        p = {k: cs[0][k] for k in SETS + ("breaks", "chain_limits", "chain_limits_through_any_history", "sample")}
        for c in cs[1:]:
            for k in p:
                p[k] = _add(p[k], c[k])
        kc = [c["keep_cost"] for c in cs]
        lengths = {}
        for c in kc:
            for n, v in c["added_by_observations"].items():
                lengths[n] = lengths.get(n, 0) + v
        p["keep_cost"] = {**{k: sum(c[k] for c in kc) for k in kc[0] if k != "added_by_observations"},
                          "added_by_observations": dict(sorted(lengths.items(), key=lambda x: int(x[0])))}
        p["fates"] = {k: sum(s["configs"][name]["fates"][k] for s in seasons) for k in FATES}
        p["prune_checks"] = {k: sum(s["configs"][name]["prune_checks"][k] for s in seasons) for k in seasons[0]["configs"][name]["prune_checks"]}
        if "holders" in cs[0]:
            hs = [{"year": s["year"], **h} for s in seasons for h in s["configs"][name]["census"]["holders"]]
            distinct = {(h["year"], h["holder"]): h for h in hs}
            lengths = {}
            for h in distinct.values():
                lengths[str(h["n_obs"])] = lengths.get(str(h["n_obs"]), 0) + 1
            p["holders"] = {"pairs": len(hs), "distinct": len(distinct), "by_observations": dict(sorted(lengths.items(), key=lambda x: int(x[0]))),
                            "ends_by_pair": {e: sum(1 for h in hs if h["end"] == e) for e in ENDS}}
        pooled[name] = p
    return pooled


def decide(pooled):
    """Section 18's two provisional marks, read on the released configuration."""
    p = pooled["released"]
    steps = p["STORED"]["reference"]["steps"]
    if len({p[n]["reference"]["steps"] for n in SETS} | {pooled[c][n]["reference"]["steps"] for c in pooled for n in SETS}) != 1:
        raise SystemExit("REFUSED: the sets were scored over different reference observations")
    keep_rise = p["KEEP"]["reference"]["accompanied"] - p["STORED"]["reference"]["accompanied"]
    base, lim = p["chain_limits"]["0"], p["chain_limits"][str(BAR_HORIZON)]
    carried_rise = lim["carried"] - base["carried"]
    longest_rise = lim["longest_single_history"] - base["longest_single_history"]
    out = {"keep": {"accompanied_rise": keep_rise, "share_rise": round(keep_rise / steps, 4), "mark_met": bool(keep_rise * KEEP_ONE_IN >= steps)},
           "recovery": {"horizon_hours": BAR_HORIZON, "carried_rise": carried_rise, "longest_rise": longest_rise, "longest_share_rise": round(longest_rise / steps, 4),
                        "developed_further": bool(carried_rise >= BAR_CARRIED or longest_rise * BAR_ONE_IN >= steps)}}
    if "chain_limits_through_any_history" in p:           # the same bar on the limit that does bound a join of whole histories
        base, lim = p["chain_limits_through_any_history"]["0"], p["chain_limits_through_any_history"][str(BAR_HORIZON)]
        c, n = lim["carried"] - base["carried"], lim["longest_single_history"] - base["longest_single_history"]
        out["recovery_through_any_history"] = {"horizon_hours": BAR_HORIZON, "carried_rise": c, "longest_rise": n, "longest_share_rise": round(n / steps, 4),
                                                "bar_met": bool(c >= BAR_CARRIED or n * BAR_ONE_IN >= steps)}
    return out


def tables_text(rec):
    P_, D = rec["pooled"], rec["decision"]
    L = ["# Lifecycle census of the continuity effort (declared section 18)", "",
         "1990 and 2006 pooled. STORED is the finished tracks, KEEP adds the pruned histories that pass the final filter, and ALL is every "
         "history the association built, at its unsmoothed positions. The reference population is the QTrack cohort tracks starting at 10 to 30 E.", ""]
    for name, p in P_.items():
        ref = p["STORED"]["reference"]
        L += [f"## {name}", "",
              "Histories by fate, whole domain: " + ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in p["fates"].items()) + ". "
              f"Pruned with both attempts spent {p['prune_checks']['pruned_with_both_attempts_spent']}, with an attempt stranded "
              f"{p['prune_checks']['pruned_with_an_attempt_stranded']}, with an attempt still due 0.", "",
              f"| set | reference observations accompanied (of {ref['steps']}) | carried | relayed | picked up late | dropped | unaccompanied at both | "
              f"longest single-history accompaniment | reference tracks accompanied (of {ref['tracks']}) | histories per accompanied reference track |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for s in SETS:
            r = p[s]["reference"]; c = r["classes"]
            per = r["histories_on_accompanied"] / r["tracks_accompanied"] if r["tracks_accompanied"] else 0.0
            L.append(f"| {s} | {r['accompanied']} ({100.0 * r['accompanied'] / r['steps']:.1f} %) | {c['carried']} | {c['relayed']} | {c['picked_up_late']} | {c['dropped']} | "
                     f"{c['unaccompanied_at_both']} | {r['longest_single_history']} | {r['tracks_accompanied']} | {per:.2f} |")
        k = p["keep_cost"]
        L += ["", f"KEEP adds {k['added_in_sector']} histories in the sector to {k['stored_in_sector']} stored tracks. A QTrack system ever accompanies "
                  f"{k['added_ever_accompanied']} of the added ({100.0 * k['added_ever_accompanied'] / max(1, k['added_in_sector']):.1f} %) and {k['stored_ever_accompanied']} of the stored "
                  f"({100.0 * k['stored_ever_accompanied'] / max(1, k['stored_in_sector']):.1f} %). Of the added, {k['added_sharing_an_observation']} share an observation with another history. "
                  f"Added histories by observations: {k['added_by_observations']}.", "",
              "| stage sample class | observations | STORED accompanies | KEEP accompanies | ALL accompanies |", "| --- | --- | --- | --- | --- |"]
        for c, n in p["sample"]["observations"].items():
            L.append(f"| {c.replace('_', ' ')} | {n} | {p['sample']['STORED'][c]} | {p['sample']['KEEP'][c]} | {p['sample']['ALL'][c]} |")
        if "holders" in p:
            h = p["holders"]
            L += ["", f"The association reading's pruned histories: {h['distinct']} distinct in {h['pairs']} pairs with a QTrack track, by observations {h['by_observations']}. "
                      "At the two attempts after the last observation, by pair: " + ", ".join(f"{e.replace('_', ' ')} {n}" for e, n in h["ends_by_pair"].items()) + "."]
        L += ["", "Breaks along the reference tracks (a change of the nearest history of ALL between successive accompanied observations): "
              + ", ".join(f"{b.replace('_', ' ')} {n}" for b, n in p["breaks"].items()) + ".", "",
              "| chaining horizon (hours) | carried, best case | longest single-history accompaniment, best case |", "| --- | --- | --- |"]
        for hz, v in p["chain_limits"].items():
            L.append(f"| {'none' if hz == '0' else hz} | {v['carried']} | {v['longest_single_history']} |")
        L += ["", "| chaining horizon (hours), any retained history inside a chain | carried, best case | longest single-history accompaniment, best case | "
                  "histories with a successor | successor links |", "| --- | --- | --- | --- | --- |"]
        for hz, v in p["chain_limits_through_any_history"].items():
            L.append(f"| {'none' if hz == '0' else hz} | {v['carried']} | {v['longest_single_history']} | {v['histories_with_a_successor']} | {v['successor_links']} |")
        L += ["", "| links of STORED in the sector, by hours of the prediction | consistent | switch | relay | undetermined | unassessed |", "| --- | --- | --- | --- | --- | --- |"]
        for hrs, v in p["STORED"]["link_status"].items():
            L.append(f"| {hrs} | {v['consistent']} | {v['switch']} | {v['relay']} | {v['undetermined']} | {v['unassessed']} |")
        L += ["", "| set | tracks in sector | flagged tracks | switch events | relay events | undetermined events | exposure pairs | QTrack systems left behind | "
                  "reference tracks with a flagged history | switch events with a jump link | supported false-link candidates | relay events with a jump link | links | jump links | stationary |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for s in ("STORED", "KEEP"):
            f, c = p[s]["flags"], p[s]["costs"]
            L.append(f"| {s} | {c['tracks_in_sector']} | {f['flagged_tracks']} | {f['events']['switch']} | {f['events']['relay']} | {f['events']['undetermined']} | {f['exposure_pairs']} | "
                     f"{f['systems_left_behind']} | {f['reference_tracks_with_a_flagged_history']} | {f['switch_events_with_a_jump_link']} | {f['supported_false_link_candidates']} | "
                     f"{f['relay_events_with_a_jump_link']} | {c['links']} | {c['jump_links']} | {c['stationary_tracks']} |")
        e = p["STORED"]["flags"]["switch_events_by_elapsed"]
        L += ["", "Switch events of STORED by time elapsed since the previous accompanied observation: " + ", ".join(f"{k.replace('_', ' ')} {n}" for k, n in e.items()) + ".", ""]
    L += ["## Per season", ""]
    for y, cfgs in rec["seasons"].items():
        for name, c in cfgs.items():
            cz = c["census"]
            L.append(f"- {y} {name}: accompanied " + ", ".join(f"{s} {cz[s]['reference']['accompanied']}" for s in SETS) + f" of {cz['STORED']['reference']['steps']}; "
                     + "best-case carried " + ", ".join(f"{'none' if h == '0' else h + ' h'} {v['carried']}" for h, v in cz["chain_limits"].items()) + "; "
                     + "best-case longest " + ", ".join(f"{'none' if h == '0' else h + ' h'} {v['longest_single_history']}" for h, v in cz["chain_limits"].items()))
    L.append("")
    K, Rv = D["keep"], D["recovery"]
    L += ["## Provisional marks, released configuration", "",
          f"KEEP raises the accompanied reference observations by {K['accompanied_rise']} ({100.0 * K['share_rise']:.1f} points), mark of {100.0 / KEEP_ONE_IN:g} points "
          f"{'met' if K['mark_met'] else 'not met'}.",
          f"Chaining at {Rv['horizon_hours']} hours raises the best-case carried count by {Rv['carried_rise']} and the best-case longest single-history accompaniment by "
          f"{Rv['longest_rise']} observations ({100.0 * Rv['longest_share_rise']:.1f} %), against a bar of {BAR_CARRIED} carried or {100.0 / BAR_ONE_IN:g} %. "
          f"Recovery developed further: {'yes' if Rv['developed_further'] else 'no'}."]
    if "recovery_through_any_history" in D:
        A_ = D["recovery_through_any_history"]
        L.append(f"With any retained history allowed inside a chain, the rises are {A_['carried_rise']} carried and {A_['longest_rise']} observations "
                 f"({100.0 * A_['longest_share_rise']:.1f} %), and the same bar is {'met' if A_['bar_met'] else 'not met'}.")
    L.append("")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("season", "combine"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--year", type=int)
    a = ap.parse_args(argv)
    if a.mode == "season":
        p = os.path.join(a.out_dir, f"continuity_lifecycle_{a.year}.json.gz")
        if os.path.exists(p):
            raise SystemExit(f"REFUSED: {p} exists and records are never overwritten")
        rec = {"generated_by": {"script": "scripts/continuity_lifecycle.py", "sha256": SCRIPT_SHA256_AT_LOAD, "evaluation_sha256": E.SCRIPT_SHA256_AT_LOAD,
                                "declared": "CONTINUITY_ACROSS_AFRICA_2026-11-09.md section 18", "configs": CONFIGS}, **run_season(a.year)}
        with gzip.open(p, "xt", encoding="utf-8") as fh:
            json.dump(rec, fh)
        print(json.dumps({n: {k: v for k, v in c.items() if k != "rows"} for n, c in rec["configs"].items()}, indent=1))
        return 0
    pj, pm = (os.path.join(a.out_dir, f"continuity_lifecycle.{e}") for e in ("json", "md"))
    for p in (pj, pm):
        if os.path.exists(p):
            raise SystemExit(f"REFUSED: {p} exists and records are never overwritten")
    seasons, digests = [], {}
    for y in S.YEARS:
        path = os.path.join(a.out_dir, f"continuity_lifecycle_{y}.json.gz")
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            r = json.load(fh)
        g = r["generated_by"]
        if g["sha256"] != SCRIPT_SHA256_AT_LOAD or g["evaluation_sha256"] != E.SCRIPT_SHA256_AT_LOAD:
            raise SystemExit(f"REFUSED: the {y} season record was not written by the current scripts")
        seasons.append(r); digests[str(y)] = C.sha256_of(path)
    pooled = pool(seasons)
    rec = {"generated_by": {"script": "scripts/continuity_lifecycle.py", "sha256": SCRIPT_SHA256_AT_LOAD, "evaluation_sha256": E.SCRIPT_SHA256_AT_LOAD, "seasons": digests},
           "seasons": {str(s["year"]): {n: {k: v for k, v in c.items() if k != "rows"} for n, c in s["configs"].items()} for s in seasons},
           "pooled": pooled, "decision": decide(pooled)}
    with open(pj, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    text = tables_text(rec)
    with open(pm, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
