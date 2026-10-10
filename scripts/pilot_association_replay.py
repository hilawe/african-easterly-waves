#!/usr/bin/env python3
"""The eastern pilot's instrumented association replay: one run, one season, the
production loop replayed from the retained case exactly as `export_protocol_case.track_case`
ran it, with detailed logging only at the steps of the named families' windows.

THE LOOP IS THE PRODUCTION LOOP, unchanged: from the case's first step, with an empty live
state, through `detect_troughs`, the region medians of the smoothed fine winds,
`associate_step`, `prune_stale_tracks`, to `finalize_tracks` at the case's last step. Nothing
is finalized early and no state is invented, because the final speed filter reads a track's
median segment speed and the five-point smoother's end windows shrink, so a run cut at a
window's end is not the archived run up to that window. THE GATE is that the finished
tracks equal the archived tracks of the run directory as a MULTISET (duplicates preserved,
since version 1's association lets two tracks claim one trough), and also in order, so a
finished index here is the archived index. A replay that fails the gate still writes its
log, marked failed, and exits nonzero.

WHAT IS LOGGED, only at the window steps and only as a side record (the live state is
never written to):

- every final candidate the association receives, in its order, with its center, point
  count, the digest of its region mask, its extents and the region medians of the smoothed
  fine winds that predict where it goes;
- every live track before the step, by a STABLE BIRTH IDENTITY ("step:candidate", assigned
  when the track is seeded and carried in a side registry keyed on the track object, with
  pruned objects kept so no identity is reused), its observation count, last position, the
  predictions it holds and whether it is due in the six-hour or the twelve-hour pass, and
  for a due track its search polygon as the pass reads it;
- the matching read for each due track (which candidates have a point inside its polygon,
  their implied speeds, their distances to the predicted center, the chosen one), computed
  read-only with the association's own primitives in its own order and then BOUND to what
  the production call did: the replay refuses if the read disagrees with an observed claim;
- the claims (track, candidate, pass), the seeds (new tracks and the candidate each came
  from), the tracks the prune dropped, and whether the step was skipped as wave-free;
- at the end, every track's fate: pruned in the loop, removed by the final speed filter, or
  finished with its archived index.

The raw history of every track alive or seeded at a logged step, and of every named
track, is recorded whatever the window: the positions it claimed, unsmoothed, with each
observation's region digest, since the finished output holds only smoothed positions.

    python3 scripts/pilot_association_replay.py --run-dir <run dir> --case <tracker_case.mat> --year 2006 \\
        --families <json> --label C --out <fresh json path>
"""
import argparse
import datetime as dt
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import exact_tracks as X  # noqa: E402
import pilot_stage_diagnostic as SD  # noqa: E402
import pilot_track_crosswalk as P  # noqa: E402

EPOCH = dt.datetime(1900, 1, 1)


def region_digest(mask):
    m = np.ascontiguousarray(np.asarray(mask, dtype=bool))
    h = hashlib.sha256(repr(m.shape).encode())
    h.update(m.tobytes())
    return h.hexdigest()


def _num(x):
    x = float(x)
    return None if math.isnan(x) else x


def window_steps(times, families):
    """The steps inside any family's window, and each family's own steps."""
    per = {}
    for fam in families:
        t0 = (dt.datetime.strptime(fam["window"][0], "%Y-%m-%dT%H") - EPOCH).total_seconds() / 86400.0
        t1 = (dt.datetime.strptime(fam["window"][1], "%Y-%m-%dT%H") - EPOCH).total_seconds() / 86400.0
        per[fam["name"]] = [k for k in range(times.size) if t0 - 1e-9 <= times[k] <= t1 + 1e-9]
        if not per[fam["name"]]:
            raise SystemExit(f"REFUSED: the window {fam['window']} holds no step of the case")
    return sorted(set(k for ks in per.values() for k in ks)), per


def in_box(lat, lon, box):
    return bool(box["lat"][0] <= lat <= box["lat"][1] and box["lon"][0] <= lon <= box["lon"][1])


def describe_candidates(waves, um, vm, families):
    out = []
    for i, w in enumerate(waves):
        la, lo = np.asarray(w["lat_wave"], float), np.asarray(w["lon_wave"], float)
        out.append({"index": i, "lat_mean": float(w["lat_mean"]), "lon_mean": float(w["lon_mean"]), "n_points": int(la.size),
                    "region_sha256": region_digest(w["region"]),
                    "lat_extent": [float(la.min()), float(la.max())] if la.size else None,
                    "lon_extent": [float(lo.min()), float(lo.max())] if lo.size else None,
                    "u_median": _num(um(w)), "v_median": _num(vm(w)),
                    "in_family_box": [f["name"] for f in families if in_box(float(w["lat_mean"]), float(w["lon_mean"]), f["box"])]})
    return out


def _poly(state, key):
    return [[float(x), float(y)] for x, y in zip(np.asarray(state["poly_lon_" + key], float).ravel(), np.asarray(state["poly_lat_" + key], float).ravel())]


def _prediction(st):
    return {"est_6hr": [_num(st["est_lat_6hr"]), _num(st["est_lon_6hr"])], "est_12hr": [_num(st["est_lat_12hr"]), _num(st["est_lon_12hr"])]}


def due_pass(state, step):
    from aew.v1port.association import NO_PREDICTION
    if state.get("est_step_6hr") == step:
        return "6hr"
    if state.get("est_step_12hr") == step and state.get("est_step_6hr") == NO_PREDICTION:
        return "12hr"
    return None


def live_snapshot(tracks, states, step, registry):
    out = []
    for tr, st in zip(tracks, states):
        due = due_pass(st, step)
        row = {"birth": registry[id(tr)], "n_obs": len(tr["time"]), "last_step": int(tr["step"][-1]),
               "last_lat": float(tr["meanlat"][-1]), "last_lon": float(tr["meanlon"][-1]),
               "est_step_6hr": st.get("est_step_6hr"), "est_step_12hr": st.get("est_step_12hr"), "due": due, **_prediction(st)}
        if due:
            row["polygon_" + due] = _poly(st, due)
        out.append(row)
    return out


def read_matching(tracks, states, waves, step, exclusive, registry):
    """The association's matching, READ with its own primitives in its own order and never
    acted on: for each due track the candidates with a point inside its polygon, their
    implied speeds, their distances to the predicted center, and what `_match` chooses.
    The caller binds this to what the production call then does."""
    from aew.v1port import association as A
    from aew.v1port.geometry import great_circle_distance, points_in_polygon
    rows, claimed = [], set()
    last_state = states[-1] if states else None
    for key, hours in (("6hr", A.STEP_HOURS), ("12hr", 2 * A.STEP_HOURS)):
        for track, state in zip(tracks, states):
            if due_pass(state, step) != key:
                continue
            read = dict(state)
            if key == "12hr" and state is last_state:
                # lines 254-265 of find_ews_f.m inflate the LAST track's polygons between the
                # passes. The production call does it in place, this read does it on a copy.
                for k2, area in (("6hr", A.POLY_AREA_6HR), ("12hr", A.POLY_AREA_12HR)):
                    if "poly_lon_" + k2 in read:
                        read["poly_lon_" + k2], read["poly_lat_" + k2] = A._inflate(np.array(read["poly_lon_" + k2], float), np.array(read["poly_lat_" + k2], float), area)
            pool = [i for i in range(len(waves)) if i not in claimed] if exclusive else list(range(len(waves)))
            inside, speeds, dist = [], {}, {}
            for i in pool:
                w = waves[i]
                if not points_in_polygon(read["poly_lon_" + key], read["poly_lat_" + key], w["lon_wave"], w["lat_wave"]).any():
                    continue
                inside.append(i)
                km = float(great_circle_distance(w["lat_mean"], w["lon_mean"], track["meanlat"][-1], track["meanlon"][-1], "km"))
                speeds[str(i)] = km * 1000.0 / (hours * 3600.0)
                dist[str(i)] = float(great_circle_distance(read["est_lat_" + key], read["est_lon_" + key], w["lat_mean"], w["lon_mean"], "nm"))
            chosen = A._match(read, track, waves, pool, key, hours)
            if chosen is not None:
                claimed.add(chosen)
            rows.append({"birth": registry[id(track)], "pass": key, "candidates_inside_polygon": inside, "implied_speed_ms": speeds,
                         "distance_to_prediction_nm": dist, "polygon_as_read": _poly(read, key), "chosen": chosen})
    return rows


def raw_history(tr):
    """What a live track claimed, step by step, unsmoothed, with each observation's region."""
    return {"steps": [int(s) for s in tr["step"]], "times": [float(x) for x in tr["time"]],
            "lat_claimed": [float(x) for x in tr["meanlat"]], "lon_claimed": [float(x) for x in tr["meanlon"]],
            "n_points": [int(x) for x in tr["n_points"]], "region_sha256": [region_digest(m) for m in tr["wave_points"]]}


def replay(case, record, families, log_steps):
    from aew.v1port import climatology as clim
    from aew.v1port import pipeline as PL
    from aew.v1port.association import associate_step, finalize_tracks, prune_stale_tracks
    from aew.v1port.detection import detect_troughs
    flags = record["dataset_specific"]["tracker_flags"]
    exclusive, absorb = bool(flags["exclusive"]), bool(flags["absorb"])
    ct, ft = float(record["dataset_specific"]["coarse_threshold"]), float(record["dataset_specific"]["fine_threshold"])
    times, lat_c, lon_c = case["time"], case["lat_c"], case["lon_c"]
    latgrid_c, longrid_c = np.meshgrid(lat_c, lon_c, indexing="ij")
    tracks, states = [], []
    registry, graveyard, fates, log, touched, histories = {}, [], {}, {}, set(), {}
    log_set = set(log_steps)
    for step in range(times.size):
        t = float(times[step])
        waves = detect_troughs(t, latgrid_c, longrid_c, case["u_c"][step], case["currv_anom_c"][step], case["advcurrv_anom_c"][step],
                               case["latgrid"], case["longrid"], case["currv_anom"][step], coarse_threshold=ct, fine_threshold=ft, absorb=absorb)
        um = PL._median_over(clim.smooth9(case["u"][step]))
        vm = PL._median_over(clim.smooth9(case["v"][step]))
        logged = step in log_set
        if logged:
            touched.update(registry[id(tr)] for tr in tracks)
            entry = {"step": step, "date": f"{SD.date_of(t):%Y-%m-%dT%H:00Z}", "skipped_wave_free": not waves, "n_live_before": len(tracks),
                     "candidates": describe_candidates(waves, um, vm, families), "live_before": live_snapshot(tracks, states, step, registry),
                     "matching": read_matching(tracks, states, waves, step, exclusive, registry) if waves else []}
        n_before = {id(tr): len(tr["time"]) for tr in tracks}
        tracks, states = associate_step(tracks, states, waves, step, um, vm, exclusive=exclusive)
        claims, seeds = [], []
        for tr, st in zip(tracks, states):
            if id(tr) in registry:
                if len(tr["time"]) > n_before[id(tr)]:
                    cand = next(i for i, w in enumerate(waves) if tr["wave_points"][-1] is w["region"])
                    claims.append({"birth": registry[id(tr)], "candidate": cand, "n_obs": len(tr["time"]), "prediction_after": _prediction(st)})
            else:
                cand = next(i for i, w in enumerate(waves) if tr["wave_points"][0] is w["region"])
                birth = f"{step}:{cand}"
                if birth in fates:
                    raise SystemExit(f"REFUSED: two tracks born at step {step} from candidate {cand}")
                registry[id(tr)] = birth
                fates[birth] = {"born_step": step, "born_candidate": cand, "fate": "live"}
                seeds.append({"birth": birth, "candidate": cand, "prediction_after": _prediction(st)})
        if logged:
            # bind the read to the production call: a due track's read choice is its claim
            for m in entry["matching"]:
                actual = next((c["candidate"] for c in claims if c["birth"] == m["birth"]), None)
                if actual != m["chosen"]:
                    raise SystemExit(f"REFUSED: at step {step} the matching read chose {m['chosen']} for track {m['birth']} "
                                     f"and the production call claimed {actual}")
            due_of = {m["birth"]: m["pass"] for m in entry["matching"]}
            for c in claims:
                if c["birth"] not in due_of:
                    raise SystemExit(f"REFUSED: at step {step} track {c['birth']} claimed candidate {c['candidate']} without being due")
                c["pass"] = due_of[c["birth"]]
            entry["claims"], entry["seeds"] = claims, seeds
            touched.update(s["birth"] for s in seeds)
        before_prune = {id(tr): tr for tr in tracks}
        tracks, states = prune_stale_tracks(tracks, states, step, t, waves)
        kept = {id(tr) for tr in tracks}
        pruned = []
        for i, tr in before_prune.items():
            if i in kept:
                continue
            birth = registry[i]
            fates[birth].update({"fate": "pruned_in_loop", "fate_step": step, "n_obs": len(tr["time"]), "last_step": int(tr["step"][-1])})
            pruned.append({"birth": birth, "n_obs": len(tr["time"]), "last_step": int(tr["step"][-1])})
            if birth in touched:
                histories[birth] = raw_history(tr)
            graveyard.append(tr)                     # kept alive so its identity is never reused
            tr["lat_wave"], tr["lon_wave"], tr["wave_points"] = [], [], []
        if logged:
            entry["pruned"] = pruned
            log[step] = entry
    finished = finalize_tracks(tracks, total_steps=times.size)
    # a finished track is a shallow copy of its live track and shares its time list
    live_of = {id(tr["time"]): tr for tr in tracks}
    finished_birth = []
    for j, f in enumerate(finished):
        birth = registry[id(live_of[id(f["time"])])]
        fates[birth].update({"fate": "finished", "finished_index": j, "n_obs": len(f["time"])})
        finished_birth.append(birth)
    for tr in tracks:
        birth = registry[id(tr)]
        if fates[birth]["fate"] == "live":
            fates[birth].update({"fate": "removed_by_final_speed_filter" if len(tr["time"]) >= 2 else "removed_as_single_observation",
                                 "n_obs": len(tr["time"]), "last_step": int(tr["step"][-1])})
    return {"finished": finished, "finished_birth": finished_birth, "fates": fates, "log": log, "touched": touched,
            "histories": histories, "live_at_end": {registry[id(tr)]: tr for tr in tracks},
            "settings": {"exclusive": exclusive, "absorb": absorb, "coarse_threshold": ct, "fine_threshold": ft}}


def gate(finished, archived):
    """The finished tracks against the archived ones: as a multiset, and in order."""
    mine = {"tracks": [{"time": np.asarray(f["time"], float), "lat": np.asarray(f["meanlat"], float), "lon": np.asarray(f["meanlon"], float)} for f in finished]}
    ident = P.identity(mine, archived)

    def ordered(run):
        return [X.canonical([{"time": t["time"], "meanlat": t["lat"], "meanlon": t["lon"]}])[0] for t in run["tracks"]]
    ident["order_identical"] = ordered(mine) == ordered(archived)
    ident["passed"] = bool(ident["passed"] and ident["order_identical"])
    return ident


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("--run-dir", "--case", "--families", "--label", "--out"):
        ap.add_argument(name, required=True)
    ap.add_argument("--year", type=int, required=True)
    args = ap.parse_args(argv)
    if os.path.exists(args.out):
        raise SystemExit(f"REFUSED: {args.out} exists and a replay record is never overwritten")
    run = P.load_run(args.run_dir, year=args.year)
    case = SD.load_case(args.case, run["record"])
    fams = json.load(open(args.families))["families"]
    log_steps, per_family = window_steps(case["time"], fams)
    t0 = time.perf_counter()
    result = replay(case, run["record"], fams, log_steps)
    elapsed = time.perf_counter() - t0
    g = gate(result["finished"], run)
    key = "control_tracks" if args.label == "B" else "treatment_tracks"
    named, histories = {}, dict(result["histories"])
    for birth in result["touched"]:
        if birth in result["live_at_end"]:
            histories[birth] = raw_history(result["live_at_end"][birth])
    for fam in fams:
        for idx in fam.get(key, []):
            if idx < len(result["finished_birth"]):
                birth = result["finished_birth"][idx]
                histories[birth] = raw_history(result["live_at_end"][birth])
                named[f"{args.label}{idx}"] = {"family": fam["name"], "finished_index": idx, "birth": birth, "fate": result["fates"][birth]}
    counts = {}
    for f in result["fates"].values():
        counts[f["fate"]] = counts.get(f["fate"], 0) + 1
    out = {"generated_by": "scripts/pilot_association_replay.py", "script_sha256": X.digest(os.path.abspath(__file__)), "year": args.year, "label": args.label,
           "run_dir": args.run_dir, "record_file": run["record_file"], "record_sha256": run["record_sha256"], "tracks_sha256": run["tracks_sha256"],
           "case_path": args.case, "case_sha256": case["sha256"], "families_file": args.families, "families_sha256": X.digest(args.families),
           "settings": result["settings"], "loop": "export_protocol_case.track_case, replayed in full from the case's first step with an empty live state to its last step, nothing finalized early",
           "gate": g, "elapsed_seconds": round(elapsed, 1), "steps_in_case": int(case["time"].size), "logged_steps": log_steps,
           "family_steps": per_family, "fate_counts": counts, "finished_birth": result["finished_birth"],
           "named_tracks": named, "steps": [result["log"][k] for k in log_steps], "histories": histories, "fates": result["fates"]}
    X.publish_json(args.out, out, exclusive=True)
    print(f"{args.label} {args.year}: {len(result['finished'])} finished tracks, gate {'PASSED' if g['passed'] else 'FAILED'} "
          f"({g['shared']} shared, {g['only_in_a']} only replayed, {g['only_in_b']} only archived, order {g['order_identical']}), "
          f"{elapsed:.0f} s, fates {counts}")
    return 0 if g["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
