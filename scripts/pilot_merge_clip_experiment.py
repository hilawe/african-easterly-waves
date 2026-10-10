#!/usr/bin/env python3
"""The bounded coarse-merge experiment of `MERGE_EXHAUSTION_PROPOSAL_2026-10-08.md`.

Two modes over one declared comparison set. BASELINE recomputes the frozen candidate's
detection, flag off, at every step of the declared windows and at the stored steps of
the declared stationary histories, records the final candidates with region digests, the
lineage of every coarse and fine input through the merges, and the baseline measures
(close pairs, counts near the comparison locations and the stored positions), and
refuses unless the final candidates reproduce the retained event trace at its steps.
VARIANT recomputes the same steps flag off, refuses unless they reproduce the baseline
record digest for digest, recomputes them with the coarse clip on, and evaluates the
proposal's measures against the baseline. The reader's outcome categories are applied in
the results document, not here. The measures:

    A recovery at the missing times and the two controls; C uniqueness at the missing
    times; B the baseline's close pairs and the newly introduced ones with evidence;
    D1 and D2 the no-addition restrictions; D3 the descriptive persistence proxy; E the
    regression windows; F1 region-selection invariance for non-exhausted inputs; F2
    exact flag-off reproduction; F3 downstream interactions; G lineage.

THE SECOND VARIANT, `--coarse-position input` (2026-10-09), gives each coarse pass-one
wave its input axis's own mean position instead of the selected region's median and
changes nothing else. Its F1 is strict, every selection, exhausted or not, must be
unchanged. Two measures serve it and the clip alike: `--segment` reports correspondence
with a declared axis segment by the R1 pilot's distance, and the input fates follow every
coarse input to its final candidate in both runs.

    python3 scripts/pilot_merge_clip_experiment.py --mode baseline --control-run <dir> \
        --control-case <mat> --wide-dir <dir> --year 2006 --calibration <audit> \
        --domain-calibration --replay <margin60_own.json.gz> --trace <trace.json.gz> \
        --window C2 972 992 5 25 10 35 --window C3 889 898 0 30 15 50 \
        --stationary C5_1286 1286 --missing 988 14.5 26 ... --out <record.json.gz>
"""
import argparse
import datetime as dt
import gzip
import hashlib
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import exact_tracks as X  # noqa: E402
import pilot_alledge_replay as A  # noqa: E402
import pilot_association_replay as R  # noqa: E402
import pilot_margin_replay as MR  # noqa: E402
import pilot_stage_diagnostic as SD  # noqa: E402
import pilot_track_crosswalk as P  # noqa: E402

CLOSE_PAIR_DEG = 5.0          # the merge distance, by the merge's own great-circle distance
MATCH_DEG = 2.0               # coordinate degrees, a candidate matched to a position or another candidate
RECOVERY_DEG = 5.0            # coordinate degrees, a missing time counted as recovered
LIST_DEG = 10.0               # coordinate degrees, candidates listed around a comparison location
SEGMENT_TOL_DEG, SEGMENT_ALT_DEG = 3.0, 5.0   # the R1 pilot reference's tolerance and the distance reported beside it


def coord_distance(lat1, lon1, lat2, lon2):
    return float(np.hypot(float(lat1) - float(lat2), float(lon1) - float(lon2)))


def gc_distance_deg(lat1, lon1, lat2, lon2):
    from aew.v1port.contours import great_circle_distance
    return float(great_circle_distance(float(lat1), float(lon1), np.array([float(lat2)]), np.array([float(lon2)]), "nm")[0]) / 60.0


def final_tuple(w):
    return [float(w["lat_mean"]), float(w["lon_mean"]), int(np.asarray(w["lat_wave"]).size), R.region_digest(w["region"])]


def close_pairs(finals, limit=CLOSE_PAIR_DEG):
    """Pairs of final candidates closer than the limit by the merge's great-circle distance."""
    out = []
    for i, j in itertools.combinations(range(len(finals)), 2):
        d = gc_distance_deg(finals[i][0], finals[i][1], finals[j][0], finals[j][1])
        if d < limit:
            out.append({"i": i, "j": j, "distance_deg": round(d, 3)})
    return out


def pair_preexisting(pair, finals_v, baseline_pairs, finals_b, tol=MATCH_DEG):
    """Whether the baseline has a pair at this step whose two members lie within `tol` of
    the variant pair's two members, in either order."""
    a, b = finals_v[pair["i"]], finals_v[pair["j"]]
    for bp in baseline_pairs:
        c, d = finals_b[bp["i"]], finals_b[bp["j"]]
        if (coord_distance(a[0], a[1], c[0], c[1]) <= tol and coord_distance(b[0], b[1], d[0], d[1]) <= tol) or \
           (coord_distance(a[0], a[1], d[0], d[1]) <= tol and coord_distance(b[0], b[1], c[0], c[1]) <= tol):
            return True
    return False


def within(finals, lat, lon, radius):
    """Indices of final candidates within `radius` coordinate degrees, nearest first."""
    hits = [(coord_distance(f[0], f[1], lat, lon), i) for i, f in enumerate(finals)]
    return [{"index": i, "distance_deg": round(d, 3), "lat": finals[i][0], "lon": finals[i][1], "n_points": finals[i][2]} for d, i in sorted(hits) if d <= radius]


def added_and_removed(finals_v, finals_b, tol=MATCH_DEG):
    """Variant candidates with no baseline candidate within `tol`, and the reverse."""
    added = [i for i, f in enumerate(finals_v) if not any(coord_distance(f[0], f[1], g[0], g[1]) <= tol for g in finals_b)]
    removed = [i for i, g in enumerate(finals_b) if not any(coord_distance(g[0], g[1], f[0], f[1]) <= tol for f in finals_v)]
    return added, removed


def in_box(lat, lon, box):
    return bool(box[0] <= lat <= box[1] and box[2] <= lon <= box[3])


def compact_entry(e):
    """A lineage entry without arrays, the regions replaced by digests."""
    out = {k: v for k, v in e.items() if k != "select"}
    sel = e.get("select")
    if sel:
        out["levels"] = [[l["cells"], round(l["lat_extent"], 2), round(l["lon_extent"], 2)] for l in sel["levels"]]
        out["level_landed"], out["exhausted"] = sel["level_landed"], bool(sel["exhausted"])
        out["selected"] = {"cells": sel["selected_cells"], "lat_extent": round(sel["selected_lat_extent"], 2), "lon_extent": round(sel["selected_lon_extent"], 2),
                           "sha256": R.region_digest(sel["region"])}
        out["base_sha256"] = R.region_digest(sel["base_region"])
        if "clip" in sel:
            out["clip"] = sel["clip"]
    return out


def chain_for_final(step_rec, f):
    """The final candidate's origin: the fine input, the coarse output, the coarse input
    axis, and whether that coarse selection ran out or was clipped."""
    fine = next((e for e in step_rec["fine_lineage"] if e.get("output") == f), None)
    if fine is None:
        return None
    coarse = next((e for e in step_rec["coarse_lineage"] if e.get("output") == fine["input"]), None)
    return {"fine_input": fine["input"], "coarse_output": fine["input"], "axis": None if coarse is None else coarse["input"],
            "coarse_exhausted": None if coarse is None else coarse.get("exhausted"), "coarse_clipped": None if coarse is None else bool(coarse.get("clip")),
            "coarse_base_sha256": None if coarse is None else coarse.get("base_sha256"), "fine_exhausted": fine.get("exhausted")}


def selection_invariance(baseline_step, variant_step, step, include_exhausted=False):
    """F1 at one step: every coarse input whose ladder did not run out in the baseline
    must select the same region in the variant, and with `include_exhausted` every input
    whatever its ladder did. One helper serves the measure and the immediate stop, so the
    gate and the record cannot drift apart."""
    variant = {e["input"]: e for e in variant_step["coarse_lineage"] if e.get("merged") is not None}
    checked, violations = 0, []
    for b in baseline_step["coarse_lineage"]:
        if b.get("merged") is None or (b.get("exhausted") and not include_exhausted):
            continue
        checked += 1
        e = variant.get(b["input"])
        if e is None or e["selected"]["sha256"] != b["selected"]["sha256"]:
            violations.append({"step": int(step), "input": b["input"]})
    return checked, violations


def variant_step(baseline_step, k, date, off, run_variant, strict=False):
    """One variant step: run the variant, then REFUSE before anything is stored or any
    further step runs if a non-exhausted selection changed, or with `strict` any
    selection (F1 is an immediate stop under the contract), otherwise return the step
    record."""
    on = run_variant()
    _, violations = selection_invariance(baseline_step, on, k, include_exhausted=strict)
    if violations:
        raise SystemExit(f"REFUSED: at step {k} a non-exhausted selection changed with the flag on (F1), nothing further runs: {violations[:5]}")
    return {"step": k, "date": date, **on, "flag_off_final": off["final"]}


def band_windows(stationary, box):
    """One window per stationary history over its whole lifetime, first to last stored
    step with the gaps between, in the declared box, for the D3 and E measures."""
    return {name: {"first": min(h["steps"]), "last": max(h["steps"]), "box": [float(x) for x in box]} for name, h in stationary.items()}


def input_fate(step_rec, inp):
    """One coarse input followed to its final candidate: kept or dropped in the duplicate
    pass, rejected for extent, dropped in the fine merge, or the final it became, with the
    coordinate distance from its own axis mean to that final."""
    e = next((x for x in step_rec["coarse_lineage"] if x["input"] == inp and x.get("merged") is not None), None)
    if e is None:
        return None
    out = e.get("output")
    final = None
    if out is not None:
        fine = next((x for x in step_rec["fine_lineage"] if x["input"] == out and x.get("merged") is not None), None)
        final = None if fine is None else fine.get("output")
    pos = None if final is None else [step_rec["final"][final][0], step_rec["final"][final][1]]
    kept = e.get("kept_after_pass_two", True if e.get("lone_wave") else None)
    removed_at = "duplicate_pass" if kept is False else "min_extent" if e.get("rejected_min_extent") else "fine_merge" if out is not None and final is None else None
    return {"kept_after_pass_two": kept, "coarse_output": out, "final": final, "final_position": pos, "removed_at": removed_at,
            "relocation_deg": None if pos is None or e.get("lat_mean") is None else round(coord_distance(e["lat_mean"], e["lon_mean"], pos[0], pos[1]), 3)}


def input_fates(base_step, var_step):
    """Every coarse input of the step, its fate in both runs, listed where they differ."""
    rows, counts = [], {"inputs": 0, "changed": 0, "removed_baseline": 0, "removed_variant": 0, "newly_removed": 0, "newly_surviving": 0}
    for e in base_step["coarse_lineage"]:
        if e.get("merged") is None:
            continue
        counts["inputs"] += 1
        fb, fv = input_fate(base_step, e["input"]), input_fate(var_step, e["input"])
        rb, rv = fb is None or fb["final"] is None, fv is None or fv["final"] is None
        counts["removed_baseline"] += rb; counts["removed_variant"] += rv
        counts["newly_removed"] += (rv and not rb); counts["newly_surviving"] += (rb and not rv)
        if fb != fv:
            counts["changed"] += 1
            rows.append({"input": e["input"], "axis": None if e.get("lat_mean") is None else [round(e["lat_mean"], 3), round(e["lon_mean"], 3)],
                         "exhausted": e.get("exhausted"), "baseline": fb, "variant": fv})
    return {"counts": counts, "changed": rows}


def segment_correspondence(step_rec, seg):
    """Final candidates against a declared axis segment by the R1 pilot's distance: the
    nearest with its origin, the counts within the tolerance and the distance beside it,
    and the candidates listed within the listing radius."""
    import pilot_r1_measurement as RM
    finals = step_rec["final"]
    d = [RM.axis_distance(f[0], f[1], seg["axis_lon"], seg["lat_range"]) for f in finals]
    order = sorted(range(len(finals)), key=lambda i: d[i])
    near = order[0] if order else None
    return {"nearest": None if near is None else {"index": near, "lat": finals[near][0], "lon": finals[near][1], "n_points": finals[near][2],
                                                  "distance_deg": round(d[near], 3), "chain": chain_for_final(step_rec, near)},
            "within_tol": sum(x <= SEGMENT_TOL_DEG for x in d), "within_alt": sum(x <= SEGMENT_ALT_DEG for x in d),
            "listed": [{"index": i, "lat": finals[i][0], "lon": finals[i][1], "distance_deg": round(d[i], 3)} for i in order if d[i] <= LIST_DEG]}


def run_step(wide, k, sel, t, ct, ft, grids, absorb, clip, from_input=False):
    fields = A.prepared_fields(wide, k, *sel)
    lin = {}
    kw = {} if clip is None else {"coarse_clip_radius_deg": clip}
    if from_input:
        kw["coarse_position_from_input"] = True
    final = A.detect_from_prepared(fields, t, ct, ft, grids, absorb=absorb, lineage=lin, **kw)
    return {"final": [final_tuple(w) for w in final], "axes": len(lin.get("axes", [])),
            "coarse_lineage": [compact_entry(e) for e in lin.get("coarse", [])],
            "fine_lineage": [compact_entry(e) for e in lin.get("fine", [])],
            "coarse_exhausted_inputs": sum(1 for e in lin.get("coarse", []) if e.get("select") and e["select"]["exhausted"]),
            "coarse_inputs": len(lin.get("coarse", []))}


def baseline_measures(steps, declared):
    """What the baseline records on its own: close pairs, counts near the comparison
    locations and the stored positions, and the exhaustion census inside the windows."""
    m = {"close_pairs": {k: close_pairs(s["final"]) for k, s in steps.items()},
         "near_missing": {}, "near_controls": {}, "near_pre_arrival": {}, "stationary": {},
         "exhaustion_in_windows": {k: {"coarse_inputs": s["coarse_inputs"], "coarse_exhausted": s["coarse_exhausted_inputs"], "final": len(s["final"])} for k, s in steps.items()}}
    for key, radius in (("missing", RECOVERY_DEG), ("controls", MATCH_DEG), ("pre_arrival", MATCH_DEG)):
        for step, lat, lon in declared.get(key, []):
            m["near_" + key][str(step)] = {"location": [lat, lon], "within": within(steps[str(step)]["final"], lat, lon, radius), "listed": within(steps[str(step)]["final"], lat, lon, LIST_DEG)}
    for name, h in declared.get("stationary", {}).items():
        m["stationary"][name] = [{"step": s, "stored": [la, lo], "within": within(steps[str(s)]["final"], la, lo, MATCH_DEG)} for s, la, lo in zip(h["steps"], h["lat"], h["lon"])]
    m["segments"] = [{**seg, **segment_correspondence(steps[str(seg["step"])], seg)} for seg in declared.get("segments", [])]
    return m


def evaluate(base, var, declared, clip_radius_deg=None):
    """The variant's measures against the baseline, step for step."""
    bs, vs = base["steps"], var["steps"]
    out = {"A": {"missing": [], "controls": []}, "C": [], "B": {"baseline_pairs": 0, "variant_pairs": 0, "new_pairs": []},
           "D1": {}, "D2": [], "D3": {}, "E": {}, "F1": {"violations": [], "non_exhausted_inputs_checked": 0}, "F3": {"differing_finals": []}, "changes": {},
           "G": {"coarse_inputs": 0, "exhausted_selections": 0, "clipped_selections": 0, "seed_beyond_radius": [], "clipped_dropped_pieces_total": 0,
                 "clipped_kept_cells": {"min": None, "max": None}}}
    for k in vs:
        for e in vs[k]["coarse_lineage"]:
            if e.get("merged") is None:
                continue
            out["G"]["coarse_inputs"] += 1
            out["G"]["exhausted_selections"] += bool(e.get("exhausted"))
            if e.get("clip"):
                out["G"]["clipped_selections"] += 1
                out["G"]["clipped_dropped_pieces_total"] += e["clip"]["dropped_pieces"]
                kc = e["clip"]["kept_cells"]; g = out["G"]["clipped_kept_cells"]
                g["min"] = kc if g["min"] is None else min(g["min"], kc); g["max"] = kc if g["max"] is None else max(g["max"], kc)
                if clip_radius_deg is not None and e.get("seed_distance_deg", 0.0) > clip_radius_deg:
                    out["G"]["seed_beyond_radius"].append({"step": int(k), "input": e["input"], "seed_distance_deg": round(e["seed_distance_deg"], 3)})
    for step, lat, lon in declared.get("missing", []):
        hits = within(vs[str(step)]["final"], lat, lon, RECOVERY_DEG)
        out["A"]["missing"].append({"step": step, "location": [lat, lon], "recovered": bool(hits), "nearest": hits[0] if hits else None})
        out["C"].append({"step": step, "within_5": len(hits), "unique": len(hits) <= 1, "listed_10": within(vs[str(step)]["final"], lat, lon, LIST_DEG)})
    for step, lat, lon in declared.get("controls", []):
        hits = within(vs[str(step)]["final"], lat, lon, MATCH_DEG)
        out["A"]["controls"].append({"step": step, "frozen": [lat, lon], "held": bool(hits), "nearest": hits[0] if hits else None, "listed_10": within(vs[str(step)]["final"], lat, lon, LIST_DEG)})
    for k in vs:
        fb, fv = bs[k]["final"], vs[k]["final"]
        pb, pv = close_pairs(fb), close_pairs(fv)
        out["B"]["baseline_pairs"] += len(pb); out["B"]["variant_pairs"] += len(pv)
        for p in pv:
            if not pair_preexisting(p, fv, pb, fb):
                ci, cj = chain_for_final(vs[k], p["i"]), chain_for_final(vs[k], p["j"])
                out["B"]["new_pairs"].append({"step": int(k), "date": vs[k]["date"], "distance_deg": p["distance_deg"],
                                              "members": [{"lat": fv[p["i"]][0], "lon": fv[p["i"]][1], "n_points": fv[p["i"]][2], "chain": ci},
                                                          {"lat": fv[p["j"]][0], "lon": fv[p["j"]][1], "n_points": fv[p["j"]][2], "chain": cj}],
                                              "either_clipped": bool((ci or {}).get("coarse_clipped") or (cj or {}).get("coarse_clipped")),
                                              "share_base_component": bool(ci and cj and ci["coarse_base_sha256"] and ci["coarse_base_sha256"] == cj["coarse_base_sha256"])})
        added, removed = added_and_removed(fv, fb)
        out["changes"][k] = {"date": vs[k]["date"], "final_baseline": len(fb), "final_variant": len(fv), "added": [fv[i] for i in added], "removed": [fb[i] for i in removed],
                             "added_chains": [chain_for_final(vs[k], i) for i in added]}
        # F1: every non-exhausted coarse input selects the same region flag on and off
        from_input = var.get("coarse_position") == "input"
        checked, violations = selection_invariance(bs[k], vs[k], k, include_exhausted=from_input)
        out["F1"]["non_exhausted_inputs_checked"] += checked
        if from_input:
            out.setdefault("input_fates", {})[k] = input_fates(bs[k], vs[k])
        out["F1"]["violations"].extend(violations)
        # F3, both directions: variant finals absent from the baseline, and baseline finals
        # absent from the variant, each with its origin; a non-exhausted origin on either
        # side is a downstream interaction (the order-dependent duplicate pass)
        for side, rec, finals, other in (("variant", vs[k], fv, fb), ("baseline", bs[k], fb, fv)):
            for i, f in enumerate(finals):
                if f not in other:
                    c = chain_for_final(rec, i)
                    out["F3"]["differing_finals"].append({"step": int(k), "side": side, "final": f, "chain": c, "downstream_interaction": bool(c and c["coarse_exhausted"] is False),
                                                          "other_side_within_2": bool(any(coord_distance(f[0], f[1], g[0], g[1]) <= MATCH_DEG for g in other))})
    out["segments"] = [{**seg, "baseline": segment_correspondence(bs[str(seg["step"])], seg), "variant": segment_correspondence(vs[str(seg["step"])], seg)}
                       for seg in declared.get("segments", [])]
    for name, h in declared.get("stationary", {}).items():
        rows = []
        for s, la, lo in zip(h["steps"], h["lat"], h["lon"]):
            nb, nv = len(within(bs[str(s)]["final"], la, lo, MATCH_DEG)), len(within(vs[str(s)]["final"], la, lo, MATCH_DEG))
            rows.append({"step": s, "stored": [la, lo], "baseline_within_2": nb, "variant_within_2": nv, "added": nv > nb})
        out["D1"][name] = {"steps": len(rows), "steps_with_addition": sum(r["added"] for r in rows), "rows": rows}
    for step, lat, lon in declared.get("pre_arrival", []):
        nb, nv = len(within(bs[str(step)]["final"], lat, lon, MATCH_DEG)), len(within(vs[str(step)]["final"], lat, lon, MATCH_DEG))
        out["D2"].append({"step": step, "location": [lat, lon], "baseline_within_2": nb, "variant_within_2": nv, "added": nv > nb})
    for wname, w in declared["windows"].items():
        box = w["box"]; prev_added = []
        rows = []
        for s in range(w["first"], w["last"] + 1):
            k = str(s)
            fb, fv = bs[k]["final"], vs[k]["final"]
            added, removed = added_and_removed(fv, fb)
            added_in_box = [fv[i] for i in added if in_box(fv[i][0], fv[i][1], box)]
            persistent = [f for f in added_in_box if any(coord_distance(f[0], f[1], g[0], g[1]) <= MATCH_DEG for g in prev_added)]
            base_in_box = [f for f in fb if in_box(f[0], f[1], box)]
            unmatched = [f for f in base_in_box if not any(coord_distance(f[0], f[1], g[0], g[1]) <= MATCH_DEG for g in fv)]
            rows.append({"step": s, "added_in_box": len(added_in_box), "added_persistent": len(persistent), "baseline_in_box": len(base_in_box), "baseline_unmatched": unmatched})
            prev_added = added_in_box
        out["D3"][wname] = {"box": box, "added_in_box_total": sum(r["added_in_box"] for r in rows), "added_persistent_total": sum(r["added_persistent"] for r in rows),
                            "steps_with_added": sum(1 for r in rows if r["added_in_box"]), "rows": [{k2: v for k2, v in r.items() if k2 != "baseline_unmatched"} for r in rows]}
        out["E"][wname] = {"box": box, "baseline_in_box_total": sum(r["baseline_in_box"] for r in rows), "unmatched_total": sum(len(r["baseline_unmatched"]) for r in rows),
                           "unmatched": [{"step": r["step"], "final": f} for r in rows for f in r["baseline_unmatched"]]}
    return out


def load_frozen(args):
    run = P.load_run(args.control_run, year=args.year)
    case_b = SD.load_case(args.control_case, run["record"])
    wide = A.load_wide(args.wide_dir, args.year)
    if wide["record"]["control_run"]["case_sha256"] != case_b["sha256"]:
        raise SystemExit("REFUSED: the wide case was not bound to this control case")
    ds = run["record"]["dataset_specific"]
    ct, ft, thresholds = A.thresholds_for_run((float(ds["coarse_threshold"]), float(ds["fine_threshold"])), args.calibration, False, args.domain_calibration, run["domain"])
    grids = MR.control_grids(case_b)
    sel = A.selection(case_b, wide)
    raw = A.raw_equality(case_b, wide, sel)
    if any(raw.values()):
        raise SystemExit(f"REFUSED: the wide case differs from the control case on the control's own cells: {raw}")
    return run, case_b, wide, ct, ft, thresholds, grids, sel


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=("baseline", "variant"), required=True)
    for name in ("--control-run", "--control-case", "--wide-dir", "--calibration", "--replay", "--out"):
        ap.add_argument(name, required=True)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--domain-calibration", action="store_true")
    ap.add_argument("--trace", help="the retained event trace; baseline mode refuses unless it reproduces its candidates at its steps")
    ap.add_argument("--baseline", help="variant mode: the baseline record this run is compared with")
    ap.add_argument("--clip-radius", type=float, help="variant mode: the predeclared clip radius in coordinate degrees")
    ap.add_argument("--coarse-position", choices=("median", "input"), default="median",
                    help="variant mode: 'input' gives coarse pass-one waves their input axis's mean position instead of the region median")
    ap.add_argument("--segment", nargs=4, type=float, action="append", default=[], metavar=("STEP", "AXIS_LON", "LAT0", "LAT1"),
                    help="a declared axis segment at a step, against which final candidates are measured by the R1 pilot's distance")
    ap.add_argument("--window", nargs=7, action="append", default=[], metavar=("NAME", "FIRST", "LAST", "LAT0", "LAT1", "LON0", "LON1"))
    ap.add_argument("--stationary", nargs=2, action="append", default=[], metavar=("NAME", "HISTORY_INDEX"))
    ap.add_argument("--missing", nargs=3, type=float, action="append", default=[], metavar=("STEP", "LAT", "LON"))
    ap.add_argument("--control", nargs=3, type=float, action="append", default=[], metavar=("STEP", "LAT", "LON"))
    ap.add_argument("--pre-arrival", nargs=3, type=float, action="append", default=[], metavar=("STEP", "LAT", "LON"))
    ap.add_argument("--band-box", nargs=4, type=float, default=[5.0, 25.0, 45.0, 60.0], metavar=("LAT0", "LAT1", "LON0", "LON1"),
                    help="the eastern band box in which D3 and E are measured over each stationary history's lifetime")
    args = ap.parse_args(argv)
    if os.path.exists(args.out):
        raise SystemExit(f"REFUSED: {args.out} exists and a record is never overwritten")
    if args.mode == "variant" and (args.baseline is None or (args.clip_radius is None) == (args.coarse_position == "median")):
        raise SystemExit("REFUSED: variant mode needs --baseline and exactly one variant, --clip-radius or --coarse-position input")
    if args.mode == "baseline" and (args.clip_radius is not None or args.coarse_position != "median"):
        raise SystemExit("REFUSED: baseline mode runs the frozen candidate only")
    run, case_b, wide, ct, ft, thresholds, grids, sel = load_frozen(args)
    flags = run["record"]["dataset_specific"]["tracker_flags"]
    absorb = bool(flags["absorb"])
    times = np.asarray(case_b["time"], float).ravel()
    with gzip.open(args.replay, "rb") as fh:
        blob = fh.read()
    replay = json.loads(blob); replay_sha = hashlib.sha256(blob).hexdigest()
    if (replay["thresholds"]["coarse"], replay["thresholds"]["fine"]) != (ct, ft) or replay["runs"]["wide"]["case_sha256"] != wide["sha256"]:
        raise SystemExit("REFUSED: the replay record is not the frozen candidate's on this wide case")
    declared = {"windows": {}, "stationary": {}, "missing": [(int(s), la, lo) for s, la, lo in args.missing],
                "controls": [(int(s), la, lo) for s, la, lo in args.control], "pre_arrival": [(int(s), la, lo) for s, la, lo in args.pre_arrival],
                "segments": [{"step": int(s), "axis_lon": float(a), "lat_range": [float(l0), float(l1)]} for s, a, l0, l1 in args.segment],
                "radii": {"close_pair_gc_deg": CLOSE_PAIR_DEG, "match_coord_deg": MATCH_DEG, "recovery_coord_deg": RECOVERY_DEG, "list_coord_deg": LIST_DEG}}
    steps_needed = set()
    for name, first, last, la0, la1, lo0, lo1 in args.window:
        declared["windows"][name] = {"first": int(first), "last": int(last), "box": [float(la0), float(la1), float(lo0), float(lo1)]}
        steps_needed.update(range(int(first), int(last) + 1))
    for name, idx in args.stationary:
        # The anchor at each stored step is the history's RAW detection position, the
        # candidate the tracker took at that step, not the smoothed track position, which
        # can sit more than 2 degrees from any candidate. Both are recorded.
        h = replay["finished"][int(idx)]
        declared["stationary"][name] = {"history_index": int(idx), "steps": [int(s) for s in h["steps"]], "lat": [float(x) for x in h["raw_lat"]], "lon": [float(x) for x in h["raw_lon"]],
                                        "smoothed_lat": [float(x) for x in h["lat"]], "smoothed_lon": [float(x) for x in h["lon"]], "anchor": "raw detection positions",
                                        "first": {"date": f"{SD.date_of(float(h['time'][0])):%Y-%m-%d %HZ}", "lat": float(h["raw_lat"][0]), "lon": float(h["raw_lon"][0])},
                                        "last": {"date": f"{SD.date_of(float(h['time'][-1])):%Y-%m-%d %HZ}", "lat": float(h["raw_lat"][-1]), "lon": float(h["raw_lon"][-1])}}
        steps_needed.update(int(s) for s in h["steps"])
    for name, w in band_windows(declared["stationary"], args.band_box).items():
        declared["windows"][name] = w
        steps_needed.update(range(w["first"], w["last"] + 1))
    for key in ("missing", "controls", "pre_arrival"):
        steps_needed.update(s for s, _, _ in declared[key])
    steps_needed.update(seg["step"] for seg in declared["segments"])
    trace_by_step, trace_sha = {}, None
    if args.trace:
        with gzip.open(args.trace, "rb") as fh:
            tblob = fh.read()
        trace_rec = json.loads(tblob); trace_sha = hashlib.sha256(tblob).hexdigest()
        if (trace_rec["thresholds"]["coarse"], trace_rec["thresholds"]["fine"]) != (ct, ft) or trace_rec["runs"]["B"]["case_sha256"] != case_b["sha256"]:
            raise SystemExit("REFUSED: the trace record is not of this control case under this pair")
        trace_by_step = {s["step"]: [[float(c["lat_mean"]), float(c["lon_mean"]), int(c["n_points"]), c["region_sha256"]] for c in s["candidates"]] for s in trace_rec["trace"]["steps"]}
    base_rec, base_sha = None, None
    if args.mode == "variant":
        with gzip.open(args.baseline, "rb") as fh:
            bblob = fh.read()
        base_rec = json.loads(bblob); base_sha = hashlib.sha256(bblob).hexdigest()
        if base_rec["mode"] != "baseline" or base_rec["year"] != args.year or base_rec["runs"]["wide"]["case_sha256"] != wide["sha256"] or base_rec["thresholds"]["coarse"] != ct:
            raise SystemExit("REFUSED: the baseline record is not of this frozen candidate")
        missing_steps = sorted(s for s in steps_needed if str(s) not in base_rec["steps"])
        if missing_steps:
            raise SystemExit(f"REFUSED: the baseline record holds no step {missing_steps[:5]}")
    steps, gates = {}, []
    for k in sorted(steps_needed):
        t = float(times[k])
        date = f"{SD.date_of(t):%Y-%m-%d %HZ}"
        off = run_step(wide, k, sel, t, ct, ft, grids, absorb, None)
        if k in trace_by_step:
            same = off["final"] == trace_by_step[k]
            gates.append({"step": k, "against": "trace", "reproduces": same})
            if not same:
                raise SystemExit(f"REFUSED: at step {k} the flag-off detection does not reproduce the retained trace")
        if base_rec is not None:
            same = off["final"] == base_rec["steps"][str(k)]["final"]
            gates.append({"step": k, "against": "baseline", "reproduces": same})
            if not same:
                raise SystemExit(f"REFUSED: at step {k} the flag-off detection does not reproduce the baseline record (F2)")
        if args.mode == "baseline":
            steps[str(k)] = {"step": k, "date": date, **off}
        else:
            steps[str(k)] = variant_step(base_rec["steps"][str(k)], k, date, off,
                                         lambda: run_step(wide, k, sel, t, ct, ft, grids, absorb, args.clip_radius, from_input=args.coarse_position == "input"),
                                         strict=args.coarse_position == "input")
        print(f"{date} step {k}: axes {steps[str(k)]['axes']}, coarse inputs {steps[str(k)]['coarse_inputs']}, exhausted {steps[str(k)]['coarse_exhausted_inputs']}, final {len(steps[str(k)]['final'])}" + ("" if args.mode == "baseline" else f" (flag off {len(off['final'])})"))
    out = {"generated_by": "scripts/pilot_merge_clip_experiment.py", "script_sha256": X.digest(os.path.abspath(__file__)), "year": args.year, "mode": args.mode,
           "runs": {"B": {"dir": args.control_run, "record_sha256": run["record_sha256"], "tracks_sha256": run["tracks_sha256"], "case_sha256": case_b["sha256"], "control_domain": run["domain"]},
                    "wide": {"dir": args.wide_dir, "record_sha256": wide["record_sha256"], "case_sha256": wide["sha256"], "box": wide["record"]["box"]}},
           "thresholds": thresholds, "tracker_flags": flags, "replay": {"path": args.replay, "sha256": replay_sha},
           "trace": None if not args.trace else {"path": args.trace, "sha256": trace_sha},
           "baseline": None if base_rec is None else {"path": args.baseline, "sha256": base_sha}, "clip_radius_deg": args.clip_radius,
           "coarse_position": args.coarse_position,
           "declared": declared, "gates": gates, "steps": steps}
    out["measures"] = baseline_measures(steps, declared) if args.mode == "baseline" else evaluate(base_rec, out, declared, args.clip_radius)
    payload = json.dumps(out, sort_keys=True, indent=1, default=float).encode()
    with gzip.open(args.out, "wb") as fh:
        fh.write(payload)
    print(f"Gates {len(gates)} checked, all reproducing. Record {args.out}, json sha256 {hashlib.sha256(payload).hexdigest()[:12]}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
