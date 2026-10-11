#!/usr/bin/env python3
"""The 2006 baseline measurement of `EASTERN_EVALUATION_ADDENDUM_2006_2026-10-10.md`.

Reads the declaration (`eval60_2006_reference_declaration.json`), the traced 2006 replay of
the frozen candidate (`scripts/pilot_alledge_replay.py` with that declaration, every family
traced, and the retained record as `--reproduces`) and the retained finished record, and
writes the design's section 5 measurements for the declared segments at their inspected
maps only. It reuses the R1 pilot's binding, input contract, observations, coverage
categories and displacement categories (`scripts/pilot_r1_measurement.py`, unchanged), and
replaces only what that script fixed for R1: segments, boxes, transitions and inspected
intervals come from the declaration, a read axis may be a longitude range, every map
carries whether its read axis lies west of 40 E with the segment summaries split on it,
and finished coverage is reported per declared segment. It reports, at every map, whether
a candidate within tolerance was seeded, claimed or left unclaimed. It makes no claim
about why a map is uncovered, which is the lineage reading's question.

    python3 scripts/pilot_eastern_measurement.py --reference <declaration.json> --trace <trace.json.gz> \\
        --finished <retained.json.gz> --out <record.json> --trace-digest <first 12 of the logged digest>

THE DISTANCE, declared before matching. From a position to a map's read geometry it is the
Euclidean distance in coordinate degrees to the nearest point of the rectangle formed by the
read longitude (a value or a declared range) and the read latitude range. With a single
read longitude it equals the R1 pilot's distance to the axis segment.
"""
import argparse
import gzip
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pilot_r1_measurement as M  # noqa: E402

STATIONARY_DEG = M.STATIONARY_DEG


def lon_range(m):
    r = m.get("axis_lon_range")
    if r is not None:
        return float(r[0]), float(r[1])
    return float(m["axis_lon"]), float(m["axis_lon"])


def ref_distance(lat, lon, m):
    """Euclidean degrees to the nearest point of the read longitude (value or range) by the
    read latitude range."""
    lo, hi = lon_range(m)
    dlon = max(lo - float(lon), 0.0, float(lon) - hi)
    la0, la1 = float(m["lat_range"][0]), float(m["lat_range"][1])
    dlat = max(la0 - float(lat), 0.0, float(lat) - la1)
    return math.hypot(dlon, dlat)


def west_of_40(m):
    return lon_range(m)[1] < 40.0


def measure_map(entry, m, box, tol, alt):
    """Detection and active association at one inspected map, with the claim status of each
    candidate near the read geometry."""
    held = {}
    for o in M.observations(entry):
        held.setdefault(o["candidate"], []).append({"birth": o["birth"], "kind": o["kind"]})
    cands = []
    for c in entry["candidates"]:
        d = ref_distance(c["lat_mean"], c["lon_mean"], m)
        cands.append({"index": c["index"], "lat": float(c["lat_mean"]), "lon": float(c["lon_mean"]), "n_points": int(c["n_points"]),
                      "region_sha256": c["region_sha256"], "distance_deg": round(d, 3), "class": M.classify(d, tol, alt),
                      "in_case_box": M.in_box(float(c["lat_mean"]), float(c["lon_mean"]), box), "held_by": held.get(c["index"], [])})
    nearest = min(cands, key=lambda c: c["distance_deg"]) if cands else None
    obs = []
    for o in M.observations(entry):
        d = ref_distance(o["lat"], o["lon"], m)
        obs.append({**o, "distance_deg": round(d, 3), "class": M.classify(d, tol, alt)})
    observed = {o["birth"] for o in obs}
    alive_without = []
    for row in entry["live_before"]:
        if row["birth"] in observed:
            continue
        d = ref_distance(row["last_lat"], row["last_lon"], m)
        if d <= tol:
            alive_without.append({"birth": row["birth"], "last_step": int(row["last_step"]), "last_lat": float(row["last_lat"]), "last_lon": float(row["last_lon"]),
                                  "distance_deg": round(d, 3), "n_obs": int(row["n_obs"]), "due": row.get("due")})
    within = [c for c in cands if c["class"] == "within"]
    return {"date": M.date_key(entry["date"]), "step": int(entry["step"]), "segment": m["segment"], "scored": bool(m["scored"]),
            "read_lon": list(lon_range(m)), "lat_range": m["lat_range"], "west_of_40E": west_of_40(m),
            "detection": {"candidates_in_case_box": sum(1 for c in cands if c["in_case_box"]), "candidates_total": len(cands),
                          "within": within, "ambiguous": [c for c in cands if c["class"] == "ambiguous"], "nearest": nearest,
                          "within_unclaimed": [c["index"] for c in within if not c["held_by"]]},
            "association": {"observations_within": [o for o in obs if o["class"] == "within"], "observations_ambiguous": [o for o in obs if o["class"] == "ambiguous"],
                            "alive_within_without_observation": alive_without, "n_live_before": int(entry["n_live_before"])}}


def transition_report(r_from, r_to, histories, fates, pruned_at, finished_index):
    """The histories within tolerance on the two maps of a declared transition, with their
    positions, regions and lifecycle, and what the configuration did. Reported, never scored."""
    obs_a, obs_b = r_from["association"]["observations_within"], r_to["association"]["observations_within"]
    on_b = {o["birth"]: o for o in obs_b}
    rows_a = []
    for o in obs_a:
        h = histories.get(o["birth"])
        if o["birth"] in on_b:
            outcome = "continues"
        elif h is None:
            outcome = "unknown, raw history not retained"
        elif int(h["steps"][-1]) <= r_from["step"]:
            outcome = "ends"
        else:
            outcome = "observes elsewhere after the first map"
        rows_a.append({"birth": o["birth"], "position_on_first_map": [o["lat"], o["lon"]], "distance_deg": o["distance_deg"], "region_sha256": o["region_sha256"],
                       "outcome": outcome, "position_on_second_map": [on_b[o["birth"]]["lat"], on_b[o["birth"]]["lon"]] if o["birth"] in on_b else None,
                       **M.lifecycle(o["birth"], histories, fates, pruned_at, finished_index)})
    a_births = {r["birth"] for r in rows_a}
    rows_b = [{"birth": o["birth"], "position_on_second_map": [o["lat"], o["lon"]], "distance_deg": o["distance_deg"], "region_sha256": o["region_sha256"],
               "newcomer": o["birth"] not in a_births, **M.lifecycle(o["birth"], histories, fates, pruned_at, finished_index)} for o in obs_b]
    continuing = [r["birth"] for r in rows_a if r["outcome"] == "continues"]
    newcomers = [r["birth"] for r in rows_b if r["newcomer"]]
    if not rows_a and not rows_b:
        what = "no history within tolerance on either map"
    elif continuing:
        what = "continues" + (", with newcomers" if newcomers else "")
    elif rows_a and newcomers:
        what = "replaced"
    elif rows_a:
        what = "ends or observes elsewhere, nothing within tolerance on the second map"
    else:
        what = "nothing within tolerance on the first map, newcomers on the second"
    return {"first_map": r_from["date"], "second_map": r_to["date"], "histories_on_first_map": rows_a, "histories_on_second_map": rows_b,
            "continuing": continuing, "newcomers_on_second_map": newcomers, "transition": what,
            "note": "identity across a transition is unresolved in the fields; this is what the configuration did, not a score"}


def interval_report(trace_steps, interval, histories, fates, pruned_at, finished_index, stationary=STATIONARY_DEG):
    """Produced histories observing inside a declared inspected interval's box at its
    12-hourly maps, which must all be logged, with candidates per map, raw histories,
    per-step and net displacements and the numerical category at both stationary
    parameters. The design's section 6 class needs the field reading and is not assigned."""
    expected = M.inspected_dates(interval["window"])
    entries = {M.date_key(e["date"]): e for e in trace_steps}
    missing = [d for d in expected if d not in entries]
    if missing:
        raise SystemExit(f"REFUSED: the interval {interval['name']}'s maps {missing} are not among the trace's logged steps")
    box = {"lat": interval["lat"], "lon": interval["lon"]}
    per_map = {d: sum(1 for c in entries[d]["candidates"] if M.in_box(float(c["lat_mean"]), float(c["lon_mean"]), box)) for d in expected}
    seen = {}
    for d in expected:
        for o in M.observations(entries[d]):
            if M.in_box(o["lat"], o["lon"], box):
                seen.setdefault(o["birth"], []).append(d)
    out, counts = [], {str(p): {} for p in stationary}
    for birth, dates in sorted(seen.items()):
        h = histories.get(birth)
        if h is None:
            raise SystemExit(f"REFUSED: the raw history of produced history {birth} is not retained in the trace")
        M.validate_history(birth, h)
        lons = [float(x) for x in h["lon_claimed"]]
        net = round(lons[-1] - lons[0], 3)
        cats = {str(p): M.displacement_category(net, p) for p in stationary}
        for p, c in cats.items():
            counts[p][c] = counts[p].get(c, 0) + 1
        out.append({"birth": birth, "maps_observed_in_box": sorted(dates), "fate": fates.get(birth), "pruned_at": pruned_at.get(birth), "finished_index": finished_index.get(birth),
                    "raw_history": {"steps": h["steps"], "lon_claimed": h["lon_claimed"], "lat_claimed": h["lat_claimed"]},
                    "per_step_lon_displacement_deg": [round(b - a, 3) for a, b in zip(lons[:-1], lons[1:])], "net_lon_displacement_deg": net, "n_obs": len(h["steps"]),
                    "displacement_category_by_stationary_parameter": cats, "section_6_class": M.SECTION_6})
    return {"interval": interval, "inspected_maps": expected, "candidates_in_box_per_map": per_map, "produced_histories": out,
            "displacement_category_counts_by_stationary_parameter": counts, "stationary_parameters_deg": list(stationary),
            "note": "inspected, not scored; the displacement category is numerical and the design's section 6 class needs the field reading"}


def finished_report(finished, maps, segments, step_of_date, tol, alt):
    """Finished histories with a raw position within the alternative at a scored map, raw and
    smoothed positions side by side, with coverage per declared segment."""
    out = []
    for idx, h in enumerate(finished):
        pos = {int(s): i for i, s in enumerate(h["steps"])}
        rows = []
        for m in maps:
            s = step_of_date.get(m["date"])
            if s is None or s not in pos:
                continue
            i = pos[s]
            raw_d = ref_distance(h["raw_lat"][i], h["raw_lon"][i], m)
            sm_d = ref_distance(h["lat"][i], h["lon"][i], m)
            rows.append({"date": m["date"], "segment": m["segment"], "scored": bool(m["scored"]), "west_of_40E": west_of_40(m),
                         "raw": [round(float(h["raw_lat"][i]), 3), round(float(h["raw_lon"][i]), 3)], "raw_distance_deg": round(raw_d, 3), "raw_class": M.classify(raw_d, tol, alt),
                         "smoothed": [round(float(h["lat"][i]), 3), round(float(h["lon"][i]), 3)], "smoothed_distance_deg": round(sm_d, 3), "smoothed_class": M.classify(sm_d, tol, alt)})
        if not any(r["raw_class"] != "none" and r["scored"] for r in rows):
            continue
        lons = [float(x) for x in h["raw_lon"]]
        crosses = any(lons[i] >= 40.0 and min(lons[i + 1:], default=99.0) < 40.0 for i in range(len(lons) - 1))
        cover = {}
        for seg in segments:
            seg_maps = [m["date"] for m in maps if m["segment"] == seg]
            got = [r["date"] for r in rows if r["segment"] == seg and r["raw_class"] == "within"]
            if seg_maps:
                cover[seg] = {"maps_within_raw": got, "share": round(len(got) / len(seg_maps), 3)}
        out.append({"finished_index": idx, "birth": h["birth"], "n_obs": len(h["steps"]), "first_step": int(h["steps"][0]), "last_step": int(h["steps"][-1]),
                    "first_raw": [round(float(h["raw_lat"][0]), 3), round(float(h["raw_lon"][0]), 3)], "last_raw": [round(float(h["raw_lat"][-1]), 3), round(float(h["raw_lon"][-1]), 3)],
                    "crosses_40E_raw": bool(crosses), "coverage": cover, "at_maps": rows})
    return out


def segment_summaries(per_map, seg, name):
    """The design's coverage categories for one segment, over all its maps and, where it has
    maps on both sides, east and west of 40 E apart."""
    maps = [r for r in per_map if r["segment"] == seg]
    out = {"all": M.segment_summary(maps, seg, name)}
    east, west = [r for r in maps if not r["west_of_40E"]], [r for r in maps if r["west_of_40E"]]
    if east and west:
        out["east_of_40E"] = M.segment_summary(east, seg, name)
        out["west_of_40E"] = M.segment_summary(west, seg, name)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("--reference", "--trace", "--finished", "--out", "--trace-digest"):
        ap.add_argument(name, required=True)
    args = ap.parse_args(argv)
    if os.path.exists(args.out):
        raise SystemExit(f"REFUSED: {args.out} exists and a record is never overwritten")
    trace_digest = M.sha256_decompressed(args.trace)
    if len(args.trace_digest) < 12 or not trace_digest.startswith(args.trace_digest):
        raise SystemExit("REFUSED: the trace's digest does not begin with the one the replay logged")
    decl = json.load(open(args.reference))
    ref, tols = decl["reference"], decl["tolerances"]
    tol, alt = float(tols["positional_deg"]), float(tols["alternative_deg"])
    with gzip.open(args.trace, "rb") as fh:
        trace = json.loads(fh.read())
    with gzip.open(args.finished, "rb") as fh:
        finished_rec = json.loads(fh.read())
    M.check_binding(trace, args.reference, finished_rec, args.finished)
    M.validate_trace(trace, finished_rec)
    tr = trace["trace"]
    entries = {M.date_key(e["date"]): e for e in tr["steps"]}
    step_of_date = {d: int(e["step"]) for d, e in entries.items()}
    segs = ref["segments"]
    for m in ref["maps"]:
        if m["date"] not in entries:
            raise SystemExit(f"REFUSED: the inspected map {m['date']} is not among the trace's logged steps")
        if m["segment"] not in segs:
            raise SystemExit(f"REFUSED: the map {m['date']} names an undeclared segment {m['segment']}")
    scored = [s for s, v in segs.items() if v["scored"]]
    results = {}
    for name, t in (("tolerance", tol), ("alternative", alt)):
        per_map = [measure_map(entries[m["date"]], m, segs[m["segment"]]["case_box"], t, alt if name == "tolerance" else t) for m in ref["maps"]]
        by_date = {}
        for r in per_map:
            by_date.setdefault(r["date"], {})[r["segment"]] = r
        transitions = []
        for tdecl in ref.get("transitions", []):
            a, b = by_date[tdecl["from"]["date"]][tdecl["from"]["segment"]], by_date[tdecl["to"]["date"]][tdecl["to"]["segment"]]
            transitions.append({"name": tdecl["name"], **transition_report(a, b, tr["histories"], tr["fates"], tr["pruned_at"], tr["finished_index"])})
        results[name] = {"degrees": t, "maps": per_map, "segments": {s: segment_summaries(per_map, s, name) for s in scored}, "transitions": transitions}
    intervals = [interval_report(tr["steps"], iv, tr["histories"], tr["fates"], tr["pruned_at"], tr["finished_index"]) for iv in ref.get("intervals", [])]
    fin = finished_report(finished_rec["finished"], ref["maps"], scored, step_of_date, tol, alt)
    summary = {}
    for name, r in results.items():
        rows = {}
        for s in scored:
            maps = [m for m in r["maps"] if m["segment"] == s]
            sm = r["segments"][s]["all"]
            rows[s] = {"maps": len(maps), "maps_with_a_candidate_within": [m["date"] for m in maps if m["detection"]["within"]],
                       "nearest_candidate_distance_by_map": {m["date"]: (m["detection"]["nearest"] or {}).get("distance_deg") for m in maps},
                       "maps_with_an_unclaimed_candidate_within": [m["date"] for m in maps if m["detection"]["within_unclaimed"]],
                       "maps_covered_by_a_history": sm["maps_covered"], "maps_uncovered_by_a_history": sm["maps_uncovered"],
                       "one_history_observes_every_map": sm["one_history_observes_every_map"], "histories_touching": sm["histories_touching_count"],
                       "overlapping_maps": len(sm["overlapping_coverage"]), "confirmed_duplicates": len(sm["confirmed_duplicates"]),
                       "west_of_40E_maps": [m["date"] for m in maps if m["west_of_40E"]]}
        summary[name] = {"segments": rows, "transitions": {t["name"]: t["transition"] for t in r["transitions"]}}
    summary["intervals"] = {iv["interval"]["name"]: iv["displacement_category_counts_by_stationary_parameter"] for iv in intervals}
    summary["finished_histories_near_a_scored_map"] = len(fin)
    out = {"generated_by": "scripts/pilot_eastern_measurement.py", "script_sha256": M.sha256_of(os.path.abspath(__file__)),
           "reuses": {"script": "scripts/pilot_r1_measurement.py", "sha256": M.sha256_of(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pilot_r1_measurement.py"))},
           "inputs": {"reference": {"path": args.reference, "sha256": M.sha256_of(args.reference)},
                      "trace": {"path": args.trace, "sha256": M.sha256_of(args.trace), "sha256_decompressed": trace_digest, "declared_digest": args.trace_digest,
                                "limit": "agreement with the fingerprint the replay logged, not the correctness or completeness of the replay's logging"},
                      "finished": {"path": args.finished, "sha256": M.sha256_of(args.finished), "sha256_decompressed": M.sha256_decompressed(args.finished)}},
           "input_contract_checked": True, "reference_declared": decl, "scored_only_at_inspected_maps": True,
           "replay": {"reproduces": trace.get("reproduces"), "elapsed_seconds": trace.get("elapsed_seconds"), "logged_steps": len(tr["steps"]), "configuration": trace.get("configuration")},
           "summary": summary, "results": results, "intervals": intervals,
           "finished_stage": {"note": "reported separately; raw and smoothed positions side by side; not required for detection", "finished_histories_near_a_scored_map": fin},
           "limits": "repeated observations of a small provisional case set, not independent validations; no claim about why a map is uncovered"}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    for name in ("tolerance", "alternative"):
        print(f"[{name} {results[name]['degrees']} deg]")
        for s, row in summary[name]["segments"].items():
            print(f"  {s}: {len(row['maps_with_a_candidate_within'])} of {row['maps']} maps with a candidate within; covered by a history {len(row['maps_covered_by_a_history'])}; "
                  f"histories touching {row['histories_touching']}; overlaps {row['overlapping_maps']}; confirmed duplicates {row['confirmed_duplicates']}; "
                  f"unclaimed-within maps {len(row['maps_with_an_unclaimed_candidate_within'])}")
        print(f"  transitions {summary[name]['transitions']}")
    print(f"intervals {summary['intervals']}; finished histories near a scored map {len(fin)}. Record {args.out}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
