#!/usr/bin/env python3
"""The inventory for the continuity diagnostic's conditional second pass (the brief's
section 8): in each season given, every cohort track of this record with first position
in 10 E to 30 E, 5 to 15 N, first observation in July, August or September, and outcome
other than Atlantic-side, listed with its candidate correspondences under the reader's
own window rule (`continuity_case_reading.candidates`, unchanged), keeping repeated
proximity (window steps within 500 km per candidate), the minimum distance, and the
number of candidates at every step. Then the predeclared selection: per season, among
the tracks with at least one candidate, the largest number of window steps within 500
km summed over candidates, ties by the earliest first observation then the lowest track
index, one per season where any, at most three in all. No field is read and no case is
judged here.

    .venv/bin/python3 scripts/continuity_inventory.py --years 1990 2002 2007 --campaign-evidence <dir> \\
        --qtrack-dir <dir> --regions-dir <dir> --measurement-artifacts <dir> --out <json>
"""

import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

import exact_tracks as X  # noqa: E402

LON = (10.0, 30.0)
LAT = (5.0, 15.0)
MONTHS = (7, 8, 9)
MAX_CASES = 3


def _sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def eligible(rec):
    return (LON[0] <= rec["first"]["lon"] < LON[1] and LAT[0] <= rec["first"]["lat"] < LAT[1]
            and int(rec["first"]["time"][5:7]) in MONTHS and rec["outcome"] != "atlantic_side")


def select(entries):
    """The predeclared rule over one season's inventory entries."""
    with_cand = [e for e in entries if e["n_candidate_systems"] > 0]
    if not with_cand:
        return None
    return sorted(with_cand, key=lambda e: (-e["window_steps_within_500_km_summed"], e["first"]["time"], e["id"]))[0]


def run(years, campaign_evidence, qtrack_dir, regions_dir, measurement_artifacts, out):
    import qtrack_pairing_pilot as M
    import season_metrics as S
    import coast_crossing_measurement as C
    import continuity_case_reading as R
    seasons, inputs = {}, {}
    for y in years:
        art_path = os.path.join(measurement_artifacts, R.ARTIFACT.format(year=y))
        art = json.load(open(art_path))
        ours_path = os.path.join(campaign_evidence, f"era5_{y}", "tracker_port.mat")
        if _sha256(ours_path) != art["inputs"]["this_record"]["sha256"]:
            raise SystemExit(f"REFUSED: the {y} tracks file is not the one the season artifact names")
        q_path = M.strict_path(qtrack_dir, M.QTRACK_FILE, y)
        if _sha256(q_path) != art["inputs"]["qtrack"]["sha256"]:
            raise SystemExit(f"REFUSED: QTrack's {y} file is not the one the season artifact names")
        recorded = art["inputs"]["region_polygons"]["sha256"]
        if {name: _sha256(os.path.join(regions_dir, f"{name}.mat")) for name in recorded} != recorded:
            raise SystemExit("REFUSED: the region polygons are not the ones the season artifact names")
        ours, _ = M.load_ours(ours_path)
        q_tracks, _ = M.load_qtrack(q_path)
        regions, _ = S.load_regions(regions_dir)
        endpoint = C.endpoint_day(y)
        side_b = {r["id"]: r for r in art["sides"]["qtrack"]["tracks"]}
        tails, _ = M.shared_tail_members_qtrack(q_path)
        by_id = {t["id"]: t for t in ours}
        entries = []
        for rec in art["sides"]["this_record"]["tracks"]:
            if not eligible(rec):
                continue
            c = R.candidates(by_id[rec["id"]], q_tracks, side_b, set(tails), regions, endpoint)
            entries.append({"id": rec["id"], "first": rec["first"], "band": rec["band"], "outcome": rec["outcome"], "follow_up_incomplete": rec["follow_up_incomplete"],
                            "n": rec["n_observations"], "group": rec.get("group"),
                            "window_observations": c["window_observations"], "n_candidate_systems": c["n_candidate_systems"],
                            "window_steps_within_500_km_summed": sum(s["window_steps_within_500_km"] for s in c["candidate_systems"]),
                            "max_candidates_at_a_step": max((s["n_candidates"] for s in c["per_step"]), default=0),
                            "per_step": [{"elapsed_hours": s["elapsed_hours"], "n_candidates": s["n_candidates"], "nearest_system": s["nearest_system"], "nearest_km": s["nearest_km"]}
                                         for s in c["per_step"]],
                            "candidates": [{"system": s["system"], "window_steps_within_500_km": s["window_steps_within_500_km"], "min_km_in_window": s["min_km_in_window"],
                                            "first_elapsed_hour_within_500_km": s["first_elapsed_hour_within_500_km"], "outcome": s["outcome"],
                                            "read_from_positions_outside_cohort": s["read_from_positions_outside_cohort"], "shared_tail_member": s["shared_tail_member"]}
                                           for s in c["candidate_systems"]],
                            "nearest_km_by_step": [s["nearest_km"] for s in c["per_step"]]})
        entries.sort(key=lambda e: (e["first"]["time"], e["id"]))
        chosen = select(entries)
        seasons[str(y)] = {"inventory": entries, "n_inventoried": len(entries), "n_with_a_candidate": sum(1 for e in entries if e["n_candidate_systems"] > 0),
                           "selected": {"id": chosen["id"], "window_steps_within_500_km_summed": chosen["window_steps_within_500_km_summed"], "candidates": [c["system"] for c in chosen["candidates"]]} if chosen else None}
        inputs[str(y)] = {"season_artifact": {"path": art_path, "sha256": _sha256(art_path)}, "this_record": {"path": ours_path, "sha256": _sha256(ours_path)},
                          "qtrack": {"path": q_path, "sha256": _sha256(q_path)}}
    selected = [(y, seasons[str(y)]["selected"]["id"]) for y in years if seasons[str(y)]["selected"]][:MAX_CASES]
    payload = {"generated_by": "scripts/continuity_inventory.py", "script_sha256": X.digest(__file__),
               "reader_script_sha256": X.digest(os.path.join(HERE, "continuity_case_reading.py")),
               "what_this_is": "the conditional second pass's inventory and selection; candidates are not identities; no case is judged here",
               "rules": {"stratum_lon": list(LON), "stratum_lat": list(LAT), "months": list(MONTHS), "outcome": "not Atlantic-side",
                         "window_hours_inclusive": R.WINDOW_HOURS, "radius_km": R.RADIUS_KM,
                         "selection": "per season, among tracks with at least one candidate, the largest window steps within 500 km summed over candidates, "
                                      "ties by earliest first observation then lowest track index; one per season, at most three"},
               "years": list(years), "inputs": inputs, "seasons": seasons, "selected_cases": [{"year": y, "id": i} for y, i in selected]}
    try:
        X.publish_json(out, payload, exclusive=True)
    except FileExistsError:
        raise SystemExit(f"REFUSED: {out} exists and artifacts are never overwritten")
    for y in years:
        s = seasons[str(y)]
        print(f"{y}: {s['n_inventoried']} inventoried, {s['n_with_a_candidate']} with a candidate, selected {s['selected']}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--years", nargs="+", type=int, required=True)
    ap.add_argument("--campaign-evidence", required=True)
    ap.add_argument("--qtrack-dir", required=True)
    ap.add_argument("--regions-dir", required=True)
    ap.add_argument("--measurement-artifacts", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return run(a.years, a.campaign_evidence, a.qtrack_dir, a.regions_dir, a.measurement_artifacts, a.out)


if __name__ == "__main__":
    sys.exit(main())
