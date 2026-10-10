#!/usr/bin/env python3
"""The eastern pilot's crossed-input merge experiment at one step: the four combinations
of a run's PREPARATION BUNDLE with a run's ORDERED PRE-MERGE CANDIDATE BUNDLE, through the
detector's two merge passes exactly as `detect_troughs` runs them.

THE PREPARATION BUNDLE of a run at a step is its coarse grid and its fine grid, its masked
coarse curvature (smoothed, cyclonic-positive, the westerly and sub-threshold cells
masked), its masked fine curvature (smoothed, cyclonic-positive, sub-threshold cells
masked) and its two thresholds: everything the two merge passes read apart from the
candidates. THE CANDIDATE BUNDLE is the ordered list of trough-axis means the run's own
masked advection contours to on its own grid, in the detector's order. The four
combinations are the two native ones (B preparation with B candidates, C with C) and the
two crossed ones. THE NATIVE COMBINATIONS MUST REPRODUCE, center for center and region
digest for region digest, the final candidates the instrumented replay logged for that run
at that step, or the experiment refuses.

CANDIDATES OUTSIDE THE RECEIVING GRID are neither dropped, clamped nor given an extended
grid silently. The merge seeds each candidate at the nearest above-threshold cell of the
receiving grid in squared degrees (column-first ties, as `merge_contours` enumerates), so a
candidate east of the control's last column seeds a cell inside it, displaced west. Every
candidate's seed cell and displacement are recorded in every combination, and each crossed
combination is run in two declared variants: AS GIVEN (every candidate of the donor run,
in its order) and INSIDE THE RECEIVING GRID ONLY (the candidates whose center lies within
the receiving coarse grid's extent, in the same order, the others listed as withheld). A
native combination's two variants are one and the same, which is asserted. The seed cell
is the one replicated computation here, and it is bound to the production merge: the
region grown from it must equal, cell for cell, the region `merge_contours` returns for
that candidate alone.

WHAT A RESULT CAN MEAN. A native-only final candidate (in one native set and more than the
match distance from every candidate of the other, inside the family's box and the common
domain) is read against the two crossed combinations. Present with its native preparation
whatever the candidates and absent with the other preparation, it follows the preparation.
Present with its native candidates whatever the preparation and absent with the other
candidates, it follows the candidates. Anything else is an interaction or unresolved. The
reading is mechanical and algorithmic. It says which input of this implementation the
candidate followed at this step, and nothing about physical waves. The association's use
of these candidates is the replay's record, not this one's.

    python3 scripts/pilot_crossed_input.py --control-run <B dir> --control-case <B case> --treatment-run <C dir> \\
        --treatment-case <C case> --year 1990 --date 1990-07-01T12 --families <json> --family B_control_lost \\
        --replay-control <B replay json> --replay-treatment <C replay json> --out <fresh json>
"""
import argparse
import datetime as dt
import itertools
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import exact_tracks as X  # noqa: E402
import pilot_association_replay as R  # noqa: E402
import pilot_replay_reading as RR  # noqa: E402
import pilot_stage_diagnostic as SD  # noqa: E402
import pilot_track_crosswalk as P  # noqa: E402

EPOCH = dt.datetime(1900, 1, 1)
MATCH_DEG = SD.MATCH_DEG
COMMON_EAST_DEG = SD.COMMON_EAST_DEG


def array_digest(a):
    a = np.ascontiguousarray(np.asarray(a, dtype=float))
    return R.hashlib.sha256(repr(a.shape).encode() + a.tobytes()).hexdigest()


def step_of(case, date):
    t = (dt.datetime.strptime(date, "%Y-%m-%dT%H") - EPOCH).total_seconds() / 86400.0
    k = np.flatnonzero(np.abs(case["time"] - t) < 1e-9)
    if k.size != 1:
        raise SystemExit(f"REFUSED: the case holds {k.size} steps at {date}")
    return int(k[0])


def bundles(case, k, ct, ft):
    """The preparation bundle and the ordered candidate bundle of one run at step k."""
    s = SD.stages(case, k, ct, ft)
    lat_c, lon_c = case["lat_c"], case["lon_c"]
    latgrid_c, longrid_c = np.meshgrid(lat_c, lon_c, indexing="ij")
    prep = {"latgrid_c": latgrid_c, "longrid_c": longrid_c, "curvature": s["prepared"], "latgrid_f": case["latgrid"], "longrid_f": case["longrid"],
            "curvature_f": s["prepared_fine"], "ct": ct, "ft": ft,
            "extent": {"lat": [float(lat_c.min()), float(lat_c.max())], "lon": [float(lon_c.min()), float(lon_c.max())]},
            "digests": {"coarse_curvature": array_digest(s["prepared"]), "fine_curvature": array_digest(s["prepared_fine"]),
                        "lon_c": array_digest(lon_c), "lat_c": array_digest(lat_c),
                        "longrid_f": array_digest(case["longrid"]), "latgrid_f": array_digest(case["latgrid"])}}
    t = float(case["time"][k])
    cands = [{"time": t, "lat_mean": float(a), "lon_mean": float(o)} for a, o in s["axis_candidates"]]
    return prep, cands, [int(la.size) for la, lo in s["axes"]]


def seed_cell(prep, cand):
    """The cell `merge_contours` seeds this candidate at: the nearest above-threshold cell
    in squared degrees, column-first ties. Replicated from its three lines and bound below."""
    from aew.v1port.contours import _binary_masks
    masks = _binary_masks(prep["curvature"], prep["ct"])
    if not masks[0].any():
        return masks, None, None
    base_cols, base_rows = np.nonzero(masks[0].T)
    d2 = (cand["lat_mean"] - prep["latgrid_c"][base_rows, base_cols]) ** 2 + (cand["lon_mean"] - prep["longrid_c"][base_rows, base_cols]) ** 2
    n = int(np.argmin(d2))
    return masks, (int(base_rows[n]), int(base_cols[n])), float(np.sqrt(d2[n]))


def candidate_rows(prep, cands):
    """Every candidate against the receiving grid: inside its extent or not, its seed cell
    and displacement, and the region the merge grows from it, bound to the production merge."""
    from aew.v1port.contours import _select_region, connected_region, merge_contours
    rows = []
    for j, c in enumerate(cands):
        inside = bool(prep["extent"]["lat"][0] <= c["lat_mean"] <= prep["extent"]["lat"][1] and prep["extent"]["lon"][0] <= c["lon_mean"] <= prep["extent"]["lon"][1])
        masks, seed, disp = seed_cell(prep, c)
        row = {"index": j, "lat_mean": c["lat_mean"], "lon_mean": c["lon_mean"], "inside_receiving_grid": inside, "seed_cell": seed,
               "seed_lat": None, "seed_lon": None, "seed_displacement_deg": disp, "region_sha256": None, "region_cells": 0}
        if seed is not None:
            region = _select_region(masks, seed, prep["latgrid_c"], prep["longrid_c"])
            alone = merge_contours([c], prep["latgrid_c"], prep["longrid_c"], prep["curvature"], prep["ct"])
            if len(alone) != 1 or not np.array_equal(alone[0]["region"], region):
                raise SystemExit(f"REFUSED: the replicated seed for candidate {j} grows a region the production merge does not return")
            # the region at the ladder's base level too, so a reader can see whether the
            # ladder stepped up (the chosen region is smaller than the base one) and how
            # far east the base component reaches on this grid
            base = connected_region(masks[0], seed)
            # `region_smaller_than_base_level` TRUE proves the ladder stepped up, since the
            # selector returns the base-level region unless an extent check advanced it.
            # FALSE does not prove it did not: a component above every level of the ladder
            # stays the same component at every level, so the selector can advance without
            # the region shrinking. The selected level itself is not exposed by the
            # production selector and is not replicated here.
            row.update({"seed_lat": float(prep["latgrid_c"][seed]), "seed_lon": float(prep["longrid_c"][seed]),
                        "region_sha256": R.region_digest(region), "region_cells": int(region.sum()),
                        "region_lon_extent": [float(prep["longrid_c"][region].min()), float(prep["longrid_c"][region].max())],
                        "region_lat_extent": [float(prep["latgrid_c"][region].min()), float(prep["latgrid_c"][region].max())],
                        "base_level_region_cells": int(base.sum()),
                        "base_level_lon_extent": [float(prep["longrid_c"][base].min()), float(prep["longrid_c"][base].max())],
                        "region_smaller_than_base_level": bool(region.sum() < base.sum())})
        rows.append(row)
    return rows


def merge_both_passes(prep, cands):
    """The two merge passes as `detect_troughs` runs them on this preparation, with each
    merged wave traced to the candidates whose lone region equals its region."""
    from aew.v1port.contours import merge_contours
    rows = candidate_rows(prep, cands)
    coarse = merge_contours(cands, prep["latgrid_c"], prep["longrid_c"], prep["curvature"], prep["ct"]) if cands else []
    coarse_rows = []
    for w in coarse:
        d = R.region_digest(w["region"])
        coarse_rows.append({"lat_mean": float(w["lat_mean"]), "lon_mean": float(w["lon_mean"]), "n_points": int(np.asarray(w["lat_wave"]).size),
                            "region_sha256": d, "from_candidates": [r["index"] for r in rows if r["region_sha256"] == d]})
    final = merge_contours(coarse, prep["latgrid_f"], prep["longrid_f"], prep["curvature_f"], prep["ft"]) if coarse else []
    lone_fine = [R.region_digest(merge_contours([w], prep["latgrid_f"], prep["longrid_f"], prep["curvature_f"], prep["ft"])[0]["region"]) for w in coarse]
    final_rows = []
    for w in final:
        d = R.region_digest(w["region"])
        final_rows.append({"lat_mean": float(w["lat_mean"]), "lon_mean": float(w["lon_mean"]), "n_points": int(np.asarray(w["lat_wave"]).size),
                           "region_sha256": d, "from_coarse_merged": [i for i, ld in enumerate(lone_fine) if ld == d]})
    return {"candidates": rows, "coarse_merged": coarse_rows, "final": final_rows}


def combination(prep, cands, variant):
    if variant == "as_given":
        used, withheld = list(cands), []
    else:
        keep = [bool(prep["extent"]["lat"][0] <= c["lat_mean"] <= prep["extent"]["lat"][1] and prep["extent"]["lon"][0] <= c["lon_mean"] <= prep["extent"]["lon"][1]) for c in cands]
        used = [c for c, k in zip(cands, keep) if k]
        withheld = [{"index": j, "lat_mean": c["lat_mean"], "lon_mean": c["lon_mean"]} for j, (c, k) in enumerate(zip(cands, keep)) if not k]
    out = merge_both_passes(prep, used)
    out["variant"] = variant
    out["candidates_used"] = len(used)
    out["candidates_withheld"] = withheld
    return out


def replay_candidates(replay, k):
    entry = next((e for e in replay["steps"] if e["step"] == k), None)
    if entry is None:
        raise SystemExit(f"REFUSED: the replay {replay['label']} {replay['year']} did not log step {k}")
    return [(c["lat_mean"], c["lon_mean"], c["region_sha256"]) for c in entry["candidates"]], entry


def native_check(result, replay_entry_candidates):
    mine = [(r["lat_mean"], r["lon_mean"], r["region_sha256"]) for r in result["final"]]
    return {"reproduces_replay_candidates": mine == replay_entry_candidates, "n_here": len(mine), "n_replay": len(replay_entry_candidates)}


def present(point, finals, tol):
    """A final candidate within `tol` degrees of the point, as a Euclidean distance in
    degrees, the same rule the stage diagnostic and the replay reading apply. A first
    version tested each coordinate separately, which accepted a diagonal offset of 1.13
    degrees under the 1-degree rule."""
    return any(float(np.hypot(f["lat_mean"] - point[0], f["lon_mean"] - point[1])) <= tol for f in finals)


def native_only_sets(results, box):
    """Each run's native-only final candidates: inside the family's box and west of the
    common edge, with no final candidate of the other native run within MATCH_DEG anywhere.
    The counterpart is searched among ALL of the other run's final candidates, so one just
    outside the box or just east of the common edge still counts (a first version searched
    the other run's in-box candidates only)."""
    def in_common_box(f):
        return SD.in_box([(f["lat_mean"], f["lon_mean"])], box) and f["lon_mean"] <= COMMON_EAST_DEG
    finals = {r: [f for f in results[(r, r)]["as_given"]["final"] if in_common_box(f)] for r in "BC"}
    out = {}
    for r, o in (("B", "C"), ("C", "B")):
        out[r] = [f for f in finals[r] if not present((f["lat_mean"], f["lon_mean"]), results[(o, o)]["as_given"]["final"], MATCH_DEG)]
    return finals, out


def reading(point, native_prep, native_cand, results, variant, tol):
    """Where a native-only candidate appears across the four combinations, and the
    mechanical reading of what it followed."""
    other_prep, other_cand = ("C" if native_prep == "B" else "B"), ("C" if native_cand == "B" else "B")
    where = {f"prep_{p}_cand_{c}": present(point, results[(p, c)][variant]["final"], tol) for p in "BC" for c in "BC"}
    with_native_prep = where[f"prep_{native_prep}_cand_{native_cand}"] and where[f"prep_{native_prep}_cand_{other_cand}"]
    without_native_prep = not where[f"prep_{other_prep}_cand_{native_cand}"] and not where[f"prep_{other_prep}_cand_{other_cand}"]
    with_native_cand = where[f"prep_{native_prep}_cand_{native_cand}"] and where[f"prep_{other_prep}_cand_{native_cand}"]
    without_native_cand = not where[f"prep_{native_prep}_cand_{other_cand}"] and not where[f"prep_{other_prep}_cand_{other_cand}"]
    if with_native_prep and without_native_prep and not (with_native_cand and without_native_cand):
        follows = "preparation"
    elif with_native_cand and without_native_cand and not (with_native_prep and without_native_prep):
        follows = "candidates"
    else:
        follows = "interaction_or_unresolved"
    return {"present_in": where, "follows": follows}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("--control-run", "--control-case", "--treatment-run", "--treatment-case", "--date", "--families", "--family", "--replay-control", "--replay-treatment", "--out"):
        ap.add_argument(name, required=True)
    ap.add_argument("--year", type=int, required=True)
    args = ap.parse_args(argv)
    if os.path.exists(args.out):
        raise SystemExit(f"REFUSED: {args.out} exists and a record is never overwritten")
    t_start = time.perf_counter()
    runs = {"B": P.load_run(args.control_run, year=args.year), "C": P.load_run(args.treatment_run, year=args.year)}
    cases = {"B": SD.load_case(args.control_case, runs["B"]["record"]), "C": SD.load_case(args.treatment_case, runs["C"]["record"])}
    replays, replay_sha = {}, {}
    for r, path in (("B", args.replay_control), ("C", args.replay_treatment)):
        replays[r], replay_sha[r] = RR.load_replay_bytes(path)
    for r in "BC":
        if replays[r]["case_sha256"] != cases[r]["sha256"] or replays[r]["tracks_sha256"] != runs[r]["tracks_sha256"] or not replays[r]["gate"]["passed"]:
            raise SystemExit(f"REFUSED: the {r} replay is not a passed replay of this case and these tracks")
    fam = next((f for f in json.load(open(args.families))["families"] if f["name"] == args.family), None)
    if fam is None:
        raise SystemExit(f"REFUSED: no family {args.family} in {args.families}")
    box = fam["box"]
    k = {r: step_of(cases[r], args.date) for r in "BC"}
    thr = {r: (float(runs[r]["record"]["dataset_specific"]["coarse_threshold"]), float(runs[r]["record"]["dataset_specific"]["fine_threshold"])) for r in "BC"}
    preps, cands, axis_sizes = {}, {}, {}
    for r in "BC":
        preps[r], cands[r], axis_sizes[r] = bundles(cases[r], k[r], *thr[r])
    results = {}
    for p, c in itertools.product("BC", "BC"):
        results[(p, c)] = {v: combination(preps[p], cands[c], v) for v in ("as_given", "inside_receiving_grid")}
        if p == c:
            a, b = results[(p, c)]["as_given"], results[(p, c)]["inside_receiving_grid"]
            if a["final"] != b["final"] or b["candidates_withheld"]:
                raise SystemExit(f"REFUSED: the native combination {p}/{c} differs between its two variants")
    native = {}
    for r in "BC":
        logged, entry = replay_candidates(replays[r], k[r])
        native[r] = native_check(results[(r, r)]["as_given"], logged)
        native[r]["replay_claims_at_step"] = entry.get("claims", [])
        native[r]["replay_seeds_at_step"] = entry.get("seeds", [])
        if not native[r]["reproduces_replay_candidates"]:
            raise SystemExit(f"REFUSED: the native combination {r}/{r} does not reproduce the candidates the {r} replay logged at step {k[r]}")

    finals, native_only = native_only_sets(results, box)
    readings = []
    for r in "BC":
        for f in native_only[r]:
            point = (f["lat_mean"], f["lon_mean"])
            readings.append({"native_run": r, "candidate": [point[0], point[1]], "region_sha256": f["region_sha256"],
                             "exact": {v: reading(point, r, r, results, v, 1e-9) for v in ("as_given", "inside_receiving_grid")},
                             "within_match_deg": {v: reading(point, r, r, results, v, MATCH_DEG) for v in ("as_given", "inside_receiving_grid")}})
    out = {"generated_by": "scripts/pilot_crossed_input.py", "script_sha256": X.digest(os.path.abspath(__file__)), "year": args.year, "date": args.date,
           "family": args.family, "box": box, "match_deg": MATCH_DEG, "common_east_deg": COMMON_EAST_DEG, "step": k,
           "runs": {r: {"dir": d, "record_sha256": runs[r]["record_sha256"], "tracks_sha256": runs[r]["tracks_sha256"], "case_sha256": cases[r]["sha256"],
                        "coarse_threshold": thr[r][0], "fine_threshold": thr[r][1]} for r, d in (("B", args.control_run), ("C", args.treatment_run))},
           "replays": {r: {"path": p, "replay_sha256": replay_sha[r], "script_sha256": replays[r]["script_sha256"], "gate": replays[r]["gate"]}
                       for r, p in (("B", args.replay_control), ("C", args.replay_treatment))},
           "elapsed_seconds": round(time.perf_counter() - t_start, 1),
           "preparation_bundles": {r: {"extent": preps[r]["extent"], "digests": preps[r]["digests"], "coarse_threshold": thr[r][0], "fine_threshold": thr[r][1]} for r in "BC"},
           "candidate_bundles": {r: {"count": len(cands[r]), "axis_vertex_counts": axis_sizes[r],
                                     "candidates": [[c["lat_mean"], c["lon_mean"]] for c in cands[r]],
                                     "east_of_common_edge": sum(1 for c in cands[r] if c["lon_mean"] > COMMON_EAST_DEG),
                                     "outside_other_grid": sum(1 for c in cands[r] if not (preps["C" if r == "B" else "B"]["extent"]["lon"][0] <= c["lon_mean"] <= preps["C" if r == "B" else "B"]["extent"]["lon"][1]))} for r in "BC"},
           "native_endpoints": native,
           "combinations": {f"prep_{p}_cand_{c}": results[(p, c)] for p, c in itertools.product("BC", "BC")},
           "native_only_in_box_and_common_domain": {r: [[f["lat_mean"], f["lon_mean"]] for f in native_only[r]] for r in "BC"},
           "readings": readings,
           "reading_rule": "a native-only final candidate (in the family box and west of the common edge, with no final candidate of the other native run within match_deg anywhere) follows the preparation when it is present in both combinations with its native preparation and absent in both with the other, follows the candidates when the same holds for the candidate bundle, and is an interaction or unresolved otherwise. Presence is a Euclidean distance in degrees between centers, 1e-9 for exact and match_deg for loose. Both variants of the out-of-grid treatment are read."}
    X.publish_json(args.out, out, exclusive=True)
    for rd in readings:
        print(f"{args.date} {rd['native_run']}-only {rd['candidate']}: exact {rd['exact']['as_given']['follows']} (as given), {rd['exact']['inside_receiving_grid']['follows']} (inside only). "
              f"Within {MATCH_DEG} deg {rd['within_match_deg']['as_given']['follows']} (as given), {rd['within_match_deg']['inside_receiving_grid']['follows']} (inside only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
