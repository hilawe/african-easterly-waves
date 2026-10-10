#!/usr/bin/env python3
"""The eastern pilot's margin experiment at one highlighted step: three preparations of
the same step, the arrays verified, and the two merge passes run with the candidate list
held fixed within each contrast, to separate the boundary's two candidate effects that the
crossed-input experiment left bundled, the smoother's edge columns and the grid's extent.

THE THREE PREPARATIONS, each a bundle in the sense of `pilot_crossed_input.py` (its coarse
and fine grids, its masked coarse curvature, its masked fine curvature, its thresholds):

- P40, the current 40 E preparation: the control case's fields smoothed and masked on the
  control grids, whose last columns (39 E coarse, 40 E fine) are the smoother's edge.
- PMARGIN, wider-grid smoothing followed by cropping: the treatment case's fields smoothed
  and masked on the treatment grids, then every prepared array cropped to the control
  grids' exact columns, so the columns are the control's and the values in them are the
  interior-smoothed values. Its trough axes are contoured on the cropped grid, as a margin
  preparation in production would contour them.
- P60, the current wider preparation: the treatment case's fields on the treatment grids.

P40 and PMARGIN share their grids and differ only where the edge smoothing differs. PMARGIN
and P60 share their values in every common column and differ only in extent. The contrast
P40 against P60 is the retained crossed-input result and is read from its record, not
recomputed. THRESHOLDS, ASSOCIATION AND PRUNING ARE NOT TOUCHED: P40 and PMARGIN use the
control run's thresholds, P60 the treatment run's, whose two-ulp difference produces
identical masks at every ladder level (the retained mask check), and nothing is replayed.

THE ARRAYS ARE VERIFIED before any merge: the raw case fields of the two runs in their
common columns, at the step and over every step of the season, and the prepared arrays of
the three bundles field by field, with the columns that differ named.

THE CONTRASTS hold the candidate list fixed. The edge-ring contrast runs P40 and PMARGIN
each with the control's native list and each with the margin's own list. The extent
contrast runs PMARGIN and P60 each with the margin's list and each with the treatment's
native list, the treatment's candidates east of 40 E handled on the cropped grid in the
two declared variants (as given, seeded at the nearest above-threshold cell with the
displacement recorded, and withheld). The native endpoints, P40 with the control's list
and P60 with the treatment's, must reproduce the candidates the replays logged.

THE EXPECTED DISTINGUISHING OUTCOMES are declared in `EXPECTATIONS` below and in the
experiment's document before the run, and the record carries them beside the observed
outcome. A null or mixed result is a result.

    python3 scripts/pilot_margin_experiment.py --control-run <B dir> --control-case <B case> --treatment-run <C dir> \\
        --treatment-case <C case> --year 1990 --date 1990-07-01T12 --families <json> --family B_control_lost \\
        --replay-control <B replay> --replay-treatment <C replay> --crossed-input <r4 record> --out <fresh json>
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import exact_tracks as X  # noqa: E402
import pilot_association_replay as R  # noqa: E402
import pilot_crossed_input as CI  # noqa: E402
import pilot_replay_reading as RR  # noqa: E402
import pilot_stage_diagnostic as SD  # noqa: E402
import pilot_track_crosswalk as P  # noqa: E402

MATCH_DEG = CI.MATCH_DEG
RAW_FIELDS = (("u_c", "coarse"), ("v_c", "coarse"), ("currv_anom_c", "coarse"), ("advcurrv_anom_c", "coarse"), ("u", "fine"), ("v", "fine"), ("currv_anom", "fine"))
PREPARED = (("coarse_curvature", "prepared", "coarse"), ("coarse_advection", "advection", "coarse"), ("coarse_zonal_wind_smoothed", "wind", "coarse"),
            ("fine_curvature", "prepared_fine", "fine"), ("fine_zonal_wind_smoothed", "fine_u", "fine"), ("fine_meridional_wind_smoothed", "fine_v", "fine"))

# Declared before the run, from the coarse regions the crossed-input records carry. The
# verdict rule below is mechanical and the record carries both this and what was observed.
EXPECTATIONS = {
    "arrays": "the raw case fields of the two runs are equal in every common column at every step; PMARGIN equals P60 in every common column of "
              "every prepared field; PMARGIN differs from P40 only in the last coarse column (39 E) and the last fine column (40 E), and in no other column",
    "1990-07-01T12": {"expected": "smoothing", "reason": "the crossed-input coarse regions grow from one seed cell to 9 cells on P40 and 7 on P60, the two extra cells in the "
                                                          "control's edge column; if the edge smoothing carries the difference, PMARGIN holds the treatment-only candidate near "
                                                          "5.5 N 39.0 E and not the control-only one near 10.5 N 36.0 E, with either fixed list"},
    "2006-08-12T12": {"expected": "extent", "reason": "the crossed-input base-level component ends at 39 E on P40 (196 cells) and at 59 E on P60 (235 cells); if the extent "
                                                       "carries the difference, PMARGIN, which has the control's extent and the treatment's values, holds the control-only "
                                                       "candidate near 15.5 N 32.5 E and not the treatment-only one near 13.0 N 32.5 E, with either fixed list"},
    "verdict_rule": "for each fixed list, PMARGIN's finals are read for the control-only and the treatment-only candidates of the retained crossed-input record, exactly and "
                    "within match_deg; 'smoothing' when PMARGIN holds the treatment-only candidate and not the control-only one under every list and both tolerances, "
                    "'extent' when the reverse holds, and 'mixed_or_null' otherwise (one tolerance or one list disagreeing, both held, or neither held)",
}


def common_columns(case_b, case_c):
    """The treatment grids' columns that are the control grids' columns, which must be the
    control's columns exactly and in order."""
    cols_c = np.isin(case_c["lon_c"], case_b["lon_c"])
    fcols_c = np.isin(case_c["longrid"][0, :], case_b["longrid"][0, :])
    if not np.array_equal(case_c["lon_c"][cols_c], case_b["lon_c"]) or not np.array_equal(case_c["longrid"][0, fcols_c], case_b["longrid"][0, :]):
        raise SystemExit("REFUSED: the control grids' columns are not a subset of the treatment grids' columns in order")
    if not np.array_equal(case_b["lat_c"], case_c["lat_c"]) or not np.array_equal(case_b["latgrid"][:, 0], case_c["latgrid"][:, 0]):
        raise SystemExit("REFUSED: the two cases do not share their latitude rows")
    return cols_c, fcols_c


def raw_equality(case_b, case_c, k, cols_c, fcols_c):
    """The raw case fields in the common columns, at step k and over every step."""
    out = {}
    for name, grid in RAW_FIELDS:
        sel = cols_c if grid == "coarse" else fcols_c
        a, b = case_b[name], case_c[name][:, :, sel]
        if a.shape != b.shape:
            raise SystemExit(f"REFUSED: {name} has shape {a.shape} in the control and {b.shape} in the treatment's common columns")
        diff_all = (a != b) & ~(np.isnan(a) & np.isnan(b))
        diff_k = diff_all[k]
        out[name] = {"cells": int(a[k].size), "differing_at_step": int(diff_k.sum()), "differing_all_steps": int(diff_all.sum()),
                     "steps": int(a.shape[0]), "max_abs_difference_all_steps": float(np.nanmax(np.abs(a - b))) if diff_all.any() else 0.0}
    return out


def prepared_bundle(case, k, ct, ft, stages=None):
    """A preparation bundle on a case's own grids, with every prepared field kept for the
    array verification."""
    s = stages if stages is not None else SD.stages(case, k, ct, ft)
    prep, cands, sizes = CI.bundles(case, k, ct, ft)
    prep["fields"] = {name: s[key] for name, key, _g in PREPARED}
    return prep, cands, sizes


def margin_bundle(case_b, case_c, k, ct, ft, cols_c, fcols_c):
    """Wider-grid smoothing and masking, then cropping to the control grids' exact columns,
    with the trough axes contoured on the cropped grid."""
    from aew.v1port.detection import trough_axes
    s = SD.stages(case_c, k, ct, ft)
    lat_c, lon_c = case_b["lat_c"], case_b["lon_c"]
    latgrid_c, longrid_c = np.meshgrid(lat_c, lon_c, indexing="ij")
    fields = {}
    for name, key, grid in PREPARED:
        fields[name] = s[key][:, cols_c] if grid == "coarse" else s[key][:, fcols_c]
    advection = s["advection"][:, cols_c]
    axes = trough_axes(latgrid_c, longrid_c, advection)
    t = float(case_b["time"][k])
    cands = [{"time": t, "lat_mean": float(np.mean(la)), "lon_mean": float(np.mean(lo))} for la, lo in axes]
    prep = {"latgrid_c": latgrid_c, "longrid_c": longrid_c, "curvature": fields["coarse_curvature"], "latgrid_f": case_b["latgrid"], "longrid_f": case_b["longrid"],
            "curvature_f": fields["fine_curvature"], "ct": ct, "ft": ft,
            "extent": {"lat": [float(lat_c.min()), float(lat_c.max())], "lon": [float(lon_c.min()), float(lon_c.max())]},
            "digests": {"coarse_curvature": CI.array_digest(fields["coarse_curvature"]), "fine_curvature": CI.array_digest(fields["fine_curvature"]),
                        "lon_c": CI.array_digest(lon_c), "lat_c": CI.array_digest(lat_c), "longrid_f": CI.array_digest(case_b["longrid"]), "latgrid_f": CI.array_digest(case_b["latgrid"])},
            "fields": fields}
    return prep, cands, [int(la.size) for la, lo in axes]


def field_columns_differing(fa, fb, lons):
    """The columns in which two prepared fields on one grid differ (masked-ness or value)."""
    diff = (np.isnan(fa) != np.isnan(fb)) | (~np.isnan(fa) & ~np.isnan(fb) & (fa != fb))
    cols = np.where(diff.any(axis=0))[0]
    return {"cells_differing": int(diff.sum()), "columns_differing": [float(lons[c]) for c in cols]}


def prepared_comparison(p40, pm, p60, case_b, cols_c, fcols_c):
    """P40 against PMARGIN on the control grids, and PMARGIN against P60 in the common columns."""
    out = {"P40_vs_PMARGIN": {}, "PMARGIN_vs_P60_common_columns": {}}
    for name, _key, grid in PREPARED:
        lons = case_b["lon_c"] if grid == "coarse" else case_b["longrid"][0, :]
        sel = cols_c if grid == "coarse" else fcols_c
        out["P40_vs_PMARGIN"][name] = field_columns_differing(p40["fields"][name], pm["fields"][name], lons)
        out["PMARGIN_vs_P60_common_columns"][name] = field_columns_differing(pm["fields"][name], p60["fields"][name][:, sel], lons)
    return out


def holds(point, finals, tol):
    return CI.present(point, finals, tol)


def content_and_order(res_a, res_b):
    """Two merge results compared as a multiset of final candidates (center and region
    digest) and as an ordered list. The association reads candidates in order, when it
    breaks a distance tie and when it seeds new tracks, so equal content in another order
    is not the same detector output."""
    ka = [(f["lat_mean"], f["lon_mean"], f["region_sha256"]) for f in res_a["final"]]
    kb = [(f["lat_mean"], f["lon_mean"], f["region_sha256"]) for f in res_b["final"]]
    return {"same_content": sorted(ka) == sorted(kb), "same_order": ka == kb, "count_a": len(ka), "count_b": len(kb)}


def build_bundles(cases, thr, k, cols_c, fcols_c):
    """The three preparation bundles and their candidate lists, with the thresholds each
    one was built with recorded beside it: P40 and PMARGIN with the control run's, P60 with
    the treatment run's."""
    p40, l40, sizes40 = prepared_bundle(cases["B"], k, *thr["B"])
    p60, l60, sizes60 = prepared_bundle(cases["C"], k, *thr["C"])
    pm, lm, sizesm = margin_bundle(cases["B"], cases["C"], k, *thr["B"], cols_c, fcols_c)
    preps = {"P40": p40, "PMARGIN": pm, "P60": p60}
    lists = {"L40": l40, "LMARGIN": lm, "L60": l60}
    sizes = {"L40": sizes40, "LMARGIN": sizesm, "L60": sizes60}
    thresholds = {name: (preps[name]["ct"], preps[name]["ft"]) for name in preps}
    return preps, lists, sizes, thresholds


def verdict(step_results, reference):
    """The mechanical reading of PMARGIN's finals against the retained native-only
    candidates, for every fixed list, both variants and both tolerances."""
    b_only, c_only = reference["B"], reference["C"]
    if len(b_only) != 1 or len(c_only) != 1:
        return {"verdict": "mixed_or_null", "note": f"the retained record names {len(b_only)} control-only and {len(c_only)} treatment-only candidates, not one each"}
    b, c = tuple(b_only[0]), tuple(c_only[0])
    reads = {}
    for (contrast, prep_name, list_name, variant), res in step_results.items():
        if prep_name != "PMARGIN":
            continue
        for tol_name, tol in (("exact", 1e-9), ("within_match_deg", MATCH_DEG)):
            reads[f"{contrast}/{list_name}/{variant}/{tol_name}"] = {"holds_control_only": holds(b, res["final"], tol), "holds_treatment_only": holds(c, res["final"], tol)}
    if all(r["holds_treatment_only"] and not r["holds_control_only"] for r in reads.values()):
        v = "smoothing"
    elif all(r["holds_control_only"] and not r["holds_treatment_only"] for r in reads.values()):
        v = "extent"
    else:
        v = "mixed_or_null"
    return {"verdict": v, "control_only": list(b), "treatment_only": list(c), "reads": reads}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("--control-run", "--control-case", "--treatment-run", "--treatment-case", "--date", "--families", "--family", "--replay-control", "--replay-treatment", "--crossed-input", "--out"):
        ap.add_argument(name, required=True)
    ap.add_argument("--year", type=int, required=True)
    args = ap.parse_args(argv)
    if os.path.exists(args.out):
        raise SystemExit(f"REFUSED: {args.out} exists and a record is never overwritten")
    if args.date not in EXPECTATIONS:
        raise SystemExit(f"REFUSED: no expectation was declared for {args.date}; this experiment runs only at the two highlighted steps")
    t_start = time.perf_counter()
    runs = {"B": P.load_run(args.control_run, year=args.year), "C": P.load_run(args.treatment_run, year=args.year)}
    cases = {"B": SD.load_case(args.control_case, runs["B"]["record"]), "C": SD.load_case(args.treatment_case, runs["C"]["record"])}
    replays, replay_sha = {}, {}
    for r, path in (("B", args.replay_control), ("C", args.replay_treatment)):
        replays[r], replay_sha[r] = RR.load_replay_bytes(path)
        if replays[r]["case_sha256"] != cases[r]["sha256"] or replays[r]["tracks_sha256"] != runs[r]["tracks_sha256"] or not replays[r]["gate"]["passed"]:
            raise SystemExit(f"REFUSED: the {r} replay is not a passed replay of this case and these tracks")
    with open(args.crossed_input, "rb") as fh:
        crossed_blob = fh.read()
    crossed = json.loads(crossed_blob)
    crossed_sha = R.hashlib.sha256(crossed_blob).hexdigest()
    if crossed["date"] != args.date or crossed["family"] != args.family or any(crossed["replays"][r]["replay_sha256"] != replay_sha[r] for r in "BC"):
        raise SystemExit("REFUSED: the crossed-input record is not for this date, family and these replays")
    fam = next((f for f in json.load(open(args.families))["families"] if f["name"] == args.family), None)
    if fam is None:
        raise SystemExit(f"REFUSED: no family {args.family} in {args.families}")
    k = {r: CI.step_of(cases[r], args.date) for r in "BC"}
    if k["B"] != k["C"]:
        raise SystemExit("REFUSED: the two cases index the date at different steps")
    k = k["B"]
    thr = {r: (float(runs[r]["record"]["dataset_specific"]["coarse_threshold"]), float(runs[r]["record"]["dataset_specific"]["fine_threshold"])) for r in "BC"}
    cols_c, fcols_c = common_columns(cases["B"], cases["C"])
    raw = raw_equality(cases["B"], cases["C"], k, cols_c, fcols_c)
    preps, lists, sizes, thresholds = build_bundles(cases, thr, k, cols_c, fcols_c)
    p40, pm, p60 = preps["P40"], preps["PMARGIN"], preps["P60"]
    arrays = {"raw_case_fields_common_columns": raw, "prepared": prepared_comparison(p40, pm, p60, cases["B"], cols_c, fcols_c),
              "grid_extents": {"P40": p40["extent"], "PMARGIN": pm["extent"], "P60": p60["extent"]}, "thresholds": thresholds}
    contrasts = {"edge_ring": {"preparations": ["P40", "PMARGIN"], "lists": ["L40", "LMARGIN"]},
                 "extent": {"preparations": ["PMARGIN", "P60"], "lists": ["LMARGIN", "L60"]}}
    results, records, variants = {}, {}, {}
    for cname, c in contrasts.items():
        for prep_name in c["preparations"]:
            for list_name in c["lists"]:
                for variant in ("as_given", "inside_receiving_grid"):
                    res = CI.combination(preps[prep_name], lists[list_name], variant)
                    results[(cname, prep_name, list_name, variant)] = res
                    records[f"{cname}/{prep_name}/{list_name}/{variant}"] = res
                # giving or withholding the out-of-grid candidates: the same final content, and the same order?
                variants[f"{cname}/{prep_name}/{list_name}"] = content_and_order(results[(cname, prep_name, list_name, "as_given")],
                                                                                 results[(cname, prep_name, list_name, "inside_receiving_grid")])
    native = {}
    for r, prep_name, list_name in (("B", "P40", "L40"), ("C", "P60", "L60")):
        logged, _entry = CI.replay_candidates(replays[r], k)
        res = CI.combination(preps[prep_name], lists[list_name], "as_given")
        native[r] = CI.native_check(res, logged)
        if not native[r]["reproduces_replay_candidates"]:
            raise SystemExit(f"REFUSED: {prep_name} with {list_name} does not reproduce the candidates the {r} replay logged at step {k}")
    reference = crossed["native_only_in_box_and_common_domain"]
    read = verdict(results, reference)
    expected = EXPECTATIONS[args.date]
    out = {"generated_by": "scripts/pilot_margin_experiment.py", "script_sha256": X.digest(os.path.abspath(__file__)), "year": args.year, "date": args.date, "step": k,
           "family": args.family, "box": fam["box"], "match_deg": MATCH_DEG,
           "runs": {r: {"dir": d, "record_sha256": runs[r]["record_sha256"], "tracks_sha256": runs[r]["tracks_sha256"], "case_sha256": cases[r]["sha256"]}
                    for r, d in (("B", args.control_run), ("C", args.treatment_run))},
           "replays": {r: {"path": p, "replay_sha256": replay_sha[r]} for r, p in (("B", args.replay_control), ("C", args.replay_treatment))},
           "crossed_input_record": {"path": args.crossed_input, "sha256": crossed_sha, "native_only": reference},
           "expectations": {"arrays": EXPECTATIONS["arrays"], "this_step": expected, "verdict_rule": EXPECTATIONS["verdict_rule"]},
           "arrays": arrays,
           "candidate_lists": {name: {"count": len(lst), "candidates": [[c["lat_mean"], c["lon_mean"]] for c in lst],
                                      "east_of_common_edge": sum(1 for c in lst if c["lon_mean"] > CI.COMMON_EAST_DEG)} for name, lst in lists.items()},
           "axis_vertex_counts": sizes,
           "contrasts": contrasts, "native_endpoints": native, "results": records,
           "out_of_grid_variants_compared": variants, "reading": read,
           "observed_matches_expected": read["verdict"] == expected["expected"], "elapsed_seconds": round(time.perf_counter() - t_start, 1)}
    X.publish_json(args.out, out, exclusive=True)
    print(f"{args.date}: arrays raw differing (all steps) {sum(v['differing_all_steps'] for v in raw.values())}, "
          f"PMARGIN vs P60 common differing {sum(v['cells_differing'] for v in arrays['prepared']['PMARGIN_vs_P60_common_columns'].values())}, "
          f"P40 vs PMARGIN columns {sorted({c for v in arrays['prepared']['P40_vs_PMARGIN'].values() for c in v['columns_differing']})}; "
          f"verdict {read['verdict']} (expected {expected['expected']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
