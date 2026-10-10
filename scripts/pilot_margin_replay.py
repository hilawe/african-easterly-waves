#!/usr/bin/env python3
"""The margin replay: the production loop of `export_protocol_case.track_case` run for one
season with its detection preparation taken either as it is (NATIVE) or from the wider
case, smoothed first and cropped to the control grids afterward (MARGIN), everything else
unchanged, through the original horizon and finalization, with every finished track's raw
history retained for the comparison in `pilot_margin_compare.py`.

WHAT THE MARGIN CHANGES, and nothing else. At every step the detector's preparation (the
nine-point smoother on the coarse zonal wind, the coarse curvature anomaly, its advection
and the fine curvature anomaly, with the sign flip south of the equator) runs on the
treatment case's wider arrays and the results are cropped to the control grids' exact
columns before the westerly and threshold masks, the contouring of the trough axes and the
two merge passes, which run on the control grids as in production. The smoothed fine winds
the association takes its region medians from are prepared the same way, smoothed on the
wider fine grid and cropped. The thresholds are the control run's. `associate_step`,
`prune_stale_tracks` and `finalize_tracks` are the production functions, the candidates
keep the order the detector produced, and the run goes from the case's first step with an
empty live state to its last step with finalization at the original horizon.

THE NATIVE MODE IS THE BIND. With the control case and no crop, the same code must give, at
every step, exactly the candidates `detect_troughs` gives (centers and region digests, in
order), and its finished tracks must equal the archived tracks as an ordered multiset, or
the run refuses. A margin run has no oracle, since it is the experiment, and it records
its own detection as produced.

WHAT IS RETAINED, for every finished track of either mode: the smoothed output arrays as
the archive holds them, the raw history underneath (the steps, times, positions and
region digests of the candidates the track claimed, unsmoothed), a birth label private to
the run ("step:candidate index", never compared across runs), and the fate counts of every
track born.

    python3 scripts/pilot_margin_replay.py --preparation margin --control-run <B dir> --control-case <B case> \\
        --treatment-case <C case> --treatment-run <C dir> --year 1990 --out <fresh json.gz>
"""
import argparse
import gzip
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import exact_tracks as X  # noqa: E402
import pilot_association_replay as R  # noqa: E402
import pilot_margin_experiment as ME  # noqa: E402
import pilot_stage_diagnostic as SD  # noqa: E402
import pilot_track_crosswalk as P  # noqa: E402


def prepare_and_detect(case, k, ct, ft, cols, fcols, grids, absorb=False):
    """One step's detection with the preparation on `case`'s own grids and the rest on
    `grids`, the control grids, after cropping by the column masks. With the control case
    and all-true masks this is `detect_troughs` step for step. Returns the candidate list
    in the detector's order and the smoothed, cropped fine winds the association reads."""
    from aew.v1port.climatology import smooth9
    from aew.v1port.contours import merge_contours
    from aew.v1port.detection import MAX_ZONAL_WIND, _prepare, trough_axes
    lat_c = np.asarray(case["lat_c"], float)
    lat_f = np.asarray(case["latgrid"], float)[:, 0]
    wind = smooth9(np.asarray(case["u_c"][k], float))[:, cols]
    curvature_c = _prepare(np.asarray(case["currv_anom_c"][k], float), lat_c)[:, cols]
    advection_c = smooth9(np.asarray(case["advcurrv_anom_c"][k], float))[:, cols]
    curvature_f = _prepare(np.asarray(case["currv_anom"][k], float), lat_f)[:, fcols]
    u_s = smooth9(np.asarray(case["u"][k], float))[:, fcols]
    v_s = smooth9(np.asarray(case["v"][k], float))[:, fcols]
    westerly = wind > MAX_ZONAL_WIND
    advection_c = np.where(westerly, np.nan, advection_c)
    curvature_c = np.where(westerly, np.nan, curvature_c)
    weak_c = curvature_c < ct
    advection_c = np.where(weak_c, np.nan, advection_c)
    curvature_c = np.where(weak_c, np.nan, curvature_c)
    curvature_f = np.where(curvature_f < ft, np.nan, curvature_f)
    axes = trough_axes(grids["latgrid_c"], grids["longrid_c"], advection_c)
    if not axes:
        return [], u_s, v_s
    t = float(case["time"][k])
    candidates = [{"time": t, "lat_mean": float(np.mean(la)), "lon_mean": float(np.mean(lo))} for la, lo in axes]
    coarse = merge_contours(candidates, grids["latgrid_c"], grids["longrid_c"], curvature_c, ct, absorb=absorb)
    if not coarse:
        return [], u_s, v_s
    return merge_contours(coarse, grids["latgrid_f"], grids["longrid_f"], curvature_f, ft, absorb=absorb), u_s, v_s


def signature(waves):
    return [(float(w["lat_mean"]), float(w["lon_mean"]), R.region_digest(w["region"])) for w in waves]


def run_season(case, grids, cols, fcols, ct, ft, flags, bind_case=None):
    """The loop, native or margin. With `bind_case` (the control case, native mode), every
    step's candidates are checked against `detect_troughs` on that case."""
    from aew.v1port import pipeline as PL
    from aew.v1port.association import associate_step, finalize_tracks, prune_stale_tracks
    from aew.v1port.detection import detect_troughs
    exclusive, absorb = bool(flags["exclusive"]), bool(flags["absorb"])
    times = np.asarray(case["time"], float).ravel()
    tracks, states = [], []
    registry, graveyard, fates = {}, [], {}
    bound_steps = 0
    for step in range(times.size):
        t = float(times[step])
        waves, u_s, v_s = prepare_and_detect(case, step, ct, ft, cols, fcols, grids, absorb=absorb)
        if bind_case is not None:
            oracle = detect_troughs(t, grids["latgrid_c"], grids["longrid_c"], bind_case["u_c"][step], bind_case["currv_anom_c"][step], bind_case["advcurrv_anom_c"][step],
                                    grids["latgrid_f"], grids["longrid_f"], bind_case["currv_anom"][step], coarse_threshold=ct, fine_threshold=ft, absorb=absorb)
            if signature(waves) != signature(oracle):
                raise SystemExit(f"REFUSED: at step {step} the native preparation does not reproduce detect_troughs")
            bound_steps += 1
        um, vm = PL._median_over(u_s), PL._median_over(v_s)
        tracks, states = associate_step(tracks, states, waves, step, um, vm, exclusive=exclusive)
        for tr in tracks:
            if id(tr) not in registry:
                cand = next(i for i, w in enumerate(waves) if tr["wave_points"][0] is w["region"])
                registry[id(tr)] = f"{step}:{cand}"
                fates[registry[id(tr)]] = "live"
        before = {id(tr): tr for tr in tracks}
        tracks, states = prune_stale_tracks(tracks, states, step, t, waves)
        kept = {id(tr) for tr in tracks}
        for i, tr in before.items():
            if i not in kept:
                fates[registry[i]] = "pruned_in_loop"
                graveyard.append(tr)
                tr["lat_wave"], tr["lon_wave"], tr["wave_points"] = [], [], []
    finished = finalize_tracks(tracks, total_steps=times.size)
    live_of = {id(tr["time"]): tr for tr in tracks}
    out = []
    for f in finished:
        live = live_of[id(f["time"])]
        birth = registry[id(live)]
        fates[birth] = "finished"
        out.append({"birth": birth, "time": [float(x) for x in f["time"]], "lat": [float(x) for x in f["meanlat"]], "lon": [float(x) for x in f["meanlon"]],
                    "steps": [int(s) for s in live["step"]], "raw_lat": [float(x) for x in live["meanlat"]], "raw_lon": [float(x) for x in live["meanlon"]],
                    "n_points": [int(x) for x in live["n_points"]], "region_sha256": [R.region_digest(m) for m in live["wave_points"]]})
    for tr in tracks:
        if fates[registry[id(tr)]] == "live":
            fates[registry[id(tr)]] = "removed_by_final_speed_filter" if len(tr["time"]) >= 2 else "removed_as_single_observation"
    counts = {}
    for v in fates.values():
        counts[v] = counts.get(v, 0) + 1
    return out, counts, bound_steps


def control_grids(case_b):
    lat_c, lon_c = case_b["lat_c"], case_b["lon_c"]
    latgrid_c, longrid_c = np.meshgrid(lat_c, lon_c, indexing="ij")
    return {"lat_c": lat_c, "lon_c": lon_c, "latgrid_c": latgrid_c, "longrid_c": longrid_c, "latgrid_f": case_b["latgrid"], "longrid_f": case_b["longrid"]}


def archive_gate(finished, run):
    mine = [{"time": np.asarray(f["time"]), "meanlat": np.asarray(f["lat"]), "meanlon": np.asarray(f["lon"])} for f in finished]
    return R.gate(mine, run)


def write_gz(path, payload):
    """Published EXCLUSIVELY, the way `exact_tracks.publish_json` publishes: compressed to a
    private temporary file beside the target and linked into place, which the filesystem
    refuses atomically when the target exists. A check-then-write would let two writers
    both pass the check and the later one truncate the earlier's record, and an interrupted
    write would leave a partial record at the final name. Neither can happen here."""
    import tempfile
    blob = json.dumps(payload, sort_keys=True).encode()
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(path) + ".writing-", dir=directory)
    os.close(fd)
    try:
        with gzip.open(tmp, "wb", compresslevel=9) as fh:
            fh.write(blob)
        try:
            os.link(tmp, path)
        except FileExistsError:
            raise SystemExit(f"REFUSED: {path} exists and a record is never overwritten")
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return hashlib.sha256(blob).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("--control-run", "--control-case", "--treatment-run", "--treatment-case", "--out"):
        ap.add_argument(name, required=True)
    ap.add_argument("--preparation", choices=("native", "margin"), required=True)
    ap.add_argument("--year", type=int, required=True)
    args = ap.parse_args(argv)
    if os.path.exists(args.out):
        raise SystemExit(f"REFUSED: {args.out} exists and a record is never overwritten")
    runs = {"B": P.load_run(args.control_run, year=args.year), "C": P.load_run(args.treatment_run, year=args.year)}
    cases = {"B": SD.load_case(args.control_case, runs["B"]["record"]), "C": SD.load_case(args.treatment_case, runs["C"]["record"])}
    flags = runs["B"]["record"]["dataset_specific"]["tracker_flags"]
    ct, ft = float(runs["B"]["record"]["dataset_specific"]["coarse_threshold"]), float(runs["B"]["record"]["dataset_specific"]["fine_threshold"])
    grids = control_grids(cases["B"])
    cols_c, fcols_c = ME.common_columns(cases["B"], cases["C"])
    raw = ME.raw_equality(cases["B"], cases["C"], 0, cols_c, fcols_c)
    if any(v["differing_all_steps"] for v in raw.values()):
        raise SystemExit("REFUSED: the two cases differ in their common columns, so a margin preparation is not the control's fields")
    t0 = time.perf_counter()
    if args.preparation == "native":
        all_c, all_f = np.ones(cases["B"]["lon_c"].size, bool), np.ones(cases["B"]["longrid"].shape[1], bool)
        finished, counts, bound = run_season(cases["B"], grids, all_c, all_f, ct, ft, flags, bind_case=cases["B"])
        gate = archive_gate(finished, runs["B"])
        if not gate["passed"]:
            raise SystemExit(f"REFUSED: the native replay does not reproduce the archived tracks: {gate}")
    else:
        finished, counts, bound = run_season(cases["C"], grids, cols_c, fcols_c, ct, ft, flags)
        gate = archive_gate(finished, runs["B"])         # the sensitivity itself, recorded and never a pass condition
    elapsed = time.perf_counter() - t0
    out = {"generated_by": "scripts/pilot_margin_replay.py", "script_sha256": X.digest(os.path.abspath(__file__)), "year": args.year, "preparation": args.preparation,
           "loop": "export_protocol_case.track_case with the detection preparation as described in the module docstring, from the first step with an empty live state to the last, finalized at the original horizon",
           "runs": {r: {"dir": d, "record_sha256": runs[r]["record_sha256"], "tracks_sha256": runs[r]["tracks_sha256"], "case_sha256": cases[r]["sha256"]} for r, d in (("B", args.control_run), ("C", args.treatment_run))},
           "thresholds": {"coarse": ct, "fine": ft, "from": "the control run's record"}, "tracker_flags": flags,
           "grids": {"coarse_columns": int(cases["B"]["lon_c"].size), "fine_columns": int(cases["B"]["longrid"].shape[1]), "coarse_easternmost_lon": float(cases["B"]["lon_c"].max()), "fine_easternmost_lon": float(cases["B"]["longrid"][0].max())},
           "raw_case_fields_common_columns": raw, "steps": int(cases["B"]["time"].size),
           "native_bind": {"steps_checked_against_detect_troughs": bound} if args.preparation == "native" else None,
           "archive_comparison": gate, "fate_counts": counts, "finished_count": len(finished), "elapsed_seconds": round(elapsed, 1), "finished": finished}
    digest = write_gz(args.out, out)
    print(f"{args.preparation} {args.year}: {len(finished)} finished tracks, archive comparison {'equal' if gate['passed'] else 'differs'} "
          f"({gate['shared']} shared, {gate['only_in_a']} only here, {gate['only_in_b']} only archived, order {gate['order_identical']}), {elapsed:.0f} s, fates {counts}; sha256 {digest[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
