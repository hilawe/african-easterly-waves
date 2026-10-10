#!/usr/bin/env python3
"""The all-edge margin replay: the production loop of `export_protocol_case.track_case`
run for one season with its detection preparation taken either as it is (NATIVE) or from
the wide case, smoothed on the wide grids and cropped to the control grids on every side
afterward (MARGIN), everything else unchanged, through the original horizon and
finalization, with every finished track's raw history retained and the smoothing and
cropping verified at every step.

WHAT THE MARGIN CHANGES, and nothing else. At every step the detector's preparation (the
nine-point smoother on the coarse zonal wind, the coarse curvature anomaly, its advection
and the fine curvature anomaly, with the sign flip south of the equator) runs on the wide
case's arrays and the results are cropped to the control grids' exact rows and columns
before the westerly and threshold masks, the contouring of the trough axes and the two
merge passes, which run on the control grids as in production. The smoothed fine winds
the association takes its region medians from are prepared the same way, smoothed on the
wide fine grid and cropped. The thresholds are the control run's numbers, or, for a
margin run given `--calibration`, a calibration audit's smooth_then_crop pair bound by
digest and read through the mask check's reader, with `--transfer` declaring that the
audit's released pair is not this control run's (the 60 E control), the control's own pair
recorded beside the applied one. Native mode never takes a calibration input, so native
reproduction stays under the released configuration. `associate_step`,
`prune_stale_tracks` and `finalize_tracks` are the production functions, the candidates
keep the order the detector produces, and the run goes from the case's first step with an
empty live state to its last step with finalization at the original horizon.

THE PREPARATION IS VERIFIED AT EVERY STEP in margin mode. The native preparation is also
computed from the control case at the same step, and for each of the six prepared fields
the cells that differ are counted in three places: the interior (every row and column
except the first and last of each grid), the edge ring, and the four corners. The run
refuses if any interior cell differs at any step, since the nine-point smoother reaches
one cell and a difference farther in would mean the two preparations are not the same
computation on the same values. The ring and corner counts are recorded.

THE NATIVE MODE IS THE BIND. With the control case and no crop, the same code must give,
at every step, exactly the candidates `detect_troughs` gives (centers and region digests,
in order), and its finished tracks must equal the archived tracks as an ordered multiset,
or the run refuses. The wide case must itself be bound to the control case by its own
record, and the run checks that binding before anything else.

THE RECORDS carry `preparation` "native" or "margin" with `margin_edges` "all", the same
`runs` and `thresholds` fields in both modes, and every finished track's smoothed arrays
and raw history, so `pilot_margin_compare.py` and `pilot_margin_cohort.py` read them as they
read the eastern-only replay's.

    python3 scripts/pilot_alledge_replay.py --preparation margin --control-run <B dir> --control-case <B case> \\
        --wide-dir <wide case dir> --year 1990 --out <fresh json.gz>
"""
import argparse
import gzip
import gzip
import hashlib
import io
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import exact_tracks as X  # noqa: E402
import pilot_association_replay as R  # noqa: E402
import pilot_margin_replay as MR  # noqa: E402
import pilot_alledge_mask_check as MK  # noqa: E402
import pilot_stage_diagnostic as SD  # noqa: E402
import pilot_track_crosswalk as P  # noqa: E402

PREPARED = (("coarse_curvature", "coarse"), ("coarse_advection", "coarse"), ("coarse_zonal_wind_smoothed", "coarse"),
            ("fine_curvature", "fine"), ("fine_zonal_wind_smoothed", "fine"), ("fine_meridional_wind_smoothed", "fine"))


def load_wide(wide_dir, year):
    """The wide case and its record, the case hashed against the record."""
    from scipy.io import loadmat
    rec_path = os.path.join(wide_dir, f"wide_case_{year}.json")
    with open(rec_path, "rb") as fh:
        rec_blob = fh.read()
    rec = json.loads(rec_blob)
    with open(os.path.join(wide_dir, "wide_case.mat"), "rb") as fh:
        blob = fh.read()
    digest = hashlib.sha256(blob).hexdigest()
    if digest != rec["wide_case_sha256"]:
        raise SystemExit("REFUSED: the wide case is not the one its record names")
    if not all(rec["bound_to_control_case"].values()):
        raise SystemExit("REFUSED: the wide case's record does not bind it to the control case")
    names = ["time", "lat_c", "lon_c", "latgrid", "longrid", "u_c", "v_c", "currv_anom_c", "advcurrv_anom_c", "currv_anom", "u", "v"]
    out = {}
    for name in names:
        out[name] = np.asarray(loadmat(io.BytesIO(blob), variable_names=[name])[name], float)
    out["time"], out["lat_c"], out["lon_c"] = out["time"].ravel(), out["lat_c"].ravel(), out["lon_c"].ravel()
    out["sha256"], out["record"], out["record_sha256"] = digest, rec, hashlib.sha256(rec_blob).hexdigest()
    return out


def selection(case_b, wide):
    """The wide grids' rows and columns that are the control grids', which must be the
    control's exactly and in order, on both grids."""
    rows_c = np.isin(wide["lat_c"], case_b["lat_c"])
    cols_c = np.isin(wide["lon_c"], case_b["lon_c"])
    rows_f = np.isin(wide["latgrid"][:, 0], case_b["latgrid"][:, 0])
    cols_f = np.isin(wide["longrid"][0, :], case_b["longrid"][0, :])
    if not (np.array_equal(wide["lat_c"][rows_c], case_b["lat_c"]) and np.array_equal(wide["lon_c"][cols_c], case_b["lon_c"])
            and np.array_equal(wide["latgrid"][rows_f, 0], case_b["latgrid"][:, 0]) and np.array_equal(wide["longrid"][0, cols_f], case_b["longrid"][0, :])):
        raise SystemExit("REFUSED: the control grids are not a subset of the wide grids in order")
    if not np.array_equal(wide["time"], case_b["time"]):
        raise SystemExit("REFUSED: the wide case and the control case do not share their steps")
    return rows_c, cols_c, rows_f, cols_f


def raw_equality(case_b, wide, sel):
    rows_c, cols_c, rows_f, cols_f = sel
    out = {}
    for name in ("u_c", "v_c", "currv_anom_c", "advcurrv_anom_c"):
        a, b = case_b[name], wide[name][:, rows_c, :][:, :, cols_c]
        out[name] = int(np.count_nonzero((a != b) & ~(np.isnan(a) & np.isnan(b))))
    for name in ("u", "v", "currv_anom"):
        a, b = case_b[name], wide[name][:, rows_f, :][:, :, cols_f]
        out[name] = int(np.count_nonzero((a != b) & ~(np.isnan(a) & np.isnan(b))))
    return out


def prepared_fields(case, k, rows_c, cols_c, rows_f, cols_f):
    """The six prepared fields at step k, before the masks, computed on `case`'s own grids
    and cropped by the selections (all-true selections for the control case itself)."""
    from aew.v1port.climatology import smooth9
    from aew.v1port.detection import _prepare
    lat_c = np.asarray(case["lat_c"], float)
    lat_f = np.asarray(case["latgrid"], float)[:, 0]

    def cc(a):
        return a[rows_c, :][:, cols_c]

    def cf(a):
        return a[rows_f, :][:, cols_f]
    return {"coarse_zonal_wind_smoothed": cc(smooth9(np.asarray(case["u_c"][k], float))),
            "coarse_curvature": cc(_prepare(np.asarray(case["currv_anom_c"][k], float), lat_c)),
            "coarse_advection": cc(smooth9(np.asarray(case["advcurrv_anom_c"][k], float))),
            "fine_curvature": cf(_prepare(np.asarray(case["currv_anom"][k], float), lat_f)),
            "fine_zonal_wind_smoothed": cf(smooth9(np.asarray(case["u"][k], float))),
            "fine_meridional_wind_smoothed": cf(smooth9(np.asarray(case["v"][k], float)))}


def detect_from_prepared(fields, t, ct, ft, grids, absorb=False, lineage=None, coarse_clip_radius_deg=None, coarse_position_from_input=False):
    """The detector from its prepared fields: the masks, the axes, the two merges.
    `lineage`, when a dict is passed, receives the two merges' lineage lists under
    "coarse" and "fine" and the axes under "axes". Recording changes nothing.
    `coarse_clip_radius_deg` is the experimental clip at exhaustion, coarse merge only.
    `coarse_position_from_input` is the experimental input-position assignment, coarse
    merge only; the fine merge is called exactly as before."""
    from aew.v1port.contours import merge_contours
    from aew.v1port.detection import MAX_ZONAL_WIND, trough_axes
    wind, curvature_c, advection_c, curvature_f = fields["coarse_zonal_wind_smoothed"], fields["coarse_curvature"], fields["coarse_advection"], fields["fine_curvature"]
    westerly = wind > MAX_ZONAL_WIND
    advection_c = np.where(westerly, np.nan, advection_c)
    curvature_c = np.where(westerly, np.nan, curvature_c)
    weak_c = curvature_c < ct
    advection_c = np.where(weak_c, np.nan, advection_c)
    curvature_c = np.where(weak_c, np.nan, curvature_c)
    curvature_f = np.where(curvature_f < ft, np.nan, curvature_f)
    axes = trough_axes(grids["latgrid_c"], grids["longrid_c"], advection_c)
    if lineage is not None:
        lineage.update({"axes": axes, "coarse": [], "fine": []})
    if not axes:
        return []
    candidates = [{"time": t, "lat_mean": float(np.mean(la)), "lon_mean": float(np.mean(lo))} for la, lo in axes]
    coarse = merge_contours(candidates, grids["latgrid_c"], grids["longrid_c"], curvature_c, ct, absorb=absorb,
                            lineage=None if lineage is None else lineage["coarse"], clip_radius_deg=coarse_clip_radius_deg,
                            position_from_input=coarse_position_from_input)
    if not coarse:
        return []
    return merge_contours(coarse, grids["latgrid_f"], grids["longrid_f"], curvature_f, ft, absorb=absorb,
                          lineage=None if lineage is None else lineage["fine"])


def ring_counts(a, b):
    """Cells differing between two prepared fields on one grid, in the interior, the edge
    ring and the four corners."""
    diff = (np.isnan(a) != np.isnan(b)) | (~np.isnan(a) & ~np.isnan(b) & (a != b))
    ring = np.zeros(a.shape, bool)
    ring[0, :] = ring[-1, :] = True
    ring[:, 0] = ring[:, -1] = True
    corners = np.zeros(a.shape, bool)
    corners[0, 0] = corners[0, -1] = corners[-1, 0] = corners[-1, -1] = True
    return {"interior": int(np.count_nonzero(diff & ~ring)), "ring": int(np.count_nonzero(diff & ring)), "corners": int(np.count_nonzero(diff & corners))}


def run_season(case_b, grids, ct, ft, flags, wide=None, sel=None, trace=None):
    """The loop. Native mode (`wide` None) prepares on the control case and binds to
    `detect_troughs`. Margin mode prepares on the wide case, crops, and verifies against the
    native preparation at every step. A TRACE (`trace` a dict with `steps`, `box`,
    `families`, `point` and `radius_deg`) is a side record of the association replay's
    kind at the given steps, never written to the live state. It holds the candidates,
    the live tracks, the matching read bound to the production call, the claims, the
    seeds, the prunes, the candidate nearest the point, and at the end the raw history
    and fate of every track seen in the box. It is returned in `trace["result"]`."""
    from aew.v1port import pipeline as PL
    from aew.v1port.association import associate_step, finalize_tracks, prune_stale_tracks
    from aew.v1port.detection import detect_troughs
    exclusive, absorb = bool(flags["exclusive"]), bool(flags["absorb"])
    times = np.asarray(case_b["time"], float).ravel()
    all_c = (np.ones(case_b["lat_c"].size, bool), np.ones(case_b["lon_c"].size, bool))
    all_f = (np.ones(case_b["latgrid"].shape[0], bool), np.ones(case_b["longrid"].shape[1], bool))
    tracks, states = [], []
    registry, graveyard, fates = {}, [], {}
    trace_steps = set(trace["steps"]) if trace is not None else set()
    trace_log, touched, born, pruned_at, histories, finished_index = [], set(), {}, {}, {}, {}
    bound_steps, verification = 0, {"steps": 0, "interior_differing_max": 0, "ring_differing_total": {n: 0 for n, _g in PREPARED}, "corner_differing_total": {n: 0 for n, _g in PREPARED},
                                    "ring_differing_steps": {n: 0 for n, _g in PREPARED}}
    for step in range(times.size):
        t = float(times[step])
        native_fields = prepared_fields(case_b, step, all_c[0], all_c[1], all_f[0], all_f[1])
        if wide is None:
            fields = native_fields
            waves = detect_from_prepared(fields, t, ct, ft, grids, absorb=absorb)
            oracle = detect_troughs(t, grids["latgrid_c"], grids["longrid_c"], case_b["u_c"][step], case_b["currv_anom_c"][step], case_b["advcurrv_anom_c"][step],
                                    grids["latgrid_f"], grids["longrid_f"], case_b["currv_anom"][step], coarse_threshold=ct, fine_threshold=ft, absorb=absorb)
            if MR.signature(waves) != MR.signature(oracle):
                raise SystemExit(f"REFUSED: at step {step} the native preparation does not reproduce detect_troughs")
            bound_steps += 1
        else:
            fields = prepared_fields(wide, step, *sel)
            for name, _grid in PREPARED:
                c = ring_counts(native_fields[name], fields[name])
                if c["interior"]:
                    raise SystemExit(f"REFUSED: at step {step} the margin preparation differs from the native one in {c['interior']} interior cells of {name}")
                verification["ring_differing_total"][name] += c["ring"]
                verification["corner_differing_total"][name] += c["corners"]
                verification["ring_differing_steps"][name] += int(c["ring"] > 0)
            verification["steps"] += 1
            waves = detect_from_prepared(fields, t, ct, ft, grids, absorb=absorb)
        um, vm = PL._median_over(fields["fine_zonal_wind_smoothed"]), PL._median_over(fields["fine_meridional_wind_smoothed"])
        logged = step in trace_steps
        if logged:                                                   # read before the production call, never acted on
            entry = {"step": step, "time": t, "date": f"{SD.date_of(t):%Y-%m-%dT%H:00Z}", "n_live_before": len(tracks),
                     "candidates": R.describe_candidates(waves, um, vm, trace["families"]),
                     "live_before": R.live_snapshot(tracks, states, step, registry),
                     "matching": R.read_matching(tracks, states, waves, step, exclusive, registry) if waves else []}
        n_before = {id(tr): len(tr["time"]) for tr in tracks}
        tracks, states = associate_step(tracks, states, waves, step, um, vm, exclusive=exclusive)
        claims, seeds = [], []
        for tr in tracks:
            if id(tr) not in registry:
                cand = next(i for i, w in enumerate(waves) if tr["wave_points"][0] is w["region"])
                registry[id(tr)] = f"{step}:{cand}"
                fates[registry[id(tr)]] = "live"
                if trace is not None:
                    born[registry[id(tr)]] = {"step": step, "time": t, "candidate": cand, "lat": float(tr["meanlat"][0]), "lon": float(tr["meanlon"][0])}
                    seeds.append({"birth": registry[id(tr)], "candidate": cand, "lat": float(tr["meanlat"][0]), "lon": float(tr["meanlon"][0])})
            elif trace is not None and len(tr["time"]) > n_before[id(tr)]:
                cand = next(i for i, w in enumerate(waves) if tr["wave_points"][-1] is w["region"])
                claims.append({"birth": registry[id(tr)], "candidate": cand, "n_obs": len(tr["time"]), "lat": float(tr["meanlat"][-1]), "lon": float(tr["meanlon"][-1])})
        if logged:
            for m in entry["matching"]:                              # bind the read to the production call
                actual = next((c["candidate"] for c in claims if c["birth"] == m["birth"]), None)
                if actual != m["chosen"]:
                    raise SystemExit(f"REFUSED: at step {step} the matching read chose {m['chosen']} for track {m['birth']} and the production call claimed {actual}")
            entry["claims"], entry["seeds"] = claims, seeds
            for row in entry["live_before"]:
                if R.in_box(row["last_lat"], row["last_lon"], trace["box"]):
                    touched.add(row["birth"])
            for s in seeds + claims:
                if R.in_box(s["lat"], s["lon"], trace["box"]):
                    touched.add(s["birth"])
            nearest = None
            if trace.get("point") is not None and waves:
                plat, plon = float(trace["point"][0]), float(trace["point"][1])
                d = [float(np.hypot(float(w["lat_mean"]) - plat, float(w["lon_mean"]) - plon)) for w in waves]
                i = int(np.argmin(d))
                held = next((c["birth"] for c in claims if c["candidate"] == i), None) or next((s["birth"] for s in seeds if s["candidate"] == i), None)
                nearest = {"candidate": i, "distance_deg": round(d[i], 3), "lat": float(waves[i]["lat_mean"]), "lon": float(waves[i]["lon_mean"]),
                           "n_points": int(np.asarray(waves[i]["lat_wave"]).size), "held_by": held, "within_radius": bool(d[i] <= float(trace.get("radius_deg", 5.0)))}
            entry["nearest_to_point"] = nearest
        before = {id(tr): tr for tr in tracks}
        tracks, states = prune_stale_tracks(tracks, states, step, t, waves)
        kept = {id(tr) for tr in tracks}
        pruned = []
        for i, tr in before.items():
            if i not in kept:
                fates[registry[i]] = "pruned_in_loop"
                if trace is not None:
                    pruned_at[registry[i]] = {"step": step, "time": t, "n_obs": len(tr["time"]), "last_step": int(tr["step"][-1])}
                    if registry[i] in touched:
                        histories[registry[i]] = R.raw_history(tr)
                    pruned.append({"birth": registry[i], "n_obs": len(tr["time"]), "last_step": int(tr["step"][-1])})
                graveyard.append(tr)
                tr["lat_wave"], tr["lon_wave"], tr["wave_points"] = [], [], []
        if logged:
            entry["pruned"] = pruned
            trace_log.append(entry)
    finished = finalize_tracks(tracks, total_steps=times.size)
    live_of = {id(tr["time"]): tr for tr in tracks}
    out = []
    for f in finished:
        live = live_of[id(f["time"])]
        birth = registry[id(live)]
        fates[birth] = "finished"
        finished_index[birth] = len(out)
        out.append({"birth": birth, "time": [float(x) for x in f["time"]], "lat": [float(x) for x in f["meanlat"]], "lon": [float(x) for x in f["meanlon"]],
                    "steps": [int(s) for s in live["step"]], "raw_lat": [float(x) for x in live["meanlat"]], "raw_lon": [float(x) for x in live["meanlon"]],
                    "n_points": [int(x) for x in live["n_points"]], "region_sha256": [R.region_digest(m) for m in live["wave_points"]]})
    for tr in tracks:
        if fates[registry[id(tr)]] == "live":
            fates[registry[id(tr)]] = "removed_by_final_speed_filter" if len(tr["time"]) >= 2 else "removed_as_single_observation"
        if trace is not None and registry[id(tr)] in touched:
            histories[registry[id(tr)]] = R.raw_history(tr)
    counts = {}
    for v in fates.values():
        counts[v] = counts.get(v, 0) + 1
    if trace is not None:
        summary = []
        for birth in sorted(touched, key=lambda b: (born[b]["step"], born[b]["candidate"])):
            b = born[birth]
            near = None if trace.get("point") is None else round(float(np.hypot(b["lat"] - float(trace["point"][0]), b["lon"] - float(trace["point"][1]))), 3)
            summary.append({"birth": birth, "seeded_step": b["step"], "seeded_date": f"{SD.date_of(b['time']):%Y-%m-%dT%H:00Z}", "seeded_lat": b["lat"], "seeded_lon": b["lon"],
                            "seeded_in_window": b["step"] in trace_steps, "seed_distance_to_point_deg": near, "fate": fates[birth],
                            "n_obs": len(histories[birth]["steps"]) if birth in histories else None, "pruned_at": pruned_at.get(birth), "finished_index": finished_index.get(birth)})
        within = [s for s in summary if s["seed_distance_to_point_deg"] is not None and s["seed_distance_to_point_deg"] <= float(trace.get("radius_deg", 5.0))]
        trace["result"] = {"steps": trace_log, "touched": sorted(touched), "born": {b: born[b] for b in touched}, "fates": {b: fates[b] for b in touched},
                           "pruned_at": {b: pruned_at[b] for b in touched if b in pruned_at}, "finished_index": {b: finished_index[b] for b in touched if b in finished_index},
                           "histories": histories, "summary": summary,
                           "seeded_within_radius": {"in_window": sum(1 for s in within if s["seeded_in_window"]), "any_time": len(within),
                                                    "note": "the summary lists every track seen in the box at a logged step, whenever it was seeded, so a seed within the radius may predate the window"}}
    return out, counts, bound_steps, (verification if wide is not None else None)


def thresholds_for_run(run_pair, calibration_path=None, transfer=False, domain_calibration=False, control_domain=None):
    """The pair the run applies, with its provenance. Without a calibration input it is the
    control run's own pair, the released configuration. With one it is the calibration
    audit's smooth_then_crop pair, read through the mask check's bound reader against the
    control's pair, or under one of two declarations when the audit's released pair is
    not this control run's. A TRANSFER applies a pair calibrated on another domain. A
    DOMAIN CALIBRATION applies the pair calibrated on this control's own domain, whose
    archived-order anchor is another run's released pair on that domain (the D run's at
    60 E, where the C control carries the transferred 40 E pair), and the audit's domain
    must be the control's. In both the audit's own bind is still checked and the
    control's pair is recorded beside the applied one. The archived control metadata is
    never edited to supply different numbers."""
    ct, ft = float(run_pair[0]), float(run_pair[1])
    if transfer and domain_calibration:
        raise SystemExit("REFUSED: declare one of --transfer and --domain-calibration, not both")
    if calibration_path is None:
        if transfer or domain_calibration:
            raise SystemExit("REFUSED: a declared transfer or domain calibration needs a calibration input")
        return ct, ft, {"coarse": ct, "fine": ft, "from": "the control run's record"}
    with open(calibration_path, "rb") as fh:
        blob = fh.read()
    rec = json.loads(blob)
    audited = MK._pair(rec.get("released_artifact"))
    if audited is None:
        raise SystemExit("REFUSED: the calibration input carries no audited released pair")
    if transfer or domain_calibration:
        if audited == (ct, ft):
            raise SystemExit("REFUSED: the audit's released pair is this control run's pair, so no flag is needed and nothing is transferred")
        if domain_calibration:
            dom = rec.get("domain") or {}
            if control_domain is None or list(dom.get("lat_range", [])) != [float(x) for x in control_domain["lat"]] or list(dom.get("lon_range", [])) != [float(x) for x in control_domain["lon"]]:
                raise SystemExit(f"REFUSED: the audit's domain {dom} is not this control run's domain {control_domain}, so it is not a domain calibration")
        pair = MK.recomputed_pair(rec, audited)
    else:
        pair = MK.recomputed_pair(rec, (ct, ft))
    record = {"coarse": pair[0], "fine": pair[1], "from": "the calibration audit's smooth_then_crop order",
              "calibration_audit": {"path": calibration_path, "sha256": hashlib.sha256(blob).hexdigest(), "generated_by": rec.get("generated_by"),
                                    "domain": rec.get("domain"), "audited_released_pair": list(audited)},
              "control_run_pair": [ct, ft], "transfer": bool(transfer), "domain_calibration": bool(domain_calibration)}
    return pair[0], pair[1], record


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("--control-run", "--control-case", "--wide-dir", "--out"):
        ap.add_argument(name, required=True)
    ap.add_argument("--preparation", choices=("native", "margin"), required=True)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--calibration", help="a calibration audit record whose smooth_then_crop pair the margin run applies. Absent, the control run's own pair")
    ap.add_argument("--transfer", action="store_true", help="declare that the audit's released pair is not this control run's, so its recomputed pair is transferred to this domain")
    ap.add_argument("--domain-calibration", action="store_true", help="declare that the audit is of this control's own domain, anchored to another run's released pair on it")
    ap.add_argument("--trace-families", help="a diagnostic families file. With --trace-family, the loop is traced at that family's window steps as a side record")
    ap.add_argument("--trace-family", help="the name of the family to trace")
    ap.add_argument("--trace-point", nargs=3, type=float, metavar=("LAT", "LON", "DAYS_SINCE_1900"), help="a documented point. At each traced step the candidate nearest it is recorded")
    ap.add_argument("--trace-radius", type=float, default=5.0, help="degrees within which a candidate or a seed counts as near the point")
    ap.add_argument("--reproduces", help="a retained replay record whose finished histories this run must reproduce exactly, refused otherwise")
    args = ap.parse_args(argv)
    if os.path.exists(args.out):
        raise SystemExit(f"REFUSED: {args.out} exists and a record is never overwritten")
    if (args.trace_family or args.trace_point) and not args.trace_families:
        raise SystemExit("REFUSED: a trace needs --trace-families and --trace-family")
    if args.preparation == "native" and (args.calibration or args.transfer or args.domain_calibration):
        raise SystemExit("REFUSED: native reproduction stays under the released configuration and takes no calibration input")
    run = P.load_run(args.control_run, year=args.year)
    case_b = SD.load_case(args.control_case, run["record"])
    wide = load_wide(args.wide_dir, args.year)
    if wide["record"]["control_run"]["case_sha256"] != case_b["sha256"]:
        raise SystemExit("REFUSED: the wide case was not bound to this control case")
    flags = run["record"]["dataset_specific"]["tracker_flags"]
    ds = run["record"]["dataset_specific"]
    ct, ft, thresholds = thresholds_for_run((float(ds["coarse_threshold"]), float(ds["fine_threshold"])), args.calibration, args.transfer, args.domain_calibration, run["domain"])
    grids = MR.control_grids(case_b)
    sel = selection(case_b, wide)
    raw = raw_equality(case_b, wide, sel)
    if any(raw.values()):
        raise SystemExit(f"REFUSED: the wide case differs from the control case on the control's own cells: {raw}")
    trace = None
    if args.trace_families:
        fams = [f for f in json.load(open(args.trace_families))["families"] if f["name"] == args.trace_family]
        if len(fams) != 1:
            raise SystemExit(f"REFUSED: {args.trace_families} holds no single family named {args.trace_family!r}")
        steps, _per = R.window_steps(np.asarray(case_b["time"], float).ravel(), fams)
        trace = {"steps": set(steps), "box": fams[0]["box"], "families": fams, "point": tuple(args.trace_point) if args.trace_point else None, "radius_deg": args.trace_radius}
    t0 = time.perf_counter()
    if args.preparation == "native":
        finished, counts, bound, verification = run_season(case_b, grids, ct, ft, flags, trace=trace)
        gate = MR.archive_gate(finished, run)
        if not gate["passed"]:
            raise SystemExit(f"REFUSED: the native replay does not reproduce the archived tracks: {gate}")
    else:
        finished, counts, bound, verification = run_season(case_b, grids, ct, ft, flags, wide=wide, sel=sel, trace=trace)
        gate = MR.archive_gate(finished, run)        # the sensitivity itself, recorded and never a pass condition
    elapsed = time.perf_counter() - t0
    reproduces = None
    if args.reproduces:                               # the trace is of the frozen candidate only if this run is that run
        with gzip.open(args.reproduces, "rb") as fh:
            blob = fh.read()
        ref = json.loads(blob)
        mine = [(f["time"], f["lat"], f["lon"]) for f in finished]
        theirs = [(f["time"], f["lat"], f["lon"]) for f in ref["finished"]]
        if mine != theirs:
            raise SystemExit(f"REFUSED: this run's {len(mine)} finished histories are not the retained record's {len(theirs)}, so the trace is not of the frozen candidate")
        reproduces = {"path": args.reproduces, "sha256": hashlib.sha256(blob).hexdigest(), "finished_identical": True, "finished": len(finished)}
    trace_out = None
    if trace is not None:
        fam = trace["families"][0]
        trace_out = {"families_file": args.trace_families, "families_sha256": X.digest(args.trace_families), "family": fam["name"], "window": fam["window"], "box": fam["box"],
                     "point": None if trace["point"] is None else {"lat": trace["point"][0], "lon": trace["point"][1], "days_since_1900": trace["point"][2]},
                     "radius_deg": trace["radius_deg"], "logged_steps": sorted(trace["steps"]), **trace["result"]}
    out = {"generated_by": "scripts/pilot_alledge_replay.py", "script_sha256": X.digest(os.path.abspath(__file__)), "year": args.year, "preparation": args.preparation, "margin_edges": "all",
           "loop": "export_protocol_case.track_case with the detection preparation as described in the module docstring, from the first step with an empty live state to the last, finalized at the original horizon",
           "runs": {"B": {"dir": args.control_run, "record_sha256": run["record_sha256"], "tracks_sha256": run["tracks_sha256"], "case_sha256": case_b["sha256"], "control_domain": run["domain"]},
                    "wide": {"dir": args.wide_dir, "record_sha256": wide["record_sha256"], "case_sha256": wide["sha256"], "box": wide["record"]["box"]}},
           "thresholds": thresholds, "tracker_flags": flags,
           "configuration": "released" if args.calibration is None else ("recomputed calibration, transferred" if args.transfer else ("recomputed calibration, the domain's own" if args.domain_calibration else "recomputed calibration")),
           "grids": {"coarse_rows": int(case_b["lat_c"].size), "coarse_columns": int(case_b["lon_c"].size), "fine_rows": int(case_b["latgrid"].shape[0]), "fine_columns": int(case_b["longrid"].shape[1])},
           "raw_case_fields_control_cells_differing": raw, "steps": int(case_b["time"].size),
           "native_bind": {"steps_checked_against_detect_troughs": bound} if args.preparation == "native" else None,
           "preparation_verification": verification,
           "archive_comparison": gate, "fate_counts": counts, "finished_count": len(finished), "elapsed_seconds": round(elapsed, 1),
           "reproduces": reproduces, "trace": trace_out, "finished": finished}
    digest = MR.write_gz(args.out, out)
    if trace_out is not None:
        w = trace_out["seeded_within_radius"]
        print(f"Trace of family {trace_out['family']} at {len(trace_out['logged_steps'])} steps, {len(trace_out['touched'])} tracks seen in the box, "
              f"{w['in_window']} seeded within the radius of the point inside the window and {w['any_time']} at any time.")
    print(f"{args.preparation} all-edge {args.year}: {len(finished)} finished tracks, archive comparison {'equal' if gate['passed'] else 'differs'} "
          f"({gate['shared']} shared, {gate['only_in_a']} only here, {gate['only_in_b']} only archived, order {gate['order_identical']}), {elapsed:.0f} s, fates {counts}. "
          f"Verification {verification}. Digest {digest[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
