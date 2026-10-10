#!/usr/bin/env python3
"""The eastern pilot's bounded visual audit: the predeclared case selection from a
crosswalk report, and one figure per case from the retained fields.

SELECTION, fixed in EASTERN_EXTENSION_PILOT_BRIEF_2026-10-05.md section 7 item 8 before any
result was viewed, at most 12 cases per season:
- the 6 paired tracks with the longest candidate recovered prefix in time steps, ties by
  the earlier treatment start;
- the 3 unpaired treatment tracks starting east of 40 E with the longest lifetime;
- 3 comparison cases, the paired tracks starting within 5 degrees west of 40 E in the
  control whose treatment counterpart starts at the same time and place (the same first
  time, latitude and longitude, whatever the later positions do), the longest lifetimes
  first.
Each case names the control and treatment track indices from the report, so the figure
and any later reading carry the same identity as the crosswalk.

FIGURE, one per case: for every time step of the candidate prefix and the two steps on
either side of it (or, for an unpaired or comparison case, the first six steps), THE
TRACKER'S OWN COARSE-STAGE VIEW from the treatment's retained case file, bound by digest to
the treatment record: the curvature-vorticity anomaly smoothed by the port's nine-point
smoother with cyclonic made positive in both hemispheres, masked where the smoothed zonal
wind exceeds 2.5 m/s or the anomaly is below the record's coarse threshold (masked cells
gray), and the trough axes the detector finds as the zero contour of the smoothed
advection anomaly inside that mask, drawn as black lines, with the 700 hPa wind as arrows,
the treatment track in red and the control track in blue. Where retained GridSat-B1
imagery covers the step, the infrared brightness temperature below 240 K is hatched in the
sequence tool's box around the track position, read through that tool's own readers
(timestamp validated, partial spatial coverage reported). A record beside the figures names
every panel's step, position, trough-axis count and satellite coverage. The figure is
context for a human reading of continuity (SUPPORTED, CONTRADICTED, UNDETERMINED, by the
rule in the brief, read from whether a trough axis is present at the track position and
moves continuously across the steps); it decides nothing, and visible convection is not a
requirement.

PROXIMITY, a record beside the figures that turns the reading's "a trough axis at the track
position" into a number: for every step a figure shows, the distance in degrees from the
reference position to the nearest of the detector's trough axes (point to polyline, plain
degrees of latitude and longitude as the crosswalk measures separation), and whether it is
within AXIS_TOLERANCE_DEG, one coarse grid cell. The reference position is the treatment
track's position at the step, else the control's, else (for the padding steps before the
treatment starts) the treatment's first position, labeled as such. The tolerance was set
after the figures were viewed, to state the readings' "within about a grid cell" exactly.
Stored track positions are five-point moving averages of axis centers (the association
stage's final smoothing), so even a well-tracked trough sits some way from its
instantaneous axis. REFERENCE therefore measures the same quantity for every track of the
run against the run's own axes, and the audited cases are read against that distribution.

    python3 scripts/pilot_visual_audit_cases.py select --report <crosswalk json> --out <cases json>
    python3 scripts/pilot_visual_audit_cases.py render --cases <cases json> --case-file <tracker_case.mat of the treatment run> \\
        --control-run <run dir B> --treatment-run <run dir C> --year 1990 --imagery data/gridsat_jas --out-dir <figures dir>
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import exact_tracks as X  # noqa: E402

MAX_PREFIX, MAX_UNPAIRED, MAX_COMPARISON = 6, 3, 3
EAST_START_DEG, COMPARISON_BAND_DEG, PREFIX_PAD_STEPS = 40.0, 5.0, 2
ANOM_LIMIT = 1.5e-5                          # the shading saturates here, about the prepared field's 95th percentile (s^-1)
AXIS_TOLERANCE_DEG = 2.0                     # one coarse grid cell, declared after the figures were viewed
PAGE_PANELS, NCOLS = 12, 2                   # a case with more steps than one page holds is split over pages


def in_season(month):
    """The brief's season membership, the first observation's month in June to September."""
    return 6 <= int(month) <= 9


def select(report, prefix_rule="strict", season_only=False):
    """The case list from a crosswalk report, by the predeclared rule. `prefix_rule`
    "strict" takes the crosswalk's PREFIX CHANGED class (an earlier start and then exactly
    the control's observations). "loose" takes the brief's section 7 item 2 definition of a
    candidate recovered eastern history, which is the same end, a treatment start earlier
    and farther east than the control's, and that treatment start east of 40 E in the added
    strip, without requiring the positions in between to match. The cases file says which
    rule was used, since "loose" widens the predeclared class and is used only when the
    strict class is empty, stated as such in the assessment. "eastern" is the brief's item 2
    class as written (a pair whose control track starts later and farther west than the
    treatment's), any pair class and any end, restricted to treatment starts east of 40 E,
    the added strip that item 2's title names. It replaces "loose", which added a same-end
    condition the brief does not have. `season_only` is a SUPPLEMENTARY
    selection, declared after the predeclared one was viewed, restricted to tracks whose
    first observation is in June to September (for a pair, the control's or the
    treatment's), so the season the pilot question is about is read. The cases file says so."""
    pairs, attrs_a, attrs_b = report["pairs"], report["attributes_a"], report["attributes_b"]
    if season_only:
        pairs = [p for p in pairs if in_season(attrs_a[p["a"]]["start_month"]) or in_season(attrs_b[p["b"]]["start_month"])]
    if prefix_rule == "strict":
        prefixes = [p for p in pairs if p["class"] == "prefix changed"]
    elif prefix_rule == "eastern":
        prefixes = [p for p in pairs if attrs_b[p["b"]]["start_time"] < attrs_a[p["a"]]["start_time"]
                    and attrs_b[p["b"]]["start_lon"] > attrs_a[p["a"]]["start_lon"] and attrs_b[p["b"]]["start_lon"] > EAST_START_DEG]
    elif prefix_rule == "loose":
        prefixes = [p for p in pairs if p["class"] in ("prefix changed", "otherwise changed") and p["start_shift_days"] < 0
                    and attrs_b[p["b"]]["start_lon"] > attrs_a[p["a"]]["start_lon"] and attrs_b[p["b"]]["start_lon"] > EAST_START_DEG]
    else:
        raise SystemExit(f"REFUSED: unknown prefix rule {prefix_rule!r}")
    prefixes.sort(key=lambda p: (-p["candidate_prefix_steps"], attrs_b[p["b"]]["start_time"]))
    unpaired = [j for j in report["unpaired_b"] if attrs_b[j]["start_lon"] > EAST_START_DEG
                and (not season_only or in_season(attrs_b[j]["start_month"]))]
    unpaired.sort(key=lambda j: (-attrs_b[j]["observations"], attrs_b[j]["start_time"]))
    # THE COMPARISON RULE IS THE BRIEF'S: the same first time and place on both sides, whatever
    # the later positions do, so the pair's class is recorded with the case rather than required
    comparison = [p for p in pairs
                  if EAST_START_DEG - COMPARISON_BAND_DEG <= attrs_a[p["a"]]["start_lon"] < EAST_START_DEG
                  and attrs_b[p["b"]]["start_time"] == attrs_a[p["a"]]["start_time"]
                  and attrs_b[p["b"]]["start_lat"] == attrs_a[p["a"]]["start_lat"]
                  and attrs_b[p["b"]]["start_lon"] == attrs_a[p["a"]]["start_lon"]]
    comparison.sort(key=lambda p: (-attrs_a[p["a"]]["observations"], attrs_a[p["a"]]["start_time"]))
    cases = []
    for p in prefixes[:MAX_PREFIX]:
        cases.append({"kind": "candidate recovered prefix", "a": p["a"], "b": p["b"], "prefix_steps": p["candidate_prefix_steps"],
                      "prefix_degrees": p["candidate_prefix_degrees"], "start_shift_days": p["start_shift_days"], "pair_class": p["class"]})
    for j in unpaired[:MAX_UNPAIRED]:
        cases.append({"kind": "unpaired eastern start", "a": None, "b": j, "observations": attrs_b[j]["observations"],
                      "start_lon": attrs_b[j]["start_lon"], "potentially_censored_start": attrs_b[j]["potentially_censored_start"]})
    for p in comparison[:MAX_COMPARISON]:
        cases.append({"kind": "comparison, same start near the old edge", "a": p["a"], "b": p["b"], "start_lon": attrs_a[p["a"]]["start_lon"],
                      "start_lat": attrs_a[p["a"]]["start_lat"], "pair_class": p["class"]})
    return {"generated_by": "scripts/pilot_visual_audit_cases.py", "script_sha256": X.digest(os.path.abspath(__file__)),
            "report": report.get("runs"), "year": report.get("year"),
            "rule": {"max_prefix": MAX_PREFIX, "max_unpaired": MAX_UNPAIRED, "max_comparison": MAX_COMPARISON,
                     "east_start_deg": EAST_START_DEG, "comparison_band_deg": COMPARISON_BAND_DEG, "prefix_rule": prefix_rule,
                     "season_only": season_only,
                     "season_note": ("SUPPLEMENTARY, declared after the predeclared selection was viewed: only tracks whose first "
                                     "observation is in June to September (a pair qualifies by either side)" if season_only else
                                     "the predeclared selection, every month"),
                     "comparison_note": "pairs with the same first time, latitude and longitude on both sides, any class",
                     "prefix_rule_note": {"strict": "the crosswalk's strict prefix class",
                                          "eastern": "the brief's item 2 class as written (the control track starts later and farther west), any pair "
                                                     "class and any end, restricted to treatment starts east of 40 E",
                                          "loose": "SUPERSEDED: same end, a treatment start earlier and farther east than the control's and east of 40 E; "
                                                   "the same-end condition is not the brief's"}[prefix_rule]},
            "candidates_available": {"prefix": len(prefixes), "unpaired_eastern": len(unpaired), "comparison": len(comparison)},
            "cases": cases, "readings": "to be entered by the human reader: supported, contradicted or undetermined, per case"}


def steps_for(case, track_b, track_a, times):
    """The time steps the figure shows: the prefix and two steps either side, else the
    first six steps of the treatment track."""
    if case["kind"] == "candidate recovered prefix" and track_a is not None:
        first_shared = float(track_a["time"][0])
        lo = max(0, int(np.searchsorted(times, track_b["time"][0])) - PREFIX_PAD_STEPS)
        hi = min(times.size - 1, int(np.searchsorted(times, first_shared)) + PREFIX_PAD_STEPS)
        return list(range(lo, hi + 1))
    k0 = int(np.searchsorted(times, track_b["time"][0]))
    return list(range(k0, min(times.size, k0 + 6)))


def tracker_view(u, anom, adv, lat_c, lon_c, coarse_threshold):
    """The coarse stage of the port's detector, step for step as detect_troughs prepares
    it before merging: smooth the zonal wind, the anomaly and its advection with the
    nine-point smoother, make cyclonic positive in both hemispheres, discard westerly flow
    above MAX_ZONAL_WIND and anomalies below the coarse threshold, and find the trough axes
    as the zero contour of the masked advection. Returns the masked prepared anomaly (NaN
    where masked) and the axes as (lat, lon) polylines. The functions and constants are
    the detector's own, so the view is what the tracker saw at this stage."""
    from aew.v1port.climatology import smooth9
    from aew.v1port.detection import MAX_ZONAL_WIND, _prepare, trough_axes
    lat_c, lon_c = np.asarray(lat_c, float), np.asarray(lon_c, float)
    wind = smooth9(np.asarray(u, float))
    curvature = _prepare(np.asarray(anom, float), lat_c)
    advection = smooth9(np.asarray(adv, float))
    masked = (wind > MAX_ZONAL_WIND) | (curvature < coarse_threshold)
    curvature = np.where(masked, np.nan, curvature)
    advection = np.where(masked, np.nan, advection)
    latgrid, longrid = np.meshgrid(lat_c, lon_c, indexing="ij")
    return curvature, trough_axes(latgrid, longrid, advection)


def nearest_axis_deg(lat, lon, axes):
    """The distance in degrees from (lat, lon) to the nearest point of any axis polyline,
    None when there is no axis."""
    best = None
    for lats, lons in axes:
        y, x = np.asarray(lats, float), np.asarray(lons, float)
        if y.size == 1:
            d = float(np.hypot(y[0] - lat, x[0] - lon))
        else:
            y0, x0, dy, dx = y[:-1], x[:-1], np.diff(y), np.diff(x)
            seg = dy * dy + dx * dx
            t = np.clip(np.where(seg > 0, ((lat - y0) * dy + (lon - x0) * dx) / np.where(seg > 0, seg, 1.0), 0.0), 0.0, 1.0)
            d = float(np.min(np.hypot(y0 + t * dy - lat, x0 + t * dx - lon)))
        best = d if best is None else min(best, d)
    return best


def _bound_inputs(cases_path, case_file, control_run, treatment_run, year):
    """The cases file, both runs and the treatment's case file, each bound by digest to the
    selection, refused otherwise. Shared by render and proximity."""
    import pilot_track_crosswalk as P
    cases = json.load(open(cases_path))
    a, b = P.load_run(control_run, year=year), P.load_run(treatment_run, year=year)
    for side, run in (("a", a), ("b", b)):
        named = cases["report"][side]
        for key in ("record_sha256", "manifest_sha256", "tracks_sha256"):
            if key not in named:
                raise SystemExit(f"REFUSED: the cases file does not carry the {side} run's {key}, so the figures cannot be bound to the selected run")
            if named[key] != run[key]:
                raise SystemExit(f"REFUSED: the cases were selected from a run whose {key} is not the {side} run's given here")
    case_digest = X.digest(case_file)
    if case_digest != b["record"]["dataset_specific"].get("case_sha256"):
        raise SystemExit(f"REFUSED: {case_file} is not the case the treatment record names")
    return cases, a, b, case_digest


def proximity(cases_path, case_file, control_run, treatment_run, year, out_path):
    """For every step every figure shows, the reference position and its distance to the
    nearest trough axis, with a per-case summary over the positioned steps and, for a
    prefix case, over the prefix steps alone."""
    from scipy.io import loadmat
    import season_metrics as S
    cases, a, b, case_digest = _bound_inputs(cases_path, case_file, control_run, treatment_run, year)
    coarse_threshold = float(b["record"]["dataset_specific"]["coarse_threshold"])
    case = loadmat(case_file, variable_names=["time", "lat_c", "lon_c", "u_c", "currv_anom_c", "advcurrv_anom_c"])
    times = np.asarray(case["time"], float).ravel()
    lat_c, lon_c = np.asarray(case["lat_c"], float).ravel(), np.asarray(case["lon_c"], float).ravel()
    out_cases = []
    for n, c in enumerate(cases["cases"], start=1):
        tb_track = b["tracks"][c["b"]]
        ta_track = a["tracks"][c["a"]] if c["a"] is not None else None
        first_shared = float(ta_track["time"][0]) if (c["kind"] == "candidate recovered prefix" and ta_track is not None) else None
        rows = []
        for k in steps_for(c, tb_track, ta_track, times):
            source, position = None, None
            for tr, label in ((tb_track, "treatment"), (ta_track, "control")):
                if tr is None:
                    continue
                at = np.where(np.abs(tr["time"] - times[k]) < 1e-9)[0]
                if at.size:
                    source, position = label, (float(tr["lat"][at[0]]), float(tr["lon"][at[0]]))
                    break
            if position is None and times[k] < tb_track["time"][0]:
                source, position = "treatment first position, before the track starts", (float(tb_track["lat"][0]), float(tb_track["lon"][0]))
            _, axes_k = tracker_view(np.asarray(case["u_c"][k], float), np.asarray(case["currv_anom_c"][k], float),
                                     np.asarray(case["advcurrv_anom_c"][k], float), lat_c, lon_c, coarse_threshold)
            d = nearest_axis_deg(*position, axes_k) if position is not None else None
            in_prefix = bool(first_shared is not None and source == "treatment" and times[k] < first_shared)
            rows.append({"step": k, "date": f"{S.date_of(times[k]):%Y-%m-%dT%H:00Z}", "position_lat_lon": position, "position_source": source,
                         "trough_axes": len(axes_k), "nearest_axis_deg": None if d is None else round(d, 3),
                         "axis_within_tolerance": None if d is None else bool(d <= AXIS_TOLERANCE_DEG), "in_prefix": in_prefix})

        def tally(sel):
            flags = [bool(r["axis_within_tolerance"]) for r in sel if r["position_source"] in ("treatment", "control")]
            share, longest = _share_and_gap(flags)
            return {"positioned_steps": len(flags), "within_tolerance": sum(1 for f in flags if f), "longest_run_without": longest,
                    "share": None if share is None else round(share, 3)}
        summary = {"positioned": tally(rows)}
        if first_shared is not None:
            summary["prefix"] = tally([r for r in rows if r["in_prefix"]])
        shown = [r["position_lat_lon"] for r in rows if r["position_source"] == "treatment"]
        geometry = {"start_lat_lon": [round(float(tb_track["lat"][0]), 3), round(float(tb_track["lon"][0]), 3)],
                    "end_lat_lon": [round(float(tb_track["lat"][-1]), 3), round(float(tb_track["lon"][-1]), 3)],
                    "observations": int(tb_track["time"].size), "start_date": f"{S.date_of(float(tb_track['time'][0])):%Y-%m-%dT%H:00Z}",
                    "whole_track_dlat": round(float(tb_track["lat"][-1] - tb_track["lat"][0]), 3),
                    "whole_track_dlon": round(float(tb_track["lon"][-1] - tb_track["lon"][0]), 3),
                    "shown_steps_dlat": round(shown[-1][0] - shown[0][0], 3) if len(shown) > 1 else None,
                    "shown_steps_dlon": round(shown[-1][1] - shown[0][1], 3) if len(shown) > 1 else None}
        out_cases.append({"case": n, "kind": c["kind"], "treatment_track": c["b"], "control_track": c["a"], "summary": summary,
                          "treatment_geometry": geometry, "steps": rows})
    out = {"generated_by": "scripts/pilot_visual_audit_cases.py proximity", "script_sha256": X.digest(os.path.abspath(__file__)), "year": year,
           "cases_file": os.path.relpath(cases_path), "cases_file_sha256": X.digest(cases_path), "case_file_sha256": case_digest,
           "runs": {"control": {"record_sha256": a["record_sha256"], "tracks_sha256": a["tracks_sha256"]},
                    "treatment": {"record_sha256": b["record_sha256"], "tracks_sha256": b["tracks_sha256"]}},
           "tolerance_deg": AXIS_TOLERANCE_DEG, "tolerance_note": "one coarse grid cell, declared after the figures were viewed",
           "distance": "plain degrees of latitude and longitude, point to the nearest axis polyline segment",
           "cases": out_cases}
    X.publish_json(out_path, out, exclusive=True)
    return out


def _share_and_gap(flags):
    longest, run = 0, 0
    for f in flags:
        run = 0 if f else run + 1
        longest = max(longest, run)
    return (sum(flags) / len(flags) if flags else None), longest


def reference(case_file, run_dir, year, out_path):
    """Every track of a run, every stored position, against the axes of the run's own case
    at that step: the share of each track's positions within AXIS_TOLERANCE_DEG of an axis
    and its longest run of positions without, per track and as percentiles over all tracks
    and over the in-season tracks starting between 20 W and 40 E, with the proportion of
    tracks meeting both the share's 25th percentile and the run's 90th percentile together.
    A stored position whose time is not on the case's time axis is refused, never skipped."""
    from scipy.io import loadmat
    import pilot_track_crosswalk as P
    import season_metrics as S
    run = P.load_run(run_dir, year=year)
    case_digest = X.digest(case_file)
    if case_digest != run["record"]["dataset_specific"].get("case_sha256"):
        raise SystemExit(f"REFUSED: {case_file} is not the case the record in {run_dir} names")
    coarse_threshold = float(run["record"]["dataset_specific"]["coarse_threshold"])
    case = loadmat(case_file, variable_names=["time", "lat_c", "lon_c", "u_c", "currv_anom_c", "advcurrv_anom_c"])
    times = np.asarray(case["time"], float).ravel()
    lat_c, lon_c = np.asarray(case["lat_c"], float).ravel(), np.asarray(case["lon_c"], float).ravel()
    axes_at = {}
    per_track = []
    for t in run["tracks"]:
        flags = []
        for tt, la, lo in zip(t["time"], t["lat"], t["lon"]):
            k = int(np.argmin(np.abs(times - tt)))
            if abs(times[k] - tt) > 1e-6:
                # EVERY STORED POSITION IS MEASURED OR THE REFERENCE REFUSES: skipping one would
                # shrink the denominator and shorten the gaps without a trace
                raise SystemExit(f"REFUSED: a stored position at time {tt} is not on the case's time axis")
            if k not in axes_at:
                axes_at[k] = tracker_view(np.asarray(case["u_c"][k], float), np.asarray(case["currv_anom_c"][k], float),
                                          np.asarray(case["advcurrv_anom_c"][k], float), lat_c, lon_c, coarse_threshold)[1]
            d = nearest_axis_deg(float(la), float(lo), axes_at[k])
            flags.append(d is not None and d <= AXIS_TOLERANCE_DEG)
        share, gap = _share_and_gap(flags)
        month = S.date_of(float(t["time"][0])).month
        per_track.append({"track": len(per_track), "share": None if share is None else round(share, 4), "longest_run_without": gap,
                          "positions": len(flags), "stored_positions": int(t["time"].size),
                          "african_season": bool(6 <= month <= 9 and -20.0 <= float(t["lon"][0]) < 40.0)})

    def dist(rows):
        sh = [r["share"] for r in rows if r["share"] is not None]
        gp = [r["longest_run_without"] for r in rows if r["share"] is not None]
        q = lambda v, p: round(float(np.percentile(v, p)), 3) if v else None
        out = {"tracks": len(sh), "share_percentiles": {str(p): q(sh, p) for p in (10, 25, 50, 75, 90)},
               "longest_run_without_percentiles": {str(p): q(gp, p) for p in (10, 50, 90)},
               "tracks_with_every_position_within": sum(1 for s in sh if s == 1.0)}
        if sh:
            # THE JOINT PROPORTION, measured rather than read off two marginal percentiles
            cut_share, cut_gap = out["share_percentiles"]["25"], out["longest_run_without_percentiles"]["90"]
            both = sum(1 for s, g in zip(sh, gp) if s >= cut_share and g <= cut_gap)
            out["meeting_both_cutoffs"] = {"share_at_least": cut_share, "longest_run_at_most": cut_gap, "tracks": both,
                                           "fraction": round(both / len(sh), 4)}
        return out
    out = {"generated_by": "scripts/pilot_visual_audit_cases.py reference", "script_sha256": X.digest(os.path.abspath(__file__)), "year": year,
           "run": {"dir": run_dir, "record_sha256": run["record_sha256"], "tracks_sha256": run["tracks_sha256"]}, "case_file_sha256": case_digest,
           "tolerance_deg": AXIS_TOLERANCE_DEG, "all_tracks": dist(per_track),
           "african_season_tracks": dist([r for r in per_track if r["african_season"]]),
           "african_season": "first observation in June to September and between 20 W and 40 E",
           "positions_measured": sum(r["positions"] for r in per_track), "positions_stored": sum(r["stored_positions"] for r in per_track),
           "per_track": per_track}
    X.publish_json(out_path, out, exclusive=True)
    return out


def gridsat_overlay(ax, imagery, when, lat, lon):
    """The retained GridSat-B1 cold-cloud hatching for one step, read through the sequence
    tool's own readers so the timestamp validation and the partial-coverage report are the
    sequence's. Returns the step's coverage record, and draws nothing when nothing is read."""
    import gridsat_case_sequence as G
    path = G.retained_file(imagery, when)
    if path is None:
        return {"coverage": "not covered" if when.year >= 1980 else "no GridSat-B1 before 1980"}
    tb, lats, lons, file_time, spatial = G.read_box(path, lat, lon)
    if file_time != when:
        return {"coverage": "file timestamp does not match the requested timestep", "file": os.path.relpath(path)}
    finite = np.isfinite(tb)
    if tb.ndim == 2 and tb.shape[0] > 1 and tb.shape[1] > 1 and finite.any():
        cold = np.where(finite, tb < G.COLD_K, False).astype(float)
        if cold.any():
            ax.contourf(lons, lats, cold, levels=[0.5, 1.5], colors="none", hatches=["////"])
    return {"coverage": "retained imagery", "file": os.path.relpath(path), "file_sha256": X.digest(path), **spatial, **G.readings(tb)}


def render(cases_path, case_file, control_run, treatment_run, year, imagery, out_dir):
    """One figure per case, bound to the runs the cases were selected from (their records,
    manifests and tracks by digest) and to the treatment's retained case file by digest,
    with a record of what each panel shows. `out_dir` must not exist: it is created here
    and refused otherwise, so no retained page is ever overwritten."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from scipy.io import loadmat
    import gridsat_case_sequence as G
    import season_metrics as S
    cases, a, b, case_digest = _bound_inputs(cases_path, case_file, control_run, treatment_run, year)
    coarse_threshold = float(b["record"]["dataset_specific"]["coarse_threshold"])
    case = loadmat(case_file, variable_names=["time", "lat_c", "lon_c", "u_c", "v_c", "currv_anom_c", "advcurrv_anom_c"])
    times = np.asarray(case["time"], float).ravel()
    lat_c, lon_c = np.asarray(case["lat_c"], float).ravel(), np.asarray(case["lon_c"], float).ravel()
    # THE OUTPUT DIRECTORY IS CLAIMED FRESH BEFORE ANY PAGE IS WRITTEN: pages carry deterministic
    # names and the record is published last, so a repeat render into an existing directory
    # would overwrite retained figures and then refuse on the record, leaving stale digests
    try:
        os.mkdir(out_dir)
    except FileExistsError:
        raise SystemExit(f"REFUSED: {out_dir} exists and figure sets are never overwritten; render into a new directory")
    made = []
    for n, c in enumerate(cases["cases"], start=1):
        tb_track = b["tracks"][c["b"]]
        ta_track = a["tracks"][c["a"]] if c["a"] is not None else None
        steps = steps_for(c, tb_track, ta_track, times)
        shown = [t for t in (ta_track, tb_track) if t is not None]
        lon_lo, lon_hi = min(float(t["lon"].min()) for t in shown) - 12, max(float(t["lon"].max()) for t in shown) + 12
        lat_lo, lat_hi = min(float(t["lat"].min()) for t in shown) - 10, max(float(t["lat"].max()) for t in shown) + 10
        panel_h = max(2.6, 7.0 * (lat_hi - lat_lo) / (lon_hi - lon_lo) + 0.6)
        stem = os.path.join(out_dir, f"case_{n:02d}_{year}_b{c['b']}" + (f"_a{c['a']}" if c["a"] is not None else ""))
        pages, panels = [], []
        for page, first in enumerate(range(0, len(steps), PAGE_PANELS), start=1):
            chunk = steps[first:first + PAGE_PANELS]
            nrows = -(-len(chunk) // NCOLS)
            fig, axes = plt.subplots(nrows, NCOLS, figsize=(7.0 * NCOLS, panel_h * nrows), squeeze=False)
            for ax in axes.ravel()[len(chunk):]:
                ax.axis("off")
            for ax, k in zip(axes.ravel(), chunk):
                u, v = np.asarray(case["u_c"][k], float), np.asarray(case["v_c"][k], float)
                curvature, axes_k = tracker_view(u, np.asarray(case["currv_anom_c"][k], float), np.asarray(case["advcurrv_anom_c"][k], float),
                                                 lat_c, lon_c, coarse_threshold)
                ax.set_facecolor("0.82")                                                     # masked cells stay gray
                ax.pcolormesh(lon_c, lat_c, np.ma.masked_invalid(curvature), cmap="YlOrRd", vmin=0.0, vmax=ANOM_LIMIT, shading="auto")
                for lats_k, lons_k in axes_k:
                    ax.plot(lons_k, lats_k, "-", color="k", lw=1.0)
                ax.quiver(lon_c, lat_c, u, v, color="0.3", scale=350, width=0.0022)
                position = None
                for tr, color, label in ((ta_track, "tab:blue", "control"), (tb_track, "tab:red", "treatment")):
                    if tr is None:
                        continue
                    past = tr["time"] <= times[k] + 1e-9
                    ax.plot(tr["lon"][past], tr["lat"][past], "-", color=color, lw=1.8, label=label)
                    if past.any():
                        ax.plot(tr["lon"][past][-1], tr["lat"][past][-1], "o", color=color, ms=6)
                    at = np.where(np.abs(tr["time"] - times[k]) < 1e-9)[0]
                    if at.size:
                        position = (float(tr["lat"][at[0]]), float(tr["lon"][at[0]]))     # the treatment's position wins when both exist
                when = G.nearest_gridsat_time(times[k])
                sat = gridsat_overlay(ax, imagery, when, *position) if position is not None else {"coverage": "no track position at this step"}
                ax.set_xlim(lon_lo, lon_hi); ax.set_ylim(lat_lo, lat_hi)
                sat_word = sat["coverage"] if sat["coverage"] != "retained imagery" else f"retained imagery, box {sat['spatial_coverage']}"
                ax.set_title(f"case {n}, step {k}, {S.date_of(times[k]):%Y-%m-%d %HZ}, {len(axes_k)} axes, GridSat {sat_word}", fontsize=8)
                ax.axvline(EAST_START_DEG, color="gray", lw=0.8, ls="--")
                panels.append({"step": k, "time_days": float(times[k]), "date": f"{S.date_of(times[k]):%Y-%m-%dT%H:00Z}",
                               "position_lat_lon": position, "trough_axes": len(axes_k), "gridsat": sat})
            handles, labels = axes.ravel()[0].get_legend_handles_labels()
            handles.append(Line2D([], [], color="k", lw=1.0, label="trough axes (detector)")); labels.append(handles[-1].get_label())
            handles.append(Patch(facecolor="0.82", label="masked: westerly > 2.5 m/s or below the coarse threshold")); labels.append(handles[-1].get_label())
            handles.append(Patch(facecolor="none", hatch="////", label=f"GridSat-B1 Tb < {G.COLD_K:.0f} K")); labels.append(handles[-1].get_label())
            axes.ravel()[0].legend(handles, labels, loc="upper right", fontsize=7)
            fig.suptitle(f"{year} case {n}, page {page}: {c['kind']} (treatment track {c['b']}" + (f", control track {c['a']})" if c["a"] is not None else ")"), fontsize=10)
            fig.tight_layout(rect=(0, 0, 1, 0.985))
            path = f"{stem}_p{page}.png"
            fig.savefig(path, dpi=100); plt.close(fig)
            pages.append({"figure": os.path.relpath(path), "figure_sha256": X.digest(path), "steps": chunk})
        covered = sum(1 for q in panels if q["gridsat"]["coverage"] == "retained imagery")
        usable = sum(1 for q in panels if q["gridsat"]["coverage"] == "retained imagery" and q["gridsat"].get("cells", 0) > 0)
        made.append({"case": n, "kind": c["kind"], "pages": pages, "treatment_track": c["b"], "control_track": c["a"],
                     "steps": steps, "gridsat_covered_steps": covered, "gridsat_usable_steps": usable, "panels": panels})
    out = {"generated_by": "scripts/pilot_visual_audit_cases.py", "script_sha256": X.digest(os.path.abspath(__file__)), "year": year,
           "cases_file": os.path.relpath(cases_path), "cases_file_sha256": X.digest(cases_path),
           "case_file": os.path.relpath(case_file), "case_file_sha256": case_digest,
           "runs": {"control": {"dir": control_run, "record_sha256": a["record_sha256"], "manifest_sha256": a["manifest_sha256"], "tracks_sha256": a["tracks_sha256"]},
                    "treatment": {"dir": treatment_run, "record_sha256": b["record_sha256"], "manifest_sha256": b["manifest_sha256"], "tracks_sha256": b["tracks_sha256"]}},
           "imagery": imagery, "cold_threshold_k": G.COLD_K, "box_half_width_deg": G.HALF_WIDTH_DEG,
           "field": {"what": "the detector's coarse-stage view: currv_anom_c through the port's nine-point smoother, cyclonic positive in both "
                             "hemispheres, masked where the smoothed zonal wind exceeds 2.5 m/s or the anomaly is below the treatment record's "
                             "coarse threshold; trough axes are the zero contour of the smoothed advection anomaly inside the mask, by the port's trough_axes",
                     "coarse_threshold": coarse_threshold, "shading_limit": ANOM_LIMIT, "masked_cells_shown": "gray"},
           "satellite_is": "infrared brightness temperature in the sequence tool's box around the track position, context only, never a requirement; "
                           "gridsat_usable_steps counts covered steps whose box holds at least one pixel",
           "readings": "to be entered by the human reader: supported, contradicted or undetermined, per case", "figures": made}
    X.publish_json(os.path.join(out_dir, f"audit_figures_{year}.json"), out, exclusive=True)
    return made


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("select"); s.add_argument("--report", required=True); s.add_argument("--out", required=True)
    s.add_argument("--prefix-rule", choices=("strict", "eastern", "loose"), default="strict")
    s.add_argument("--season-only", action="store_true", help="the supplementary June to September selection, declared in the cases file")
    r = sub.add_parser("render")
    for name in ("--cases", "--case-file", "--control-run", "--treatment-run", "--out-dir"):
        r.add_argument(name, required=True)
    r.add_argument("--year", type=int, required=True); r.add_argument("--imagery", default="data/gridsat_jas")
    f = sub.add_parser("reference")
    for name in ("--case-file", "--run", "--out"):
        f.add_argument(name, required=True)
    f.add_argument("--year", type=int, required=True)
    q = sub.add_parser("proximity")
    for name in ("--cases", "--case-file", "--control-run", "--treatment-run", "--out"):
        q.add_argument(name, required=True)
    q.add_argument("--year", type=int, required=True)
    args = ap.parse_args(argv)
    if args.mode == "reference":
        try:
            out = reference(args.case_file, args.run, args.year, args.out)
        except FileExistsError:
            raise SystemExit(f"REFUSED: {args.out} exists and artifacts are never overwritten")
        print(f"{args.year}: all {out['all_tracks']}; African season {out['african_season_tracks']}")
        return 0
    if args.mode == "proximity":
        try:
            out = proximity(args.cases, args.case_file, args.control_run, args.treatment_run, args.year, args.out)
        except FileExistsError:
            raise SystemExit(f"REFUSED: {args.out} exists and artifacts are never overwritten")
        for c in out["cases"]:
            s = c["summary"]
            print(f"case {c['case']}: positioned {s['positioned']}" + (f", prefix {s['prefix']}" if "prefix" in s else ""))
        return 0
    if args.mode == "select":
        out = select(json.load(open(args.report)), args.prefix_rule, args.season_only)
        try:
            X.publish_json(args.out, out, exclusive=True)
        except FileExistsError:
            raise SystemExit(f"REFUSED: {args.out} exists and artifacts are never overwritten")
        print(f"{len(out['cases'])} cases selected of {out['candidates_available']}; wrote {args.out}")
        return 0
    made = render(args.cases, args.case_file, args.control_run, args.treatment_run, args.year, args.imagery, args.out_dir)
    for m in made:
        print(f"case {m['case']}: {m['pages'][0]['figure']} and {len(m['pages']) - 1} more page(s) ({len(m['steps'])} steps, "
              f"GridSat on {m['gridsat_covered_steps']}, usable {m['gridsat_usable_steps']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
