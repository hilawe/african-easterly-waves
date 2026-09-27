#!/usr/bin/env python3
"""One track's reading for the continuity diagnostic (the brief of 2026-09-27 as amended
before the run), from retained evidence only. For a named track of this record in a
season it records:

- the track's own span, band, outcome, group and duplicate-group members from the
  season's coast-crossing artifact, which must name the tracks file by digest;
- CANDIDATE CORRESPONDENCES: every QTrack system with an observation within 500 km at an
  equal timestamp inside the WINDOW, the case's observations with elapsed time 0 to 48
  hours inclusive since its first observation, kept with identifiers, per-step
  distances, the first elapsed hour within 500 km and the number of candidates per
  step, so ambiguity is preserved, and what a candidate does after the window is
  recorded apart and marked as beyond it;
- successors and predecessors in this record (a first observation within 0 to 12 hours
  after the case's last one and within 500 km, or a last observation within 0 to 12
  hours before its first and within 500 km), and QTrack systems starting within 0 to 12
  hours and 500 km of the case's end;
- for tracks and systems outside the cohort, whose outcome the artifact does not hold,
  a reading of their positions under the same region rule, marked as outside the
  cohort;
- a fixed two-dimensional VIEW of the tracker case's own fields (the curvature-vorticity
  anomaly shaded, the 700 hPa wind as arrows) at the case's last time and 24 and 48 hours
  later, 20 degrees west to 10 degrees east of the last position and 10 degrees either
  side, with the case, its successors and its candidates overlaid, recorded as a figure
  and as the anomaly at the last position and the window's maximum. A missing signature
  in any view establishes nothing about physical disappearance.

Decision history (which candidate the tracker took or dropped, what was absorbed) is
not retained anywhere identified and is recorded as unavailable.

    .venv/bin/python3 scripts/continuity_case_reading.py --year 1990 --track 816 --role case \\
        --campaign-evidence <dir> --tracker-cases <dir> --qtrack-dir <dir> --regions-dir <dir> \\
        --measurement-artifacts <dir> --out <json> --figure <png>
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

import exact_tracks as X  # noqa: E402

RADIUS_KM = 500.0
WINDOW_HOURS = 48.0                  # inclusive
ADJACENT_HOURS = 12.0                # inclusive, for successors, predecessors and QTrack starts near the end
VIEW_WEST, VIEW_EAST, VIEW_LAT = 20.0, 10.0, 10.0
VIEW_HOURS = (0.0, 24.0, 48.0)
ARTIFACT = "coast_crossing_{year}_2026-09-27.json"


def _sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def km(lat1, lon1, lat2, lon2):
    from aew.v1port.geometry import great_circle_distance
    return float(great_circle_distance(lat1, lon1, lat2, lon2, "km"))


def hours(a, b):
    return round((float(a) - float(b)) * 24.0, 6)


def read_outside_cohort(track, regions, endpoint):
    """The outcome a track outside the cohort would have under the same event rule, read
    from its positions and marked as such."""
    import coast_crossing_measurement as C
    c = C.classify(track, regions, endpoint)
    return {"outcome": c["outcome"], "follow_up_incomplete": c["follow_up_incomplete"], "read_from_positions_outside_cohort": True}


def outcome_of(tid, side_records, track, regions, endpoint):
    if tid in side_records:
        r = side_records[tid]
        return {"outcome": r["outcome"], "follow_up_incomplete": r["follow_up_incomplete"], "read_from_positions_outside_cohort": False}
    return read_outside_cohort(track, regions, endpoint)


def candidates(subject, q_tracks, q_records, tails, regions, endpoint):
    """Candidate correspondences in the window, with ambiguity preserved."""
    t0 = float(subject["time"][0])
    window = [k for k in range(subject["time"].size) if 0.0 <= hours(subject["time"][k], t0) <= WINDOW_HOURS]
    per_step, all_dists = [], {}
    for k in window:
        tk = float(subject["time"][k])
        dists = []
        for q in q_tracks:
            j = np.where(np.abs(q["time"] - tk) < 1e-9)[0]
            if j.size:
                j = int(j[0])
                dists.append((km(subject["lat"][k], subject["lon"][k], q["lat"][j], q["lon"][j]), q["id"], j))
        dists.sort()
        within = [(d, sid) for d, sid, _ in dists if d <= RADIUS_KM]
        per_step.append({"elapsed_hours": hours(tk, t0), "nearest_system": dists[0][1] if dists else None,
                         "nearest_km": round(dists[0][0], 1) if dists else None, "systems_within_500_km": [sid for _, sid in within],
                         "n_candidates": len(within)})
        for d, sid, _ in dists:
            all_dists.setdefault(sid, {})[str(hours(tk, t0))] = d                     # unrounded: every threshold decision uses this
    # a candidate is any system within 500 km at some window step; its distance is kept at EVERY
    # window step it shares, inside or outside 500 km, so an excursion is visible
    systems = {}
    for sid, by_hour in all_dists.items():
        inside = [float(h) for h, d in by_hour.items() if d <= RADIUS_KM]
        if inside:
            systems[sid] = {"system": sid, "distances_km_by_elapsed_hour": {h: round(d, 1) for h, d in by_hour.items()},
                            "first_elapsed_hour_within_500_km": min(inside),
                            "window_steps_shared": len(by_hour), "window_steps_beyond_500_km": sum(1 for d in by_hour.values() if d > RADIUS_KM),
                            "window_steps_within_500_km": sum(1 for d in by_hour.values() if d <= RADIUS_KM),
                            "min_km_in_window": round(min(by_hour.values()), 1)}
    by_id = {q["id"]: q for q in q_tracks}
    out = []
    for sid, rec in systems.items():
        q = by_id[sid]
        # every shared timestamp, in and beyond the window, recorded apart
        shared, ia, ib = np.intersect1d(subject["time"], q["time"], return_indices=True)
        beyond = [(hours(shared[m], t0), round(km(subject["lat"][ia[m]], subject["lon"][ia[m]], q["lat"][ib[m]], q["lon"][ib[m]]), 1))
                  for m in range(shared.size) if hours(shared[m], t0) > WINDOW_HOURS]
        steps = [round(km(q["lat"][i], q["lon"][i], q["lat"][i + 1], q["lon"][i + 1]), 1) for i in range(q["time"].size - 1)]
        jump = (max(steps), int(np.argmax(steps))) if steps else (None, None)
        import season_metrics as S
        rec.update({"system_start": str(S.date_of(q["time"][0])), "system_end": str(S.date_of(q["time"][-1])), "system_n": int(q["time"].size),
                    "system_start_relative_to_case_hours": hours(q["time"][0], t0),
                    "shared_tail_member": bool(sid in tails),
                    "largest_single_step_km": jump[0], "largest_single_step_at": str(S.date_of(q["time"][jump[1]])) if jump[1] is not None else None,
                    "beyond_window": {"shared_steps": len(beyond), "within_500_km": sum(1 for _, d in beyond if d <= RADIUS_KM),
                                      "last_shared_elapsed_hours": beyond[-1][0] if beyond else None,
                                      "distances_km_by_elapsed_hour": {str(h): d for h, d in beyond}},
                    **outcome_of(sid, q_records, q, regions, endpoint)})
        out.append(rec)
    return {"window_hours_inclusive": WINDOW_HOURS, "window_observations": len(window), "per_step": per_step,
            "candidate_systems": sorted(out, key=lambda r: r["min_km_in_window"]), "n_candidate_systems": len(out)}


def adjacent(subject, tracks, records, regions, endpoint, which):
    """Successors (start within 0 to 12 hours after the subject's end and 500 km of its
    last position) or predecessors (end within 0 to 12 hours before its start and 500
    km of its first position), with their outcomes."""
    out = []
    for t in tracks:
        if t["id"] == subject["id"] or not t["time"].size:
            continue
        if which == "successor":
            dh = hours(t["time"][0], subject["time"][-1])
            d = km(subject["lat"][-1], subject["lon"][-1], t["lat"][0], t["lon"][0])
        else:
            dh = hours(subject["time"][0], t["time"][-1])
            d = km(subject["lat"][0], subject["lon"][0], t["lat"][-1], t["lon"][-1])
        if 0.0 <= dh <= ADJACENT_HOURS and d <= RADIUS_KM:
            import season_metrics as S
            out.append({"id": t["id"], "hours_apart": dh, "km_apart": round(d, 1), "n": int(t["time"].size),
                        "first": str(S.date_of(t["time"][0])), "last": str(S.date_of(t["time"][-1])),
                        "first_lon": float(t["lon"][0]), "first_lat": float(t["lat"][0]), "last_lon": float(t["lon"][-1]), "last_lat": float(t["lat"][-1]),
                        "positions": [{"time": str(S.date_of(tt)), "lon": float(lo), "lat": float(la)} for tt, lo, la in zip(t["time"], t["lon"], t["lat"])],
                        **outcome_of(t["id"], records, t, regions, endpoint)})
    return sorted(out, key=lambda r: (r["hours_apart"], r["km_apart"]))


def field_view(case_path, subject, successors, cand_systems, q_tracks, figure):
    """The fixed two-dimensional view from the tracker case's own fields, drawn and
    summarized. Returns the summary; the figure is written to `figure`."""
    from scipy.io import loadmat
    import season_metrics as S
    m = loadmat(case_path, variable_names=["latgrid", "longrid", "time", "currv_anom", "u", "v", "rean", "level"])
    lat = m["latgrid"][:, 0]
    lon = m["longrid"][0, :]
    t = m["time"].ravel()
    t_end = float(subject["time"][-1])
    lon_e, lat_e = float(subject["lon"][-1]), float(subject["lat"][-1])
    requested = {"lon": [lon_e - VIEW_WEST, lon_e + VIEW_EAST], "lat": [lat_e - VIEW_LAT, lat_e + VIEW_LAT]}
    lo_lon, hi_lon = max(lon.min(), lon_e - VIEW_WEST), min(lon.max(), lon_e + VIEW_EAST)
    lo_lat, hi_lat = max(lat.min(), lat_e - VIEW_LAT), min(lat.max(), lat_e + VIEW_LAT)
    clipped = {"west": bool(lo_lon > requested["lon"][0]), "east": bool(hi_lon < requested["lon"][1]),
               "south": bool(lo_lat > requested["lat"][0]), "north": bool(hi_lat < requested["lat"][1])}
    ii = np.where((lat >= lo_lat) & (lat <= hi_lat))[0]
    jj = np.where((lon >= lo_lon) & (lon <= hi_lon))[0]
    if ii.size < 3 or jj.size < 3:
        raise SystemExit("REFUSED: the view window falls outside the case grid")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from aew.plotting import panel_label
    fig, axes = plt.subplots(1, len(VIEW_HOURS), figsize=(5.2 * len(VIEW_HOURS), 4.2), squeeze=False)
    summary = []
    by_id = {q["id"]: q for q in q_tracks}
    for ax, h in zip(axes[0], VIEW_HOURS):
        tt = t_end + h / 24.0
        k = np.where(np.abs(t - tt) < 1e-9)[0]
        if not k.size:
            summary.append({"hours_after_end": h, "time": None, "available": False})
            ax.text(0.5, 0.5, "beyond the retained year", transform=ax.transAxes, ha="center")
            continue
        k = int(k[0])
        A = m["currv_anom"][k][np.ix_(ii, jj)]
        U, V = m["u"][k][np.ix_(ii, jj)], m["v"][k][np.ix_(ii, jj)]
        vmax = float(np.nanmax(np.abs(A))) or 1.0
        ax.pcolormesh(lon[jj], lat[ii], A, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="nearest")
        step = max(1, ii.size // 10)
        ax.quiver(lon[jj][::step], lat[ii][::step], U[::step, ::step], V[::step, ::step], scale=300, width=0.003, color="k")
        ax.plot(subject["lon"], subject["lat"], color="#1f5fa8", lw=1.2, zorder=4)
        ax.scatter([lon_e], [lat_e], marker="x", color="#1f5fa8", s=50, zorder=5)
        pos = subject["lon"][np.abs(subject["time"] - tt) < 1e-9]
        if pos.size:
            ax.scatter(pos, subject["lat"][np.abs(subject["time"] - tt) < 1e-9], color="#1f5fa8", s=30, zorder=5)
        for s in successors:
            ax.plot([p["lon"] for p in s["positions"]], [p["lat"] for p in s["positions"]], color="#1f5fa8", lw=0.8, ls="--", zorder=4)
            at = [p for p in s["positions"] if p["time"] == str(S.date_of(tt))]
            if at:
                ax.scatter([at[0]["lon"]], [at[0]["lat"]], marker="D", color="#1f5fa8", s=26, zorder=5)
        for c in cand_systems:
            q = by_id[c["system"]]
            ax.plot(q["lon"], q["lat"], color="#c8552d", lw=0.8, zorder=3)
            sel = np.abs(q["time"] - tt) < 1e-9
            if sel.any():
                ax.scatter(q["lon"][sel], q["lat"][sel], marker="s", color="#c8552d", s=30, zorder=5)
        ax.set_xlim(lo_lon, hi_lon)
        ax.set_ylim(lo_lat, hi_lat)
        ax.set_title(f"{S.date_of(tt)} ({h:g} h after the end)", fontsize=8)
        panel_label(ax, "abc"[list(VIEW_HOURS).index(h)], size=10)
        a_end = float(m["currv_anom"][k][int(np.argmin(np.abs(lat - lat_e))), int(np.argmin(np.abs(lon - lon_e)))])
        imax = np.unravel_index(int(np.nanargmax(A)), A.shape)
        summary.append({"hours_after_end": h, "time": str(S.date_of(tt)), "available": True, "window_clipped": any(clipped.values()),
                        "anomaly_at_last_position": a_end,
                        "window_max_anomaly": float(A[imax]), "window_max_at": {"lon": float(lon[jj][imax[1]]), "lat": float(lat[ii][imax[0]])}})
    fig.suptitle(f"Track {subject['id']}: the tracker's curvature-vorticity anomaly (shaded) and 700 hPa wind, this record blue, candidate QTrack systems red", fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(figure, dpi=140)
    plt.close(fig)
    return {"field": "curvature-vorticity anomaly and 700 hPa wind from the tracker case", "reanalysis": str(m["rean"][0]), "level_hPa": float(m["level"].ravel()[0]),
            "window_requested": requested, "window": {"lon": [float(lo_lon), float(hi_lon)], "lat": [float(lo_lat), float(hi_lat)]},
            "window_clipped_by_grid": clipped, "times": summary,
            "caution": "a missing signature in any one view establishes nothing about physical disappearance"}


def run(year, track_id, role, campaign_evidence, tracker_cases, qtrack_dir, regions_dir, measurement_artifacts, out, figure):
    import qtrack_pairing_pilot as M
    import season_metrics as S
    import coast_crossing_measurement as C
    from aew import qtrack as Q
    art_path = os.path.join(measurement_artifacts, ARTIFACT.format(year=year))
    art = json.load(open(art_path))
    ours_path = os.path.join(campaign_evidence, f"era5_{year}", "tracker_port.mat")
    if art["inputs"]["this_record"]["path"] != ours_path and not os.path.samefile(art["inputs"]["this_record"]["path"], ours_path):
        raise SystemExit("REFUSED: the season artifact names a different tracks file")
    if _sha256(ours_path) != art["inputs"]["this_record"]["sha256"]:
        raise SystemExit("REFUSED: the tracks file is not the one the season artifact names")
    q_path = M.strict_path(qtrack_dir, M.QTRACK_FILE, year)
    if _sha256(q_path) != art["inputs"]["qtrack"]["sha256"]:
        raise SystemExit("REFUSED: QTrack's file is not the one the season artifact names")
    ours, case_id = M.load_ours(ours_path)
    q_tracks, q_meta = M.load_qtrack(q_path)
    recorded = art["inputs"]["region_polygons"]["sha256"]
    # digests compared BEFORE the polygons are parsed, so a changed file is refused rather than failing in the reader
    if {name: _sha256(os.path.join(regions_dir, f"{name}.mat")) for name in recorded} != recorded:
        raise SystemExit("REFUSED: the region polygons are not the ones the season artifact names")
    regions, digests = S.load_regions(regions_dir)
    if digests != recorded:
        raise SystemExit("REFUSED: the region polygons are not the ones the season artifact names")
    endpoint = C.endpoint_day(year)
    side_a = {r["id"]: r for r in art["sides"]["this_record"]["tracks"]}
    side_b = {r["id"]: r for r in art["sides"]["qtrack"]["tracks"]}
    if track_id not in side_a:
        raise SystemExit(f"REFUSED: track {track_id} is not in the {year} cohort")
    subject = next(t for t in ours if t["id"] == track_id)
    rec = side_a[track_id]
    groups = {g["group"]: g for g in art["sides"]["this_record"]["grouping_sensitivity"]["group_units"]}
    tails, _ = M.shared_tail_members_qtrack(q_path)
    cands = candidates(subject, q_tracks, side_b, set(tails), regions, endpoint)
    succ = adjacent(subject, ours, side_a, regions, endpoint, "successor")
    pred = adjacent(subject, ours, side_a, regions, endpoint, "predecessor")
    q_near_end = []
    for q in q_tracks:
        if not q["time"].size:
            continue
        dh = hours(q["time"][0], subject["time"][-1])
        d = km(subject["lat"][-1], subject["lon"][-1], q["lat"][0], q["lon"][0])
        if 0.0 <= dh <= ADJACENT_HOURS and d <= RADIUS_KM:
            q_near_end.append({"system": q["id"], "hours_after_end": dh, "km_from_end": round(d, 1), "n": int(q["time"].size), **outcome_of(q["id"], side_b, q, regions, endpoint)})
    by_id_ours = {t["id"]: t for t in ours}
    members = []
    if rec.get("group") is not None and rec["group"] in groups:
        for mid in groups[rec["group"]]["members"]:
            if mid == track_id:
                continue
            m = by_id_ours[mid]
            members.append({"id": mid, "n": int(m["time"].size), "first": str(S.date_of(m["time"][0])), "last": str(S.date_of(m["time"][-1])),
                            "first_lon": float(m["lon"][0]), "first_lat": float(m["lat"][0]), "last_lon": float(m["lon"][-1]), "last_lat": float(m["lat"][-1]),
                            **outcome_of(mid, side_a, m, regions, endpoint)})
    t_first, t_last = float(subject["time"][0]), float(subject["time"][-1])
    alive = [q for q in q_tracks if q["time"].size and q["time"][0] <= t_last and q["time"][-1] >= t_first]
    east = []
    for q in alive:
        sel = (q["time"] >= t_first) & (q["time"] <= t_last) & (q["lon"] > -17.0)
        if sel.any():
            dmin = min(km(subject["lat"][i], subject["lon"][i], q["lat"][j], q["lon"][j])
                       for i in range(subject["time"].size) for j in np.where(sel & (np.abs(q["time"] - subject["time"][i]) < 1e-9))[0]) if any(
                np.any(sel & (np.abs(q["time"] - subject["time"][i]) < 1e-9)) for i in range(subject["time"].size)) else None
            east.append({"system": q["id"], "min_km_at_equal_times": round(dmin, 1) if dmin is not None else None})
    inventory = {"systems_alive_during_the_track": len(alive), "with_any_position_east_of_17_W_during_it": east,
                 "rule": "alive: any observation between the track's first and last time; east: any such observation with longitude above 17 W"}
    case_path = os.path.join(tracker_cases, f"era5_{year}", "tracker_case.mat")
    if os.path.exists(figure):
        raise SystemExit(f"REFUSED: {figure} exists and figures beside artifacts are never overwritten")
    view = field_view(case_path, subject, succ, cands["candidate_systems"], q_tracks, figure)
    payload = {
        "generated_by": "scripts/continuity_case_reading.py", "script_sha256": X.digest(__file__), "year": year, "role": role,
        "what_this_is": "one track's reading for the continuity diagnostic from retained evidence; candidate correspondences are not identities; "
                        "decision history is unavailable",
        "rules": {"radius_km": RADIUS_KM, "window_hours_inclusive": WINDOW_HOURS, "adjacent_hours_inclusive": ADJACENT_HOURS,
                  "view": {"west_deg": VIEW_WEST, "east_deg": VIEW_EAST, "lat_deg": VIEW_LAT, "hours_after_end": list(VIEW_HOURS)}},
        "inputs": {"season_artifact": {"path": art_path, "sha256": _sha256(art_path)}, "this_record": {"path": ours_path, "sha256": _sha256(ours_path), "case_id": case_id},
                   "qtrack": {"path": q_path, "sha256": _sha256(q_path)}, "tracker_case": {"path": case_path, "sha256": _sha256(case_path)},
                   "region_polygons": {"dir": regions_dir, "sha256": digests}},
        "decision_history": "unavailable: the tracker case holds input fields, the port output's absorb=0 is a configuration flag, no retained source identified",
        "track": {"id": track_id, "n": int(subject["time"].size), "first": {"time": str(S.date_of(subject["time"][0])), "lon": float(subject["lon"][0]), "lat": float(subject["lat"][0])},
                  "last": {"time": str(S.date_of(subject["time"][-1])), "lon": float(subject["lon"][-1]), "lat": float(subject["lat"][-1])},
                  "band": rec["band"], "outcome": rec["outcome"], "follow_up_incomplete": rec["follow_up_incomplete"], "sectors_visited": rec["sectors_visited"],
                  "group": rec.get("group"), "group_members": groups[rec["group"]]["members"] if rec.get("group") is not None and rec["group"] in groups else None,
                  "group_member_spans": members,
                  "positions": [{"time": str(S.date_of(tt)), "lon": float(lo), "lat": float(la)} for tt, lo, la in zip(subject["time"], subject["lon"], subject["lat"])]},
        "candidate_correspondences": cands, "successors": succ, "predecessors": pred, "qtrack_systems_starting_near_end": q_near_end,
        "qtrack_inventory_during_the_track": inventory,
        "field_view": view, "figure": figure}
    try:
        X.publish_json(out, payload, exclusive=True)
    except FileExistsError:
        raise SystemExit(f"REFUSED: {out} exists and artifacts are never overwritten")
    print(f"{year} track {track_id} ({role}): {subject['time'].size} observations, {cands['n_candidate_systems']} candidate systems in the window, "
          f"{len(succ)} successors, {len(pred)} predecessors, {len(q_near_end)} QTrack starts near the end")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--track", type=int, required=True)
    ap.add_argument("--role", choices=("case", "control"), required=True)
    ap.add_argument("--campaign-evidence", required=True)
    ap.add_argument("--tracker-cases", required=True)
    ap.add_argument("--qtrack-dir", required=True)
    ap.add_argument("--regions-dir", required=True)
    ap.add_argument("--measurement-artifacts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--figure", required=True)
    a = ap.parse_args(argv)
    return run(a.year, a.track, a.role, a.campaign_evidence, a.tracker_cases, a.qtrack_dir, a.regions_dir, a.measurement_artifacts, a.out, a.figure)


if __name__ == "__main__":
    sys.exit(main())
