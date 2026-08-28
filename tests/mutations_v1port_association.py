"""Mutations for the version 1 track association port.

The list was written before the assertions existed and is reproduced in the docstring of
tests/test_v1port_association.py. Run with the repository's mutation checker against
src/aew/v1port/association.py.

Two mutations target geometry.points_in_polygon, which association calls. Run those with
--target src/aew/v1port/geometry.py; they are kept here because the behavior they bind is
version 1's `inpolygon` semantics, which this stage is what depends on.
"""


def _sub(old, new):
    return lambda s: s.replace(old, new)


MUTATIONS = {
    # A1 the exclusivity flag does nothing, so the repair silently fails to repair
    "exclusivity_flag_is_inert": _sub(
        "            pool = [i for i in available if i not in claimed] if exclusive "
        "else available",
        "            pool = available"),
    # A2 exclusivity is always on, so version 1 can no longer be reproduced
    "exclusivity_always_on": _sub(
        "            pool = [i for i in available if i not in claimed] if exclusive "
        "else available",
        "            pool = [i for i in available if i not in claimed]"),
    # A3 the polygon test is skipped, so anything matches
    "polygon_test_skipped": _sub(
        "        if not inside.any():\n            continue",
        "        if False:\n            continue"),
    # A4 the speed ceiling is not enforced
    "speed_ceiling_not_enforced": _sub(
        "        if (travelled_km * 1000.0) / (hours * 3600.0) < MAX_SPEED_MS:\n"
        "            hits.append(i)",
        "        hits.append(i)"),
    # A5 the first match wins rather than the nearest to the prediction
    "match_is_not_the_nearest": _sub(
        "    return hits[int(np.argmin(distances))]",
        "    return hits[0]"),
    # A6 an unmatched track keeps its prediction instead of standing down
    "unmatched_track_keeps_its_prediction": _sub(
        '                state["est_step_" + key] = NO_PREDICTION\n'
        '                state["est_lat_" + key] = NO_PREDICTION\n'
        '                state["est_lon_" + key] = NO_PREDICTION\n'
        "                continue",
        "                continue"),
    # A7 the twelve-hour gate's second condition is dropped, so a track stranded by a
    # wave-free timestep gets recovered where version 1 leaves it stranded
    "twelve_hour_gate_ignores_the_six_hour_prediction": _sub(
        '            if key == "12hr" and state.get("est_step_6hr") != NO_PREDICTION:',
        "            if False:"),
    # A7 the twelve-hour phase is removed altogether
    "twelve_hour_phase_removed": _sub(
        '    for key, hours in (("6hr", STEP_HOURS), ("12hr", 2 * STEP_HOURS)):',
        '    for key, hours in (("6hr", STEP_HOURS),):'),
    # A8 claimed candidates also seed new tracks, dropping the one exclusion version 1
    # does enforce
    "claimed_candidates_also_seed": _sub(
        "        if i in claimed:\n            continue",
        "        if False:\n            continue"),
    # A9 the predicted displacement ignores the wind
    "prediction_ignores_the_wind": _sub(
        "    dlat, dlon = meters_to_degrees(u_median * seconds, v_median * seconds, "
        "latitude)",
        "    dlat, dlon = meters_to_degrees(0.0, 0.0, latitude)"),
    # A10 the polygon is not displaced by the predicted motion
    "polygon_not_displaced": _sub(
        "            plons, plats = hull[0] + dlon, hull[1] + dlat",
        "            plons, plats = hull[0], hull[1]"),
    # A11 the minimum-area inflation never fires
    "inflation_never_fires": _sub(
        "    if area > minimum_area:\n        return lons, lats",
        "    return lons, lats"),
    # A12 the inflation is symmetric, losing the original's along-track stretch
    "inflation_is_symmetric": _sub(
        "        lons = (lons - lon_c) * np.sqrt((1.0 + 1.0 / 3.0) * scale) + lon_c\n"
        "        lats = (lats - lat_c) * np.sqrt((2.0 / 3.0) * scale) + lat_c",
        "        lons = (lons - lon_c) * np.sqrt(scale) + lon_c\n"
        "        lats = (lats - lat_c) * np.sqrt(scale) + lat_c"),
    # A13 the six-hour pass inflates every extended track, not only the last one it saw
    "six_hour_pass_inflates_every_track": _sub(
        "                                inflate=(key == \"12hr\"))",
        "                                inflate=True)"),
    # A13 the residual-track inflation is dropped entirely
    "six_hour_pass_inflates_nothing": _sub(
        '        if key == "12hr" and last_six_hour_state is not None:',
        "        if False:"),
    # A29 the inflation acts on the last DUE track rather than the last track in the list
    "inflation_acts_on_the_last_due_track": _sub(
        "    last_six_hour_state = states[-1] if states else None",
        "    last_six_hour_state = next(\n"
        "        (s for s in reversed(states) if s.get(\"est_step_6hr\") == step), None)"),
    # A14 the zero-area case is guarded rather than dividing by zero
    "zero_area_polygon_guarded": _sub(
        "    if area > minimum_area:\n        return lons, lats",
        "    if area <= 0 or area > minimum_area:\n        return lons, lats"),
    # A15 the convex hull is left open, moving the inflation center onto the centroid
    "hull_left_open": _sub(
        "        order = np.append(hull.vertices, hull.vertices[0])",
        "        order = hull.vertices"),
    # A16 a wave-free timestep clears predictions instead of being skipped whole
    "wave_free_timestep_not_skipped": _sub(
        "    if not waves:\n        return tracks, states",
        "    if False:\n        return tracks, states"),
    # A17 a candidate with no trough points seeds a track
    "empty_candidate_seeds_a_track": _sub(
        '        if np.asarray(waves[i]["lat_wave"]).size == 0:',
        "        if False:"),
    # A18 the prune tests span rather than observation count and recency
    "prune_tests_span_not_count_and_recency": _sub(
        '            if (now - track["time"][-1]) <= recent_days\n'
        '            or len(track["time"]) > lifetime_steps]',
        '            if (track["step"][-1] - track["step"][0]) >= lifetime_steps]'),
    # A18 the recency clause alone is dropped
    "prune_drops_the_recency_clause": _sub(
        '            if (now - track["time"][-1]) <= recent_days\n'
        '            or len(track["time"]) > lifetime_steps]',
        '            if len(track["time"]) > lifetime_steps]'),
    # A18 the count comparison loses its strictness
    "prune_count_is_not_strict": _sub(
        '            or len(track["time"]) > lifetime_steps]',
        '            or len(track["time"]) >= lifetime_steps]'),
    # A30 the prune runs on a wave-free timestep, which version 1's `continue` skips
    "prune_runs_on_a_wave_free_timestep": _sub(
        "    if not waves:\n        return tracks, states\n"
        "    if step + 1 < lifetime_steps:",
        "    if step + 1 < lifetime_steps:"),
    # A19 the prune runs from the first timestep
    "prune_runs_from_the_start": _sub(
        "    if step + 1 < lifetime_steps:",
        "    if False:"),
    # A20 the speed test uses end-to-end displacement rather than the median segment speed
    "speed_test_is_end_to_end": _sub(
        "        if not np.median(speeds) >= min_speed_ms:\n            continue",
        "        total = float(great_circle_distance(lat[0], lon[0], lat[-1], lon[-1],\n"
        '                                           "km"))\n'
        "        if not (total * 1000.0) / seconds.sum() >= min_speed_ms:\n"
        "            continue"),
    # A21 the segment durations assume a six-hour spacing
    "segment_durations_assume_six_hours": _sub(
        "        seconds = np.diff(times) * 24.0 * 3600.0",
        "        seconds = np.full(times.size - 1, STEP_HOURS * 3600.0)"),
    # A22 the five-point smoothing is dropped
    "smoothing_dropped": _sub(
        "        if smooth:\n"
        '            out["meanlat"] = list(_moving_average_5(lat))\n'
        '            out["meanlon"] = list(_moving_average_5(lon))',
        "        if False:\n"
        '            out["meanlat"] = list(_moving_average_5(lat))\n'
        '            out["meanlon"] = list(_moving_average_5(lon))'),
    # A22 the moving average does not shrink its window at the ends
    "moving_average_does_not_shrink_at_the_ends": _sub(
        "        half = min(2, i, n - 1 - i)",
        "        half = 2"),
    # A23 the final filter runs on a run shorter than the minimum lifetime
    "short_run_gate_removed": _sub(
        "    if total_steps is not None and total_steps < lifetime_steps:\n"
        "        return list(tracks)",
        "    if False:\n        return list(tracks)"),
    # A24 the trough mask is not carried onto the track
    "trough_mask_not_carried_on_seed": _sub(
        '        "wave_points": [wave.get("region")],',
        '        "wave_points": [],'),
    "trough_mask_not_carried_on_extend": _sub(
        '    track["wave_points"].append(wave.get("region"))',
        "    pass"),
    # A28 the mask is stored under the key the record writer reads for the basin, so a
    # track becomes unfilable. This is not hypothetical: the first draft did exactly this.
    "trough_mask_shadows_the_basin_key": _sub(
        '        "wave_points": [wave.get("region")],',
        '        "region_name": [wave.get("region")],'),
}

# Mutations for geometry.points_in_polygon, which this stage depends on for version 1's
# `inpolygon` semantics. Run with --target src/aew/v1port/geometry.py.
GEOMETRY_MUTATIONS = {
    # A26 an edge point is excluded, which is where MATLAB's inpolygon and the obvious
    # library substitutes disagree
    "edge_points_excluded": _sub(
        "        if np.any((np.abs(cross) <= tol) & within):\n"
        "            inside[i] = True\n"
        "            continue",
        "        pass"),
    # A25 and A27 the polygon is replaced by its bounding box, so a concave notch admits
    # points and the ray cast is not exercised at all
    "polygon_treated_as_its_bounding_box": _sub(
        "        inside[i] = bool(np.count_nonzero(straddles & (x < x_at_y)) % 2)",
        "        inside[i] = bool(poly_x.min() <= x <= poly_x.max() and\n"
        "                         poly_y.min() <= y <= poly_y.max())"),
    # A25 non-straddling edges are counted as crossings
    "non_straddling_edges_counted_as_crossings": _sub(
        "        inside[i] = bool(np.count_nonzero(straddles & (x < x_at_y)) % 2)",
        "        inside[i] = bool(np.count_nonzero(x < x_at_y) % 2)"),
    # a non-finite polygon silently admits points instead of containing nothing, which is
    # what makes version 1's divide-by-zero inflation strand a track
    "non_finite_polygon_not_refused": _sub(
        "    if poly_x.size < 3 or not (np.all(np.isfinite(poly_x)) and "
        "np.all(np.isfinite(poly_y))):",
        "    if poly_x.size < 3:"),
}
