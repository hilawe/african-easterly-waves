"""Track association, ported from the second half of find_ews_f.m.

Stage 6 of the port, the last of the tracker, and the one that matters most: THIS IS WHERE
VERSION 1'S DUPLICATION DEFECT LIVES.

HOW ASSOCIATION WORKS. Each live track carries a predicted position six and twelve hours
ahead, and a search polygon around each. At a timestep, a candidate matches a track when
any of its trough points falls inside that track's polygon and the implied propagation
speed is below the maximum. Of the matches, the one nearest the predicted center wins and
extends the track. Candidates that no track claimed seed new tracks. The prediction comes
from the median wind over the wave's own points, so a wave is advected by the flow it sits
in.

THE DEFECT, stated exactly, because the repair depends on the detail. The original
accumulates every claimed candidate into a variable `excl`:

    if exist('excl') == 0;   excl = id;   else   excl = [excl;id];   end

written once in the six-hour branch and once in the twelve-hour branch. But `excl` is read
in exactly one place, at the start of the new-track seeding block:

    if exist('excl') == 1;
      excl = unique(excl);
      id2 = setdiff(id,excl);

So exclusivity is enforced between EXTENDING a track and SEEDING a new one, and never
between one extension and another. Both matching loops iterate over every candidate with
no knowledge of what an earlier track already took, so two tracks at the same timestep can
claim the same trough, and from then on they carry byte-identical observations.

WHAT THAT ESTABLISHES, AND WHAT IT DOES NOT. The source establishes the mechanism: version 1
cannot prevent two tracks from taking one trough in a timestep. It does not establish that
this mechanism produced the duplication rate measured in the published record. Attributing
the measured rate to this mechanism needs a replay, which is what M3 is for. Until then the
claim here is the mechanism, not its share.

THE REPAIR IS ONE FLAG. `exclusive=False` reproduces version 1. `exclusive=True` removes a
candidate from consideration as soon as a track claims it, which is the single rule the
rebuild needs. Both live in one code path on purpose: reproducing version 1 faithfully is
what validates the port, and the corrected record has to come from the same code or the
comparison means nothing.

FOUR MORE VERSION 1 BEHAVIORS, all confirmed at source and all reproduced here. Each is
strange enough that a reader would take it for a porting error, so each is named where it
happens as well as here.

  1. A TIMESTEP WITH NO CANDIDATES IS SKIPPED ENTIRELY, not treated as a failed match.
     Every no-wave path in the detection half is a `continue` (lines 124-131, 153-162,
     165-172), so nothing is cleared and no track stands down. A track whose six-hour
     prediction fell on that timestep keeps a stale prediction forever, and the twelve-hour
     gate's second conjunct then excludes it from recovery. A wave-free timestep therefore
     strands every track waiting on it. See `associate_step`.

  2. THE SIX-HOUR PASS INFLATES ONLY THE LAST TRACK'S POLYGONS. The `for st` loop ends at
     line 252 and the inflation block sits at 254-265, outside it, referring to `poly(st)`
     with the residual loop value. Every other track extended in that pass keeps an
     un-inflated polygon. The twelve-hour pass (341-352) and the seeding block (422-433)
     inflate inside their loops, so this asymmetry belongs to the six-hour pass alone.

  3. A ZERO-AREA POLYGON IS INFLATED BY DIVIDING BY ZERO. The guard is
     `polyarea(...) <= threshold`, which a degenerate polygon satisfies, so `scale` is
     infinite and every vertex becomes non-finite. Such a polygon then contains nothing,
     which strands the track. Reachable from a single-point wave or a collinear fallback.

  4. THE HULL IS CLOSED. MATLAB's `convhull` repeats the first vertex at the end, and the
     inflation centers the polygon on the arithmetic mean of that vertex list, so the
     repeated vertex is counted twice and the center is not the centroid.
"""

import numpy as np

from .geometry import great_circle_distance, meters_to_degrees, points_in_polygon

# find_ews_f.m parameters
MAX_SPEED_MS = 25.0          # max_speed
MIN_SPEED_MS = 2.0           # min_speed
MIN_LIFETIME_DAYS = 2.0      # tlm_thr
POLY_AREA_6HR = 75.0         # poly_area_6hrthr
POLY_AREA_12HR = 100.0       # poly_area_12hrthr
STEP_HOURS = 6.0

# tlm_thr * 4, the number of six-hourly steps in the minimum lifetime. Version 1 uses this
# one number for three different things: the timestep from which pruning starts, the
# observation count a track must exceed to survive a prune, and the gate on the final
# speed filter.
LIFETIME_STEPS = int(MIN_LIFETIME_DAYS * 4)
RECENT_DAYS = 0.5            # the "seen within half a day" clause of the prune

NO_PREDICTION = -999


def _hull_polygon(lons, lats):
    """Convex hull of the wave points, CLOSED, or None when one cannot be formed.

    FAITHFUL. MATLAB's `convhull` returns a closed index sequence with the first vertex
    repeated at the end, and version 1 passes that list straight into the inflation, which
    centers on `mean(...)` of the vertices. The repeat therefore weights the first vertex
    twice and shifts the center off the centroid. Reproducing the closure is what makes the
    inflated polygon match. `scipy.spatial.ConvexHull.vertices` is open, so the first index
    is appended here.

THE STARTING VERTEX IS ROTATED TO THE LOWEST INDEX, and that is a repair rather than a
    detail. Closing the ring doubles whichever vertex comes first, so where the cycle starts
    changes the mean the inflation centers on. MATLAB and scipy start it in different
    places, which was recorded as an unresolved divergence until MATLAB was finally run on
    2026-08-30. On a real 200-point wave footprint:

        MATLAB  convhull(...,'simplify',true) -> [1 3 187 200 116 40 26 2 1]
        scipy   ConvexHull(...).vertices      -> [3 187 200 116 40 26 2 1]

    The same cycle in the same direction, rotated by one. Without the rotation below the
    port doubles a DIFFERENT vertex from version 1 on 72 percent of real wave footprints,
    moving the inflation center by a median of 0.50 degrees, past 0.5 for half of them and
    up to 4.0, on a grid whose spacing is 2 degrees. That is not a rounding-level concern.

    THE RULE IS INFERRED FROM TWO OBSERVATIONS and should be read that way. Both MATLAB
    answers began at the lowest-index hull vertex, and in both the lowest happened to be
    index 1, so "start at the lowest index" fits the evidence without being strongly tested
    by it. What would falsify it is a hull whose lowest index is not 1 starting somewhere
    else. `test_the_hull_starts_at_the_lowest_index` pins the behavior either way.

    The vertex SET half of the question is settled and the port already matched. MATLAB
    passes `'simplify',true`, which drops collinear points, and scipy drops them too;
    measured 2026-08-30 as 5 indices against plain convhull's 9 on a square with a midpoint
    on every edge. `test_the_hull_drops_collinear_points` covers it.

    The original wraps `convhull` in a try and falls back to a seven-point diamond built
    from the wave's extreme latitudes when it fails.
    """
    if lons.size < 3:
        return None
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(np.column_stack([lons, lats]))
        vertices = np.asarray(hull.vertices)
        # rotate so the cycle begins where MATLAB's does, preserving direction
        vertices = np.roll(vertices, -int(np.argmin(vertices)))
        order = np.append(vertices, vertices[0])
        return lons[order], lats[order]
    except Exception:                                    # noqa: BLE001
        return None


def _fallback_polygon(track, dlat, dlon):
    """The seven-point diamond the original builds when the hull fails.

    Its vertices run through the wave's minimum, mean and maximum latitude, each offset
    both ways by the predicted displacement, and close back on the start.
    """
    minlat, maxlat = track["minlat"][-1], track["maxlat"][-1]
    meanlat = track["meanlat"][-1]
    lon_min, lon_max = track["meanlon_minlat"][-1], track["meanlon_maxlat"][-1]
    meanlon = track["meanlon"][-1]
    lats = np.array([minlat - dlat, minlat + dlat, meanlat + dlat, maxlat + dlat,
                     maxlat - dlat, meanlat - dlat, minlat - dlat])
    lons = np.array([lon_min - dlon, lon_min + dlon, meanlon + dlon, lon_max + dlon,
                     lon_max - dlon, meanlon - dlon, lon_min - dlon])
    return lons, lats


def _polygon_area(lons, lats):
    """Shoelace area, matching MATLAB `polyarea`."""
    return 0.5 * np.abs(np.dot(lons, np.roll(lats, 1)) - np.dot(lats, np.roll(lons, 1)))


def _inflate(lons, lats, minimum_area):
    """Grow a polygon about the mean of its vertices until it reaches the minimum area.

    FAITHFUL in three ways that all matter.

    THE ASYMMETRY. The original scales longitude by sqrt((1 + 1/3) * scale) and latitude by
    sqrt((2/3) * scale), so the search region is stretched more along the direction of
    travel than across it. The two factors do not multiply to `scale`, so the inflated
    polygon does not exactly reach the target area.

    THE CENTER. `mean(...)` of the vertex list, which for a closed hull counts the first
    vertex twice and is therefore not the centroid.

    THE DEGENERATE CASE. The guard is `area <= minimum_area`, which a zero-area polygon
    satisfies, so version 1 divides by zero and the vertices become non-finite. That is
    reproduced rather than guarded, because a guard would silently keep a track alive that
    version 1 stranded. A non-finite polygon contains nothing, which is what
    `points_in_polygon` reports.
    """
    area = _polygon_area(lons, lats)
    if area > minimum_area:
        return lons, lats
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = minimum_area / area
        lon_c, lat_c = np.mean(lons), np.mean(lats)
        lons = (lons - lon_c) * np.sqrt((1.0 + 1.0 / 3.0) * scale) + lon_c
        lats = (lats - lat_c) * np.sqrt((2.0 / 3.0) * scale) + lat_c
    return lons, lats


def predict_displacement(u_median, v_median, latitude, hours):
    """Displacement in degrees from the median wind over the wave's points."""
    seconds = hours * 3600.0
    dlat, dlon = meters_to_degrees(u_median * seconds, v_median * seconds, latitude)
    return float(dlat), float(dlon)


def _new_track(wave, step):
    lat_w, lon_w = wave["lat_wave"], wave["lon_wave"]
    return {
        "time": [wave["time"]],
        "step": [step],
        "meanlat": [wave["lat_mean"]], "meanlon": [wave["lon_mean"]],
        "maxlat": [float(np.max(lat_w))],
        "minlat": [float(np.min(lat_w))],
        "meanlon_maxlat": [float(np.mean(lon_w[lat_w == np.max(lat_w)]))],
        "meanlon_minlat": [float(np.mean(lon_w[lat_w == np.min(lat_w)]))],
        "lat_wave": [lat_w], "lon_wave": [lon_w],
        "n_points": [int(lat_w.size)],
        # The trough mask, version 1's `wave_points`. Carried because
        # generate_ew_stats_f.m composites its satellite and radiation fields over exactly
        # these cells, so a port of that stage needs them and cannot recover them later.
        #
        # NAMED `wave_points`, NOT `region`, deliberately. The merge stage calls this mask
        # `region`, and version 1 uses `region` and `region_name` on a track for something
        # else entirely: the geographic basin the wave came from, which is what the record
        # writer files each track under. Two meanings on one key would have made a track
        # unfilable, and did until this was caught.
        "wave_points": [wave.get("region")],
    }


def _extend(track, wave, step):
    track["time"].append(wave["time"])
    track["step"].append(step)
    track["meanlat"].append(wave["lat_mean"])
    track["meanlon"].append(wave["lon_mean"])
    lat_w, lon_w = wave["lat_wave"], wave["lon_wave"]
    track["maxlat"].append(float(np.max(lat_w)))
    track["minlat"].append(float(np.min(lat_w)))
    track["meanlon_maxlat"].append(float(np.mean(lon_w[lat_w == np.max(lat_w)])))
    track["meanlon_minlat"].append(float(np.mean(lon_w[lat_w == np.min(lat_w)])))
    track["lat_wave"].append(lat_w)
    track["lon_wave"].append(lon_w)
    track["n_points"].append(int(lat_w.size))
    track["wave_points"].append(wave.get("region"))


def _refresh_prediction(state, track, wave, step, u_median, v_median, inflate=True):
    """Recompute both predicted positions and both search polygons.

    `inflate=False` builds the polygons and leaves them at their natural size, which is
    what the six-hour pass does for every track except the last one it touches. See the
    module docstring, behavior 2.
    """
    lat, lon = wave["lat_mean"], wave["lon_mean"]
    for hours, key, area in ((STEP_HOURS, "6hr", POLY_AREA_6HR),
                             (2 * STEP_HOURS, "12hr", POLY_AREA_12HR)):
        dlat, dlon = predict_displacement(u_median, v_median, lat, hours)
        state["est_lat_" + key] = lat + dlat
        state["est_lon_" + key] = lon + dlon
        state["est_step_" + key] = step + (1 if key == "6hr" else 2)
        hull = _hull_polygon(wave["lon_wave"], wave["lat_wave"])
        if hull is None:
            plons, plats = _fallback_polygon(track, dlat, dlon)
        else:
            plons, plats = hull[0] + dlon, hull[1] + dlat
        if inflate:
            plons, plats = _inflate(plons, plats, area)
        state["poly_lon_" + key] = plons
        state["poly_lat_" + key] = plats


def _inflate_state(state):
    """Apply both inflations to one track's stored polygons, as lines 254-265 do."""
    for key, area in (("6hr", POLY_AREA_6HR), ("12hr", POLY_AREA_12HR)):
        if "poly_lon_" + key not in state:
            continue
        state["poly_lon_" + key], state["poly_lat_" + key] = _inflate(
            state["poly_lon_" + key], state["poly_lat_" + key], area)


def _match(state, track, waves, available, key, hours):
    """Candidates inside the polygon whose implied speed is under the maximum."""
    hits = []
    for i in available:
        wave = waves[i]
        inside = points_in_polygon(state["poly_lon_" + key], state["poly_lat_" + key],
                                   wave["lon_wave"], wave["lat_wave"])
        if not inside.any():
            continue
        travelled_km = float(great_circle_distance(
            wave["lat_mean"], wave["lon_mean"],
            track["meanlat"][-1], track["meanlon"][-1], "km"))
        if (travelled_km * 1000.0) / (hours * 3600.0) < MAX_SPEED_MS:
            hits.append(i)
    if not hits:
        return None
    distances = [float(great_circle_distance(state["est_lat_" + key],
                                             state["est_lon_" + key],
                                             waves[i]["lat_mean"],
                                             waves[i]["lon_mean"], "nm")) for i in hits]
    return hits[int(np.argmin(distances))]


def associate_step(tracks, states, waves, step, u_median, v_median, exclusive=False):
    """One timestep of association, extending tracks and seeding new ones.

    `u_median` and `v_median` are callables taking a wave and returning the median wind
    over its points, which is what the original computes from the fine-grid winds.

    THEY MUST IGNORE MISSING VALUES. The original calls `nanmedian`, and that is not a
    defensive habit: the fields being averaged are the smoothed fine-grid winds, and the
    smoothing stage propagates NaN on purpose because `smth9_f`'s own missing-value guard is
    dead code. A plain median over those points returns NaN wherever a wave touches masked
    ground, and a NaN displacement puts the prediction and its polygon nowhere, silently
    ending the track.

    Set `exclusive=True` for the repair: a claimed candidate leaves the pool immediately,
    so no two tracks can take the same trough. The default reproduces version 1.
    """
    # FAITHFUL, and the single most consequential of the four behaviors named in the
    # module docstring. A timestep that produced no candidates is skipped whole, so no
    # prediction is cleared and no track stands down. That is not a shortcut: it is what
    # makes the twelve-hour gate's second conjunct load-bearing, because a track waiting on
    # a wave-free timestep keeps a stale six-hour prediction and is then barred from
    # twelve-hour recovery.
    if not waves:
        return tracks, states

    available = list(range(len(waves)))
    claimed = set()
    # The state the six-hour pass's inflation acts on. It is the LAST TRACK IN THE LIST,
    # not the last one that matched or was even due, because MATLAB's `for st = 1:ew_num`
    # leaves `st` at `ew_num` however many iterations hit its `continue`. A first draft read
    # this as the last DUE track, which is a different track whenever the newest track is
    # not due. Both the block and the loop are guarded by `ew_num > 0`.
    last_six_hour_state = states[-1] if states else None

    for key, hours in (("6hr", STEP_HOURS), ("12hr", 2 * STEP_HOURS)):
        if key == "12hr" and last_six_hour_state is not None:
            # Lines 254-265, which sit BETWEEN the two passes rather than after both. The
            # order matters whenever the residual track is one the twelve-hour pass then
            # extends: version 1 inflates the polygon that pass is about to replace, not
            # the one it produces.
            _inflate_state(last_six_hour_state)
        for track, state in zip(tracks, states):
            if state.get("est_step_" + key) != step:
                continue
            if key == "12hr" and state.get("est_step_6hr") != NO_PREDICTION:
                # FAITHFUL to `find(est_time_12hr(:,1) == t & est_time_6hr(:,1) == -999)`.
                # The second conjunct is load-bearing: a track that sat out a wave-free
                # timestep still holds a stale six-hour prediction, and this is what stops
                # it being recovered here.
                continue
            pool = [i for i in available if i not in claimed] if exclusive else available
            chosen = _match(state, track, waves, pool, key, hours)
            if chosen is None:
                state["est_step_" + key] = NO_PREDICTION
                state["est_lat_" + key] = NO_PREDICTION
                state["est_lon_" + key] = NO_PREDICTION
                continue
            wave = waves[chosen]
            _extend(track, wave, step)
            claimed.add(chosen)
            # The six-hour pass builds the polygons inside the loop and inflates outside
            # it; the twelve-hour pass does both inside.
            _refresh_prediction(state, track, wave, step,
                                u_median(wave), v_median(wave),
                                inflate=(key == "12hr"))
    # New tracks come from candidates nobody claimed. This exclusion exists in the
    # original and is the only place `excl` is read.
    for i in range(len(waves)):
        if i in claimed:
            continue
        if np.asarray(waves[i]["lat_wave"]).size == 0:
            # FAITHFUL to lines 374-377: a candidate with no trough points is skipped
            # rather than seeded. Without this the extremes below have no values to take.
            continue
        track = _new_track(waves[i], step)
        state = {}
        _refresh_prediction(state, track, waves[i], step,
                            u_median(waves[i]), v_median(waves[i]))
        tracks.append(track)
        states.append(state)
    return tracks, states


def prune_stale_tracks(tracks, states, step, now, waves,
                       lifetime_steps=LIFETIME_STEPS, recent_days=RECENT_DAYS):
    """The in-loop lifetime prune, from lines 441-466.

    NOT A FINAL FILTER, which is the point. Version 1 runs this INSIDE its timestep loop,
    from the eighth timestep onward, and drops tracks from the live set along with their
    predictions and polygons. A track survives if EITHER its last observation is no more
    than half a day old OR it has more than `lifetime_steps` observations. So the test is
    on observation COUNT and on RECENCY, not on the span between first and last
    observation, and a track pruned here never reaches the speed filter at all.

    `waves` IS REQUIRED, and it is the whole reason this signature is not simpler. Version
    1's no-wave paths are `continue` statements in the middle of the timestep loop, and
    that loop does not close until line 467, AFTER this prune at 441-466. So a `continue`
    skips the prune along with the association. Reproducing the skip in `associate_step`
    alone is not enough: a caller that pruned on every step would still diverge, because
    version 1 does not prune on a wave-free one, and at the end of a run that is the
    difference between a short stale track reaching the speed filter and never getting
    there. Passing the same wave list both functions saw is what keeps the pair honest,
    and it has no default for the same reason `days_since_1900` has no default epoch.

    `now` is the current timestep's time in the same units as the track times, which for
    this record is days. Returns the surviving tracks and their states.
    """
    if not waves:
        return tracks, states
    if step + 1 < lifetime_steps:
        # `ct_time` counts timesteps from one, so the prune starts at the eighth.
        return tracks, states
    kept = [(track, state) for track, state in zip(tracks, states)
            if (now - track["time"][-1]) <= recent_days
            or len(track["time"]) > lifetime_steps]
    if not kept:
        return [], []
    return [t for t, _ in kept], [s for _, s in kept]


def _moving_average_5(values):
    """MATLAB `smooth(x, 5)`, a five-point moving average with shrinking end windows.

    At the ends the window shrinks to the largest odd span that fits, so the first and last
    values are returned unchanged and the second and second-last are three-point means.
    """
    x = np.asarray(values, dtype=float)
    n = x.size
    out = np.empty(n, dtype=float)
    for i in range(n):
        half = min(2, i, n - 1 - i)
        out[i] = x[i - half:i + half + 1].mean()
    return out


def finalize_tracks(tracks, total_steps=None, min_speed_ms=MIN_SPEED_MS,
                    lifetime_steps=LIFETIME_STEPS, smooth=True):
    """The final speed filter and smoothing, from lines 472-490.

    THREE THINGS THIS IS NOT, each of which an earlier draft got wrong.

    It is NOT a lifetime filter. Lifetime pruning happens inside the timestep loop, in
    `prune_stale_tracks`. Nothing here looks at how long a track lived.

    The speed test is on the MEDIAN of the consecutive-segment speeds, not on end-to-end
    displacement, and it keeps a track whose median is at or above the minimum. A track
    that loops back on itself has fast segments and little net displacement, and version 1
    keeps it.

    The segment durations come from the track's own TIME values, in days, not from an
    assumed six-hour spacing. A track resumed by the twelve-hour branch has a twelve-hour
    segment in it, and treating that as six hours doubles its apparent speed.

    Surviving tracks then get a five-point moving average applied to their mean latitude
    and longitude. That is a real transformation of the output record, not a display
    convenience, so it happens here.

    `total_steps` is how many timesteps the run covered. Version 1 gates this whole filter
    on `ct_time >= tlm_thr*4`, so a run shorter than the minimum lifetime is returned
    unfiltered. Passing None skips the gate.
    """
    if total_steps is not None and total_steps < lifetime_steps:
        return list(tracks)
    kept = []
    for track in tracks:
        times = np.asarray(track["time"], dtype=float)
        if times.size < 2:
            continue
        lat = np.asarray(track["meanlat"], dtype=float)
        lon = np.asarray(track["meanlon"], dtype=float)
        distance_km = np.array([float(great_circle_distance(
            lat[i], lon[i], lat[i + 1], lon[i + 1], "km"))
            for i in range(times.size - 1)])
        seconds = np.diff(times) * 24.0 * 3600.0
        with np.errstate(divide="ignore", invalid="ignore"):
            speeds = (distance_km * 1000.0) / seconds
        if not np.median(speeds) >= min_speed_ms:
            continue
        out = dict(track)
        if smooth:
            out["meanlat"] = list(_moving_average_5(lat))
            out["meanlon"] = list(_moving_average_5(lon))
        kept.append(out)
    return kept


def duplicate_observations(tracks):
    """Count observations shared by more than one track, the defect's own signature.

    Two tracks that claimed the same trough carry identical (time, latitude, longitude)
    entries from that point. This is the measure the published record was audited with,
    and it is what `exclusive=True` should drive to zero. Run it before
    `finalize_tracks`, whose smoothing changes the coordinates.
    """
    seen, duplicates = {}, 0
    for number, track in enumerate(tracks):
        for t, lat, lon in zip(track["time"], track["meanlat"], track["meanlon"]):
            key = (t, round(float(lat), 9), round(float(lon), 9))
            if key in seen and seen[key] != number:
                duplicates += 1
            else:
                seen.setdefault(key, number)
    return duplicates
