"""Mutations for the version 1 contour merging port.

The list was written before the assertions existed and is reproduced in the docstring of
tests/test_v1port_contours.py. Run with the repository's mutation checker against
src/aew/v1port/contours.py.
"""


def _sub(old, new):
    return lambda s: s.replace(old, new)


MUTATIONS = {
    # K1 four-connected instead of eight
    "region_is_four_connected": _sub(
        '    labels, _ = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))',
        '    labels, _ = ndimage.label(mask)'),
    # K2 the whole mask returned rather than the seed's component
    "region_returns_the_whole_mask": _sub(
        "    return labels == labels[row, col]",
        "    return mask.copy()"),
    # K3 a seed on an empty cell still returns something
    # K3 IS SUPERSEDED BY K15. It asserted the opposite behaviour, that a seed on an
    # empty cell gives an empty region, and its anchor is gone.
    # K15 the below-threshold seed goes back to returning nothing, which is what the
    # port did before 2026-08-30 and is not what isolate_region_f.m does
    "below_threshold_seed_returns_nothing": _sub(
        "    if mask[row, col]:\n        return labels == labels[row, col]",
        "    if not mask[row, col]:\n        return np.zeros_like(mask)\n"
        "    if mask[row, col]:\n        return labels == labels[row, col]"),
    # K16 the seed cell is left out, so only the neighbouring components come back
    "seed_cell_not_marked": _sub(
        "    region[row, col] = True",
        "    region[row, col] = False"),
    # K17 only one adjacent component is collected rather than all of them
    "only_one_adjacent_component": _sub(
        "    for label in touching:",
        "    for label in sorted(touching)[:1]:"),
    # K18 the neighbour search is 4-connected, so a diagonally adjacent component is
    # missed even though the original tests all eight
    "adjacent_search_is_four_connected": _sub(
        "            if delta_row == 0 and delta_col == 0:\n                continue",
        "            if delta_row == 0 and delta_col == 0:\n                continue\n"
        "            if delta_row != 0 and delta_col != 0:\n                continue"),
    # K20 the out-of-bounds guard is dropped, so a seed off the grid no longer gives the
    # empty region the original's early return produces
    "out_of_bounds_seed_not_refused": _sub(
        "    if not (0 <= row < mask.shape[0] and 0 <= col < mask.shape[1]):",
        "    if False:"),
    # K21 the neighbour scan is skipped for a seed on a grid edge, so a boundary seed
    # comes back alone. An independent check built exactly this and it passed the first
    # version of these tests, every one of which seeded the interior.
    "edge_seed_skips_its_neighbours": _sub(
        "    touching = set()",
        "    if row in (0, mask.shape[0] - 1) or col in (0, mask.shape[1] - 1):\n"
        "        return region\n"
        "    touching = set()"),
    # K22 absorption happens by default, which is the behaviour version 1's broken
    # inpolygon call prevents and which the port performed until 2026-08-30
    "absorption_on_by_default": _sub(
        "def merge_contours(candidates, latgrid, longrid, curvature, threshold, absorb=False):",
        "def merge_contours(candidates, latgrid, longrid, curvature, threshold, absorb=True):"),
    # K23 the flag is accepted and ignored, so absorb=True changes nothing
    "flag_accepted_and_ignored": _sub(
        "        if absorb and others:",
        "        if False and others:"),
    # K24 with absorption off the loop still consumes another candidate, so waves vanish
    # for a different reason than absorption
    "candidates_consumed_without_absorbing": _sub(
        "        drop = set(absorbed) | {first}",
        "        drop = set(absorbed) | {first} | set(others[:1])"),
    # K25 the wave takes another candidate's time even when nothing was absorbed
    "time_taken_from_an_unabsorbed_candidate": _sub(
        '        time_source = candidates[absorbed[0]]["time"] if absorbed else here["time"]',
        '        time_source = candidates[others[0]]["time"] if others else here["time"]'),
    # K19, WITHDRAWN AND MEASURED RATHER THAN ARGUED. It asked whether the
    # above-threshold branch could differ from the below-threshold path applied to the
    # same seed. It cannot: with 8-connectivity any above-threshold cell adjacent to an
    # above-threshold seed is by definition already in the seed's own component, so
    # unioning the adjacent components adds nothing. Checked over 4,000 random masks
    # with a randomly chosen above-threshold seed, and exhaustively over every 3-by-3
    # mask and above-threshold seed, 2,304 cases: the paths agreed everywhere. The
    # branch is an optimisation, not a behavioural distinction. The reproducer is
    # tests/reproducers/k19_paths_agree.py, so the withdrawal can be rechecked.
    # K4 the ladder is never climbed
    "ladder_never_climbs": _sub(
        "        if (_extent(lats) >= MAX_LAT_EXTENT_DEG\n"
        "                or _extent(lons) >= MAX_LON_EXTENT_DEG):",
        "        if False:"),
    # K5 the ladder climbs unconditionally
    "ladder_always_climbs": _sub(
        "        if (_extent(lats) >= MAX_LAT_EXTENT_DEG\n"
        "                or _extent(lons) >= MAX_LON_EXTENT_DEG):",
        "        if True:"),
    # K6 the hull absorbs everything
    # K6 the hull absorbs everything. THE ANCHOR HAD ROTTED: it named
    # `polygon.contains_points`, which this module has never called, so the mutation
    # reported INERT and had been testing nothing for as long as it existed. Re-pointed
    # at the call the module actually makes.
    "hull_absorbs_everything": _sub(
        "        return points_in_polygon(lons[hull.vertices], lats[hull.vertices],\n"
        "                                 points_lon, points_lat)",
        "        return np.ones(len(points_lon), dtype=bool)"),
    # K7 a degenerate region still absorbs
    "degenerate_region_absorbs": _sub(
        "    if lons.size <= 2:\n        return np.zeros(len(points_lon), dtype=bool)",
        "    if lons.size <= 2:\n        return np.ones(len(points_lon), dtype=bool)"),
    # K8 near duplicates are kept apart
    "near_duplicates_not_merged": _sub(
        "            close = {others[i] for i in np.flatnonzero(distance_deg <= MERGE_DISTANCE_DEG)}",
        "            close = set()"),
    # K9 the merge distance compared in nautical miles rather than degrees
    "merge_distance_wrong_units": _sub(
        '                np.array([merged[i]["lon_mean"] for i in others]), "nm") / 60.0',
        '                np.array([merged[i]["lon_mean"] for i in others]), "nm")'),
    # K10 the minimum-extent pass keeps everything
    "minimum_extent_never_rejects": _sub(
        "        if span_deg >= MIN_EXTENT_DEG:",
        "        if True:"),
    # K11 the extent measured east to west
    "extent_measured_zonally": _sub(
        "        top = lats == np.max(lats)\n        bottom = lats == np.min(lats)",
        "        top = lons == np.max(lons)\n        bottom = lons == np.min(lons)"),
    # K12 the refined centre replaces rather than averages
    "refined_centre_replaces_the_median": _sub(
        '            wave["lat_mean"] = (float(np.mean(lats[peak])) + wave["lat_mean"]) / 2.0\n'
        '            wave["lon_mean"] = (float(np.mean(lons[peak])) + wave["lon_mean"]) / 2.0',
        '            wave["lat_mean"] = float(np.mean(lats[peak]))\n'
        '            wave["lon_mean"] = float(np.mean(lons[peak]))'),
    # K13 NaN treated as above threshold
    "nan_counts_as_above_threshold": _sub(
        "    return [np.nan_to_num(curvature, nan=-np.inf) >= threshold * step",
        "    return [np.nan_to_num(curvature, nan=np.inf) >= threshold * step"),
    # K14 the lone-wave early return removed, the obvious cleanup
    "lone_wave_early_return_removed": _sub(
        "    if len(merged) <= 1:\n        return merged",
        "    if False:\n        return merged"),
    # K27 the nearest-cell tie broken row-first, NumPy's order, instead of column-first,
    # the original's `find` order (the 2026-09-18 real case at 33035.75 of 1990)
    "nearest_tie_broken_row_first": _sub(
        "    base_cols, base_rows = np.nonzero(masks[0].T)",
        "    base_rows, base_cols = np.nonzero(masks[0])"),
}
