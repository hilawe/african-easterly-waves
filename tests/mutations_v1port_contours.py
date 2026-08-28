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
    "seed_on_empty_cell_returns_a_region": _sub(
        "    if not mask[row, col]:\n        return np.zeros_like(mask)",
        "    if False:\n        return np.zeros_like(mask)"),
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
    "hull_absorbs_everything": _sub(
        "        return polygon.contains_points(np.column_stack([points_lon, points_lat]))",
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
}
