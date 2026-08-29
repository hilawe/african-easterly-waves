"""Mutations for the basin assignment.

The list was written before the assertions existed and is reproduced in the docstring of
tests/test_v1port_regions.py. Run with the repository's mutation checker against
src/aew/v1port/regions.py.

The strongest test these are checked against needs both version 1's archived polygons and
its published record, and both are untracked data, so that test SKIPS here. What catches
these is the synthetic rule-level suite, which is the reason it exists alongside the
twelve-thousand-track check rather than being replaced by it.
"""


def _sub(old, new):
    return lambda s: s.replace(old, new)


MUTATIONS = {
    # G1 the basin comes from the last observation rather than the first
    "filed_by_where_it_ended": _sub(
        "    name = region_of_point(lon[0], lat[0], regions)",
        "    name = region_of_point(lon[-1], lat[-1], regions)"),
    # G1 the basin comes from the track's mean position
    "filed_by_the_mean_position": _sub(
        "    name = region_of_point(lon[0], lat[0], regions)",
        "    name = region_of_point(lon.mean(), lat.mean(), regions)"),
    # G2 the quarter-degree snap is skipped
    "snap_skipped": _sub(
        "    x, y = float(snap(lon)), float(snap(lat))",
        "    x, y = float(lon), float(lat)"),
    # G2 the snap uses the wrong grid
    "snap_to_the_wrong_grid": _sub(
        "SNAP_DEGREES = 0.25",
        "SNAP_DEGREES = 1.0"),
    # G3 and G4 the polygons are tested in reverse, so the catch-all takes everything
    "polygons_tested_in_reverse": _sub(
        "    for name, poly_lon, poly_lat in regions:",
        "    for name, poly_lon, poly_lat in reversed(regions):"),
    # G3 the first match does not win
    "last_match_wins": _sub(
        "        if points_in_polygon(poly_lon, poly_lat, [x], [y])[0]:\n"
        "            return name\n"
        "    return None",
        "    found = None\n"
        "    for name, poly_lon, poly_lat in regions:\n"
        "        if points_in_polygon(poly_lon, poly_lat, [x], [y])[0]:\n"
        "            found = name\n"
        "    return found"),
    # G5 a wave matching nothing is filed under the last polygon anyway
    "unmatched_waves_get_a_fallback": _sub(
        "    return None\n\n\ndef assign_region",
        "    return regions[-1][0]\n\n\ndef assign_region"),
    # G7 the stored polygon order is not preserved
    "stored_order_not_preserved": _sub(
        "    return tuple(out)",
        "    return tuple(sorted(out, key=lambda r: r[0]))"),
    # G8 unassigned waves are not counted, so a run missing waves looks clean
    "unassigned_not_counted": _sub(
        "        if assign_region(track, regions) is None:\n            unassigned += 1",
        "        assign_region(track, regions)"),
    # a missing polygon file is worked around instead of refused
    "missing_polygon_file_not_refused": _sub(
        "    if not os.path.exists(path):",
        "    if False:"),
}
