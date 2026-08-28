"""Mutations for the version 1 geometry helpers.

The list was written before the assertions existed and is reproduced in the docstring of
tests/test_v1port_geometry.py. Run with the repository's mutation checker against
src/aew/v1port/geometry.py.
"""


def _sub(old, new):
    return lambda s: s.replace(old, new)


MUTATIONS = {
    # G1 advection sign flipped
    "advection_sign_flipped": _sub(
        "    out[:, rr, cc] = -(u[:, rr, cc] * ddx) - (v[:, rr, cc] * ddy)",
        "    out[:, rr, cc] = (u[:, rr, cc] * ddx) + (v[:, rr, cc] * ddy)"),
    # G2 two rings masked instead of three
    "advection_masks_two_rings": _sub(
        "ADVECTION_EDGE_WIDTH = 3",
        "ADVECTION_EDGE_WIDTH = 2"),
    # G3 the right-hand mask symmetrized on a global grid
    "advection_global_mask_symmetrized": _sub(
        "    cols = (np.arange(0, nlon - ADVECTION_EDGE_WIDTH) if global_grid",
        "    cols = (np.arange(0, nlon) if global_grid"),
    # G4 the two derivative terms swapped
    "advection_terms_swapped": _sub(
        "    out[:, rr, cc] = -(u[:, rr, cc] * ddx) - (v[:, rr, cc] * ddy)",
        "    out[:, rr, cc] = -(u[:, rr, cc] * ddy) - (v[:, rr, cc] * ddx)"),
    # G5 metres to degrees using sine
    "degrees_use_sine": _sub(
        "            * (180.0 / (np.pi * EARTH_RADIUS_M * np.cos(np.deg2rad(lat)))))",
        "            * (180.0 / (np.pi * EARTH_RADIUS_M * np.sin(np.deg2rad(lat)))))"),
    # G6 the latitude factor applied to the latitude spacing
    "degrees_latitude_factor_misplaced": _sub(
        "    dlat = np.asarray(dy, dtype=float) * (180.0 / (np.pi * EARTH_RADIUS_M))",
        "    dlat = np.asarray(dy, dtype=float) * (180.0 / (np.pi * EARTH_RADIUS_M "
        "* np.cos(np.deg2rad(np.asarray(latitude_deg, dtype=float)))))"),
    # G7 a wrong unit factor
    "distance_wrong_unit_factor": _sub(
        '    "km": 1.852,',
        '    "km": 1.609,'),
    # G8 the cosine term dropped from the haversine
    "distance_drops_the_cosine_term": _sub(
        "    inner = (np.sin((lat1 - lat2) / 2.0) ** 2\n"
        "             + np.cos(lat1) * np.cos(lat2) * np.sin((lon1 - lon2) / 2.0) ** 2)",
        "    inner = (np.sin((lat1 - lat2) / 2.0) ** 2\n"
        "             + np.sin((lon1 - lon2) / 2.0) ** 2)"),
    # G9 the quarter-to-whole factor dropped
    "wavelength_drops_the_factor_of_four": _sub(
        "                output = 4.0 * float(quarter)",
        "                output = float(quarter)"),
    # G10 the heading not rounded, so nothing matches
    "wavelength_heading_not_rounded": _sub(
        "    return np.round(radians / np.deg2rad(1.0)) * np.deg2rad(1.0)",
        "    return radians"),
    # G11 the ray-start threshold check removed
    "wavelength_ignores_the_trough_check": _sub(
        "    if profile[0] >= threshold:",
        "    if True:"),
    # G12 the crossing interpolated from the wrong pair
    "wavelength_wrong_bracket": _sub(
        "            pair_c = profile[k - 1:k + 1]\n            pair_d = distance[k - 1:k + 1]",
        "            pair_c = profile[k:k + 2]\n            pair_d = distance[k:k + 2]"),
}
