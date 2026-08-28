"""Mutations for the version 1 vorticity port, one per property the suite claims.

The list was written before the assertions existed and is reproduced in the docstring of
tests/test_v1port_vorticity.py. Run with the repository's mutation checker:

    python3 <mutation-check tool> --repo . \
        --target src/aew/v1port/vorticity.py \
        --catalogue tests/mutations_v1port_vorticity.py \
        --test-cmd ".venv/bin/python -m pytest -q tests/test_v1port_vorticity.py"
"""


def _sub(old, new):
    return lambda s: s.replace(old, new)


MUTATIONS = {
    # V1 no quadrant wrapping, so two quadrants come back negative
    "refdir_unwrapped": _sub(
        "    return np.mod(np.arctan2(u, v), 2.0 * np.pi)",
        "    return np.arctan2(u, v)"),
    # V2 the mathematical convention instead of the meteorological one
    "refdir_from_east_not_north": _sub(
        "    return np.mod(np.arctan2(u, v), 2.0 * np.pi)",
        "    return np.mod(np.arctan2(v, u), 2.0 * np.pi)"),
    # V3 the original's sin((lat+90) deg) misread as sin(lat)
    "zonal_spacing_uses_sine": _sub(
        "    dx = np.deg2rad(1.0) * EARTH_RADIUS_M * np.cos(np.deg2rad(lat)) * dlon_deg",
        "    dx = np.deg2rad(1.0) * EARTH_RADIUS_M * np.sin(np.deg2rad(lat)) * dlon_deg"),
    # V4 the latitude factor dropped entirely
    "zonal_spacing_ignores_latitude": _sub(
        "    dx = np.deg2rad(1.0) * EARTH_RADIUS_M * np.cos(np.deg2rad(lat)) * dlon_deg",
        "    dx = np.deg2rad(1.0) * EARTH_RADIUS_M * np.ones_like(lat) * dlon_deg"),
    # V5 the latitude edge mask not applied
    "no_latitude_edge_mask": _sub(
        "    rows = np.arange(EDGE_WIDTH, nlat - EDGE_WIDTH)",
        "    rows = np.arange(STENCIL, nlat - STENCIL)"),
    # V6 curvature sign flipped
    "curvature_sign_flipped": _sub(
        "    curvature[:, rr, cc] = rv - sv",
        "    curvature[:, rr, cc] = sv - rv"),
    # V7 centered difference divided by one spacing rather than two
    "single_spacing_difference": _sub(
        "    span = 2.0 * STENCIL",
        "    span = 1.0 * STENCIL"),
    # V8 the zonal neighbour does not wrap
    "zonal_neighbour_does_not_wrap": _sub(
        "    east = (cols + STENCIL) % nlon\n    west = (cols - STENCIL) % nlon",
        "    east = np.clip(cols + STENCIL, 0, nlon - 1)\n"
        "    west = np.clip(cols - STENCIL, 0, nlon - 1)"),
    # V9 the two derivative terms of relative vorticity swapped
    "relative_terms_swapped": _sub(
        "    rv = ((v_east - v_west) / (span * dx)) - ((u_north - u_south) / (span * dy))",
        "    rv = ((u_north - u_south) / (span * dy)) - ((v_east - v_west) / (span * dx))"),
    # V10 a calm wind projects to NaN rather than the zero the original defines
    "calm_wind_projects_to_nan": _sub(
        "    return np.where((u == 0) & (v == 0), 0.0, projection)",
        "    return projection"),
    # V11 the right-hand mask symmetrized on a global grid, the tempting cleanup that
    # would put the port quietly out of agreement with version 1
    "global_mask_symmetrized": _sub(
        "        cols = np.arange(0, nlon - EDGE_WIDTH)",
        "        cols = np.arange(0, nlon)"),
}
