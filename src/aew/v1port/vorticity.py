"""Vorticity decomposition, ported from the archived AEWC version 1 MATLAB source.

This is the bottom of the version 1 dependency chain and the quantity its detection
thresholds are percentiles of, so it is the first thing to port and the first thing the
ERA5 climatology needs.

FAITHFULNESS IS THE POINT. The purpose of this port is to reproduce version 1 exactly,
because faithful reproduction is what validates the port. Where the original does
something odd, this module reproduces it and says so rather than quietly improving it.
Two such places are marked FAITHFUL below, and both are real asymmetries in the original:

- The right-hand column mask sits OUTSIDE the global-grid guard in the MATLAB, so a
  global grid wraps its left edge but still masks its right edge.
- `calculate_dxdy_f` writes the cosine of latitude as `sin((lat + 90) * pi / 180)`, which
  is the same number by a trigonometric identity but reads as a different quantity.

Source files this replaces:
    calculate_compvort_multi_f.m    the decomposition
    calculate_refdir_array_f.m      the reference wind direction
    calculate_ptdir_array_f.m       projection onto that direction
    calculate_dxdy_f.m              grid spacing in meters

The decomposition itself is the standard natural-coordinate one. Relative vorticity is
computed by centered differences, shear vorticity is computed from the wind components
projected onto the local reference direction, and curvature vorticity is the residual.
"""

import numpy as np

# The MATLAB uses the equatorial radius, not a mean radius. Keep its value exactly.
EARTH_RADIUS_M = 6378100.0

# calculate_compvort_multi_f.m sets uncalc = 2, so two rows at each latitude edge and
# two columns at the right edge carry no result. dnopts = 1 is the centered-difference
# half-width.
EDGE_WIDTH = 2
STENCIL = 1

# The MATLAB fills unresolved points with this sentinel. This module returns NaN and
# offers the sentinel on request, so a caller comparing against MATLAB output can ask
# for the original encoding.
MATLAB_FILL = -999.0


def reference_direction(u, v):
    """The reference wind direction, in radians clockwise from north, in [0, 2*pi).

    calculate_refdir_array_f.m enumerates eight quadrant and axis cases plus the
    both-zero case. Every branch reduces to the same thing: the meteorological direction
    angle measured from north, which is atan2(u, v) rather than the mathematical
    atan2(v, u). Verified case by case against the original's branches, including the
    convention that a zero wind returns 0.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    return np.mod(np.arctan2(u, v), 2.0 * np.pi)


def project_onto_direction(u, v, refdir):
    """Signed magnitude of (u, v) projected onto the direction `refdir`.

    calculate_ptdir_array_f.m computes |V| * cos(refdir - dir(u, v)) and defines the
    projection of a zero wind as exactly zero rather than as an indeterminate angle.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    magnitude = np.hypot(u, v)
    direction = np.mod(np.arctan2(u, v), 2.0 * np.pi)
    projection = magnitude * np.cos(refdir - direction)
    # both components zero: the original assigns 0 and never consults an angle
    return np.where((u == 0) & (v == 0), 0.0, projection)


def grid_spacing_m(lat_deg, dlon_deg, dlat_deg, nlat=None, nlon=None, global_grid=False):
    """Grid spacing in meters at a given latitude, from calculate_dxdy_f.m.

    FAITHFUL: the regional branch of the original computes the zonal spacing as
    `sin((lat + 90) / 180 * pi)`, which is identically `cos(lat)`. This uses the cosine
    form, which is the same number and says what it means.
    """
    if global_grid:
        if nlat is None or nlon is None:
            raise ValueError("a global grid needs nlat and nlon")
        # the original indexes latitude by row rather than by value in this branch
        row = np.asarray(lat_deg, dtype=float)
        colatitude = (row / (nlat - 1)) * np.pi
        dx = (np.sin(colatitude) * EARTH_RADIUS_M * 2.0 * np.pi) / nlon
        dy = np.full_like(np.asarray(dx, dtype=float),
                          (np.pi * EARTH_RADIUS_M) / (nlat - 1))
        return dx, dy

    lat = np.asarray(lat_deg, dtype=float)
    dx = np.deg2rad(1.0) * EARTH_RADIUS_M * np.cos(np.deg2rad(lat)) * dlon_deg
    dy = np.full_like(dx, np.deg2rad(1.0) * EARTH_RADIUS_M * dlat_deg)
    return dx, dy


def component_vorticity(latgrid, longrid, u, v, global_grid=False, fill=np.nan):
    """Relative, shear and curvature vorticity, ported from calculate_compvort_multi_f.m.

    Parameters
    ----------
    latgrid, longrid : 2-D arrays, shape (nlat, nlon)
        Coordinate meshes, as produced by ``numpy.meshgrid(lon, lat)``.
    u, v : arrays, shape (ntime, nlat, nlon) or (nlat, nlon)
        Wind components on that mesh.
    global_grid : bool
        The MATLAB `globflag`. When true the zonal derivative wraps at the left edge and
        the spacing is computed from the row index rather than the latitude value.
    fill : float
        Value for points the original does not compute. Pass ``MATLAB_FILL`` to match
        the original's encoding exactly.

    Returns
    -------
    (relative, shear, curvature), each with the shape of ``u``.

    FAITHFUL, and this is a real asymmetry in the original rather than a porting choice.
    In `calculate_compvort_multi_f.m` the masking test is

        elseif ( (globflag == 0) & (h <= uncalc) ) | (h > (nlon-uncalc));

    The right-hand term sits OUTSIDE the `globflag` guard, so on a global grid the left
    edge wraps and is computed while the right edge is masked anyway. This reproduces
    that. Do not silently symmetrize it: the whole value of the port is that it agrees
    with version 1, and any deliberate change belongs in a later, separate step.
    """
    latgrid = np.asarray(latgrid, dtype=float)
    longrid = np.asarray(longrid, dtype=float)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    squeeze_time = u.ndim == 2
    if squeeze_time:
        u = u[np.newaxis, ...]
        v = v[np.newaxis, ...]
    if u.shape != v.shape:
        raise ValueError("u and v must have the same shape")

    nlat, nlon = latgrid.shape
    if u.shape[-2:] != (nlat, nlon):
        raise ValueError("u and v must match the coordinate mesh, got %s against %s"
                         % (u.shape[-2:], (nlat, nlon)))
    if nlat <= 2 * EDGE_WIDTH or nlon <= 2 * EDGE_WIDTH:
        raise ValueError("grid too small to compute: needs more than %d rows and columns"
                         % (2 * EDGE_WIDTH))

    lat_values = latgrid[:, 0]
    lon_values = longrid[0, :]
    dlon = (lon_values.max() - lon_values.min()) / (nlon - 1)
    dlat = (lat_values.max() - lat_values.min()) / (nlat - 1)

    relative = np.full(u.shape, float(fill))
    shear = np.full(u.shape, float(fill))
    curvature = np.full(u.shape, float(fill))

    rows = np.arange(EDGE_WIDTH, nlat - EDGE_WIDTH)
    if global_grid:
        cols = np.arange(0, nlon - EDGE_WIDTH)
    else:
        cols = np.arange(EDGE_WIDTH, nlon - EDGE_WIDTH)
    if rows.size == 0 or cols.size == 0:
        return _finish(relative, shear, curvature, squeeze_time)

    # column neighbours, wrapped exactly as the MATLAB wraps them
    east = (cols + STENCIL) % nlon
    west = (cols - STENCIL) % nlon
    north = rows + STENCIL
    south = rows - STENCIL

    if global_grid:
        dx, dy = grid_spacing_m(rows.astype(float), dlon, dlat,
                                nlat=nlat, nlon=nlon, global_grid=True)
    else:
        dx, dy = grid_spacing_m(lat_values[rows], dlon, dlat)
    dx = dx[:, np.newaxis]
    dy = dy[:, np.newaxis]

    rr = rows[:, np.newaxis]
    cc = cols[np.newaxis, :]

    u_here, v_here = u[:, rr, cc], v[:, rr, cc]
    u_east, v_east = u[:, rr, east[np.newaxis, :]], v[:, rr, east[np.newaxis, :]]
    u_west, v_west = u[:, rr, west[np.newaxis, :]], v[:, rr, west[np.newaxis, :]]
    u_north = u[:, north[:, np.newaxis], cc]
    v_north = v[:, north[:, np.newaxis], cc]
    u_south = u[:, south[:, np.newaxis], cc]
    v_south = v[:, south[:, np.newaxis], cc]

    span = 2.0 * STENCIL
    rv = ((v_east - v_west) / (span * dx)) - ((u_north - u_south) / (span * dy))

    refdir = reference_direction(u_here, v_here)

    def along(uu, vv):
        p = project_onto_direction(uu, vv, refdir)
        return p * np.sin(refdir), p * np.cos(refdir)

    _, v_ref_east = along(u_east, v_east)
    _, v_ref_west = along(u_west, v_west)
    u_ref_north, _ = along(u_north, v_north)
    u_ref_south, _ = along(u_south, v_south)

    sv = ((v_ref_east - v_ref_west) / (span * dx)
          - (u_ref_north - u_ref_south) / (span * dy))

    relative[:, rr, cc] = rv
    shear[:, rr, cc] = sv
    curvature[:, rr, cc] = rv - sv

    return _finish(relative, shear, curvature, squeeze_time)


def _finish(relative, shear, curvature, squeeze_time):
    if squeeze_time:
        return relative[0], shear[0], curvature[0]
    return relative, shear, curvature
