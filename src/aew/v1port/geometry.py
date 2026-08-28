"""Geometry helpers, ported from the archived AEWC version 1 MATLAB source.

Stage 3 of the port. These sit between the vorticity decomposition and the detection
step: the tracker is handed the advection of curvature vorticity as an input, and reports
a wavelength for each wave it finds.

Source files this replaces:
    calculate_advvort_f.m       advection of a vorticity field by the wind
    calculate_dlatdlon_f.m      meters back to degrees, the inverse of calculate_dxdy_f
    grcirc_dist_f.m             great-circle distance
    calculate_wavelength_f.m    wavelength along the wave's direction of travel

FAITHFUL reproductions are marked. As in the other stages, an oddity in the original is
kept and labeled rather than quietly repaired, because agreement with version 1 is what
validates this port.
"""

import numpy as np

from .vorticity import EARTH_RADIUS_M, MATLAB_FILL, STENCIL, grid_spacing_m

# calculate_advvort_f.m masks uncalc+1 = 3 rows and columns rather than the 2 that
# calculate_compvort_multi_f.m masks. It has to: it differentiates the vorticity field,
# which is itself undefined in the outer two rings.
ADVECTION_EDGE_WIDTH = 3

# grcirc_dist_f.m converts radians to its output unit through nautical miles.
_UNIT_PER_NAUTICAL_MILE = {
    "nm": 1.0,
    "km": 1.852,
    "mi": 1.15077945,
}


def meters_to_degrees(dx, dy, latitude_deg):
    """Grid spacing in degrees, ported from calculate_dlatdlon_f.m.

    The inverse of `grid_spacing_m`: the meridional spacing is latitude-independent and
    the zonal spacing carries a 1/cos(latitude) factor.
    """
    lat = np.asarray(latitude_deg, dtype=float)
    dlat = np.asarray(dy, dtype=float) * (180.0 / (np.pi * EARTH_RADIUS_M))
    dlon = (np.asarray(dx, dtype=float)
            * (180.0 / (np.pi * EARTH_RADIUS_M * np.cos(np.deg2rad(lat)))))
    return dlat, dlon


def great_circle_distance(lat1, lon1, lat2, lon2, units="km"):
    """Great-circle distance, ported from grcirc_dist_f.m.

    The original is the haversine form, returning radians unless a unit is named. The
    unit conversions all route through nautical miles, which is why the kilometre factor
    is 60 * 1.852 rather than a radius multiplication.
    """
    lat1, lon1 = np.deg2rad(np.asarray(lat1, float)), np.deg2rad(np.asarray(lon1, float))
    lat2, lon2 = np.deg2rad(np.asarray(lat2, float)), np.deg2rad(np.asarray(lon2, float))
    inner = (np.sin((lat1 - lat2) / 2.0) ** 2
             + np.cos(lat1) * np.cos(lat2) * np.sin((lon1 - lon2) / 2.0) ** 2)
    radians = 2.0 * np.arcsin(np.sqrt(np.clip(inner, 0.0, 1.0)))
    if units == "radians":
        return radians
    if units == "degrees":
        return np.rad2deg(radians)
    if units not in _UNIT_PER_NAUTICAL_MILE:
        raise ValueError("unknown unit %r; expected nm, km, mi, degrees or radians"
                         % (units,))
    return np.rad2deg(radians) * 60.0 * _UNIT_PER_NAUTICAL_MILE[units]


def advection_of_vorticity(latgrid, longrid, u, v, vorticity, global_grid=False,
                           fill=np.nan):
    """Advection of a vorticity field by the wind, from calculate_advvort_f.m.

        adv = -(u * d(vort)/dx) - (v * d(vort)/dy)

    Version 1 passes curvature vorticity here, so the tracker's input is the advection of
    curvature vorticity. The sign convention is the meteorological one: positive means the
    flow is carrying higher values toward the point.

    FAITHFUL, and worth recording because it looks like a bug and is not one in effect.
    In the original, the two masked branches assign to `advcurRV`, a variable that exists
    nowhere else in the function, rather than to the `advRV` being returned:

        advRV = ones(size(RV)).*-999;
        ...
          if (g <= uncalc+1) | g > (nlat-(uncalc+1));
            advcurRV(:,g,h) = -999;        % a throwaway variable

    The masked points still come back as -999 only because `advRV` was initialized to it.
    The behavior is therefore correct and the writes are dead. This port masks properly
    and produces the same values.

    FAITHFUL: the right-hand column mask again sits outside the global-grid guard, the
    same asymmetry `component_vorticity` carries. A global grid wraps its left edge and
    masks its right.
    """
    latgrid = np.asarray(latgrid, dtype=float)
    longrid = np.asarray(longrid, dtype=float)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    vorticity = np.asarray(vorticity, dtype=float)

    squeeze_time = u.ndim == 2
    if squeeze_time:
        u, v, vorticity = u[np.newaxis], v[np.newaxis], vorticity[np.newaxis]
    if not (u.shape == v.shape == vorticity.shape):
        raise ValueError("u, v and vorticity must have the same shape")

    nlat, nlon = latgrid.shape
    if u.shape[-2:] != (nlat, nlon):
        raise ValueError("fields must match the coordinate mesh, got %s against %s"
                         % (u.shape[-2:], (nlat, nlon)))
    if nlat <= 2 * ADVECTION_EDGE_WIDTH or nlon <= 2 * ADVECTION_EDGE_WIDTH:
        raise ValueError("grid too small to compute: needs more than %d rows and columns"
                         % (2 * ADVECTION_EDGE_WIDTH))

    lat_values = latgrid[:, 0]
    lon_values = longrid[0, :]
    dlon = (lon_values.max() - lon_values.min()) / (nlon - 1)
    dlat = (lat_values.max() - lat_values.min()) / (nlat - 1)

    out = np.full(u.shape, float(fill))
    rows = np.arange(ADVECTION_EDGE_WIDTH, nlat - ADVECTION_EDGE_WIDTH)
    cols = (np.arange(0, nlon - ADVECTION_EDGE_WIDTH) if global_grid
            else np.arange(ADVECTION_EDGE_WIDTH, nlon - ADVECTION_EDGE_WIDTH))
    if rows.size == 0 or cols.size == 0:
        return out[0] if squeeze_time else out

    if global_grid:
        dx, dy = grid_spacing_m(rows.astype(float), dlon, dlat,
                                nlat=nlat, nlon=nlon, global_grid=True)
    else:
        dx, dy = grid_spacing_m(lat_values[rows], dlon, dlat)
    dx, dy = dx[:, np.newaxis], dy[:, np.newaxis]

    east = (cols + STENCIL) % nlon
    west = (cols - STENCIL) % nlon
    rr, cc = rows[:, np.newaxis], cols[np.newaxis, :]
    span = 2.0 * STENCIL

    ddx = (vorticity[:, rr, east[np.newaxis, :]]
           - vorticity[:, rr, west[np.newaxis, :]]) / (span * dx)
    ddy = (vorticity[:, rows[:, np.newaxis] + STENCIL, cc]
           - vorticity[:, rows[:, np.newaxis] - STENCIL, cc]) / (span * dy)

    out[:, rr, cc] = -(u[:, rr, cc] * ddx) - (v[:, rr, cc] * ddy)
    return out[0] if squeeze_time else out


def _snap(value, resolution):
    """Round to the nearest multiple of the grid resolution, as the original does."""
    return np.round(np.asarray(value, dtype=float) / resolution) * resolution


def _direction_to_degree(u, v):
    """Reference direction rounded to the nearest degree, as calculate_wavelength_f does.

    The rounding is what makes an exact equality test between two directions meaningful:
    without it no grid point would ever match the wave's heading exactly.
    """
    from .vorticity import reference_direction
    radians = reference_direction(u, v)
    return np.round(radians / np.deg2rad(1.0)) * np.deg2rad(1.0)


def wavelength_along_ray(lat1, lon1, lat2, lon2, latgrid, longrid, curvature,
                         threshold, resolution, max_threshold):
    """Wavelength along the wave's direction of travel, from calculate_wavelength_f.m.

    The method: take the heading from the previous wave center to the current one, find
    every grid point lying on that heading from the current center, walk outward along it,
    and call the distance at which the curvature vorticity anomaly first crosses zero a
    quarter wavelength. Multiply by four.

    Returns ``MATLAB_FILL`` when no wavelength can be formed, exactly as the original
    returns -999.

    Both of the original's cases are reproduced, including the fallback for a profile
    that decreases and then rises again before reaching zero.
    """
    curvature = np.asarray(curvature, dtype=float).ravel()
    latflat = np.asarray(latgrid, dtype=float).ravel()
    lonflat = np.asarray(longrid, dtype=float).ravel()
    if curvature.size != latflat.size:
        raise ValueError("curvature has %d points but the mesh has %d"
                         % (curvature.size, latflat.size))

    heading = _direction_to_degree(_snap(lon2, resolution) - _snap(lon1, resolution),
                                   _snap(lat2, resolution) - _snap(lat1, resolution))
    point_heading = _direction_to_degree(_snap(lonflat, resolution) - _snap(lon2, resolution),
                                         _snap(latflat, resolution) - _snap(lat2, resolution))

    on_ray = np.flatnonzero(point_heading == heading)
    if on_ray.size < 3:
        return MATLAB_FILL

    distance = great_circle_distance(_snap(lat2, resolution), _snap(lon2, resolution),
                                     _snap(latflat[on_ray], resolution),
                                     _snap(lonflat[on_ray], resolution), "km")
    order = np.argsort(distance, kind="stable")
    distance = distance[order]
    on_ray = on_ray[order]
    profile = curvature[on_ray]

    output = MATLAB_FILL

    # The ray must start inside a trough, or it is not measuring a wave. Note WHICH point
    # that is: the wave center does not lie on its own ray, because the direction from a
    # point to itself is zero rather than the wave's heading, so `profile[0]` is the first
    # grid point one step out. This check also guarantees the first zero-crossing cannot
    # be at index 0, so the interpolation below always has a point behind it to bracket
    # with, which is what keeps the original's `sid(1)-1` from indexing off the front.
    if profile[0] >= threshold:
        negative = np.flatnonzero(profile < 0)
        if negative.size > 1:
            k = negative[0]
            pair_c = profile[k - 1:k + 1]
            pair_d = distance[k - 1:k + 1]
            if pair_c[0] != pair_c[1]:
                quarter = np.interp(0.0, pair_c[::-1], pair_d[::-1])
                output = 4.0 * float(quarter)

    # Case 2 in the original: no crossing, or one implying an implausibly long wave.
    if output == MATLAB_FILL or output > max_threshold:
        drop = profile[:-1] - profile[1:]
        falling = np.flatnonzero(drop > 0)
        if falling.size:
            start = falling[0]
            tail = np.flatnonzero((drop[start:] < 0)
                                  & (profile[start:-1] >= threshold))
            if tail.size and on_ray.size >= 3:
                candidate = 4.0 * float(distance[start + tail[0] - 1])
                if ((output != MATLAB_FILL and 4.0 * distance[tail[0]] < output)
                        or (output == MATLAB_FILL and 4.0 * distance[tail[0]] > 0)):
                    output = candidate
    return output


def points_in_polygon(polygon_lons, polygon_lats, lons, lats, tol=1e-9):
    """Which points lie inside or on the polygon, matching MATLAB `inpolygon`.

    WRITTEN OUT RATHER THAN DELEGATED, for two reasons.

    THE BOUNDARY. MATLAB's `inpolygon` counts a point ON an edge as inside, and returns
    that in its first output, which is the one version 1 reads, both in the association
    stage and in merge_contours_f.m. The obvious Python substitute, matplotlib's
    `Path.contains_points`, leaves boundary points undefined at its default radius. The
    difference is not theoretical here: the polygons are convex hulls of one wave's grid
    points and the points tested belong to another wave on the same regular grid, so a
    point landing exactly on a hull edge is an ordinary event. The edge test below runs
    first and settles it.

    THE DEPENDENCY. Delegating pulled matplotlib into the tracker's core, where it was its
    only use, and matplotlib is a plotting extra rather than a core requirement of this
    package. A core installation would have seeded tracks and then failed on the first
    matching pass. A ray cast is a dozen lines and removes the import.

    On the straddle test in the ray cast: `>` and `>=` are both valid conventions and they
    decide how a ray grazing a vertex is counted. Over 24,000 random grid-aligned point and
    polygon pairs the two disagreed 207 times and every disagreement was on a point ON the
    boundary, which the edge test has already resolved. Either reading gives the same
    answer here.
    """
    poly_x = np.asarray(polygon_lons, dtype=float).ravel()
    poly_y = np.asarray(polygon_lats, dtype=float).ravel()
    px = np.asarray(lons, dtype=float).ravel()
    py = np.asarray(lats, dtype=float).ravel()
    inside = np.zeros(px.shape, dtype=bool)
    if poly_x.size < 3 or not (np.all(np.isfinite(poly_x)) and np.all(np.isfinite(poly_y))):
        # A polygon with a non-finite vertex contains nothing. Version 1 reaches that state
        # by inflating a zero-area polygon, which divides by zero; see association._inflate.
        return inside

    x1, y1 = poly_x, poly_y
    x2, y2 = np.roll(poly_x, -1), np.roll(poly_y, -1)
    for i in range(px.size):
        x, y = px[i], py[i]
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        # On an edge counts as inside, which is what `inpolygon` reports.
        cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
        within = ((np.minimum(x1, x2) - tol <= x) & (x <= np.maximum(x1, x2) + tol) &
                  (np.minimum(y1, y2) - tol <= y) & (y <= np.maximum(y1, y2) + tol))
        if np.any((np.abs(cross) <= tol) & within):
            inside[i] = True
            continue
        # Ray cast along increasing x, counting edge crossings.
        straddles = (y1 > y) != (y2 > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_at_y = np.where(straddles, x1 + (y - y1) * (x2 - x1) / (y2 - y1), np.inf)
        inside[i] = bool(np.count_nonzero(straddles & (x < x_at_y)) % 2)
    return inside
