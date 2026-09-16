"""Trough detection, ported from the first half of find_ews_f.m.

Stage 5 of the port. This is everything the original does at a single timestep before it
starts matching candidates to existing tracks: smooth, mask, find trough axes, and merge
the resulting candidates into waves. The association half is a separate unit.

THE TROUGH AXIS IS THE ZERO CONTOUR OF CURVATURE-VORTICITY ADVECTION. The original sets

    tr_thr = 0.*10^-5;   %Advection of Curvature Vorticity (Trough Axis Identification)

which is exactly zero, and contours the advection field at that level. That is a
physically meaningful definition rather than a tuned one: where the advection of curvature
vorticity changes sign, the curvature vorticity is at an extremum along the flow, which is
the trough axis.

Two masks run first. Anywhere the zonal wind exceeds +2.5 m/s is discarded, which removes
westerly flow and keeps the easterly regime the waves live in. Anywhere the curvature
vorticity anomaly is below its threshold is discarded, coarse threshold for the coarse
field and fine for the fine.

Then two merge passes, coarse grid first and fine grid second, both through
`contours.merge_contours`.
"""

import numpy as np

from .climatology import smooth9, southern_hemisphere_sign
from .contours import merge_contours

# find_ews_f.m parameters
TROUGH_LEVEL = 0.0        # tr_thr, the advection contour that defines a trough axis
MAX_ZONAL_WIND = 2.5      # u_thr, in meters per second


def _prepare(field, lat_values):
    """Smooth, then make cyclonic positive in both hemispheres."""
    return southern_hemisphere_sign(smooth9(field)[np.newaxis, ...], lat_values)[0]


def trough_axes(latgrid, longrid, advection, level=TROUGH_LEVEL):
    """Trough axes as polylines, replacing the original's use of MATLAB `contours`.

    Returns a list of (lat, lon) vertex arrays, one per contour.

    A DELIBERATE DIVERGENCE, and a more consequential one than the region search in the
    contour stage, so it is stated in full.

    MATLAB's `contours` returns a flat 2-by-N matrix in which a contour is a header column
    holding [level; number_of_points] followed by that many vertex columns. To find the
    headers, find_ews_f.m searches for columns whose FIRST ROW equals the level:

        id = find(ch(1,:) == tr_thr);

    That is only safe when no vertex x-coordinate can equal the level. Here the level is
    ZERO and the x-coordinate is LONGITUDE, and the version 1 tracking domain runs from
    140 W to 40 E on a 2.5-degree grid, so longitude zero is a grid line and contour
    vertices land on it. The search therefore returns real headers AND every vertex on the
    prime meridian.

    The original defends with two guards: the supposed point count must be a whole number
    greater than one, and for all but the first it must agree with where the previous
    contour ended. Those reject many spurious hits but not all, since a vertex at
    longitude zero and a whole-number latitude above one passes the first guard.

    This port walks the contours properly, so the ambiguity does not arise.

    MEASURED 2026-08-30, AND IT DOES NOT FIRE ON VERSION 1'S OWN GRID. The parser was
    transcribed and run under Octave over 75 timesteps of 1990 and 8,512 contours: every
    column matching `find(ch(1,:) == 0)` was a real header, none was a vertex, and no
    real contour was lost. The reason is the grid. Version 1 retrieves ONE DEGREE data
    and decimates with a stride of two, so its tracking longitudes are ODD, zero is not
    a grid line, and no vertex interpolated exactly onto it.

    THE MECHANISM IS REAL AND A CONTROL SHOWS IT, because a measurement that returns
    zero may be measuring nothing. Repeating it with the longitudes shifted one degree,
    so zero becomes a grid line, gives 173 spurious matches and 121 real contours lost,
    1.4 percent. The ambiguity is genuine and this configuration avoids it, which is not
    the same as the parse being sound.

    An earlier version of this note said the domain runs on a 2.5 degree grid, on which
    zero IS a grid line. That rested on the pre-2026-08-29 belief about the input
    resolution: the decimation gives 2.0 degrees from one degree input, not 2.5.
    """
    from contourpy import contour_generator
    lat_values = latgrid[:, 0]
    lon_values = longrid[0, :]
    field = np.asarray(advection, dtype=float)
    # contourpy treats NaN as absent, which is what the masks above intend
    generator = contour_generator(x=lon_values, y=lat_values, z=field,
                                  name="serial", corner_mask=True)
    out = []
    for segment in generator.lines(float(level)):
        seg = np.asarray(segment, dtype=float)
        if seg.ndim != 2 or seg.shape[0] < 2:
            continue
        out.append((seg[:, 1], seg[:, 0]))       # (lat, lon)
    return out


def detect_troughs(time, latgrid_coarse, longrid_coarse, u_coarse,
                   curvature_anomaly_coarse, advection_anomaly_coarse,
                   latgrid_fine, longrid_fine, curvature_anomaly_fine,
                   coarse_threshold, fine_threshold, absorb=False):
    """One timestep of version 1 trough detection.

    Parameters mirror what find_ews_f.m has in hand inside its time loop: the coarse grid
    carries the wind, the curvature anomaly and its advection, and the fine grid carries
    the curvature anomaly again at input resolution.

    Returns the merged wave list from `merge_contours`, which is what the association
    stage consumes. `absorb=True` enables the first-pass absorption version 1's own code
    attempts and never completes; see `contours.merge_contours`.
    """
    lat_c = np.asarray(latgrid_coarse, dtype=float)[:, 0]
    lat_f = np.asarray(latgrid_fine, dtype=float)[:, 0]

    wind = smooth9(np.asarray(u_coarse, dtype=float))
    curvature_c = _prepare(np.asarray(curvature_anomaly_coarse, dtype=float), lat_c)
    advection_c = smooth9(np.asarray(advection_anomaly_coarse, dtype=float))
    curvature_f = _prepare(np.asarray(curvature_anomaly_fine, dtype=float), lat_f)

    # Mask 1: discard westerly flow. Note the original tests the signed wind rather than
    # its magnitude, so strong EASTERLIES are kept, which is the regime of interest.
    westerly = wind > MAX_ZONAL_WIND
    advection_c = np.where(westerly, np.nan, advection_c)
    curvature_c = np.where(westerly, np.nan, curvature_c)

    # Mask 2: discard anomalies below threshold, each field against its own.
    #
    # FAITHFUL: the original masks the FINE field by curvature only. Its zonal-wind mask
    # for the fine grid is present but commented out:
    #     %id = find(ut2 > u_thr);
    #     %crvt2(id) = nan;
    # so the wind mask applies to the coarse fields alone. Reproduced.
    weak_c = curvature_c < coarse_threshold
    advection_c = np.where(weak_c, np.nan, advection_c)
    curvature_c = np.where(weak_c, np.nan, curvature_c)
    curvature_f = np.where(curvature_f < fine_threshold, np.nan, curvature_f)

    axes = trough_axes(latgrid_coarse, longrid_coarse, advection_c)
    if not axes:
        return []

    candidates = [{"time": time,
                   "lat_mean": float(np.mean(lats)),
                   "lon_mean": float(np.mean(lons))}
                  for lats, lons in axes]

    # BOTH passes take the flag. Version 1 runs the same merge twice and its absorption
    # is broken in both, so honouring it in one and not the other would be neither
    # version 1 nor the repair.
    coarse = merge_contours(candidates, latgrid_coarse, longrid_coarse,
                            curvature_c, coarse_threshold, absorb=absorb)
    if not coarse:
        return []
    return merge_contours(coarse, latgrid_fine, longrid_fine,
                          curvature_f, fine_threshold, absorb=absorb)
