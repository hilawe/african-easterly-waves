"""The whole tracker, one year at a time, ported from p2_track_*.m and find_ews_f.m's loop.

Stage 8, and the one that makes the other seven mean anything. Until this existed every
stage was tested alone and the chain had never produced a record, which is the difference
between a port and a set of modules.

WHAT VERSION 1 DOES PER YEAR, from p2_track_eraint_700hPa.m, in its order:

  1. Load u and v at the level on the native grid, and the curvature vorticity derived
     from them.
  2. Subtract the climatological mean for each timestep's calendar position, giving the
     curvature anomaly the whole method works on.
  3. Compute the advection of that anomaly by the wind. VERSION 1 FLIPS LATITUDE both
     going in and coming out, which is reproduced below and explained at `_advection`.
  4. Decimate u, v, the anomaly and its advection from native resolution to the coarse
     tracking grid, 2.5 degrees.
  5. Subset both grids to the published domain.
  6. Run the timestep loop.
  7. Filter and write.

WHAT THE TIMESTEP LOOP DOES, from find_ews_f.m, and the ORDER IS THE POINT because two of
its steps are conditional on each other:

  detect  ->  if nothing found, SKIP THE REST OF THIS TIMESTEP  ->  associate  ->  prune

The skip is a `continue` in the original, and the timestep loop does not close until after
the prune, so a wave-free timestep runs neither association nor pruning. Getting that wrong
in either place changes the record: `associate_step` and `prune_stale_tracks` both take the
candidate list so neither can be called wrongly by accident.

THE MEDIAN WINDS ARE THE SMOOTHED FINE WINDS. Version 1 smooths u and v on the fine grid
once at the top of each timestep and takes `nanmedian` over each wave's own trough cells to
predict where it goes. `nanmedian`, not `median`: the smoother propagates NaN deliberately,
so a plain median returns NaN wherever a wave touches masked ground and silently ends the
track.

WHAT THIS MODULE DOES NOT DO. It does not composite the satellite and radiation fields;
that is generate_ew_stats_f.m, which is not ported, so the record it writes has those
variables at their fill value and no basin assigned. It also does not compute wavelengths,
for the same reason: version 1 fills them in the statistics stage.
"""

import numpy as np

from . import climatology as clim
from .association import associate_step, finalize_tracks, prune_stale_tracks
from .detection import detect_troughs
from .geometry import advection_of_vorticity
from .vorticity import component_vorticity

# p2_track_eraint_700hPa.m
DOMAIN_LAT = (-35.0, 35.0)
DOMAIN_LON = (-140.0, 40.0)
COARSE_RESOLUTION_DEG = 2.5

# find_ews_f.m, hardcoded per reanalysis and level rather than computed. Version 1 states
# them as the 55th and 66th percentiles of that reanalysis's own anomaly distribution, so
# they are NOT transferable between reanalyses and a new one needs its own pair.
V1_THRESHOLDS = {
    ("ERA-Int", 600): (7.43e-7, 2.88e-6),
    ("ERA-Int", 700): (7.16e-7, 2.80e-6),
    ("ERA-Int", 850): (5.95e-7, 2.40e-6),
    ("ERA-40", 600): (7.97e-7, 3.17e-6),
    ("ERA-40", 700): (7.48e-7, 2.96e-6),
    ("ERA-40", 850): (6.16e-7, 2.46e-6),
    ("NCEP", 600): (6.72e-7, 2.75e-6),
    ("NCEP", 700): (6.45e-7, 2.67e-6),
    ("NCEP", 850): (5.34e-7, 2.23e-6),
    ("CFS", 600): (1.08e-6, 5.05e-6),
    ("CFS", 700): (1.09e-6, 5.25e-6),
    ("CFS", 850): (9.28e-7, 4.71e-6),
}


def thresholds_for(reanalysis, level):
    """Version 1's own coarse and fine thresholds, or a refusal.

    THERE IS NO DEFAULT AND NO FALLBACK. A reanalysis version 1 never saw has no entry
    here, and guessing one from a neighbour would produce a record that looks like version
    1's and was detected at a different sensitivity. The percentiles have to be computed
    from that reanalysis's own distribution, which is what `climatology.detection_thresholds`
    is for.
    """
    key = (reanalysis, int(level))
    if key not in V1_THRESHOLDS:
        raise ValueError(
            f"version 1 has no thresholds for {reanalysis!r} at {level} hPa. It hardcodes "
            f"them per reanalysis, as percentiles of that reanalysis's own curvature "
            f"anomaly distribution, so they cannot be borrowed. Compute them with "
            f"climatology.detection_thresholds and pass them explicitly. Known: "
            f"{sorted(set(k[0] for k in V1_THRESHOLDS))}")
    return V1_THRESHOLDS[key]


def latitude_descends(latgrid):
    """True when row 0 is the northernmost, which is how reanalysis files usually come."""
    return float(latgrid[1, 0]) < float(latgrid[0, 0])


def _advection(latgrid, longrid, u, v, anomaly):
    """Advection of the curvature anomaly, reproducing version 1's latitude flip.

    THE FLIP IS NOT DECORATION AND IT IS NOT SYMMETRIC. The driver writes:

        advcurrv_anom(:,end:-1:1,:) = calculate_advvort_f(latgrid(end:-1:1,:), longrid,
            u(:,end:-1:1,:), v(:,end:-1:1,:), currv_anom(:,end:-1:1,:), 0);

    and what it is compensating for is inside `calculate_advvort_f`, which derives its
    latitude spacing as `ll_lat = min(latgrid(:,1))` with `nglat_int` always positive. That
    is an ASSUMPTION THAT LATITUDE ASCENDS WITH ROW INDEX. Version 1's own fields arrive
    the other way round, north first, so the driver reverses them, gets the right answer,
    and reverses the result back.

    Which means the flip GIVES THE RIGHT ANSWER FOR DESCENDING INPUT AND THE WRONG SIGN FOR
    ASCENDING INPUT. It is not a no-op either way. Measured on a northward wind across a
    latitudinal gradient: the correct advection is -5.67e-06, the flip on descending input
    gives -5.67e-06, and the flip on ascending input gives +5.67e-06.

    So the row order has to be established rather than assumed, and this orients to
    descending first. An earlier version took the input order on trust and every test in
    this file, all of which build ascending grids, was quietly running on a sign-flipped
    advection field. It went unnoticed because the trough axis is the ZERO contour of this
    field, and flipping a field's sign does not move its zeros.
    """
    latgrid = np.asarray(latgrid, dtype=float)
    if latitude_descends(latgrid):
        oriented = (latgrid, u, v, anomaly)
        restore = False
    else:
        # Present the fields the way version 1's own data arrives, so its flip does the job
        # it was written to do, then put the result back in the caller's order.
        oriented = (latgrid[::-1, :], u[:, ::-1, :], v[:, ::-1, :], anomaly[:, ::-1, :])
        restore = True
    lat_d, u_d, v_d, anomaly_d = oriented

    flip = slice(None, None, -1)
    out = np.full(np.shape(anomaly_d), np.nan, dtype=float)
    out[:, flip, :] = advection_of_vorticity(
        lat_d[flip, :], longrid, u_d[:, flip, :], v_d[:, flip, :], anomaly_d[:, flip, :])
    return out[:, ::-1, :] if restore else out


def coarse_grid(latgrid, longrid, input_resolution, output_resolution):
    """The decimated coordinate meshes, SUBSAMPLED rather than smoothed.

    decimate_f.m filters the DATA and plain-subsamples the coordinate vectors, and the
    difference is not cosmetic. Running the coordinates through the same filter distorts
    them wherever the convolution has nothing to average against, which on version 1's own
    domain at 0.75 degrees moves a coordinate by up to eleven degrees and leaves the grid
    NOT MONOTONIC: the first three latitudes come out as -24.2, -27.3, -25.4 instead of
    -35.0, -32.75, -30.5. Everything downstream reads positions off this grid.

    Split out of `track_year` so it can be tested at the real domain size, where the defect
    is large, rather than only through a small fixture where it happens not to show.
    """
    _, stride = clim.decimation_shape(input_resolution, output_resolution)
    lat_c, lon_c = np.meshgrid(
        clim.subsample_coordinates(longrid[0, :], stride),
        clim.subsample_coordinates(latgrid[:, 0], stride))[::-1]
    return lat_c, lon_c


def _subset(latgrid, longrid, lat_range, lon_range):
    """Row and column selectors for the published domain, as the driver's `find` does."""
    lat_values = latgrid[:, 0]
    lon_values = longrid[0, :]
    rows = np.flatnonzero((lat_values >= lat_range[0]) & (lat_values <= lat_range[1]))
    cols = np.flatnonzero((lon_values >= lon_range[0]) & (lon_values <= lon_range[1]))
    return rows, cols


def _median_over(field_2d):
    """A callable giving the median of a smoothed field over a wave's own trough cells.

    Uses `nanmedian`, which is what version 1 uses and is not a defensive habit: the
    smoother propagates NaN on purpose, so a plain median would return NaN for any wave
    touching masked ground and end its track without saying so.
    """
    def median(wave):
        mask = wave.get("region")
        if mask is None:
            return float("nan")
        values = field_2d[np.asarray(mask, dtype=bool)]
        if values.size == 0:
            return float("nan")
        with np.errstate(invalid="ignore"):
            return float(np.nanmedian(values))
    return median


def track_year(times, latgrid, longrid, u, v, curvature_anomaly,
               coarse_threshold, fine_threshold,
               coarse_resolution_deg=COARSE_RESOLUTION_DEG,
               lat_range=DOMAIN_LAT, lon_range=DOMAIN_LON,
               exclusive=False, progress=None):
    """One year of tracking, from prepared fine-grid fields to finished tracks.

    `u`, `v` and `curvature_anomaly` are (time, lat, lon) on the native grid, already
    anomalies where that applies. `times` are in days, the units the record stores.

    Returns the finished tracks. `exclusive=True` applies the duplication repair; the
    default reproduces version 1.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    anomaly = np.asarray(curvature_anomaly, dtype=float)
    times = np.asarray(times, dtype=float)
    latgrid = np.asarray(latgrid, dtype=float)
    longrid = np.asarray(longrid, dtype=float)

    advection = _advection(latgrid, longrid, u, v, anomaly)

    native_deg = abs(float(latgrid[1, 0] - latgrid[0, 0]))
    if coarse_resolution_deg < native_deg:
        raise ValueError(
            f"the coarse resolution {coarse_resolution_deg} is finer than the native grid "
            f"{native_deg}; version 1 decimates and cannot refine")
    coarse = {name: clim.gaussian_decimate(field, native_deg, coarse_resolution_deg)
              for name, field in (("u", u), ("v", v), ("anomaly", anomaly),
                                  ("advection", advection))}
    lat_c, lon_c = coarse_grid(latgrid, longrid, native_deg, coarse_resolution_deg)

    rows_f, cols_f = _subset(latgrid, longrid, lat_range, lon_range)
    rows_c, cols_c = _subset(lat_c, lon_c, lat_range, lon_range)
    if rows_c.size < 3 or cols_c.size < 3:
        raise ValueError("the coarse grid has fewer than three rows or columns inside the "
                         "domain, so nothing can be contoured on it")

    def cut_fine(field):
        return field[:, rows_f, :][:, :, cols_f]

    def cut_coarse(field):
        return field[:, rows_c, :][:, :, cols_c]

    latgrid_f = latgrid[np.ix_(rows_f, cols_f)]
    longrid_f = longrid[np.ix_(rows_f, cols_f)]
    latgrid_c = lat_c[np.ix_(rows_c, cols_c)]
    longrid_c = lon_c[np.ix_(rows_c, cols_c)]
    u_f, anomaly_f = cut_fine(u), cut_fine(anomaly)
    v_f = cut_fine(v)
    u_c, anomaly_c = cut_coarse(coarse["u"]), cut_coarse(coarse["anomaly"])
    advection_c = cut_coarse(coarse["advection"])

    tracks, states = [], []
    for step in range(times.size):
        waves = detect_troughs(
            float(times[step]), latgrid_c, longrid_c, u_c[step],
            anomaly_c[step], advection_c[step],
            latgrid_f, longrid_f, anomaly_f[step],
            coarse_threshold=coarse_threshold, fine_threshold=fine_threshold)
        # The smoothed fine winds, computed once per timestep as version 1 does, and used
        # only for the median that predicts where each wave goes next.
        u_median = _median_over(clim.smooth9(u_f[step]))
        v_median = _median_over(clim.smooth9(v_f[step]))
        tracks, states = associate_step(tracks, states, waves, step,
                                        u_median, v_median, exclusive=exclusive)
        tracks, states = prune_stale_tracks(tracks, states, step, float(times[step]),
                                            waves)
        if progress is not None:
            progress(step, times.size, len(tracks), len(waves))

    return finalize_tracks(tracks, total_steps=times.size)


RECORD_EPOCH = np.datetime64("1900-01-01T00:00:00", "ns")


def days_to_datetime64(days):
    """Days since 1900, the record's own time convention, as datetime64.

    The climatology works in calendar positions and therefore needs real dates, while every
    other stage and the record itself carry days since 1900 as a float. Converting in one
    named place keeps the two from being confused, which they were: a first version handed
    floats straight to `curvature_climatology`, where numpy read them as nanoseconds since
    1970 and put the whole record in January 1970.
    """
    days = np.asarray(days, dtype=float)
    return RECORD_EPOCH + (days * 86400.0 * 1e9).astype("timedelta64[ns]")


def climatology_from_years(curvature_by_year, times_by_year, expected_years=None):
    """The mean curvature field per calendar timestep, across the given years.

    Version 1 computes this once over 1981 to 2010 and every year's anomaly is a departure
    from it, so the window is part of the method and not a preference.

    `times_by_year` are in days since 1900, matching every other function here. Returns the
    dict `climatology.curvature_climatology` produces, which `curvature_anomaly` consumes.
    """
    curvature = np.concatenate([np.asarray(c, dtype=float) for c in curvature_by_year])
    days = np.concatenate([np.asarray(t, dtype=float) for t in times_by_year])
    if expected_years is None:
        expected_years = len(curvature_by_year)
    return clim.curvature_climatology(curvature, days_to_datetime64(days),
                                      expected_years=expected_years)


def anomaly_from_climatology(curvature, times_days, climatology):
    """Departure from the climatological mean for each timestep's calendar position."""
    return clim.curvature_anomaly(curvature, days_to_datetime64(times_days), climatology)


def curvature_from_winds(latgrid, longrid, u, v):
    """Curvature vorticity for a (time, lat, lon) wind stack, one timestep at a time.

    ORIENTS TO ASCENDING FIRST, reproducing version 1's driver. calculate_compvort_multi_f
    indexes its meridional neighbours as `s = g+1` and `t = g-1` and divides by a spacing
    that is always positive, so it assumes latitude ascends with row index, and
    p1_data_eraint.m flips latitude, the winds and the result around the call:

        [rv(:,end:-1:1,:),shrrv(:,end:-1:1,:),currv(:,end:-1:1,:)] =
            calculate_compvort_multi_f(latgrid(end:-1:1,:),longrid,
                squeeze(u(:,end:-1:1,:)),squeeze(v(:,end:-1:1,:)),0);

    This is the same assumption `_advection` orients for, in the function one stage
    earlier, and an earlier version of THIS function passed the file's own row order
    straight through. On the real north-first files that put the wrong sign on every
    meridional derivative: measured on the 1990-07-15T12 timestep, the misoriented
    curvature correlated 0.03 with the correctly oriented field while keeping a similar
    magnitude distribution. The percentile checks run on that field showed nothing
    anomalous, while only a fifth of version 1's observations had a port detection
    within 500 km. Found by
    working a single timestep against version 1's published positions after every
    aggregate comparison had looked reasonable stage by stage.
    """
    latgrid = np.asarray(latgrid, dtype=float)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    if u.ndim != 3 or v.ndim != 3:
        # Refuse both orientations the same way. Without this, ascending 2-D input died
        # in component_vorticity's shape check while descending 2-D input hit an
        # IndexError in the flip above it, so the failure depended on row order.
        raise ValueError("curvature_from_winds takes (time, lat, lon) stacks; got "
                         "u with %d and v with %d dimensions" % (u.ndim, v.ndim))
    if latitude_descends(latgrid):
        oriented = curvature_from_winds(latgrid[::-1, :], longrid,
                                        u[:, ::-1, :], v[:, ::-1, :])
        return oriented[:, ::-1, :]
    out = np.empty(np.shape(u), dtype=float)
    for t in range(np.shape(u)[0]):
        _, _, curvature = component_vorticity(latgrid, longrid, u[t], v[t])
        out[t] = curvature
    return out
