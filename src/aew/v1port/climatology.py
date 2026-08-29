"""The version 1 curvature-vorticity climatology, ported from p1_data_eraint_climate.m.

WHAT THIS IS, because the name is misleading. The version 1 "climatology" is NOT a
percentile and not a single number. It is a mean field indexed by CALENDAR TIMESTEP: for
each of the 1464 six-hourly steps in a leap year, the average curvature vorticity across
the climatology years, at every grid point. Its purpose is to be subtracted, giving the
curvature vorticity ANOMALY that the tracker actually detects on.

The scalar detection thresholds are a separate quantity. `find_ews_f.m` masks with

    id = find(crvt < curv_thr_c);      % coarse grid, 55th percentile
    id = find(crvt2 < curv_thr_f);     % fine grid, 66th percentile

and its comment names them "Curvature Vorticity Anomalies", so the percentiles are of the
SMOOTHED, SIGN-ADJUSTED ANOMALY, not of raw curvature vorticity. `anomaly_thresholds`
below reproduces that recipe. Getting this wrong is the easiest way to produce a v2 that
detects a different population than v1 did, which is why it is spelled out here.

Ported from:
    p1_data_eraint_climate.m    the mean field and its leap-day treatment
    smth9_f.m                   the nine-point smoother
    find_ews_f.m                the sign convention and the percentile definition
"""

import numpy as np

# p1_data_eraint_climate.m warns when a calendar step does not have exactly one sample
# per climatology year, which is its check that the input is complete.
LEAP_DAY_MONTH, LEAP_DAY_DAY = 2, 29

# find_ews_f.m: coarse mask at the 55th percentile, fine mask at the 66th.
COARSE_PERCENTILE = 55.0
FINE_PERCENTILE = 66.0

# smth9_f.m is always called as smth9_f(field, 0.5, 0.25, nan).
SMOOTH_P, SMOOTH_Q = 0.5, 0.25


def smooth9(field, p=SMOOTH_P, q=SMOOTH_Q):
    """Nine-point smoother, ported from smth9_f.m.

        f0 <- f0 + (p/4)(f2+f4+f6+f8 - 4 f0) + (q/4)(f1+f3+f5+f7 - 4 f0)

    with the four edge-adjacent neighbours in the p term and the four diagonals in the q
    term. The outermost ring is returned unchanged, exactly as the MATLAB loop leaves it.

    FAITHFUL: the original takes a `missing` argument and guards each window with
    `sum(temp == missing) == 0`. It is always called with `missing = nan`, and in MATLAB
    `NaN == NaN` is false, so that sum is always zero and the guard always passes. The
    missing-value handling is therefore dead code and NaN propagates through the window.
    This reproduces that behavior rather than the behavior the argument implies.
    """
    field = np.asarray(field, dtype=float)
    if field.ndim != 2:
        raise ValueError("smooth9 expects a 2-D field, got %d dimensions" % field.ndim)
    out = field.copy()
    if field.shape[0] < 3 or field.shape[1] < 3:
        return out

    c = field[1:-1, 1:-1]
    north, south = field[:-2, 1:-1], field[2:, 1:-1]
    west, east = field[1:-1, :-2], field[1:-1, 2:]
    nw, ne = field[:-2, :-2], field[:-2, 2:]
    sw, se = field[2:, :-2], field[2:, 2:]

    out[1:-1, 1:-1] = (c
                       + (p / 4.0) * (north + west + south + east - 4.0 * c)
                       + (q / 4.0) * (ne + nw + sw + se - 4.0 * c))
    return out


def calendar_key(times):
    """Month, day and hour of each timestamp, which is what the original groups on.

    p1_data_eraint_climate.m does this by reformatting every timestamp into a common
    year, so two dates in different years with the same month, day and hour land in the
    same bin.
    """
    times = np.asarray(times, dtype="datetime64[ns]")
    months = times.astype("datetime64[M]").astype(int) % 12 + 1
    days = ((times.astype("datetime64[D]") - times.astype("datetime64[M]"))
            .astype(int) + 1)
    hours = ((times.astype("datetime64[h]") - times.astype("datetime64[D]"))
             .astype(int))
    return months, days, hours


def curvature_climatology(curvature, times, expected_years=None,
                          interpolate_leap_day=True):
    """Mean curvature vorticity per calendar timestep, from p1_data_eraint_climate.m.

    Parameters
    ----------
    curvature : array, shape (ntime, nlat, nlon)
        Curvature vorticity, already computed on the working grid.
    times : array of datetime64
        Timestamps matching the first axis.
    expected_years : int, optional
        How many samples each calendar step should have. When given, steps that do not
        have exactly this many are reported, which is the original's completeness check.
    interpolate_leap_day : bool
        Reproduce the original's treatment of 29 February. See below.

    Returns
    -------
    dict with `keys` (month, day, hour tuples, sorted), `mean` (nkeys, nlat, nlon),
    `counts` (nkeys,), and `short_steps`, the calendar steps whose sample count differed
    from `expected_years`.

    FAITHFUL, and this one is a deliberate choice rather than an accident. The original
    overwrites the four 29 February steps by linear interpolation between the last step
    of 28 February and the first step of 1 March:

        currv_clim(237:240,:,:) = interp1([0;5],currv_clim([236;241],:,:),[1;2;3;4]);

    It does that because a thirty-year window holds only seven or eight 29 Februaries, so
    the leap-day mean is far noisier than every other step. Reproduced here.
    """
    curvature = np.asarray(curvature, dtype=float)
    if curvature.ndim != 3:
        raise ValueError("curvature must be (time, lat, lon), got %d dimensions"
                         % curvature.ndim)
    times = np.asarray(times, dtype="datetime64[ns]")
    if len(times) != curvature.shape[0]:
        raise ValueError("times has %d entries but curvature has %d timesteps"
                         % (len(times), curvature.shape[0]))
    if len(times) == 0:
        raise ValueError("no timesteps given, so no climatology can be formed")

    months, days, hours = calendar_key(times)
    keys = list(zip(months.tolist(), days.tolist(), hours.tolist()))
    unique_keys = sorted(set(keys))
    index = {k: i for i, k in enumerate(unique_keys)}

    total = np.zeros((len(unique_keys),) + curvature.shape[1:])
    counts = np.zeros(len(unique_keys), dtype=int)
    for position, key in enumerate(keys):
        i = index[key]
        total[i] += np.nan_to_num(curvature[position], nan=0.0)
        counts[i] += 1

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / counts[:, None, None]

    short_steps = []
    if expected_years is not None:
        for k, i in index.items():
            if counts[i] != expected_years and (k[0], k[1]) != (LEAP_DAY_MONTH,
                                                                LEAP_DAY_DAY):
                short_steps.append((k, int(counts[i])))

    if interpolate_leap_day:
        mean = _interpolate_leap_day(mean, unique_keys)

    return {"keys": unique_keys, "mean": mean, "counts": counts,
            "short_steps": sorted(short_steps)}


def _interpolate_leap_day(mean, keys):
    """Replace the 29 February steps by interpolation across the gap, as the original."""
    leap = [i for i, k in enumerate(keys) if (k[0], k[1]) == (LEAP_DAY_MONTH,
                                                              LEAP_DAY_DAY)]
    if not leap:
        return mean
    before = min(leap) - 1
    after = max(leap) + 1
    if before < 0 or after >= len(keys):
        return mean
    mean = mean.copy()
    span = after - before
    for offset, i in enumerate(sorted(leap), start=1):
        weight = offset / span
        mean[i] = (1.0 - weight) * mean[before] + weight * mean[after]
    return mean


def curvature_anomaly(curvature, times, climatology):
    """Subtract the climatological mean for each timestep's calendar position."""
    curvature = np.asarray(curvature, dtype=float)
    months, days, hours = calendar_key(times)
    index = {k: i for i, k in enumerate(climatology["keys"])}
    out = np.empty_like(curvature)
    for position, key in enumerate(zip(months.tolist(), days.tolist(), hours.tolist())):
        if key not in index:
            raise ValueError("no climatological value for calendar step %s; the "
                             "climatology does not cover this field" % (key,))
        out[position] = curvature[position] - climatology["mean"][index[key]]
    return out


def southern_hemisphere_sign(field, lat_values):
    """Negate the field south of the equator, as find_ews_f.m does before thresholding.

    The original comment is "Set Southern Hemisphere Curvature Vorticity Anomalies > 0
    (where < 0) for tracking", which makes a cyclonic anomaly positive in both
    hemispheres so that one threshold serves both.
    """
    field = np.asarray(field, dtype=float)
    lat_values = np.asarray(lat_values, dtype=float)
    if field.shape[-2] != lat_values.size:
        raise ValueError("latitude axis is %d long but %d latitudes were given"
                         % (field.shape[-2], lat_values.size))
    out = field.copy()
    south = lat_values < 0
    out[..., south, :] *= -1.0
    return out


def decimation_shape(input_resolution, output_resolution):
    """The filter width and the subsampling stride, which are NOT the same number.

    decimate_f.m computes them separately and they differ whenever the target resolution
    is an exact multiple of the input:

        if floor(reso/resi) == reso/resi;  num = reso/resi + 1;  else  num = floor(reso/resi);
        ...
        xo = xi(1:floor(reso/resi):end);

    So the stride is always `floor(reso/resi)`, while the FILTER is one wider than that
    when the ratio divides exactly. An earlier version of this function took a single
    factor and derived the width from it as `factor + 1` always, which is right for the
    exact-multiple case and wrong otherwise. It went unnoticed because every test used an
    exact multiple; the first non-exact ratio in real use is ERA-Interim's own, 2.5 over
    0.75, where the width should be 3 and that version used 4.

    Returns (width, stride).
    """
    ratio = output_resolution / input_resolution
    stride = int(np.floor(ratio))
    if stride < 1:
        raise ValueError(
            "the output resolution %r is finer than the input %r; decimate_f coarsens and "
            "has no path that refines" % (output_resolution, input_resolution))
    width = stride + 1 if np.isclose(np.floor(ratio), ratio) else stride
    return width, stride


def gaussian_kernel(width):
    """`fspecial('gaussian', width, width/5)`, normalized, as decimate_f.m builds it."""
    sigma = width / 5.0
    axis = np.arange(width) - (width - 1) / 2.0
    gx, gy = np.meshgrid(axis, axis)
    kernel = np.exp(-(gx ** 2 + gy ** 2) / (2.0 * sigma ** 2))
    return kernel / kernel.sum()


def subsample_coordinates(values, stride):
    """Coarse coordinates, taken by STRIDE and never smoothed.

    decimate_f.m filters the DATA and plain-subsamples the coordinate vectors. Running a
    coordinate array through the smoother instead distorts it at the edges, where the
    convolution has nothing to average against, and the resulting grid is not monotonic.
    A first version of the pipeline did exactly that and the spacing came out negative.
    """
    return np.asarray(values)[::stride]


def gaussian_decimate(field, input_resolution, output_resolution):
    """Coarsen by Gaussian smoothing then subsampling, ported from decimate_f.m.

    Takes the two RESOLUTIONS rather than a factor, because the filter width and the
    subsampling stride are computed differently from them and cannot be recovered from
    each other. See `decimation_shape`.

    This is one of the three toolbox calls in the version 1 source. The kernel is a
    normalized isotropic Gaussian, which numpy reproduces exactly.
    """
    field = np.asarray(field, dtype=float)
    if field.ndim == 2:
        field = field[np.newaxis, ...]
        squeeze = True
    else:
        squeeze = False
    width, stride = decimation_shape(input_resolution, output_resolution)

    from scipy.signal import convolve2d
    kernel = gaussian_kernel(width)
    smoothed = np.stack([convolve2d(step, kernel, mode="same") for step in field])
    out = smoothed[:, ::stride, ::stride]
    return (out[0] if squeeze else out)


def anomaly_threshold(anomaly, lat_values, percentile, smooth=True):
    """One detection threshold, as a percentile of the field it actually gates.

    THE FIELD MATTERS AS MUCH AS THE PERCENTILE, and this is deliberately a single-field
    function for that reason. In find_ews_f.m the two masks compare against two DIFFERENT
    fields:

        id = find(crvt  < curv_thr_c);   % crvt  is the anomaly on the COARSE 2.5 deg grid
        id = find(crvt2 < curv_thr_f);   % crvt2 is the anomaly on the FINE input grid

    so the 55th percentile belongs to the decimated field and the 66th to the fine one.
    An earlier version of this module took both percentiles from a single array, which
    would have produced a coarse threshold calibrated on the wrong distribution. Compute
    the coarse field with `gaussian_decimate` first and pass it here separately.

    The recipe applied here is the rest of what find_ews_f.m does before comparing:
    smooth with the nine-point smoother, then negate south of the equator.
    """
    anomaly = np.asarray(anomaly, dtype=float)
    if anomaly.ndim == 2:
        anomaly = anomaly[np.newaxis, ...]
    if smooth:
        anomaly = np.stack([smooth9(step) for step in anomaly])
    adjusted = southern_hemisphere_sign(anomaly, lat_values)
    sample = adjusted[np.isfinite(adjusted)]
    if sample.size == 0:
        raise ValueError("no finite anomaly values, so no threshold can be taken")
    return float(np.percentile(sample, percentile))


def detection_thresholds(anomaly_fine, lat_fine, decimation_factor):
    """Both version 1 thresholds, each taken from the grid its mask compares against.

    Returns ``(coarse, fine)``: the coarse value is the 55th percentile of the decimated
    anomaly and the fine value is the 66th percentile of the input-resolution anomaly.
    """
    coarse_field = gaussian_decimate(anomaly_fine, 1.0, decimation_factor)
    lat_coarse = np.asarray(lat_fine, dtype=float)[::decimation_factor]
    coarse = anomaly_threshold(coarse_field, lat_coarse, COARSE_PERCENTILE)
    fine = anomaly_threshold(anomaly_fine, lat_fine, FINE_PERCENTILE)
    return coarse, fine
