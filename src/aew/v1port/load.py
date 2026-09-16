"""Read retrieved reanalysis files into what the tracker takes.

The gap between `scripts/download_eraint_v1port.py` and `pipeline.track_year`. Three jobs,
and each is somewhere a silent error could live.

TIME. The files carry `valid_time` in seconds since 1970 and every stage of the port works
in days since 1900, which is what the record itself stores. The offset is 25567 days. Get
it wrong and nothing complains: the fields are fine, the tracks are fine, and the record
carries timestamps decades out with the right units attribute.

ORIENTATION. These files come north-first. The tracker handles either order now, but only
because that was found and fixed; it is checked here rather than assumed, because a file
that arrived south-first would take a different path through the advection.

THE CLIMATOLOGY, which is why this module exists at all rather than being three lines in a
script. Version 1's anomaly is a departure from a mean field computed per calendar timestep
across 1981 to 2010. Thirty years of curvature at this resolution is about eight gigabytes
if it is concatenated first, so it is ACCUMULATED YEAR BY YEAR instead. That is an
optimisation with a correctness obligation attached: the streaming result has to equal what
`climatology.curvature_climatology` gives for the same input, including its treatment of
missing values and of 29 February, and `test_v1port_load.py` holds it to that on a case
small enough to compute both ways.
"""

import glob
import os

import numpy as np

from . import climatology as clim

# datetime.date(1970, 1, 1).toordinal() - datetime.date(1900, 1, 1).toordinal()
EPOCH_OFFSET_DAYS = 25567
SECONDS_PER_DAY = 86400.0


def seconds_since_1970_to_days_since_1900(seconds):
    """The file's time convention to the record's. See the module docstring."""
    return np.asarray(seconds, dtype=float) / SECONDS_PER_DAY + EPOCH_OFFSET_DAYS


def _time_days(dataset):
    for name in ("valid_time", "time"):
        if name not in dataset.variables:
            continue
        variable = dataset.variables[name]
        values = np.asarray(variable[:], dtype=float)
        units = str(getattr(variable, "units", ""))
        if "since 1970" in units and units.startswith("seconds"):
            return seconds_since_1970_to_days_since_1900(values)
        if "since 1900" in units and units.startswith("days"):
            return values
        raise ValueError(
            f"the time variable {name!r} has units {units!r}, which this loader does not "
            f"convert. Every stage downstream works in days since 1900 and a wrong "
            f"conversion is invisible, so add the case explicitly rather than guessing.")
    raise ValueError("no time variable found; expected 'valid_time' or 'time'")


def _field(dataset, names):
    for name in names:
        if name in dataset.variables:
            data = np.asarray(dataset.variables[name][:], dtype=float)
            # The retrieval asks for one pressure level, which arrives as a length-one
            # axis rather than being dropped.
            while data.ndim > 3:
                axis = next(i for i, n in enumerate(data.shape[1:], start=1) if n == 1)
                data = np.squeeze(data, axis=axis)
            return data
    raise ValueError(f"none of {names} in the file; found {list(dataset.variables)}")


def load_year(year, directory="data/eraint/v1port", prefix="eraint"):
    """One year of winds, as (times_days, latgrid, longrid, u, v).

    Latitude and longitude come back as full 2-D meshes because that is what every stage
    of the port takes, and in the file's own row order, which the tracker handles.
    """
    import netCDF4 as nc

    paths = {}
    for var in ("u700", "v700"):
        path = os.path.join(directory, f"{prefix}_{var}_{year}_6h_region.nc")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} is missing. The retrieval writes one file per variable-year and "
                f"skips what is already there, so re-running it fills only the gaps.")
        paths[var] = path

    with nc.Dataset(paths["u700"]) as ds:
        times = _time_days(ds)
        lat = np.asarray(ds.variables["latitude"][:], dtype=float)
        lon = np.asarray(ds.variables["longitude"][:], dtype=float)
        u = _field(ds, ("u", "u700"))
    with nc.Dataset(paths["v700"]) as ds:
        v_times = _time_days(ds)
        v = _field(ds, ("v", "v700"))

    # ABSOLUTE TOLERANCE, and rtol=0 on purpose. Days since 1900 are numbers around 30,000,
    # so numpy's default relative tolerance of 1e-5 is about 0.3 days here: a u and v file
    # six hours out of step would compare as equal. A first version used the default and
    # this check silently passed. One second is far tighter than the six-hour spacing and
    # far looser than the file's own precision.
    one_second = 1.0 / 86400.0
    if (v_times.shape != times.shape
            or not np.allclose(v_times, times, rtol=0.0, atol=one_second)):
        raise ValueError(
            f"the u and v files for {year} do not share a time axis, so pairing them by "
            f"position would silently combine different timesteps")
    if u.shape != v.shape:
        raise ValueError(f"u has shape {u.shape} and v has {v.shape} for {year}")

    longrid, latgrid = np.meshgrid(lon, lat)
    return times, latgrid, longrid, u, v


def available_years(directory="data/eraint/v1port", prefix="eraint"):
    """Years for which BOTH wind files are present, so a partial retrieval is usable."""
    years = set()
    for path in glob.glob(os.path.join(directory, f"{prefix}_u700_*_6h_region.nc")):
        year = os.path.basename(path).split("_")[2]
        if os.path.exists(os.path.join(directory,
                                       f"{prefix}_v700_{year}_6h_region.nc")):
            years.add(int(year))
    return sorted(years)


class ClimatologyAccumulator:
    """The climatological mean per calendar timestep, built one year at a time.

    Equivalent to `climatology.curvature_climatology` over the concatenation of everything
    added, and the test suite holds it to that rather than to a description of it. The
    equivalence is not obvious: missing values are summed as zero while still counting
    toward the divisor, which is what the batch version does, and the 29 February steps are
    interpolated across only at the end, when the full set of calendar keys is known.
    """

    def __init__(self):
        self._total = {}
        self._counts = {}
        self._shape = None

    def add(self, curvature, times_days):
        """Add one year. `times_days` are days since 1900."""
        curvature = np.asarray(curvature, dtype=float)
        if curvature.ndim != 3:
            raise ValueError("curvature must be (time, lat, lon), got %d dimensions"
                             % curvature.ndim)
        if len(times_days) != curvature.shape[0]:
            raise ValueError("times has %d entries but curvature has %d timesteps"
                             % (len(times_days), curvature.shape[0]))
        if self._shape is None:
            self._shape = curvature.shape[1:]
        elif curvature.shape[1:] != self._shape:
            raise ValueError(
                f"this year is on a {curvature.shape[1:]} grid and the accumulator holds "
                f"{self._shape}; a climatology cannot mix grids")

        from .pipeline import days_to_datetime64
        months, days, hours = clim.calendar_key(days_to_datetime64(times_days))
        for position, key in enumerate(zip(months.tolist(), days.tolist(),
                                           hours.tolist())):
            if key not in self._total:
                self._total[key] = np.zeros(self._shape)
                self._counts[key] = 0
            self._total[key] += np.nan_to_num(curvature[position], nan=0.0)
            self._counts[key] += 1
        return self

    def finalize(self, expected_years=None, interpolate_leap_day=True):
        """The same dict `curvature_climatology` returns."""
        if self._shape is None:
            raise ValueError("nothing was added, so no climatology can be formed")
        keys = sorted(self._total)
        counts = np.array([self._counts[k] for k in keys], dtype=int)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.stack([self._total[k] for k in keys]) / counts[:, None, None]

        short_steps = []
        if expected_years is not None:
            for k in keys:
                if (self._counts[k] != expected_years
                        and (k[0], k[1]) != (clim.LEAP_DAY_MONTH, clim.LEAP_DAY_DAY)):
                    short_steps.append((k, int(self._counts[k])))
        if interpolate_leap_day:
            mean = clim._interpolate_leap_day(mean, keys)
        return {"keys": keys, "mean": mean, "counts": counts,
                "short_steps": sorted(short_steps)}


def curvature_for_year(year, directory="data/eraint/v1port", prefix="eraint",
                       subsample=1):
    """Curvature vorticity for one year, with its times and grid.

    `subsample` takes every Nth grid point, and IT IS APPLIED TO THE WINDS, before the
    derivative rather than after it. That ordering is the whole point of the parameter
    living here instead of in the caller. Curvature vorticity is built from spatial
    derivatives, so striding a curvature field computed at half a degree gives a different
    quantity from the curvature a whole-degree wind field would have produced: measured on
    eight real ERA5 timesteps the two disagree at every finite cell, by 45.8 percent of the
    field's own root-mean-square. Striding the winds is what a coarser retrieval request
    would have done, so it is the operation this reproduces.

    An earlier arrangement had the caller stride the returned curvature, which was both the
    wrong operation and a crash: the climatology was still built on the full grid, so the
    anomaly subtraction met mismatched shapes.
    """
    from .pipeline import curvature_from_winds

    times, latgrid, longrid, u, v = load_year(year, directory, prefix)
    latgrid, longrid, u, v = _stride(subsample, latgrid, longrid, u, v)
    return times, latgrid, longrid, curvature_from_winds(latgrid, longrid, u, v)


def _stride(subsample, latgrid, longrid, u, v):
    """Every Nth point of a wind field and its grid, refusing a stride that cannot work."""
    subsample = int(subsample)
    if subsample < 1:
        raise ValueError(f"subsample must be at least 1, got {subsample}")
    if subsample == 1:
        return latgrid, longrid, u, v
    # A stride that leaves too few rows to differentiate produces a field the tracker
    # cannot use, and it would otherwise surface much later as an empty or degenerate
    # result rather than here where the cause is legible.
    if min(latgrid.shape[0], longrid.shape[1]) < 3 * subsample:
        raise ValueError(
            f"a stride of {subsample} leaves fewer than three points on a grid of "
            f"{latgrid.shape}, which cannot carry a spatial derivative")
    return (latgrid[::subsample, ::subsample], longrid[::subsample, ::subsample],
            u[:, ::subsample, ::subsample], v[:, ::subsample, ::subsample])


def build_climatology(years, directory="data/eraint/v1port", prefix="eraint",
                      progress=None, subsample=1):
    """The 1981-2010 style climatological mean, accumulated year by year.

    `progress` is called with (year, index, total) after each year, because this reads and
    differentiates thirty years of wind and a silent hour is hard to tell from a hang.

    `subsample` is passed straight to `curvature_for_year`, so THE CLIMATOLOGY IS BUILT ON
    THE SAME GRID the yearly fields will be sampled on. A caller that strides one and not
    the other gets shapes that cannot be subtracted, which is how this was found.
    """
    accumulator = ClimatologyAccumulator()
    years = list(years)
    for n, year in enumerate(years, start=1):
        times, _, _, curvature = curvature_for_year(year, directory, prefix, subsample)
        accumulator.add(curvature, times)
        del curvature
        if progress is not None:
            progress(year, n, len(years))
    return accumulator.finalize(expected_years=len(years))


def sample_anomaly_values(curvature, times_days, climatology, lat_values,
                          decimation_factor=None, native_resolution=None,
                          coarse_resolution=None, per_step=400, rng=None,
                          lon_values=None, lat_range=None, lon_range=None):
    """A random sample of the anomaly values a threshold would be taken over.

    THE SPATIAL BOX IS AN ARGUMENT, not a fixed whole-domain draw, so that the region
    travels with any number computed here instead of being implied by a default. The
    archive DOES record the sample, in src_readme.docx, as the full period of each
    reanalysis over the tracking domain, and running that sample does not reproduce the
    published pair. An African box reproduces the fine threshold more closely, but it was
    chosen because it did so, which is selecting the sample on the answer, and it is
    retired as a justification. The project's threshold notes record that finding.

    `lat_range` and `lon_range` are inclusive (low, high) pairs in degrees. Passing either
    requires `lon_values`, since a longitude box cannot be applied to a latitude axis
    alone. Omitting both samples the whole retrieved grid, which is what earlier runs did.

    WHY A SAMPLE. The thresholds are percentiles of the anomaly over the whole climatology
    period, and thirty years of it at ERA5's resolution is about eighteen gigabytes. A
    percentile of a large random sample estimates the percentile of the population, and
    unlike a per-year percentile averaged afterwards it estimates the right quantity.
    `threshold_sampling_error` reports a within-sample half-sample range beside it, which
    is a sensitivity diagnostic and NOT a measure of what that sampling costs, because it
    cannot see variation across this draw. Several seeds measure that, and a `per_step` at or above the
    finite-cell count removes it.

    Returns (coarse_sample, fine_sample), the smoothed and sign-adjusted values from each
    grid, which is what `climatology.anomaly_threshold` takes its percentile of.
    """
    from .climatology import (anomaly_threshold, gaussian_decimate,  # noqa: F401
                              smooth9, southern_hemisphere_sign)

    rng = rng if rng is not None else np.random.default_rng(0)
    anomaly = clim.curvature_anomaly(curvature, _as_datetime(times_days), climatology)
    if (lat_range is not None or lon_range is not None) and lon_values is None:
        raise ValueError("a spatial box needs lon_values, not just the latitude axis")

    def box_indices(lats, lons):
        """Rows and columns inside the box, as index arrays, or None for the whole grid.

        SLICE RATHER THAN MASK, because that is the order the tracker uses and the two are
        NOT equivalent. `p2_track_eraint_700hPa.m` decimates the full field, then crops
        both grids to lat1/lat2 and lon1/lon2 under "Parse Data to Final Lat/Lon Domain",
        and passes the CROPPED arrays into `find_ews_f.m`, which smooths them at lines 84
        to 91. So the smoother meets the tracking domain's edge as a boundary.

        THE DIFFERENCE IS THE WHOLE PERIMETER, not a rounding detail. `smth9_f` loops from
        2 to size-1 and starts from `out = x`, so it leaves the outer ring of whatever it
        is handed untouched. Cropping first keeps the tracking domain's edge cells raw.
        Smoothing a buffered field and masking afterwards smooths those same cells using
        halo neighbours the archived smoother never sees.

        An earlier version of this function masked after smoothing and its comment claimed
        that doing so gave "what the tracker would compute at the same place". A review
        checked that against the source and it was false. The comment asserted a fidelity
        the code did not have, which is the defect shape this project keeps finding.
        """
        if lat_range is None and lon_range is None:
            return None, None
        lats = np.asarray(lats)
        rows = np.arange(lats.size)
        if lat_range is not None:
            rows = rows[(lats >= lat_range[0]) & (lats <= lat_range[1])]
        cols = None
        if lons is not None:
            lons = np.asarray(lons)
            cols = np.arange(lons.size)
            if lon_range is not None:
                cols = cols[(lons >= lon_range[0]) & (lons <= lon_range[1])]
        return rows, cols

    def crop(field, rows, cols):
        if rows is None:
            return field
        return field[np.ix_(rows, cols if cols is not None
                            else np.arange(field.shape[1]))]

    fine_rows, fine_cols = box_indices(lat_values, lon_values)
    coarse_rows = coarse_cols = None
    coarse_ready = False
    coarse_values, fine_values = [], []
    for step in range(anomaly.shape[0]):
        cropped = crop(anomaly[step], fine_rows, fine_cols)
        fine_lats = (np.asarray(lat_values) if fine_rows is None
                     else np.asarray(lat_values)[fine_rows])
        fine = southern_hemisphere_sign(smooth9(cropped)[np.newaxis, ...], fine_lats)[0]
        fine_values.append(_draw(fine, per_step, rng))
        if native_resolution is not None and coarse_resolution is not None:
            # DECIMATE THE FULL FIELD FIRST, then crop, then smooth. The archive decimates
            # before it parses to the final domain, so the Gaussian sees the halo and the
            # nine-point smoother does not.
            coarse_field = gaussian_decimate(anomaly[step], native_resolution,
                                             coarse_resolution)
            _, stride = clim.decimation_shape(native_resolution, coarse_resolution)
            coarse_lat_axis = np.asarray(lat_values)[::stride]
            if not coarse_ready:
                coarse_rows, coarse_cols = box_indices(
                    coarse_lat_axis,
                    None if lon_values is None else np.asarray(lon_values)[::stride])
                coarse_ready = True
            coarse_cropped = crop(coarse_field, coarse_rows, coarse_cols)
            coarse_lats = (coarse_lat_axis if coarse_rows is None
                           else coarse_lat_axis[coarse_rows])
            coarse = southern_hemisphere_sign(
                smooth9(coarse_cropped)[np.newaxis, ...], coarse_lats)[0]
            coarse_values.append(_draw(coarse, per_step, rng))
    return (np.concatenate(coarse_values) if coarse_values else np.array([]),
            np.concatenate(fine_values))


def _as_datetime(times_days):
    from .pipeline import days_to_datetime64
    return days_to_datetime64(times_days)


def _draw(field, count, rng):
    """`count` finite values drawn without replacement, or all of them if there are fewer."""
    flat = field[np.isfinite(field)]
    if flat.size == 0:
        return flat
    if flat.size <= count:
        return flat
    return flat[rng.choice(flat.size, size=count, replace=False)]


def threshold_sampling_error(sample, percentile, repeats=25, rng=None):
    """How much the percentile moves when THIS sample is re-halved, as a fractional spread.

    THE POINT IS TO REFUSE TO GUESS. A threshold estimated from a sample is only usable if
    it is stable against how the sample was taken, and this measures one narrow part of
    that. Returns (estimate,
    relative_spread) where the spread is the full range across half-samples divided by the
    estimate.

    WHAT IT IS NOT, because a review found the result of this being misread. It draws
    halves of the sample ALREADY DRAWN, so it is a WITHIN-SAMPLE SENSITIVITY DIAGNOSTIC. It
    is not a confidence interval, and it cannot see variation across the ORIGINAL draw from
    the field, which is the variation that decides whether two seeds would give the same
    threshold. A run needing that must draw under several seeds and compare, or take the
    exact percentile over every finite cell. Do not compare this number against a precision
    rule stated in terms of independent seeds, because they measure different things.
    """
    rng = rng if rng is not None else np.random.default_rng(1)
    sample = np.asarray(sample, dtype=float)
    estimate = float(np.percentile(sample, percentile))
    half = max(sample.size // 2, 1)
    draws = [float(np.percentile(rng.choice(sample, size=half, replace=False), percentile))
             for _ in range(repeats)]
    spread = (max(draws) - min(draws)) / abs(estimate) if estimate else float("inf")
    return estimate, float(spread)
