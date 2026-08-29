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


def curvature_for_year(year, directory="data/eraint/v1port", prefix="eraint"):
    """Curvature vorticity for one year, with its times and grid."""
    from .pipeline import curvature_from_winds

    times, latgrid, longrid, u, v = load_year(year, directory, prefix)
    return times, latgrid, longrid, curvature_from_winds(latgrid, longrid, u, v)


def build_climatology(years, directory="data/eraint/v1port", prefix="eraint",
                      progress=None):
    """The 1981-2010 style climatological mean, accumulated year by year.

    `progress` is called with (year, index, total) after each year, because this reads and
    differentiates thirty years of wind and a silent hour is hard to tell from a hang.
    """
    accumulator = ClimatologyAccumulator()
    years = list(years)
    for n, year in enumerate(years, start=1):
        times, _, _, curvature = curvature_for_year(year, directory, prefix)
        accumulator.add(curvature, times)
        del curvature
        if progress is not None:
            progress(year, n, len(years))
    return accumulator.finalize(expected_years=len(years))


def sample_anomaly_values(curvature, times_days, climatology, lat_values,
                          decimation_factor=None, native_resolution=None,
                          coarse_resolution=None, per_step=400, rng=None):
    """A random sample of the anomaly values a threshold would be taken over.

    WHY A SAMPLE. The thresholds are percentiles of the anomaly over the whole climatology
    period, and thirty years of it at ERA5's resolution is about eighteen gigabytes. A
    percentile of a large random sample estimates the percentile of the population, and
    unlike a per-year percentile averaged afterwards it estimates the right quantity.
    `threshold_sampling_error` measures how much precision that costs rather than assuming
    it is enough.

    Returns (coarse_sample, fine_sample), the smoothed and sign-adjusted values from each
    grid, which is what `climatology.anomaly_threshold` takes its percentile of.
    """
    from .climatology import (anomaly_threshold, gaussian_decimate,  # noqa: F401
                              smooth9, southern_hemisphere_sign)

    rng = rng if rng is not None else np.random.default_rng(0)
    anomaly = clim.curvature_anomaly(curvature, _as_datetime(times_days), climatology)

    coarse_values, fine_values = [], []
    for step in range(anomaly.shape[0]):
        fine = southern_hemisphere_sign(smooth9(anomaly[step])[np.newaxis, ...],
                                        lat_values)[0]
        fine_values.append(_draw(fine, per_step, rng))
        if native_resolution is not None and coarse_resolution is not None:
            coarse_field = gaussian_decimate(anomaly[step], native_resolution,
                                             coarse_resolution)
            _, stride = clim.decimation_shape(native_resolution, coarse_resolution)
            coarse = southern_hemisphere_sign(smooth9(coarse_field)[np.newaxis, ...],
                                              lat_values[::stride])[0]
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
    """How much the percentile moves when the sample is redrawn, as a fractional spread.

    THE POINT IS TO REFUSE TO GUESS. A threshold estimated from a sample is only usable if
    the sampling noise is small against the number itself, and the way to know that is to
    resample and look. Returns (estimate, relative_spread) where the spread is the full
    range across half-samples divided by the estimate.
    """
    rng = rng if rng is not None else np.random.default_rng(1)
    sample = np.asarray(sample, dtype=float)
    estimate = float(np.percentile(sample, percentile))
    half = max(sample.size // 2, 1)
    draws = [float(np.percentile(rng.choice(sample, size=half, replace=False), percentile))
             for _ in range(repeats)]
    spread = (max(draws) - min(draws)) / abs(estimate) if estimate else float("inf")
    return estimate, float(spread)
