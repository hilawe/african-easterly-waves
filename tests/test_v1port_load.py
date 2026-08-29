"""Tests for reading retrieved reanalysis files into what the tracker takes.

THE MUTATION LIST, WRITTEN BEFORE THE ASSERTIONS EXISTED, executable as
tests/mutations_v1port_load.py.

    L1   the epoch offset is wrong or missing, so every timestamp is decades out while the
         units attribute still reads correctly
    L2   a time variable in units the loader does not convert is accepted anyway
    L3   u and v are paired by position without checking their time axes agree
    L4   the length-one pressure-level axis is not dropped
    L5   a missing file yields an empty year instead of a refusal
    L6   a year with only one of the two wind files is listed as available
    L7   the streaming climatology differs from the batch one
    L8   a missing value is left out of the divisor as well as the sum
    L9   the leap-day interpolation runs before all the calendar keys are known
    L10  the accumulator mixes grids of different shapes
    L11  the anomaly sampler draws missing values, or draws with replacement
    L12  the sampling error is not measured, so a noisy threshold looks solid

THE ONE THAT MATTERS MOST IS L7, and it is why the accumulator exists as a class rather
than as a loop in a script. Streaming the climatology is an optimisation forced by memory,
about eight gigabytes if thirty years are concatenated first, and an optimisation that
changes the answer is a defect. The equivalence is checked against the batch function on a
case small enough to compute both ways, including a leap year and a missing value, rather
than argued from the code.
"""

import datetime

import numpy as np
import pytest

nc = pytest.importorskip("netCDF4")

from aew.v1port import climatology as clim  # noqa: E402
from aew.v1port import load as L  # noqa: E402
from aew.v1port.pipeline import days_to_datetime64  # noqa: E402


def days_since_1900(year, month=1, day=1):
    return float((datetime.date(year, month, day) - datetime.date(1900, 1, 1)).days)


def write_wind_file(path, variable, times_seconds, lat, lon, values=None,
                    time_name="valid_time", time_units="seconds since 1970-01-01",
                    with_level=True):
    """A file shaped like the ones the retrieval produces."""
    with nc.Dataset(path, "w") as ds:
        ds.createDimension(time_name, len(times_seconds))
        ds.createDimension("latitude", len(lat))
        ds.createDimension("longitude", len(lon))
        t = ds.createVariable(time_name, "i8", (time_name,))
        t.units = time_units
        t[:] = np.asarray(times_seconds)
        ds.createVariable("latitude", "f8", ("latitude",))[:] = lat
        ds.createVariable("longitude", "f8", ("longitude",))[:] = lon
        shape = (time_name, "latitude", "longitude")
        if with_level:
            ds.createDimension("pressure_level", 1)
            shape = (time_name, "pressure_level", "latitude", "longitude")
        var = ds.createVariable(variable, "f4", shape)
        if values is None:
            values = np.zeros((len(times_seconds), len(lat), len(lon)))
        values = np.asarray(values, dtype=float)
        var[:] = values[:, None, :, :] if with_level else values


@pytest.fixture
def year_files(tmp_path):
    """One synthetic year-pair, at four timesteps, on a small descending grid."""
    def _make(year=1981, n=4, directory=None, **kwargs):
        directory = directory or tmp_path
        lat = np.array([10.0, 9.25, 8.5, 7.75])          # descending, as the files come
        lon = np.array([-5.0, -4.25, -3.5])
        start = (datetime.date(year, 1, 1) - datetime.date(1970, 1, 1)).days * 86400
        seconds = [start + 21600 * i for i in range(n)]
        for var, name in (("u700", "u"), ("v700", "v")):
            values = np.full((n, lat.size, lon.size), -8.0 if name == "u" else 1.0)
            write_wind_file(str(directory / f"eraint_{var}_{year}_6h_region.nc"),
                            name, seconds, lat, lon, values, **kwargs)
        return directory
    return _make


# --- time conversion --------------------------------------------------------------------

def test_the_epoch_offset_is_the_gap_between_1970_and_1900():
    """L1. Nothing downstream can detect this being wrong: the fields are fine, the tracks
    are fine, and the record carries timestamps decades out under a correct units string."""
    assert L.EPOCH_OFFSET_DAYS == 25567
    assert L.seconds_since_1970_to_days_since_1900(0) == pytest.approx(
        days_since_1900(1970))
    assert L.seconds_since_1970_to_days_since_1900(86400) == pytest.approx(
        days_since_1900(1970) + 1)


def test_a_known_date_round_trips_to_the_calendar():
    seconds = (datetime.date(2005, 1, 1) - datetime.date(1970, 1, 1)).days * 86400
    days = L.seconds_since_1970_to_days_since_1900(seconds)
    assert days == pytest.approx(days_since_1900(2005))
    assert str(days_to_datetime64([days])[0])[:10] == "2005-01-01"


def test_the_loaded_times_are_days_since_1900(year_files):
    directory = year_files(year=1981)
    times, _, _, _, _ = L.load_year(1981, str(directory))
    assert times[0] == pytest.approx(days_since_1900(1981))
    assert times[1] - times[0] == pytest.approx(0.25), "six-hourly"


def test_a_time_variable_in_unconvertible_units_is_refused(tmp_path, year_files):
    """L2. Guessing here is the failure the epoch test above describes, so the loader says
    it does not know rather than assuming."""
    year_files(year=1990, directory=tmp_path)
    lat, lon = np.array([10.0, 9.0]), np.array([0.0, 1.0])
    write_wind_file(str(tmp_path / "eraint_u700_1991_6h_region.nc"), "u",
                    [0, 6, 12], lat, lon, time_units="hours since 2000-01-01")
    write_wind_file(str(tmp_path / "eraint_v700_1991_6h_region.nc"), "v",
                    [0, 6, 12], lat, lon, time_units="hours since 2000-01-01")
    with pytest.raises(ValueError, match="does not"):
        L.load_year(1991, str(tmp_path))


def test_days_since_1900_are_passed_through(tmp_path):
    lat, lon = np.array([10.0, 9.0]), np.array([0.0, 1.0])
    for var, name in (("u700", "u"), ("v700", "v")):
        write_wind_file(str(tmp_path / f"eraint_{var}_1995_6h_region.nc"), name,
                        [38351, 38352], lat, lon,
                        time_units="days since 1900-01-01", time_name="time")
    times, _, _, _, _ = L.load_year(1995, str(tmp_path))
    assert times[0] == pytest.approx(38351.0)


# --- shape and pairing --------------------------------------------------------------------

def test_the_pressure_level_axis_is_dropped(year_files):
    """L4. The retrieval asks for one level and it arrives as a length-one axis; leaving it
    in makes every downstream shape check wrong."""
    directory = year_files(year=1981)
    _, latgrid, longrid, u, v = L.load_year(1981, str(directory))
    assert u.ndim == 3 and v.ndim == 3
    assert u.shape == (4, 4, 3)
    assert latgrid.shape == longrid.shape == (4, 3)


def test_the_grid_comes_back_in_the_files_own_row_order(year_files):
    """The tracker handles either order, but only because that was found and fixed. The
    order is reported rather than silently normalised, so a caller can see it."""
    directory = year_files(year=1981)
    _, latgrid, _, _, _ = L.load_year(1981, str(directory))
    assert float(latgrid[1, 0]) < float(latgrid[0, 0]), "these files come north-first"


def test_u_and_v_with_different_time_axes_are_refused(tmp_path):
    """L3. Pairing them by position would combine different timesteps into one wind
    vector, which no later stage could detect."""
    lat, lon = np.array([10.0, 9.0]), np.array([0.0, 1.0])
    write_wind_file(str(tmp_path / "eraint_u700_1981_6h_region.nc"), "u",
                    [0, 21600, 43200], lat, lon)
    write_wind_file(str(tmp_path / "eraint_v700_1981_6h_region.nc"), "v",
                    [0, 21600, 64800], lat, lon)      # third step differs by six hours
    with pytest.raises(ValueError, match="time axis"):
        L.load_year(1981, str(tmp_path))


def test_a_six_hour_offset_is_not_hidden_by_a_relative_tolerance(tmp_path):
    """L3, and the reason the comparison uses an absolute tolerance.

    Days since 1900 are numbers around thirty thousand, so numpy's default relative
    tolerance of 1e-5 is about a third of a day here. A first version of the check used the
    default and a six-hour disagreement compared as equal, which is the whole failure it
    was written to catch.
    """
    a = np.array([29585.5])
    b = np.array([29585.75])                          # six hours apart
    assert np.allclose(a, b), "the default tolerance really does hide this"
    assert not np.allclose(a, b, rtol=0.0, atol=1.0 / 86400.0)


def test_an_absent_file_is_refused_not_treated_as_an_empty_year(tmp_path):
    """L5. A retrieval still running is the normal state, so a gap has to be loud.

    THE PATTERN MATCHES THIS MODULE'S OWN WORDS, not a word that could appear anywhere in
    an error message. A first version matched on "missing" and passed even when the check
    was removed, because pytest names its temporary directory after the test, the test's
    name contained "missing", and netCDF's own file-not-found error quotes the path. The
    regex was matching the directory name.
    """
    with pytest.raises(FileNotFoundError, match="retrieval writes one file"):
        L.load_year(1981, str(tmp_path))


def test_only_years_with_both_files_are_listed(tmp_path, year_files):
    """L6. Half a year is not a usable year, and the retrieval writes u before v."""
    year_files(year=1981, directory=tmp_path)
    lat, lon = np.array([10.0, 9.0]), np.array([0.0, 1.0])
    write_wind_file(str(tmp_path / "eraint_u700_1982_6h_region.nc"), "u",
                    [0, 21600], lat, lon)             # u only
    assert L.available_years(str(tmp_path)) == [1981]


# --- the streaming climatology -------------------------------------------------------------

def synthetic_years(years=(2000, 2001, 2002), n=40, seed=4):
    """Curvature and times for several years over the same calendar window.

    The window straddles 29 February and one field carries a missing value, because those
    are the two places the batch version does something other than average.
    """
    rng = np.random.default_rng(seed)
    per_year, times = [], []
    for year in years:
        start = days_since_1900(year, 2, 26)
        block = rng.normal(size=(n, 5, 7))
        block[3, 2, 2] = np.nan
        per_year.append(block)
        times.append(start + 0.25 * np.arange(n))
    return per_year, times


def test_the_streaming_climatology_equals_the_batch_one():
    """L7, L8 and L9 together, and the reason the accumulator is worth its own class.

    Streaming is forced by memory: thirty years concatenated is about eight gigabytes. An
    optimisation that changes the answer is a defect, so the equivalence is measured on a
    case small enough to compute both ways rather than argued from the code. The fixture
    includes a leap year and a missing value, which are the two places the batch version
    does something other than take a mean.
    """
    per_year, times = synthetic_years()
    accumulator = L.ClimatologyAccumulator()
    for block, t in zip(per_year, times):
        accumulator.add(block, t)
    streamed = accumulator.finalize(expected_years=3)

    batch = clim.curvature_climatology(
        np.concatenate(per_year), days_to_datetime64(np.concatenate(times)),
        expected_years=3)

    assert streamed["keys"] == batch["keys"]
    assert np.array_equal(streamed["counts"], batch["counts"])
    assert np.allclose(streamed["mean"], batch["mean"], equal_nan=True)
    assert streamed["short_steps"] == batch["short_steps"]


def test_the_calendar_keys_come_back_sorted_whatever_order_the_years_arrive_in():
    """L7. The batch version sorts its calendar keys, so the streaming one has to as well
    or the two disagree on which mean belongs to which step, and the leap-day interpolation
    reads its neighbours from the wrong places.

    THE YEARS ARE ADDED OUT OF CALENDAR ORDER on purpose. Python preserves insertion order
    in a dict, so a fixture that adds them in order cannot tell a sorted result from an
    unsorted one, and an earlier version of this file could not.
    """
    accumulator = L.ClimatologyAccumulator()
    # June first, then January, so insertion order and calendar order disagree
    accumulator.add(np.ones((2, 3, 3)),
                    [days_since_1900(2001, 6, 1), days_since_1900(2001, 6, 1) + 0.25])
    accumulator.add(np.ones((2, 3, 3)) * 2.0,
                    [days_since_1900(2001, 1, 5), days_since_1900(2001, 1, 5) + 0.25])
    out = accumulator.finalize()
    assert out["keys"] == sorted(out["keys"])
    assert out["keys"][0][0] == 1, "January must come first however the years arrived"
    assert out["mean"][0, 0, 0] == pytest.approx(2.0), (
        "the mean must travel with its own key, not with its insertion position")


def test_the_fixture_actually_covers_a_leap_day_and_a_missing_value():
    """The test above proves nothing if its fixture avoids both special cases."""
    per_year, times = synthetic_years()
    accumulator = L.ClimatologyAccumulator()
    for block, t in zip(per_year, times):
        accumulator.add(block, t)
    out = accumulator.finalize(expected_years=3)
    assert any(k[:2] == (2, 29) for k in out["keys"]), "no 29 February in the fixture"
    assert any(np.isnan(block).any() for block in per_year), "no missing value either"


def test_a_missing_value_still_counts_toward_the_divisor():
    """L8. The batch version sums a missing value as zero and still counts it, so the mean
    is pulled toward zero rather than computed over fewer samples. Reproduced, because the
    two give different numbers and the streaming version has to give the batch one."""
    block_a = np.full((1, 2, 2), 4.0)
    block_b = np.full((1, 2, 2), np.nan)
    times = [days_since_1900(2001, 6, 1)], [days_since_1900(2002, 6, 1)]
    accumulator = L.ClimatologyAccumulator()
    accumulator.add(block_a, times[0])
    accumulator.add(block_b, times[1])
    out = accumulator.finalize()
    assert out["counts"][0] == 2
    assert out["mean"][0, 0, 0] == pytest.approx(2.0), "4 and a missing value average to 2"


def test_the_accumulator_refuses_to_mix_grids():
    """L10. Two grids in one climatology is a mistake with no sensible answer."""
    accumulator = L.ClimatologyAccumulator()
    accumulator.add(np.zeros((2, 3, 4)), [days_since_1900(2001), days_since_1900(2001) + 1])
    with pytest.raises(ValueError, match="cannot mix grids"):
        accumulator.add(np.zeros((2, 5, 6)),
                        [days_since_1900(2002), days_since_1900(2002) + 1])


def test_an_empty_accumulator_refuses_rather_than_returning_nothing():
    with pytest.raises(ValueError, match="nothing was added"):
        L.ClimatologyAccumulator().finalize()


def test_mismatched_times_and_fields_are_refused():
    accumulator = L.ClimatologyAccumulator()
    with pytest.raises(ValueError, match="times has"):
        accumulator.add(np.zeros((3, 2, 2)), [0.0, 1.0])


# --- sampling the anomaly for a threshold ------------------------------------------------

def test_the_sampler_draws_only_finite_values():
    """A threshold is a percentile of the values the mask actually compares against, and
    the edges of every field are not-a-number by construction, so drawing them would pull
    the percentile toward nothing."""
    field = np.full((6, 6), 3.0)
    field[0, :] = np.nan
    drawn = L._draw(field, 100, np.random.default_rng(0))
    assert drawn.size == 30, "the six not-a-number cells are excluded"
    assert np.all(drawn == 3.0)


def test_the_sampler_takes_everything_when_there_is_little():
    field = np.arange(9.0).reshape(3, 3)
    drawn = L._draw(field, 100, np.random.default_rng(0))
    assert sorted(drawn) == list(range(9))


def test_the_sampler_draws_without_replacement():
    field = np.arange(100.0).reshape(10, 10)
    drawn = L._draw(field, 40, np.random.default_rng(0))
    assert drawn.size == 40
    assert len(set(drawn.tolist())) == 40, "a repeated value would bias the percentile"


def test_an_all_missing_field_yields_nothing_rather_than_zeros():
    drawn = L._draw(np.full((4, 4), np.nan), 10, np.random.default_rng(0))
    assert drawn.size == 0


def test_the_sampling_error_reports_more_noise_for_a_smaller_sample():
    """The estimate is only usable if its own noise is small against it, so the noise is
    measured and returned rather than assumed. A function that reported the same spread
    however much data it had would make the check meaningless."""
    rng = np.random.default_rng(3)
    population = rng.normal(size=200000)
    big, big_spread = L.threshold_sampling_error(population, 55.0)
    small, small_spread = L.threshold_sampling_error(population[:2000], 55.0)
    assert big == pytest.approx(np.percentile(population, 55.0))
    assert small_spread > big_spread * 3, (
        "a fortieth of the data must show visibly more sampling noise")


def test_the_sampling_error_estimate_is_the_percentile_itself():
    values = np.arange(1000.0)
    estimate, _ = L.threshold_sampling_error(values, 66.0)
    assert estimate == pytest.approx(np.percentile(values, 66.0))
