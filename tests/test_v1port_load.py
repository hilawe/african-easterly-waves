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
    L12  the half-sample range is not computed, so an unstable threshold looks solid
    L17  a per_step at or above the finite-cell count is still treated as a random draw,
         so an exact percentile is reported as sampled
    L13  the grid is coarsened AFTER curvature is taken rather than before, so the
         derivative runs on the fine spacing and is then sampled
    L14  the climatology is accumulated on the full grid while the yearly fields are
         coarsened, so the anomaly subtracts two different grids
    L15  the stride is dropped on one side of the chain and the shapes happen to agree

THE ONE THAT MATTERS MOST IS L7, and it is why the accumulator exists as a class rather
than as a loop in a script. Streaming the climatology is an optimisation forced by memory,
about eight gigabytes if thirty years are concatenated first, and an optimisation that
changes the answer is a defect. The equivalence is checked against the batch function on a
case small enough to compute both ways, including a leap year and a missing value, rather
than argued from the code.

L13 THROUGH L15 ARE ONE SEAM, and a repository-access review found it rather than these
tests, which is the part worth keeping. `--subsample 2` is how a half-degree ERA5
retrieval is meant to be put on version 1's whole degrees. It crashed, because the
climatology was built on the full grid while the yearly curvature was strided, and no test
ran the path end to end: the unit tests handed `sample_anomaly_values` a curvature field
and a climatology that already agreed, so the seam between them was never crossed.

The repair is not to make the shapes agree. Curvature vorticity is a spatial derivative,
so striding before and striding after are different operations, and on real ERA5 they
differ by 45.8 percent of the field's own root-mean-square. The stride belongs at the WIND
stage, where a coarser retrieval request would have applied it, and that ordering is what
L13 binds.
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


def test_the_half_sample_range_widens_for_a_smaller_sample():
    """The range must widen when the sample shrinks, or the diagnostic diagnoses nothing.

    WHAT THE STATISTIC IS, stated here because its earlier name and this docstring both
    called it measured noise. It is a WITHIN-SAMPLE half-sample range, a sensitivity
    diagnostic, and it is not a precision measure for the original draw. What this test
    binds is only its one useful property, that it responds to sample size, since a
    function reporting the same spread however much data it had would diagnose nothing."""
    rng = np.random.default_rng(3)
    population = rng.normal(size=200000)
    big, big_spread = L.threshold_sampling_error(population, 55.0)
    small, small_spread = L.threshold_sampling_error(population[:2000], 55.0)
    assert big == pytest.approx(np.percentile(population, 55.0))
    assert small_spread > big_spread * 3, (
        "a fortieth of the data must show a visibly wider half-sample range")


def test_the_half_sample_range_estimate_is_the_percentile_itself():
    values = np.arange(1000.0)
    estimate, _ = L.threshold_sampling_error(values, 66.0)
    assert estimate == pytest.approx(np.percentile(values, 66.0))


def test_the_threshold_sample_honours_a_spatial_box():
    """The box that makes a threshold reproducible, bound rather than trusted.

    MUTATIONS THIS BINDS, listed before the assertions were written:
      B1 the box is ignored, so the sample is the whole grid again
      B2 latitude and longitude bounds are swapped
      B3 only one of the two bounds is applied
      B4 the coarse grid gets the fine grid's coordinates, so its box is the wrong size

    WHY IT MATTERS. version 1's archive records the percentile and the field, not the
    sample. This project measured that the same recipe reproduces version 1's published
    ERA-Interim fine threshold to about 2 percent over an African box and misses by 11 to
    20 percent over wider ones, so the region is part of the answer and has to be a stated
    argument rather than an implied default.
    """
    lat = np.arange(-30.0, 30.1, 2.0)
    lon = np.arange(-60.0, 60.1, 2.0)
    longrid, latgrid = np.meshgrid(lon, lat)
    # a field whose value encodes WHERE it came from, so the sample's provenance is
    # checkable rather than merely plausible
    curvature = (latgrid * 1000.0 + longrid)[np.newaxis, ...].repeat(4, axis=0)
    times = 38351.0 + 0.25 * np.arange(4)
    # a zero climatology keyed on the fields' own calendar steps, so the anomaly IS the
    # field and every sampled value still decodes to the place it came from
    from aew.v1port.pipeline import days_to_datetime64
    stamps = days_to_datetime64(times)
    keys = [(int(str(t)[5:7]), int(str(t)[8:10]), int(str(t)[11:13])) for t in stamps]
    climatology = {"keys": keys, "mean": np.zeros_like(curvature),
                   "counts": np.ones(len(keys)), "short_steps": []}

    box_lat, box_lon = (-10.0, 10.0), (-20.0, 20.0)

    def decode(value, lat_tol, lon_tol):
        """Which (lat, lon) a sampled value came from, or None if it decodes nowhere.

        The field is lat*1000 + lon, so a value carries its own position. The southern
        hemisphere sign flip negates it south of the equator, so both signs are tried. The
        nine-point smoother leaves a LINEAR field unchanged in the interior and perturbs it
        only at the grid edge, which is what the tolerances allow for.
        """
        for sign in (1.0, -1.0):
            w = sign * float(value)
            lat_hat = round(w / 1000.0)
            lon_hat = w - lat_hat * 1000.0
            if (box_lat[0] - lat_tol <= lat_hat <= box_lat[1] + lat_tol
                    and box_lon[0] - lon_tol <= lon_hat <= box_lon[1] + lon_tol):
                return lat_hat, lon_hat
        return None

    coarse, fine = L.sample_anomaly_values(
        curvature, times, climatology, lat, per_step=10_000,
        rng=np.random.default_rng(0), lon_values=lon,
        native_resolution=2.0, coarse_resolution=6.0,
        lat_range=box_lat, lon_range=box_lon)
    assert fine.size, "the box must not empty the fine sample"
    assert coarse.size, "the box must not empty the coarse sample"

    # BOTH bounds on BOTH grids. Checking latitude alone leaves a mutation that drops the
    # longitude bound alive, and checking the fine grid alone leaves one that builds the
    # coarse box from the fine coordinates alive. Both were found that way.
    bad = [float(v) for v in fine if decode(v, 1, 2) is None]
    assert not bad, f"{len(bad)} fine values decode outside the box, e.g. {bad[:3]}"
    bad_c = [float(v) for v in coarse if decode(v, 3, 6) is None]
    assert not bad_c, f"{len(bad_c)} coarse values decode outside the box, e.g. {bad_c[:3]}"

    whole = L.sample_anomaly_values(
        curvature, times, climatology, lat, per_step=10_000,
        rng=np.random.default_rng(0), lon_values=lon,
        native_resolution=2.0, coarse_resolution=6.0)[1]
    assert whole.size > fine.size, \
        "the boxed sample must be smaller than the whole grid, or the box did nothing"


def test_the_profiles_bind_the_configuration_they_name():
    """The named baselines, checked against the facts they exist to freeze.

    MUTATIONS THIS BINDS, listed before the assertions:
      P1 the faithful profile points at the unbuffered 0.75 degree tree
      P2 it silently gains the duplication repair, so "faithful" is no longer faithful
      P3 the faithful and repaired profiles stop differing in exactly one setting
      P4 an unknown profile name falls back to a default instead of refusing
      P5 an ERA5 profile appears while its threshold gate is still open

    WHY. A science-direction review found two entry points able to recreate disproven
    configurations without announcing it. Correcting a default fixes one path; naming the
    configuration is what lets a RESULT cite what produced it.
    """
    from aew.v1port import profiles

    faithful = profiles.get("faithful-eraint-700")
    assert faithful.directory.endswith("v1port_buffered"), \
        "the faithful baseline must use the buffered one-degree tree"
    assert faithful.native_resolution == 1.0, "version 1 ran at one degree"
    assert faithful.absorb is False and faithful.exclusive is False, \
        "the faithful baseline reproduces version 1's defects, which is its purpose"
    assert faithful.coarse_threshold == 7.16e-7 and faithful.fine_threshold == 2.80e-6, \
        "the archived pair, not a recomputed one"

    repaired = profiles.get("repaired-eraint-700")
    differing = [k for k, v in faithful.describe().items()
                 if k not in ("name", "purpose", "notes") and repaired.describe()[k] != v]
    assert differing == ["exclusive"], \
        f"the repaired profile must differ in exactly one setting, differs in {differing}"

    with pytest.raises(SystemExit, match="unknown profile"):
        profiles.get("no-such-profile")

    assert not any("era5" in name for name in profiles.PROFILES), \
        "no ERA5 profile until its threshold gate closes"


@pytest.fixture
def varying_year(tmp_path):
    """A year-pair whose winds vary in space, so a derivative is not degenerate.

    The constant-wind fixture above cannot tell striding-before from striding-after apart,
    because every curvature it produces is the same number. That is exactly how a seam
    like this stays invisible: the cheapest fixture is the one that cannot see it.

    THE WAVENUMBERS ARE HIGH DELIBERATELY. A first version of this used slowly varying
    sinusoids, and the two orderings agreed to about a tenth of a percent, because a field
    oversampled by its grid is differentiated almost identically at either spacing. That
    version could not have failed. Real wind fields carry structure near the grid scale,
    which is why the same comparison on real ERA5 gives 45.8 percent, so the fixture has to
    vary near the grid scale too or it tests nothing.
    """
    def _make(year=1981, n=4, directory=None, prefix="era5"):
        directory = directory or tmp_path
        # descending latitudes at half a degree, as an ERA5 retrieval comes
        lat = 20.0 - 0.5 * np.arange(24)
        lon = -10.0 + 0.5 * np.arange(28)
        LO, LA = np.meshgrid(lon, lat)
        start = (datetime.date(year, 1, 1) - datetime.date(1970, 1, 1)).days * 86400
        seconds = [start + 21600 * i for i in range(n)]
        for var, name in (("u700", "u"), ("v700", "v")):
            # periods of a few degrees, so the half-degree and whole-degree spacings see
            # genuinely different fields, as they do in real wind
            base = (np.sin(np.deg2rad(120 * LO)) * np.cos(np.deg2rad(150 * LA))
                    if name == "u" else
                    np.cos(np.deg2rad(140 * LO)) * np.sin(np.deg2rad(110 * LA)))
            values = np.stack([-8.0 * base + 0.4 * i for i in range(n)]) \
                if name == "u" else np.stack([3.0 * base - 0.2 * i for i in range(n)])
            write_wind_file(str(directory / f"{prefix}_{var}_{year}_6h_region.nc"),
                            name, seconds, lat, lon, values)
        return directory
    return _make


def test_the_stride_is_applied_to_the_winds_not_to_the_curvature(varying_year, tmp_path):
    """L13. Coarsening must happen before the derivative, not after it.

    WHY THIS IS NOT PEDANTRY. Curvature vorticity is built from spatial derivatives of the
    wind, so taking every second point of a curvature field computed at half a degree is
    not the same quantity as the curvature a whole-degree wind field would have given. On
    real ERA5 the two differ by 45.8 percent normalized, at every finite cell, in the very
    field whose quantiles set the detection thresholds.

    The assertion is two-sided ON PURPOSE. Equalling the wind-first path is what makes the
    operation right; DIFFERING from the curvature-first path is what makes the test able to
    fail. A one-sided version would pass on a degenerate field.
    """
    from aew.v1port.pipeline import curvature_from_winds

    varying_year(year=1981, directory=tmp_path, prefix="era5")
    _, latgrid, longrid, strided = L.curvature_for_year(
        1981, str(tmp_path), "era5", subsample=2)

    _, lat_full, lon_full, u, v = L.load_year(1981, str(tmp_path), "era5")
    wind_first = curvature_from_winds(lat_full[::2, ::2], lon_full[::2, ::2],
                                      u[:, ::2, ::2], v[:, ::2, ::2])
    curvature_first = curvature_from_winds(lat_full, lon_full, u, v)[:, ::2, ::2]

    assert strided.shape == wind_first.shape, "the stride must reach the returned grid"
    assert latgrid.shape == strided.shape[1:], "the grid must be strided with the field"
    np.testing.assert_allclose(
        strided, wind_first, rtol=1e-12, atol=0,
        err_msg="the stride must be applied to the winds, before curvature is taken")

    finite = np.isfinite(wind_first) & np.isfinite(curvature_first)
    assert finite.sum() > 50, "the fixture must produce a field worth comparing"
    assert not np.allclose(wind_first[finite], curvature_first[finite]), \
        ("the fixture cannot tell the two orderings apart, so this test could not fail; "
         "it needs winds that actually vary in space")


def test_a_stride_runs_the_whole_threshold_path_without_mixing_grids(varying_year,
                                                                    tmp_path):
    """L14 and L15. The climatology and the yearly fields must be coarsened alike.

    THE SEAM THIS CROSSES is the one the unit tests above never did. They hand
    `sample_anomaly_values` a curvature field and a climatology built from the same array,
    so the two agree by construction. This builds the climatology through its own entry
    point and samples a year through its own, which is what the threshold program does,
    and it is where `--subsample 2` failed with a broadcast error on real ERA5 data.
    """
    for year in (1981, 1982):
        varying_year(year=year, directory=tmp_path, prefix="era5")

    climatology = L.build_climatology([1981, 1982], str(tmp_path), "era5", subsample=2)
    times, latgrid, longrid, curvature = L.curvature_for_year(
        1981, str(tmp_path), "era5", subsample=2)

    assert climatology["mean"].shape[1:] == curvature.shape[1:], (
        "the climatology and the yearly field must be on the same grid; they were not, "
        "and the anomaly subtraction raised a broadcast error")

    native = abs(float(latgrid[1, 0] - latgrid[0, 0]))
    assert native == pytest.approx(1.0), \
        "a stride of two on a half-degree retrieval must give whole degrees"

    coarse, fine = L.sample_anomaly_values(
        curvature, times, climatology, latgrid[:, 0], native_resolution=native,
        coarse_resolution=2.5, per_step=8, rng=np.random.default_rng(0),
        lon_values=longrid[0, :])
    assert coarse.size and fine.size, "the path must yield samples, not empty arrays"
    assert np.all(np.isfinite(coarse)) and np.all(np.isfinite(fine))


def test_the_box_is_cropped_before_smoothing_as_the_tracker_does(varying_year, tmp_path):
    """L16. The archive crops to the tracking domain and THEN smooths, and so must this.

    THE ORDER IS VISIBLE IN THE SOURCE. `p2_track_eraint_700hPa.m` decimates the full
    field, then under "Parse Data to Final Lat/Lon Domain" crops both the fine and the
    coarse grids to lat1/lat2 and lon1/lon2, and passes the CROPPED arrays into
    `find_ews_f.m`, which calls `smth9_f` on them at lines 84 to 91. The smoother therefore
    sees the domain edge as a boundary.

    WHY THAT CHANGES VALUES RATHER THAN JUST BOOKKEEPING. `smth9_f` starts its loops at 2
    and stops at size-1, so it leaves the OUTER RING of whatever it is handed untouched.
    Cropping first means the tracking domain's perimeter keeps its raw values. Smoothing a
    buffered field and masking afterwards smooths that perimeter using halo neighbours the
    archived smoother never sees, so every cell around the edge of the sample differs.

    An earlier version of `sample_anomaly_values` did exactly that, and its comment claimed
    the opposite, that masking after smoothing gives "what the tracker would compute at the
    same place". A review checked it against the source. It does not.
    """
    from aew.v1port.climatology import smooth9, southern_hemisphere_sign

    varying_year(year=1981, directory=tmp_path, prefix="era5")
    times, latgrid, longrid, curvature = L.curvature_for_year(1981, str(tmp_path), "era5")
    climatology = L.build_climatology([1981], str(tmp_path), "era5")

    lat_range, lon_range = (5.0, 15.0), (-6.0, 4.0)
    native, coarse_res = 0.5, 1.0
    coarse_sample, fine = L.sample_anomaly_values(
        curvature, times, climatology, latgrid[:, 0], per_step=100000,
        rng=np.random.default_rng(0), lon_values=longrid[0, :],
        lat_range=lat_range, lon_range=lon_range,
        native_resolution=native, coarse_resolution=coarse_res)

    anomaly = clim.curvature_anomaly(
        curvature, L._as_datetime(times), climatology)
    lats, lons = latgrid[:, 0], longrid[0, :]
    rows = np.where((lats >= lat_range[0]) & (lats <= lat_range[1]))[0]
    cols = np.where((lons >= lon_range[0]) & (lons <= lon_range[1]))[0]
    assert rows.size > 3 and cols.size > 3, "the box must be big enough to have an interior"

    crop_then_smooth, smooth_then_crop = [], []
    for step in range(anomaly.shape[0]):
        cropped = anomaly[step][np.ix_(rows, cols)]
        crop_then_smooth.append(southern_hemisphere_sign(
            smooth9(cropped)[np.newaxis, ...], lats[rows])[0])
        whole = southern_hemisphere_sign(
            smooth9(anomaly[step])[np.newaxis, ...], lats)[0]
        smooth_then_crop.append(whole[np.ix_(rows, cols)])
    crop_then_smooth = np.concatenate([a.ravel() for a in crop_then_smooth])
    smooth_then_crop = np.concatenate([a.ravel() for a in smooth_then_crop])

    # THE FIXTURE MUST BE ABLE TO TELL THEM APART, or this test cannot fail.
    assert not np.allclose(crop_then_smooth, smooth_then_crop), (
        "the two orderings agree on this fixture, so it cannot distinguish them; the box "
        "needs a perimeter whose neighbours differ from the domain boundary")

    # `_draw` returns only finite values, and curvature is NaN at the grid boundary, so
    # the comparison is over the finite cells of each ordering rather than every cell.
    expected = np.sort(crop_then_smooth[np.isfinite(crop_then_smooth)])
    other = np.sort(smooth_then_crop[np.isfinite(smooth_then_crop)])
    got = np.sort(fine)
    assert got.size == expected.size, (
        f"the sample must be every finite cell in the box, since per_step exceeds it; "
        f"got {got.size} against {expected.size}")
    np.testing.assert_allclose(
        got, expected, rtol=1e-12, atol=0,
        err_msg="the sampler must crop to the box and THEN smooth, as the tracker does")
    assert not (got.size == other.size and np.allclose(got, other)), \
        "the sampler still matches the smooth-then-crop ordering, which the tracker does not use"

    # THE COARSE PATH IS THE HALF THAT MATTERS MOST and the first version of this test did
    # not exercise it at all, because it called the sampler without a coarse resolution. A
    # review mutation-tested that directly: the fine-grid order was bound and the coarse
    # order survived. The coarse threshold is the one that will not reproduce, so leaving
    # its order unbound was leaving the consequential half unverified.
    #
    # The archive DECIMATES THE FULL FIELD FIRST, so the Gaussian sees the halo, and only
    # then crops and smooths. That is the order reproduced here.
    from aew.v1port.climatology import decimation_shape, gaussian_decimate

    _, stride = decimation_shape(native, coarse_res)
    coarse_lats = lats[::stride]
    coarse_lons = lons[::stride]
    crows = np.where((coarse_lats >= lat_range[0]) & (coarse_lats <= lat_range[1]))[0]
    ccols = np.where((coarse_lons >= lon_range[0]) & (coarse_lons <= lon_range[1]))[0]
    assert crows.size > 2 and ccols.size > 2, "the coarse box needs an interior too"

    coarse_expected, coarse_other = [], []
    for step in range(anomaly.shape[0]):
        decimated = gaussian_decimate(anomaly[step], native, coarse_res)
        cropped_c = decimated[np.ix_(crows, ccols)]
        coarse_expected.append(southern_hemisphere_sign(
            smooth9(cropped_c)[np.newaxis, ...], coarse_lats[crows])[0])
        whole_c = southern_hemisphere_sign(
            smooth9(decimated)[np.newaxis, ...], coarse_lats)[0]
        coarse_other.append(whole_c[np.ix_(crows, ccols)])
    coarse_expected = np.concatenate([a.ravel() for a in coarse_expected])
    coarse_other = np.concatenate([a.ravel() for a in coarse_other])
    coarse_expected = np.sort(coarse_expected[np.isfinite(coarse_expected)])
    coarse_other = np.sort(coarse_other[np.isfinite(coarse_other)])

    assert not (coarse_expected.size == coarse_other.size
                and np.allclose(coarse_expected, coarse_other)), \
        "the coarse fixture cannot tell the two orderings apart, so this could not fail"
    got_coarse = np.sort(coarse_sample)
    assert got_coarse.size == coarse_expected.size, (
        f"the coarse sample must be every finite cell in the coarse box; got "
        f"{got_coarse.size} against {coarse_expected.size}")
    np.testing.assert_allclose(
        got_coarse, coarse_expected, rtol=1e-12, atol=0,
        err_msg="the COARSE path must decimate, then crop, then smooth, as the tracker does")


def test_a_large_enough_per_step_makes_the_draw_exact_and_seed_independent():
    """L17. What `--per-step` actually does, because a docstring once said it does nothing.

    THE CLAIM THAT WAS WRONG. While correcting a separate misuse of the half-sample range,
    the threshold program's documentation acquired the sentence that raising `--per-step`
    "does not address" between-seed variation. It does. `_draw` takes a larger random sample
    as the count rises, which ordinarily reduces that variation, and once the count reaches
    the number of finite cells it returns the whole field, so the spatial sampling component
    is gone and the result no longer depends on the seed at all.

    WHAT REMAINS TRUE is narrower, since raising it cannot MEASURE the precision that
    remains from a single seed. Reducing a source of variation and quantifying it are
    different things, and conflating them is what produced the wrong sentence.

    This test exists so the estimator cannot be built on the false version. The exact branch
    is what the primary cases are meant to use at one degree.
    """
    field = np.arange(100.0).reshape(10, 10)
    field[0, 0] = np.nan                                  # 99 finite cells
    finite = int(np.isfinite(field).sum())
    assert finite == 99

    partial = [L._draw(field, 40, np.random.default_rng(s)) for s in (0, 1)]
    assert all(d.size == 40 for d in partial)
    assert not np.array_equal(np.sort(partial[0]), np.sort(partial[1])), \
        "below the finite-cell count the draw must depend on the seed, or the rest is moot"

    # ONE BELOW THE FINITE COUNT is the other side of the same boundary, and it is checked
    # because a comparison loosened by one in that direction returns the whole field where
    # a sample was asked for. That would silently hand back more values than requested and
    # make a run that believes it sampled an exact one.
    just_under = L._draw(field, finite - 1, np.random.default_rng(3))
    assert just_under.size == finite - 1, (
        f"asking for {finite - 1} of {finite} finite cells must return {finite - 1}, not "
        f"the whole field")

    for count in (finite, finite + 1, finite * 3):
        exact = [L._draw(field, count, np.random.default_rng(s)) for s in (0, 1, 2)]
        assert all(d.size == finite for d in exact), \
            f"a per_step of {count} over {finite} finite cells must return every one"
        assert np.array_equal(np.sort(exact[0]), np.sort(exact[1])), \
            "at or above the finite-cell count the draw must not depend on the seed"
        np.testing.assert_allclose(
            np.percentile(exact[0], 55.0),
            np.percentile(field[np.isfinite(field)], 55.0),
            err_msg="the exact branch must give the population percentile, not an estimate")

    # AND IT MUST NOT CONSUME RANDOMNESS, which is the assertion that gives this test teeth.
    # Comparing the returned values alone cannot distinguish the exact branch from a
    # permutation of every element, because `rng.choice(n, size=n, replace=False)` returns
    # the same SET, so a sorted comparison passes and the percentile is identical. Two mutations that
    # remove the exact branch survived on that basis until this was added.
    #
    # The difference is observable because the sampler threads ONE generator across every
    # timestep. A draw that consumes randomness where it should not shifts every later
    # timestep's sample, so the defect would surface far from its cause.
    # THE BOUNDARY IS count == finite EXACTLY, and it is checked first because that is the
    # only place an off-by-one in the comparison shows. With count comfortably above the
    # finite count, `<=` and `<` behave identically and the mutation hides.
    for count in (finite, finite + 1, finite * 2):
        shared = np.random.default_rng(7)
        L._draw(field, count, shared)
        reference = np.random.default_rng(7)
        assert shared.random() == reference.random(), (
            f"a per_step of {count} over {finite} finite cells must return without touching "
            f"the generator; consuming it here silently changes every subsequent timestep's "
            f"draw, and at count == finite a permutation of the whole field would otherwise "
            f"be indistinguishable from the exact branch")
