"""The threshold estimators, implementing the contracts frozen before this code existed.

WHAT THIS MODULE IS. The machinery behind `scripts/compute_thresholds.py`, split out so
every rule the project's written threshold contract fixed in advance is a small
function a test can bind and a mutation can attack. The contracts were written
first, deliberately, and this module makes no choice they did not already make.

THE TWO PERIODS ARE SEPARATE EVERYWHERE. The archive's own tracking driver loops
1979-2010 while hardcoding a 1981-2010 climatology file, so the climatology years and the
percentile-population years are different inputs with different meanings, and the coupled
`--years` of the previous program could not express the documented experiment at all.
Nothing in this module accepts a single period.

THE TRANSFORMATIONS ARE THE FROZEN SIX. T0 is the code as archived, one nine-point pass
and an unscaled coarse percentile. T1 doubles the smoothing per the paper's description.
T2 and T3 scale the COARSE threshold by the "(90% for smoothing)" comment's two possible
readings, up and down, both mandatory so the helpful direction cannot be reported as
derived. T4 and T5 are the interactions. The scale applies to the coarse RESULT only,
after selection, and never to the fine threshold, which carries no such note.

THE POPULATION IS BUILT ONCE, ON DISK. Each year's anomaly is transformed exactly as the
tracker transforms it, cropped to the stated domain and THEN smoothed (the archive's
order, where the nine-point smoother meets the domain edge as a boundary), sign-adjusted,
and appended raw to one file per grid. Both estimators then walk the files, so the
expensive atmospheric chain runs once however many passes the selection needs, and the
memory-bounded percentile reads monthly-sized blocks. The builder asserts that the values
written equal rows x cols x steps exactly, which is the domain upper bound the finite
count is later checked against.

THE FALLBACK IS THE CONTRACT'S LADDER, mechanically. Seeds 0 to 4 on a generator PINNED
to PCG64 (a seed alone does not name a stream once NumPy changes its default), per-step
levels 400 to 6400 doubling, the per-seed estimate an exact percentile of that seed's
draw, the reported threshold the MEDIAN of the five, stability the relative between-seed
range (max - min) / abs(median) with a zero or non-finite median refusing the run, the
gate strict `< 0.01` on BOTH thresholds independently, the first passing level the
result, exhaustion an explicit non-result, and every attempted level recorded with
SELECTED and AVAILABLE counts so exactness is computed, never asserted. Draws are
regenerated from a fresh pinned generator on every walk, which is what lets the exact
percentile stream over them without anything being stored.
"""
import os

import numpy as np

from . import climatology as clim
from .percentile import exact_percentile

COARSE_Q = float(clim.COARSE_PERCENTILE)          # 55.0
FINE_Q = float(clim.FINE_PERCENTILE)              # 66.0

# id -> (smoothing passes, coarse scale direction)
TRANSFORMATIONS = {
    "T0": (1, None),
    "T1": (2, None),
    "T2": (1, "up"),
    "T3": (1, "down"),
    "T4": (2, "up"),
    "T5": (2, "down"),
}

LADDER = (400, 800, 1600, 3200, 6400)
SEEDS = (0, 1, 2, 3, 4)
GATE = 0.01
BLOCK_STEPS = 124                                  # monthly-sized reads of the files


def pinned_generator(seed):
    """A PCG64 stream by construction, never through default_rng.

    default_rng uses PCG64 today and NumPy is free to change that default, at which
    point the same seeds would draw different samples and no artifact would say why.
    """
    return np.random.Generator(np.random.PCG64(seed))


def apply_coarse_scale(value, direction):
    """The "(90% for smoothing)" adjustment, in the stated direction, coarse only."""
    if direction is None:
        return value
    if direction == "up":
        return value / 0.9
    if direction == "down":
        return value * 0.9
    raise ValueError(f"unknown coarse scale direction {direction!r}")


def transform_step(cropped, lats, passes):
    """One timestep's field, already cropped, smoothed `passes` times and sign-adjusted.

    CROPPED FIRST IS THE CALLER'S DUTY and the reason this takes the cropped array: the
    archive parses to the tracking domain and then smooths, so the smoother meets the
    domain edge as a boundary, and smoothing a buffered field before cropping was a
    measured source-fidelity error. `passes` is 1 for the code's behaviour and 2 for the
    paper's description; they are different transformations, not a refinement.
    """
    if passes < 1:
        raise ValueError(f"smoothing passes must be at least 1, got {passes}")
    out = cropped
    for _ in range(passes):
        out = clim.smooth9(out)
    return clim.southern_hemisphere_sign(out[np.newaxis, ...], lats)[0]


class PopulationFiles:
    """The two transformed populations on disk, plus the bookkeeping both estimators need."""

    def __init__(self, scratch_dir):
        self.fine_path = os.path.join(scratch_dir, "fine_population.f64")
        self.coarse_path = os.path.join(scratch_dir, "coarse_population.f64")
        self.fine_cells = None
        self.coarse_cells = None
        self.fine_shape = None                     # (rows, cols) inside the domain
        self.coarse_shape = None
        self.fine_lats = self.fine_lons = None     # the cropped coordinate vectors
        self.coarse_lats = self.coarse_lons = None
        self.steps = 0

    def upper_bound(self, which):
        cells = self.fine_cells if which == "fine" else self.coarse_cells
        return cells * self.steps


def build_population_files(years, directory, prefix, climatology, scratch_dir, *,
                           passes, lat_range, lon_range, coarse_resolution,
                           subsample=1):
    """Write the transformed fine and coarse populations for `years`, one pass, to disk.

    `years` are the POPULATION years. The climatology is handed in already built over its
    OWN years, and this function never sees them, which is the period separation made
    structural rather than remembered.
    """
    from . import load as L
    from . import pipeline as P

    files = PopulationFiles(scratch_dir)
    with open(files.fine_path, "wb") as fine_out, \
            open(files.coarse_path, "wb") as coarse_out:
        for year in sorted(years):
            times, latgrid, longrid, curvature = L.curvature_for_year(
                year, directory, prefix, subsample=subsample)
            anomaly = clim.curvature_anomaly(
                curvature, P.days_to_datetime64(times), climatology)
            del curvature
            lats, lons = latgrid[:, 0], longrid[0, :]
            rows = np.where((lats >= lat_range[0]) & (lats <= lat_range[1]))[0]
            cols = np.where((lons >= lon_range[0]) & (lons <= lon_range[1]))[0]
            native = abs(float(lats[1] - lats[0]))
            _, stride = clim.decimation_shape(native, coarse_resolution)
            clats, clons = lats[::stride], lons[::stride]
            crows = np.where((clats >= lat_range[0]) & (clats <= lat_range[1]))[0]
            ccols = np.where((clons >= lon_range[0]) & (clons <= lon_range[1]))[0]

            fine_cells = rows.size * cols.size
            coarse_cells = crows.size * ccols.size
            if files.fine_cells is None:
                files.fine_cells, files.coarse_cells = fine_cells, coarse_cells
                files.fine_shape = (int(rows.size), int(cols.size))
                files.coarse_shape = (int(crows.size), int(ccols.size))
                files.fine_lats, files.fine_lons = lats[rows].copy(), lons[cols].copy()
                files.coarse_lats = clats[crows].copy()
                files.coarse_lons = clons[ccols].copy()
            elif (files.fine_cells, files.coarse_cells) != (fine_cells, coarse_cells):
                raise ValueError(
                    f"{year} yields a {fine_cells}/{coarse_cells}-cell domain where "
                    f"earlier years gave {files.fine_cells}/{files.coarse_cells}; a "
                    f"population cannot mix grids")

            for step in range(anomaly.shape[0]):
                fine = transform_step(anomaly[step][np.ix_(rows, cols)],
                                      lats[rows], passes)
                fine.astype(np.float64).tofile(fine_out)
                decimated = clim.gaussian_decimate(anomaly[step], native,
                                                   coarse_resolution)
                coarse = transform_step(decimated[np.ix_(crows, ccols)],
                                        clats[crows], passes)
                coarse.astype(np.float64).tofile(coarse_out)
            files.steps += int(anomaly.shape[0])
            del anomaly

    for which, path, cells in (("fine", files.fine_path, files.fine_cells),
                               ("coarse", files.coarse_path, files.coarse_cells)):
        written = os.path.getsize(path) // 8
        expected = cells * files.steps
        if written != expected:
            raise AssertionError(
                f"the {which} population file holds {written} values where the domain "
                f"arithmetic says {expected}; the writer and the bookkeeping disagree "
                f"and neither can be trusted")
    return files


def file_chunks(path, total_values, block_values):
    """A chunks() callable over a raw float64 file, in fixed blocks, for exact_percentile."""
    def chunks():
        offset = 0
        while offset < total_values:
            count = min(block_values, total_values - offset)
            yield np.fromfile(path, dtype=np.float64, count=count, offset=offset * 8)
            offset += count
    return chunks


def exact_threshold(files, which, q):
    """The exact percentile of one population file, with its finite count, upper-checked.

    THE COUNT RULE, executable: a finite count above the domain upper bound is an ERROR
    and is refused. On the buffered one-degree tree the count EQUALS the bound (both
    primary cells recorded it so): the halo is wider than the derivative's two-cell edge,
    the smoother leaves its outer ring raw rather than non-finite, and the retrieval
    carries no masked cells, so no non-finite value reaches the domain. A deficit is
    therefore not the expected case on that tree and calls for inspection of the
    non-finite mask before the run is used. An earlier version of this docstring said a
    deficit was expected, which described a chain that does not reach the domain.
    """
    path = files.fine_path if which == "fine" else files.coarse_path
    cells = files.fine_cells if which == "fine" else files.coarse_cells
    total = cells * files.steps
    value, count = exact_percentile(
        file_chunks(path, total, cells * BLOCK_STEPS), q)
    if count > total:
        raise AssertionError(
            f"the {which} finite count {count} exceeds the domain upper bound {total}, "
            f"which cannot happen for a population the builder wrote; refuse everything")
    return value, count, total


def ladder_draw_chunks(path, cells, steps, per_step, seed):
    """Draws for one seed at one level, REGENERATED identically on every walk.

    A fresh pinned generator per call is what makes the sample a value rather than a
    stored object: exact_percentile walks the data three times, and each walk must see
    the same draws. Within a timestep whose finite count is at or under `per_step` the
    whole field is taken WITHOUT consuming the generator, mirroring `_draw`'s exact
    branch, so a timestep going exact does not shift any later timestep's sample.
    """
    def chunks():
        rng = pinned_generator(seed)
        for start in range(0, steps, BLOCK_STEPS):
            stop = min(start + BLOCK_STEPS, steps)
            block = np.fromfile(path, dtype=np.float64,
                                count=(stop - start) * cells, offset=start * cells * 8)
            drawn = []
            for t in range(stop - start):
                finite = block[t * cells:(t + 1) * cells]
                finite = finite[np.isfinite(finite)]
                if finite.size <= per_step:
                    drawn.append(finite)
                else:
                    drawn.append(finite[rng.choice(finite.size, size=per_step,
                                                   replace=False)])
            yield np.concatenate(drawn) if drawn else np.array([])
    return chunks


def ladder_counts(path, cells, steps, per_step):
    """Per-level count bookkeeping, identical for every seed by construction.

    Returns (selected_total, available_total, exact_timestep_count, total_timesteps).
    The draw sizes are min(per_step, available) deterministically, so the counts do not
    depend on the seed, and the artifact records the same arrays five times because the
    schema says per-seed and honesty says they are equal.
    """
    selected = available = exact_steps = 0
    for start in range(0, steps, BLOCK_STEPS):
        stop = min(start + BLOCK_STEPS, steps)
        block = np.fromfile(path, dtype=np.float64,
                            count=(stop - start) * cells, offset=start * cells * 8)
        for t in range(stop - start):
            n_finite = int(np.isfinite(block[t * cells:(t + 1) * cells]).sum())
            n_sel = min(per_step, n_finite)
            selected += n_sel
            available += n_finite
            exact_steps += int(n_sel == n_finite)
    return selected, available, exact_steps, steps


def between_seed(estimates):
    """The contract's stability block for one threshold at one level.

    Median of the five, relative range (max - min) / abs(median), and a REFUSAL rather
    than a division when the median is zero or non-finite. The absolute value is not
    decoration; this project has produced a negative single-seed threshold estimate.
    """
    estimates = [float(e) for e in estimates]
    median = float(np.median(estimates))
    if median == 0.0 or not np.isfinite(median):
        raise ValueError(
            f"the between-seed median is {median}, which cannot anchor a relative "
            f"range; the fallback run is invalid rather than divided through")
    relative_range = (max(estimates) - min(estimates)) / abs(median)
    return {"estimates": estimates, "median": median,
            "relative_range": float(relative_range),
            "passes": bool(relative_range < GATE)}


def run_ladder(files, seeds=SEEDS, ladder=LADDER):
    """The full fallback, per the contract: first level where BOTH thresholds pass.

    Returns (outcome, level_records). `outcome` is the passing level's record or None
    for exhaustion, which the caller must treat as an INVALID RUN, never a result;
    exhaustion means exact mode or nothing. Every attempted level is in `level_records`
    including the failures, so escalation is legible and the sample size cannot be tuned
    invisibly.
    """
    records = []
    for per_step in ladder:
        entry = {"per_step": int(per_step)}
        for which, path, cells in (("coarse", files.coarse_path, files.coarse_cells),
                                   ("fine", files.fine_path, files.fine_cells)):
            q = COARSE_Q if which == "coarse" else FINE_Q
            estimates = []
            for seed in seeds:
                value, _ = exact_percentile(
                    ladder_draw_chunks(path, cells, files.steps, per_step, seed), q)
                estimates.append(value)
            sel, avail, exact_steps, total_steps = ladder_counts(
                path, cells, files.steps, per_step)
            block = between_seed(estimates)
            block["selected_counts"] = [sel] * len(seeds)
            block["available_finite_counts"] = [avail] * len(seeds)
            block["exact_timestep_count"] = exact_steps
            block["total_timestep_count"] = total_steps
            entry[which] = block
        records.append(entry)
        if entry["coarse"]["passes"] and entry["fine"]["passes"]:
            return entry, records
    return None, records


# --- the matrix-case registry, so a label cannot outrun its settings ---------------------

# The twelve frozen cells of the project's threshold run matrix (two period
# definitions crossed with six transformations), as executable settings. A review demonstrated a ladder T5 diagnostic labelled "P1-T0" writing a
# successful artifact, which defeats the point of freezing the matrix before results, so
# a case id matching this pattern is a CLAIM the program checks, not a comment it copies.
import re as _re

MATRIX_CASE_PATTERN = _re.compile(r"^P[12]-T[0-5]$")
V1_ERAINT_EXPECT = (7.16e-7, 2.80e-6)


def matrix_case(case_id):
    """The frozen settings a matrix identifier claims, or None for a free-form label."""
    if case_id is None or not MATRIX_CASE_PATTERN.match(case_id):
        return None
    period, transformation = case_id.split("-")
    return {
        "climatology_years": (1981, 2010) if period == "P1" else (1980, 2010),
        "population_years": (1979, 2010) if period == "P1" else (1980, 2010),
        "transformation": transformation,
        "estimator": "exact",
        "lat_range": (-35.0, 35.0),
        "lon_range": (-140.0, 40.0),
        "coarse_resolution": 2.5,
        "subsample": 1,
        "prefix": "eraint",
        "expect": V1_ERAINT_EXPECT,
        # the grids the domain must produce on the one-degree tree; cell counts alone
        # cannot catch a shifted grid with the same dimensions, so shapes are part of it.
        # coarse_shape also protects the PARITY condition in expected_buffered_grid():
        # a buffer starting on an odd latitude gives 36 coarse rows, not 35, and its
        # coarse cells would no longer be version 1's
        "fine_shape": (71, 181),
        "coarse_shape": (35, 90),
    }


def validate_case_settings(case, settings):
    """Every way the supplied settings differ from what the case id claims.

    Returns a list of violation strings, empty when the run is what it says it is. The
    caller refuses on any violation BEFORE computing anything, because a mislabelled
    artifact is worse than no artifact.
    """
    out = []
    for key in ("climatology_years", "population_years", "transformation", "estimator",
                "lat_range", "lon_range", "coarse_resolution", "subsample", "prefix",
                "expect"):
        want, got = case[key], settings.get(key)
        if isinstance(want, tuple):
            mismatch = got is None or tuple(got) != tuple(want)
        else:
            mismatch = got != want
        if mismatch:
            out.append(f"{key}: the case requires {want!r}, the run supplies {got!r}")
    if not settings.get("out"):
        out.append("out: a matrix cell requires an output artifact")
    return out


# --- the input preflight, so a truncated or drifting tree cannot pass silently -----------

def _singleton_level_hpa(ds):
    """(level_hpa, None), (None, None) for a file with no level structure, or a refusal.

    A level DIMENSION without a readable coordinate is a refusal, since the level then
    cannot be established, and unestablished never means correct.
    """
    for name in ("pressure_level", "level", "plev", "isobaricInhPa"):
        if name in ds.variables:
            values = np.asarray(ds[name][:], dtype=float).ravel()
            if values.size != 1:
                return None, f"{name} holds {values.size} levels, expected one"
            units = str(getattr(ds[name], "units", "")).lower()
            if units in ("hpa", "millibars", "millibar", "mb"):
                return float(values[0]), None
            if units in ("pa", "pascal", "pascals"):
                return float(values[0]) / 100.0, None
            return None, f"{name} has unrecognised units {units!r}"
        if name in ds.dimensions and len(ds.dimensions[name]) >= 1:
            return None, (f"a {name} dimension exists with no coordinate variable, so "
                          f"the level cannot be established")
    return None, None


def preflight_years(years, directory, prefix, *, require_full_calendar=True,
                    expected_lat=None, expected_lon=None, expected_level_hpa=700.0):
    """Refuse inputs the estimators would otherwise quietly compute over.

    Checks, per year and per wind component: the u and v files carry IDENTICAL latitude
    and longitude arrays (the loader reads coordinates from u alone, so a divergent v
    would pass unseen); timestamps are strictly six-hourly with no gaps or duplicates;
    and, when `require_full_calendar`, the year covers its whole calendar from January 1
    00Z to December 31 18Z, 1460 or 1464 steps. Across years, the coordinate arrays must
    be BIT-IDENTICAL to the first year's, because a grid that shifts while keeping its
    shape survives every cell-count check downstream.

    Returns the metadata the artifact records: per-year step counts, the grid's bounds,
    resolution, orientation and shape. Raises ValueError on any violation, naming it.
    """
    import calendar as _calendar
    import datetime as _datetime

    import netCDF4 as _nc

    from . import load as _L

    ref_lat = ref_lon = None
    steps_per_year = {}
    if expected_lat is not None:
        expected_lat = np.asarray(expected_lat, dtype=float)
        expected_lon = np.asarray(expected_lon, dtype=float)
    for year in sorted(set(int(y) for y in years)):
        coords = {}
        times = {}
        levels = {}
        for var, name in (("u700", "u"), ("v700", "v")):
            path = os.path.join(directory, f"{prefix}_{var}_{year}_6h_region.nc")
            with _nc.Dataset(path) as ds:
                coords[name] = (np.asarray(ds["latitude"][:], dtype=float),
                                np.asarray(ds["longitude"][:], dtype=float))
                times[name] = _L._time_days(ds)
                levels[name] = _singleton_level_hpa(ds)
        # THE PRESSURE LEVEL IS REQUIRED EVIDENCE WHEN ONE IS EXPECTED. The loader
        # squeezes a singleton level axis without reading its coordinate, so an 850 hPa
        # field in a file named u700 would run a named case at the wrong level silently.
        # A FIRST VERSION CHECKED THE VALUE ONLY WHEN A FILE HAPPENED TO CARRY ONE, so
        # expecting 700 hPa did not require evidence of 700 hPa: a review executed a
        # complete level-free pair and the preflight accepted it while the downloader
        # refused the same file, and "unestablished never means passed" was being
        # claimed by a function that passed the unestablished. With an expectation set,
        # each file must PROVIDE a readable singleton coordinate, the u and v values
        # must agree, and the value must be the expected one. Level-free synthetic
        # fixtures opt out EXPLICITLY with expected_level_hpa=None.
        for name, (value, reason) in levels.items():
            if reason is not None:
                raise ValueError(f"{year} {name}: {reason}")
            if expected_level_hpa is not None:
                if value is None:
                    raise ValueError(
                        f"{year} {name}: no pressure-level coordinate, and "
                        f"{expected_level_hpa} hPa is expected; absence is not "
                        f"evidence, so the level must be established, not assumed")
                if value != float(expected_level_hpa):
                    raise ValueError(
                        f"{year} {name}: pressure level {value} hPa where "
                        f"{expected_level_hpa} was expected; the wrong level would "
                        f"run silently to a wrong threshold")
        # AGREEMENT IS CHECKED UNCONDITIONALLY when both files state a level. Inside
        # the expectation gate this clause was shadowed, since each file is compared to
        # the expectation first, and a mutation deleting it survived; its independent
        # force is under the explicit opt-out, where u at 700 and v at 850 is an
        # integrity failure no expectation is needed to see.
        if (levels["u"][0] is not None and levels["v"][0] is not None
                and levels["u"][0] != levels["v"][0]):
            raise ValueError(
                f"{year}: the u and v files carry different pressure levels, "
                f"{levels['u'][0]} against {levels['v'][0]} hPa")
        level_seen = levels["u"][0]
        if not (np.array_equal(coords["u"][0], coords["v"][0])
                and np.array_equal(coords["u"][1], coords["v"][1])):
            raise ValueError(
                f"{year}: the u and v files carry different coordinate arrays, and the "
                f"loader reads coordinates from u alone, so v would be mislocated "
                f"silently")
        # THE V TIME AXIS TOO, same one-second absolute tolerance as the loader, zero
        # relative. A first version read both axes and validated only u's, so the
        # guarantee the docstring claimed was the loader's, not this function's.
        one_second = 1.0 / 86400.0
        if (times["u"].shape != times["v"].shape
                or not np.allclose(times["u"], times["v"], rtol=0.0, atol=one_second)):
            raise ValueError(
                f"{year}: the u and v files do not share a time axis; pairing them by "
                f"position downstream would silently combine different timesteps")
        lat, lon = coords["u"]
        if ref_lat is None:
            ref_lat, ref_lon = lat, lon
            # A NAMED CASE FREEZES THE EXACT COORDINATE VECTORS, not just shapes. A
            # review moved one latitude from 0.0 to 0.125 on a full-size buffered grid
            # and every shape check passed while the derivative changed, so for matrix
            # runs the arrays themselves are compared, by value, before thirty years of
            # computation starts.
            if expected_lat is not None and not (
                    np.array_equal(lat, expected_lat)
                    and np.array_equal(lon, expected_lon)):
                raise ValueError(
                    f"{year}: the grid's coordinates differ from the frozen case's "
                    f"expected vectors (shapes {lat.shape}/{lon.shape} against "
                    f"{expected_lat.shape}/{expected_lon.shape}); a shifted coordinate "
                    f"changes the derivative while passing every shape check")
        elif not (np.array_equal(lat, ref_lat) and np.array_equal(lon, ref_lon)):
            raise ValueError(
                f"{year}: the coordinate arrays differ from the first year's; a grid "
                f"that shifts between years while keeping its shape survives every "
                f"cell-count check, so it is refused here by value")
        t = times["u"]
        diffs = np.diff(t)
        if t.size < 2 or np.any(np.abs(diffs - 0.25) > one_second):
            raise ValueError(
                f"{year}: timestamps are not strictly six-hourly (gaps, duplicates or "
                f"irregular spacing); the population would silently misweight the year")
        if require_full_calendar:
            days = 366 if _calendar.isleap(year) else 365
            expected = days * 4
            jan1 = float((_datetime.date(year, 1, 1)
                          - _datetime.date(1900, 1, 1)).days)
            if t.size != expected or abs(t[0] - jan1) > one_second:
                raise ValueError(
                    f"{year}: {t.size} timesteps starting at day {t[0]:.2f} where a "
                    f"complete calendar year is {expected} from day {jan1:.2f}; a "
                    f"truncated retrieval must not become an artifact claiming the "
                    f"full year")
        steps_per_year[year] = int(t.size)
    return {
        "steps_per_year": steps_per_year,
        "complete_calendar_required": bool(require_full_calendar),
        "n_lat": int(ref_lat.size), "n_lon": int(ref_lon.size),
        "lat_first": float(ref_lat[0]), "lat_last": float(ref_lat[-1]),
        "lon_first": float(ref_lon[0]), "lon_last": float(ref_lon[-1]),
        "native_resolution": float(abs(ref_lat[1] - ref_lat[0])),
        "row_order": "descending" if ref_lat[0] > ref_lat[-1] else "ascending",
        "lat_sha256": _array_sha256(ref_lat), "lon_sha256": _array_sha256(ref_lon),
        "level_hpa": level_seen,
    }


def _array_sha256(values):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(values, dtype=np.float64)
                          .tobytes()).hexdigest()


def expected_buffered_grid():
    """The frozen coordinate vectors of version 1's buffered one-degree retrieval.

    Latitude 50 down to -50 and longitude -155 to 55, whole degrees, 101 by 211,
    descending rows, which is what p1_data_eraint.m LOADS for the vorticity calculation
    and what every named case runs on. It is not the tree version 1's tracker ran on:
    p1_data_eraint.m SAVES an 81 by 191 crop (40N-40S, 145W-45E) and
    p2_track_eraint_700hPa.m decimates and crops that. Inside the tracking domain the
    two routes give bit-identical coarse cells (checked on a real timestep), and the
    reason is parity: both trees start on an even latitude and an odd longitude, so a
    stride-2 subsample from index 0 lands on the same cells and the 2 by 2 window reads
    the same neighbours. A buffer starting at 45 would put the coarse grid on odd
    latitudes; the `coarse_shape` check in matrix_case() is what refuses that.
    """
    return 50.0 - np.arange(101, dtype=float), -155.0 + np.arange(211, dtype=float)


FREE_CASE_PREFIXES = ("diag-", "smoke-", "test-")


def classify_case_id(case_id):
    """What a case id is allowed to mean: 'none', 'matrix', 'free', or 'reserved'.

    THE MATRIX-LOOKING NAMESPACE IS RESERVED, because a typo in an intended matrix
    identifier (P1-T6, p1-t0, P1_T0) previously fell through to a free-form label and
    ran UNPROTECTED, which made a misspelled primary case weaker than a spelled one.
    Anything that is not a valid cell must either carry an explicit diagnostic prefix
    or be refused with the valid names listed.
    """
    if case_id is None:
        return "none"
    if MATRIX_CASE_PATTERN.match(case_id):
        return "matrix"
    if any(case_id.startswith(p) for p in FREE_CASE_PREFIXES):
        return "free"
    return "reserved"
