"""Tests for the threshold estimators against their frozen contracts.

THE MUTATION LIST, WRITTEN BEFORE THE ASSERTIONS EXISTED, executable as
tests/mutations_v1port_thresholds.py.

    H1  the smoothing pass count is ignored, so T1 silently runs T0's transformation
    H2  the sign adjustment is dropped from the transformed population
    H3  the coarse scale is applied in the wrong direction, or not at all
    H4  the ladder median is replaced by the mean
    H5  the absolute value is dropped from the between-seed range denominator
    H6  the gate accepts equality, <= 0.01 instead of < 0.01
    H7  ladder exhaustion returns the last level as if it were a result
    H8  the first passing level is not honoured
    H9  the generator is default_rng instead of pinned PCG64
    H10 the exact-draw branch consumes the generator, shifting later timesteps' draws
    H11 the finite count's upper-bound refusal is dropped
    H12 the builder's written-equals-arithmetic assertion is dropped
    H13 selected counts report the per_step REQUEST instead of what was drawn
    H14 a zero between-seed median is divided by instead of refused

WHY THE CONTRACTS ARE THE SPECIFICATION HERE. Every rule these tests bind was frozen in
the project's written threshold contract (the two primary cases, the fallback
estimator and the exact-mode implementation contract) before this module existed,
several of them corrected repeatedly by review on the way. The tests cite the rule, not the code, so a
future edit that contradicts the contract fails a test naming the contract.
"""
import datetime
import os

import numpy as np
import pytest

nc = pytest.importorskip("netCDF4")

from aew.v1port import climatology as clim  # noqa: E402
from aew.v1port import load as L  # noqa: E402
from aew.v1port import pipeline as P  # noqa: E402
from aew.v1port import thresholds as T  # noqa: E402


# --- fixtures ---------------------------------------------------------------------------

def write_year(directory, year, n_steps=8):
    """One synthetic year on a one-degree grid, winds varying at the grid scale."""
    lat = 22.0 - 1.0 * np.arange(20)                 # 22N..3N
    lon = -12.0 + 1.0 * np.arange(24)                # 12W..11E
    LO, LA = np.meshgrid(lon, lat)
    start = (datetime.date(year, 1, 1) - datetime.date(1970, 1, 1)).days * 86400
    seconds = [start + 21600 * i for i in range(n_steps)]
    for var, name in (("u700", "u"), ("v700", "v")):
        base = (np.sin(np.deg2rad(120 * LO)) * np.cos(np.deg2rad(150 * LA))
                if name == "u" else
                np.cos(np.deg2rad(140 * LO)) * np.sin(np.deg2rad(110 * LA)))
        scale = -8.0 if name == "u" else 3.0
        values = np.stack([scale * base + 0.31 * step * (1 if name == "u" else -1)
                           + 0.17 * year % 3 for step in range(n_steps)])
        with nc.Dataset(os.path.join(
                directory, f"era5_{var}_{year}_6h_region.nc"), "w") as ds:
            ds.createDimension("valid_time", n_steps)
            ds.createDimension("pressure_level", 1)
            ds.createDimension("latitude", lat.size)
            ds.createDimension("longitude", lon.size)
            t = ds.createVariable("valid_time", "i8", ("valid_time",))
            t.units = "seconds since 1970-01-01"
            t[:] = np.asarray(seconds)
            lv = ds.createVariable("pressure_level", "f8", ("pressure_level",))
            lv.units = "hPa"
            lv[:] = [700.0]
            ds.createVariable("latitude", "f8", ("latitude",))[:] = lat
            ds.createVariable("longitude", "f8", ("longitude",))[:] = lon
            ds.createVariable(name, "f4",
                              ("valid_time", "latitude", "longitude"))[:] = values


LAT_RANGE, LON_RANGE = (5.0, 20.0), (-10.0, 9.0)
COARSE_RES = 2.5


@pytest.fixture
def tree(tmp_path):
    for year in (1981, 1982, 1983):
        write_year(str(tmp_path), year)
    return str(tmp_path)


def brute_force_population(tree_dir, population_years, climatology_years, passes):
    """The transformed populations computed the slow, held-in-memory way, as the oracle."""
    climatology = L.build_climatology(climatology_years, tree_dir, "era5")
    fine_all, coarse_all = [], []
    for year in sorted(population_years):
        times, latgrid, longrid, curvature = L.curvature_for_year(year, tree_dir, "era5")
        anomaly = clim.curvature_anomaly(
            curvature, P.days_to_datetime64(times), climatology)
        lats, lons = latgrid[:, 0], longrid[0, :]
        rows = np.where((lats >= LAT_RANGE[0]) & (lats <= LAT_RANGE[1]))[0]
        cols = np.where((lons >= LON_RANGE[0]) & (lons <= LON_RANGE[1]))[0]
        native = abs(float(lats[1] - lats[0]))
        _, stride = clim.decimation_shape(native, COARSE_RES)
        clats, clons = lats[::stride], lons[::stride]
        crows = np.where((clats >= LAT_RANGE[0]) & (clats <= LAT_RANGE[1]))[0]
        ccols = np.where((clons >= LON_RANGE[0]) & (clons <= LON_RANGE[1]))[0]
        for step in range(anomaly.shape[0]):
            fine_all.append(T.transform_step(
                anomaly[step][np.ix_(rows, cols)], lats[rows], passes).ravel())
            decimated = clim.gaussian_decimate(anomaly[step], native, COARSE_RES)
            coarse_all.append(T.transform_step(
                decimated[np.ix_(crows, ccols)], clats[crows], passes).ravel())
    return np.concatenate(fine_all), np.concatenate(coarse_all)


def build(tree_dir, scratch, population_years, climatology_years, passes=1):
    climatology = L.build_climatology(climatology_years, tree_dir, "era5")
    return T.build_population_files(
        population_years, tree_dir, "era5", climatology, scratch,
        passes=passes, lat_range=LAT_RANGE, lon_range=LON_RANGE,
        coarse_resolution=COARSE_RES)


# --- the exact estimator ----------------------------------------------------------------

def test_exact_thresholds_are_bit_equal_to_the_held_computation(tree, tmp_path):
    """The whole point: files plus streaming equals the population held in memory."""
    scratch = str(tmp_path / "scratch"); os.makedirs(scratch)
    files = build(tree, scratch, [1982, 1983], [1981, 1982, 1983])
    fine_ref, coarse_ref = brute_force_population(
        tree, [1982, 1983], [1981, 1982, 1983], passes=1)
    for which, ref, q in (("fine", fine_ref, T.FINE_Q), ("coarse", coarse_ref, T.COARSE_Q)):
        value, count, upper = T.exact_threshold(files, which, q)
        finite = ref[np.isfinite(ref)]
        assert count == finite.size, f"{which}: the finite count must match the oracle"
        assert count <= upper, "the count rule's expected side"
        assert value == float(np.percentile(finite, q, method="linear")), \
            f"{which}: the streamed threshold must equal the held one bit for bit"


def test_the_periods_are_genuinely_separate(tree, tmp_path):
    """H-class regression for the defect the old program had. Climatology years and
    population years produce different numbers when they differ, and the builder must be
    using each where it belongs. The oracle recomputes both splits independently."""
    scratch_a = str(tmp_path / "a"); os.makedirs(scratch_a)
    scratch_b = str(tmp_path / "b"); os.makedirs(scratch_b)
    files_a = build(tree, scratch_a, [1982, 1983], [1981, 1982, 1983])
    files_b = build(tree, scratch_b, [1982, 1983], [1982, 1983])
    va, _, _ = T.exact_threshold(files_a, "fine", T.FINE_Q)
    vb, _, _ = T.exact_threshold(files_b, "fine", T.FINE_Q)
    assert va != vb, ("different climatology periods must move the anomaly and so the "
                      "threshold, or the climatology input is not being used")
    fine_ref, _ = brute_force_population(tree, [1982, 1983], [1981, 1982, 1983], 1)
    assert va == float(np.percentile(fine_ref[np.isfinite(fine_ref)], T.FINE_Q,
                                     method="linear"))


def test_two_pass_smoothing_is_a_different_transformation(tree, tmp_path):
    """H1. T1 must not collapse into T0."""
    s1 = str(tmp_path / "p1"); os.makedirs(s1)
    s2 = str(tmp_path / "p2"); os.makedirs(s2)
    f1 = build(tree, s1, [1982], [1981, 1982], passes=1)
    f2 = build(tree, s2, [1982], [1981, 1982], passes=2)
    v1, _, _ = T.exact_threshold(f1, "fine", T.FINE_Q)
    v2, _, _ = T.exact_threshold(f2, "fine", T.FINE_Q)
    assert v1 != v2, "one pass and two passes must give different thresholds"
    with pytest.raises(ValueError, match="at least 1"):
        T.transform_step(np.zeros((4, 4)), np.linspace(10, 5, 4), 0)


def test_the_sign_adjustment_reaches_the_population(tree, tmp_path):
    """H2. A southern-hemisphere row must arrive sign-flipped, or the population is not
    what the tracker gates. Driven directly on transform_step with a row below the
    equator, since the tree fixture is northern-hemisphere only."""
    field = np.ones((3, 4))
    lats = np.array([5.0, 0.0, -5.0])
    out = T.transform_step(field, lats, 1)
    assert np.all(out[0] > 0) and np.all(out[2] < 0), \
        "southern rows must be sign-flipped in the transformed population"


def test_the_coarse_scale_directions_and_refusal():
    """H3. Up is division by 0.9, down is multiplication, None is identity, and an
    unknown direction refuses. The fine threshold NEVER passes through this function in
    the callers, which the CLI test asserts separately."""
    assert T.apply_coarse_scale(0.9, None) == 0.9
    assert T.apply_coarse_scale(0.9, "up") == pytest.approx(1.0)
    assert T.apply_coarse_scale(1.0, "down") == pytest.approx(0.9)
    assert T.apply_coarse_scale(0.9, "up") > 0.9 > T.apply_coarse_scale(0.9, "down")
    with pytest.raises(ValueError, match="unknown"):
        T.apply_coarse_scale(1.0, "sideways")
    assert set(T.TRANSFORMATIONS) == {"T0", "T1", "T2", "T3", "T4", "T5"}
    assert T.TRANSFORMATIONS["T0"] == (1, None)
    assert T.TRANSFORMATIONS["T1"] == (2, None)
    assert T.TRANSFORMATIONS["T4"] == (2, "up") and T.TRANSFORMATIONS["T5"] == (2, "down")


# --- the builder's own guards -----------------------------------------------------------

def test_a_builder_whose_writer_and_arithmetic_disagree_refuses(tree, tmp_path,
                                                                monkeypatch):
    """H12. The written-equals-rows-times-cols-times-steps assertion must be live."""
    scratch = str(tmp_path / "s"); os.makedirs(scratch)
    real = T.transform_step
    calls = {"n": 0}

    def lossy(cropped, lats, passes):
        calls["n"] += 1
        out = real(cropped, lats, passes)
        return out[:-1] if calls["n"] == 5 else out      # drop a row once, mid-stream

    monkeypatch.setattr(T, "transform_step", lossy)
    with pytest.raises((AssertionError, ValueError)):
        build(tree, scratch, [1982], [1981, 1982])


def test_the_upper_bound_refusal_is_live(tree, tmp_path, monkeypatch):
    """H11. A finite count above the domain arithmetic is refused, not recorded."""
    scratch = str(tmp_path / "s"); os.makedirs(scratch)
    files = build(tree, scratch, [1982], [1981, 1982])
    monkeypatch.setattr(T, "exact_percentile",
                        lambda chunks, q, **kw: (1.0e-7, files.upper_bound("fine") + 1))
    with pytest.raises(AssertionError, match="exceeds the domain upper bound"):
        T.exact_threshold(files, "fine", T.FINE_Q)


def test_file_chunks_reassemble_the_file_exactly(tmp_path):
    data = np.arange(1000, dtype=np.float64) * 1.7e-7
    path = str(tmp_path / "pop.f64")
    data.tofile(path)
    chunks = T.file_chunks(path, data.size, 137)
    for walk in range(2):                                # more than one walk, like the user
        got = np.concatenate(list(chunks()))
        np.testing.assert_array_equal(got, data)


# --- the ladder -------------------------------------------------------------------------

def test_the_generator_is_pinned_to_pcg64():
    """H9. The stream is named by CONSTRUCTION, not inherited from a movable default.

    Today `default_rng(seed)` and `Generator(PCG64(seed))` produce identical streams, so
    no behavioural test can tell them apart, and the mutation swapping them survived a
    type check on its output. The contract's whole point is what happens when NumPy
    changes its default, which only the construction determines, so the construction is
    what this asserts: the source names PCG64 and does not reach for default_rng. A
    source assertion is unusual and it is the honest test of a construction contract.
    """
    import inspect

    gen = T.pinned_generator(3)
    assert type(gen.bit_generator).__name__ == "PCG64"
    source = inspect.getsource(T.pinned_generator)
    assert "np.random.PCG64(" in source, \
        "the generator must be constructed from PCG64 by name"
    # the CALL pattern, with the parenthesis, because the function's own docstring
    # mentions default_rng by name while explaining why it is avoided, and a first
    # version of this assertion caught the documentation instead of the code
    assert "default_rng(" not in source, \
        "default_rng inherits whatever NumPy ships, which is the thing being pinned away"


def test_draws_regenerate_identically_and_the_exact_branch_spares_the_generator(tmp_path):
    """H10 and the determinism the streaming percentile depends on.

    Timestep 0 has fewer finite values than per_step and must be taken whole WITHOUT
    consuming the generator, so timestep 1's draw equals the draw a fresh generator
    would produce for it alone. A version that consumes on the exact branch shifts every
    later sample and the between-seed structure quietly changes meaning.
    """
    cells, per_step = 50, 10
    step0 = np.full(cells, np.nan); step0[:4] = [1.0, 2.0, 3.0, 4.0]   # 4 finite: exact
    step1 = np.linspace(10.0, 20.0, cells)                             # 50 finite: sampled
    path = str(tmp_path / "p.f64")
    np.concatenate([step0, step1]).astype(np.float64).tofile(path)

    chunks = T.ladder_draw_chunks(path, cells, 2, per_step, seed=7)
    first_walk = np.concatenate(list(chunks()))
    second_walk = np.concatenate(list(chunks()))
    np.testing.assert_array_equal(first_walk, second_walk)

    rng = T.pinned_generator(7)
    expected_step1 = step1[rng.choice(cells, size=per_step, replace=False)]
    np.testing.assert_array_equal(first_walk, np.concatenate([[1.0, 2.0, 3.0, 4.0],
                                                              expected_step1]))


def test_ladder_counts_are_what_was_drawn_not_what_was_requested(tmp_path):
    """H13. selected is min(per_step, available) summed, and the exact timestep count
    tells the mixed state apart."""
    cells = 6
    step0 = np.array([1.0, 2.0, np.nan, np.nan, np.nan, np.nan])       # 2 finite, exact
    step1 = np.arange(6, dtype=float)                                  # 6 finite, capped
    path = str(tmp_path / "p.f64")
    np.concatenate([step0, step1]).astype(np.float64).tofile(path)
    selected, available, exact_steps, total = T.ladder_counts(path, cells, 2, per_step=4)
    assert (selected, available) == (2 + 4, 2 + 6)
    assert (exact_steps, total) == (1, 2)


def test_between_seed_median_range_and_refusals():
    """H4, H5, H6, H14. The block's arithmetic, including the negative-median case the
    absolute value exists for and the strictness of the gate."""
    block = T.between_seed([1.0e-7, 1.02e-7, 0.99e-7, 1.01e-7, 1.0e-7])
    assert block["median"] == 1.0e-7
    assert block["relative_range"] == pytest.approx(0.03 / 1.0, rel=1e-9)
    assert block["passes"] is False

    negative = T.between_seed([-1.0e-7, -1.001e-7, -0.999e-7, -1.0e-7, -1.0e-7])
    assert negative["relative_range"] > 0, "abs(median) must keep the range positive"
    assert negative["passes"] is True

    # THE BOUNDARY EXACTLY, in float arithmetic that actually lands on it. A first
    # version used [1.0 x4, 1.01], whose subtraction gives 0.010000000000000009, so both
    # < and <= rejected it and the strictness mutation survived. (101 - 100) / 100 is
    # exact division landing on float(0.01) precisely, and only there do the two
    # comparisons part ways.
    exactly_gate = T.between_seed([100.0, 100.0, 100.0, 100.5, 101.0])
    assert exactly_gate["relative_range"] == 0.01, "the fixture must hit the boundary"
    assert exactly_gate["passes"] is False, "the gate is strict, equality fails"

    with pytest.raises(ValueError, match="invalid"):
        T.between_seed([1.0, -1.0, 0.0, 0.5, -0.5])          # median exactly zero
    with pytest.raises(ValueError, match="invalid"):
        T.between_seed([np.nan] * 5)


def test_a_fully_exact_ladder_passes_at_the_first_level(tree, tmp_path):
    """The integration case: a tiny grid where 400 exceeds every timestep's finite count,
    so all five seeds take the whole field, the estimates are identical, the range is
    zero, and the first level is the result with the exact state visible in the counts."""
    scratch = str(tmp_path / "s"); os.makedirs(scratch)
    files = build(tree, scratch, [1982, 1983], [1981, 1982, 1983])
    outcome, records = T.run_ladder(files)
    assert outcome is not None and outcome["per_step"] == 400
    assert len(records) == 1, "a passing first level must not run further levels"
    for which in ("coarse", "fine"):
        block = outcome[which]
        assert len(set(block["estimates"])) == 1, \
            "five whole-field draws are five identical estimates"
        assert block["relative_range"] == 0.0 and block["passes"] is True
        assert block["selected_counts"] == block["available_finite_counts"], \
            "exactness is the equality of the two counts, recomputable from them"
        assert block["exact_timestep_count"] == block["total_timestep_count"]
        # and the estimate is the exact percentile of the population itself
        q = T.COARSE_Q if which == "coarse" else T.FINE_Q
        value, _, _ = T.exact_threshold(files, which, q)
        assert block["median"] == value


def test_escalation_stops_at_the_first_passing_level_and_exhaustion_is_no_result(
        tmp_path, monkeypatch):
    """H7 and H8, on scripted estimates so the level arithmetic is isolated.

    Level 400 spreads wide, level 800 is tight, so the outcome must be the 800 record
    with both attempts recorded. Then every level spreads wide, and the outcome must be
    None, never the last record dressed as a result: the contract says exhaustion means
    exact mode or an invalid run, nothing else.
    """
    files = T.PopulationFiles(str(tmp_path))
    files.fine_cells = files.coarse_cells = 4
    files.steps = 1
    np.full(4, 1.0).tofile(files.fine_path)
    np.full(4, 1.0).tofile(files.coarse_path)

    def scripted(spread_by_level):
        calls = {"i": 0}

        def fake_percentile(chunks, q, **kw):
            level = T.LADDER[min(calls["i"] // 10, len(T.LADDER) - 1)]
            seed = (calls["i"] // 2) % 5
            calls["i"] += 1
            spread = spread_by_level(level)
            return 1.0e-7 * (1.0 + spread * seed), 4
        return fake_percentile

    monkeypatch.setattr(T, "exact_percentile",
                        scripted(lambda level: 0.05 if level == 400 else 0.0005))
    outcome, records = T.run_ladder(files)
    assert outcome is not None and outcome["per_step"] == 800, \
        "the first level where BOTH thresholds pass is the result"
    assert [r["per_step"] for r in records] == [400, 800], \
        "the failing attempt is recorded and no later level runs"

    monkeypatch.setattr(T, "exact_percentile", scripted(lambda level: 0.05))
    outcome, records = T.run_ladder(files)
    assert outcome is None, "exhaustion is an explicit non-result, never the last level"
    assert [r["per_step"] for r in records] == list(T.LADDER), \
        "every attempted level is recorded, including all the failures"


# --- the case registry and the preflight ------------------------------------------------

def test_the_matrix_registry_matches_the_frozen_cells():
    """H15-class. The registry is the contract table, executable."""
    p1 = T.matrix_case("P1-T0")
    assert p1["climatology_years"] == (1981, 2010)
    assert p1["population_years"] == (1979, 2010)
    p2 = T.matrix_case("P2-T5")
    assert p2["climatology_years"] == (1980, 2010) == p2["population_years"]
    assert p2["transformation"] == "T5"
    for case_id in ("P1-T0", "P2-T3"):
        c = T.matrix_case(case_id)
        assert c["estimator"] == "exact"
        assert c["lat_range"] == (-35.0, 35.0) and c["lon_range"] == (-140.0, 40.0)
        assert c["prefix"] == "eraint" and c["expect"] == (7.16e-7, 2.80e-6)
        assert c["fine_shape"] == (71, 181) and c["coarse_shape"] == (35, 90)
    for other in (None, "smoke-3yr", "P3-T0", "P1-T6", "p1-t0", "P1_T0"):
        assert T.matrix_case(other) is None, \
            f"{other!r} is not a cell; whether it may RUN is classify_case_id's question"


def test_case_validation_names_every_difference():
    """A mislabelled run is refused with a diff, and a matching one passes clean."""
    case = T.matrix_case("P1-T0")
    good = {"climatology_years": (1981, 2010), "population_years": (1979, 2010),
            "transformation": "T0", "estimator": "exact",
            "lat_range": (-35.0, 35.0), "lon_range": (-140.0, 40.0),
            "coarse_resolution": 2.5, "subsample": 1, "prefix": "eraint",
            "expect": (7.16e-7, 2.80e-6), "out": "somewhere.json"}
    assert T.validate_case_settings(case, good) == []
    forged = dict(good, transformation="T5", estimator="ladder",
                  population_years=(1982, 1983), expect=None, out=None)
    violations = "\n".join(T.validate_case_settings(case, forged))
    for named in ("transformation", "estimator", "population_years", "expect", "out"):
        assert named in violations, f"the diff must name {named}"
    assert "climatology_years" not in violations, "matching settings are not violations"


def preflight_tree(tmp_path, year=1981, *, steps=None, start_jan1=True, drop_step=None,
                   v_lat_shift=0.0, lat_shift=0.0, level=700.0, omit_level=False):
    """Tiny wind files purpose-built for the preflight, a few KiB per year, carrying the
    700 hPa coordinate genuine retrievals carry unless a test omits it on purpose."""
    import calendar

    lat = np.array([12.0, 11.0, 10.0]) + lat_shift
    lon = np.array([-3.0, -2.0, -1.0, 0.0])
    days = 366 if calendar.isleap(year) else 365
    n = steps if steps is not None else days * 4
    start = (datetime.date(year, 1, 1) - datetime.date(1970, 1, 1)).days * 86400
    if not start_jan1:
        start += 86400 * 30
    seconds = [start + 21600 * i for i in range(n)]
    if drop_step is not None:
        seconds = seconds[:drop_step] + seconds[drop_step + 1:]
    for var, name in (("u700", "u"), ("v700", "v")):
        use_lat = lat + (v_lat_shift if name == "v" else 0.0)
        with nc.Dataset(os.path.join(str(tmp_path),
                                     f"era5_{var}_{year}_6h_region.nc"), "w") as ds:
            ds.createDimension("valid_time", len(seconds))
            ds.createDimension("latitude", use_lat.size)
            ds.createDimension("longitude", lon.size)
            t = ds.createVariable("valid_time", "i8", ("valid_time",))
            t.units = "seconds since 1970-01-01"
            t[:] = np.asarray(seconds)
            if not omit_level:
                ds.createDimension("pressure_level", 1)
                lv = ds.createVariable("pressure_level", "f8", ("pressure_level",))
                lv.units = "hPa"
                lv[:] = [level]
            ds.createVariable("latitude", "f8", ("latitude",))[:] = use_lat
            ds.createVariable("longitude", "f8", ("longitude",))[:] = lon
            ds.createVariable(name, "f4", ("valid_time", "latitude", "longitude"))[:] = \
                np.zeros((len(seconds), use_lat.size, lon.size))


def test_preflight_passes_a_complete_healthy_year(tmp_path):
    preflight_tree(tmp_path, 1981)
    meta = T.preflight_years([1981], str(tmp_path), "era5")
    assert meta["steps_per_year"] == {1981: 1460}
    assert meta["row_order"] == "descending" and meta["native_resolution"] == 1.0
    assert (meta["lat_first"], meta["lat_last"]) == (12.0, 10.0)


def test_preflight_refuses_a_truncated_year(tmp_path):
    """A partial retrieval must not become an artifact claiming the full period."""
    preflight_tree(tmp_path, 1981, steps=1200)
    with pytest.raises(ValueError, match="complete calendar year"):
        T.preflight_years([1981], str(tmp_path), "era5")
    meta = T.preflight_years([1981], str(tmp_path), "era5",
                             require_full_calendar=False)
    assert meta["steps_per_year"] == {1981: 1200}, \
        "the opt-out records what is actually there"


def test_preflight_refuses_a_year_not_starting_at_january_first(tmp_path):
    preflight_tree(tmp_path, 1981, steps=1460, start_jan1=False)
    with pytest.raises(ValueError, match="complete calendar year"):
        T.preflight_years([1981], str(tmp_path), "era5")


def test_preflight_refuses_a_gap_in_the_timestamps(tmp_path):
    """Regularity is checked even when the calendar requirement is waived."""
    preflight_tree(tmp_path, 1981, drop_step=700)
    with pytest.raises(ValueError, match="six-hourly"):
        T.preflight_years([1981], str(tmp_path), "era5",
                          require_full_calendar=False)


def test_preflight_refuses_uv_files_with_different_coordinates(tmp_path):
    """The loader reads coordinates from u alone, so a divergent v passes unseen there
    and must be caught here."""
    preflight_tree(tmp_path, 1981, v_lat_shift=0.5)
    with pytest.raises(ValueError, match="different coordinate arrays"):
        T.preflight_years([1981], str(tmp_path), "era5")


def test_preflight_refuses_a_grid_that_drifts_between_years(tmp_path):
    """Same shape, shifted values: the case every cell-count check downstream misses."""
    preflight_tree(tmp_path, 1981)
    preflight_tree(tmp_path, 1982, lat_shift=0.5)
    with pytest.raises(ValueError, match="differ from the first year"):
        T.preflight_years([1981, 1982], str(tmp_path), "era5")


def test_preflight_refuses_uv_files_with_divergent_time_axes(tmp_path):
    """A v file shifted by one timestep must be caught HERE, because a first version
    read both time axes and validated only u's, so the guarantee this function's
    docstring claimed was really the loader's."""
    preflight_tree(tmp_path, 1981)
    path = os.path.join(str(tmp_path), "era5_v700_1981_6h_region.nc")
    with nc.Dataset(path, "a") as ds:
        times = ds["valid_time"][:]
        ds["valid_time"][:] = times + 21600            # six hours late, all steps
    with pytest.raises(ValueError, match="share a time axis"):
        T.preflight_years([1981], str(tmp_path), "era5")


def test_preflight_refuses_coordinates_that_differ_from_the_frozen_case(tmp_path):
    """The review's demonstration: one latitude moved from its expected value, every
    shape intact, and the derivative changes while all counts pass. The frozen vectors
    catch it by value, before any computation."""
    preflight_tree(tmp_path, 1981)
    lat_want = np.array([12.0, 11.0, 10.0])
    lon_want = np.array([-3.0, -2.0, -1.0, 0.0])
    meta = T.preflight_years([1981], str(tmp_path), "era5",
                             expected_lat=lat_want, expected_lon=lon_want)
    assert meta["n_lat"] == 3, "the matching grid must pass"
    distorted = lat_want.copy(); distorted[1] = 11.125
    with pytest.raises(ValueError, match="frozen case's expected vectors"):
        T.preflight_years([1981], str(tmp_path), "era5",
                          expected_lat=distorted, expected_lon=lon_want)


def test_the_matrix_looking_namespace_is_reserved():
    """A typo in an intended matrix id must not run unprotected as a free label."""
    assert T.classify_case_id(None) == "none"
    assert T.classify_case_id("P1-T0") == "matrix"
    assert T.classify_case_id("smoke-3yr") == "free"
    assert T.classify_case_id("diag-x") == "free"
    assert T.classify_case_id("test-T0") == "free"
    for typo in ("P1-T6", "P3-T0", "p1-t0", "P1_T0", "P2T3", "P1-T0-extra"):
        assert T.classify_case_id(typo) == "reserved", \
            f"{typo!r} must be refused, not silently treated as a diagnostic"


def test_the_frozen_buffered_grid_is_version_ones():
    lat, lon = T.expected_buffered_grid()
    assert lat.size == 101 and lon.size == 211
    assert (lat[0], lat[-1]) == (50.0, -50.0) and (lon[0], lon[-1]) == (-155.0, 55.0)
    assert np.all(np.diff(lat) == -1.0) and np.all(np.diff(lon) == 1.0)


def test_preflight_requires_level_evidence_when_a_level_is_expected(tmp_path):
    """The contract a review had to state twice. Expecting 700 hPa REQUIRES a readable
    700 hPa coordinate in each file; a wrong level refuses, an ABSENT level refuses,
    because absence is not evidence, and a first version accepted a complete level-free
    pair while claiming otherwise. Level-free fixtures opt out with None, explicitly."""
    preflight_tree(tmp_path, 1981)
    meta = T.preflight_years([1981], str(tmp_path), "era5")
    assert meta["level_hpa"] == 700.0, "the genuine-shaped fixture carries 700"

    preflight_tree(tmp_path, 1981, level=850.0)
    with pytest.raises(ValueError, match="850"):
        T.preflight_years([1981], str(tmp_path), "era5", expected_level_hpa=700.0)
    meta = T.preflight_years([1981], str(tmp_path), "era5", expected_level_hpa=850.0)
    assert meta["level_hpa"] == 850.0


def test_preflight_refuses_a_missing_level_when_one_is_expected(tmp_path):
    """The review's executed case: a complete, well-timed, well-gridded pair with NO
    level structure at all, which the downloader refused and the preflight accepted."""
    preflight_tree(tmp_path, 1981, omit_level=True)
    with pytest.raises(ValueError, match="absence is not evidence"):
        T.preflight_years([1981], str(tmp_path), "era5", expected_level_hpa=700.0)
    meta = T.preflight_years([1981], str(tmp_path), "era5", expected_level_hpa=None)
    assert meta["level_hpa"] is None, "the explicit opt-out records the absence"


def test_preflight_refuses_uv_files_at_different_levels(tmp_path):
    """Agreement holds even under the explicit opt-out, where no expectation shields
    it: u at 700 and v at 850 is an integrity failure regardless of what was expected,
    and inside the expectation gate this check was shadowed by the per-file comparison,
    which a surviving mutation exposed."""
    preflight_tree(tmp_path, 1981)
    path = os.path.join(str(tmp_path), "era5_v700_1981_6h_region.nc")
    with nc.Dataset(path, "a") as ds:
        ds["pressure_level"][:] = [850.0]
    with pytest.raises(ValueError, match="different pressure levels|850"):
        T.preflight_years([1981], str(tmp_path), "era5", expected_level_hpa=700.0)
    with pytest.raises(ValueError, match="different pressure levels"):
        T.preflight_years([1981], str(tmp_path), "era5", expected_level_hpa=None)


def test_preflight_refuses_a_level_axis_it_cannot_read(tmp_path):
    """The unestablished case in isolation: a pressure_level DIMENSION with no
    coordinate variable. Nothing says what level this is, and unestablished never means
    passed, so it refuses rather than assuming. The 850 test above cannot catch a
    mutation here, because there the coordinate exists and is merely wrong."""
    preflight_tree(tmp_path, 1981, omit_level=True)
    for var in ("u700", "v700"):
        path = os.path.join(str(tmp_path), f"era5_{var}_1981_6h_region.nc")
        with nc.Dataset(path, "a") as ds:
            ds.createDimension("pressure_level", 1)   # a dimension and no variable
    with pytest.raises(ValueError, match="cannot be established"):
        T.preflight_years([1981], str(tmp_path), "era5")
    # and the explicit opt-out does NOT excuse an unreadable axis, only a missing one
    with pytest.raises(ValueError, match="cannot be established"):
        T.preflight_years([1981], str(tmp_path), "era5", expected_level_hpa=None)


def test_transform_step_matches_a_literal_transcription_of_smth9_f():
    """AN ORACLE THAT SHARES NOTHING WITH smooth9. The bit-equality test above imports
    T.transform_step into its brute-force population, so it binds the file plumbing and
    the percentile, not the smoother's weights. This one transcribes smth9_f.m as the
    MATLAB double loop, applies the same sign flip by hand, and compares bit for bit on
    a field with a southern row, one pass and two. Found by an implementation review."""
    rng = np.random.default_rng(7)
    field = rng.normal(size=(9, 11))
    lats = np.linspace(4.0, -4.0, 9)          # crosses the equator, descending rows
    p, q = 0.5, 0.25

    def smth9_literal(f):
        ny, nx = f.shape
        out = f.copy()
        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                f0 = f[i, j]
                # smth9_f.m's own summation order, which bit-equality depends on:
                #   x(i-1,j)+x(i,j-1)+x(i+1,j)+x(i,j+1)  and
                #   x(i-1,j+1)+x(i-1,j-1)+x(i+1,j-1)+x(i+1,j+1)
                edge = f[i - 1, j] + f[i, j - 1] + f[i + 1, j] + f[i, j + 1]
                diag = f[i - 1, j + 1] + f[i - 1, j - 1] + f[i + 1, j - 1] + f[i + 1, j + 1]
                out[i, j] = f0 + (p / 4.0) * (edge - 4.0 * f0) + (q / 4.0) * (diag - 4.0 * f0)
        return out

    for passes in (1, 2):
        want = field
        for _ in range(passes):
            want = smth9_literal(want)
        want = want.copy()
        want[lats < 0.0] *= -1.0
        got = T.transform_step(field, lats, passes)
        assert np.array_equal(got, want), f"{passes} pass(es) differ from the transcription"
    # the outer ring is untouched by the smoother and flipped only by the sign step
    got = T.transform_step(field, lats, 1)
    assert np.array_equal(got[0], field[0]) and np.array_equal(got[-1], -field[-1])
