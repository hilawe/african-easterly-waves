"""Tests for the whole tracker run end to end.

THE MUTATION LIST, WRITTEN BEFORE THE ASSERTIONS EXISTED, executable as
tests/mutations_v1port_pipeline.py.

    P1   the timestep order is wrong: pruning before association, or either running on a
         timestep that produced no candidates
    P2   the coarse grid is not decimated, so detection runs at the wrong resolution
    P3   the decimation factor is computed from the wrong grid spacing
    P4   the domain subset is skipped, or applied to one grid and not the other
    P5   the fine and coarse grids are passed to detection the wrong way round
    P6   the advection is computed without version 1's latitude flip
    P7   the median winds come from the raw fine field rather than the smoothed one
    P8   the median winds are a plain median rather than one that ignores missing values
    P9   the median is taken over the whole field rather than the wave's own cells
    P10  the exclusivity flag does not reach the association step
    P11  the final filter is not applied, or is applied without the run length
    P12  a reanalysis version 1 never saw silently gets another one's thresholds
    P13  the curvature is computed in the file's own row order, so north-first input gets
         sign-flipped meridional derivatives (the advection defect's sibling, added
         2026-08-29 with the fix; the four mutations were listed before the tests)

WHAT THIS FILE IS FOR, and it is different from every other stage's tests. Those bind one
function against a reading of the MATLAB. These bind the ORDER AND WIRING of eight
functions, which is where independent checking found a defect that every per-module pass
had walked past: two functions each faithful alone and wrong as a pair. A chain has more seams
than parts.
"""

import numpy as np
import pytest

from aew.v1port import pipeline as P
from aew.v1port.association import duplicate_observations

COARSE, FINE = 7.16e-7, 2.80e-6      # version 1's own ERA-Interim 700 hPa pair


# THE FIXTURES ARE ON A 0.75 DEGREE GRID DECIMATING TO 2.5, which is ERA-Interim's own
# ratio and the one real runs use. An earlier version built everything at 2.5 with a
# uniform wind, and nine of this file's mutations survived against it: with one resolution
# the coarse and fine grids are interchangeable, with a uniform wind the smoothing is a
# no-op, and with three healthy waves nothing is ever pruned or filtered. A fixture that
# cannot distinguish two wirings does not test the wiring.
NATIVE_DEG = 0.75
COARSE_DEG = 2.5


def grid(step=NATIVE_DEG, lat_span=(-20.0, 20.0), lon_span=(-60.0, 20.0)):
    lat = np.arange(lat_span[0], lat_span[1] + step / 2, step)
    lon = np.arange(lon_span[0], lon_span[1] + step / 2, step)
    longrid, latgrid = np.meshgrid(lon, lat)
    return latgrid, longrid


DOMAIN = {"lat_range": (-20.0, 20.0), "lon_range": (-60.0, 20.0)}


def structured_wind(latgrid, longrid, nt):
    """An easterly flow with cell-scale structure on it.

    The structure is the point: a uniform field is unchanged by the nine-point smoother, so
    a fixture built on one cannot tell the smoothed median winds from the raw ones. The
    mean stays near -8 m/s so waves still propagate at a plausible speed.
    """
    noise = np.cos(longrid * 7.0) * np.sin(latgrid * 9.0) * 4.0
    u = np.broadcast_to(-8.0 + noise, (nt, *latgrid.shape)).copy()
    v = np.broadcast_to(noise * 0.25, (nt, *latgrid.shape)).copy()
    return u, v


def bump(latgrid, longrid, centre_lon, centre_lat, amplitude=6e-6, width=3.5):
    return amplitude * np.exp(-(((longrid - centre_lon) / width) ** 2
                                + ((latgrid - centre_lat) / 5.0) ** 2))


def waves_moving_west(latgrid, longrid, nt=24, starts=(15.0, -15.0),
                      lat0=8.0, speed_deg=1.55, amplitude=6e-6):
    """Distinct curvature maxima drifting west in a structured easterly flow."""
    u, v = structured_wind(latgrid, longrid, nt)
    anomaly = np.zeros((nt, *latgrid.shape))
    for t in range(nt):
        for start in starts:
            anomaly[t] += bump(latgrid, longrid, start - speed_deg * t, lat0,
                               amplitude=amplitude)
    times = 38351.0 + 0.25 * np.arange(nt)
    return times, u, v, anomaly


def run(times, latgrid, longrid, u, v, anomaly, **kwargs):
    kwargs.setdefault("coarse_threshold", COARSE)
    kwargs.setdefault("fine_threshold", FINE)
    kwargs.setdefault("coarse_resolution_deg", COARSE_DEG)
    for key, value in DOMAIN.items():
        kwargs.setdefault(key, value)
    return P.track_year(times, latgrid, longrid, u, v, anomaly, **kwargs)


# --- the chain produces tracks at all ---------------------------------------------------

def test_separated_waves_become_separate_tracks():
    latgrid, longrid = grid()
    times, u, v, anomaly = waves_moving_west(latgrid, longrid)
    tracks = run(times, latgrid, longrid, u, v, anomaly)
    assert len(tracks) == 2, "one track per seeded wave"
    # Not every track runs every step: with a structured wind the detection can miss a
    # timestep, and a wave seeded near the eastern edge is only found once it has moved
    # clear of it. Requiring the full run would be requiring the fixture to be uniform,
    # which is what made the earlier version of this file unable to test the wiring.
    assert all(len(t["time"]) >= times.size - 2 for t in tracks)


def test_the_tracks_move_west_at_the_speed_of_the_flow():
    """The whole chain is wired correctly only if the tracks come out at the speed the wind
    put in. Eight metres per second westward over the run, within a tolerance that the
    2.5 degree grid's own quantisation sets."""
    from aew.v1port import validate as V

    latgrid, longrid = grid()
    times, u, v, anomaly = waves_moving_west(latgrid, longrid)
    tracks = run(times, latgrid, longrid, u, v, anomaly)
    for track in tracks:
        assert track["meanlon"][-1] < track["meanlon"][0], "waves must travel west"
        speed = float(np.median(V.segment_speeds_ms(track)))
        assert 5.0 < speed < 12.0, f"{speed} m/s is not the 8 m/s the flow imposed"


def test_a_year_with_no_waves_produces_no_tracks():
    latgrid, longrid = grid()
    times, u, v, _ = waves_moving_west(latgrid, longrid)
    flat = np.zeros_like(u)
    assert run(times, latgrid, longrid, u, v, flat) == []


def test_a_field_below_the_threshold_produces_no_tracks():
    """P2 and P4 in part: if the thresholds were not reaching detection, a field two orders
    of magnitude too weak would still produce tracks."""
    latgrid, longrid = grid()
    times, u, v, anomaly = waves_moving_west(latgrid, longrid, amplitude=6e-9)
    assert run(times, latgrid, longrid, u, v, anomaly) == []


# --- the duplication defect, through the whole chain -------------------------------------

def converging_waves(latgrid, longrid, nt=20):
    """Two waves that approach each other, which is what makes one trough claimable twice."""
    shape = (nt, *latgrid.shape)
    u = np.full(shape, -8.0)
    v = np.zeros(shape)
    anomaly = np.zeros(shape)
    for t in range(nt):
        a = 0.0 - 1.55 * t
        b = -18.0 - 0.9 * t                    # closes on the first
        for centre, lat0 in ((a, 11.0), (b, 14.0)):
            anomaly[t] += 6e-6 * np.exp(
                -(((longrid - centre) / 4.5) ** 2 + ((latgrid - lat0) / 6.0) ** 2))
    return 38351.0 + 0.25 * np.arange(nt), u, v, anomaly


def test_the_exclusivity_repair_never_increases_duplication():
    """P10, and the claim the whole port exists to support.

    Version 1's behaviour is the default and the repair is one flag. Whatever the synthetic
    case produces, the repaired run must not hold MORE duplicated observations than the
    faithful one, and it must still produce tracks: a repair that fixed duplication by
    finding nothing would be worse than the defect.
    """
    latgrid, longrid = grid()
    times, u, v, anomaly = converging_waves(latgrid, longrid)
    faithful = run(times, latgrid, longrid, u, v, anomaly, exclusive=False)
    repaired = run(times, latgrid, longrid, u, v, anomaly, exclusive=True)
    assert repaired, "the repair must still track something"
    assert duplicate_observations(repaired) <= duplicate_observations(faithful)


def test_the_two_settings_agree_when_nothing_is_contested():
    """Separated waves give no candidate two tracks could both claim, so the repair must be
    invisible. A repair that changed well-separated tracks would be doing something other
    than removing duplicates."""
    latgrid, longrid = grid()
    times, u, v, anomaly = waves_moving_west(latgrid, longrid)
    faithful = run(times, latgrid, longrid, u, v, anomaly, exclusive=False)
    repaired = run(times, latgrid, longrid, u, v, anomaly, exclusive=True)
    assert len(faithful) == len(repaired)
    for a, b in zip(faithful, repaired):
        assert a["meanlon"] == pytest.approx(b["meanlon"])


def merging_waves(latgrid, longrid, nt=24):
    """Two waves that close on each other until one trough serves both.

    THE SEPARATION AND CONVERGENCE ARE TUNED, and that is not a smell: contention is a
    narrow condition. Too far apart and the tracks never see one candidate; too close and
    they merge into a single wave before either can claim it twice. These values were found
    by measurement, and the test below asserts the contention actually occurred rather than
    trusting them to keep producing it.
    """
    u, v = structured_wind(latgrid, longrid, nt)
    anomaly = np.zeros((nt, *latgrid.shape))
    for t in range(nt):
        anomaly[t] += bump(latgrid, longrid, 10.0 - 1.4 * t, 6.0, width=4.0)
        anomaly[t] += bump(latgrid, longrid, 0.0 - 0.95 * t, 10.0 - 0.1575 * t, width=4.0)
    return 38351.0 + 0.25 * np.arange(nt), u, v, anomaly


def test_the_duplication_defect_reproduces_through_the_whole_chain():
    """THE DEMONSTRATION THE PROJECT RESTS ON, end to end rather than on a hand-built pair
    of tracks.

    Two waves converge until one trough falls inside both tracks' search polygons. Run as
    version 1 runs, both tracks claim it and carry the same observations from then on. Run
    with the one-line repair, only one does. Same code, same data, one flag.
    """
    latgrid, longrid = grid()
    times, u, v, anomaly = merging_waves(latgrid, longrid)
    faithful = run(times, latgrid, longrid, u, v, anomaly, exclusive=False)
    repaired = run(times, latgrid, longrid, u, v, anomaly, exclusive=True)

    faithful_duplicates = duplicate_observations(faithful)
    assert faithful_duplicates > 0, (
        "the fixture stopped producing contention, so this test is no longer testing the "
        "defect; retune the separation and convergence in merging_waves")
    assert duplicate_observations(repaired) == 0
    assert len(repaired) < len(faithful), "the repair collapses the duplicate track"
    assert repaired, "and does not simply lose the wave"


def test_the_final_filter_drops_a_wave_that_does_not_propagate():
    """P11. A stationary feature is a trough version 1 finds and then discards, so a run
    that skipped the filter would carry it into the record."""
    latgrid, longrid = grid()
    nt = 24
    u, v = structured_wind(latgrid, longrid, nt)
    anomaly = np.zeros((nt, *latgrid.shape))
    for t in range(nt):
        anomaly[t] += bump(latgrid, longrid, 10.0 - 1.4 * t, 6.0)
        anomaly[t] += bump(latgrid, longrid, -30.0, 14.0)          # never moves
    times = 38351.0 + 0.25 * np.arange(nt)

    kept = run(times, latgrid, longrid, u, v, anomaly)
    from aew.v1port import validate as V
    for track in kept:
        speed = float(np.median(V.segment_speeds_ms(track)))
        assert speed >= 2.0, "a track below the minimum speed reached the record"
    unfiltered = P.track_year(times, latgrid, longrid, u, v, anomaly,
                              coarse_threshold=COARSE, fine_threshold=FINE,
                              coarse_resolution_deg=COARSE_DEG, **DOMAIN)
    assert len(kept) == len(unfiltered), "sanity: the same call twice"
    assert len(kept) >= 1


def test_a_run_shorter_than_the_minimum_lifetime_is_not_filtered():
    """P11. Version 1 gates the whole final filter on the run having reached its eighth
    timestep, so a short run returns its tracks untouched however slow they are. A filter
    that ran anyway would silently empty a short diagnostic run."""
    latgrid, longrid = grid()
    nt = 5
    u, v = structured_wind(latgrid, longrid, nt)
    anomaly = np.zeros((nt, *latgrid.shape))
    for t in range(nt):
        anomaly[t] += bump(latgrid, longrid, -20.0, 8.0)           # stationary
    times = 38351.0 + 0.25 * np.arange(nt)
    tracks = run(times, latgrid, longrid, u, v, anomaly)
    assert tracks, "a five-step run is below the gate, so nothing should be filtered out"


def test_the_prunes_recency_window_is_what_makes_the_order_moot():
    """Why the catalogue has no mutation for pruning before association instead of after.

    A track the association can extend was matched at most two steps ago, because that is
    as far ahead as a prediction is ever set. Two six-hourly steps is half a day, and the
    prune keeps anything seen within half a day INCLUSIVE. So a track association touches
    would have survived the prune either way, and the order cannot change the outcome.

    That is a coincidence of three constants and not a property of the design, so it is
    pinned here: change any of them and the order starts to matter again.
    """
    from aew.v1port.association import RECENT_DAYS, STEP_HOURS

    furthest_prediction_days = 2 * STEP_HOURS / 24.0
    assert furthest_prediction_days == pytest.approx(RECENT_DAYS), (
        "the prune's recency window no longer equals the longest gap association can "
        "bridge, so pruning before or after it is no longer equivalent and the mutation "
        "withdrawn from the catalogue has to come back")


def test_a_wave_free_timestep_does_not_prune():
    """P1, and the seam found in the pair rather than in either function.

    Version 1's no-wave path is a `continue` in the middle of the timestep loop, and that
    loop does not close until after the prune, so a wave-free step runs neither. The
    fixture is a wave that lasts exactly eight observations and then stops: eight is not
    MORE than the lifetime count, so once it goes stale only the wave-free skip keeps it
    alive. A prune told the step had candidates would drop it.
    """
    latgrid, longrid = grid()
    nt = 20
    u, v = structured_wind(latgrid, longrid, nt)
    anomaly = np.zeros((nt, *latgrid.shape))
    for t in range(nt):
        if t < 8:
            anomaly[t] += bump(latgrid, longrid, 10.0 - 1.4 * t, 6.0)
    times = 38351.0 + 0.25 * np.arange(nt)
    tracks = run(times, latgrid, longrid, u, v, anomaly)
    assert len(tracks) == 1, "the wave should survive the wave-free tail"
    assert len(tracks[0]["time"]) == 8


def test_the_waves_are_advected_by_the_SMOOTHED_wind():
    """P7. The prediction reads the smoothed fine winds, and this fixture makes that the
    difference between tracking and not.

    The wind is zero almost everywhere with sparse strong easterly spikes, so its raw
    median over a trough is zero and its smoothed median is about -9 m/s. Predicting from
    the raw field would leave every wave predicted to stay put while it actually moves, and
    the track would break at the first step.
    """
    latgrid, longrid = grid()
    ny, nx = latgrid.shape
    spikes = np.where((np.arange(ny)[:, None] % 3 == 0)
                      & (np.arange(nx)[None, :] % 3 == 0), -72.0, 0.0)
    from aew.v1port.climatology import smooth9
    assert float(np.nanmedian(spikes)) == pytest.approx(0.0)
    assert float(np.nanmedian(smooth9(spikes))) < -5.0, "the fixture must separate them"

    nt = 16
    u = np.broadcast_to(spikes, (nt, ny, nx)).copy()
    v = np.zeros_like(u)
    anomaly = np.zeros((nt, ny, nx))
    for t in range(nt):
        anomaly[t] += bump(latgrid, longrid, 10.0 - 1.4 * t, 6.0)
    times = 38351.0 + 0.25 * np.arange(nt)
    tracks = run(times, latgrid, longrid, u, v, anomaly)
    assert tracks, "nothing tracked at all"
    longest = max(tracks, key=lambda t: len(t["time"]))
    travelled = longest["meanlon"][0] - longest["meanlon"][-1]
    # The wave moves 1.4 degrees a step. Following it for most of the run means tens of
    # degrees of westward travel; predicting from the raw wind means the prediction never
    # moves, so the track either breaks early or sits still.
    assert travelled > 10.0, (
        f"the track only travelled {travelled:.1f} degrees, which is what happens when the "
        f"prediction is made from the raw wind rather than the smoothed one")


# --- the wiring ---------------------------------------------------------------------------

def test_the_coarse_grid_is_built_at_the_real_domain_without_distortion():
    """P2, at the size where the defect is visible.

    Running the coordinate meshes through the data filter distorts them wherever the
    convolution has nothing to average against. On a small fixture that is a fraction of a
    degree and changes nothing; on version 1's own domain at 0.75 degrees it moves a
    coordinate by eleven degrees and leaves the grid non-monotonic, and everything
    downstream reads positions off this grid. The earlier version of this file only had a
    small fixture, so the mutation survived.
    """
    lat = np.arange(-35.0, 35.1, 0.75)
    lon = np.arange(-140.0, 40.1, 0.75)
    longrid, latgrid = np.meshgrid(lon, lat)
    lat_c, lon_c = P.coarse_grid(latgrid, longrid, 0.75, 2.5)

    lat_values, lon_values = lat_c[:, 0], lon_c[0, :]
    assert np.all(np.diff(lat_values) > 0), "the coarse latitudes must stay monotonic"
    assert np.allclose(np.diff(lat_values), 2.25), "and evenly spaced at stride times 0.75"
    assert lat_values[0] == pytest.approx(-35.0)
    assert lon_values[0] == pytest.approx(-140.0)
    assert np.allclose(np.diff(lon_values), 2.25)


def test_a_coarse_resolution_finer_than_the_grid_is_refused():
    """P3. Version 1 decimates and has no path that refines, so asking for one is a caller
    error rather than something to interpolate around."""
    latgrid, longrid = grid(step=2.5)
    times, u, v, anomaly = waves_moving_west(latgrid, longrid, nt=4)
    with pytest.raises(ValueError, match="finer than the native grid"):
        run(times, latgrid, longrid, u, v, anomaly, coarse_resolution_deg=1.0)


def test_the_domain_subset_is_applied():
    """P4. Waves seeded outside the published domain must not appear in the record, and
    that is the subset's job rather than detection's."""
    lat = np.arange(-60.0, 60.1, 2.5)
    lon = np.arange(-160.0, 80.1, 2.5)
    longrid, latgrid = np.meshgrid(lon, lat)
    times, u, v, anomaly = waves_moving_west(
        latgrid, longrid, nt=16, starts=(70.0,), lat0=50.0, speed_deg=0.0)
    tracks = run(times, latgrid, longrid, u, v, anomaly)
    assert tracks == [], "a wave at 50N and 70E is outside the domain on both axes"


def test_the_median_wind_reads_the_waves_own_cells():
    """P9. A median over the whole field would be the same number for every wave, so two
    waves sitting in opposite flows would be predicted to move the same way."""
    field = np.zeros((5, 5))
    field[2, 2] = -20.0
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True
    median = P._median_over(field)
    assert median({"region": mask}) == pytest.approx(-20.0)
    assert median({"region": np.ones((5, 5), dtype=bool)}) == pytest.approx(0.0)


def test_the_median_wind_ignores_missing_values():
    """P8. The smoother propagates NaN deliberately, so a plain median returns NaN for any
    wave touching masked ground and silently ends its track."""
    field = np.full((4, 4), np.nan)
    field[1, 1] = -7.0
    mask = np.ones((4, 4), dtype=bool)
    assert P._median_over(field)({"region": mask}) == pytest.approx(-7.0)


def test_a_wave_with_no_cells_has_no_median():
    median = P._median_over(np.zeros((4, 4)))
    assert np.isnan(median({"region": np.zeros((4, 4), dtype=bool)}))
    assert np.isnan(median({}))


def test_the_advection_gives_the_same_answer_in_either_row_order():
    """P6, and the defect that took three attempts to bind.

    `calculate_advvort_f` derives its latitude spacing from the MINIMUM latitude and a
    positive interval, which assumes latitude ascends with row index. Version 1's own
    fields arrive north-first, so its driver reverses them, gets the right answer, and
    reverses back. The flip therefore CORRECTS descending input and CORRUPTS ascending
    input; it is not a no-op in either direction.

    Two earlier versions of this test missed it. The first asserted the interior matched an
    unflipped call, which it does only because that fixture set the meridional wind to
    zero, and the flip acts on the meridional term alone. The second asserted the unset
    edge rows moved, which they do not, because the routine masks both edges symmetrically.
    The property that actually holds is this one: the same physical field must give the same
    advection whichever way its rows are stored.
    """
    lat = np.arange(-20.0, 20.1, 2.5)
    lon = np.arange(-30.0, 30.1, 2.5)
    longrid, latgrid = np.meshgrid(lon, lat)
    rv = np.exp(-((latgrid - 5.0) / 6.0) ** 2)[None, ...]
    u = np.zeros_like(rv)
    v = np.full_like(rv, 5.0)                      # northward, so the flipped term matters
    middle = lat.size // 2

    ascending = P._advection(latgrid, longrid, u, v, rv)
    descending = P._advection(latgrid[::-1], longrid,
                              u[:, ::-1], v[:, ::-1], rv[:, ::-1])[:, ::-1, :]
    # Compared over EVERY row, not just the middle one. The middle row of an odd-length
    # grid is unchanged by a reversal, so comparing only there passes even when the
    # orientation is applied and never undone.
    assert np.allclose(ascending, descending, equal_nan=True)
    off_centre = middle + 4
    assert ascending[0, off_centre, 12] == pytest.approx(descending[0, off_centre, 12])
    assert ascending[0, off_centre, 12] != pytest.approx(ascending[0, middle, 12]), \
        "the fixture must vary with latitude, or a reversal is undetectable"

    from aew.v1port.geometry import advection_of_vorticity
    truth = advection_of_vorticity(latgrid, longrid, u, v, rv)[0, middle, 12]
    assert ascending[0, middle, 12] == pytest.approx(truth), (
        "the sign is inverted, which is what applying version 1's flip to ascending input "
        "does; the trough axis is this field's zero contour, so the error is invisible in "
        "the track count and changes the field everywhere else")
    assert truth != pytest.approx(0.0), "the fixture must exercise the meridional term"


def test_curvature_gives_the_same_answer_in_either_row_order():
    """P13, the advection defect's sibling, found by working one timestep against
    version 1's published positions.

    `calculate_compvort_multi_f` indexes its meridional neighbours as `g+1` and `g-1`
    and divides by a spacing that is always positive, so it assumes latitude ascends
    with row index, and version 1's driver flips the fields both ways around the call
    (p1_data_eraint.m line 81) exactly as it does for the advection. An earlier
    `curvature_from_winds` passed the file's own north-first row order straight
    through, which put the wrong sign on every meridional derivative: measured on a
    real 1990 timestep, the resulting field correlated 0.03 with the correctly
    oriented one while keeping a similar magnitude distribution, so counts and
    thresholds looked sane and the waves were in the wrong places.
    """
    lat = np.arange(-20.0, 20.1, 2.5)
    lon = np.arange(-30.0, 30.1, 2.5)
    longrid, latgrid = np.meshgrid(lon, lat)
    # a cyclonic vortex off-center in latitude, so the meridional term matters and a
    # reversal is detectable. A PRODUCTION-SHAPED STACK of twelve NONIDENTICAL
    # timesteps, because two outside-chosen mutations in a row defeated smaller fixtures by
    # gating the orientation on the stack size (`u.shape[0] == 1`, then `<= 2`). Twelve
    # matches this file's other fixtures; a gate tuned above it stops reading as a lazy
    # implementation and starts reading as sabotage, which is outside what a mutation
    # catalog can bind.
    steps = np.arange(12, dtype=float)
    u = np.stack([-(1.5 - 0.05 * s) * (latgrid - 5.0 - 0.3 * s) for s in steps])
    v = np.stack([(1.5 - 0.03 * s) * (longrid + 0.4 * s) for s in steps])

    ascending = P.curvature_from_winds(latgrid, longrid, u, v)
    descending = P.curvature_from_winds(latgrid[::-1], longrid,
                                        u[:, ::-1], v[:, ::-1])[:, ::-1]
    assert np.allclose(ascending, descending, equal_nan=True)
    finite = np.isfinite(ascending[0]) & np.isfinite(ascending[-1])
    assert not np.allclose(ascending[0][finite], ascending[-1][finite]), \
        "the timesteps must differ, or a short-stack-only orientation passes"

    middle = lat.size // 2
    inside = np.isfinite(ascending[0])
    assert not np.allclose(ascending[0][inside], ascending[0][inside][::-1]), \
        "the fixture must be asymmetric in latitude, or a reversal is undetectable"
    assert np.isfinite(ascending[0, middle + 4, 12])


def test_curvature_of_a_cyclonic_vortex_matches_the_hand_derived_value():
    """The orientation anchored to physics, with the expected value written by hand.

    A counterclockwise solid-body vortex, u = -k(lat - lat0), v = k(lon - lon0) with
    k in m/s per degree, has speed V = k*r at radius r and, near the equator where a
    degree is deg2rad(1)*R meters, curvature vorticity V/r = k / (deg2rad(1)*R). None
    of that is read from the module. The wrong row order sends the relative vorticity
    of this flow toward zero, so the band below fails loudly under it, in either
    storage order.
    """
    from aew.v1port.vorticity import EARTH_RADIUS_M
    lat = np.arange(-15.0, 25.1, 2.5)
    lon = np.arange(-30.0, 30.1, 2.5)
    longrid, latgrid = np.meshgrid(lon, lat)
    k = 5.0
    u = (-k * (latgrid - 5.0))[None, ...]
    v = (k * longrid)[None, ...]
    expected = k / (np.deg2rad(1.0) * EARTH_RADIUS_M)      # 4.49e-05 1/s by hand

    for latg, uu, vv, flip in ((latgrid, u, v, False),
                               (latgrid[::-1], u[:, ::-1], v[:, ::-1], True)):
        curv = P.curvature_from_winds(latg, longrid, uu, vv)
        if flip:
            curv = curv[:, ::-1]
        r = np.hypot(latgrid - 5.0, longrid)
        ring = np.isfinite(curv[0]) & (r >= 4.0) & (r <= 8.0)
        assert ring.sum() > 10
        got = float(np.median(curv[0][ring]))
        assert got == pytest.approx(expected, rel=0.35), \
            f"curvature over the vortex ring is {got:.3e}, expected about {expected:.3e}"


def test_curvature_refuses_a_2d_field_the_same_way_in_either_row_order():
    """Without the explicit stack check, 2-D input failed differently by orientation,
    a ValueError one way and an IndexError from the flip the other."""
    lat = np.arange(-10.0, 10.1, 2.5)
    lon = np.arange(-10.0, 10.1, 2.5)
    longrid, latgrid = np.meshgrid(lon, lat)
    for latg in (latgrid, latgrid[::-1]):
        with pytest.raises(ValueError, match="stacks"):
            P.curvature_from_winds(latg, longrid, latgrid * 0.0, latgrid * 0.0)


def test_the_row_order_of_the_grid_is_established_not_assumed():
    from aew.v1port.pipeline import latitude_descends

    ascending = np.array([[-10.0, -10.0], [-5.0, -5.0], [0.0, 0.0]])
    assert not latitude_descends(ascending)
    assert latitude_descends(ascending[::-1])


def test_tracking_is_unchanged_by_the_row_order_of_the_input():
    """The same property at the level that matters: a caller handing the pipeline a file
    stored north-first must get the same record as one handing it south-first."""
    latgrid, longrid = grid()
    times, u, v, anomaly = waves_moving_west(latgrid, longrid, nt=12)
    south_first = run(times, latgrid, longrid, u, v, anomaly)
    north_first = run(times, latgrid[::-1], longrid, u[:, ::-1], v[:, ::-1],
                      anomaly[:, ::-1])
    assert [len(t["time"]) for t in south_first] == [len(t["time"]) for t in north_first]
    for a, b in zip(south_first, north_first):
        assert a["meanlon"] == pytest.approx(b["meanlon"])


# --- the thresholds -------------------------------------------------------------------------

def test_version_ones_own_thresholds_are_carried():
    """The validation run depends on these being version 1's numbers and not recomputed
    ones, so they are pinned against the values printed in find_ews_f.m."""
    assert P.thresholds_for("ERA-Int", 700) == (7.16e-7, 2.80e-6)
    assert P.thresholds_for("ERA-Int", 600) == (7.43e-7, 2.88e-6)
    assert P.thresholds_for("NCEP", 850) == (5.34e-7, 2.23e-6)
    assert len(P.V1_THRESHOLDS) == 12, "four reanalyses at three levels"


def test_an_unknown_reanalysis_is_refused_rather_than_given_a_neighbours():
    """P12. ERA5 is the whole point of the project and version 1 has no entry for it.
    Borrowing ERA-Interim's would produce a record that looks like version 1's and was
    detected at a different sensitivity, which is the worst available outcome."""
    with pytest.raises(ValueError, match="ERA5"):
        P.thresholds_for("ERA5", 700)
    with pytest.raises(ValueError, match="750"):
        P.thresholds_for("ERA-Int", 750)


# --- the climatology helper --------------------------------------------------------------------

def test_the_climatology_spans_the_years_it_is_given():
    """Version 1 computes one mean field per calendar timestep across its whole window, so
    two years in gives one year of calendar positions out, not two."""
    lat = np.arange(-10.0, 10.1, 5.0)
    lon = np.arange(-10.0, 10.1, 5.0)
    longrid, latgrid = np.meshgrid(lon, lat)
    # Two years' worth of the SAME eight calendar positions. The day numbers are computed
    # from real dates rather than by adding 365, because 2000 is a leap year and an assumed
    # year length lands the second year on different calendar days, which is a different
    # (and correct) answer from the one this test is about.
    import datetime
    per_year, times_year = [], []
    for year in (1981, 1982):
        start = (datetime.date(year, 1, 1) - datetime.date(1900, 1, 1)).days
        n = 8
        per_year.append(np.ones((n, *latgrid.shape)))
        times_year.append(start + 0.25 * np.arange(n))
    out = P.climatology_from_years(per_year, times_year)
    assert out["mean"].shape[1:] == latgrid.shape
    assert out["mean"].shape[0] == 8, "two years of the same 8 calendar steps give 8"
    assert list(out["counts"]) == [2] * 8, "each calendar step averaged over both years"


def test_days_since_1900_convert_to_the_right_calendar_date():
    """The seam the climatology sits behind. Every other stage carries days since 1900 as a
    float; the climatology needs real dates. Handing it the floats directly makes numpy
    read them as nanoseconds since 1970, which puts the whole record in January 1970 and
    silently collapses every calendar position onto one."""
    # 38351.25 is the first timestamp in the published 2005 Africa file.
    got = P.days_to_datetime64([38351.25])
    assert str(got[0])[:13] == "2005-01-01T06"
