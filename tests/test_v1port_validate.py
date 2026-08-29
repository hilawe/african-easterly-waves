"""Tests for the check of the port against version 1's published record.

THE MUTATION LIST, WRITTEN BEFORE THE ASSERTIONS EXISTED, executable as
tests/mutations_v1port_validate.py.

    W1   the ragged array is read back with the wrong offsets, so tracks are mixed
    W2   fill values are read as data rather than as missing
    W3   the reconstructed step numbers assume contiguous six-hourly spacing, which the
         published tracks do not have
    W4   the segment speeds use an assumed spacing rather than the track's own times
    W5   the speed check reports success without having tested anything
    W6   the speed check counts a rejection as a pass, or the reverse
    W7   the duplication counter attributes an observation to its own track
    W8   the duplication counter's key is precise enough that two tracks recording one
         trough never match
    W9   the region and month filters do not filter
    W10  the smoothing sensitivity reports a ratio for a pass it did not run

WHAT THESE TESTS ARE FOR, and it is not the same as the other stages. Everywhere else the
mutation catalog binds the port against a reading of the MATLAB. Here the tests bind the
MEASURING INSTRUMENT, because the measurement itself is against real data and its result is
whatever it is. An instrument that cannot be trusted produces a number that cannot either,
and the two numbers this module produces are load-bearing: one is the project's headline
claim about version 1, and the other is the only evidence anywhere that the port's final
filter agrees with the original.
"""

import os

import numpy as np
import pytest

nc = pytest.importorskip("netCDF4")

from aew.v1port import record as R  # noqa: E402
from aew.v1port import validate as V  # noqa: E402

PUBLISHED_DIR = "data/aewc"
published = pytest.mark.skipif(
    not os.path.isdir(PUBLISHED_DIR) or not os.listdir(PUBLISHED_DIR),
    reason="the published C00784 files are not in this clone")


def write_record(tmp_path, tracks, name="ERA-Int_ew_700hPa_2005_AFR.nc"):
    """Round-trip through the port's own writer, so the reader is tested against a file
    laid out the way the published ones are rather than against a hand-built fixture."""
    path = tmp_path / name
    R.write_region(str(path), tracks, year=2005, level=700, reanalysis="ERA-Int",
                   region="AFR", date_created="2014-03-01")
    return path


def track(n, lat0=10.0, lon0=0.0, day0=38351.0, step_days=0.25, lon_step=-1.5):
    return {"time": [day0 + step_days * i for i in range(n)],
            "meanlat": [lat0] * n,
            "meanlon": [lon0 + lon_step * i for i in range(n)],
            "maxlat": [lat0 + 1.0] * n, "minlat": [lat0 - 1.0] * n,
            "meanlon_maxlat": [lon0] * n, "meanlon_minlat": [lon0] * n}


# --- reading the ragged array back -------------------------------------------------------

def test_tracks_come_back_with_their_own_observations(tmp_path):
    """W1. Three tracks of different lengths, each at its own latitude, so a mix-up is
    visible rather than merely a wrong total."""
    written = [track(4, lat0=10.0), track(7, lat0=-20.0), track(3, lat0=30.0)]
    got = V.read_tracks(str(write_record(tmp_path, written)))
    assert [len(t["time"]) for t in got] == [4, 7, 3]
    for original, back in zip(written, got):
        assert back["meanlat"] == pytest.approx(original["meanlat"], abs=1e-4)
        assert back["meanlon"] == pytest.approx(original["meanlon"], abs=1e-4)


def test_a_field_the_writer_filled_comes_back_as_missing(tmp_path):
    """W2. Reading -999 as a wavelength of minus 999 kilometres would poison any statistic
    computed over it."""
    got = V.read_tracks(str(write_record(tmp_path, [track(4)])))
    assert np.all(np.isnan(got[0]["wavelength"]))


def test_steps_are_reconstructed_from_the_times_not_assumed(tmp_path):
    """W3. Published tracks contain twelve-hour gaps, from version 1's resumption branch;
    over half of the real ones do. Numbering the observations 0, 1, 2 would silently
    compress those gaps."""
    gapped = track(4)
    gapped["time"] = [38351.0, 38351.25, 38351.75, 38352.0]      # one twelve-hour gap
    got = V.read_tracks(str(write_record(tmp_path, [gapped])))
    assert got[0]["step"] == [0, 1, 3, 4]


def test_the_reader_finds_every_file_in_a_directory(tmp_path):
    write_record(tmp_path, [track(4)], "ERA-Int_ew_700hPa_2005_AFR.nc")
    write_record(tmp_path, [track(5), track(6)], "ERA-Int_ew_700hPa_2006_AFR.nc")
    got = V.read_record(str(tmp_path))
    assert len(got) == 3
    assert len({t["source_file"] for t in got}) == 2


# --- the speed measurement ----------------------------------------------------------------

def test_segment_speeds_use_the_tracks_own_elapsed_times():
    """W4. A twelve-hour segment covering twice the ground is the SAME speed as a six-hour
    one covering half, and a fixture with both catches an assumed spacing."""
    t = track(3)
    t["time"] = [38351.0, 38351.25, 38351.75]          # 6 h then 12 h
    t["meanlon"] = [0.0, -1.0, -3.0]                   # 1 degree then 2 degrees
    speeds = V.segment_speeds_ms(t)
    assert speeds.size == 2
    # Not exact: a great-circle arc over two degrees is fractionally shorter than twice
    # the arc over one. The tolerance is far tighter than the factor of two an assumed
    # six-hour spacing would produce, which is what this test is for.
    assert speeds[0] == pytest.approx(speeds[1], rel=1e-4)


def test_a_single_observation_track_has_no_speed():
    assert V.segment_speeds_ms(track(1)).size == 0


def test_the_speed_check_actually_rejects_something():
    """W5 and W6. The check is only worth running if it can fail, so a stationary track
    must be counted as rejected and a moving one must not."""
    stationary = track(12, lon_step=0.0)
    moving = track(12, lon_step=-2.0)
    assert V.check_speed_filter([stationary])["rejected"] == 1
    assert V.check_speed_filter([moving])["rejected"] == 0
    mixed = V.check_speed_filter([stationary, moving])
    assert mixed["tracks"] == 2 and mixed["rejected"] == 1


def test_a_marginal_rejection_is_reported_separately():
    """The smoothing bias is one-sided, so a track failing by a hair is weaker evidence
    than one failing by a factor. The two are counted apart."""
    from aew.v1port.association import MIN_SPEED_MS
    lon_step = -(MIN_SPEED_MS * 0.95) * 6 * 3600 / (111320.0 * np.cos(np.radians(10.0)))
    marginal = track(12, lon_step=lon_step)
    out = V.check_speed_filter([marginal])
    assert out["rejected"] == 1 and out["rejected_within_ten_percent"] == 1


# --- the duplication counter ---------------------------------------------------------------

def test_a_track_does_not_duplicate_itself():
    """W7. The defect being measured is TWO tracks holding one observation, so a track that
    somehow held the same observation twice must not inflate the count.

    THE FIXTURE IS DEGENERATE ON PURPOSE and version 1 cannot produce it, because its track
    times strictly increase. A first version of this test used a stationary track, which
    revisits a POSITION but never a (time, position) key, so it could not tell the two
    readings apart and the mutation removing the guard survived. Testing the contract needs
    a case the contract covers, not the nearest realistic one.
    """
    doubled = track(4, lon_step=0.0)
    doubled["time"] = [38351.0, 38351.0, 38351.25, 38351.5]
    assert V.duplication_rate([doubled])["duplicated"] == 0
    # and the same key across two tracks IS counted, so the guard is not just suppressing
    assert V.duplication_rate([doubled, dict(doubled)])["duplicated"] == 4


def test_two_tracks_sharing_observations_are_counted():
    a = track(6)
    b = dict(track(6))                     # identical, as the defect produces
    out = V.duplication_rate([a, b])
    assert out["duplicated"] == 6
    assert out["fraction"] == pytest.approx(0.5)


def test_the_key_tolerates_single_precision_coordinates():
    """W8. The published coordinates are 32-bit, so two tracks that recorded one trough
    agree to about seven digits and not beyond. A key rounded to nine decimals would find
    no duplicates at all in the real record."""
    a = track(3)
    b = dict(track(3))
    b["meanlat"] = [v + 1e-7 for v in a["meanlat"]]
    assert V.duplication_rate([a, b])["duplicated"] == 3


def test_the_region_and_month_filters_filter():
    """W9."""
    a = track(6, lat0=10.0, lon0=0.0, day0=38351.0)       # January, near zero longitude
    b = dict(a)
    everywhere = V.duplication_rate([a, b])
    assert everywhere["observations"] == 12
    outside_longitude = V.duplication_rate([a, b], lon_range=(60.0, 70.0))
    assert outside_longitude["observations"] == 0
    outside_latitude = V.duplication_rate([a, b], lat_range=(-40.0, -30.0))
    assert outside_latitude["observations"] == 0
    wrong_month = V.duplication_rate([a, b], months={7})
    assert wrong_month["observations"] == 0
    right_month = V.duplication_rate([a, b], months={1})
    assert right_month["observations"] == 12


# --- the smoothing sensitivity ---------------------------------------------------------------

def test_smoothing_reduces_speed_and_each_pass_reduces_it_less():
    """W10, and the measurement the speed check's interpretation rests on. Smoothing a path
    shortens it, so every ratio is below one; and because a second pass acts on already
    smoothed coordinates it removes less, which is why correcting the speed check by one
    pass understates the original effect rather than overstating it."""
    rng = np.random.default_rng(5)
    wiggly = []
    for _ in range(300):
        t = track(12)
        t["meanlat"] = list(10.0 + rng.normal(0.0, 0.6, 12))
        t["meanlon"] = list(np.cumsum(rng.normal(-1.5, 0.8, 12)))
        wiggly.append(t)
    ratios = V.smoothing_sensitivity(wiggly, passes=3)
    assert len(ratios) == 3
    assert ratios[0] < 0.95, "the first pass must shorten the path appreciably"
    assert ratios[0] < ratios[1] < ratios[2], "each pass must cost less than the last"
    # By the third pass the path has converged and the ratio sits at one, which is the
    # same statement from the other end: there is nothing left for smoothing to remove.
    assert ratios[2] == pytest.approx(1.0, abs=0.02)


# --- against the record itself -----------------------------------------------------------------

@published
def test_the_published_record_reads_back_at_its_known_size():
    """Pins the oracle. If this number moves, the files under data/aewc changed and every
    measurement below is about a different dataset."""
    tracks = V.read_record(PUBLISHED_DIR)
    assert len(tracks) == 12163
    assert sum(len(t["time"]) for t in tracks) == 194309


@published
def test_the_ports_speed_filter_agrees_with_version_one_on_almost_every_track():
    """THE ONE REAL CHECK ON `finalize_tracks`, and the strongest evidence in the project
    that the tracker's tail matches version 1.

    Every published track already passed version 1's own minimum-speed filter, so the port
    rejecting one is the port disagreeing with the original about a case the original ruled
    on. It rejects 109 of 12,163, which is 0.9 percent.

    That is not a clean pass and the residual is not dismissed here. Version 1 filters on
    UNSMOOTHED coordinates and publishes SMOOTHED ones, so these speeds are a lower bound;
    the rejected tracks lose more to smoothing than average (a ratio of 0.858 against 0.907)
    and cluster at eight to ten observations, where a five-point window is heavy relative to
    the track; and the available correction is measurably too small, since each smoothing
    pass costs less than the last. So the disagreement is consistent with the bias and its
    size, and it is NOT PROOF that the filter is right. The bound is asserted loosely on
    purpose: tightening it would be pretending to a precision the smoothing denies.
    """
    tracks = V.read_record(PUBLISHED_DIR)
    out = V.check_speed_filter(tracks)
    assert out["rejected"] / out["tracks"] < 0.02, (
        "the port rejects more than two percent of the tracks version 1 kept, which is "
        "past what the one-sided smoothing bias can account for")
    assert out["median_speed_ms"] > 4.0, "African easterly waves propagate"


@published
def test_the_duplication_rate_of_the_published_record():
    """The project's headline claim about version 1, measured rather than cited.

    About 36 percent of the Africa record's observations are held by more than one track,
    and about 39 percent within a West African summer corridor. The claim is usually stated
    as "about 40 percent", which holds for the corridor and not for the whole record, so
    both are pinned here and the region is part of the number.
    """
    tracks = V.read_record(PUBLISHED_DIR)
    whole = V.duplication_rate(tracks)
    corridor = V.duplication_rate(tracks, lon_range=(-20, 20), lat_range=(5, 20),
                                  months={6, 7, 8, 9})
    assert 0.34 < whole["fraction"] < 0.39
    assert 0.36 < corridor["fraction"] < 0.42
    assert corridor["fraction"] > whole["fraction"], (
        "the corridor is where the claim is made and where the rate is highest")
