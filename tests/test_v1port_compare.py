"""Tests for the comparison of a port-produced record against version 1's published one.

THE MUTATION LIST, WRITTEN BEFORE THE ASSERTIONS EXISTED, executable as
tests/mutations_v1port_compare.py.

    X1   the separation is computed between timesteps the two tracks do not share
    X2   the time tolerance is so wide that any two tracks look simultaneous
    X3   a track is matched to more than one counterpart
    X4   matching is not best-first, so a poor pair blocks a good one
    X5   the position tolerance is not applied, so everything matches
    X6   a pair with almost no overlap counts as a match
    X7   the gradual-drift signature fires on tracks whose separation is flat
    X8   the gradual-drift signature never fires
    X9   the near-meridian signature counts tracks far from zero longitude
    X10  the fractions are computed over the wrong denominator

WHAT THIS FILE IS FOR. The comparison is the instrument the port's fidelity will be judged
on, and its answer is whatever the data says. So these tests bind the INSTRUMENT: that it
reports an exact match as exact, that each divergence signature fires on its own signature
and not on the others, and that nothing is counted twice. A signature detector that fires on
everything would let any result be explained away, which is the failure mode that matters
here.
"""

import numpy as np
import pytest

from aew.v1port import compare as C

DAY = 0.25


def track(n, lat=10.0, lon0=0.0, lon_step=-1.5, day0=38351.0, lat_step=0.0):
    return {"time": [day0 + DAY * i for i in range(n)],
            "meanlat": [lat + lat_step * i for i in range(n)],
            "meanlon": [lon0 + lon_step * i for i in range(n)]}


# --- separation ----------------------------------------------------------------------

def test_a_track_compared_with_itself_is_everywhere_identical():
    times, distances = C.separation_km(track(6), track(6))
    assert times.size == 6
    assert np.allclose(distances, 0.0)


def test_only_shared_timesteps_are_compared():
    """X1. Two tracks that overlap for part of their lives are compared only where both
    exist; pairing an early step of one with a late step of the other would report a
    separation neither track ever had."""
    early = track(6, day0=38351.0)
    late = track(6, day0=38351.0 + 4 * DAY)
    times, distances = C.separation_km(early, late)
    assert times.size == 2, "they coexist for two steps"
    assert times[0] == pytest.approx(38351.0 + 4 * DAY)


def test_tracks_that_never_coexist_share_nothing():
    times, distances = C.separation_km(track(4, day0=38351.0),
                                       track(4, day0=38400.0))
    assert times.size == 0 and distances.size == 0


def test_the_time_tolerance_is_tighter_than_the_record_spacing():
    """X2. The record is six-hourly, so a tolerance at or above that would let a track be
    compared against its neighbour's next observation."""
    a = track(4)
    b = track(4, day0=38351.0 + DAY)          # offset by exactly one step
    times, _ = C.separation_km(a, b)
    assert times.size == 3, "three steps line up exactly; none should match across a step"


def test_a_displaced_track_reports_a_real_distance():
    a = track(4, lat=10.0)
    b = track(4, lat=11.0)                    # one degree of latitude apart
    _, distances = C.separation_km(a, b)
    assert np.all(distances > 100.0) and np.all(distances < 120.0)


# --- matching -------------------------------------------------------------------------

def test_identical_records_match_completely():
    tracks = [track(6, lon0=0.0), track(6, lon0=40.0), track(6, lon0=-40.0)]
    out = C.compare(tracks, tracks)
    assert out["matched"] == 3
    assert out["match_rate"] == pytest.approx(1.0)
    assert out["median_separation_km"] == pytest.approx(0.0)


def test_no_published_track_is_matched_twice():
    """X3. Two port tracks lying on top of one published track must not both claim it."""
    published = [track(6)]
    port = [track(6), track(6)]
    out = C.match_tracks(port, published)
    assert len(out["pairs"]) == 1
    assert len(out["port_unmatched"]) == 1


def test_no_port_track_is_matched_twice():
    """X3, the other direction, which the test above cannot see. One port track sitting on
    two published ones must claim only one of them, or the match count exceeds the number
    of tracks that exist and the match rate becomes meaningless."""
    published = [track(6), track(6)]
    port = [track(6)]
    out = C.match_tracks(port, published)
    assert len(out["pairs"]) == 1
    assert len(out["published_unmatched"]) == 1


def test_the_best_pair_wins_when_two_compete():
    """X4. Taken in file order, the first port track would claim the published one and the
    closer second would be left over. Best-first is what makes a pair mean 'the same wave'
    rather than 'whichever came first'."""
    published = [track(6, lat=10.0)]
    port = [track(6, lat=12.0), track(6, lat=10.0)]     # the second is exact
    out = C.match_tracks(port, published)
    assert out["pairs"][0]["port"] == 1
    assert out["pairs"][0]["median_separation_km"] == pytest.approx(0.0)


def test_a_track_beyond_the_tolerance_does_not_match():
    """X5."""
    published = [track(6, lat=10.0)]
    port = [track(6, lat=30.0)]                          # about 2200 km away
    out = C.match_tracks(port, published, tolerance_km=500.0)
    assert out["pairs"] == []
    assert out["port_unmatched"] == [0] and out["published_unmatched"] == [0]


def test_a_pair_with_too_little_overlap_does_not_match():
    """X6. Two waves that happen to cross for one timestep are not the same wave.

    THE FIXTURE HAS TO BE CO-LOCATED WHERE THEY MEET, or the position tolerance rejects the
    pair first and the overlap rule goes unexercised. An earlier version let the two drift
    apart, so removing the overlap requirement changed nothing and the mutation survived.
    """
    published = [track(8, day0=38351.0, lon_step=0.0)]
    port = [track(8, day0=38351.0 + 7 * DAY, lon_step=0.0)]   # same place, one shared step
    shared, distances = C.separation_km(port[0], published[0])
    assert shared.size == 1 and distances[0] == pytest.approx(0.0), (
        "the fixture must overlap in exactly one step and be co-located there")
    assert C.match_tracks(port, published, min_shared_steps=3)["pairs"] == []
    assert C.match_tracks(port, published, min_shared_steps=1)["pairs"], (
        "and must match once the requirement is lowered, or this proves nothing")


# --- the divergence signatures ------------------------------------------------------------

def drifting_copy(tracks, per_step=0.02):
    out = []
    for t in tracks:
        d = dict(t)
        d["meanlon"] = [lo - per_step * i for i, lo in enumerate(t["meanlon"])]
        out.append(d)
    return out


def test_the_drift_signature_fires_on_a_growing_separation():
    """X8, and the signature that matters most: the convex hull's starting vertex would
    offset every search polygon systematically, which shows up as tracks that agree at
    first and slowly part rather than breaking at one step."""
    published = [track(16, lon0=float(k) * 30.0) for k in range(4)]
    port = drifting_copy(published)
    out = C.compare(port, published)
    assert out["matched"] == 4
    assert out["signatures"]["gradual_drift_fraction"] == pytest.approx(1.0)


def test_the_drift_signature_stays_quiet_on_a_constant_offset():
    """X7, and the other half of the same requirement. A detector that fired on any
    difference would let every result be explained as the hull vertex."""
    published = [track(16, lon0=float(k) * 30.0) for k in range(4)]
    port = [dict(t, meanlat=[la + 0.5 for la in t["meanlat"]]) for t in published]
    out = C.compare(port, published)
    assert out["matched"] == 4
    assert out["median_separation_km"] > 40.0, "the fixture must actually be displaced"
    assert out["signatures"]["gradual_drift_fraction"] == pytest.approx(0.0)


def test_the_meridian_signature_fires_on_unmatched_tracks_near_zero_longitude():
    """X9. Version 1's contour parse can match vertices on the prime meridian as well as
    real contour headers, so its phantom candidates would sit there."""
    near = [track(6, lon0=2.0), track(6, lon0=-3.0)]
    far = [track(6, lon0=80.0), track(6, lon0=-90.0)]
    published = near + far
    out = C.compare(far, published)          # the port lacks both near-meridian tracks
    assert out["signatures"]["published_unmatched"] == 2
    assert out["signatures"]["meridian_enrichment"] == pytest.approx(2.0), (
        "half the record is near the meridian and all the unmatched are, so twice")


def test_the_meridian_signature_stays_quiet_when_the_gaps_are_elsewhere():
    near = [track(6, lon0=2.0), track(6, lon0=-3.0)]
    far = [track(6, lon0=80.0), track(6, lon0=-90.0)]
    published = near + far
    out = C.compare(near, published)         # the port lacks only the distant tracks
    assert out["signatures"]["published_unmatched"] == 2
    assert out["signatures"]["meridian_enrichment"] == pytest.approx(0.0)


def test_the_meridian_signature_reads_one_when_the_gaps_match_the_base_rate():
    """X9, AND THE DEFECT THAT MADE THIS TEST NECESSARY.

    A first version reported the bare share of unmatched tracks near the meridian, and on
    the first real comparison that came out at 24.7 percent, which reads like a signal until
    you notice that 24.2 percent of ALL the published tracks are near the meridian too. It
    was reporting the geography of the record, not anything about the difference.

    Here the unmatched are a representative slice of the record, one near and one far out of
    two of each, so the enrichment must be exactly one and the bare share must NOT be zero.
    A detector that cannot tell this case from a real signal cannot support any conclusion.
    """
    near = [track(6, lon0=2.0), track(6, lon0=-3.0)]
    far = [track(6, lon0=80.0), track(6, lon0=-90.0)]
    published = near + far
    port = [near[0], far[0]]                 # one of each is matched, one of each is not
    out = C.compare(port, published)
    signatures = out["signatures"]
    assert signatures["published_unmatched"] == 2
    assert signatures["near_meridian_base_rate"] == pytest.approx(0.5)
    assert signatures["unmatched_near_meridian_fraction"] == pytest.approx(0.5), (
        "the bare share is a half here, which looks like something and is not")
    assert signatures["meridian_enrichment"] == pytest.approx(1.0), (
        "against the base rate it is exactly nothing, which is the truth")


def test_the_truncation_signature_counts_longer_port_tracks():
    """Version 1's region search can return a partial region, which would let it lose a
    wave the port keeps following."""
    published = [track(6, lon0=float(k) * 30.0) for k in range(3)]
    port = [track(10, lon0=float(k) * 30.0) for k in range(3)]
    out = C.compare(port, published)
    assert out["matched"] == 3
    assert out["signatures"]["port_tracks_longer_fraction"] == pytest.approx(1.0)


def test_the_fractions_are_taken_over_their_own_denominators():
    """X10. The drift fraction is over MATCHED PAIRS and the meridian fraction is over
    UNMATCHED PUBLISHED tracks; sharing one denominator would make both meaningless."""
    published = [track(16, lon0=0.0), track(16, lon0=90.0), track(16, lon0=-120.0)]
    port = drifting_copy(published[:1])
    out = C.compare(port, published)
    assert out["signatures"]["matched_pairs"] == 1
    assert out["signatures"]["published_unmatched"] == 2
    assert out["signatures"]["gradual_drift_fraction"] == pytest.approx(1.0)
    # one of the two unmatched sits at zero longitude, the other does not
    assert out["signatures"]["unmatched_near_meridian_fraction"] == pytest.approx(0.0)


def test_an_empty_comparison_does_not_divide_by_zero():
    out = C.compare([], [])
    assert out["matched"] == 0
    assert np.isnan(out["median_separation_km"])
    assert out["signatures"]["gradual_drift_fraction"] == pytest.approx(0.0)
