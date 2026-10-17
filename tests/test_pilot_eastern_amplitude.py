"""The amplitude comparison's overlap value on known sets: complete separation, reversal,
ties counted as half, and an empty set reported as unavailable."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import pilot_eastern_amplitude as AM  # noqa: E402


def test_the_overlap_value_ranks_near_against_away_with_ties_as_half():
    assert AM.overlap([5.0, 6.0], [1.0, 2.0, 3.0]) == 1.0
    assert AM.overlap([1.0], [2.0, 3.0]) == 0.0
    assert AM.overlap([2.0, 4.0], [2.0, 3.0]) == (0.5 + 0.0 + 1.0 + 1.0) / 4
    assert AM.overlap([], [1.0]) is None and AM.overlap([1.0], []) is None
