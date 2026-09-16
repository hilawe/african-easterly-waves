"""The oracle for the memory-bounded exact percentile.

THE MUTATION LIST, WRITTEN BEFORE THE ASSERTIONS EXISTED, executable as
tests/mutations_percentile.py.

    E1  infinities are not excluded, so the population differs from np.isfinite's
    E2  an empty finite population returns a number instead of refusing
    E3  the rank arithmetic is off by one, selecting a neighbouring order statistic
    E4  the interpolation is reimplemented instead of delegated, drifting by rounding
    E5  the counting pass and the collecting pass disagree about bin membership and the
        mismatch is not refused
    E6  refinement mislocalises the rank, so a near-constant field selects the wrong value
    E7  the small-population path and the streaming path give different answers for the
        same data

WHY BIT-EQUALITY AND NOT CLOSENESS. This function exists so exact mode can claim "the
percentile over every finite cell", and the contract in THRESHOLD_PRIMARY_CASES.md pins
the target to np.percentile(x[np.isfinite(x)], q, method="linear"). A tolerance would turn
that claim into "nearly the percentile", which is the fallback's territory, and would also
hide E4-class drift forever. Every comparison here is ==, not approx.

THE CASE LIST IS THE CONTRACT'S, not a convenience sample. It covers NaN excluded, positive and
negative infinity excluded, duplicates, even and odd counts, values spanning signs, exactly
one finite value, and refusal on zero finite values. The production population is defined
by np.isfinite, so NaN-only exclusion tests would not bind it.
"""
import numpy as np
import pytest

from aew.v1port.percentile import exact_percentile

# Percentiles the project actually uses, plus the edges where rank arithmetic breaks.
QS = (0.0, 55.0, 66.0, 100.0)


def reference(x, q):
    return np.percentile(np.asarray(x, dtype=float)[np.isfinite(x)], q, method="linear")


def streamed(x, q, n_chunks=7, **kw):
    """The same data pushed through the streaming path in uneven chunks."""
    x = np.asarray(x, dtype=float).ravel()
    pieces = np.array_split(x, n_chunks)
    value, count = exact_percentile(lambda: iter(pieces), q, **kw)
    return value, count


def force_streaming(**kw):
    """Parameters that push even tiny arrays through the histogram route.

    collect_threshold=0 disables the small-population shortcut, and a handful of bins with
    a small candidate cap forces refinement, so the machinery the thirty-year run will use
    is the machinery under test rather than a delegation to np.percentile.
    """
    out = {"collect_threshold": 0, "bins": 8, "max_candidates": 16}
    out.update(kw)
    return out


@pytest.mark.parametrize("q", QS)
def test_matches_numpy_bit_for_bit_on_random_data(q):
    rng = np.random.default_rng(11)
    x = rng.normal(scale=3e-6, size=10_001)          # odd count, threshold-scale values
    value, count = streamed(x, q, **force_streaming())
    assert count == x.size
    assert value == reference(x, q), "the streamed value must equal NumPy's exactly"


@pytest.mark.parametrize("q", QS)
def test_even_count_and_sign_spanning_values(q):
    rng = np.random.default_rng(12)
    x = np.concatenate([rng.normal(-2e-7, 1e-7, 5000), rng.normal(4e-7, 2e-7, 5000)])
    value, count = streamed(x, q, **force_streaming())
    assert count == 10_000
    assert value == reference(x, q)


@pytest.mark.parametrize("q", QS)
def test_nan_and_both_infinities_are_excluded(q):
    """E1. The population is np.isfinite's, so +inf and -inf must vanish, not dominate.

    An infinity that leaks into the range computation stretches the histogram over an
    infinite width, and one that leaks into the count shifts every rank, so this is the
    case that distinguishes isfinite from ~isnan.
    """
    rng = np.random.default_rng(13)
    clean = rng.normal(size=4001)
    dirty = np.concatenate([clean, [np.nan] * 40, [np.inf] * 7, [-np.inf] * 7])
    rng.shuffle(dirty)
    value, count = streamed(dirty, q, **force_streaming())
    assert count == clean.size, "the count must be the finite count, nothing else"
    assert value == reference(clean, q)


def test_duplicates_including_a_near_constant_field():
    """E6. A field that piles most of its mass into one bin forces refinement.

    90 percent of the values are identical, so with 8 bins the wanted rank's bin holds far
    more than max_candidates=16 and the selection must descend. A refinement that
    mislocalises the rank returns a value from the wrong side of the pile.
    """
    rng = np.random.default_rng(14)
    x = np.concatenate([np.full(9000, 5.0), rng.uniform(0, 10, 1000)])
    rng.shuffle(x)
    for q in QS:
        value, count = streamed(x, q, **force_streaming())
        assert count == 10_000
        assert value == reference(x, q)


def test_an_entirely_constant_field_terminates_and_is_exact():
    """The degenerate end of E6: min == max, refinement has nowhere to descend."""
    x = np.full(5000, 7.25e-7)
    value, count = streamed(x, 55.0, **force_streaming())
    assert count == 5000
    assert value == 7.25e-7


def test_exactly_one_finite_value_is_that_value():
    x = np.array([np.nan, np.inf, 3.5e-6, -np.inf, np.nan])
    for q in QS:
        value, count = streamed(x, q, **force_streaming())
        assert count == 1
        assert value == 3.5e-6, "every percentile of one value is that value"


def test_zero_finite_values_are_refused():
    """E2. An answer about nothing is a refusal, not a number."""
    x = np.array([np.nan, np.inf, -np.inf, np.nan])
    with pytest.raises(ValueError, match="refused"):
        streamed(x, 55.0, **force_streaming())
    with pytest.raises(ValueError, match="refused"):
        exact_percentile(lambda: iter([np.array([])]), 55.0)


def test_the_small_path_and_the_streaming_path_agree():
    """E7. Two routes to the same number must be the same number, on identical data."""
    rng = np.random.default_rng(15)
    x = rng.normal(size=3000)
    small, n_small = streamed(x, 66.0)                        # under the default threshold
    big, n_big = streamed(x, 66.0, **force_streaming())       # forced through the machinery
    assert n_small == n_big == 3000
    assert small == big == reference(x, 66.0)


def test_chunking_does_not_change_the_answer():
    """The chunk boundaries are an implementation accident and must be invisible."""
    rng = np.random.default_rng(16)
    x = rng.normal(size=8192)
    answers = {streamed(x, 55.0, n_chunks=k, **force_streaming())[0]
               for k in (1, 2, 5, 64)}
    assert answers == {reference(x, 55.0)}, \
        "the same population in different chunkings must give one identical value"


def test_a_bare_iterator_is_refused():
    """The data is walked more than once, so an exhaustible iterator is an error the
    caller should hear about, not a silent second pass over nothing."""
    with pytest.raises(TypeError, match="callable"):
        exact_percentile(iter([np.arange(5.0)]), 55.0)


def test_the_count_is_returned_with_the_value():
    """The contract records the finite count in the artifact, so it must come from the
    same walk that produced the value, not from a separate accounting."""
    x = np.concatenate([np.arange(100.0), [np.nan] * 5])
    value, count = streamed(x, 55.0, **force_streaming())
    assert count == 100
    assert value == reference(x, 55.0)


def test_the_interpolation_is_numpys_own_not_a_reimplementation():
    """E4, on a found counterexample rather than hoping random data hits one.

    NumPy's _lerp switches formula at t >= 0.5 for rounding symmetry, so the plausible
    reimplementation a + (b - a) * t can differ in the LAST BIT. The pair below was found
    by search through the real q pipeline: naive gives ...791, NumPy gives ...792. The
    fixture-validity assertion comes first, so if a future NumPy changes _lerp and the
    counterexample stops being one, this test refuses loudly instead of silently no longer
    binding anything.
    """
    a = -5.369532353602851e-07
    b = 5.811181041963531e-07
    q = 77.47968438365298
    naive = a + (b - a) * (q / 100.0)
    ref = reference(np.array([a, b]), q)
    assert naive != ref, (
        "the counterexample no longer distinguishes naive lerp from NumPy's; find a new "
        "pair, because without one this test does not bind the delegation")
    value, count = streamed(np.array([a, b]), q, **force_streaming())
    assert count == 2
    assert value == ref, "the last bit must be NumPy's, which only delegation guarantees"


def test_a_chunks_callable_that_changes_its_data_is_refused():
    """E5. The counting pass and the collecting pass must see the same population.

    The contract for `chunks` is a callable returning the SAME data every call. If a
    caller violates that (a file rewritten mid-run, a generator with state), the counting
    pass and the collecting pass disagree, and the selection must refuse rather than
    return a value quietly computed over two different populations. This drives the
    consistency guard directly, because on correct data it can never fire, and a guard no
    test can reach is decoration.
    """
    rng = np.random.default_rng(17)
    full = rng.uniform(0.0, 10.0, size=1000)
    calls = {"n": 0}

    def unstable():
        calls["n"] += 1
        # pass 1 (count/range) and pass 2 (histogram) see everything; the collecting
        # pass sees a population with fifty values missing
        return iter([full if calls["n"] <= 2 else full[:-50]])

    with pytest.raises(AssertionError, match="agree exactly"):
        exact_percentile(unstable, 55.0, collect_threshold=0, bins=8,
                         max_candidates=2000)


def test_an_infinity_that_survived_filtering_is_refused_by_range_validation():
    """The range refusal, reached when the finite filter itself is broken.

    THIS IS NOT THE PROGRESS-GUARD TEST ANY MORE, and its earlier name and docstring said
    it was. When the range validation was added it began intercepting a leaked infinity
    before refinement could start, so the guard became unreachable from here, its
    disabling mutation survived, and the patched-`_bin_extent` test below became the
    guard's real test. What this one binds now is that a population whose range has gone
    non-finite is refused loudly on entry rather than computed through."""
    from aew.v1port import percentile as P

    original = P._finite
    P._finite = lambda chunk: np.asarray(chunk, dtype=np.float64).ravel()  # let inf through
    try:
        bad = np.array([1.0, 2.0, 3.0, np.inf])
        with pytest.raises((AssertionError, ValueError)):
            exact_percentile(lambda: iter([bad]), 55.0, collect_threshold=0,
                             bins=8, max_candidates=1)
    finally:
        P._finite = original


def test_the_found_counterexamples_at_both_production_percentiles():
    """E4's sharper half: the percentage round trip, on FOUND populations.

    A review discovered that delegating with frac * 100.0 through np.percentile does not
    round-trip its internal division by 100, and the last bit moves. The populations here
    were found by search, regenerated deterministically from their seed and trial number.
    The fixture-validity assertions confirm each still distinguishes the round-tripping
    formula from the direct one, so if a future NumPy closes the gap this test refuses
    loudly instead of silently binding nothing.
    """
    for q, count, seed, trial in ((55.0, 26, 0, 0), (66.0, 16, 0, 5)):
        rng = np.random.default_rng(seed)
        for _ in range(trial + 1):
            x = rng.normal(scale=3e-7, size=count)
        ref = reference(x, q)
        # fixture validity: the round-tripping formula must still disagree with NumPy
        n = x.size
        f = (q / 100.0) * (n - 1)
        lo_r, hi_r = int(np.floor(f)), int(np.ceil(f))
        s = np.sort(x)
        round_tripped = float(np.percentile(np.array([s[lo_r], s[hi_r]]),
                                            (f - lo_r) * 100.0, method="linear"))
        assert round_tripped != ref, (
            f"the q={q} counterexample no longer distinguishes the formulas; find a new "
            f"population or this regression binds nothing")
        value, got_n = streamed(x, q, **force_streaming())
        assert got_n == count
        assert value == ref, (
            f"q={q}: the streamed value must be NumPy's bit for bit, which requires "
            f"delegating with the FRACTION, never frac * 100")


@pytest.mark.parametrize("q", (-1.0, 100.0000001, 101.0, float("nan"), float("inf")))
def test_a_percentile_outside_the_valid_range_is_refused_on_both_routes(q):
    """NumPy refuses these on the held path. The streaming path once answered q=-1 with
    a number and q=101 with an unrelated IndexError, so validation lives before the
    route split and both refuse identically."""
    x = np.arange(100.0)
    with pytest.raises(ValueError, match="outside"):
        streamed(x, q)                                      # small route
    with pytest.raises(ValueError, match="outside"):
        streamed(x, q, **force_streaming())                 # streaming route


def test_an_ordinary_percentile_makes_three_full_traversals():
    """The pass count is part of the performance contract, and it was misstated once.

    A first version documented two passes while making five, selecting each order
    statistic independently. The two ranks share one histogram and one collection, so an
    ordinary noninteger percentile is exactly three walks of the data: count and range,
    histogram, collection. This counts the walks rather than trusting the docstring.
    """
    rng = np.random.default_rng(18)
    x = rng.normal(size=5000)
    walks = {"n": 0}

    def counted():
        walks["n"] += 1
        return iter([x])

    value, n = exact_percentile(counted, 55.0, collect_threshold=0, bins=64,
                                max_candidates=5000)
    assert value == reference(x, 55.0) and n == 5000
    assert walks["n"] == 3, (
        f"an ordinary percentile with no refinement walked the data {walks['n']} times; "
        f"the contract says three")


def test_a_population_spanning_the_float64_line_is_refused_on_both_routes():
    """The guarantee's domain boundary, found by review rather than declared.

    On [-max, max] at q=0, NumPy's own interpolation subtraction overflows and returns
    nan, while the streaming route returned the finite minimum, a better answer that
    silently broke both the bit-equality claim and route agreement. Emulating the overflow
    would preserve a pathology, so the population is refused, identically, before either
    route runs. The atmospheric fields, near 1e-4, sit hundreds of orders of magnitude
    inside the boundary.
    """
    big = np.finfo(float).max
    x = np.array([-big, big, 0.0])
    with pytest.raises(ValueError, match="overflows"):
        streamed(x, 0.0)                                     # held route
    with pytest.raises(ValueError, match="overflows"):
        streamed(x, 0.0, **force_streaming())                # streaming route
    # and the boundary is the RANGE, not the magnitude: one huge value with a nearby
    # companion has a representable range and must still be answered
    y = np.array([big, big / 2])
    value, count = streamed(y, 55.0, **force_streaming(max_candidates=4))
    assert count == 2
    assert value == reference(y, 55.0)


def test_a_refinement_that_stops_narrowing_is_refused_not_looped(monkeypatch):
    """The progress guard, driven directly, because nothing else can reach it any more.

    With finite floats refinement always narrows or collapses to lo == hi, and the range
    validation now refuses non-finite spans before refinement begins, so on correct code
    the guard is unreachable. It exists for the state this test manufactures: a descent
    whose bounds stop moving. `_bin_extent` is patched to return the parent range
    unchanged, so the second iteration repeats the first exactly, and the guard must
    refuse rather than loop. Under the mutation that widens the guard's state with the
    filter-chain length, the state never repeats and this test hangs, which the mutation
    harness reports as caught; here it must raise.
    """
    from aew.v1port import percentile as P

    monkeypatch.setattr(P, "_bin_extent", lambda chunks, filters, lo, width, bins,
                        target: (lo, lo + width))
    cluster = np.full(99, 1.0) + np.linspace(0, 1e-9, 99)
    x = np.concatenate([cluster, [100.0]])
    with pytest.raises(AssertionError, match="no progress"):
        exact_percentile(lambda: iter([x]), 55.0, collect_threshold=0, bins=8,
                         max_candidates=8)
