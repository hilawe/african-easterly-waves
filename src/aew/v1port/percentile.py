"""An exact percentile over data too large to hold, bit-equal to NumPy's.

WHY THIS EXISTS. The threshold contract's exact mode takes a percentile over the full
thirty-year population, 563,233,628 fine values as an upper bound, 4.196 GiB as float64
before non-finite exclusion, on a 16 GiB host. The existing route appends per-year arrays
and concatenates before calling `np.percentile`, which holds several full-size copies at
once. The contract forbids promoting that route, so this computes the same number without
ever holding the population.

WHAT IT GUARANTEES, and the oracle in `tests/test_percentile.py` holds it to this rather
than taking it from here. The result equals

    np.percentile(x[np.isfinite(x)], q, method="linear")

EXACTLY, bit for bit, for the concatenation `x` of every chunk supplied, FOR ACCEPTED
POPULATIONS: those whose finite range hi - lo is itself representable in float64. A
population spanning nearly the whole float64 line overflows NumPy's own interpolation
subtraction (np.percentile returns nan on [-max, max] at q=0), and a first version of this
routine returned the finite minimum there, a mathematically better answer that silently
broke both the bit-equality claim and the two-routes-agree property. Emulating the
overflow would be preserving a pathology, so the population is VALIDATED instead and both
routes refuse it identically. The atmospheric fields, near 1e-4 at their largest, sit
hundreds of orders of magnitude inside the float64 boundary (an earlier version said
twelve, which was wrong by a factor of twenty-five), so the refusal is a guardrail, not a
working path.

The finite population is defined by `np.isfinite`, so NaN and both infinities are
excluded, matching the production path. Zero finite values is a REFUSAL, never a numeric
answer, and a `q` outside [0, 100] is refused BEFORE any route is chosen, because NumPy
rejects it on the small path and an unvalidated streaming path once answered q=-1 with a
number.

HOW, in two ideas. `method="linear"` needs at most two order statistics, the values at
ranks floor(f) and ceil(f) for f = q/100 * (n - 1), plus NumPy's own interpolation between
them. The order statistics are found by counting rather than sorting. One pass histograms
the finite values over their range, the bins holding BOTH wanted ranks are identified from
cumulative counts, and one further pass collects both bins' members together, which are few
enough to sort in memory. A bin still too large (a near-constant field piles everything
into one) is REFINED, re-histogrammed over its own sub-range, as many times as needed, and
a bin whose values are all identical terminates immediately.

THE FINAL INTERPOLATION IS DELEGATED WITH THE FRACTION, not a percentage. For a
two-element array, `np.quantile([a, b], frac, method="linear")` has rank exactly `frac`,
so NumPy's `_lerp` produces the final bits. A first version passed `frac * 100.0` to
`np.percentile`, whose internal division by 100 does not round-trip. At q=55 the very
first random 26-value population tried differed from NumPy in the last bit. The
regression tests carry found counterexamples at both production percentiles.

PASSES OVER THE DATA, stated honestly because a first version claimed two. The common
noninteger-percentile case makes THREE full traversals, one for count and range, one
histogram, one collection, the two order statistics SHARING the histogram and the
collection. A first version selected each rank independently, which was five traversals.
Refinement adds traversals for the affected rank only, and an integer rank position needs
one order statistic in the same three passes.

MEMORY, measured rather than asserted, by `scripts/bench_percentile_memory.py`, which
reports GROWTH above the pre-call high-water mark and ABSOLUTE PEAK ru_maxrss separately,
over three repetitions, with the environment recorded, because single figures presented
as fixed turned out to vary by over 100 MiB between runs. Observed on this host (macOS
arm64, Python 3.12.13, NumPy 2.5.0): a production-sized annual chunk of 18,762,460 values
grew 814 to 978 MiB, several times the 143 MiB chunk, because the finite copy, the
boolean mask and the int64 bin indices are each chunk-sized and coexist, while monthly
chunks of about 12 MiB grew 137 to 139 MiB with an absolute peak of 203 to 205 MiB. A
first version claimed one chunk plus 8 MiB plus the candidate cap, and measurement said
otherwise. The working set scales with the LARGEST CHUNK, not the population, so callers
control the bound through chunk size, and monthly chunks are the recommended shape. The
histogram adds `bins` int64 counts (8 MiB at the default) and collection is capped by
`max_candidates`. Rerun the benchmark rather than quoting these numbers elsewhere.

DETERMINISM. Bin membership is computed from (lo, width, bins) by the same arithmetic in
the counting pass and the collecting pass, so a value cannot fall in one bin while being
counted and another while being collected. Refinement filters by the CHAIN of bin indices
from every level, recomputed identically each pass, rather than by value ranges whose edge
rounding could drift between passes.

The caller supplies `chunks` as a CALLABLE returning an iterable of arrays, because the
data is walked more than once and a bare iterator would be exhausted after the first pass.
"""
import numpy as np

DEFAULT_BINS = 1 << 20
DEFAULT_MAX_CANDIDATES = 8_000_000
# Populations at or under this are collected whole and handed to np.percentile directly.
# The streaming machinery is for data that cannot be held, and below this size holding it
# is both cheaper and trivially bit-equal.
DEFAULT_COLLECT_THRESHOLD = 2_000_000


def _finite(chunk):
    chunk = np.asarray(chunk, dtype=np.float64).ravel()
    return chunk[np.isfinite(chunk)]


def _apply_filters(values, filters):
    """The values surviving the refinement chain, recomputed identically every pass.

    `filters` is the chain from every completed level, oldest first, each a
    (lo, width, bins, wanted_bin) tuple. A value participates at this level only if it
    fell in the wanted bin at every earlier one.
    """
    for lo, width, level_bins, wanted in filters:
        idx = np.clip(((values - lo) / width * level_bins).astype(np.int64),
                      0, level_bins - 1)
        values = values[idx == wanted]
    return values


def _histogram_pass(chunks, filters, lo, width, bins):
    counts = np.zeros(bins, dtype=np.int64)
    for chunk in chunks():
        values = _apply_filters(_finite(chunk), filters)
        if values.size == 0:
            continue
        idx = np.clip(((values - lo) / width * bins).astype(np.int64), 0, bins - 1)
        counts += np.bincount(idx, minlength=bins)
    return counts


def _collect_pass(chunks, filters, lo, width, bins, wanted_bins):
    gathered = {b: [] for b in wanted_bins}
    for chunk in chunks():
        values = _apply_filters(_finite(chunk), filters)
        if values.size == 0:
            continue
        idx = np.clip(((values - lo) / width * bins).astype(np.int64), 0, bins - 1)
        for b in wanted_bins:
            members = values[idx == b]
            if members.size:
                gathered[b].append(members)
    return {b: (np.concatenate(parts) if parts else np.array([]))
            for b, parts in gathered.items()}


def _bin_extent(chunks, filters, lo, width, bins, target):
    """The min and max of one bin's members, for bounding a refinement descent."""
    lowest = highest = None
    for chunk in chunks():
        values = _apply_filters(_finite(chunk), filters)
        if values.size == 0:
            continue
        idx = np.clip(((values - lo) / width * bins).astype(np.int64), 0, bins - 1)
        members = values[idx == target]
        if members.size:
            low, high = float(members.min()), float(members.max())
            lowest = low if lowest is None else min(lowest, low)
            highest = high if highest is None else max(highest, high)
    return lowest, highest


def _refine(chunks, rank, filters, lo, hi, bins, max_candidates):
    """The value at local `rank` within an oversized bin, by descending into it."""
    previous = None
    while True:
        # The state is (lo, hi, rank) and NOTHING ELSE. A first version also included
        # the filter-chain length, which grows by one on every refinement, so the state
        # never repeated and the guard meant to stop an infinite loop was itself the
        # infinite loop. Range and rank are what progress means here.
        state = (lo, hi, rank)
        if state == previous:
            raise AssertionError(
                f"refinement made no progress at range [{lo}, {hi}]; the population is "
                f"not behaving like finite floats and the selection is refused rather "
                f"than looped")
        previous = state
        if lo == hi:
            return lo                       # every remaining candidate is this value
        width = hi - lo
        counts = _histogram_pass(chunks, filters, lo, width, bins)
        cumulative = np.cumsum(counts)
        target = int(np.searchsorted(cumulative, rank + 1))
        below = int(cumulative[target - 1]) if target > 0 else 0
        in_bin = int(counts[target])
        if in_bin <= max_candidates:
            candidates = _collect_pass(chunks, filters, lo, width, bins,
                                       [target])[target]
            if candidates.size != in_bin:
                raise AssertionError(
                    f"the counting pass saw {in_bin} values in bin {target} and the "
                    f"collecting pass gathered {candidates.size}; the two passes must "
                    f"agree exactly or the selection is not trustworthy")
            candidates.sort()
            return float(candidates[rank - below])
        new_lo, new_hi = _bin_extent(chunks, filters, lo, width, bins, target)
        filters = filters + [(lo, width, bins, target)]
        rank = rank - below
        lo, hi = new_lo, new_hi


def _order_statistics(chunks, ranks, lo, hi, bins, max_candidates):
    """Values at the 0-based `ranks`, sharing the histogram and collection passes.

    A first version selected each rank independently, five full traversals for an
    ordinary percentile where three suffice. The two wanted ranks always share one
    histogram, and their bins, often the same bin, are collected together in one pass.
    """
    if lo == hi:
        return {rank: lo for rank in ranks}
    width = hi - lo
    counts = _histogram_pass(chunks, [], lo, width, bins)
    cumulative = np.cumsum(counts)
    plan = {}
    for rank in ranks:
        target = int(np.searchsorted(cumulative, rank + 1))
        below = int(cumulative[target - 1]) if target > 0 else 0
        plan[rank] = (target, below, int(counts[target]))
    small_bins = sorted({t for t, _, in_bin in plan.values()
                         if in_bin <= max_candidates})
    collected = (_collect_pass(chunks, [], lo, width, bins, small_bins)
                 if small_bins else {})
    out = {}
    for rank, (target, below, in_bin) in plan.items():
        if in_bin <= max_candidates:
            candidates = collected[target]
            if candidates.size != in_bin:
                raise AssertionError(
                    f"the counting pass saw {in_bin} values in bin {target} and the "
                    f"collecting pass gathered {candidates.size}; the two passes must "
                    f"agree exactly or the selection is not trustworthy")
            candidates = np.sort(candidates)
            out[rank] = float(candidates[rank - below])
        else:
            new_lo, new_hi = _bin_extent(chunks, [], lo, width, bins, target)
            out[rank] = _refine(chunks, rank - below,
                                [(lo, width, bins, target)],
                                new_lo, new_hi, bins, max_candidates)
    return out


def exact_percentile(chunks, q, *, bins=DEFAULT_BINS,
                     max_candidates=DEFAULT_MAX_CANDIDATES,
                     collect_threshold=DEFAULT_COLLECT_THRESHOLD):
    """The q-th percentile of the finite values across all chunks, and their count.

    Returns (value, finite_count). Bit-equal to
    `np.percentile(concatenated[np.isfinite(concatenated)], q, method="linear")`.

    `chunks` is a callable returning a fresh iterable of arrays each time it is called.
    Raises ValueError when the population holds no finite values, because an answer about
    nothing is a refusal, not a number, and when `q` is outside [0, 100], validated HERE
    so both routes refuse identically.
    """
    if not callable(chunks):
        raise TypeError("chunks must be a callable returning an iterable of arrays, "
                        "because the data is walked more than once")
    q = float(q)
    if not np.isfinite(q) or q < 0.0 or q > 100.0:
        raise ValueError(
            f"percentile q={q} is outside [0, 100]; NumPy refuses this on the held "
            f"path and the streaming path must refuse it identically")
    # first pass: count, range, and collect while small
    n = 0
    lo = hi = None
    small = []
    small_ok = True
    for chunk in chunks():
        values = _finite(chunk)
        if values.size == 0:
            continue
        n += int(values.size)
        cmin, cmax = float(values.min()), float(values.max())
        lo = cmin if lo is None else min(lo, cmin)
        hi = cmax if hi is None else max(hi, cmax)
        if small_ok:
            small.append(values)
            if n > collect_threshold:
                small, small_ok = [], False
    if n == 0:
        raise ValueError(
            "no finite values in the population; a percentile of nothing is refused "
            "rather than answered")
    # THE RANGE MUST BE REPRESENTABLE, checked before either route runs. hi - lo
    # overflowing to inf makes NumPy's interpolation return nan on the held path while
    # the histogram arithmetic here degrades into warnings and a different answer, so the
    # two routes disagree and neither is the stated reference. Both refuse instead.
    if not np.isfinite(hi - lo):
        raise ValueError(
            f"the finite population spans [{lo}, {hi}], whose range overflows float64; "
            f"the interpolation is not representable and the percentile is refused on "
            f"both routes rather than answered differently by each")
    if small_ok:
        pooled = np.concatenate(small)
        return float(np.percentile(pooled, q, method="linear")), n

    # method="linear": ranks floor(f) and ceil(f) of f = q/100 * (n - 1)
    f = (q / 100.0) * (n - 1)
    rank_lo = int(np.floor(f))
    rank_hi = int(np.ceil(f))
    stats = _order_statistics(chunks, sorted({rank_lo, rank_hi}), lo, hi, bins,
                              max_candidates)
    x_lo = stats[rank_lo]
    if rank_hi == rank_lo:
        return float(x_lo), n
    x_hi = stats[rank_hi]
    # Delegate the interpolation with the FRACTION, never a percentage. For a two-element
    # array np.quantile's linear rank is exactly `frac`, so NumPy's own _lerp produces
    # the final bits. Passing frac * 100.0 to np.percentile does not round-trip through
    # its internal division by 100, and differed from NumPy in the last bit on the first
    # random 26-value population tried at q=55.
    frac = f - rank_lo
    value = np.quantile(np.array([x_lo, x_hi]), frac, method="linear")
    return float(value), n
