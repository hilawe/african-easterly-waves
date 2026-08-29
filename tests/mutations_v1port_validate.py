"""Mutations for the check of the port against version 1's published record.

The list was written before the assertions existed and is reproduced in the docstring of
tests/test_v1port_validate.py. Run with the repository's mutation checker against
src/aew/v1port/validate.py.

NOTE THAT THE TESTS AGAINST THE REAL RECORD SKIP HERE, because the mutation checker stages
only tracked files and data/aewc is untracked. So these mutations are caught by the
synthetic round-trip tests alone, which is the reason those tests exist at all: an
instrument bound only by the data it measures is bound by nothing when the data is absent.
"""


def _sub(old, new):
    return lambda s: s.replace(old, new)


MUTATIONS = {
    # W1 the ragged array is read back at a fixed stride, so tracks are mixed
    "ragged_offsets_ignore_the_counts": _sub(
        "    offsets = np.concatenate([[0], np.cumsum(counts)])",
        "    offsets = np.arange(len(counts) + 1) * (counts[0] if len(counts) else 0)"),
    # W1 every track reads from the start of the arrays
    "every_track_reads_from_offset_zero": _sub(
        "        lo, hi = offsets[i], offsets[i + 1]",
        "        lo, hi = 0, offsets[i + 1] - offsets[i]"),
    # W2 the fill value is read as data
    "fill_values_read_as_data": _sub(
        "            columns[key] = np.ma.filled(np.ma.masked_invalid(raw.astype(float)), "
        "np.nan)",
        "            columns[key] = np.asarray(raw, dtype=float)"),
    # W3 the step numbers assume contiguous six-hourly spacing
    "steps_assume_contiguous_spacing": _sub(
        '        track["step"] = [int(round((t - track["time"][0]) * 4)) '
        'for t in track["time"]]',
        '        track["step"] = list(range(len(track["time"])))'),
    # W4 the segment durations assume six hours rather than reading the times
    "segment_durations_assumed": _sub(
        "    seconds = np.diff(times) * 24.0 * 3600.0",
        "    seconds = np.full(times.size - 1, 6.0 * 3600.0)"),
    # W5 the speed check never rejects anything
    "speed_check_never_rejects": _sub(
        "        if not median >= MIN_SPEED_MS:",
        "        if False:"),
    # W6 the speed check rejects what it should keep
    "speed_check_is_inverted": _sub(
        "        if not median >= MIN_SPEED_MS:",
        "        if median >= MIN_SPEED_MS:"),
    # W6 the marginal count is not separated from the rest
    "marginal_rejections_not_counted": _sub(
        "            if median >= MIN_SPEED_MS * 0.9:\n                marginal += 1",
        "            pass"),
    # W7 an observation is counted as duplicating its own track
    "a_track_duplicates_itself": _sub(
        "            if key in seen and seen[key] != number:",
        "            if key in seen:"),
    # W8 the key is rounded past what single precision preserves
    "duplicate_key_is_too_precise": _sub(
        "            key = (round(float(t), 4), round(float(lat), 3), "
        "round(float(lon), 3))",
        "            key = (float(t), float(lat), float(lon))"),
    # W9 the region filter does not filter
    "longitude_filter_inert": _sub(
        "            if lon_range and not lon_range[0] <= lon <= lon_range[1]:\n"
        "                continue",
        "            if False:\n                continue"),
    # W9 the latitude filter does not filter
    "latitude_filter_inert": _sub(
        "            if lat_range and not lat_range[0] <= lat <= lat_range[1]:\n"
        "                continue",
        "            if False:\n                continue"),
    # W9 the month filter does not filter
    "month_filter_inert": _sub(
        "                if date.month not in months:\n                    continue",
        "                pass"),
    # W10 the smoothing sensitivity never advances, so every pass reports the first
    "smoothing_passes_do_not_advance": _sub(
        "            current, previous = nxt, now",
        "            pass"),
}
