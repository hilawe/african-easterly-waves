"""Mutations for the downloader's file validation.

The list matches the counterfeits a review executed against the earlier validator, each
of which it accepted. Run with the repository's mutation checker against
scripts/download_eraint_v1port.py.
"""


def _sub(old, new):
    return lambda s: s.replace(old, new)


MUTATIONS = {
    # the level check goes silent, so an 850 hPa field named u700 runs at the wrong level
    "level_check_dropped": _sub(
        "    if level != float(LEVEL):",
        "    if False:"),
    # a level dimension without a coordinate is given the benefit of the doubt
    "unestablished_level_accepted": _sub(
        "    if level_reason is not None:\n        return level_reason",
        "    if level_reason is not None:\n        pass"),
    # the year check goes silent, so a complete 1982 satisfies a 1981 request
    "year_check_dropped": _sub(
        "    if abs(times[0] - jan1) > one_second:",
        "    if False:"),
    # the cadence check goes silent
    "cadence_check_dropped": _sub(
        "    if np.any(np.abs(np.diff(times) - 0.25) > one_second):",
        "    if False:"),
    # exact coordinate comparison relaxes to allclose, blessing a slightly shifted grid
    "coordinates_compared_approximately": _sub(
        "    if not (np.array_equal(lat, want_lat) and np.array_equal(lon, want_lon)):",
        "    if not (np.allclose(lat, want_lat) and np.allclose(lon, want_lon)):"),
    # time is read raw instead of through the loader's unit rules, resurrecting the
    # wrong-epoch and zero-timestamp counterfeits
    "time_read_raw": _sub(
        "            times = L._time_days(ds)",
        '            tname = "valid_time" if "valid_time" in ds.variables else "time"\n'
        "            times = (np.asarray(ds[tname][:], dtype=float) / 86400.0\n"
        "                     + 25567.0)"),
    # the dry run writes again
    "dry_run_moves_files": _sub(
        "            elif repair:",
        "            elif True:"),
}
