"""Mutations for the comparison against version 1's published record.

The list was written before the assertions existed and is reproduced in the docstring of
tests/test_v1port_compare.py. Run with the repository's mutation checker against
src/aew/v1port/compare.py.
"""


def _sub(old, new):
    return lambda s: s.replace(old, new)


MUTATIONS = {
    # X1 timesteps the two tracks do not share are compared anyway
    "unshared_timesteps_compared": _sub(
        "        if abs(tb[j] - t) > tolerance:\n            continue",
        "        if False:\n            continue"),
    # X2 the time tolerance is wider than the record's own spacing
    "time_tolerance_wider_than_a_step": _sub(
        "    tolerance = 1.0 / 6.0",
        "    tolerance = 1.0"),
    # X3 a published track can be claimed by more than one port track
    "published_tracks_matched_twice": _sub(
        "        if i in used_port or j in used_published:\n            continue",
        "        if i in used_port:\n            continue"),
    # X3 a port track can be claimed twice
    "port_tracks_matched_twice": _sub(
        "        if i in used_port or j in used_published:\n            continue",
        "        if j in used_published:\n            continue"),
    # X4 pairs are taken in discovery order rather than best first
    "matching_is_not_best_first": _sub(
        "    candidates.sort(key=lambda c: (c[0], -c[1]))",
        "    pass"),
    # X5 the position tolerance is not applied
    "position_tolerance_not_applied": _sub(
        "            if median <= tolerance_km:\n"
        "                candidates.append((median, shared.size, i, j))",
        "            candidates.append((median, shared.size, i, j))"),
    # X6 a pair with almost no overlap counts
    "minimum_overlap_not_required": _sub(
        "            if shared.size < min_shared_steps:\n                continue",
        "            if False:\n                continue"),
    # X7 the drift signature fires whenever there is any separation at all
    "drift_signature_fires_on_any_offset": _sub(
        "            if np.median(distances[half:]) > np.median(distances[:half]) * 1.5:",
        "            if np.median(distances) > 0:"),
    # X8 the drift signature never fires
    "drift_signature_never_fires": _sub(
        "            if np.median(distances[half:]) > np.median(distances[:half]) * 1.5:",
        "            if False:"),
    # X9 the meridian signature counts tracks far from zero longitude
    "meridian_signature_ignores_longitude": _sub(
        "            if lon.size and abs(float(np.mean(lon))) <= 10.0:",
        "            if lon.size:"),
    # X9 the signature is reported as a bare share, so it reads the record's geography
    # rather than the difference
    "meridian_signature_not_against_the_base_rate": _sub(
        "    enrichment = (unmatched_share / baseline_share) if baseline_share else "
        'float("nan")',
        "    enrichment = unmatched_share"),
    # X10 the drift fraction is taken over the unmatched count instead of the pairs
    "drift_fraction_over_the_wrong_denominator": _sub(
        '    return {"gradual_drift_fraction": growing / pairs,',
        '    return {"gradual_drift_fraction": growing / max(unmatched, 1),'),
    # the truncation signature stops distinguishing longer port tracks
    "longer_port_tracks_not_counted": _sub(
        '        if len(a["time"]) > len(b["time"]):\n            longer += 1',
        "        pass"),
    # the self-comparison stops being exact, which would make every result unreadable
    "separation_is_not_a_distance": _sub(
        "        distances.append(float(great_circle_distance(la_a[i], lo_a[i],\n"
        "                                                     la_b[j], lo_b[j], \"km\")))",
        "        distances.append(float(abs(la_a[i] - la_b[j])))"),
}
