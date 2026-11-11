"""Planted defects for scripts/continuity_join_timing.py, run with the mutation checker
against tests/test_continuity_join_timing.py. Each must make a test fail."""

MUTATIONS = {
    "gap_in_wrong_units": lambda s: s.replace("round(24.0 * (e[\"time\"] - acc[k - 1]))", "round(4.0 * (e[\"time\"] - acc[k - 1]))"),
    "missing_position_not_refused": lambda s: s.replace("            if not dist:\n                raise", "            if False:\n                raise"),
    "additions_included": lambda s: s.replace('        if e["kind"] not in ("switch", "relay"):\n            continue', '        if False:\n            continue'),
    "mean_for_median": lambda s: s.replace('"gap_hours_median": float(np.median(g))', '"gap_hours_median": float(np.mean(g))'),
    "overwrite_allowed": lambda s: s.replace("    if os.path.exists(p):\n        raise SystemExit", "    if False:\n        raise SystemExit"),
}
