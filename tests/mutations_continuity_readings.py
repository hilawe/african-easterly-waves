"""Planted defects for scripts/continuity_readings.py, run with the mutation checker
against tests/test_continuity_readings.py. Each must make a test fail."""

MUTATIONS = {
    "ridge_order_counted": lambda s: s.replace("v[i] <= 0.0 < v[i + 1]", "(v[i] <= 0.0 < v[i + 1] or v[i] > 0.0 >= v[i + 1])"),
    "zero_not_northerly": lambda s: s.replace("v[i] <= 0.0 < v[i + 1]", "v[i] < 0.0 < v[i + 1]"),
    "crossing_limit_5deg": lambda s: s.replace("CROSSING_DEG = 4.0", "CROSSING_DEG = 5.0"),
    "one_row_enough": lambda s: s.replace("ROWS_NEEDED = 2", "ROWS_NEEDED = 1"),
    "maximum_not_strict": lambda s: s.replace("            if x > nb.max():", "            if x >= nb.max():"),
    "maximum_ignores_threshold": lambda s: s.replace("            if x < ft:\n                continue\n", ""),
    "maximum_radius_500": lambda s: s.replace("MAXIMUM_KM = 300.0", "MAXIMUM_KM = 500.0"),
    "support_needs_one": lambda s: s.replace('return ("unsupported", "partial", "supported")[n]', 'return ("unsupported", "supported", "supported")[n]'),
    "seed_ignored": lambda s: s.replace(' or \\\n        next((s["birth"] for s in entry["seeds"] if s["candidate"] == idx), None)', ""),
    "radius_ignored": lambda s: s.replace("    if d[i] > radius_km:\n        return", "    if False:\n        return"),
    "pruned_count_dropped": lambda s: s.replace('        out["n_obs"] = result["pruned_at"][holder]["n_obs"]\n', ""),
    "stored_distance_from_candidate": lambda s: s.replace('out["stored_km"] = round(float(C.km(qlat, qlon, f["lat"][k], f["lon"][k])), 1)', 'out["stored_km"] = round(d[i], 1)'),
    "gate_removed": lambda s: s.replace('        if not gate["passed"]:\n            raise SystemExit', '        if False:\n            raise SystemExit'),
    "any_season_read": lambda s: s.replace("    if year not in S.YEARS:\n        raise SystemExit(f\"REFUSED: {year} is not a development season of section 8\")\n    import pilot_alledge_replay",
                                           "    if False:\n        raise SystemExit(f\"REFUSED: {year} is not a development season of section 8\")\n    import pilot_alledge_replay"),
    "unrecorded_holder_accepted": lambda s: s.replace('    if holder not in result["fates"]:\n        raise SystemExit', '    if False:\n        raise SystemExit'),
    "overwrite_allowed": lambda s: s.replace("        if os.path.exists(p):\n            raise SystemExit(f\"REFUSED: {p} exists", "        if False:\n            raise SystemExit(f\"REFUSED: {p} exists"),
}
