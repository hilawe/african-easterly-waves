"""Planted defects for scripts/continuity_merge_variants.py, run with the mutation checker
against tests/test_continuity_merge_variants.py. Each must make a test fail."""

MUTATIONS = {
    "sector_drops_latitude_test": lambda s: s.replace(' and sector[2] <= float(w["lat_mean"]) <= sector[3]]', "]"),
    "close_pairs_inclusive_far": lambda s: s.replace("            if float(C.km(la[i], lo[i], la[j], lo[j])) < radius_km:", "            if float(C.km(la[i], lo[i], la[j], lo[j])) < radius_km * 2:"),
    "recovery_any_class": lambda s: s.replace('rec = sum(1 for r in rows if r["stage_class"] == "placed_away" and r[variant] <= radius_km)', 'rec = sum(1 for r in rows if r[variant] <= radius_km)'),
    "regression_ignores_released": lambda s: s.replace('reg = sum(1 for r in rows if r["released"] <= radius_km and r[variant] > radius_km)', 'reg = sum(1 for r in rows if r[variant] > radius_km)'),
    "recovery_exclusive_boundary": lambda s: s.replace('rec = sum(1 for r in rows if r["stage_class"] == "placed_away" and r[variant] <= radius_km)', 'rec = sum(1 for r in rows if r["stage_class"] == "placed_away" and r[variant] < radius_km)'),
    "clip_radius_changed": lambda s: s.replace("CLIP_DEG = 5.0", "CLIP_DEG = 8.0"),
    "vin_not_input_position": lambda s: s.replace('"VIN": {"coarse_position_from_input": True}', '"VIN": {"coarse_position_from_input": False}'),
    "any_season_read": lambda s: s.replace("    if year not in S.YEARS:\n        raise SystemExit(f\"REFUSED: {year} is not a development season of section 8\")\n    import pilot_alledge_replay",
                                           "    if False:\n        raise SystemExit(f\"REFUSED: {year} is not a development season of section 8\")\n    import pilot_alledge_replay"),
    "overwrite_allowed": lambda s: s.replace("        if os.path.exists(p):\n            raise SystemExit(f\"REFUSED: {p} exists", "        if False:\n            raise SystemExit(f\"REFUSED: {p} exists"),
}
