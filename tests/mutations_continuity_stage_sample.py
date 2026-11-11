"""Planted defects for scripts/continuity_stage_sample.py, run with the mutation checker
against tests/test_continuity_stage_sample.py. Each must make a test fail."""

MUTATIONS = {
    "candidate_outranks_tracked": lambda s: s.replace('    if stored_km <= radius:\n        return "tracked"\n    if final_km <= radius:\n        return "candidate_not_tracked"',
                                                      '    if final_km <= radius:\n        return "candidate_not_tracked"\n    if stored_km <= radius:\n        return "tracked"'),
    "radius_exclusive": lambda s: s.replace("    if final_km <= radius:", "    if final_km < radius:"),
    "axis_ignored": lambda s: s.replace('    if axis_km <= radius:\n        return "placed_away"\n', ""),
    "westerly_outranks_axis": lambda s: s.replace('    if axis_km <= radius:\n        return "placed_away"\n    if westerly:\n        return "masked_by_westerlies"',
                                                  '    if westerly:\n        return "masked_by_westerlies"\n    if axis_km <= radius:\n        return "placed_away"'),
    "above_ignores_westerly": lambda s: s.replace("    above = (not westerly) and np.isfinite(curv) and curv >= ct", "    above = np.isfinite(curv) and curv >= ct"),
    "holder_index_swapped": lambda s: s.replace("if bool(np.asarray(w[\"region\"])[i, j])]", "if bool(np.asarray(w[\"region\"])[min(j, w['region'].shape[0] - 1), min(i, w['region'].shape[1] - 1)])]"),
    "ties_by_higher_id": lambda s: s.replace("key=lambda r: (first[r[\"id\"]], r[\"id\"])", "key=lambda r: (first[r[\"id\"]], -r[\"id\"])"),
    "three_per_class": lambda s: s.replace("PER_CLASS = 2", "PER_CLASS = 3"),
    "gate_removed": lambda s: s.replace("    if MR.signature(st[\"final\"]) != MR.signature(oracle):\n        raise SystemExit", "    if False:\n        raise SystemExit"),
    "any_season_sampled": lambda s: s.replace("    if year not in YEARS:\n        raise SystemExit", "    if False:\n        raise SystemExit"),
    "overwrite_allowed": lambda s: s.replace("        if os.path.exists(p):\n            raise SystemExit(f\"REFUSED: {p} exists", "        if False:\n            raise SystemExit(f\"REFUSED: {p} exists"),
}
