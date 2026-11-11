"""Planted defects for scripts/continuity_census.py, run with the mutation checker
against tests/test_continuity_census.py. Each must make a test fail."""

MUTATIONS = {
    "radius_boundary_exclusive": lambda s: s.replace("if dk <= RADIUS_KM and", "if dk < RADIUS_KM and"),
    "start_window_54h": lambda s: s.replace("START_HOURS = 48.0", "START_HOURS = 54.0"),
    "start_window_ignores_index": lambda s: s.replace("    for k in range(upto + 1):\n        if (float(track[\"time\"][k]) - t0)",
                                                      "    for k in range(len(track[\"time\"])):\n        if (float(track[\"time\"][k]) - t0)"),
    "carried_relayed_swapped": lambda s: s.replace('return "carried" if start & at_event else "relayed"', 'return "relayed" if start & at_event else "carried"'),
    "late_and_dropped_swapped": lambda s: s.replace('        return "picked_up_late"\n    if start:\n        return "dropped"',
                                                    '        return "dropped"\n    if start:\n        return "picked_up_late"'),
    "companions_any_time": lambda s: s.replace("    hit = index.get(time_key(day))\n", "    hit = index.get(time_key(day)) or next(iter(index.values()), None)\n"),
    "reserved_not_refused": lambda s: s.replace("    if year in RESERVED or year in HELD_OUT:\n        raise SystemExit", "    if False:\n        raise SystemExit"),
    "season_list_keeps_2005": lambda s: s.replace("if y not in RESERVED + HELD_OUT)", "if y not in RESERVED)"),
    "digest_not_checked": lambda s: s.replace('        if sha256_of(p) != art["inputs"][key]["sha256"]:', "        if False:"),
    "first_position_not_checked": lambda s: s.replace('    if abs(float(track["lat"][0]) - f["lat"]) > 1e-6 or abs(float(track["lon"][0]) - f["lon"]) > 1e-6:', "    if False:"),
    "band_upper_inclusive": lambda s: s.replace('return START_LONS[0] <= r["first"]["lon"] < START_LONS[1]', 'return START_LONS[0] <= r["first"]["lon"] <= START_LONS[1]'),
    "later_includes_earlier_times": lambda s: s.replace('if float(qt["time"][j]) > t_end + 1e-9', 'if float(qt["time"][j]) > t_end - 10.0'),
    "share_off_by_one": lambda s: s.replace("    n = upto + 1\n", "    n = upto\n"),
    "event_index_last_used": lambda s: s.replace('upto = r["first_atlantic_side"]["index"] if r["outcome"] == "atlantic_side" else r["last_used"]["index"]',
                                                 'upto = r["last_used"]["index"]'),
    "candidate_kind_inverted": lambda s: s.replace('"kind": "candidate_discontinuity" if later else "not_followed"', '"kind": "not_followed" if later else "candidate_discontinuity"'),
    "overwrite_allowed": lambda s: s.replace("        if os.path.exists(p):\n            raise SystemExit(f\"REFUSED: {p} exists", "        if False:\n            raise SystemExit(f\"REFUSED: {p} exists"),
}
