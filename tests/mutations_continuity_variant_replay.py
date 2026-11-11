"""Planted defects for scripts/continuity_variant_replay.py, run with the mutation checker
against tests/test_continuity_variant_replay.py. Each must make a test fail."""


def _off(cond):
    return lambda s: s.replace(cond, "if False:")


MUTATIONS = {
    "multi_system_ignores_times": lambda s: s.replace("    return len(ids) >= 2 and len(times) >= 2", "    return len(ids) >= 2"),
    "stationary_exclusive_three_days": lambda s: s.replace(">= days - 1e-9 and", "> days + 1e-9 and"),
    "stationary_radius_doubled": lambda s: s.replace('track["lat"][-1], track["lon"][-1])) < C.RADIUS_KM', 'track["lat"][-1], track["lon"][-1])) < 2 * C.RADIUS_KM'),
    "sector_any_latitude": lambda s: s.replace(' & (t["lat"] >= sector[2]) & (t["lat"] <= sector[3]))]', '))]'),
    "groups_whole_record": lambda s: s.replace("    groups = {g for tid, g in membership.items() if tid in ids}", "    groups = set(membership.values())"),
    "share_threshold_raised": lambda s: s.replace("SHARE_RISE = 0.05", "SHARE_RISE = 0.06"),
    "share_threshold_without_tolerance": lambda s: s.replace('p["share"] - base["share"] >= SHARE_RISE - TOL', 'p["share"] - base["share"] >= SHARE_RISE'),
    "false_join_tighter": lambda s: s.replace("v <= FALSE_JOIN_RISE + TOL", "v <= FALSE_JOIN_RISE - 0.01"),
    "carried_rise_not_required": lambda s: s.replace(' and p["carried"] > base["carried"]', ""),
    "rise_from_zero_free": lambda s: s.replace('(float("inf") if p[k] else 0.0)', "0.0"),
    "qtrack_steps_check_skipped": _off('if len({pooled[n]["qtrack_steps"] for n in CONFIGS}) != 1:'),
    "gate_skipped": _off('if not gate["passed"]:'),
    "census_check_skipped": _off('if name == "released" and corr != census_row(census, year):'),
    "census_digest_skipped": _off('if census["generated_by"]["sha256"] != C.SCRIPT_SHA256_AT_LOAD:'),
    "coast_rows_check_skipped": _off("if mine != theirs:"),
    "band_ignored": lambda s: s.replace("        if not (C.START_LONS[0] <= r[\"first\"][\"lon\"] < C.START_LONS[1]):\n            continue\n", ""),
    "upto_always_last_used": lambda s: s.replace('upto = r["first_atlantic_side"]["index"] if r["outcome"] == "atlantic_side" else r["last_used"]["index"]', 'upto = r["last_used"]["index"]'),
    "a_variant_dropped": lambda s: s.replace('CONFIGS = {"released": {}, **V.VARIANTS}', 'CONFIGS = {"released": {}, "VIN": V.VARIANTS["VIN"]}'),
    "any_season_read": _off("if year not in S.YEARS:"),
    "season_overwrite_allowed": lambda s: s.replace("        if os.path.exists(p):\n            raise SystemExit(f\"REFUSED: {p} exists and records are never overwritten\")\n        rec =",
                                                    "        if False:\n            raise SystemExit(f\"REFUSED: {p} exists and records are never overwritten\")\n        rec ="),
}
