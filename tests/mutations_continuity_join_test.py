"""Planted defects for scripts/continuity_join_test.py, run with the mutation checker
against tests/test_continuity_join_test.py. Each must make a test fail."""


def _off(cond):
    return lambda s: s.replace(cond, "if False:")


def _true(key):
    return lambda s: s.replace(f'"{key}": within("{key}")', f'"{key}": True')


MUTATIONS = {
    "switch_without_position": lambda s: s.replace('kind = "switch" if with_position else', 'kind = "switch" if alive else'),
    "alive_strictly_after": lambda s: s.replace('q_spans[a]["last"] >= float(d) - TOL', 'q_spans[a]["last"] > float(d) + TOL'),
    "addition_not_recognized": lambda s: s.replace("            if not lost:\n                kind = \"addition\"", "            if False:\n                kind = \"addition\""),
    "previous_set_accumulates": lambda s: s.replace("        prev = now\n", "        prev = now if prev is None else prev | now\n"),
    "class_order_swapped": lambda s: s.replace('for k in ("switch", "relay", "undetermined"):', 'for k in ("relay", "switch", "undetermined"):'),
    "sector_ignored": lambda s: s.replace("    for t in R.in_sector_tracks(tracks):\n        if R.multi_system", "    for t in tracks:\n        if R.multi_system"),
    "undetermined_not_flagged": lambda s: s.replace('r["classes"]["switch"] + r["classes"]["undetermined"]', 'r["classes"]["switch"]'),
    "rate_rise_raised": lambda s: s.replace("RATE_RISE = 0.25", "RATE_RISE = 0.5"),
    "rate_without_tolerance": lambda s: s.replace("(1.0 + RATE_RISE) * base[key] + TOL", "(1.0 + RATE_RISE) * base[key]"),
    "share_check_dropped": lambda s: s.replace('"share_rise": v["share"] - base["share"] >= R.SHARE_RISE - TOL', '"share_rise": True'),
    "carried_check_dropped": lambda s: s.replace('"carried_rise": v["carried"] > base["carried"]', '"carried_rise": True'),
    "flagged_check_dropped": _true("flagged_rate"),
    "stationary_check_dropped": _true("stationary_rate"),
    "duplicate_check_dropped": _true("duplicate_rate"),
    "stage_check_skipped": _off('if counts != stage["configs"][name]:'),
    "gate_skipped": lambda s: s.replace('if name == "released" and not MR.archive_gate(finished, run)["passed"]:', "if False:"),
    "class_sum_skipped": _off('if sum(classes.values()) != counts["multi_system_tracks"]:'),
    "case_check_skipped": _off('if case["sha256"] != stage["case_sha256"]:'),
    "stage_digest_skipped": _off('if stage["generated_by"]["sha256"] != R.SCRIPT_SHA256_AT_LOAD:'),
    "vin_in_place_of_vclip": lambda s: s.replace('CONFIGS = {"released": {}, "VCLIP": R.CONFIGS["VCLIP"]}', 'CONFIGS = {"released": {}, "VCLIP": R.CONFIGS["VIN"]}'),
    "any_season_read": _off("if year not in S.YEARS:"),
    "combine_overwrite_allowed": lambda s: s.replace("    for p in (pj, pm):\n        if os.path.exists(p):", "    for p in (pj, pm):\n        if False:"),
}
