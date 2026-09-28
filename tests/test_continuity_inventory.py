"""The second pass's inventory lists the stratum's July to September non-Atlantic tracks
with their candidates and applies the predeclared selection, and the monthly tabulation of
the stratum binds to the origin re-tabulation's totals."""
import importlib.util
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, HERE)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_inventory_lists_candidates_and_selects_by_the_rule(tmp_path):
    import test_continuity_case_reading as T
    I = _load("continuity_inventory")
    evidence, cases, qdir, regions = T._world(tmp_path)
    art_dir = T._season_artifact(tmp_path, evidence, qdir, regions)
    out = tmp_path / "inventory.json"
    assert I.main(["--years", "1990", "--campaign-evidence", evidence, "--qtrack-dir", qdir, "--regions-dir", regions,
                   "--measurement-artifacts", str(art_dir), "--out", str(out)]) == 0
    inv = json.load(open(out))
    s = inv["seasons"]["1990"]
    ids = [e["id"] for e in s["inventory"]]
    assert ids == [4, 0, 5]                                      # tracks 4, 0 and 5 start in September in the stratum with outcome none (4 shares no timestamp with QTrack); 1 is Atlantic side and outside the cohort
    e0 = [e for e in s["inventory"] if e["id"] == 0][0]
    assert [c["system"] for c in e0["candidates"]] == [1, 4] and e0["window_steps_within_500_km_summed"] == 12 and e0["max_candidates_at_a_step"] == 2
    assert s["n_with_a_candidate"] == 2 and s["selected"]["id"] == 0            # a tie on 12 steps and on start time, broken by the lower index
    assert inv["selected_cases"] == [{"year": 1990, "id": 0}]
    assert I.select([]) is None and I.select([{"n_candidate_systems": 0}]) is None
    ranked = I.select([{"id": 9, "n_candidate_systems": 1, "window_steps_within_500_km_summed": 2, "first": {"time": "1990-08-01 00:00:00"}},
                       {"id": 3, "n_candidate_systems": 1, "window_steps_within_500_km_summed": 5, "first": {"time": "1990-08-09 00:00:00"}}])
    assert ranked["id"] == 3                                                       # more steps within 500 km wins over an earlier start
    ranked = I.select([{"id": 2, "n_candidate_systems": 1, "window_steps_within_500_km_summed": 5, "first": {"time": "1990-08-09 00:00:00"}},
                       {"id": 7, "n_candidate_systems": 1, "window_steps_within_500_km_summed": 5, "first": {"time": "1990-08-01 00:00:00"}}])
    assert ranked["id"] == 7                                                       # equal steps: the earlier start wins although its index is higher
    assert all("per_step" in e and len(e["per_step"]) == e["window_observations"] for e in s["inventory"])   # ambiguity kept at every step
    with pytest.raises(SystemExit):                                                # never overwritten
        I.main(["--years", "1990", "--campaign-evidence", evidence, "--qtrack-dir", qdir, "--regions-dir", regions, "--measurement-artifacts", str(art_dir), "--out", str(out)])
    with open(os.path.join(evidence, "era5_1990", "tracker_port.mat"), "ab") as fh:
        fh.write(b"\0")
    with pytest.raises(SystemExit, match="not the one the season artifact names"):
        I.main(["--years", "1990", "--campaign-evidence", evidence, "--qtrack-dir", qdir, "--regions-dir", regions, "--measurement-artifacts", str(art_dir), "--out", str(tmp_path / "again.json")])


def test_monthly_tabulation_binds_to_the_origin_totals(tmp_path):
    import test_coast_crossing_measurement as T
    C = _load("coast_crossing_measurement")
    O = _load("coast_crossing_origin")
    B = _load("coast_crossing_by_month")
    evidence, qdir, regions = T._world(tmp_path)
    art_dir = tmp_path / "artifacts"
    art_dir.mkdir()
    season = art_dir / "coast_crossing_1990_2026-09-27.json"
    assert C.main(["--year", "1990", "--campaign-evidence", evidence, "--qtrack-dir", qdir, "--regions-dir", regions, "--out", str(season)]) == 0
    origin = art_dir / "origin.json"
    assert O.main(["--artifacts", str(art_dir), "--years", "1990", "--out", str(origin), "--pattern", "coast_crossing_{year}_2026-09-27.json"]) == 0
    out = tmp_path / "by_month.json"
    assert B.main(["--artifacts", str(art_dir), "--origin", str(origin), "--years", "1990", "--out", str(out)]) == 0
    rows = json.load(open(out))["by_month"]
    o = json.load(open(origin))["pooled_all_years"]
    for n in ("this_record", "qtrack"):
        comp = o[n]["by_longitude_band"]["10 E to 30 E"]["latitude_composition"]
        assert rows[n]["total"]["cohort"] == comp["5 to 10"]["cohort"] + comp["10 to 15"]["cohort"]
        assert sum(rows[n][m]["cohort"] for m in ("6", "7", "8", "9")) == rows[n]["total"]["cohort"]
    tampered = json.loads(origin.read_text())
    tampered["pooled_all_years"]["this_record"]["by_longitude_band"]["10 E to 30 E"]["latitude_composition"]["5 to 10"]["cohort"] += 1
    (art_dir / "origin2.json").write_text(json.dumps(tampered))
    with pytest.raises(SystemExit, match="do not equal the origin"):
        B.main(["--artifacts", str(art_dir), "--origin", str(art_dir / "origin2.json"), "--years", "1990", "--out", str(tmp_path / "b2.json")])
