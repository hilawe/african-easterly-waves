"""The replay reading is a lookup: it names what the association did with each in-box
candidate from the replay's own claims and seeds, pairs it with the other run's nearest
candidate, carries the fates, and refuses a replay that failed its gate."""
import importlib.util
import json
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def _load():
    spec = importlib.util.spec_from_file_location("pilot_replay_reading_under_test", os.path.join(ROOT, "scripts", "pilot_replay_reading.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _replay(label, cand_lon, claim_birth, seed_birth):
    cands = [{"index": 0, "lat_mean": 12.0, "lon_mean": cand_lon, "n_points": 5, "region_sha256": "a" * 64, "u_median": -5.0, "v_median": 0.0, "in_family_box": ["f"]},
             {"index": 1, "lat_mean": -20.0, "lon_mean": -100.0, "n_points": 3, "region_sha256": "b" * 64, "u_median": -1.0, "v_median": 0.0, "in_family_box": []}]
    entry = {"step": 5, "date": "1990-07-01T12:00Z", "skipped_wave_free": False, "n_live_before": 1, "candidates": cands,
             "live_before": [{"birth": claim_birth, "n_obs": 3, "last_step": 4, "last_lat": 12.0, "last_lon": cand_lon + 1.0, "due": "6hr", "est_6hr": [12.0, cand_lon], "est_12hr": [12.0, cand_lon - 1.0]}],
             "matching": [{"birth": claim_birth, "pass": "6hr", "candidates_inside_polygon": [0], "implied_speed_ms": {"0": 5.0}, "distance_to_prediction_nm": {"0": 1.0}, "polygon_as_read": [], "chosen": 0}],
             "claims": [{"birth": claim_birth, "candidate": 0, "n_obs": 4, "pass": "6hr"}], "seeds": [{"birth": seed_birth, "candidate": 1}], "pruned": []}
    return {"label": label, "year": 1990, "script_sha256": "s", "case_sha256": "c", "tracks_sha256": "t", "gate": {"passed": True}, "steps": [entry],
            "family_steps": {"f": [5]}, "fates": {claim_birth: {"born_step": 2, "born_candidate": 0, "fate": "finished", "finished_index": 7, "n_obs": 9},
                                                 seed_birth: {"born_step": 5, "born_candidate": 1, "fate": "pruned_in_loop", "fate_step": 9, "n_obs": 2}},
            "histories": {claim_birth: {"steps": [2, 3, 4, 5], "lon_claimed": [cand_lon + 3, cand_lon + 2, cand_lon + 1, cand_lon], "lat_claimed": [12.0] * 4}},
            "named_tracks": {f"{label}7": {"family": "f", "finished_index": 7, "birth": claim_birth, "fate": {"fate": "finished"}}}}


def test_a_replay_record_is_bound_by_the_bytes_it_is_read_from(tmp_path):
    M = _load()
    import hashlib
    p = tmp_path / "r.json"
    p.write_text('{"label": "B", "gate": {"passed": true}}')
    rec, sha = M.load_replay_bytes(str(p))
    assert rec["label"] == "B" and sha == hashlib.sha256(p.read_bytes()).hexdigest()
    p.write_text('{"label": "B", "gate": {"passed": true}, "x": 1}')
    assert M.load_replay_bytes(str(p))[1] != sha


def test_the_reading_looks_up_claims_seeds_fates_and_the_other_runs_nearest_candidate(tmp_path):
    M = _load()
    rb, rc = _replay("B", 36.0, "2:0", "5:1"), _replay("C", 36.4, "1:3", "5:1")
    fam = {"name": "f", "window": ["1990-07-01T12", "1990-07-01T12"], "box": {"lat": [0.0, 25.0], "lon": [20.0, 50.0]}}
    out = M.read_family(fam, rb, rc)
    row = out["steps"][0]
    assert row["B"][0]["handling"] == [{"claimed_by": "2:0", "pass": "6hr", "n_obs_after": 4}] and len(row["B"]) == 1                  # the southern candidate is outside the box
    assert row["B"][0]["nearest_other_run"]["distance_deg"] == pytest.approx(0.4) and row["B"][0]["nearest_other_run"]["handling_in_other_run"] == [{"claimed_by": "1:3", "pass": "6hr", "n_obs_after": 4}]
    assert row["C"][0]["nearest_other_run"]["same_region_digest"] is True
    assert out["fates_of_tracks_involved"]["B"]["2:0"]["fate"] == "finished" and out["fates_of_tracks_involved"]["B"]["2:0"]["steps"] == [2, 3, 4, 5]
    assert out["named_tracks"]["B7"]["birth"] == "2:0" and out["named_tracks"]["C7"]["birth"] == "1:3"
    assert row["due_in_box"]["B"] == [{"birth": "2:0", "n_obs": 3, "last": [12.0, 37.0], "due": "6hr", "est": [12.0, 36.0]}]
    assert row["matching_in_box_B"][0]["chosen"] == 0
    rb_failed = dict(rb, gate={"passed": False})
    for name, obj in (("b.json", rb_failed), ("c.json", rc)):
        (tmp_path / name).write_text(json.dumps(obj))
    (tmp_path / "f.json").write_text(json.dumps({"families": [fam]}))
    with pytest.raises(SystemExit, match="did not pass its gate"):
        M.main(["--replay-control", str(tmp_path / "b.json"), "--replay-treatment", str(tmp_path / "c.json"), "--families", str(tmp_path / "f.json"), "--out", str(tmp_path / "o.json")])
