"""The association replay runs the production loop unchanged, so its finished tracks equal
`track_case`'s on the same case. It names every track by a stable birth identity, logs
claims, seeds and prunes only at the window steps, binds its matching read to the
production call, and its gate refuses a missing track or a reordered archive."""
import importlib.util
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def _load(name):
    spec = importlib.util.spec_from_file_location(name + "_under_test", os.path.join(ROOT, "scripts", name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def moving_trough_case(steps=14, lon_hi=39.0, start_lon=30.0, speed_deg_per_step=-1.5):
    """A case whose one trough drifts west at a steady speed, on the stage diagnostic's grids,
    with a uniform easterly so the association's prediction follows it."""
    lat_c = np.arange(34.0, -35.0, -2.0)
    lon_c = np.arange(-139.0, lon_hi, 2.0)
    latgrid, longrid = np.meshgrid(np.arange(35.0, -36.0, -1.0), np.arange(-140.0, lon_hi + 0.5, 1.0), indexing="ij")
    LA, LO = np.meshgrid(lat_c, lon_c, indexing="ij")
    anom, adv, anom_f = [], [], []
    for k in range(steps):
        c = start_lon + k * speed_deg_per_step
        a = 2e-5 * np.exp(-((LO - c) / 4.0) ** 2) * np.exp(-((LA - 12.0) / 6.0) ** 2)
        anom.append(np.where(LA < 0, -a, a))
        adv.append(1e-10 * (LO - c) * np.exp(-((LA - 12.0) / 6.0) ** 2))
        anom_f.append(2e-5 * np.exp(-((longrid - c) / 4.0) ** 2) * np.exp(-((latgrid - 12.0) / 6.0) ** 2))
    shape = (steps,) + LA.shape
    return {"time": 100.0 + 0.25 * np.arange(steps), "lat_c": lat_c, "lon_c": lon_c, "latgrid": latgrid, "longrid": longrid,
            "u_c": np.full(shape, -7.0), "v_c": np.zeros(shape), "currv_anom_c": np.stack(anom), "advcurrv_anom_c": np.stack(adv),
            "currv_anom": np.stack(anom_f), "u": np.full((steps,) + latgrid.shape, -7.0), "v": np.zeros((steps,) + latgrid.shape), "sha256": "x" * 64}


RECORD = {"dataset_specific": {"tracker_flags": {"exclusive": False, "absorb": False}, "coarse_threshold": 4.0e-7, "fine_threshold": 2.5e-6}}
FAMILIES = [{"name": "f", "window": ["1900-04-11T12", "1900-04-12T12"], "box": {"lat": [0.0, 25.0], "lon": [0.0, 39.0]}}]


def _archived(tracks):
    return {"tracks": [{"time": np.asarray(t["time"], float), "lat": np.asarray(t["meanlat"], float), "lon": np.asarray(t["meanlon"], float)} for t in tracks]}


def test_the_replay_reproduces_the_production_loop_and_logs_only_the_window():
    R = _load("pilot_association_replay")
    E = _load("export_protocol_case")
    case = moving_trough_case()
    steps, per = R.window_steps(case["time"], FAMILIES)
    assert steps == [2, 3, 4, 5, 6] and per == {"f": [2, 3, 4, 5, 6]}
    res = R.replay(case, RECORD, FAMILIES, steps)
    ref = E.track_case({k: v for k, v in case.items() if k != "sha256"}, 4.0e-7, 2.5e-6, exclusive=False, absorb=False)
    g = R.gate(res["finished"], _archived(ref))
    assert g["passed"] and g["order_identical"] and g["tracks_a"] == len(ref) >= 1
    assert sorted(res["log"]) == steps                                                    # nothing logged outside the window
    first = res["log"][2]
    assert first["candidates"] and first["candidates"][0]["region_sha256"] and first["candidates"][0]["u_median"] == pytest.approx(-7.0)
    assert first["candidates"][0]["in_family_box"] == ["f"]
    long_track = max(res["fates"], key=lambda b: res["fates"][b].get("n_obs", 0))
    assert res["fates"][long_track]["fate"] == "finished" and res["fates"][long_track]["born_step"] == 0
    assert long_track == res["finished_birth"][res["fates"][long_track]["finished_index"]]
    claims = [c for k in steps for c in res["log"][k]["claims"]]
    assert claims and all(c["pass"] in ("6hr", "12hr") for c in claims) and any(c["birth"] == long_track for c in claims)
    due = [m for k in steps for m in res["log"][k]["matching"] if m["birth"] == long_track]
    assert due and all(m["chosen"] is not None and m["chosen"] in m["candidates_inside_polygon"] for m in due)
    hist = res["histories"].get(long_track) or R.raw_history(res["live_at_end"][long_track])
    assert hist["steps"][0] == 0 and len(hist["region_sha256"]) == len(hist["steps"]) and hist["lon_claimed"][0] > hist["lon_claimed"][-1]


def test_the_gate_refuses_a_missing_track_a_duplicate_and_a_reordered_archive():
    R = _load("pilot_association_replay")
    E = _load("export_protocol_case")
    case = moving_trough_case()
    ref = E.track_case({k: v for k, v in case.items() if k != "sha256"}, 4.0e-7, 2.5e-6, exclusive=False, absorb=False)
    res = R.replay(case, RECORD, FAMILIES, [2])
    assert R.gate(res["finished"], _archived(ref))["passed"]
    assert not R.gate(res["finished"], _archived(ref[:-1]))["passed"]                        # one archived track missing
    assert not R.gate(res["finished"], _archived(ref + ref[-1:]))["passed"]                  # one archived twice, multiplicity counts
    if len(ref) > 1:
        g = R.gate(res["finished"], _archived(ref[::-1]))
        assert g["shared"] == len(ref) and not g["order_identical"] and not g["passed"]       # same multiset, other order
    import copy
    nudged = copy.deepcopy(ref)
    nudged[0]["meanlat"][0] = float(nudged[0]["meanlat"][0]) + 1e-12                          # one coordinate, one representable step
    assert not R.gate(res["finished"], _archived(nudged))["passed"]
    later = copy.deepcopy(ref)
    later[0]["time"][-1] = float(later[0]["time"][-1]) + 1e-9
    assert not R.gate(res["finished"], _archived(later))["passed"]
    shorter = copy.deepcopy(ref)
    for k in ("time", "meanlat", "meanlon"):
        shorter[0][k] = list(shorter[0][k])[:-1]                                              # one observation fewer
    assert not R.gate(res["finished"], _archived(shorter))["passed"]


def test_a_claim_whose_matching_row_is_missing_is_refused(monkeypatch):
    R = _load("pilot_association_replay")
    case = moving_trough_case()
    real = R.read_matching

    def forgetful(tracks, states, waves, step, exclusive, registry):
        return [m for m in real(tracks, states, waves, step, exclusive, registry) if m["chosen"] is None]
    monkeypatch.setattr(R, "read_matching", forgetful)
    with pytest.raises(SystemExit, match="without being due"):
        R.replay(case, RECORD, FAMILIES, [2, 3, 4, 5, 6])


def test_a_matching_read_that_disagrees_with_the_production_claim_is_refused(monkeypatch):
    R = _load("pilot_association_replay")
    case = moving_trough_case()
    real = R.read_matching

    def lying(tracks, states, waves, step, exclusive, registry):
        rows = real(tracks, states, waves, step, exclusive, registry)
        for m in rows:
            m["chosen"] = None
        return rows
    monkeypatch.setattr(R, "read_matching", lying)
    with pytest.raises(SystemExit, match="matching read chose"):
        R.replay(case, RECORD, FAMILIES, [2, 3, 4, 5, 6])


def test_an_existing_output_is_refused(tmp_path):
    R = _load("pilot_association_replay")
    out = tmp_path / "r.json"
    out.write_text("{}")
    with pytest.raises(SystemExit, match="never overwritten"):
        R.main(["--run-dir", str(tmp_path), "--case", "x", "--families", "y", "--label", "B", "--out", str(out), "--year", "1990"])
