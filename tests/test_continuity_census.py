"""Tests for scripts/continuity_census.py (`CONTINUITY_ACROSS_AFRICA_2026-11-09.md`, section 3)."""

import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
import continuity_census as C  # noqa: E402

DAY0 = 33022.0                     # 1990-06-01 00Z in days since 1900


def track(tid, lons, lats=10.0, start=DAY0):
    lons = np.asarray(lons, float)
    return {"id": tid, "time": start + 0.25 * np.arange(lons.size), "lat": np.full(lons.size, lats, float), "lon": lons}


def test_companions_are_same_time_within_the_radius_nearest_first():
    a = track(1, [20.0, 19.0]); b = track(2, [20.0 + 4.0, 15.0]); c = track(3, [21.0, 19.0], start=DAY0 + 0.25)
    idx = C.index_by_time([b, c])
    d = C.km(10.0, 20.0, 10.0, 24.0)
    assert d < C.RADIUS_KM and C.companions(10.0, 20.0, DAY0, idx) == [(2, pytest.approx(d))]
    assert C.companions(10.0, 19.0, DAY0 + 0.25, idx)[0][0] == 3, "track 3 is 2 degrees away at that time, track 2 is 4"
    far = track(4, [20.0 + 4.6]); assert C.km(10.0, 20.0, 10.0, 24.6) > C.RADIUS_KM
    assert C.companions(10.0, 20.0, DAY0, C.index_by_time([far])) == []
    assert C.companions(10.0, 20.0, DAY0 + 10.0, idx) == [], "no observation at that time means no companion"


def test_the_radius_boundary_is_inclusive_and_off_grid_times_are_refused(monkeypatch):
    monkeypatch.setattr(C, "km", lambda *a: np.array([500.0, 500.0001]))
    idx = {C.time_key(DAY0): (np.array([0.0, 0.0]), np.array([0.0, 0.0]), [7, 8])}
    assert C.companions(0.0, 0.0, DAY0, idx) == [(7, 500.0)]
    with pytest.raises(SystemExit, match="six-hourly grid"):
        C.time_key(DAY0 + 0.1)


def test_the_start_window_is_48_hours_inclusive_and_stops_at_the_index():
    q = track(1, np.linspace(30.0, 0.0, 12))
    near_at_48h = track(9, [q["lon"][8]], start=DAY0 + 2.0)      # exactly 48 hours after the start
    near_at_54h = track(10, [q["lon"][9]], start=DAY0 + 2.25)
    idx = C.index_by_time([near_at_48h, near_at_54h])
    assert C.start_set(q, idx, 11) == {9}
    assert C.start_set(q, idx, 5) == set(), "observations after the index are not read"


def test_the_five_classes():
    assert C.classify_q({1}, {1, 2}) == "carried"
    assert C.classify_q({1}, {2}) == "relayed"
    assert C.classify_q(set(), {2}) == "picked_up_late"
    assert C.classify_q({1}, set()) == "dropped"
    assert C.classify_q(set(), set()) == "unaccompanied_at_both"


def test_reserved_and_held_out_seasons_are_refused():
    assert set(C.RESERVED).isdisjoint(C.SEASONS) and 2005 not in C.SEASONS and len(C.SEASONS) == 24
    for y in C.RESERVED + C.HELD_OUT:
        with pytest.raises(SystemExit, match="reserved or held out"):
            C.season(y)


def _synthetic_season(tmp_path, first_shift=0.0):
    """QTrack 1 runs from 20 E to the coast, beside AEWC v2 track 10 early and track 11 late
    (relayed); QTrack 2 is beside AEWC v2 track 12 throughout (carried); AEWC v2 track 10 ends
    inland beside QTrack 1 (a candidate discontinuity, since track 11 follows); track 13 has no
    QTrack track near it."""
    q1 = track(1, np.linspace(20.0, -16.0, 13)); q2 = track(2, np.linspace(25.0, -12.0, 13), lats=14.0)
    a10 = track(10, q1["lon"][:5] + 1.0); a11 = track(11, q1["lon"][9:] - 1.0, start=DAY0 + 9 * 0.25)     # from 54 h, after the start window
    a12 = track(12, q2["lon"] + 0.5, lats=14.0); a13 = track(13, np.linspace(28.0, 24.0, 6), lats=4.0)
    q3 = track(3, [30.0, 29.0], lats=22.0)                       # starts exactly at 30 E, outside [10, 30)
    a14 = track(14, q1["lon"][1:3] + 0.5, lats=10.5, start=DAY0 + 0.25)   # beside QTrack 1 only before track 10 ends, outside the cohort
    ours, q = [a10, a11, a12, a13, a14], [q1, q2, q3]
    f_ours, f_q = tmp_path / "ours.bin", tmp_path / "q.bin"
    f_ours.write_bytes(b"ours"); f_q.write_bytes(b"qtrack")

    def row(t, outcome, ev=None, shift=0.0):
        r = {"id": t["id"], "band": "10 to 15", "outcome": outcome,
             "first": {"lat": float(t["lat"][0]), "lon": float(t["lon"][0]) + shift},
             "last_used": {"index": t["lon"].size - 1}}
        if ev is not None:
            r["first_atlantic_side"] = {"index": ev}
        return r
    art = {"inputs": {"this_record": {"path": str(f_ours), "sha256": C.sha256_of(f_ours)},
                      "qtrack": {"path": str(f_q), "sha256": C.sha256_of(f_q)}},
           "sides": {"qtrack": {"tracks": [row(q1, "atlantic_side", 11), row(q2, "atlantic_side", 12, first_shift), row(q3, "none")]},
                     "this_record": {"tracks": [row(a10, "none"), row(a11, "atlantic_side", 0), row(a12, "atlantic_side", 10), row(a13, "none")]}}}
    d = tmp_path / "art"; d.mkdir()
    (d / C.ARTIFACT.format(year=1990)).write_text(json.dumps(art))
    return str(d), (lambda p: (ours, "case"), lambda p: (q, {}))


def test_a_synthetic_season_classes_relay_carry_and_a_candidate_discontinuity(tmp_path):
    d, loaders = _synthetic_season(tmp_path)
    s = C.season(1990, artifact_dir=d, loaders=loaders)
    by_q = {r["id"]: r for r in s["qtrack"]}
    assert by_q[1]["class"] == "relayed" and by_q[1]["start_companions"] == [10, 14] and by_q[1]["event_companions"] == [11]
    assert by_q[2]["class"] == "carried" and by_q[2]["accompanied_steps"] == by_q[2]["steps"] == 13
    assert set(by_q) == {1, 2}, "a track starting at exactly 30 E is outside the cohort band"
    assert by_q[1]["steps"] == 12 and by_q[1]["accompanied_steps"] == 8, "q1 is accompanied at indices 0 to 4 and 9 to 11"
    by_a = {r["id"]: r for r in s["this_record"]}
    assert by_a[10]["beside_atlantic_qtrack"] == {"qtrack_id": 1, "later_aewc_tracks": [11], "kind": "candidate_discontinuity"}
    assert not by_a[13]["accompanied_start"] and not by_a[13]["accompanied_last"] and "beside_atlantic_qtrack" not in by_a[13]
    T = C.totals([s])
    assert T["qtrack_atlantic_classes"]["relayed"] == 1 and T["qtrack_atlantic_classes"]["carried"] == 1
    assert T["aewc_inland_end_beside_atlantic_qtrack"] == {"n": 1, "candidate_discontinuity": 1, "not_followed": 0}
    assert T["aewc_by_start_and_outcome"]["unaccompanied_start"] == {"n": 1, "atlantic_side": 0}
    text = C.tables_text({"seasons": [s], "totals": T})
    assert "candidate discontinuities" in text and "not identity" in text


def test_a_changed_tracks_file_or_a_shifted_first_position_is_refused(tmp_path):
    d, loaders = _synthetic_season(tmp_path, first_shift=0.5)
    with pytest.raises(SystemExit, match="does not start where its artifact says"):
        C.season(1990, artifact_dir=d, loaders=loaders)
    sub = tmp_path / "x"; sub.mkdir()
    d2, loaders2 = _synthetic_season(sub)
    (sub / "q.bin").write_bytes(b"changed")
    with pytest.raises(SystemExit, match="is not the file"):
        C.season(1990, artifact_dir=d2, loaders=loaders2)


def test_records_are_never_overwritten(tmp_path):
    (tmp_path / "continuity_census.md").write_text("x")
    with pytest.raises(SystemExit, match="never overwritten"):
        C.main(["--out-dir", str(tmp_path)])


def test_nearest_offsets_stay_in_the_sector_and_sign_the_latitude_offset(tmp_path):
    import continuity_nearest_offsets as N
    north = track(20, [10.0], lats=18.0); americas = track(21, [-70.0], lats=10.0)
    idx = C.index_by_time([north, americas])
    d, off = N.nearest_in_sector(10.0, 10.0, DAY0, idx)
    assert off == pytest.approx(8.0) and d == pytest.approx(C.km(10.0, 10.0, 18.0, 10.0))
    assert N.nearest_in_sector(10.0, -70.0, DAY0, C.index_by_time([americas])) is None, "positions outside the sector do not count"
    assert N.nearest_in_sector(10.0, 10.0, DAY0 + 5.0, idx) is None
    parts = [{"none_in_sector": 1, "distance_km": [100.0, 900.0, 1200.0], "lat_offset": [0.5, 6.0, -2.0], "qtrack_lat": [10.0, 10.0, 12.0]}]
    S = N.summary(parts)
    assert S["observations"] == 4 and S["beyond_radius"] == 2 and S["beyond_share_north"] == 0.5
    assert S["beyond_nearest_lat_median"] == pytest.approx(13.0) and S["share_within_radius"] == pytest.approx(1 / 3, abs=1e-4)
    with pytest.raises(SystemExit, match="reserved or held out"):
        N.season_offsets(2005)
