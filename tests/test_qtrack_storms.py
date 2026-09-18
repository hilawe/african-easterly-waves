"""Linking QTrack storm tags to IBTrACS genesis events, on a synthetic best-track file.

MUTATION LIST, written before the assertions:
  S1 match ignores the SEASON: INERT under any offset window shorter than a year,
     because a same-name storm from another season is always outside the window;
     the season filter is explicit documentation rather than a bound behaviour;
  S2 match ignores the NAME;
  S3 the FARTHEST candidate event is chosen instead of the nearest;
  S4 the offset window is not enforced;
  S5 the ambiguity flag is dropped (two candidates within the window report "matched");
  S6 the wave observation reads the first valid basin code instead of the code at the
     genesis step;
  S7 pandas reads the basin string "NA" as missing (keep_default_na dropped);
  S8 link_year yields one row per TRACK instead of per storm key;
  S9 candidate_names returns the tag unchanged;
  S10 ibtracs_name_forms returns the name unchanged (a colon-joined crossover name is
      never found by either of its parts or by the hyphenated tag);
  S11 name forms match by substring ("JOAN" finds "JOANNE").
  E1 the genesis event is the FIRST observation, not the first non-DB/LO/WV record
     (a disturbance that crosses a subbasin boundary before qualifying gets the wrong
     subbasin, and a storm whose event is 78 h after its first observation is rejected);
  E2 the exclusion covers DB only, so a LO or WV record qualifies as the event;
  D1 the spatial check is dropped (a candidate 4,000 km from the wave is "matched");
  D2 the distance tolerance is ignored (any distance passes);
  D3 the shared-identifier flag is dropped (two keys keep one SID as "matched");
  R1 a genesis outside the file's time axis returns the endpoint's code silently;
  R2 the nearest-step tolerance is ignored (a genesis 10 h from any step is read).
Added after a further review:
  A1 ambiguity is decided on time alone, before the distance criterion is applied
     (two candidates in the window with only one within 500 km read as ambiguous);
  B1 the returned basin comes from the first observation rather than the event (a
     crossover storm first seen in the North Atlantic and qualifying in the eastern
     Pacific reports NA).
"""
import datetime as dt

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from aew import qtrack_storms as QS  # noqa: E402

HEADER = ("SID,SEASON,NUMBER,BASIN,SUBBASIN,NAME,ISO_TIME,NATURE,LAT,LON,WMO_WIND,"
          "WMO_PRES,WMO_AGENCY,TRACK_TYPE,USA_STATUS\n")
UNITS = " ,Year, , , , , , ,degrees_north,degrees_east,kts,mb, , , \n"


def write_ibtracs(path, rows):
    with open(path, "w") as fh:
        fh.write(HEADER)
        fh.write(UNITS)
        for sid, season, basin, sub, name, iso, lat, lon, tt, status in rows:
            fh.write(f"{sid},{season},1,{basin},{sub},{name},{iso},TS,{lat},{lon},35,1000,"
                     f"hurdat_atl,{tt},{status}\n")


@pytest.fixture
def events(tmp_path):
    path = str(tmp_path / "ib.csv")
    write_ibtracs(path, [
        # CLAUDETTE-like: a wave in the open Atlantic on 07-07, qualifies (TD) in the
        # Caribbean on 07-08 18Z: first observation NA, genesis event CS
        ("2013A", 2013, "NA", "NA", "CLAUD", "2013-07-07 00:00:00", 12.0, -60.0, "main", "WV"),
        ("2013A", 2013, "NA", "NA", "CLAUD", "2013-07-08 00:00:00", 12.5, -66.0, "main", "LO"),
        ("2013A", 2013, "NA", "CS", "CLAUD", "2013-07-08 18:00:00", 13.0, -70.0, "main", "TD"),
        ("2013A", 2013, "NA", "CS", "CLAUD", "2013-07-09 00:00:00", 13.5, -71.0, "main", "TS"),
        # a same-name storm in the Gulf whose event is 78 h after CLAUD's: outside a
        # 72 h window from CLAUD's own event, inside it from a tag between the two
        ("2013B", 2013, "NA", "GM", "CLAUD", "2013-07-12 00:00:00", 25.0, -90.0, "main", "TD"),
        # 2012's CLAUD never competes for a 2013 tag
        ("2012C", 2012, "NA", "NA", "CLAUD", "2012-07-08 18:00:00", 12.0, -50.0, "main", "TD"),
        # two UNNAMED storms in September, one near 41W, one far away at 150W
        ("2013F", 2013, "NA", "NA", "UNNAMED", "2013-09-01 12:00:00", 16.0, -41.0, "main", "TD"),
        ("2013H", 2013, "EP", "MM", "UNNAMED", "2013-09-20 12:00:00", 16.0, -150.0, "main", "TD"),
        # a storm with only disturbance records: no genesis event at all
        ("2013J", 2013, "NA", "NA", "GHOST", "2013-08-01 00:00:00", 15.0, -40.0, "main", "DB"),
        # a CROSSOVER: first seen as a wave in the North Atlantic, qualifies in the
        # eastern Pacific, so first_basin NA and event_basin EP
        ("2013K", 2013, "NA", "CS", "CROSS", "2013-08-10 00:00:00", 10.0, -80.0, "main", "WV"),
        ("2013K", 2013, "EP", "MM", "CROSS", "2013-08-11 00:00:00", 10.0, -88.0, "main", "TD"),
        ("2013J", 2013, "NA", "NA", "GHOST", "2013-08-01 06:00:00", 15.0, -41.0, "main", "LO"),
        ("2013G", 2013, "NA", "NA", "SPUR", "2013-09-01 12:00:00", 16.0, -41.0, "spur", "TS"),
    ])
    return QS.storm_events(QS.load_ibtracs_records(path))


def test_records_keep_the_north_atlantic_basin_string_and_drop_spurs(events):
    assert "NA" in set(events["first_basin"]), "the string NA must not be read as missing"
    assert "EP" in set(events["event_basin"])
    assert "2013G" not in set(events["SID"]), "spur tracks are excluded"


def test_the_genesis_event_is_the_first_non_disturbance_record(events):
    claud = events[events["SID"] == "2013A"].iloc[0]
    assert claud["first_time"] == pd.Timestamp("2013-07-07 00:00")
    assert claud["first_subbasin"] == "NA" and claud["first_status"] == "WV"
    assert claud["event_time"] == pd.Timestamp("2013-07-08 18:00")
    assert claud["event_subbasin"] == "CS" and claud["event_status"] == "TD"
    ghost = events[events["SID"] == "2013J"].iloc[0]
    assert ghost["event_time"] is None or pd.isna(ghost["event_time"])
    cross = events[events["SID"] == "2013K"].iloc[0]
    assert cross["first_basin"] == "NA" and cross["event_basin"] == "EP"
    assert cross["first_subbasin"] == "CS" and cross["event_subbasin"] == "MM"


def test_the_event_basin_not_the_first_observation_basin_is_returned(events):
    m = QS.match_storm(events, 2013, "CROSS", dt.datetime(2013, 8, 11, 0), (10.0, -88.0))
    assert m["status"] == "matched" and m["offset_hours"] == 0.0 and m["wave_to_event_km"] == 0.0
    assert m["event_basin"] == "EP" and m["event_subbasin"] == "MM"
    assert m["first_basin"] == "NA" and m["first_subbasin"] == "CS"
    assert "basin" not in m, "no field named plainly basin, which was the ambiguity"


def test_distance_is_applied_before_ambiguity_is_decided(events):
    # two UNNAMED events lie within a 30 day window of a 09-10 tag, but only 2013F is
    # within 500 km of a wave at 41W: that is a match, not an ambiguity
    m = QS.match_storm(events, 2013, "UNNAMED", dt.datetime(2013, 9, 10, 12), (16.0, -41.0),
                       max_offset_hours=720)
    assert m["status"] == "matched" and m["sid"] == "2013F"
    assert m["candidates_within_window"] == 2
    # with both events within reach of the wave, it IS ambiguous
    m = QS.match_storm(events, 2013, "UNNAMED", dt.datetime(2013, 9, 10, 12), (16.0, -41.0),
                       max_offset_hours=720, max_distance_km=20000)
    assert m["status"] == "ambiguous" and m["candidates_within_both"] == 2
    # and with neither within reach, rejected with the nearest in time recorded
    m = QS.match_storm(events, 2013, "UNNAMED", dt.datetime(2013, 9, 10, 12), (0.0, 100.0),
                       max_offset_hours=720)
    assert m["status"] == "rejected_distance" and m["nearest_sid"] == "2013F"


def test_matching_uses_the_event_not_the_first_observation(events):
    wave = (13.2, -70.5)                      # the wave sits by CLAUD's event
    # tag at the event time: 42 h after the first observation, 0 h from the event
    m = QS.match_storm(events, 2013, "CLAUD", dt.datetime(2013, 7, 8, 18), wave)
    assert m["status"] == "matched" and m["sid"] == "2013A" and m["offset_hours"] == 0.0
    assert m["event_subbasin"] == "CS" and m["first_subbasin"] == "NA"
    assert m["first_observation"] == "2013-07-07 00Z" and m["event_time"] == "2013-07-08 18Z"
    assert m["wave_to_event_km"] < 60
    # a tag 78 h after the first observation but 36 h after the event: inside a 40 h
    # window by the event (2013B's event is 42 h later, outside it), which the
    # first-observation rule would have rejected
    m = QS.match_storm(events, 2013, "CLAUD", dt.datetime(2013, 7, 10, 6), wave,
                       max_offset_hours=40)
    assert m["status"] == "matched" and m["sid"] == "2013A" and m["offset_hours"] == 36.0
    # a tag between the two events with both inside the window, but only 2013A's
    # event within 500 km of the wave: matched, the distance having decided it
    m = QS.match_storm(events, 2013, "CLAUD", dt.datetime(2013, 7, 10, 9), wave)
    assert m["status"] == "matched" and m["sid"] == "2013A"
    assert m["candidates_within_window"] == 2
    m = QS.match_storm(events, 2013, "CLAUD", dt.datetime(2013, 7, 30, 0), wave)
    assert m["status"] == "outside_window" and m["nearest_sid"] == "2013B"
    assert m["offset_hours"] == 432.0
    assert QS.match_storm(events, 2013, "OCTAVE", dt.datetime(2013, 7, 8, 18), wave)["status"] \
        == "no_candidate"
    assert QS.match_storm(events, 2013, "GHOST", dt.datetime(2013, 8, 1, 0), wave)["status"] \
        == "no_candidate", "a storm with no genesis event is no candidate"


def test_a_candidate_far_from_the_wave_is_rejected_with_the_distance_recorded(events):
    # only 2013H (150W) lies within the window of a 09-20 tag; the wave is at 41W
    m = QS.match_storm(events, 2013, "UNNAMED", dt.datetime(2013, 9, 20, 12), (16.0, -41.0))
    assert m["status"] == "rejected_distance" and m["nearest_sid"] == "2013H"
    assert 11000 < m["wave_to_event_km"] < 12000 and "tolerance 500" in m["reason"]
    # the same candidate matches a wave that is actually there
    m = QS.match_storm(events, 2013, "UNNAMED", dt.datetime(2013, 9, 20, 12), (16.5, -149.0))
    assert m["status"] == "matched" and m["wave_to_event_km"] < 500
    # and no wave position means no validation
    m = QS.match_storm(events, 2013, "UNNAMED", dt.datetime(2013, 9, 20, 12), None)
    assert m["status"] == "rejected_distance" and m["reason"] == "no wave position at genesis"


def test_matching_does_not_ignore_the_season(events):
    m = QS.match_storm(events, 2012, "CLAUD", dt.datetime(2012, 7, 8, 18), (12.0, -50.0))
    assert m["status"] == "matched" and m["sid"] == "2012C"


def test_hurdat_style_tags_find_their_ibtracs_names(events):
    assert QS.candidate_names("TEN") == ["UNNAMED"]
    assert QS.candidate_names("JOAN-MIRIAM") == ["JOAN-MIRIAM", "JOAN", "MIRIAM"]
    assert QS.candidate_names("TWENTY-ONE") == ["UNNAMED"]
    m = QS.match_storm(events, 2013, "NINE", dt.datetime(2013, 9, 1, 12), (16.0, -41.0))
    assert m["status"] == "matched" and m["sid"] == "2013F" and m["matched_name"] == "UNNAMED"
    m = QS.match_storm(events, 2013, "OCTAVE-CLAUD", dt.datetime(2013, 7, 8, 18), (13.2, -70.5))
    assert m["status"] == "matched" and m["matched_name"] == "CLAUD"


def test_wave_observation_at_genesis_respects_the_time_axis_and_the_step_tolerance():
    times = [dt.datetime(2013, 10, 13, h) for h in (0, 6, 12, 18)]
    basin = [7.0, 6.0, np.nan, 1.0]
    lon = [-60.0, -61.0, np.nan, -63.0]
    lat = [10.0, 10.5, np.nan, 11.5]
    w = QS.qtrack_basin_at_genesis(basin, lon, lat, times, dt.datetime(2013, 10, 13, 7))
    assert w["code"] == 6 and w["lat_lon"] == [10.5, -61.0] and w["hours_from_step"] == -1.0
    w = QS.qtrack_basin_at_genesis(basin, lon, lat, times, dt.datetime(2013, 10, 13, 13))
    assert w["code"] is None and "no position" in w["reason"]
    # BEFORE the axis: unavailable, not the first code
    w = QS.qtrack_basin_at_genesis(basin, lon, lat, times, dt.datetime(2013, 4, 19, 0))
    assert w["code"] is None and "outside" in w["reason"]
    assert w["axis"] == ["2013-10-13 00Z", "2013-10-13 18Z"]
    # AFTER the axis: unavailable, not the last code
    w = QS.qtrack_basin_at_genesis(basin, lon, lat, times, dt.datetime(2013, 11, 3, 0))
    assert w["code"] is None and "outside" in w["reason"]
    # inside the axis but 4 h from any step with a 3 h tolerance: unavailable
    sparse = [dt.datetime(2013, 10, 13, 0), dt.datetime(2013, 10, 13, 8)]
    w = QS.qtrack_basin_at_genesis([7.0, 6.0], [-60.0, -61.0], [10.0, 10.5], sparse,
                                   dt.datetime(2013, 10, 13, 4))
    assert w["code"] is None and "4.0 h away" in w["reason"]


def test_link_year_keeps_one_row_per_key_and_flags_shared_identifiers(events):
    times = [dt.datetime(2013, 7, 8, h) for h in (12, 18)] + [dt.datetime(2013, 7, 9, 0)]
    data = {"name": np.array(["CLAUD", "CLAUD", "N/A", "NINE", "CLAUD"]),
            "system": np.array([3.0, 8.0, 9.0, 11.0, 12.0]),
            "basin": np.array([[7, 6, 6], [7, 6, 6], [5, 5, 5], [7, 7, 7], [6, 6, 6]],
                              dtype=float),
            "lon": np.array([[-69.0, -70.5, -71.0], [-69.0, -70.5, -71.0],
                             [-100.0, -101.0, -102.0], [-41.0, -41.5, -42.0],
                             [-70.0, -70.6, -71.2]]),
            "lat": np.array([[13.0, 13.2, 13.5], [13.0, 13.2, 13.5], [12.0, 12.0, 12.0],
                             [16.0, 16.0, 16.0], [13.1, 13.3, 13.6]])}
    g = dt.datetime(2013, 7, 8, 18)
    genesis = [g, g, None, dt.datetime(2013, 9, 1, 12), dt.datetime(2013, 7, 9, 0)]
    rows = QS.link_year(data, times, genesis, events, keep=[True] * 5)
    assert len(rows) == 3, "a storm tagged on two tracks is one row"
    a = next(r for r in rows if r["qtrack_genesis"] == "2013-07-08 18Z")
    assert a["systems"] == [3, 8] and a["wave_at_genesis"]["code"] == 6
    # the 07-09 00Z CLAUD key: 2013A's event 6 h before, 2013B's 72 h after, both
    # inside the window, only 2013A within 500 km of system 12: matched to 2013A,
    # which then SHARES the identifier with the 07-08 18Z key
    b = next(r for r in rows if r["qtrack_genesis"] == "2013-07-09 00Z")
    assert b["status"] == "shared_identifier" and b["nearest_sid"] == "2013A"
    nine = next(r for r in rows if r["name"] == "NINE")
    # its genesis (09-01) lies outside this three-step axis: unavailable, and with no
    # wave position the spatial check cannot run
    assert nine["wave_at_genesis"]["code"] is None and nine["status"] == "rejected_distance"
    assert a["status"] == "shared_identifier" and a["nearest_sid"] == "2013A"
    assert sorted(a["shared_with_keys"]) == ["2013-07-09 00Z"]
    # with the second CLAUD key moved to another storm's time, the first is matched
    genesis2 = [g, g, None, dt.datetime(2013, 9, 1, 12), dt.datetime(2013, 7, 12, 0)]
    rows2 = QS.link_year(data, times, genesis2, events, keep=[True] * 5)
    a2 = next(r for r in rows2 if r["qtrack_genesis"] == "2013-07-08 18Z")
    assert a2["status"] == "matched" and a2["sid"] == "2013A"
    assert QS.link_year(data, times, genesis, events, keep=[False] * 5) == []


def test_a_colon_joined_ibtracs_name_is_found_by_the_tag_and_by_either_part(events):
    """Joan 1988 is one IBTrACS entry named JOAN:MIRIAM in both basin files; it was the
    one key of 512 with no candidate anywhere until this form was handled. The tag,
    each part, and the hyphenated tag all find it; a whole-name rule, so JOAN does not
    find a storm named JOANNE."""
    assert QS.ibtracs_name_forms("JOAN:MIRIAM") == {"JOAN:MIRIAM", "JOAN-MIRIAM", "JOAN",
                                                    "MIRIAM"}
    assert QS.ibtracs_name_forms("CESAR") == {"CESAR"}
    e = events.copy()
    e.loc[e["SID"] == "2013F", "NAME"] = "JOAN:MIRIAM"
    e.loc[e["SID"] == "2013H", "NAME"] = "JOANNE"
    for tag in ("JOAN-MIRIAM", "JOAN", "MIRIAM"):
        m = QS.match_storm(e, 2013, tag, dt.datetime(2013, 9, 1, 12), (16.0, -41.0))
        assert m["status"] == "matched" and m["sid"] == "2013F", (tag, m)
        assert m["matched_name"] == "JOAN:MIRIAM"
    # a JOAN tag on JOANNE's own date and position: the only candidate is still the
    # JOAN:MIRIAM entry nineteen days away, so the state is a window rejection naming
    # it, and JOANNE (2013H) never entered the candidate set
    m = QS.match_storm(e, 2013, "JOAN", dt.datetime(2013, 9, 20, 12), (16.0, -150.0))
    assert m["status"] == "outside_window" and m["nearest_sid"] == "2013F"
