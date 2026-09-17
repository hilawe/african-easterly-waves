"""Link QTrack's storm tags to the IBTrACS best track, and read the basin at genesis.

WHY. QTrack tags a developing wave with a storm NAME and a genesis time, and nothing else.
A name is not an identifier (1981 carries four "UNNAMED" storms) and the wave track's
basin field at genesis is a property of an inferred key, so two things are needed before
any development statistic: the authoritative storm identifier, and the storm's genesis
basin from its own record. IBTrACS carries both: the `SID`, and the `SUBBASIN` of the
record at which the storm's genesis EVENT is defined. The North Atlantic file on disk
(`ibtracs.NA.list.v04r01.csv`) holds systems that occur within that basin, which is not
an origin guarantee, so the basin and subbasin are read from the selected event's own
record (`event_basin`, `event_subbasin`), and the first observation's are kept beside
them (`first_basin`, `first_subbasin`) because a system can cross between them before
it qualifies.

THE GENESIS EVENT is not the first observation. The installed QTrack (tracking.py, the
block that builds TC_gen_time from HURDAT through tropycal) drops the records whose type
is DB, LO or WV (disturbance, low, wave) and takes the FIRST REMAINING record as the
storm's genesis. `genesis_event` applies that rule to the local `USA_STATUS` field, which
is fully populated from 1981 on. The first observation of any stage is kept beside it as
a separate field, because a disturbance can cross a subbasin boundary before it
qualifies (Claudette 2003, Isaias 2020, Fred 2021, Debby 2024 are recorded as open
Atlantic at their first observation and Caribbean at their genesis event). A first
version of this module labeled the first observation as genesis; a review found the
four subbasin changes and three named storms (Marco 1996, Noel 2007, Bonnie 2022) that
the installed rule matches exactly and the first-observation rule rejected.

HOW A TAG IS LINKED, in states that are kept apart. Both declared tolerances are
applied to EVERY candidate of the same season and normalized name before any state is
decided (a first version decided ambiguity on time alone, and all eight of its
ambiguous cases had exactly one candidate satisfying both):
  matched            exactly one candidate's genesis event lies within `max_offset_hours`
                     of the tag AND within `max_distance_km` of the wave track's
                     position at the genesis step;
  ambiguous          more than one candidate satisfies both tolerances;
  rejected_distance  candidates lie within the window but none within the distance (the
                     nearest in time is recorded; a first version accepted three unnamed
                     candidates 3,784 to 6,370 km from the wave), or the wave has no
                     position at genesis so the distance cannot be checked;
  outside_window     no candidate event within the window (the nearest is recorded);
  no_candidate       no storm of that season and name in the file;
  shared_identifier  the chosen SID is also chosen by another key of the year, set on
                     both rows after the year is linked, for reconciliation.
The distance tolerance is QTrack's own genesis-association distance, 500 km
(`TC_merge_dist`, the same default as its merge distance), not a value chosen to remove
the outliers. The offset (tag minus event, hours) travels with every row.

THE WAVE OBSERVATION AT GENESIS, `qtrack_basin_at_genesis`, is the track's basin code at
the valid step nearest the genesis time, returned WITH the time difference, and it is
UNAVAILABLE with a stated reason when the genesis lies outside the file's time axis or
farther than `max_step_hours` from the nearest step (a first version returned the first
or last code silently for a genesis months outside the June to October axis).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "IBTRACS_COLUMNS",
    "NON_GENESIS_STATUS",
    "DEFAULT_MAX_OFFSET_HOURS",
    "DEFAULT_MAX_DISTANCE_KM",
    "DEFAULT_MAX_STEP_HOURS",
    "candidate_names",
    "haversine_km",
    "load_ibtracs_records",
    "genesis_event",
    "storm_events",
    "match_storm",
    "qtrack_basin_at_genesis",
    "link_year",
]

IBTRACS_COLUMNS = ["SID", "SEASON", "BASIN", "SUBBASIN", "NAME", "ISO_TIME", "NATURE",
                   "LAT", "LON", "USA_STATUS", "TRACK_TYPE"]
NON_GENESIS_STATUS = ("DB", "LO", "WV")          # the installed QTrack exclusion
DEFAULT_MAX_OFFSET_HOURS = 72.0
DEFAULT_MAX_DISTANCE_KM = 500.0                   # QTrack's TC_merge_dist default
DEFAULT_MAX_STEP_HOURS = 3.0                      # half a six-hourly step

NUMBER_WORDS = {"ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE",
                "TEN", "ELEVEN", "TWELVE", "THIRTEEN", "FOURTEEN", "FIFTEEN", "SIXTEEN",
                "SEVENTEEN", "EIGHTEEN", "NINETEEN", "TWENTY", "TWENTY-ONE", "TWENTY-TWO",
                "TWENTY-THREE", "TWENTY-FOUR", "TWENTY-FIVE", "TWENTY-SIX", "TWENTY-SEVEN",
                "TWENTY-EIGHT", "TWENTY-NINE", "THIRTY"}


def candidate_names(name):
    """The IBTrACS names a QTrack tag may appear under.

    QTrack's tags follow HURDAT naming: a depression that never earned a name carries
    its number spelled out ("TEN"), which IBTrACS records as "UNNAMED", and a storm that
    crossed between basins carries both names hyphenated ("JOAN-MIRIAM"), which IBTrACS
    records under each name separately.
    """
    name = str(name).strip().upper()
    if name in NUMBER_WORDS:
        return ["UNNAMED"]
    if "-" in name:
        return [name] + [p for p in name.split("-") if p]
    return [name]


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(2 * r * np.arcsin(np.sqrt(a)))


def load_ibtracs_records(path):
    """Every main-track record with its status and position, sorted by storm and time.

    Read with `keep_default_na=False`, because the basin code for the North Atlantic is
    the string "NA", which pandas would otherwise read as missing. The records are kept
    whole so the genesis event can be selected later, rather than reduced to a first
    observation here.
    """
    import pandas as pd

    ib = pd.read_csv(path, skiprows=[1], usecols=IBTRACS_COLUMNS, keep_default_na=False,
                     low_memory=False)
    ib = ib[ib["TRACK_TYPE"] == "main"].copy()
    ib["ISO_TIME"] = pd.to_datetime(ib["ISO_TIME"])
    ib["SEASON"] = ib["SEASON"].astype(int)
    ib["USA_STATUS"] = ib["USA_STATUS"].astype(str).str.strip()
    return ib.sort_values(["SID", "ISO_TIME"]).reset_index(drop=True)


def genesis_event(records):
    """The genesis event of one storm's records: the first whose status is not DB, LO
    or WV, as the installed QTrack defines it; None when no record qualifies."""
    ok = ~records["USA_STATUS"].isin(NON_GENESIS_STATUS)
    if not ok.any():
        return None
    return records[ok].iloc[0]


def storm_events(records):
    """One row per storm: identity, first observation, and genesis event (or none)."""
    import pandas as pd

    rows = []
    for sid, g in records.groupby("SID", sort=False):
        first = g.iloc[0]
        ev = genesis_event(g)
        rows.append({"SID": sid, "SEASON": int(first["SEASON"]), "NAME": str(first["NAME"]),
                     "first_basin": str(first["BASIN"]),
                     "event_basin": None if ev is None else str(ev["BASIN"]),
                     "first_time": first["ISO_TIME"], "first_subbasin": str(first["SUBBASIN"]),
                     "first_status": str(first["USA_STATUS"]),
                     "first_lat": float(first["LAT"]), "first_lon": float(first["LON"]),
                     "event_time": None if ev is None else ev["ISO_TIME"],
                     "event_subbasin": None if ev is None else str(ev["SUBBASIN"]),
                     "event_status": None if ev is None else str(ev["USA_STATUS"]),
                     "event_lat": None if ev is None else float(ev["LAT"]),
                     "event_lon": None if ev is None else float(ev["LON"])})
    return pd.DataFrame(rows)


def match_storm(events, season, name, genesis, wave_lat_lon,
                max_offset_hours=DEFAULT_MAX_OFFSET_HOURS,
                max_distance_km=DEFAULT_MAX_DISTANCE_KM):
    """Link one tag to the storm whose GENESIS EVENT it names, or say why it has none.

    `wave_lat_lon` is the wave track's position at the genesis step, or None when that
    observation is unavailable, in which case the spatial check cannot run and the
    state is `rejected_distance` with reason "no wave position".
    """
    import pandas as pd

    genesis = pd.Timestamp(genesis)
    if genesis.tzinfo is not None:
        genesis = genesis.tz_convert(None)
    cands = events[(events["SEASON"] == int(season))
                   & (events["NAME"].isin(candidate_names(name)))
                   & events["event_time"].notna()]
    if len(cands) == 0:
        return {"status": "no_candidate"}
    offsets = ((genesis - pd.to_datetime(cands["event_time"])).dt.total_seconds()
               / 3600.0).to_numpy()
    if wave_lat_lon is None:
        dists = np.full(len(cands), np.nan)
    else:
        dists = np.array([haversine_km(wave_lat_lon[0], wave_lat_lon[1], la, lo)
                          for la, lo in zip(cands["event_lat"], cands["event_lon"])])

    def describe(k):
        c = cands.iloc[k]
        out = {"nearest_sid": str(c["SID"]), "offset_hours": float(offsets[k]),
               "event_time": c["event_time"].strftime("%Y-%m-%d %HZ"),
               "event_status": c["event_status"],
               "first_observation": c["first_time"].strftime("%Y-%m-%d %HZ"),
               "first_subbasin": c["first_subbasin"], "first_basin": c["first_basin"],
               "event_subbasin": c["event_subbasin"], "event_basin": c["event_basin"],
               "event_lat_lon": [c["event_lat"], c["event_lon"]],
               "matched_name": str(c["NAME"])}
        if dists[k] == dists[k]:
            out["wave_to_event_km"] = float(dists[k])
        return out

    in_time = np.abs(offsets) <= max_offset_hours
    if not in_time.any():
        return {"status": "outside_window", **describe(int(np.argmin(np.abs(offsets))))}
    if wave_lat_lon is None:
        return {"status": "rejected_distance", "reason": "no wave position at genesis",
                "candidates_within_window": int(in_time.sum()),
                **describe(int(np.argmin(np.where(in_time, np.abs(offsets), np.inf))))}
    both = in_time & (dists <= max_distance_km)
    if both.sum() == 1:
        k = int(np.where(both)[0][0])
        return {"status": "matched", "sid": str(cands.iloc[k]["SID"]),
                "candidates_within_window": int(in_time.sum()), **describe(k)}
    if both.sum() > 1:
        k = int(np.argmin(np.where(both, np.abs(offsets), np.inf)))
        return {"status": "ambiguous", "candidates_within_window": int(in_time.sum()),
                "candidates_within_both": int(both.sum()), **describe(k)}
    k = int(np.argmin(np.where(in_time, np.abs(offsets), np.inf)))
    return {"status": "rejected_distance",
            "reason": f"nearest event {dists[k]:.0f} km from the wave, tolerance "
                      f"{max_distance_km:.0f}, none of {int(in_time.sum())} within the "
                      "window is closer",
            "candidates_within_window": int(in_time.sum()), **describe(k)}


def qtrack_basin_at_genesis(basin_row, lon_row, lat_row, times, genesis,
                            max_step_hours=DEFAULT_MAX_STEP_HOURS):
    """The track's basin code and position at the step nearest the genesis time.

    Returns a dict: `code`, `lat_lon` and `hours_from_step` when available, else
    `code` None with a `reason`: the genesis lies outside the file's time axis, the
    nearest step is farther than `max_step_hours`, or the track has no position there.
    """
    import pandas as pd

    basin_row = np.asarray(basin_row, dtype=float)
    lon_row = np.asarray(lon_row, dtype=float)
    lat_row = np.asarray(lat_row, dtype=float)
    t = pd.to_datetime(pd.Series(list(times)))
    genesis = pd.Timestamp(genesis)
    if genesis.tzinfo is not None:
        genesis = genesis.tz_convert(None)
    if genesis < t.iloc[0] or genesis > t.iloc[-1]:
        return {"code": None, "reason": "genesis outside the file's time axis",
                "axis": [t.iloc[0].strftime("%Y-%m-%d %HZ"), t.iloc[-1].strftime("%Y-%m-%d %HZ")]}
    diffs = (t - genesis).dt.total_seconds().to_numpy() / 3600.0
    k = int(np.argmin(np.abs(diffs)))
    if abs(diffs[k]) > max_step_hours:
        return {"code": None, "reason": f"nearest step {abs(diffs[k]):.1f} h away",
                "hours_from_step": float(diffs[k])}
    if np.isnan(basin_row[k]) or np.isnan(lon_row[k]) or np.isnan(lat_row[k]):
        return {"code": None, "reason": "track has no position at the genesis step",
                "hours_from_step": float(diffs[k])}
    return {"code": int(basin_row[k]), "lat_lon": [float(lat_row[k]), float(lon_row[k])],
            "hours_from_step": float(diffs[k])}


def link_year(data, times, genesis_datetimes, events, keep,
              max_offset_hours=DEFAULT_MAX_OFFSET_HOURS,
              max_distance_km=DEFAULT_MAX_DISTANCE_KM,
              max_step_hours=DEFAULT_MAX_STEP_HOURS):
    """Link every kept, tagged system of one year; one row per distinct storm key.

    A storm tagged on several tracks yields ONE row listing them all, with the wave
    observation taken from the first listed track. After all keys are linked, any SID
    chosen by more than one key turns those rows' state into `shared_identifier`.
    """
    rows = {}
    names = np.asarray(data["name"]).astype(str)
    sysno = np.asarray(data["system"]).astype(int)
    for i in np.where(np.asarray(keep, dtype=bool))[0]:
        if names[i] == "N/A" or genesis_datetimes[i] is None:
            continue
        key = (names[i], genesis_datetimes[i])
        if key in rows:
            rows[key]["systems"].append(int(sysno[i]))
            continue
        g = genesis_datetimes[i]
        wave = qtrack_basin_at_genesis(data["basin"][i], data["lon"][i], data["lat"][i],
                                       times, g, max_step_hours)
        m = match_storm(events, int(g.year), names[i], g, wave.get("lat_lon"),
                        max_offset_hours, max_distance_km)
        rows[key] = {"name": names[i], "qtrack_genesis": g.strftime("%Y-%m-%d %HZ"),
                     "systems": [int(sysno[i])], "wave_at_genesis": wave, **m}
    seen = {}
    for row in rows.values():
        if row["status"] == "matched":
            seen.setdefault(row["sid"], []).append(row)
    for sid, group in seen.items():
        if len(group) > 1:
            for row in group:
                row["status"] = "shared_identifier"
                row["shared_with_keys"] = [r["qtrack_genesis"] for r in group
                                           if r is not row]
    return list(rows.values())
