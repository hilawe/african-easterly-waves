"""Filters for the QTrack version 2 track files (ERA5_WITH_EPAC), Atlantic and repeats.

WHAT THE FILES ARE. One netCDF per year, systems along a `system` dimension and
six-hourly steps along `time`, with `AEW_lon`, `AEW_lat`, `AEW_strength`, `TC_name` (a
string per system, "N/A" for a non-developer), `TC_gen_time`, and two integer basin
fields, `basin_des` (system x time) and `first_basin_des` (system), which carry no
attributes. Their key was DERIVED from where tracks change code
(scripts/diagnose_qtrack_basins.py, artifact docs/aewc_v2/artifacts/qtrack_basin_codes.json)
and is recorded as an inference awaiting the authors' confirmation:

    1  over land in the Americas          5  eastern Pacific, 140W to the coast
    2  Africa, east of the 17W coast       6  Caribbean and Gulf, west of 60W
    4  central Pacific, west of 140W       7  Atlantic, 60W to 17W
    8  east of the African window (rare)   9  south of the equator

TWO RULES, both DECISIONS recorded here rather than facts read from the archive.

ATLANTIC. A system is Atlantic when ANY valid step carries code 2, 6 or 7 AND its first
valid code is not a Pacific one (4 or 5). Waves move westward, so a system that was ever
on the Atlantic side is an Atlantic wave whatever it did afterwards, and a system whose
steps are only 1, 4, 5, 8 or 9 never was. `first_basin_des` alone is deliberately NOT
the criterion: a wave first detected over Central American land (code 1) that then runs
along the Caribbean is Atlantic. The Pacific-origin exclusion exists because a review
found six systems that begin in code 5 and later touch an Atlantic-side code: three
named eastern Pacific hurricanes whose stored tracks extend across North America (Rick
2009, Simon 2014, Darby 2016), and three that touch code 6 at the Panama border and
then run west into the Pacific, Dora 1999 (one code-6 step near 80W 8.7N) and two
untagged tracks. Under the any-step rule alone the four named ones counted as Atlantic
developers. The stored trajectories do not establish the physical identity of a
post-storm remnant, so the criterion is stated as it is measured: first valid code
Pacific, some later step Atlantic-side.

SHARED TAILS, formerly "repeats". The dataset's authors report repeated systems needing
filtering. Measured on the 44 files, every pair of Atlantic-side systems that shares four
or more identical (longitude, latitude) positions has ONE geometry: different histories
before the first shared step, then identical positions and validity through a common
endpoint. 107 such pairs, forming 97 groups (92 pairs and 5 triples). None is a track
contained within another. The INSTALLED QTrack implementation (the `qtrack` package in
this environment, `tracking.py`, the block guarded by `two_wave_distance <
merge_distance`) CAN produce that shape: when two tracked waves come within its merge
distance (default 500 km) it compares the last finite LONGITUDE of each track and copies
the remaining longitudes and latitudes of the one ending further west into the other from
that step on. That is what the installed code computes, read from the code, not from its
comment about lengths. WHETHER THIS IS HOW THE ARCHIVE'S FILES WERE PRODUCED is not
established: the producing revision and its settings have not been confirmed with the
authors, so the archive's procedure is an inference from an implementation capable of
imposing the shape. The differing heads are separate detections under that inference.
Whether they are separate physical waves, one wave detected twice, or a tracking error is
NOT established by the stored tracks, and is a question for the archive's authors.

THE POLICY, a decision: KEEP EVERY BASIN-SELECTED TRACK. Removing the shorter member of
each pair, the first version of this module's rule, discarded 96 tracks and with them
1,009 coordinate-and-time records present in no retained track; it also left 263 copied
records in place, from the six pairs retained for a storm name and two pairs sharing
fewer than four positions. Nothing in the geometry justifies choosing one head over the
other, so nothing is removed. The old rule survives only as `repeat_policy="drop_shorter"`,
so the summary can still report what it would have done. The shared tails are recorded
per year (both systems, first and last shared step, shared count) and their effect is
MEASURED by `observation_counts`, which counts distinct (time step, longitude, latitude)
records among the kept tracks: with everything kept, 117,769 valid records hold 114,318
distinct ones and 3,451 copies. Any density or longitude-time analysis owns the choice of
which copy to count, and reads that choice from one place rather than inventing it.

STORM IDENTITY. Counting developers by TC_name collapses distinct storms: in 1981 four
systems tagged "UNNAMED" carry four different genesis times. `storm_keys` therefore
identifies a storm by (name, TC_gen_time), which is nanoseconds since 1970 in every
file, and a storm tagged on both heads of a shared-tail pair counts once. A TAGGED TRACK
WHOSE GENESIS IS MISSING, ZERO OR NON-FINITE HAS AN UNRESOLVED IDENTITY: it is counted
separately and never merged with another track of the same name, because a missing
identity component is not a shared one (a first version keyed such tracks by name alone,
which silently restored name-only counting). The key is a consistency convention within
these files, not a link to the authoritative best track; 512 distinct keys in the
Atlantic-side population is a count of tags, not of verified storms or of their genesis
basins. Two of them (Priscilla and Octave 2013) are storms that formed in the eastern
Pacific on Atlantic-origin tracks, which the tag cannot show.

TIME. 43 files store the time axis as hours since 1900-01-01 and 2024 as seconds since
1970-01-01; TC_gen_time is nanoseconds since 1970 in all 44. Decode each through its
own units. This module never decodes the time axis; it works in time-step INDEX within a
file, which is what identical stored positions share.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

__all__ = [
    "ATLANTIC_CODES",
    "PACIFIC_CODES",
    "NON_ATLANTIC_CODES",
    "BASIN_KEY",
    "DEFAULT_MIN_SHARED",
    "read_year",
    "atlantic_systems",
    "duplicate_clusters",
    "keep_longest",
    "keep_longest_and_named",
    "filter_year",
    "storm_keys",
    "observation_counts",
    "observation_owners",
    "shared_tail_pairs",
    "write_subset",
]

BASIN_KEY = {1: "land, Americas", 2: "Africa", 4: "central Pacific",
             5: "eastern Pacific", 6: "Caribbean and Gulf", 7: "Atlantic",
             8: "east of the African window", 9: "south of the equator"}
ATLANTIC_CODES = frozenset({2, 6, 7})
PACIFIC_CODES = frozenset({4, 5})
NON_ATLANTIC_CODES = frozenset({1, 4, 5, 8, 9})
DEFAULT_MIN_SHARED = 4


def read_year(path):
    """The per-system arrays a filter needs, NaN where a track has no position."""
    import netCDF4 as nc

    d = nc.Dataset(path)
    try:
        out = {"lon": np.ma.filled(d["AEW_lon"][:], np.nan).astype(float),
               "lat": np.ma.filled(d["AEW_lat"][:], np.nan).astype(float),
               "basin": np.ma.filled(d["basin_des"][:], np.nan).astype(float),
               "name": np.asarray(d["TC_name"][:]).astype(str),
               "gen_time": np.ma.filled(d["TC_gen_time"][:], np.nan).astype(float)
               if "TC_gen_time" in d.variables else np.full(len(d.dimensions["system"]), np.nan),
               "system": np.asarray(d["system"][:])}
    finally:
        d.close()
    if not (out["lon"].shape == out["lat"].shape == out["basin"].shape):
        raise ValueError("AEW_lon, AEW_lat and basin_des must share one shape")
    return out


def atlantic_systems(basin):
    """True per system when any valid step is Atlantic-side and the first is not Pacific."""
    basin = np.asarray(basin, dtype=float)
    if basin.ndim != 2:
        raise ValueError("basin must be (system, time)")
    codes = np.where(np.isnan(basin), -1, basin)
    any_atlantic = np.isin(codes, list(ATLANTIC_CODES)).any(axis=1)
    first = np.full(basin.shape[0], -1.0)
    for i in range(basin.shape[0]):
        valid = codes[i][codes[i] >= 0]
        if valid.size:
            first[i] = valid[0]
    pacific_origin = np.isin(first, list(PACIFIC_CODES))
    return any_atlantic & ~pacific_origin


def duplicate_clusters(lon, lat, min_shared=DEFAULT_MIN_SHARED):
    """Clusters of systems that share at least `min_shared` identical positions.

    Identity is exact equality of BOTH longitude and latitude at the same timestep,
    counted over steps where both systems have a position. Sharing is transitive: if a
    shares with b and b with c, the three are one cluster even when a and c share
    nothing, because the middle track is what ties the record together. Returns a list
    of sorted index lists, singletons included, covering every system once.
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    if lon.shape != lat.shape or lon.ndim != 2:
        raise ValueError("lon and lat must be (system, time) and share one shape")
    if min_shared < 1:
        raise ValueError("min_shared must be at least 1")
    n = lon.shape[0]
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    valid = ~np.isnan(lon) & ~np.isnan(lat)
    for i in range(n):
        for j in range(i + 1, n):
            both = valid[i] & valid[j]
            if both.sum() < min_shared:
                continue
            same = (lon[i][both] == lon[j][both]) & (lat[i][both] == lat[j][both])
            if same.sum() >= min_shared:
                parent[find(i)] = find(j)
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return sorted(sorted(g) for g in groups.values())


def keep_longest(clusters, n_valid):
    """One survivor per cluster: the most valid steps, ties to the lower index."""
    n_valid = np.asarray(n_valid)
    keep = np.zeros(len(n_valid), dtype=bool)
    for group in clusters:
        best = min(group, key=lambda i: (-int(n_valid[i]), i))
        keep[best] = True
    return keep


def keep_longest_and_named(clusters, n_valid, names):
    """keep_longest, plus every developer whose name the cluster's survivor lacks.

    A cluster of two different storms keeps both; a cluster whose longer track is
    untagged and whose shorter track is a storm keeps both; a cluster whose members are
    the same storm keeps the longest only; a cluster with two twins of one missing storm
    keeps the longer twin only. "N/A" is not a name.
    """
    names = np.asarray(names).astype(str)
    n_valid = np.asarray(n_valid)
    keep = keep_longest(clusters, n_valid)
    for group in clusters:
        survivor = min(group, key=lambda i: (-int(n_valid[i]), i))
        missing = {names[i] for i in group if names[i] != "N/A"} - {names[survivor]}
        for name in missing:
            bearers = [i for i in group if names[i] == name]
            keep[min(bearers, key=lambda i: (-int(n_valid[i]), i))] = True
    return keep


def shared_tail_pairs(lon, lat, min_shared=DEFAULT_MIN_SHARED, among=None):
    """Every pair sharing at least `min_shared` identical positions, with its span.

    Returns a list of dicts with the two indices, the first and last shared time-step
    index, and the shared count. `among` restricts the pairs to those indices (the
    Atlantic-side systems); every pair is reported, not only the transitive groups.
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    idx = list(range(lon.shape[0])) if among is None else [int(i) for i in among]
    valid = ~np.isnan(lon) & ~np.isnan(lat)
    out = []
    for p, i in enumerate(idx):
        for j in idx[p + 1:]:
            both = valid[i] & valid[j]
            if both.sum() < min_shared:
                continue
            same = both & (lon[i] == lon[j]) & (lat[i] == lat[j])
            if same.sum() >= min_shared:
                where = np.where(same)[0]
                out.append({"a": i, "b": j, "first_shared": int(where[0]),
                            "last_shared": int(where[-1]), "n_shared": int(same.sum())})
    return out


def storm_keys(names, gen_time):
    """One stable key per developer, (name, genesis time); None for a non-developer;
    the string "unresolved" for a tagged track whose genesis is missing, zero or
    non-finite.

    Name alone collapses distinct storms ("UNNAMED" appears four times in 1981 with four
    genesis times); name plus genesis time separates them and lets a storm tagged on both
    heads of a shared-tail pair count once. A missing genesis is NOT a key component:
    two tagged tracks with the same name and no genesis are two unresolved identities,
    not one storm, since the data justify neither one nor two.
    """
    names = np.asarray(names).astype(str)
    gen_time = np.asarray(gen_time, dtype=float)
    if names.shape != gen_time.shape:
        raise ValueError("names and gen_time must have one entry per system")
    keys = []
    for nm, g in zip(names, gen_time):
        if nm == "N/A":
            keys.append(None)
        elif g != g or not np.isfinite(g) or g <= 0:
            keys.append("unresolved")
        else:
            keys.append((nm, int(g)))
    return keys


def observation_counts(lon, lat, keep):
    """Valid, distinct and copied (time step, longitude, latitude) records among `keep`.

    Distinctness is exact equality at the same time-step index within one file, matching
    the shared positions observed in the archive. A record shared by three tracks is one
    distinct record and two copies, so groups larger than a pair are handled without
    subtracting the same position once per pair.
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    keep = np.asarray(keep, dtype=bool)
    seen = set()
    total = 0
    for i in np.where(keep)[0]:
        valid = np.where(~np.isnan(lon[i]) & ~np.isnan(lat[i]))[0]
        total += int(valid.size)
        for t in valid:
            seen.add((int(t), float(lon[i, t]), float(lat[i, t])))
    return {"valid_records": total, "distinct_records": len(seen),
            "copied_records": total - len(seen)}


def observation_owners(lon, lat, keep, system):
    """One owning track per (time step, longitude, latitude) record among `keep`.

    THE CONVENTION, declared once here so every density or longitude-time analysis
    applies the same one: a record held by several kept tracks is owned by the track
    with the MOST valid records, ties to the LOWER published system number. Returns a
    boolean `owned` mask (system x time), True where a track's record is the one to
    count, and the owner's system number per record for inspection. Summing `owned`
    over kept tracks equals `observation_counts(...)["distinct_records"]` by
    construction, and a test holds that identity. Ownership decides counting only; no
    coordinate is changed, and intensity or other track-specific fields at a shared
    position still need their own explicit rule.
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    keep = np.asarray(keep, dtype=bool)
    system = np.asarray(system).astype(int)
    valid = ~np.isnan(lon) & ~np.isnan(lat)
    n_valid = valid.sum(axis=1)
    owner = {}
    for i in np.where(keep)[0]:
        for t in np.where(valid[i])[0]:
            key = (int(t), float(lon[i, t]), float(lat[i, t]))
            cur = owner.get(key)
            if cur is None or (n_valid[i], -system[i]) > (n_valid[cur], -system[cur]):
                owner[key] = int(i)
    owned = np.zeros(lon.shape, dtype=bool)
    for (t, _lo, _la), i in owner.items():
        owned[i, t] = True
    owner_system = {k: int(system[i]) for k, i in owner.items()}
    return {"owned": owned, "owner_system": owner_system}


def filter_year(data, min_shared=DEFAULT_MIN_SHARED, repeat_policy="keep"):
    """Apply the basin rule and the chosen shared-tail policy to one year's arrays.

    `repeat_policy` is "keep" (the decision: nothing removed) or "drop_shorter" (the
    retired rule, kept so the summary can report what it would have removed). Returns
    the boolean masks `atlantic`, `duplicate` (removed under the policy) and `keep`,
    the transitive clusters, the shared-tail pairs among Atlantic systems, the
    observation counts among the kept tracks, the distinct storm keys, and counts.
    """
    if repeat_policy not in ("keep", "drop_shorter"):
        raise ValueError(f"unknown repeat_policy {repeat_policy!r}")
    atl = atlantic_systems(data["basin"])
    n_valid = (~np.isnan(data["lon"]) & ~np.isnan(data["lat"])).sum(axis=1)
    idx = np.where(atl)[0]
    clusters_local = duplicate_clusters(data["lon"][idx], data["lat"][idx], min_shared)
    clusters = [[int(idx[i]) for i in g] for g in clusters_local]
    pairs = shared_tail_pairs(data["lon"], data["lat"], min_shared, among=idx)
    if repeat_policy == "keep":
        keep = atl.copy()
    else:
        keep = keep_longest_and_named(clusters, n_valid, data["name"])
    duplicate = atl & ~keep
    developers = data["name"] != "N/A"
    keys = storm_keys(data["name"], data.get("gen_time", np.full(len(atl), np.nan)))
    kept_keys = {k for k, kp in zip(keys, keep) if kp and k is not None and k != "unresolved"}
    unresolved = sum(1 for k, kp in zip(keys, keep) if kp and k == "unresolved")
    return {"atlantic": atl, "duplicate": duplicate, "keep": keep, "clusters": clusters,
            "shared_tail_pairs": pairs,
            "observations": observation_counts(data["lon"], data["lat"], keep),
            "storm_keys": keys, "n_distinct_storms_kept": len(kept_keys),
            "n_unresolved_storm_identities_kept": int(unresolved),
            "n_systems": int(len(atl)), "n_atlantic": int(atl.sum()),
            "n_removed": int(duplicate.sum()), "n_kept": int(keep.sum()),
            "n_developer_tags_kept": int((keep & developers).sum()),
            "n_shared_tail_pairs": len(pairs),
            "min_shared": int(min_shared), "repeat_policy": repeat_policy}


def write_subset(src, dst, keep, min_shared=DEFAULT_MIN_SHARED):
    """Copy a year file keeping only the systems where `keep` is True.

    Every dimension, variable and attribute is copied; variables with a `system`
    dimension are subset along it, everything else (time, the grid, curv_data_mean) is
    copied whole. The `system` coordinate keeps its ORIGINAL numbers so a kept track
    can be traced back to the published file. `min_shared` is the shared-position
    threshold the caller actually used, written into the file's own metadata as a number
    as well as in words, because a review ran the driver at three and the file still
    said four.
    """
    min_shared = int(min_shared)
    import netCDF4 as nc

    keep = np.asarray(keep, dtype=bool)
    s = nc.Dataset(src)
    try:
        if keep.shape != (len(s.dimensions["system"]),):
            raise ValueError("keep must have one entry per system")
        d = nc.Dataset(dst, "w")
        try:
            d.setncatts({k: s.getncattr(k) for k in s.ncattrs()})
            d.setncattr("aew_filter", "Atlantic (any step in basin codes 2, 6, 7 and "
                        "first valid code not 4 or 5, under a key inferred from the "
                        "tracks). Nothing removed for shared positions: tracks sharing "
                        f"{min_shared} or more identical positions are listed in the "
                        "summary as shared-tail pairs, a shape the installed QTrack "
                        "post-processing can impose by copying coordinates; the "
                        "archive's producing revision is unconfirmed. System numbers "
                        "are the originals")
            d.setncattr("aew_filter_min_shared", min_shared)
            for name, dim in s.dimensions.items():
                d.createDimension(name, int(keep.sum()) if name == "system"
                                  else (None if dim.isunlimited() else len(dim)))
            for name, v in s.variables.items():
                dtype = str if v.dtype == str or v.dtype.kind in "OSU" else v.dtype
                fill = getattr(v, "_FillValue", None)
                out = d.createVariable(name, dtype, v.dimensions,
                                       fill_value=fill if dtype is not str else None)
                out.setncatts({k: v.getncattr(k) for k in v.ncattrs()
                               if k != "_FillValue"})
                values = v[:]
                if "system" in v.dimensions:
                    axis = v.dimensions.index("system")
                    values = np.compress(keep, values, axis=axis)
                if dtype is str:
                    for i, val in enumerate(np.asarray(values).astype(str)):
                        out[i] = str(val)
                else:
                    out[:] = values
        finally:
            d.close()
    finally:
        s.close()
