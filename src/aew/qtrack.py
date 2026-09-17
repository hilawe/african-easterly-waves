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

ATLANTIC. A system is Atlantic when ANY valid step carries code 2, 6 or 7. Waves move
westward, so a system that was ever on the Atlantic side is an Atlantic wave whatever
it did afterwards, and a system whose steps are only 1, 4, 5, 8 or 9 never was.
`first_basin_des` is deliberately NOT the criterion: a wave first detected over
Central American land (code 1) that then runs along the Caribbean is Atlantic.

REPEATS. The dataset's authors report repeated systems needing filtering. Measured on
the 44 files, the repeats are POSITIONAL: pairs of systems that share identical
(longitude, latitude) values at four or more valid steps, 182 such pairs, and the
duplicate storm-name tags follow from them (1988 Gilbert is two systems of 110 and 106
steps). A recurring "UNNAMED" tag is not a repeat, since HURDAT labels several distinct
unnamed storms that way. Systems are clustered transitively on shared positions and the
LONGEST track of each cluster is kept, ties broken by the lower system index, AND any
developer in the cluster whose storm name the survivor does not carry is kept as well,
because dropping it would remove a named storm from the record. Measured on the 44
files, 31 clusters hold a developer; in 25 the survivor carries the same name, in 2 the
longer twin is untagged (Bertha 1996, Harvey 2017), and in 4 the two tracks are
different storms sharing positions (Priscilla and Octave 2013, Hermine and Fiona 2016,
Gamma and Delta 2020, Philippe and Rina 2023). Those six are retained and listed for a
person to adjudicate, not resolved by the code. The threshold of shared steps is a
parameter, and the driver reports the count of removed systems at 2, 4 and 8 so the
choice of 4 (one day) is legible.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

__all__ = [
    "ATLANTIC_CODES",
    "NON_ATLANTIC_CODES",
    "BASIN_KEY",
    "DEFAULT_MIN_SHARED",
    "read_year",
    "atlantic_systems",
    "duplicate_clusters",
    "keep_longest",
    "keep_longest_and_named",
    "filter_year",
    "write_subset",
]

BASIN_KEY = {1: "land, Americas", 2: "Africa", 4: "central Pacific",
             5: "eastern Pacific", 6: "Caribbean and Gulf", 7: "Atlantic",
             8: "east of the African window", 9: "south of the equator"}
ATLANTIC_CODES = frozenset({2, 6, 7})
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
               "system": np.asarray(d["system"][:])}
    finally:
        d.close()
    if not (out["lon"].shape == out["lat"].shape == out["basin"].shape):
        raise ValueError("AEW_lon, AEW_lat and basin_des must share one shape")
    return out


def atlantic_systems(basin):
    """True per system when any valid step carries an Atlantic-side code."""
    basin = np.asarray(basin, dtype=float)
    if basin.ndim != 2:
        raise ValueError("basin must be (system, time)")
    codes = np.where(np.isnan(basin), -1, basin)
    return np.isin(codes, list(ATLANTIC_CODES)).any(axis=1)


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
    the same storm keeps the longest only. "N/A" is not a name.
    """
    names = np.asarray(names).astype(str)
    keep = keep_longest(clusters, n_valid)
    for group in clusters:
        survivor = next(i for i in group if keep[i])
        for i in group:
            if names[i] != "N/A" and names[i] != names[survivor]:
                keep[i] = True
    return keep


def filter_year(data, min_shared=DEFAULT_MIN_SHARED):
    """Apply both rules to one year's arrays.

    Returns a dict with the boolean masks `atlantic`, `duplicate` (removed as a repeat
    among Atlantic systems) and `keep`, the clusters, and the counts a summary needs.
    Deduplication runs among the Atlantic systems only, so a Pacific twin of an Atlantic
    track can neither remove it nor be counted as its repeat.
    """
    atl = atlantic_systems(data["basin"])
    n_valid = (~np.isnan(data["lon"]) & ~np.isnan(data["lat"])).sum(axis=1)
    idx = np.where(atl)[0]
    clusters_local = duplicate_clusters(data["lon"][idx], data["lat"][idx], min_shared)
    clusters = [[int(idx[i]) for i in g] for g in clusters_local]
    # clusters hold ORIGINAL system indices, and n_valid is indexed the same way
    longest = keep_longest(clusters, n_valid)
    keep = keep_longest_and_named(clusters, n_valid, data["name"])
    retained = keep & ~longest
    duplicate = atl & ~keep
    developers = data["name"] != "N/A"
    return {"atlantic": atl, "duplicate": duplicate, "keep": keep, "clusters": clusters,
            "retained_for_name": retained,
            "n_systems": int(len(atl)), "n_atlantic": int(atl.sum()),
            "n_duplicates_removed": int(duplicate.sum()), "n_kept": int(keep.sum()),
            "n_developers_kept": int((keep & developers).sum()),
            "n_developers_removed_as_duplicate": int((duplicate & developers).sum()),
            "n_retained_for_name": int(retained.sum()),
            "min_shared": int(min_shared)}


def write_subset(src, dst, keep):
    """Copy a year file keeping only the systems where `keep` is True.

    Every dimension, variable and attribute is copied; variables with a `system`
    dimension are subset along it, everything else (time, the grid, curv_data_mean) is
    copied whole. The `system` coordinate keeps its ORIGINAL numbers so a kept track
    can be traced back to the published file.
    """
    import netCDF4 as nc

    keep = np.asarray(keep, dtype=bool)
    s = nc.Dataset(src)
    try:
        if keep.shape != (len(s.dimensions["system"]),):
            raise ValueError("keep must have one entry per system")
        d = nc.Dataset(dst, "w")
        try:
            d.setncatts({k: s.getncattr(k) for k in s.ncattrs()})
            d.setncattr("aew_filter", "Atlantic (any step in basin codes 2, 6, 7), "
                        "positional repeats removed except developers whose storm name "
                        "the surviving twin lacks. System numbers are the originals")
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
