"""Check the port against the record version 1 actually produced.

WHY THIS EXISTS. Every other test in this port compares the code to a reading of the
MATLAB, and repeated independent checking found twenty-three places that reading was wrong. Reading
harder is not the fix. The fix is an oracle that does not come from the reading, and one
is already here: NCEI dataset C00784, version 1's own published output, in `data/aewc`.

WHAT CAN BE CHECKED WITHOUT ANY REANALYSIS, which is what this module does. The published
files carry finished tracks: times, centroid positions, latitude extremes, wavelengths and
vorticity statistics. Version 1 applied its own filters before writing them, so the record
is a set of 12,163 worked examples of what those filters accept. Two parts of the port can
be held against that directly.

  THE FINAL FILTER. Every published track passed version 1's minimum-speed test. So
  `finalize_tracks` run over the published tracks must keep ALL of them. Anything it drops
  is a track version 1 kept and the port would not, which is a defect in the port and not a
  matter of opinion. This is the strongest available check on that function, and it needs
  no wind field at all.

  THE DUPLICATION COUNTER. The project's headline claim about version 1 is that a large
  share of its observations duplicate another track's. `duplicate_observations` is the
  measure that claim rests on, and until now it had only been run on synthetic fixtures.
  Running it on the record both exercises it at scale and puts a number on the claim.

WHAT THIS CANNOT CHECK, stated so the module is not read as more than it is. Detection,
contour merging, and the matching rules inside a timestep all need the reanalysis fields
version 1 saw, and those are not here. This is a partial oracle covering the tail of the
pipeline. The full check is the ERA-Interim run, which needs data that has to be retrieved.
"""

import glob
import os

import numpy as np


def read_tracks(path):
    """Read one published file back into the track shape the port uses.

    The file stores a contiguous ragged array: `count` gives each track's length and the
    sample variables run end to end, so track n is the slice starting at the sum of the
    counts before it. Fill values come back as a masked array and are converted to NaN.
    """
    import netCDF4 as nc

    fields = {"time": "time", "meanlat": "lat", "meanlon": "lon",
              "maxlat": "maxlat", "minlat": "minlat",
              "meanlon_maxlat": "meanlonatmaxlat",
              "meanlon_minlat": "meanlonatminlat",
              "wavelength": "wavelength"}
    with nc.Dataset(path) as ds:
        counts = np.asarray(ds.variables["count"][:], dtype=int)
        # `np.asarray` on a masked array DROPS THE MASK, so wrapping the read in it before
        # filling turns every fill value into a real -999. That is what a first version did,
        # and it is silent: the wavelengths came back as measurements of minus 999 km. Fill
        # first, convert after.
        columns = {}
        for key, var in fields.items():
            if var not in ds.variables:
                continue
            raw = ds.variables[var][:]
            columns[key] = np.ma.filled(np.ma.masked_invalid(raw.astype(float)), np.nan)
    offsets = np.concatenate([[0], np.cumsum(counts)])
    tracks = []
    for i, n in enumerate(counts):
        lo, hi = offsets[i], offsets[i + 1]
        track = {key: list(col[lo:hi]) for key, col in columns.items()}
        # `step` is not stored; version 1's records are six-hourly, so it is recoverable
        # from the times themselves rather than assumed to be contiguous.
        track["step"] = [int(round((t - track["time"][0]) * 4)) for t in track["time"]]
        tracks.append(track)
    return tracks


def read_record(directory="data/aewc", pattern="*.nc"):
    """Every track in the published files under `directory`, with the file each came from."""
    out = []
    for path in sorted(glob.glob(os.path.join(directory, pattern))):
        for track in read_tracks(path):
            track["source_file"] = os.path.basename(path)
            out.append(track)
    return out


def segment_speeds_ms(track):
    """Consecutive-segment speeds in metres per second, as the final filter measures them."""
    from .geometry import great_circle_distance

    lat = np.asarray(track["meanlat"], dtype=float)
    lon = np.asarray(track["meanlon"], dtype=float)
    times = np.asarray(track["time"], dtype=float)
    if times.size < 2:
        return np.array([])
    distance_km = np.array([float(great_circle_distance(lat[i], lon[i],
                                                        lat[i + 1], lon[i + 1], "km"))
                            for i in range(times.size - 1)])
    seconds = np.diff(times) * 24.0 * 3600.0
    with np.errstate(divide="ignore", invalid="ignore"):
        return (distance_km * 1000.0) / seconds


def check_speed_filter(tracks):
    """Does the port's final filter keep every track version 1 kept?

    Returns a dict. `rejected` counts the published tracks the port would have dropped,
    which should be zero: version 1 applied this filter before writing them, so a rejection
    is the port disagreeing with the original about a case the original has already ruled
    on.

    THE SMOOTHING IS THE ONE COMPLICATION and it is worth stating rather than working
    around silently. Version 1 smooths the coordinates of the tracks it KEEPS, after
    deciding, so the positions in the file are not the positions the filter saw. Smoothing
    shortens a wiggly path, so a median segment speed computed from the published
    coordinates is a LOWER BOUND on the one version 1 tested. A track that passes on the
    published coordinates therefore certainly passed on the originals, and one that fails
    by a hair may not be a real disagreement. Both counts are returned.
    """
    from .association import MIN_SPEED_MS

    rejected, marginal, medians = [], 0, []
    for track in tracks:
        speeds = segment_speeds_ms(track)
        if speeds.size == 0:
            continue
        median = float(np.median(speeds))
        medians.append(median)
        if not median >= MIN_SPEED_MS:
            rejected.append(track)
            if median >= MIN_SPEED_MS * 0.9:
                marginal += 1
    return {"tracks": len(tracks), "rejected": len(rejected),
            "rejected_within_ten_percent": marginal,
            "median_speed_ms": float(np.median(medians)) if medians else float("nan"),
            "slowest_median_ms": float(np.min(medians)) if medians else float("nan"),
            "rejected_tracks": rejected}


def smoothing_sensitivity(tracks, passes=2):
    """How much each extra smoothing pass reduces a track's median segment speed.

    WHY THIS IS NOT A CURIOSITY. The check above compares against coordinates version 1
    smoothed after filtering, so its speeds are a lower bound and the natural correction is
    to ask what one more pass costs and scale by the inverse. That correction is WRONG IN A
    KNOWN DIRECTION, and this measures by how much: smoothing has strong diminishing
    returns, so a pass applied to already-smoothed coordinates removes far less than the
    first pass removed from the raw ones. On the published record the first extra pass
    costs about 11 percent of the median speed and the second about 3, so scaling by the
    first pass's inverse UNDERSTATES the original effect, and any track the corrected check
    still rejects may well have passed in version 1.

    Returns the median ratio for each successive pass, so the shape is visible rather than
    asserted.
    """
    ratios = [[] for _ in range(passes)]
    for track in tracks:
        current = track
        previous = np.median(segment_speeds_ms(current))
        if not (previous > 0):
            continue
        for p in range(passes):
            from .association import _moving_average_5
            nxt = dict(current)
            nxt["meanlat"] = list(_moving_average_5(current["meanlat"]))
            nxt["meanlon"] = list(_moving_average_5(current["meanlon"]))
            speeds = segment_speeds_ms(nxt)
            if speeds.size == 0:
                break
            now = np.median(speeds)
            if previous > 0:
                ratios[p].append(float(now / previous))
            current, previous = nxt, now
    return [float(np.median(r)) if r else float("nan") for r in ratios]


def duplication_rate(tracks, lon_range=None, lat_range=None, months=None):
    """Share of observations that another track also holds, measured on the record.

    Uses the same key as `association.duplicate_observations`, rounded to the precision the
    file stores rather than to nine decimals, because the published coordinates are single
    precision and two tracks that recorded the same trough agree only to that.

    THE ANSWER DEPENDS ON WHERE YOU LOOK, which is why the filters exist. Over the whole
    Africa record it is about 37 percent; over a tight West African summer corridor it is
    about 39; over 15W to 15E in June to September it is about 33. A claim of "about 40
    percent" is defensible only alongside the region it was measured over.
    """
    import datetime

    seen, duplicated, total = {}, 0, 0
    for number, track in enumerate(tracks):
        for t, lat, lon in zip(track["time"], track["meanlat"], track["meanlon"]):
            if not (np.isfinite(t) and np.isfinite(lat) and np.isfinite(lon)):
                continue
            if lon_range and not lon_range[0] <= lon <= lon_range[1]:
                continue
            if lat_range and not lat_range[0] <= lat <= lat_range[1]:
                continue
            if months:
                date = datetime.date(1900, 1, 1) + datetime.timedelta(days=float(t))
                if date.month not in months:
                    continue
            total += 1
            key = (round(float(t), 4), round(float(lat), 3), round(float(lon), 3))
            if key in seen and seen[key] != number:
                duplicated += 1
            else:
                seen.setdefault(key, number)
    return {"observations": total, "duplicated": duplicated,
            "fraction": duplicated / total if total else float("nan")}


def summarize(directory="data/aewc"):
    """Run every check that needs only the published record. Returns a dict of results."""
    tracks = read_record(directory)
    if not tracks:
        raise FileNotFoundError(
            f"no published files under {directory}; this check needs NCEI C00784")
    speed = check_speed_filter(tracks)
    speed.pop("rejected_tracks")
    return {"files": len({t["source_file"] for t in tracks}),
            "tracks": len(tracks),
            "speed_filter": speed,
            "smoothing_pass_ratios": smoothing_sensitivity(tracks[:2000]),
            "duplication_whole_record": duplication_rate(tracks),
            "duplication_summer_corridor": duplication_rate(
                tracks, lon_range=(-20, 20), lat_range=(5, 20), months={6, 7, 8, 9})}
