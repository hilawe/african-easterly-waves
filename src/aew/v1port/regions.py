"""Which basin a wave came from, ported from generate_ew_stats_f.m.

The record files one track per basin, and until this existed the port could produce tracks
and not file them. It is the only part of generate_ew_stats_f.m the tracker needs: the rest
of that function composites satellite fields the project does not yet have a source for.

WHAT VERSION 1 DOES, and every clause of it matters:

    if j == 1;
      for rg = 1:size(region,2);
        if inpolygon(round(...meanlon(j)/0.25)*0.25, round(...meanlat(j)/0.25)*0.25,
                     region(rg).lon, region(rg).lat) == 1;
          ew_tracks(i).region = rg;
          ew_tracks(i).region_name = region(rg).name;
          break

  THE FIRST OBSERVATION ONLY. A wave is filed by where it STARTED, not where it spent its
  life or where it ended. A track that forms over Africa and dies in the Atlantic is an
  Africa wave.

  ROUNDED TO A QUARTER DEGREE FIRST. The original's own comment says the polygons were
  drawn on a 0.25 land-sea grid, "hence the rounding", so a point is snapped to that grid
  before being tested. Skipping it moves the answer for any track starting within an eighth
  of a degree of a boundary.

  FIRST MATCH WINS, IN THE STORED ORDER. The polygons overlap, and OTH is the whole domain
  box, so it only ever matches what nothing else did. Testing in a different order, or
  taking the best match rather than the first, files waves under different basins.

  A WAVE MATCHING NOTHING KEEPS NO REGION. The MATLAB simply leaves the field unset, and
  the writer's `strcmp` then never matches it, so the wave appears in no file at all. That
  is reproduced as None rather than quietly filed under OTH, because silently inventing a
  basin is worse than reporting that the wave has none.

THE POLYGONS ARE THE ORIGINAL'S OWN, read from regions.mat in the archived source rather
than redrawn. They are large, 500 to 1200 vertices each, and redrawing them by hand would
be a different definition wearing the same names.
"""

import functools
import os

import numpy as np

from .geometry import points_in_polygon

# generate_ew_stats_f.m: "Regional Polygons Created from 0.25 x 0.25 Land-Sea Grid, Hence
# the Rounding"
SNAP_DEGREES = 0.25

DEFAULT_REGIONS_FILE = os.path.join(
    "data", "aewc_v2_pilot", "v1_src", "regions.mat")


@functools.lru_cache(maxsize=4)
def load_regions(path=DEFAULT_REGIONS_FILE):
    """The eight basin polygons, IN THEIR STORED ORDER, which is part of the definition.

    Returns a tuple of (name, lon, lat). Order is preserved because the assignment takes
    the first match and the last entry is a catch-all covering the whole domain.
    """
    from scipy.io import loadmat

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} is missing. The basin polygons are version 1's own, archived with its "
            f"source, and are not reconstructible from the published record; redrawing "
            f"them would be a different definition under the same names.")
    entries = loadmat(path)["regions"][0]
    out = []
    for entry in entries:
        name = str(entry["name"][0])
        lon = np.asarray(entry["lon"], dtype=float).ravel()
        lat = np.asarray(entry["lat"], dtype=float).ravel()
        out.append((name, lon, lat))
    return tuple(out)


def snap(value, step=SNAP_DEGREES):
    """Round to the grid the polygons were drawn on, as the original does before testing."""
    return np.round(np.asarray(value, dtype=float) / step) * step


def region_of_point(lon, lat, regions=None):
    """The first basin whose polygon contains the point, or None.

    None rather than a fallback: version 1 leaves the field unset when nothing matches, and
    the writer then files the wave nowhere. Inventing a basin would put a wave in a record
    version 1 never put it in.
    """
    regions = regions if regions is not None else load_regions()
    x, y = float(snap(lon)), float(snap(lat))
    for name, poly_lon, poly_lat in regions:
        if points_in_polygon(poly_lon, poly_lat, [x], [y])[0]:
            return name
    return None


def assign_region(track, regions=None):
    """Set `region_name` on one track from its FIRST observation, and return it."""
    lon = np.asarray(track["meanlon"], dtype=float)
    lat = np.asarray(track["meanlat"], dtype=float)
    if lon.size == 0:
        raise ValueError("a track with no observations has no origin to file it by")
    name = region_of_point(lon[0], lat[0], regions)
    if name is not None:
        track["region_name"] = name
    return name


def assign_regions(tracks, regions=None):
    """Assign every track, and report how many matched nothing.

    Returns (tracks, unassigned_count). The count is returned rather than logged because a
    run where many waves match no basin is a run whose record is missing waves, and that
    should be visible to the caller rather than buried.
    """
    regions = regions if regions is not None else load_regions()
    unassigned = 0
    for track in tracks:
        if assign_region(track, regions) is None:
            unassigned += 1
    return tracks, unassigned
