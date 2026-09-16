"""Trough-contour merging, ported from the archived AEWC version 1 MATLAB source.

Stage 4 of the port, and the last piece before the detection and association core.

WHAT IT DOES. Detection produces candidate wave troughs, several of which may belong to
one physical wave. This merges them: for each candidate it grows the connected region of
above-threshold curvature vorticity anomaly around it, and any other candidate whose
center falls inside the convex hull of that region is absorbed. Two further passes drop
near-duplicates and reject regions too small to be a wave.

Source files this replaces:
    merge_contours_f.m      the three merging passes
    isolate_region_f.m      the connected-region search the first pass uses

ONE DELIBERATE DIVERGENCE FROM THE ORIGINAL, which is the most consequential decision in
this file and is spelled out under `connected_region` below. Every other difference is
mechanical.
"""

import numpy as np

from .geometry import great_circle_distance, points_in_polygon

# merge_contours_f.m parameters, kept at their original values and names in spirit.
MERGE_DISTANCE_DEG = 5.0     # mrg_thr: centers closer than this are the same wave
MIN_EXTENT_DEG = 1.0         # min_thr: a wave must span at least this, north to south
MAX_LAT_EXTENT_DEG = 25.0    # lat_thr: a region wider than this triggers a stricter cut
MAX_LON_EXTENT_DEG = 25.0    # lon_thr

# The original builds six masks at these multiples of the base threshold and steps up
# through them whenever the region it grows is implausibly large.
THRESHOLD_LADDER = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5)


def connected_region(mask, seed):
    """Version 1's region search, replacing isolate_region_f.

    Returns a boolean array of the same shape and dtype as `mask`.

    WHAT COMES BACK depends on the seed, and the below-threshold case is not the obvious
    one. A seed on a True cell gives the 8-connected component containing it. A seed on a
    FALSE cell gives that cell PLUS the above-threshold components touching it, which is
    a single cell only when nothing touches it, because the original marks its seed
    before testing anything. An out-of-bounds seed gives an empty region, since the
    original returns before marking. The block in the body carries the source for that.

    A DELIBERATE DIVERGENCE, and the reason matters more than the change.

    `isolate_region_f.m` is a recursive flood fill that marks each visited cell with 2 and
    calls itself on every newly found neighbour. Its recursive block is wrapped in

        try
          [output2,Zn] = isolate_region_f(X,Y,Zn,output(i,:));
          output = [output;output2];
        catch err
          return
        end

    so when MATLAB's recursion limit is reached the error is swallowed and the function
    returns whatever it had found so far. The default limit is 500 and the walk's depth
    grows with region size in a strongly shape-dependent way, measured at about half
    the cell count for a dense rectangular blob and around a sixth for the real
    components of one worked timestep (880 cells reached depth 140), so a large region
    COULD be SILENTLY TRUNCATED, while whether the limit is actually reached on real
    fields is not established in either direction.

    This port computes the complete region. Reproducing the truncation is not possible in
    any principled way, because what gets returned depends on MATLAB's stack depth and on
    the order the recursion happens to visit cells, neither of which is a property of the
    science. So this is the one place the port deliberately does something the original
    did not, and it is recorded here, in the project plan, and as an item to check when
    version 1 and the port are finally compared: REGIONS MAY LEGITIMATELY DIFFER FOR LARGE
    FEATURES, and a difference there is evidence of the original's truncation rather than
    of a porting error.
    """
    from scipy import ndimage
    mask = np.asarray(mask, dtype=bool)
    row, col = int(seed[0]), int(seed[1])
    if not (0 <= row < mask.shape[0] and 0 <= col < mask.shape[1]):
        # The original returns BEFORE marking anything when the seed is out of bounds,
        # so nothing is visited and the region really is empty here.
        return np.zeros_like(mask)
    # 8-connected, matching the original's eight explicit neighbour tests
    labels, _ = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    if mask[row, col]:
        return labels == labels[row, col]

    # A SEED BELOW THRESHOLD IS STILL MARKED, and the port returned nothing here until
    # 2026-08-30. `isolate_region_f.m` marks the seed with no test that it qualifies,
    #
    #     Zn(pos(1),pos(2)) = 2;
    #     output = pos;
    #
    # and only THEN tests the eight neighbours against the original field, recursing
    # into each one that is above threshold. So the region is the seed cell plus the
    # above-threshold components touching it, which is a single cell only when nothing
    # touches it. Subject to the recursion caveat above: version 1 can return a PARTIAL
    # component for a large fill, so this describes what it sets out to collect rather
    # than a guarantee about what it returns.
    #
    # It is reachable in normal use rather than pathological: `_select_region` seeds
    # every level of the threshold ladder with the same cell, and a region large enough
    # to trigger escalation is often seeded by a cell that does not survive the stricter
    # level. Measured BEFORE this fix, on the 0.75 and the buffered runs alike, that path
    # emptied the region for 15.2 percent of the observations where an axis reached a
    # wave version 1 found and the merge produced none.
    region = np.zeros_like(mask)
    region[row, col] = True
    touching = set()
    for delta_row in (-1, 0, 1):
        for delta_col in (-1, 0, 1):
            if delta_row == 0 and delta_col == 0:
                continue
            r, c = row + delta_row, col + delta_col
            if 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1] and mask[r, c]:
                touching.add(int(labels[r, c]))
    for label in touching:
        region |= labels == label
    return region


def _binary_masks(curvature, threshold):
    """The six binary masks of the threshold ladder.

    The original writes each one as a pair of assignments (zero everything below the
    level, then set everything still at or above the base threshold to one), which
    reduces to a single comparison because the survivors are all above the base.
    """
    curvature = np.asarray(curvature, dtype=float)
    return [np.nan_to_num(curvature, nan=-np.inf) >= threshold * step
            for step in THRESHOLD_LADDER]


def _extent(values):
    if values.size == 0:
        return 0.0
    return float(np.abs(np.max(values) - np.min(values)))


def _select_region(masks, seed, latgrid, longrid):
    """Grow the region at each threshold and step up while it is implausibly large.

    FAITHFUL, and the original's phrasing of this is worth preserving. Each escalation
    test examines the region at a FIXED level rather than the currently selected one:

        if extent(level 1) too large -> take level 2
        if extent(level 2) too large -> take level 3
        ...

    Because a stricter threshold can only shrink the region, the levels are nested and
    this cascade lands on the first level that is small enough. It reads as though it
    might skip a level and it cannot.
    """
    regions = [connected_region(m, seed) for m in masks]
    chosen = regions[0]
    for level in range(len(regions) - 1):
        lats = latgrid[regions[level]]
        lons = longrid[regions[level]]
        if (_extent(lats) >= MAX_LAT_EXTENT_DEG
                or _extent(lons) >= MAX_LON_EXTENT_DEG):
            chosen = regions[level + 1]
    return chosen


def _hull_contains(lons, lats, points_lon, points_lat):
    """Which of the given points fall inside the convex hull of a region.

    The original calls `convhull` inside a try block and does nothing on failure, which
    is what happens for a degenerate region (fewer than three points, or all collinear).
    Reproduced: a hull that cannot be formed absorbs nothing.

    The containment test is `geometry.points_in_polygon` rather than a library one, for the
    same reason the association stage uses it: merge_contours_f.m calls MATLAB `inpolygon`,
    which counts a point ON the hull boundary as inside, and the points being tested are
    other candidates' centers on the same grid the hull vertices came from.
    """
    if lons.size <= 2:
        return np.zeros(len(points_lon), dtype=bool)
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(np.column_stack([lons, lats]))
        return points_in_polygon(lons[hull.vertices], lats[hull.vertices],
                                 points_lon, points_lat)
    except Exception:                                   # noqa: BLE001
        # degenerate geometry: the original's catch block leaves `check` unset
        return np.zeros(len(points_lon), dtype=bool)


def merge_contours(candidates, latgrid, longrid, curvature, threshold, absorb=False):
    """Merge trough candidates belonging to one wave, from merge_contours_f.m.

    THE FIRST PASS ABSORBS NOTHING BY DEFAULT, BECAUSE VERSION 1'S DOES NOT. Its
    absorption test is

        if inpolygon(pot_wv(idd(j)).lon_mean,pot_wv(idd(j)).lat_mean,tlon(k),tlat(k),
                     'simplify',true) == 1;

    and `inpolygon` takes four arguments in both MATLAB and Octave. The two extra ones
    raise, the whole block sits inside a `try ... catch err ... end` whose catch is
    empty, and `check` is therefore never set. Every candidate becomes its own wave and
    the count is brought down entirely by the five-degree pass and the minimum extent.

    ESTABLISHED BY EXECUTION, not by reading. Version 1's own merge_contours_f was run
    under Octave on six real timesteps of the buffered 1990 record: convhull succeeded
    417 times, the absorption loop ran to completion 6 times, and the catch was reached
    411 times with "inpolygon: function called with too many inputs". The 6 completions
    are the iterations where no other candidate remained to test. Against the same
    arrays the archived source returns 34, 28, 28, 29, 25 and 29 waves while the source
    with the call repaired returns 33, 28, 27, 29, 25 and 29.

    `absorb=True` performs the absorption the code was written to do. It is not version
    1 and its record is not version 1's, which is the same shape as `exclusive=True` in
    the association stage: the defect is reproduced by default and the repair is a flag.

    Parameters
    ----------
    candidates : sequence of dict
        Each needs `lat_mean`, `lon_mean` and `time`.
    latgrid, longrid : 2-D coordinate meshes
    curvature : 2-D curvature vorticity anomaly on that mesh
    threshold : float
        The base trough threshold. The ladder multiplies it.

    Returns
    -------
    list of dict, each with `time`, `lat_mean`, `lon_mean`, `region` (a boolean mask),
    `lat_wave` and `lon_wave`.
    """
    latgrid = np.asarray(latgrid, dtype=float)
    longrid = np.asarray(longrid, dtype=float)
    curvature = np.asarray(curvature, dtype=float)
    if latgrid.shape != longrid.shape or latgrid.shape != curvature.shape:
        raise ValueError("mesh and curvature must share a shape, got %s, %s and %s"
                         % (latgrid.shape, longrid.shape, curvature.shape))
    candidates = list(candidates)
    if not candidates:
        return []

    masks = _binary_masks(curvature, threshold)
    if not masks[0].any():
        return []

    base_rows, base_cols = np.nonzero(masks[0])
    base_lat = latgrid[base_rows, base_cols]
    base_lon = longrid[base_rows, base_cols]

    # --- pass one: grow a region per candidate and absorb those inside its hull -------
    remaining = list(range(len(candidates)))
    merged = []
    while remaining:
        first = remaining[0]
        others = remaining[1:]
        here = candidates[first]

        # nearest above-threshold grid point, in squared degrees as the original
        d2 = ((here["lat_mean"] - base_lat) ** 2 + (here["lon_mean"] - base_lon) ** 2)
        nearest = int(np.argmin(d2))
        seed = (base_rows[nearest], base_cols[nearest])

        region = _select_region(masks, seed, latgrid, longrid)
        lats, lons = latgrid[region], longrid[region]

        absorbed = []
        if absorb and others:
            # Only reachable with the flag set. Version 1 computes the hull and then
            # fails before it can use it, so this is what its code describes rather than
            # what it does.
            inside = _hull_contains(lons, lats,
                                    np.array([candidates[i]["lon_mean"] for i in others]),
                                    np.array([candidates[i]["lat_mean"] for i in others]))
            absorbed = [others[i] for i in np.flatnonzero(inside)]

        # FAITHFUL: when candidates are absorbed the original takes the TIME of the first
        # absorbed one rather than of the candidate being processed. With absorption
        # disabled nothing is ever absorbed, so a wave always carries its own candidate's
        # time and this branch is unreachable.
        time_source = candidates[absorbed[0]]["time"] if absorbed else here["time"]
        merged.append({
            "time": time_source,
            "lat_mean": float(np.median(lats)) if lats.size else here["lat_mean"],
            "lon_mean": float(np.median(lons)) if lons.size else here["lon_mean"],
            "region": region,
            "lat_wave": lats,
            "lon_wave": lons,
        })
        drop = set(absorbed) | {first}
        remaining = [i for i in remaining if i not in drop]

    # FAITHFUL, and surprising enough to be worth naming. The original returns here when
    # pass one left one wave or fewer:
    #
    #     %Second Check: Remove all duplicate waves
    #     if size(pot_wv2,2) <= 1;
    #       output = pot_wv2;
    #       return
    #     end
    #
    # so a LONE WAVE SKIPS BOTH REMAINING PASSES. It never has its center refined toward
    # the peak anomaly, and it is never tested against the minimum extent, which means a
    # single one-row streak too short to be a wave is published as one. Reproduced. A
    # test pins this, because tidying it away is the obvious cleanup and would put the
    # port out of agreement with version 1.
    if len(merged) <= 1:
        return merged

    # --- pass two: drop centers closer together than the merge distance ---------------
    kept = []
    pending = list(range(len(merged)))
    while pending:
        first = pending[0]
        others = pending[1:]
        kept.append(merged[first])
        if others:
            distance_deg = great_circle_distance(
                merged[first]["lat_mean"], merged[first]["lon_mean"],
                np.array([merged[i]["lat_mean"] for i in others]),
                np.array([merged[i]["lon_mean"] for i in others]), "nm") / 60.0
            close = {others[i] for i in np.flatnonzero(distance_deg <= MERGE_DISTANCE_DEG)}
        else:
            close = set()
        pending = [i for i in pending if i != first and i not in close]

    # --- pass three: reject waves too short north to south, and refine the center -----
    out = []
    for wave in kept:
        lats, lons = wave["lat_wave"], wave["lon_wave"]
        if lats.size == 0:
            continue
        values = curvature[wave["region"]]
        peak = values == np.nanmax(values)
        if peak.any():
            wave = dict(wave)
            wave["lat_mean"] = (float(np.mean(lats[peak])) + wave["lat_mean"]) / 2.0
            wave["lon_mean"] = (float(np.mean(lons[peak])) + wave["lon_mean"]) / 2.0

        top = lats == np.max(lats)
        bottom = lats == np.min(lats)
        top_lon = (np.min(lons[top]) + np.max(lons[top])) / 2.0
        bottom_lon = (np.min(lons[bottom]) + np.max(lons[bottom])) / 2.0
        span_deg = great_circle_distance(np.min(lats), bottom_lon,
                                         np.max(lats), top_lon, "nm") / 60.0
        if span_deg >= MIN_EXTENT_DEG:
            out.append(wave)
    return out
