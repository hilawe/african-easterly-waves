#!/usr/bin/env python3
"""How deep version 1's region search actually recurses, over a real window.

WHY THIS EXISTS. `isolate_region_f.m` is a recursive flood fill whose recursive call sits
in a `try ... catch err; return; end`, so reaching MATLAB's recursion limit truncates the
region SILENTLY and returns whatever was found so far. Whether real fields reach that
limit has been recorded as an open divergence since the port began, and a six-timestep
sample found a maximum depth of 140, which was written up as the truncation being
"measured ABSENT".

Six timesteps is not the record, and a later run supplied direct evidence that deeper
regions exist: the whole-tracker oracle run stopped inside timestep 41 of a 60-step window
by exhausting a 544 KB interpreter stack. This measures the depth over a real window
instead.

THIS IS A PYTHON REIMPLEMENTATION OF THE WALK, AND IT IS VALIDATED AGAINST THE ORIGINAL.
It counts recursion depth, not stack bytes, and reproducing a recursion order in another
language is an assumption until it is checked. It was checked:
`scripts/export_depth_cases.py` exports real masks and seeds and
`scripts/octave/depth_check.m` runs the ARCHIVED isolate_region_f on them under Octave,
recording its actual `numel(dbstack)`. On twelve cases spanning depths of 1 to 246 the two
agree EXACTLY, twelve of twelve. What that validates is the depth number. It says nothing
about how many bytes of native stack a frame costs, which is a separate question answered
separately by running the real function until it dies.

THE SIMULATION IS OF THE MATLAB, not of the port. The port's `connected_region` is iterative and
deliberately does not truncate, so it cannot answer this. The walk below reproduces
`isolate_region_f`'s own order, including the two details that change the depth:

  - neighbours are TESTED against the caller's `Z` but MARKED in `Zn`, and the child is
    handed `Zn`, so a cell already marked by a sibling is not revisited by a later child
  - all eight neighbours are appended before any recursion happens, in the source's own
    order, and the recursion then runs over them in that order
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from aew.v1port import climatology as clim  # noqa: E402
from aew.v1port import load as L  # noqa: E402
from aew.v1port import pipeline as P  # noqa: E402
from aew.v1port.climo_cache import load_or_build  # noqa: E402
from aew.v1port.detection import MAX_ZONAL_WIND, _prepare, trough_axes  # noqa: E402

# the source's own neighbour order, (drow, dcol)
NEIGHBOURS = ((-1, 0), (0, -1), (-1, -1), (1, 0), (0, 1), (1, 1), (-1, 1), (1, -1))


def walk_depth(mask, seed):
    """The deepest recursion `isolate_region_f` would reach from `seed`.

    An explicit stack rather than Python recursion, so the measurement is not itself
    limited by the interpreter it runs in. `marked` is the threaded `Zn`: a cell marked by
    any earlier call is invisible to later ones, which is what keeps the walk finite.
    """
    rows, cols = mask.shape
    r0, c0 = seed
    if not (0 <= r0 < rows and 0 <= c0 < cols):
        return 0
    marked = np.zeros(mask.shape, dtype=bool)

    def children(r, c):
        """Neighbours this call would append, marking them as it goes."""
        found = []
        for dr, dc in NEIGHBOURS:
            rr, cc = r + dr, c + dc
            if not (0 <= rr < rows and 0 <= cc < cols):
                continue
            if mask[rr, cc] and not marked[rr, cc]:
                marked[rr, cc] = True
                found.append((rr, cc))
        return found

    marked[r0, c0] = True
    deepest = 1
    # each frame is (its children, the index of the next one to descend into)
    stack = [[children(r0, c0), 0]]
    while stack:
        frame = stack[-1]
        if frame[1] >= len(frame[0]):
            stack.pop()
            continue
        r, c = frame[0][frame[1]]
        frame[1] += 1
        stack.append([children(r, c), 0])
        deepest = max(deepest, len(stack))
    return deepest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--year", type=int, default=1990)
    ap.add_argument("--start", type=int, default=600)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--directory", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "eraint", "v1port_buffered"))
    args = ap.parse_args(argv)

    # BOTH MERGE PASSES, because find_ews_f.m calls merge_contours_f twice: once on the
    # coarse grid at line 158 and again on the FINE grid at line 167, with the fine
    # threshold. An earlier version of this measured only the coarse pass, reported a
    # maximum depth of 114, and concluded the recursion could not be what exhausted the
    # interpreter stack. The fine grid is 71 by 181 against the coarse 35 by 90, so its
    # regions are the ones that can actually get deep, and leaving it out measured the
    # easier half of the question.
    climo = load_or_build(L.available_years(args.directory, "eraint"),
                          args.directory, "eraint", os.environ["AEW_CLIMO_CACHE"])
    times, latgrid, longrid, u, v = L.load_year(args.year, args.directory, "eraint")
    curvature = P.curvature_from_winds(latgrid, longrid, u, v)
    anomaly = P.anomaly_from_climatology(curvature, times, climo)
    del curvature
    advection = P._advection(latgrid, longrid, u, v, anomaly)
    native = abs(float(latgrid[1, 0] - latgrid[0, 0]))
    coarse = {n: clim.gaussian_decimate(f, native, P.COARSE_RESOLUTION_DEG)
              for n, f in (("u", u), ("anomaly", anomaly), ("advection", advection))}
    del advection
    lat_c, lon_c = P.coarse_grid(latgrid, longrid, native, P.COARSE_RESOLUTION_DEG)
    rows_c, cols_c = P._subset(lat_c, lon_c, P.DOMAIN_LAT, P.DOMAIN_LON)
    rows_f, cols_f = P._subset(latgrid, longrid, P.DOMAIN_LAT, P.DOMAIN_LON)
    LG, LO = lat_c[np.ix_(rows_c, cols_c)], lon_c[np.ix_(rows_c, cols_c)]
    FG, FO = latgrid[np.ix_(rows_f, cols_f)], longrid[np.ix_(rows_f, cols_f)]
    coarse_threshold, fine_threshold = P.thresholds_for("ERA-Int", 700)
    print(f"coarse grid {LG.shape}, fine grid {FG.shape}", flush=True)

    MATLAB_LIMIT = 500
    # THE GUI BINARY'S CEILING FOR THIS FUNCTION, BRACKETED BY MEASUREMENT rather than
    # taken from a probe. Running the archived isolate_region_f itself on real masks under
    # `octave --no-gui`, a walk of depth 61 completes and the next case, 173, kills the
    # process. An earlier version of this quoted 190, which was a ceiling measured on a
    # small probe FUNCTION whose frames are smaller than isolate_region_f's, so it
    # overstated how deep the real walk can go. Both bounds are reported because the gap
    # between them has not been narrowed.
    GUI_LOW, GUI_HIGH = 61, 173
    depths = {"coarse": [], "fine": []}
    for step in range(args.start, args.start + args.steps):
        wind = clim.smooth9(coarse["u"][step][np.ix_(rows_c, cols_c)])
        field = _prepare(coarse["anomaly"][step][np.ix_(rows_c, cols_c)], LG[:, 0])
        adv = clim.smooth9(coarse["advection"][step][np.ix_(rows_c, cols_c)])
        westerly = wind > MAX_ZONAL_WIND
        adv = np.where(westerly, np.nan, adv)
        field = np.where(westerly, np.nan, field)
        weak = field < coarse_threshold
        adv = np.where(weak, np.nan, adv)
        field = np.where(weak, np.nan, field)
        mask = np.nan_to_num(field, nan=-np.inf) >= coarse_threshold

        # the fine field the SECOND merge pass is handed, gated by the fine threshold
        # SMOOTHED FIRST, then sign-adjusted, which is find_ews_f.m's own order (it
        # smooths the fine curvature at line 91 before the fine merge at line 167). An
        # earlier version of this line computed the smoothed field and then overwrote it
        # with the unsmoothed one, so the fine depths were measured on a rougher field
        # than version 1 ever grows regions on.
        fine = _prepare(clim.smooth9(anomaly[step][np.ix_(rows_f, cols_f)]), FG[:, 0])
        fine_mask = np.nan_to_num(fine, nan=-np.inf) >= fine_threshold

        centres = []
        for axis_lat, axis_lon in trough_axes(LG, LO, adv):
            centres.append((float(np.mean(axis_lat)), float(np.mean(axis_lon))))
        step_depth = {"coarse": 0, "fine": 0}
        for lat_mean, lon_mean in centres:
            r = int(np.argmin(np.abs(LG[:, 0] - lat_mean)))
            c = int(np.argmin(np.abs(LO[0, :] - lon_mean)))
            d = walk_depth(mask, (r, c))
            depths["coarse"].append(d)
            step_depth["coarse"] = max(step_depth["coarse"], d)
            rf = int(np.argmin(np.abs(FG[:, 0] - lat_mean)))
            cf = int(np.argmin(np.abs(FO[0, :] - lon_mean)))
            df = walk_depth(fine_mask, (rf, cf))
            depths["fine"].append(df)
            step_depth["fine"] = max(step_depth["fine"], df)
        print(f"  step {step} ({float(times[step]):.2f}): deepest coarse "
              f"{step_depth['coarse']:4d}, deepest FINE {step_depth['fine']:5d}",
              flush=True)

    print(f"\nOVER {args.steps} TIMESTEPS, "
          f"{len(depths['coarse'])} candidate regions grown on each grid:\n")
    print(f"  {'':>10} {'deepest':>8} {'median':>7} {'90th':>6} "
          f"{'past ' + str(GUI_LOW):>8} {'past ' + str(GUI_HIGH):>9} "
          f"{'past ' + str(MATLAB_LIMIT):>9}")
    for which in ("coarse", "fine"):
        d = np.array(depths[which])
        print(f"  {which:>10} {d.max():8d} {np.median(d):7.0f} "
              f"{np.percentile(d, 90):6.0f} {int((d > GUI_LOW).sum()):8d} "
              f"{int((d > GUI_HIGH).sum()):9d} {int((d > MATLAB_LIMIT).sum()):9d}")
    both = np.array(depths["coarse"] + depths["fine"])
    print(f"\nTRUNCATION UNDER MATLAB'S OWN LIMIT OF {MATLAB_LIMIT} "
          f"{'OCCURS' if (both > MATLAB_LIMIT).any() else 'DOES NOT OCCUR'} "
          f"on this window.")
    print(f"THE GUI BINARY'S CEILING for this function is between {GUI_LOW} and "
          f"{GUI_HIGH}, measured by running the archived isolate_region_f on real masks "
          f"under `octave --no-gui`.")
    print(f"Regions past the LOW bound: {int((both > GUI_LOW).sum())} of {both.size}. "
          f"Past the HIGH bound: {int((both > GUI_HIGH).sum())}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
