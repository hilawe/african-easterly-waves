#!/usr/bin/env python3
"""Export real merge inputs for version 1's own MATLAB to run on, under Octave.

THIS IS THE ORACLE THE PROJECT HAS NEVER HAD. Every check so far compares the port
against somebody's reading of the archived source. This hands version 1's own
merge_contours_f.m the exact arrays the port's merge sees at a real timestep and records
what it returns, so the two can be compared without a reading in between.

WHAT IS EXPORTED, per timestep: the coarse coordinate meshes, the MASKED curvature the
merge is given, the threshold, and the candidate positions from the trough axes. Also
the port's own merged output, so the comparison needs nothing recomputed on either side.

ARGUMENT ORDER IS A TRAP AND IS HANDLED IN THE OCTAVE SCRIPT, NOT HERE.
merge_contours_f takes (pot_wv, longrid, latgrid, cRVt, thr), longitude BEFORE latitude,
where the port's merge_contours takes latitude first.
"""
import os
import sys

import numpy as np
from scipy.io import savemat

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "src"))

from aew.v1port import climatology as clim
from aew.v1port import load as L
from aew.v1port import pipeline as P
from aew.v1port.contours import merge_contours
from aew.v1port.detection import MAX_ZONAL_WIND, _prepare, trough_axes

# Where the exchange files live. Kept OUT of the repository by default, because they
# are multi-megabyte arrays and are regenerable from this script.
S = os.environ.get("AEW_ORACLE_DIR", os.path.join(os.path.sep, "tmp", "aew-oracle"))
os.makedirs(S, exist_ok=True)
D = "data/eraint/v1port_buffered"
# Timesteps across the season, including 32886.5, which the merge trace singled out
# as a case where a wave version 1 keeps is absorbed in the coarse pass, and 33030.5,
# which is pair 63's one detection difference. At 33030.5 the port's pass 2 drops the
# feature as a duplicate of a one-point wave 4.38 degrees away, and pass 2 is ORDER
# DEPENDENT, so whether version 1 makes the same choice is what this export answers.
# See scripts/trace_pair63_merge.py for the port-side trace of that rejection.
TARGETS = [32886.5, 33030.5, 33067.5, 33067.75, 33118.5, 33157.0, 33234.75]

with np.load(os.environ.get("AEW_CLIMO_CACHE",
                            os.path.join(S, "climo_buffered.npz")),
             allow_pickle=False) as z:
    climo = {"keys": [tuple(int(x) for x in r) for r in z["keys"]], "mean": z["mean"],
             "counts": z["counts"], "short_steps": []}
times, latgrid, longrid, u, v = L.load_year(1990, D, "eraint")
curvature = P.curvature_from_winds(latgrid, longrid, u, v)
anomaly = P.anomaly_from_climatology(curvature, times, climo)
del curvature
advection = P._advection(latgrid, longrid, u, v, anomaly)
native = abs(float(latgrid[1, 0] - latgrid[0, 0]))
C = {n: clim.gaussian_decimate(f, native, P.COARSE_RESOLUTION_DEG)
     for n, f in (("u", u), ("anomaly", anomaly), ("advection", advection))}
del advection
lat_c, lon_c = P.coarse_grid(latgrid, longrid, native, P.COARSE_RESOLUTION_DEG)
rows_c, cols_c = P._subset(lat_c, lon_c, P.DOMAIN_LAT, P.DOMAIN_LON)
LG, LO = lat_c[np.ix_(rows_c, cols_c)], lon_c[np.ix_(rows_c, cols_c)]
lat_values = LG[:, 0]
coarse_threshold, _ = P.thresholds_for("ERA-Int", 700)

cases = {}
for target in TARGETS:
    step = int(np.argmin(np.abs(times - target)))
    if abs(float(times[step]) - target) > 1e-6:
        print(f"  no timestep at {target}, skipped")
        continue
    wind = clim.smooth9(C["u"][step][np.ix_(rows_c, cols_c)])
    field = _prepare(C["anomaly"][step][np.ix_(rows_c, cols_c)], lat_values)
    advection_c = clim.smooth9(C["advection"][step][np.ix_(rows_c, cols_c)])
    westerly = wind > MAX_ZONAL_WIND
    advection_c = np.where(westerly, np.nan, advection_c)
    field = np.where(westerly, np.nan, field)
    weak = field < coarse_threshold
    advection_c = np.where(weak, np.nan, advection_c)
    field = np.where(weak, np.nan, field)

    axes = trough_axes(LG, LO, advection_c)
    if not axes:
        continue
    candidates = [{"time": float(times[step]), "lat_mean": float(np.mean(a)),
                   "lon_mean": float(np.mean(b))} for a, b in axes]
    port = merge_contours(candidates, LG, LO, field, coarse_threshold)
    key = f"t{str(target).replace('.', 'p')}"
    cases[key] = {
        "time": float(times[step]),
        "latgrid": LG, "longrid": LO,
        # NaN is deliberate and is what the merge is handed. MATLAB's `find(cRVt < thr)`
        # and `find(cRVt >= thr)` are both FALSE at NaN, so a NaN cell stays NaN in the
        # binary field and `Z == 1` never matches it, which is the same as the port's
        # nan_to_num(-inf) comparison. Carrying it through rather than filling keeps the
        # two sides handed identical input.
        "cRVt": field,
        "thr": float(coarse_threshold),
        "cand_lat": np.array([c["lat_mean"] for c in candidates], dtype=float),
        "cand_lon": np.array([c["lon_mean"] for c in candidates], dtype=float),
        "cand_time": np.array([c["time"] for c in candidates], dtype=float),
        "port_lat": np.array([w["lat_mean"] for w in port], dtype=float),
        "port_lon": np.array([w["lon_mean"] for w in port], dtype=float),
        "port_cells": np.array([int(w["region"].sum()) for w in port], dtype=float),
    }
    print(f"  {target}: {len(candidates)} candidates -> {len(port)} port waves",
          flush=True)

out = os.path.join(S, "merge_cases.mat")
savemat(out, cases, do_compression=True)
print(f"\nwrote {len(cases)} cases to {out}")
print(f"grid {LG.shape}, threshold {coarse_threshold:.4e}")
