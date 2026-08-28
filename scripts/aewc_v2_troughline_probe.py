"""Trough-line (enstools-feature) extended-domain probe, season 2004.

The trough-line counterpart of the QTrack probe: the Fischer et al. (2024)
AEW identification and tracking pipeline runs end to end on local ERA5
700 hPa winds, twice, differing only in the eastern data boundary (the
published 45 E against an extension to 60 E), plus a comparison of the
control against the published wts2004 archive.

APPROXIMATION, stated first. The method thresholds curvature-vorticity
anomalies against an hour-of-year climatology; the published record uses a
1979-2019 (41 calendar years) climatology on 0.5-degree input. Neither is
available locally, so this probe builds a FIVE-YEAR (2000-2004) climatology
on the native 1.5-degree grid, shared by both runs. Under that approximation
the local control OVER-DETECTS relative to the published archive, so the
reproduction comparison bounds feasibility only.

BOUNDARY CORRECTNESS (an external review's finding, folded). The method as
published computes curvature vorticity AFTER cropping to its data domain,
with a cyclic wrap that is only valid on a full longitude band, so two runs
with different eastern boundaries would differ in their wrap seam and not
only in extent, and the published archive itself carries that seam at its
own edges. This probe therefore computes curvature vorticity ONCE on the
full global longitude band, where the wrap is valid, and hands the same
field to both runs; the remaining per-run steps (smoothing, anomaly,
advection) still act on each run's crop, so detection can differ within a
few cells of the control's eastern edge. The probe ASSERTS exact equality of
the two runs' detection masks over the shared domain west of the edge
margin, and pins the count of differing cells inside the margin.

Environment patches, all to the scratch clone or the venv and none to this
repository's own code, each pinned by hash before the run (the clone's
working tree is additionally asserted clean apart from the two patched
files, and the venv shims and cached derived inputs are hash-pinned too):
  1. feature/__init__.py: distutils.spawn replaced by shutil.which (py3.12).
  2. feature/identification/identification.py: the map_blocks index field is
     complex-encoded and chunked like the dataset (modern xarray asserts
     chunk compatibility of args; the object-dtype tuple field predates
     that).
  3. A minimal enstools shim (misc, io) in the venv, since the framework
     imports only four small utilities from the real enstools.
The clone is pinned to commit a33236f (2024-09-18).

Run from the repo root:  .venv/bin/python scripts/aewc_v2_troughline_probe.py
  --mutate=noext   the "east" run uses the published 45 E boundary too; the
                   eastern-delivery checks must fail and the mutated run must
                   equal the control track for track.
Wall clock about 4 minutes when the climatology and input files exist
(they are built on first run; add about 10 s).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "data" / "aewc_v2_pilot" / "troughline_run"
CLONE = ROOT / "data" / "aewc_v2_pilot" / "enstools-feature"

INPUT_SHA256 = {
    "data/era5/era5_u700_2000-2004_6h_global.nc":
        "819095f9f6df44778f4ebad59f2b3fdf19087f4e4a63994f1fbadb6dc36bbc7b",
    "data/era5/era5_v700_2000-2004_6h_global.nc":
        "d4fff9798b2effb2cb294af988f071f7a38dfa3bdc693f8c049aa9fc56fe0057",
    "data/aewc_v2_pilot/wts2004.json":
        "31beef8a888d06766593a742ac83c50b8882a11927c62091660d81b7c56f2702",
}

CLONE_COMMIT = "a33236fccf190ad8142fdd2a6ee9504f03741b63"
CLONE_ALLOWED_MODIFIED = {"feature/__init__.py",
                          "feature/identification/identification.py"}
SHIM_SHA256 = {
    ".venv/lib/python3.12/site-packages/enstools/misc.py": "5e2b399b22588c61bafc2e501085e13373aab7b3f8a3f17abfd9a21e6bd00ac7",
    ".venv/lib/python3.12/site-packages/enstools/io.py": "1efafdaa8fe1260b64f32cdab9126c8555db3c5b37101fd002c236eecbfb11ae",
}
DERIVED_SHA256 = {
    "cv_clim_local.nc": "bb9a9aa8f56048a46fd0b4ddd443378fbcb6b73f95cb49129eed98a5e4cc1df7",
    "uv2004.nc": "3c81c58dddf7f7c2f90652d867d6f2061d1daa062e84f21cbecf3cbf69549080",
}
PATCHED_FILES_SHA256 = {
    "feature/__init__.py": "2495eb86b6f6033536a22664d7b54fd1ccbc1f7880238bdfddc5b1a3bcc370ff",
    "feature/identification/identification.py": "e5ffa5ccd13f4d554a7ae95aa5a77f080e2c3015b59f1c56d15d27a7d87e580e",
}

EDGE_MARGIN_DEG = 6.0   # smoothing (2 cells) + two derivative passes; the
                        # equality assertion runs west of 45 - 6 = 39 E
EXPECTED = {
    "published": {"tracks": 66, "east_any": 0, "east_first": 0,
                  "max_track_lon": (31.2, 0.1), "obj_east": 16,
                  "max_vertex_lon": (45.0, 1e-6)},
    "control": {"tracks": 113, "east_any": 4, "east_first": 4,
                "max_track_lon": (42.4, 0.1), "obj_east": 86,
                "max_vertex_lon": (45.0, 1e-6)},
    "east": {"tracks": 116, "east_any": 7, "east_first": 7,
             "max_track_lon": (56.6, 0.1), "obj_east": 364,
             "max_vertex_lon": (60.0, 1e-6)},
    "control_vs_published_matches": 47,
    "east_starters": 7,
    "east_starters_west_of_20E": 1,
    "east_starters_matching_published": 1,
    "crosser_min_lon": (-18.26, 0.05),
    "mask_margin_diff_cells": 4796,
    "mask_margin_total_cells": 44064,
}

MUTATION = None
for a in sys.argv[1:]:
    if a.startswith("--mutate="):
        MUTATION = a.split("=", 1)[1]
if MUTATION is not None and MUTATION != "noext":
    sys.exit(f"unknown mutation {MUTATION!r}; the only defined one is noext")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _bind_environment():
    import subprocess
    head = subprocess.run(["git", "-C", str(CLONE), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    assert head == CLONE_COMMIT, f"clone at {head[:12]}, expected {CLONE_COMMIT[:12]}"
    status = subprocess.run(["git", "-C", str(CLONE), "status", "--porcelain"],
                            capture_output=True, text=True).stdout
    modified = {line[3:].strip() for line in status.splitlines() if line.strip()}
    assert modified <= CLONE_ALLOWED_MODIFIED, (
        f"clone tree carries unreviewed modifications: "
        f"{sorted(modified - CLONE_ALLOWED_MODIFIED)}")
    for rel, want in PATCHED_FILES_SHA256.items():
        got = _sha256(CLONE / rel)
        assert got == want, (f"{rel}: sha256 {got[:12]}... != pinned; the "
                             f"environment patches this probe was reviewed "
                             f"under are not in place")
    for rel, want in SHIM_SHA256.items():
        got = _sha256(ROOT / rel)
        assert got == want, f"shim {rel}: sha256 {got[:12]}... != pinned"
    for rel, want in INPUT_SHA256.items():
        got = _sha256(ROOT / rel)
        assert got == want, f"{rel}: sha256 {got[:12]}... != pinned"


def _build_inputs():
    """Climatology and season input, built once, then hash-pinned.

    Curvature vorticity is computed ONCE on the full global longitude band,
    where the method's cyclic wrap is valid, and both the climatology and the
    season input carry that field, so the two tracking runs share identical
    curvature vorticity and differ only in their crop. The climatology's
    quantile population is longitudes -100 to 60, latitudes 4 to 29.5 (the
    local files' band), 2000-2004, stated in the findings note.
    """
    import xarray as xr
    import metpy.calc as mpcalc
    sys.path.insert(0, str(CLONE))
    from feature.identification.african_easterly_waves.processing import compute_cv

    if not ((RUN / "cv_clim_local.nc").exists() and (RUN / "uv2004.nc").exists()):
        RUN.mkdir(parents=True, exist_ok=True)
        u = xr.open_dataset(ROOT / "data/era5/era5_u700_2000-2004_6h_global.nc")
        v = xr.open_dataset(ROOT / "data/era5/era5_v700_2000-2004_6h_global.nc")
        ds = xr.Dataset({"u": u["u700"].squeeze("pressure_level", drop=True),
                         "v": v["v700"].squeeze("pressure_level", drop=True)})
        ds = ds.sortby("latitude").sel(latitude=slice(3, 29.5))
        ds = compute_cv(ds, "u", "v", "cv")   # global longitudes: valid wrap
        dsr = ds.sel(longitude=slice(-100, 60))
        cv = mpcalc.smooth_n_point(dsr["cv"], n=9, passes=2).metpy.dequantify()
        cv = cv.assign_coords(hourofyear=cv.time.dt.strftime("%m-%d %H"))
        clim = cv.groupby("hourofyear").mean("time")
        anom = cv.groupby("hourofyear") - clim
        q_hoy = anom.groupby("hourofyear").quantile(0.66, dim=...)
        out = xr.Dataset()
        out["cv"] = clim.rename({"latitude": "lat", "longitude": "lon"}) \
                        .expand_dims(plev=[70000.0])
        out["cva_quantile_hoy"] = q_hoy.expand_dims(plev=[70000.0])
        out["cva_quantile_full"] = anom.quantile(0.66, dim=...) \
                                       .expand_dims(plev=[70000.0])
        out.to_netcdf(RUN / "cv_clim_local.nc")
        season = ds[["u", "v", "cv"]].sel(time=slice("2004-05-25", "2004-11-05"),
                                          longitude=slice(-100, 60))
        # give every variable an explicit 700 hPa level dim so precompute's
        # level handling treats the pre-supplied cv like u and v
        season = season.expand_dims(level=[700]).transpose(
            "level", "time", "latitude", "longitude")
        season.to_netcdf(RUN / "uv2004.nc")
    for name, want in DERIVED_SHA256.items():
        got = _sha256(RUN / name)
        assert got == want, (f"derived input {name}: sha256 {got[:12]}... != "
                             f"pinned; delete it to rebuild from the pinned "
                             f"raw inputs")


def _run_pipeline(mode):
    """One identification+tracking run; returns the JSON path."""
    sys.path.insert(0, str(CLONE))
    import feature.identification.african_easterly_waves.configuration as cfg
    cfg.aew_clim_dir = str(RUN / "cv_clim_local.nc")
    cfg.data_lat = (3, 29.5)
    east_e = 45 if MUTATION == "noext" else 60
    if mode == "control":
        cfg.data_lon = (-100, 45)
        cfg.wave_filter_lon = (-110, 55)
    else:
        cfg.data_lon = (-100, east_e)
        cfg.wave_filter_lon = (-110, east_e + 5)

    from feature.pipeline import FeaturePipeline
    from feature.identification.african_easterly_waves import AEWIdentification
    from feature.tracking.african_easterly_waves import AEWTracking
    from feature.identification._proto_gen import african_easterly_waves_pb2
    from feature.identification.african_easterly_waves.processing import interpolate_wts

    pipeline = FeaturePipeline(african_easterly_waves_pb2,
                               processing_mode="2d", workers=1)
    pipeline.set_data_path(str(RUN / "uv2004.nc"))
    i_strat = AEWIdentification(wt_out_file=False, cv="cv", year_summer=2004)
    t_strat = AEWTracking()
    t_strat.set_threads(8)
    pipeline.set_identification_strategy(i_strat)
    pipeline.set_tracking_strategy(t_strat, generate_tracks=True)
    pipeline.execute()
    ob = pipeline.get_feature_desc()
    interpolate_wts(ob, african_easterly_waves_pb2)
    suffix = "_mut_noext" if MUTATION else ""
    out = RUN / f"wts2004_{mode}{suffix}.json"
    pipeline.save_result(description_type="json", description_path=str(out))
    return out


def _node_obj(n):
    # the current proto serializes the field as 'feature'; the published
    # archive's generation called it 'object'
    return n.get("object") or n["feature"]


def load_wts(path):
    d = json.loads(Path(path).read_text())
    s = d["sets"][0]
    tracks = []
    for trk in s["tracks"]:
        seen = {}
        for e in trk["edges"]:
            for n in [e["parent"]] + list(e.get("children", [])):
                o = _node_obj(n)
                key = (n["time"], o["id"])
                if key in seen:
                    continue
                pts = o["properties"]["linePts"]
                seen[key] = (float(np.mean([p["lat"] for p in pts])),
                             float(np.mean([p["lon"] for p in pts])))
        by = defaultdict(list)
        for (t, _), c in seen.items():
            by[t].append(c)
        ser = sorted((np.datetime64(t),
                      float(np.mean([c[0] for c in v])),
                      float(np.mean([c[1] for c in v])))
                     for t, v in by.items())
        if len(ser) >= 4:
            tracks.append((np.array([x[0] for x in ser]),
                           np.array([x[1] for x in ser]),
                           np.array([x[2] for x in ser])))
    obj_e = 0
    max_vtx = -180.0
    for ts in s["timesteps"]:
        for o in (ts.get("objects") or ts.get("features") or []):
            mx = max(p["lon"] for p in o["properties"]["linePts"])
            max_vtx = max(max_vtx, mx)
            if mx > 40:
                obj_e += 1
    return tracks, obj_e, max_vtx


def east_stats(tracks):
    any_e = first_e = 0
    mx = -180.0
    for t, la, lo in tracks:
        mx = max(mx, float(lo.max()))
        if (lo > 40).any():
            any_e += 1
        if lo[0] > 40:
            first_e += 1
    return any_e, first_e, mx


def match(A, B, R=500.0):
    r = np.pi / 180
    cands = []
    for i, (t1, la1, lo1) in enumerate(A):
        ix = {int(x): k for k, x in
              enumerate(t1.astype("datetime64[h]").astype(np.int64))}
        for j, (t2, la2, lo2) in enumerate(B):
            sh = [(ix[int(x)], k) for k, x in
                  enumerate(t2.astype("datetime64[h]").astype(np.int64))
                  if int(x) in ix]
            if len(sh) < 4:
                continue
            i1 = np.array([a for a, _ in sh])
            i2 = np.array([b for _, b in sh])
            a = (np.sin((la2[i2] - la1[i1]) * r / 2) ** 2
                 + np.cos(la1[i1] * r) * np.cos(la2[i2] * r)
                 * np.sin((lo2[i2] - lo1[i1]) * r / 2) ** 2)
            d = 6371 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
            if d.mean() <= R:
                cands.append((float(d.mean()), i, j))
    cands.sort()
    ua, ub = set(), set()
    pairs = []
    for _, i, j in cands:
        if i in ua or j in ub:
            continue
        ua.add(i)
        ub.add(j)
        pairs.append((i, j))
    return pairs


def detection_mask(east_bound):
    """Replicate precompute's trough-mask construction for one configuration,
    from the shared global-curvature-vorticity input, for the equality check."""
    import xarray as xr
    import metpy.calc as mpcalc
    sys.path.insert(0, str(CLONE))
    from feature.identification.african_easterly_waves.util import calc_adv

    ds = xr.open_dataset(RUN / "uv2004.nc").sel(level=700)
    ds = ds.sel(longitude=slice(-100, east_bound), latitude=slice(3, 29.5),
                time=slice("2004-06-01", "2004-10-31T23:00"))
    clim = xr.open_dataset(RUN / "cv_clim_local.nc") \
             .sel(lon=slice(-100, east_bound)) \
             .rename({"lat": "latitude", "lon": "longitude"}) \
             .sel(plev=70000.0)
    cv = mpcalc.smooth_n_point(ds["cv"], n=9, passes=2).metpy.dequantify()
    cv = cv.assign_coords(hourofyear=cv.time.dt.strftime("%m-%d %H"))
    cv_anom = cv.groupby("hourofyear") - clim.cv
    adv1, adv2 = calc_adv(cv_anom, ds["u"], ds["v"])
    cv_anom_h = cv_anom.swap_dims(dims_dict={"time": "hourofyear"})
    perc_h = cv_anom_h.where(
        cv_anom_h > clim.cva_quantile_hoy.sel(
            dict(hourofyear=cv_anom.hourofyear.data)))
    perc = perc_h.swap_dims(dims_dict={"hourofyear": "time"})
    mask = adv1.where(np.logical_and(
        ~np.isnan(perc), adv2.values > 0.0), other=np.nan)
    mask = mask.where(ds["u"].values < 0.0)
    return mask


def mask_equality():
    """Exact equality west of the edge margin; differing-cell count inside it."""
    m_c = detection_mask(45)
    m_e = detection_mask(60).sel(longitude=m_c.longitude)
    west = m_c.longitude <= (45.0 - EDGE_MARGIN_DEG)
    a_w = m_c.where(west, drop=True)
    b_w = m_e.where(west, drop=True)
    same = np.array_equal(a_w.values, b_w.values, equal_nan=True)
    margin = ~west
    a_m = m_c.where(margin, drop=True).values
    b_m = m_e.where(margin, drop=True).values
    diff = int(np.sum(~((a_m == b_m) | (np.isnan(a_m) & np.isnan(b_m)))))
    return same, diff, int(a_m.size)


def check_run(name, tracks, obj_e, max_vtx):
    e = EXPECTED[name]
    any_e, first_e, mx = east_stats(tracks)
    got = {"tracks": len(tracks), "east_any": any_e, "east_first": first_e,
           "max_track_lon": round(mx, 1), "obj_east": obj_e,
           "max_vertex_lon": round(max_vtx, 1)}
    ok = (len(tracks) == e["tracks"] and any_e == e["east_any"]
          and first_e == e["east_first"]
          and abs(mx - e["max_track_lon"][0]) <= e["max_track_lon"][1]
          and obj_e == e["obj_east"]
          and abs(max_vtx - e["max_vertex_lon"][0]) <= e["max_vertex_lon"][1])
    return ok, got


def main():
    _bind_environment()
    _build_inputs()
    t0 = time.time()
    p_ctrl = _run_pipeline("control")
    p_east = _run_pipeline("east")
    dur = round(time.time() - t0, 1)

    pub = load_wts(ROOT / "data/aewc_v2_pilot/wts2004.json")
    ctrl = load_wts(p_ctrl)
    east = load_wts(p_east)

    ok_pub, got_pub = check_run("published", *pub)
    assert ok_pub, f"published archive measurements changed: {got_pub}"
    ok_c, got_c = check_run("control", *ctrl)
    assert ok_c, f"control: {got_c} != expected"
    ok_e, got_e = check_run("east", *east)

    if MUTATION == "noext":
        # oracle 1: the mutated eastern run equals the control track for track
        same = (len(east[0]) == len(ctrl[0]) and all(
            np.array_equal(a[0], b[0]) and np.allclose(a[1], b[1])
            and np.allclose(a[2], b[2]) for a, b in zip(east[0], ctrl[0])))
        assert same, "noext: mutated east run does not equal control"
        # oracle 2: the extension predicate is killed
        if ok_e:
            print("MUTATION noext: eastern checks still passed (a finding)")
            sys.exit(2)
        print(f"MUTATION noext: east checks FAILED as required "
              f"(measured {got_e}, equal to control)")
        return

    assert ok_e, f"east: {got_e} != expected"

    same, margin_diff, margin_total = mask_equality()
    assert same, ("the two runs' detection masks differ west of the edge "
                  "margin; the shared-field construction is broken")
    assert margin_diff == EXPECTED["mask_margin_diff_cells"], margin_diff
    assert margin_total == EXPECTED["mask_margin_total_cells"], margin_total

    m = len(match(ctrl[0], pub[0]))
    assert m == EXPECTED["control_vs_published_matches"], m
    starters = [(t, la, lo) for t, la, lo in east[0] if lo[0] > 40]
    assert len(starters) == EXPECTED["east_starters"], len(starters)
    w20 = sum(float(lo.min()) < 20 for _, _, lo in starters)
    assert w20 == EXPECTED["east_starters_west_of_20E"], w20
    spairs = match(starters, pub[0])
    assert len(spairs) == EXPECTED["east_starters_matching_published"], len(spairs)
    if starters:
        crosser_idx = min(range(len(starters)),
                          key=lambda i: float(starters[i][2].min()))
        exp_lon, tol = EXPECTED["crosser_min_lon"]
        got_lon = float(starters[crosser_idx][2].min())
        assert abs(got_lon - exp_lon) <= tol, got_lon
        # the westward crosser and the published-matched starter are the same
        # track (an external review confirmed this held but was unasserted)
        if EXPECTED["east_starters_matching_published"] == 1:
            assert spairs[0][0] == crosser_idx, (
                "the published-matched starter is no longer the crosser")

    print("published:", got_pub)
    print("control:  ", got_c)
    print("east:     ", got_e)
    print(f"detection masks equal west of {45 - EDGE_MARGIN_DEG} E; "
          f"{margin_diff} differing cells in the edge margin")
    print(f"control vs published matches at 500 km: {m}")
    print(f"eastern starters: {len(starters)}, west of 20E: {w20}, "
          f"matching a published track: {len(spairs)}")
    print(f"pipeline wall clock (two runs, an observation): {dur} s")
    print("ALL ASSERTIONS PASSED")


if __name__ == "__main__":
    main()
