"""QTrack feasibility and eastern-extension probe, season 2004.

Runs the QTrack pipeline end to end on local ERA5 700 hPa winds twice, with
one parameter changed between runs: the initiation boundary, published at
40 E, against an extension to 59 E. Everything else identical. This is the
controlled experiment the nine-season east check called for, for the
point-tracker family: is eastern delivery configuration or method?

Inputs (pinned by SHA-256): data/era5/era5_u700_2000-2004_6h_global.nc and
the v700 counterpart (1.5 degree, 6-hourly, latitude -5 to 29.5). The probe
subsets June-October 2004 and 120 W to 60 E, interpolates linearly to
1 degree (the package's documented input resolution; at 1.5 degrees its
radial-average mask crashes on an internal hardcoded resolution), and runs
detection, tracking, and postprocessing per the package defaults.

KNOWN DEPARTURES from the published archive's configuration, so this is a
feasibility probe and not a reproduction claim: the local files span only
5 S to 29.5 N where the published input spans 19 S to 60 N (both the
southern band and the northern extension are absent here), the package's
nondivergent-wind step is inoperative (it requires a 'upsi' variable no
pipeline stage creates, per its own documentation caveat), the input is
interpolated from 1.5 degrees, and left_right_bounds is (-120, 60) rather
than the package default (-180, 40), identically in both runs. Under those
departures the control run still matches the published 2004 archive at 52
one-to-one pairs of 61 against 62 systems (500 km rule).

The package as installed needs one compatibility patch (numpy 2 removed
implicit 1-element-array-to-scalar conversion inside its haversine); the
probe refuses to run if the patch is absent, so the result is bound to the
code that produced it.

Run from the repo root: .venv/bin/python scripts/aewc_v2_qtrack_probe.py
  --mutate=noext   the "extended" run uses the published 40 E boundary too;
                   the eastern-delivery assertions must then FAIL, proving
                   they observe the configuration difference and not the
                   season.
Wall-clock about 2.5 minutes. Work happens in data/aewc_v2_pilot/qtrack_run/.
"""

from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "data" / "aewc_v2_pilot" / "qtrack_run"

INPUT_SHA256 = {
    "data/era5/era5_u700_2000-2004_6h_global.nc":
        "819095f9f6df44778f4ebad59f2b3fdf19087f4e4a63994f1fbadb6dc36bbc7b",
    "data/era5/era5_v700_2000-2004_6h_global.nc":
        "d4fff9798b2effb2cb294af988f071f7a38dfa3bdc693f8c049aa9fc56fe0057",
}

EXPECTED = {
    "control": {"systems": 61, "east_any": 1, "east_first": 1,
                "max_lon": (41.0, 0.1)},
    "east": {"systems": 75, "east_any": 18, "east_first": 18,
             "max_lon": (48.7, 0.1)},
    "published_2004_systems": 62,
    "control_vs_published_matches": 52,
    "starters_west_of_20E": 3,
    "starters_west_of_0": 2,
    "starters_median_lifetime_days": (3.0, 0.1),
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


PATCHED_TRACKING_SHA256 = \
    "a6606b52bead226a137766ec23f1533b357895b398416fb6999083e86edc2592"


def _require_patch():
    import qtrack
    import qtrack.tracking as qt
    assert getattr(qtrack, "__version__", "0.0.4") in ("0.0.4",), (
        f"qtrack version {getattr(qtrack,'__version__','?')} is not the "
        f"pinned 0.0.4")
    tracking = Path(qt.__file__)          # the module actually imported
    text = tracking.read_text()
    assert text.count("float(_np.asarray(x).item())") == 2, (
        "the imported qtrack lacks the numpy-2 scalar patch at both haversine "
        "sites; re-apply it or expect a TypeError")
    got = _sha256(tracking)
    assert got == PATCHED_TRACKING_SHA256, (
        f"imported qtrack/tracking.py sha256 {got[:12]}... differs from the "
        f"pinned patched file; results are not bound to the reviewed code")


def _tracks(path):
    d = xr.open_dataset(path)
    lon, lat = d["AEW_lon"].values, d["AEW_lat"].values
    t = d["time"].values
    out = []
    for s in range(lon.shape[0]):
        g = np.isfinite(lon[s]) & np.isfinite(lat[s])
        if g.sum() >= 4:
            out.append((t[g], lat[s][g], lon[s][g]))
    d.close()
    return out


def _east_stats(tracks):
    any_e = first_e = 0
    mx = -180.0
    for t, la, lo in tracks:
        mx = max(mx, float(lo.max()))
        if (lo > 40).any():
            any_e += 1
        if lo[0] > 40:
            first_e += 1
    return any_e, first_e, mx


def _match(A, B, R=500.0):
    r = np.pi / 180
    used = set()
    m = 0
    for t1, la1, lo1 in A:
        ix = {int(x): i for i, x in
              enumerate(t1.astype("datetime64[h]").astype(np.int64))}
        best = None
        for j, (t2, la2, lo2) in enumerate(B):
            if j in used:
                continue
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
            if d.mean() <= R and (best is None or d.mean() < best[0]):
                best = (d.mean(), j)
        if best:
            used.add(best[1])
            m += 1
    return m


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    _require_patch()
    for rel, want in INPUT_SHA256.items():
        got = _sha256(ROOT / rel)
        assert got == want, f"{rel}: sha256 {got[:12]}... != pinned {want[:12]}..."
    from qtrack.core import prep_data, COMPUTE_CURV_VORT_NON_DIV_UPDATE
    from qtrack.tracking import run_tracking, run_postprocessing

    timings = {}

    # 1. subset and regrid
    t0 = time.time()
    u = xr.open_dataset(ROOT / "data/era5/era5_u700_2000-2004_6h_global.nc")
    v = xr.open_dataset(ROOT / "data/era5/era5_v700_2000-2004_6h_global.nc")
    ds = xr.Dataset({"u": u["u700"].squeeze("pressure_level", drop=True),
                     "v": v["v700"].squeeze("pressure_level", drop=True)})
    ds = ds.sel(time=slice("2004-06-01", "2004-10-31T18"),
                longitude=slice(-120, 60))
    ds = ds.interp(longitude=np.arange(-120, 60.1, 1.0),
                   latitude=np.arange(-5, 29.1, 1.0), method="linear")
    ds.to_netcdf(RUN / "era5_uv700_2004_1deg.nc")
    timings["subset_regrid_s"] = round(time.time() - t0, 1)

    # 2. prep and curvature vorticity (the nondivergent-wind step is
    #    inoperative in the installed package; nondiv=False is the working path)
    t0 = time.time()
    import os
    os.chdir(RUN)
    prep_data("era5_uv700_2004_1deg.nc", cut_lev_val=700,
              data_out="prepped_2004.nc")
    COMPUTE_CURV_VORT_NON_DIV_UPDATE("prepped_2004.nc", "curv_vort_2004.nc",
                                     res=1, radius=600, njobs=6, nondiv=False)
    timings["curv_vort_s"] = round(time.time() - t0, 1)

    # 3. two tracking runs differing in one parameter
    east_bound = 40 if MUTATION == "noext" else 59
    configs = {"control": 40, "east": east_bound}
    # mutation outputs go to separate filenames so a mutation run cannot
    # overwrite the clean run's artifacts (a clean-run rerun after the first
    # mutation run was needed to restore them; this prevents the recurrence)
    suffix = f"_mut_{MUTATION}" if MUTATION else ""
    for name, init_e in configs.items():
        t0 = time.time()
        run_tracking(input_file="curv_vort_2004.nc",
                     save_file=f"AEW_raw_{name}{suffix}.nc",
                     initiation_bounds=(-35, init_e),
                     left_right_bounds=(-120, 60),
                     spatial_res=1, temporal_res=6, run_animation=False)
        run_postprocessing(input_file=f"AEW_raw_{name}{suffix}.nc",
                           curv_data_file="curv_vort_2004.nc",
                           real_year_used=2004, TC_pairing=False,
                           hovmoller_save=False,
                           save_obj_file=f"AEW_pp_{name}{suffix}.pkl",
                           save_nc_file=f"AEW_pp_{name}{suffix}.nc")
        timings[f"track_{name}_s"] = round(time.time() - t0, 1)

    # 4. measure and assert
    results = {}
    for name in configs:
        tr = _tracks(RUN / f"AEW_pp_{name}{suffix}.nc")
        any_e, first_e, mx = _east_stats(tr)
        results[name] = {"systems": len(tr), "east_any": any_e,
                         "east_first": first_e, "max_lon": round(mx, 1)}
        e = EXPECTED[name]
        ok = (len(tr) == e["systems"] and any_e == e["east_any"]
              and first_e == e["east_first"]
              and abs(mx - e["max_lon"][0]) <= e["max_lon"][1])
        if MUTATION == "noext" and name == "east":
            # oracle 1: the mutated eastern run must EQUAL the control run
            # track for track (same configuration, same output), not merely
            # differ from the extension expectation
            ctl = _tracks(RUN / f"AEW_pp_control{suffix}.nc")
            same = (len(tr) == len(ctl) and all(
                len(t1[0]) == len(t2[0])
                and np.array_equal(t1[0], t2[0])
                and np.allclose(t1[1], t2[1]) and np.allclose(t1[2], t2[2])
                for t1, t2 in zip(tr, ctl)))
            assert same, ("noext: mutated eastern run does not equal the "
                          "control run track for track")
            # oracle 2: the extension predicate is killed
            if ok:
                print("MUTATION noext: eastern assertions still passed; the "
                      "probe does not observe the configuration (a finding)")
                sys.exit(2)
            print(f"MUTATION noext: '{name}' assertions FAILED as required "
                  f"(measured {results[name]}, equal to control)")
            continue
        assert ok, f"{name}: measured {results[name]} != expected {e}"

    if MUTATION is None:
        east_tr = _tracks(RUN / "AEW_pp_east.nc")
        starters = [(t, la, lo) for t, la, lo in east_tr if lo[0] > 40]
        west = [float(lo.min()) for _, _, lo in starters]
        lifetimes = [float((t.max() - t.min()) / np.timedelta64(1, "D"))
                     for t, _, _ in starters]
        assert sum(w < 20 for w in west) == EXPECTED["starters_west_of_20E"]
        assert sum(w < 0 for w in west) == EXPECTED["starters_west_of_0"]
        med = float(np.median(lifetimes))
        lo_, tol = EXPECTED["starters_median_lifetime_days"]
        assert abs(med - lo_) <= tol, f"starter median lifetime {med}"

        pub = _tracks(ROOT / "data/aewc_v2_pilot/"
                             "AEW_tracks_post_processed_year_2004.nc")
        assert len(pub) == EXPECTED["published_2004_systems"], len(pub)
        # the three westward-propagating eastern starters each match a
        # published system (>= 4 shared times, mean <= 500 km): the extension
        # recovers missing eastern prefixes of archived waves, it does not
        # discover waves absent from the archive (external-review finding)
        west_starters = [st for st in starters if float(st[2].min()) < 20]
        assert len(west_starters) == 3
        n_matched_starters = _match(west_starters, pub)
        assert n_matched_starters == 3, (
            f"only {n_matched_starters} of 3 westward eastern starters match "
            f"a published system; the prefix-recovery claim fails")
        # the Atlantic-crosser starts at 47.4 E in the extension while its
        # archived counterpart family begins near 36 E
        crosser = min(west_starters, key=lambda st: float(st[2].min()))
        assert abs(float(crosser[2].min()) - (-42.5)) < 0.1
        assert abs(float(crosser[2][0]) - 47.4) < 0.1, float(crosser[2][0])
        ctrl = _tracks(RUN / "AEW_pp_control.nc")
        m = _match(ctrl, pub)
        assert m == EXPECTED["control_vs_published_matches"], m

        print("control:", results["control"])
        print("east:   ", results["east"])
        print(f"reproduction: {m} of {len(ctrl)} control vs {len(pub)} "
              f"published systems matched at 500 km")
        print(f"eastern starters: {len(starters)}, west of 20E "
              f"{sum(w < 20 for w in west)}, west of 0 "
              f"{sum(w < 0 for w in west)}, median lifetime {med:.1f} d")
        print("timings:", timings)
        print("ALL ASSERTIONS PASSED")


if __name__ == "__main__":
    main()
