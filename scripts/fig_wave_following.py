#!/usr/bin/env python
"""Wave-following (trough-relative) composite of MCS about the moving AEW trough.

Uses AEWC (NCEI C00784) ERA-Interim curvature-vorticity trough trajectories over West
Africa and the original ISCCP MCS (csct_africa_cs245.nc). For every trough observation,
MCS within +/- 3 h are binned by longitude RELATIVE to the trough (east positive) and
latitude, accumulated over all troughs.

Null and inference per REPAIR_SPEC.md R1: the whole-wave anchor-permutation null
(each wave keeps its track shape and takes another same-year-month wave's anchor, so
the null's longitude distribution is the observed one, with no wrap and no seam),
N draws with one stored wave-by-draw assignment matrix reused by every panel and
tercile, and a selection-aware Monte Carlo p-value on the maximum band-profile excess
over the prespecified search interval (-10 to +10 degrees relative longitude). The
figure band is the pointwise 2.5-97.5 percent null envelope, labeled as such, with the
simultaneous 95 percent half-width drawn dashed.

Three panels share the longitude axis: (a) the latitude-longitude excess over the null
mean, (b) the 5-15 N band excess with the envelopes and the test result, (c) the band
excess split into weak/strong curvature-vorticity terciles under the same assignment
matrix. Draw profiles and the matrix are cached (deposit/null_r1_fig2.npz) keyed by
seed, draw count, and trough count, so the figure re-renders without recomputation.
"""

import argparse
import os

import numpy as np
import pandas as pd
import xarray as xr

from aew.composites import (_digest, amplitude_difference_test, assignment_artifact,
                            catalog_support_check, randomization_test,
                            wave_relative_counts)
from aew.data.aewc import load_aewc_troughs
from aew.plotting import panel_label

SEARCH = (-10.0, 10.0)


def band_profiles_for_draws(tr, masks, apply_draw, n_null, cs_time, cs_lon, cs_lat,
                            rel_c, lat_c, band):
    """Null band profiles per subset mask, plus the running 2D null-mean field for the
    full set (the first mask), under one shared assignment matrix."""
    profs = {name: np.empty((n_null, rel_c.size)) for name in masks}
    sum2d = None
    pad = float(rel_c.max())
    for b in range(n_null):
        lon_b = apply_draw(b)
        catalog_support_check(lon_b, pad, cs_lon)
        for j, (name, m) in enumerate(masks.items()):
            c, _ = wave_relative_counts(tr.time[m], lon_b[m], cs_time, cs_lon, cs_lat,
                                        rel_c, lat_c, time_tol_hours=3.0)
            if j == 0:
                sum2d = c if sum2d is None else sum2d + c
            profs[name][b] = c[band].mean(axis=0)
        if (b + 1) % 100 == 0:
            print(f"  null draw {b + 1}/{n_null}", flush=True)
    return profs, sum2d / n_null


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aewc-glob", default="data/aewc/ERA-Int_ew_700hPa_*_AFR.nc")
    ap.add_argument("--csct", default="data/original/csct/csct_africa_cs245.nc")
    ap.add_argument("--out", default="fig_wave_following.png")
    ap.add_argument("--n-null", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default="deposit/null_r1_fig2.npz")
    a = ap.parse_args()

    # AEWC troughs: West African AEW region, JAS (the anatomy cohort, full season by
    # design; the R3 eligibility gate is for response-conditioned estimands only)
    tr = (load_aewc_troughs(a.aewc_glob)
          .filter_region(min_lat=5, max_lat=20, min_lon=-40, max_lon=40)
          .filter_months([7, 8, 9]))
    print(f"AEWC trough observations (West Africa, JAS): {len(tr)}")

    cs = xr.open_dataset(a.csct)
    cs_time = pd.DatetimeIndex(cs["time"].values).values
    cs_lon = np.asarray(cs["lon"].values, dtype=float)
    cs_lat = np.asarray(cs["lat"].values, dtype=float)
    cs.close()

    rel_c = np.arange(-30.0, 30.1, 2.0)
    lat_c = np.arange(0.0, 25.1, 2.0)
    band = (lat_c >= 5) & (lat_c <= 15)

    # subset masks in full-trough index space; the tercile panel reuses the same
    # assignment matrix (one null serves every panel)
    crv = tr.variables["crv"]
    good = np.isfinite(crv)
    lo_t, hi_t = np.nanquantile(crv[good], [1.0 / 3.0, 2.0 / 3.0])
    masks = {
        "all": np.ones(len(tr), dtype=bool),
        "weak": good & (crv <= lo_t),
        "strong": good & (crv >= hi_t),
    }
    print(f"stratify by curvature vorticity: weak n={int(masks['weak'].sum())} "
          f"(<= {lo_t:.3g}), strong n={int(masks['strong'].sum())} (>= {hi_t:.3g})")

    # observed composites
    counts, n = wave_relative_counts(tr.time, tr.lon, cs_time, cs_lon, cs_lat,
                                     rel_c, lat_c, time_tol_hours=3.0)
    print(f"trough observations matched to MCS: {n}; "
          f"total MCS binned: {int(counts.sum())}")
    obs_prof = {name: None for name in masks}
    obs_prof["all"] = counts[band].mean(axis=0)
    for name in ("weak", "strong"):
        c, _ = wave_relative_counts(tr.time[masks[name]], tr.lon[masks[name]],
                                    cs_time, cs_lon, cs_lat, rel_c, lat_c,
                                    time_tol_hours=3.0)
        obs_prof[name] = c[band].mean(axis=0)

    # the null: the ONE assignment artifact, then manifest-validated profile caches
    perm, apply_draw, _ = assignment_artifact(
        tr.variables["traj_id"], tr.time, tr.lon, a.n_null, a.seed,
        "deposit/null_r1_assignment.npz")
    manifest = dict(seed=a.seed, n_null=a.n_null, n_troughs=len(tr),
                    h_tid=_digest(tr.variables["traj_id"]),
                    h_lon=_digest(tr.lon),
                    h_events=_digest(cs_time.astype("int64"), cs_lon, cs_lat),
                    h_axes=_digest(rel_c, lat_c))
    cached = None
    if os.path.exists(a.cache):
        z = np.load(a.cache, allow_pickle=False)
        ok = all(k in z.files
                 and str(z[k].item() if getattr(z[k], "shape", None) == () else z[k])
                 == str(v) for k, v in manifest.items())
        if ok and set(z.files) >= {f"profs_{k}" for k in masks}:
            cached = z
            print(f"loaded null profiles from {a.cache} (manifest verified)")
    if cached is None:
        print(f"computing {a.n_null} anchor-permutation null draws ...", flush=True)
        profs, null2d = band_profiles_for_draws(tr, masks, apply_draw, a.n_null,
                                                cs_time, cs_lon, cs_lat,
                                                rel_c, lat_c, band)
        os.makedirs(os.path.dirname(a.cache) or ".", exist_ok=True)
        np.savez_compressed(a.cache, perm=perm, null2d=null2d,
                            **{f"profs_{k}": v for k, v in profs.items()},
                            **{k: np.asarray(v) for k, v in manifest.items()})
        print(f"wrote {a.cache}")
    else:
        profs = {k: cached[f"profs_{k}"] for k in masks}
        null2d = cached["null2d"]

    tests = {name: randomization_test(obs_prof[name], profs[name], rel_c, SEARCH)
             for name in masks}
    stats_rows = [dict(figure="F2", subset=name, peak_excess=r["peak_excess"],
                       peak_rel_lon=r["peak_rel_lon"], peak_ratio=r["peak_ratio"],
                       p_value=r["p_value"], n_draws=r["n_draws"],
                       simult_half_width=r["simult_half_width"],
                       n_troughs=len(tr), n_matched=n, n_binned=int(counts.sum()))
                  for name, r in tests.items()]
    # selection-aware test that the strong tercile's peak exceeds the weak tercile's
    # (the amplitude-scaling claim; shared null draws give an exact paired difference)
    amp = amplitude_difference_test(obs_prof["strong"], profs["strong"],
                                    obs_prof["weak"], profs["weak"], rel_c, SEARCH)
    stats_rows.append(dict(figure="F2", subset="strong_minus_weak",
                           peak_excess=amp["peak_excess"],
                           peak_rel_lon=amp["peak_rel_lon"], peak_ratio=np.nan,
                           p_value=amp["p_value"], n_draws=amp["n_draws"],
                           simult_half_width=np.nan))
    print(f"R1 strong>weak: peak difference {amp['peak_excess']:.0f} at "
          f"{amp['peak_rel_lon']:+.0f} deg, p = {amp['p_value']:.4g} "
          f"({amp['n_draws']} draws)")
    pd.DataFrame(stats_rows).to_csv("deposit/null_r1_fig2_stats.csv", index=False,
                                    float_format="%.6f")
    for name, r in tests.items():
        print(f"R1 {name:6s}: peak {r['peak_excess']:.0f} at "
              f"{r['peak_rel_lon']:+.0f} deg, ratio {r['peak_ratio']:.3f}, "
              f"p = {r['p_value']:.4g} ({r['n_draws']} draws), "
              f"simultaneous 95% half-width {r['simult_half_width']:.0f}")

    anom = counts - null2d

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(8, 11), sharex=True, gridspec_kw={"height_ratios": [1.5, 1, 1]})

    # (a) latitude-longitude excess over the null mean, nice-round colorbar ticks
    amax = np.nanmax(np.abs(anom))
    nice = np.array([2.5, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000])
    j = int(np.argmax(amax / nice <= 4))
    tick_step = float(nice[j])
    fill_step = float(nice[max(0, j - 2)])
    vmax = np.ceil(amax / fill_step) * fill_step
    tmax = np.floor(amax / tick_step) * tick_step
    levels = np.arange(-vmax, vmax + fill_step * 0.5, fill_step)
    ticks = np.arange(-tmax, tmax + tick_step * 0.5, tick_step)
    pc = ax1.contourf(rel_c, lat_c, anom, levels=levels, cmap="RdBu_r", extend="both")
    ax1.axvline(0, color="green", lw=2)
    ax1.set_ylabel("Latitude (N)")
    ax1.set_title("MCS excess over the anchor-permutation null, "
                  "relative to the moving AEW trough")
    panel_label(ax1, "a", 15)
    cax = make_axes_locatable(ax1).append_axes("right", size="3%", pad=0.15)
    fig.colorbar(pc, cax=cax, label="MCS count minus null mean", ticks=ticks)
    make_axes_locatable(ax2).append_axes("right", size="3%", pad=0.15).set_axis_off()
    make_axes_locatable(ax3).append_axes("right", size="3%", pad=0.15).set_axis_off()

    # (b) band excess with the labeled envelopes and the randomization result
    r = tests["all"]
    ax2.fill_between(rel_c, r["point_lo"], r["point_hi"], color="grey", alpha=0.25,
                     label="null 2.5-97.5% pointwise envelope")
    ax2.axhline(r["simult_half_width"], color="grey", lw=0.9, ls="--",
                label="simultaneous 95% half-width")
    ax2.axhline(-r["simult_half_width"], color="grey", lw=0.9, ls="--")
    ax2.plot(rel_c, r["excess"], color="tab:red", label="observed minus null mean")
    ax2.axvline(0, color="green", lw=2)
    ax2.axhline(0, color="k", lw=0.6)
    ax2.set_ylabel("MCS excess, 5-15N mean")
    ax2.set_title(f"All troughs  (peak {r['peak_excess']:.0f} at "
                  f"{r['peak_rel_lon']:+.0f} deg, ratio {r['peak_ratio']:.2f}, "
                  f"p = {r['p_value']:.3g})", fontsize=10)
    panel_label(ax2, "b", 15)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # (c) tercile split under the same assignment matrix; the weak tercile's pointwise
    # envelope is drawn (the strong tercile's is comparable)
    rw, rs = tests["weak"], tests["strong"]
    ax3.fill_between(rel_c, rw["point_lo"], rw["point_hi"], color="tab:blue",
                     alpha=0.12, label="null pointwise envelopes (per tercile)")
    ax3.fill_between(rel_c, rs["point_lo"], rs["point_hi"], color="tab:red",
                     alpha=0.12)
    for rr, col in ((rw, "tab:blue"), (rs, "tab:red")):
        ax3.axhline(rr["simult_half_width"], color=col, lw=0.8, ls="--", alpha=0.7)
        ax3.axhline(-rr["simult_half_width"], color=col, lw=0.8, ls="--", alpha=0.7)
    ax3.plot(rel_c, rw["excess"], color="tab:blue",
             label=f"weak waves (n={int(masks['weak'].sum())}, "
                   f"peak {rw['peak_excess']:.0f}, p = {rw['p_value']:.3g})")
    ax3.plot(rel_c, rs["excess"], color="tab:red",
             label=f"strong waves (n={int(masks['strong'].sum())}, "
                   f"peak {rs['peak_excess']:.0f}, p = {rs['p_value']:.3g})")
    ax3.axvline(0, color="green", lw=2)
    ax3.axhline(0, color="k", lw=0.6)
    ax3.set_xlabel("Longitude relative to trough (deg; east positive)")
    ax3.set_ylabel("MCS excess, 5-15N mean")
    ax3.set_title("Stratified by wave amplitude (curvature-vorticity terciles)",
                  fontsize=10)
    panel_label(ax3, "c", 15)
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)

    fig.suptitle(f"Wave-following composite  ({n} trough obs, JAS)", fontsize=12)
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
