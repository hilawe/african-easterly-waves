#!/usr/bin/env python
"""Continuous-response control model for the moisture-gating claim.

The trough-relative composite and the developing/non-developing split are outcome-
conditioned, which a referee can read as a bare precursor association. This asks the
sharper question directly: does the pre-trough 700 hPa inflow moisture still predict the
forward mesoscale-convective response after controlling for the other environmental
axes a reviewer would name, and does the relationship hold out of sample?

Design. Every corridor trough (5-20 N, 30 W to 40 E, JAS) is one observation. The
response is the forward CS-245 count in a trough-relative box over the next 24 h (the
same count that defines the MCS-active and MCS-quiet classes, used here as a continuous
outcome). The predictors are standardized to pooled unit variance so their coefficients
are comparable:

  inflow_rh   the along-inflow 700 hPa relative humidity 72 h before passage (the paper's
              Lagrangian mechanism variable; response-independent, since the back-
              trajectories never see the outcome)
  box_rh      the fixed-frame 700 hPa box relative humidity 24 h before passage (the
              diluted Eulerian measure, run as an alternative moisture predictor)
  antecedent  the prior-day CS-245 count near the box (t-48 h .. t-24 h)
  amplitude   the trough-mean curvature vorticity (wave-amplitude proxy)
  shear       the 600-925 hPa box shear 24 h before passage
  tcwv        the total column water vapour box 24 h before passage

with longitude-bin-by-month and calendar-year fixed effects. The model is a Poisson GLM
with cluster-robust standard errors on the wave (traj_id), which handles both the count
overdispersion and the within-wave autocorrelation, matching the wave-cluster bootstrap
used elsewhere. The model is fitted on the development sample (2000-2004), on the 20
held-out seasons, and on the pooled record; a moisture coefficient that stays positive
and significant across the three tiers is the out-of-sample validation.

Writes control_model_summary.txt and control_model_coeffs.csv under --outdir.
"""

import argparse
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import xarray as xr

from aew.data.aewc import load_aewc_trajectories
from aew.data.era5 import load_region_6h
from aew.environment import forward_response, lead_field_box
from aew.trajectory import Gridded, back_trajectories
from validate_heldout import parse_years

LAT_LO, LAT_HI = 5.0, 15.0
LEAD_H = 24.0
BACK_H = 48.0
RESP_WIN_H = 24.0
DLON = 8.0
BOX_DLON = 5.0
TOL_H = 3.0
SEED_DLON = (-4.0, 0.0, 4.0)
SEED_LATS = (7.0, 10.0, 13.0)
LON_EDGES = np.arange(-30, 41, 10.0)

TIERS = {"dev": "2000-2004", "heldout": "1983-1999,2005-2007", "pooled": "1983-2007"}
PREDICTORS = ["inflow_rh", "box_rh", "antecedent", "amplitude", "shear", "tcwv"]


def build_design(years, csct_path):
    """Per-trough design matrix over all corridor troughs in ``years``."""
    aewc_paths = [f"data/aewc/ERA-Int_ew_700hPa_{y}_AFR.nc" for y in years]
    tr = (load_aewc_trajectories(aewc_paths)
          .filter_region(min_lat=5, max_lat=20, min_lon=-30, max_lon=40)
          .filter_months([7, 8, 9]))
    cs = xr.open_dataset(csct_path)
    cst_all = pd.DatetimeIndex(cs["time"].values)
    csx = np.asarray(cs["lon"].values, float)
    csy = np.asarray(cs["lat"].values, float)
    cs.close()
    inyr = np.isin(cst_all.year, years)
    cst, csx, csy = cst_all[inyr].values, csx[inyr], csy[inyr]

    n = len(tr)
    print(f"corridor troughs {n}, systems {cst.size}", flush=True)
    resp = forward_response(tr.time, tr.lon, cst, csx, csy, RESP_WIN_H, DLON,
                            LAT_LO, LAT_HI)
    prior = tr.time.astype("datetime64[ns]") - np.timedelta64(48, "h")
    antecedent = forward_response(prior, tr.lon, cst, csx, csy, 24.0, DLON + 2.0,
                                  LAT_LO, LAT_HI)
    month = pd.DatetimeIndex(tr.time).month.values
    year = pd.DatetimeIndex(tr.time).year.values
    lonbin = np.digitize(tr.lon, LON_EDGES)

    # Eulerian boxes at -24 h: RH700, TCWV, 600-925 shear
    rt, rlat, rlon, rfield = load_region_6h("r700", years=years)
    box_rh = lead_field_box(tr.time, tr.lon, rt.values, rlat, rlon, rfield,
                            LEAD_H, tol_h=TOL_H, dlon=BOX_DLON,
                            lat_lo=LAT_LO, lat_hi=LAT_HI)
    tt, tlat, tlon, tfield = load_region_6h("tcwv", years=years)
    tcwv = lead_field_box(tr.time, tr.lon, tt.values, tlat, tlon, tfield,
                          LEAD_H, tol_h=TOL_H, dlon=BOX_DLON,
                          lat_lo=LAT_LO, lat_hi=LAT_HI)
    del tfield
    ft, flat, flon, u6 = load_region_6h("u600", years=years)
    _, _, _, u9 = load_region_6h("u925", years=years)
    _, _, _, v6 = load_region_6h("v600", years=years)
    _, _, _, v9 = load_region_6h("v925", years=years)
    shear_field = np.sqrt((u6 - u9) ** 2 + (v6 - v9) ** 2)
    del u6, u9, v6, v9
    shear = lead_field_box(tr.time, tr.lon, ft.values, flat, flon, shear_field,
                           LEAD_H, tol_h=TOL_H, dlon=BOX_DLON,
                           lat_lo=LAT_LO, lat_hi=LAT_HI)
    del shear_field

    # Lagrangian along-inflow RH at -72 h for every trough (9 parcels each)
    tu, wlat, wlon, uu = load_region_6h("u700", years=years)
    u = Gridded(tu.values, wlat, wlon, uu)
    del uu
    tv, _, _, vv = load_region_6h("v700", years=years)
    v = Gridded(tv.values, wlat, wlon, vv)
    del vv
    seed_time = tr.time.astype("datetime64[ns]") - np.timedelta64(int(LEAD_H * 3600), "s")
    gd, gl = np.meshgrid(SEED_DLON, SEED_LATS)
    npar = gd.size
    seeds_t = np.repeat(seed_time, npar)
    seeds_lon = (tr.lon[:, None] + gd.ravel()[None, :]).ravel()
    seeds_lat = np.broadcast_to(gl.ravel()[None, :], (n, npar)).ravel().copy()
    print(f"integrating {seeds_t.size} parcels {BACK_H:.0f} h backward at 700 hPa ...",
          flush=True)
    elapsed, plat, plon = back_trajectories(u, v, seeds_t, seeds_lat, seeds_lon,
                                            hours=BACK_H, dt_hours=1.0)
    del u, v
    rh = Gridded(rt.values, rlat, rlon, rfield)
    del rfield
    k = int(round(BACK_H / (elapsed[1] - elapsed[0])))
    t_abs = (seeds_t.astype("datetime64[ns]").astype("int64") / 3.6e12) - BACK_H
    inflow_rh = np.nanmean(rh.sample(t_abs, plat[k], plon[k]).reshape(n, npar), axis=1)

    df = pd.DataFrame({
        "response": resp.astype(int),
        "inflow_rh": inflow_rh,
        "box_rh": box_rh,
        "antecedent": antecedent,
        "amplitude": tr.variables["crv"],
        "shear": shear,
        "tcwv": tcwv,
        "lonmonth": [f"L{int(b)}M{int(m)}" for b, m in zip(lonbin, month)],
        "year": year,
        "wave": tr.variables["traj_id"],
    })
    before = len(df)
    df = df[np.isfinite(df[PREDICTORS]).all(axis=1)].reset_index(drop=True)
    print(f"analyzable troughs (finite predictors): {len(df)} of {before}", flush=True)
    return df


def fit_tier(df, years, moisture, scaler_mean, scaler_std, controls):
    """Poisson GLM on the tier subset, cluster-robust on the wave. ``moisture`` is the
    single moisture predictor (inflow_rh or box_rh); ``controls`` is the explicit list of
    non-moisture controls, so the primary spec (one moisture measure) and the full panel
    (plus the collinear TCWV) are both expressible."""
    sub = df[df.year.isin(years)].copy()
    terms = [moisture] + controls
    for c in terms:
        sub[f"z_{c}"] = (sub[c] - scaler_mean[c]) / scaler_std[c]
    # drop fixed-effect levels with no within-level variation in the subset
    rhs = " + ".join(f"z_{c}" for c in terms) + " + C(lonmonth) + C(year)"
    model = smf.glm(f"response ~ {rhs}", data=sub,
                    family=sm.families.Poisson())
    res = model.fit(cov_type="cluster", cov_kwds={"groups": sub["wave"].values})
    return res, sub, terms


def ladder(df, scaler_mean, scaler_std):
    """The inflow-moisture coefficient as controls are added (pooled, cluster-robust).
    Makes the absorption of the moisture signal by the convective-regime controls
    transparent, rather than reporting only the fully-loaded panel."""
    d = df.copy()
    for c in PREDICTORS:
        d[f"z_{c}"] = (d[c] - scaler_mean[c]) / scaler_std[c]
    fe = " + C(lonmonth) + C(year)"
    specs = [
        ("moisture + fixed effects only", "z_inflow_rh" + fe),
        ("+ wave amplitude", "z_inflow_rh + z_amplitude" + fe),
        ("+ shear", "z_inflow_rh + z_amplitude + z_shear" + fe),
        ("+ antecedent convection (PRIMARY)",
         "z_inflow_rh + z_amplitude + z_shear + z_antecedent" + fe),
        ("+ total column water vapour (full panel)",
         "z_inflow_rh + z_amplitude + z_shear + z_antecedent + z_tcwv" + fe),
    ]
    out = ["\nCONTROL LADDER, inflow_rh coefficient as controls are added (pooled):"]
    for label, rhs in specs:
        res = smf.glm(f"response ~ {rhs}", data=d,
                      family=sm.families.Poisson()).fit(
            cov_type="cluster", cov_kwds={"groups": d["wave"].values})
        c, p = res.params["z_inflow_rh"], res.pvalues["z_inflow_rh"]
        s = "significant" if p < 0.05 else "ns"
        out.append(f"  {label:44s} {c:+.4f}  p={p:.2e}  {s}  "
                   f"(IRR/SD {np.exp(c):.3f})")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csct", default="data/original/csct/csct_africa_cs245.nc")
    ap.add_argument("--cache", default=None,
                    help="CSV cache of the design matrix; built and written if absent")
    ap.add_argument("--outdir", default="deposit")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    pooled_years = parse_years(TIERS["pooled"])
    if a.cache and os.path.exists(a.cache):
        df = pd.read_csv(a.cache)
        print(f"loaded cached design {len(df)} from {a.cache}", flush=True)
    else:
        df = build_design(pooled_years, a.csct)
        if a.cache:
            df.to_csv(a.cache, index=False)
    # one scaler from the pooled analyzable sample, applied to every tier
    scaler_mean = {c: float(df[c].mean()) for c in PREDICTORS}
    scaler_std = {c: float(df[c].std()) for c in PREDICTORS}

    lines, rows = [ladder(df, scaler_mean, scaler_std)], []
    # PRIMARY spec = one moisture measure + the three non-moisture controls; FULL spec
    # adds the collinear TCWV. inflow_rh is the Lagrangian mechanism variable; box_rh is
    # the diluted fixed-frame measure, reported as a robustness comparison.
    SPECS = [("primary", ["antecedent", "amplitude", "shear"]),
             ("full+tcwv", ["antecedent", "amplitude", "shear", "tcwv"])]
    for moisture in ("inflow_rh", "box_rh"):
        for spec_name, controls in SPECS:
            lines.append(f"\n{'=' * 78}\nMOISTURE {moisture}  |  SPEC {spec_name} "
                         f"(controls: {', '.join(controls)})\n{'=' * 78}")
            for tier in ("dev", "heldout", "pooled"):
                years = parse_years(TIERS[tier])
                res, sub, terms = fit_tier(df, years, moisture, scaler_mean,
                                           scaler_std, controls)
                coef = res.params[f"z_{moisture}"]
                se = res.bse[f"z_{moisture}"]
                p = res.pvalues[f"z_{moisture}"]
                irr = np.exp(coef)
                ci = res.conf_int().loc[f"z_{moisture}"]
                sig = "significant" if p < 0.05 else "ns"
                lines.append(
                    f"  tier {tier:7s} (n={len(sub)}, waves={sub.wave.nunique()}): "
                    f"coef {coef:+.4f} (SE {se:.4f}), p={p:.2e}, {sig}, "
                    f"IRR/SD {irr:.3f} [{np.exp(ci[0]):.3f}, {np.exp(ci[1]):.3f}]")
                rows.append(dict(moisture=moisture, spec=spec_name, tier=tier,
                                 n=len(sub), waves=int(sub.wave.nunique()), coef=coef,
                                 se=se, pvalue=p, irr=irr, irr_lo=float(np.exp(ci[0])),
                                 irr_hi=float(np.exp(ci[1]))))
                if tier == "pooled" and spec_name == "primary" and moisture == "inflow_rh":
                    lines.append("    full standardized panel (this fit):")
                    for c in terms:
                        cc, pp = res.params[f"z_{c}"], res.pvalues[f"z_{c}"]
                        s = "sig" if pp < 0.05 else "ns"
                        lines.append(f"      {c:12s} {cc:+.4f}  p={pp:.2e}  {s}")

    # coefficient tables for the supplement figure: the full pooled panel (inflow_rh
    # spec) and the control ladder
    d = df.copy()
    for c in PREDICTORS:
        d[f"z_{c}"] = (d[c] - scaler_mean[c]) / scaler_std[c]
    panel_terms = ["inflow_rh", "antecedent", "amplitude", "shear", "tcwv"]
    rhs = " + ".join(f"z_{c}" for c in panel_terms) + " + C(lonmonth) + C(year)"
    pres = smf.glm(f"response ~ {rhs}", data=d, family=sm.families.Poisson()).fit(
        cov_type="cluster", cov_kwds={"groups": d["wave"].values})
    ci = pres.conf_int()
    panel_rows = []
    for c in panel_terms:
        panel_rows.append(dict(
            predictor=c, coef=float(pres.params[f"z_{c}"]),
            ci_lo=float(ci.loc[f"z_{c}"][0]), ci_hi=float(ci.loc[f"z_{c}"][1]),
            pvalue=float(pres.pvalues[f"z_{c}"]),
            irr=float(np.exp(pres.params[f"z_{c}"])),
            irr_lo=float(np.exp(ci.loc[f"z_{c}"][0])),
            irr_hi=float(np.exp(ci.loc[f"z_{c}"][1]))))
    pd.DataFrame(panel_rows).to_csv(
        os.path.join(a.outdir, "control_model_panel.csv"), index=False,
        float_format="%.6f")

    # multicollinearity check: pairwise correlations among the standardized predictors
    # (pooled). Written alongside the panel so the supplement can state the maximum.
    corr = d[[f"z_{c}" for c in panel_terms]].corr()
    corr.index = panel_terms
    corr.columns = panel_terms
    corr.to_csv(os.path.join(a.outdir, "control_model_predictor_corr.csv"),
                float_format="%.3f")
    tri = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    imax = tri.abs().stack().idxmax()
    print(f"\npredictor correlations (pooled, standardized): max |r| = "
          f"{abs(tri.loc[imax]):.2f} ({imax[0]} vs {imax[1]})")
    print(corr.round(2).to_string())

    ladder_specs = [
        ("fixed effects only", []),
        ("+ amplitude", ["amplitude"]),
        ("+ shear", ["amplitude", "shear"]),
        ("+ antecedent", ["amplitude", "shear", "antecedent"]),
        ("+ column vapour", ["amplitude", "shear", "antecedent", "tcwv"]),
    ]
    lrows = []
    for label, ctrls in ladder_specs:
        terms = ["inflow_rh"] + ctrls
        rr = " + ".join(f"z_{c}" for c in terms) + " + C(lonmonth) + C(year)"
        r = smf.glm(f"response ~ {rr}", data=d, family=sm.families.Poisson()).fit(
            cov_type="cluster", cov_kwds={"groups": d["wave"].values})
        cc = r.conf_int().loc["z_inflow_rh"]
        lrows.append(dict(step=label, coef=float(r.params["z_inflow_rh"]),
                          ci_lo=float(cc[0]), ci_hi=float(cc[1]),
                          pvalue=float(r.pvalues["z_inflow_rh"]),
                          irr=float(np.exp(r.params["z_inflow_rh"])),
                          irr_lo=float(np.exp(cc[0])), irr_hi=float(np.exp(cc[1]))))
    pd.DataFrame(lrows).to_csv(
        os.path.join(a.outdir, "control_model_ladder.csv"), index=False,
        float_format="%.6f")

    summary = "\n".join(lines)
    print(summary)
    with open(os.path.join(a.outdir, "control_model_summary.txt"), "w") as f:
        f.write("Continuous-response control model for the moisture-gating claim.\n")
        f.write("Poisson GLM, cluster-robust SE on the wave; predictors standardized "
                "to pooled unit variance; longitude-month and year fixed effects.\n")
        f.write(summary + "\n")
    pd.DataFrame(rows).to_csv(
        os.path.join(a.outdir, "control_model_coeffs.csv"), index=False,
        float_format="%.6f")
    print(f"\nwrote {a.outdir}/control_model_summary.txt and control_model_coeffs.csv")


if __name__ == "__main__":
    main()
