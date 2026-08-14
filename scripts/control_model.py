#!/usr/bin/env python
"""Continuous-response control model for the moisture-conditioning claim.

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
with cluster-robust standard errors on the wave (traj_id). Clustering protects the
inference against within-wave dependence and variance misspecification; it does not make
the Poisson mean-variance relation an overdispersion model, which is why the
negative-binomial sensitivity exists. The clustering matches the wave-cluster bootstrap
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
from aew.environment import complete_window_mask, forward_response, lead_field_box
from aew.terrain import DELTA_HPA, mask_level_inplace
from aew.trajectory import Gridded, aggregate_parcels, back_trajectories
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

def _file_digest(path):
    """SHA-256 of a file's bytes, or None when absent."""
    import hashlib
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


ANCHORS = ("troughs_pooled.csv", "cases_pooled_700.csv")

# Comparison tolerance. The deposit is written at four decimal places while the cache
# carries full precision, so exact equality is impossible by construction.
TOL = 1e-3

# cache column -> deposited counterpart in cases_pooled_700.csv
COMPARED = (("inflow_rh", "rh_m72"), ("box_rh", "box_rh_m24"),
            ("antecedent", "antecedent"))

KEY = ["time_k", "lon_k", "wave_k"]


def read_design_csv(path):
    """Read a cached design or a deposited anchor with an EXACT float round-trip.

    pandas' default C parser is fast but not correctly rounded, so a plain read and
    rewrite moves the last bit of some values. Measured on the real shear column,
    11.323954631638083 on disk reads back and rewrites as 11.323954631638085. The
    published control_model_design.csv is written from the cache on the cached path and
    from the fresh build otherwise, so without this the two paths produce numerically
    different tables and the deposited file depends on whether a cache happened to
    exist. The guard's own comparisons are at 1e-3 and are indifferent to this, which is
    exactly why it would have gone unnoticed.
    """
    return pd.read_csv(path, float_precision="round_trip")


def _freshness_path(cache):
    return cache + ".sources.json"


def write_freshness_record(cache, dep_dir):
    """Record the digests of the two anchor tables the cache was built against.

    THIS RECORDS NOTHING ABOUT THE CACHE ITSELF, and the omission is the design. Round
    10 showed that a manifest hashing the cache is self-authorizing, because anything
    this program writes about its own output can be rewritten by running this program
    again. So the record answers only the one question no other mechanism can answer,
    which is whether the anchors on disk are the ones the cache was built from. Row
    counts, column lists and a digest of the cache all lived here before and none was
    load-bearing once the cohort is checked against a table this program does not write.

    See docs/DESIGN_CACHE_SPEC.md for what the guard establishes and what it does not.
    """
    import json
    rec = dict(sources={name: _file_digest(os.path.join(dep_dir, name))
                        for name in ANCHORS})
    with open(_freshness_path(cache), "w") as fh:
        json.dump(rec, fh, indent=2, sort_keys=True)
    return rec


PREDICTORS = ["inflow_rh", "box_rh", "antecedent", "amplitude", "shear", "tcwv"]


def complete_case(df):
    """The modelled cohort, every eligible trough whose six predictors are all finite.

    Applied identically on the cached and the freshly built path, which is what lets the
    cache hold the ELIGIBLE cohort rather than this one. Storing the eligible cohort is
    what makes the cache's row set checkable, because it then equals the deposited
    trough set exactly, whereas the complete-case cohort is not derivable from the
    deposit at all: of the 2,082 eligible troughs dropped here, all 2,082 have a finite
    deposited inflow_rh, so the deposit cannot say which troughs the filter removes.
    """
    return df[np.isfinite(df[PREDICTORS]).all(axis=1)].reset_index(drop=True)


def build_design(years, csct_path):
    """Per-trough design matrix over all corridor troughs in ``years``."""
    aewc_paths = [f"data/aewc/ERA-Int_ew_700hPa_{y}_AFR.nc" for y in years]
    tr = (load_aewc_trajectories(aewc_paths)
          .filter_region(min_lat=5, max_lat=20, min_lon=-30, max_lon=40)
          .filter_months([7, 8, 9]))
    n_all = len(tr)
    tr = tr.filter(complete_window_mask(tr.time))  # R3 complete-window eligibility
    print(f"eligible troughs {len(tr)} of {n_all} (complete windows)", flush=True)
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

    # Eulerian boxes at -24 h: RH700, TCWV, 600-925 shear (terrain-masked, R2)
    spt, splat, splon, spf = load_region_6h("sp", years=years)
    rt, rlat, rlon, rfield = load_region_6h("r700", years=years)
    if not (spt.equals(rt) and np.array_equal(splat, rlat)
            and np.array_equal(splon, rlon)):
        raise ValueError("surface-pressure grid/time differs from r700")
    mask_level_inplace(rfield, spf, 700.0, DELTA_HPA)
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
    for f_, lev_ in ((u6, 600.0), (v6, 600.0), (u9, 925.0), (v9, 925.0)):
        mask_level_inplace(f_, spf, lev_, DELTA_HPA)   # R2: common-valid shear
    shear_field = np.sqrt((u6 - u9) ** 2 + (v6 - v9) ** 2)
    del u6, u9, v6, v9
    shear = lead_field_box(tr.time, tr.lon, ft.values, flat, flon, shear_field,
                           LEAD_H, tol_h=TOL_H, dlon=BOX_DLON,
                           lat_lo=LAT_LO, lat_hi=LAT_HI)
    del shear_field

    # Lagrangian along-inflow RH at -72 h for every trough (9 parcels each)
    tu, wlat, wlon, uu = load_region_6h("u700", years=years)
    u = Gridded(tu.values, wlat, wlon, mask_level_inplace(uu, spf, 700.0, DELTA_HPA))
    del uu
    tv, _, _, vv = load_region_6h("v700", years=years)
    v = Gridded(tv.values, wlat, wlon, mask_level_inplace(vv, spf, 700.0, DELTA_HPA))
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
    inflow_rh, _ = aggregate_parcels(rh.sample(t_abs, plat[k], plon[k]), n, npar)

    df = pd.DataFrame({
        "time": pd.DatetimeIndex(tr.time),
        "lon": tr.lon,
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
    # Returns the ELIGIBLE cohort, one row per corridor trough that passed the
    # complete-window rule, with non-finite predictors left in place. The complete-case
    # filter is complete_case() and is applied by the caller, identically on the fresh
    # and the cached path. Keeping the two separate is what makes the cache's row set
    # checkable against the deposit, since the eligible cohort equals the deposited
    # trough set while the complete-case cohort is not derivable from the deposit.
    print(f"eligible design rows {len(df)}, of which "
          f"{int(np.isfinite(df[PREDICTORS]).all(axis=1).sum())} have all six "
          f"predictors finite", flush=True)
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



class _Refuse(Exception):
    """A check that could not be performed. It is never a pass.

    Every path that cannot establish what it is supposed to establish raises this, so
    the guard has no branch that reports success over an unperformed check. Round 10
    found four fail-open shapes in the previous version, where an absent cases file
    recorded a null digest and validation then accepted with zero predictors compared,
    and where malformed JSON raised out of the guard entirely instead of refusing.
    """


EXACT_INT = 2 ** 53   # beyond this, float equality stops meaning integer equality


def _numeric(frame, col, what, *, require_numeric_dtype=False):
    """Coerce a column to numeric, refusing when a NON-NULL value fails to convert.

    A plain to_numeric(errors="coerce") turns text into NaN, and NaN then flows into a
    comparison as either a false difference or, when both sides carry text, as agreement.
    Acceptance was reachable four separate ways through that hole,
    including a non-numeric deposited response that the guard reported as identical, and
    non-numeric predictors that were accepted and then raised a TypeError downstream in
    complete_case(). A value that is absent stays absent; a value that is present and
    not a number is a refusal.
    """
    raw = frame[col]
    if getattr(raw, "ndim", 1) != 1:
        raise _Refuse(f"{what}: duplicate column label {col}")
    if require_numeric_dtype and not pd.api.types.is_numeric_dtype(raw):
        # coercible is not the same as numeric. A string-typed year compared equal here
        # and then selected zero rows in the tier filter downstream, and object-dtype
        # predictors were accepted and then raised a TypeError in complete_case().
        raise _Refuse(f"{what}: {col} is {raw.dtype}, not a numeric column")
    num = pd.to_numeric(raw, errors="coerce")
    bad = int((num.isna() & raw.notna()).sum())
    if bad:
        raise _Refuse(f"{what}: {bad} non-numeric value(s) in {col}")
    return num


def _exact_int(series, col, what):
    """Refuse values too large for float equality to mean integer equality.

    Adjacent integers above 2**53 collapse to the same float, so two different responses
    or wave identifiers compared equal and the guard then printed "EXACTLY equal". Real
    values here are convective-system counts and trajectory identifiers, both far inside
    the range, so the bound costs nothing.
    """
    big = int((series.abs() >= EXACT_INT).sum())
    if big:
        raise _Refuse(f"{what}: {big} value(s) in {col} are too large for exact "
                      f"comparison (>= 2**53)")
    return series


def _keyed(frame, what, lon_col="lon", wave_col="wave"):
    """Attach the comparison key (time, longitude to 3 dp, wave) and require it unique.

    Uniqueness is REQUIRED rather than imposed. The previous version called
    drop_duplicates on both sides, which silently collapsed conflicting duplicate rows
    before any comparison ran while the model was fitted on the original duplicated
    frame. The key is unique in the real record, measured at 0 duplicates across the
    cache, troughs_pooled.csv and eligible_troughs.csv, so requiring it costs nothing on
    a healthy input and refuses the state the old code hid.
    """
    missing = [c for c in ("time", lon_col, wave_col) if c not in frame.columns]
    if missing:
        raise _Refuse(f"{what}: missing column(s) {', '.join(missing)}")
    f = frame.copy()
    # utc=True on BOTH sides, so a timezone-aware cache and a naive anchor normalize to
    # one representation instead of raising out of the merge. These are model times in
    # UTC throughout, so treating a naive value as UTC is the correct reading.
    f["time_k"] = pd.to_datetime(f["time"], errors="coerce", utc=True)
    if f["time_k"].isna().any():
        raise _Refuse(f"{what}: {int(f['time_k'].isna().sum())} unparseable time value(s)")
    f["lon_k"] = _numeric(f, lon_col, what).round(3)
    if f["lon_k"].isna().any():
        raise _Refuse(f"{what}: {int(f['lon_k'].isna().sum())} missing longitude value(s)")
    # the wave key is validated as numeric too. A string-typed wave column otherwise
    # raised an uncaught pandas ValueError out of the merge, and a null wave was
    # accepted because only time and longitude were checked for nulls.
    f["wave_k"] = _numeric(f, wave_col, what)
    if f["wave_k"].isna().any():
        raise _Refuse(f"{what}: {int(f['wave_k'].isna().sum())} missing wave value(s)")
    dup = int(f.duplicated(KEY).sum())
    if dup:
        raise _Refuse(f"{what}: {dup} duplicate key row(s), so a one-to-one "
                      f"correspondence is undefined")
    return f


def derived_fixed_effects(time, lon):
    """The fixed-effect labels implied by a trough's own time and longitude.

    ``year`` and ``lonmonth`` are not free columns. They are exact functions of the key,
    computed this way in build_design, so a cached value that disagrees with its own row
    is wrong and can be refused with certainty rather than declared unverifiable.

    This exists because permuting these two columns in a real
    11,457-row cache, the guard accepted it, and the pooled primary term moved from
    1.013844 (0.994046 to 1.034037) to 1.002920 (0.983313 to 1.022918). They enter every
    fitted model here and the within-between model downstream, and the first version of
    this guard checked neither, because the required-column list covered the response and
    the six predictors only.
    """
    t = pd.DatetimeIndex(time)
    lonbin = np.digitize(np.asarray(lon, float), LON_EDGES)
    return (t.year.values,
            np.array([f"L{int(b)}M{int(m)}" for b, m in zip(lonbin, t.month.values)]))


def _merge_keyed(left, right, what, **kw):
    """Join on the comparison key, turning a merge-time failure into a refusal.

    ONLY the merge is wrapped. An earlier version caught ValueError across the whole
    validator, which meant a genuine programming error anywhere inside became an
    ordinary refusal and the pipeline merely rebuilt, so the guard could be broken for
    good while looking conservative. The dtype checks should make these unreachable;
    they are a refusal of last resort, never an escape.
    """
    try:
        return left.merge(right, on=KEY, **kw)
    except (pd.errors.MergeError, ValueError) as e:
        raise _Refuse(f"{what}: keys cannot be compared ({e.__class__.__name__}: {e})")


def _read_anchor(dep_dir, name):
    """Load one of the two tables build_deposit.py writes and this program does not."""
    path = os.path.join(dep_dir, name)
    if not os.path.exists(path):
        raise _Refuse(f"{name}: absent at {path}, so the cohort cannot be checked")
    try:
        f = read_design_csv(path)
    except Exception as e:
        raise _Refuse(f"{name}: unreadable ({e.__class__.__name__})")
    if f.empty:
        raise _Refuse(f"{name}: no rows")
    return f


def _check_freshness(cache, dep_dir):
    """Refuse unless the anchors on disk are the ones the cache was built against."""
    import json
    path = _freshness_path(cache)
    if not os.path.exists(path):
        raise _Refuse("no freshness record beside the cache")
    try:
        with open(path) as fh:
            rec = json.load(fh)
    except (ValueError, OSError) as e:
        raise _Refuse(f"freshness record unreadable ({e.__class__.__name__})")
    if not isinstance(rec, dict):
        raise _Refuse("freshness record is not a JSON object")
    src = rec.get("sources")
    if not isinstance(src, dict) or not src:
        raise _Refuse("freshness record carries no sources")
    out = []
    for name in ANCHORS:
        want = src.get(name)
        if not want:
            raise _Refuse(f"freshness record has no digest for {name}")
        try:
            got = _file_digest(os.path.join(dep_dir, name))
        except OSError as e:
            # present but unreadable, for example a permissions problem. The spec says
            # an anchor that cannot be read is a refusal, and without this it would
            # propagate as a crash instead.
            raise _Refuse(f"{name} unreadable ({e.__class__.__name__})")
        if got is None:
            raise _Refuse(f"{name} absent, so freshness cannot be established")
        if want != got:
            out.append(f"{name} changed since the cache was built")
    return out


def validate_design_cache(df, cache, dep_dir=None):
    """Decide whether a cached ELIGIBLE-cohort design may be reused. Returns (ok, msgs).

    ``ok`` is true only when ``msgs`` is empty. There is no third state and no argument
    by which a caller can declare a check unavailable.

    WHAT THIS ESTABLISHES. The cache describes the same cohort of trough observations as
    the deposit currently on disk, one row per deposited trough in both directions, and
    every quantity the deposit independently carries agrees with the cache.

    WHAT IT DOES NOT ESTABLISH, stated here because the previous three versions each
    claimed more in their success message than they checked. It does not establish that
    amplitude, shear or tcwv are correct, since nothing in the deposit carries them. It
    does not establish inflow_rh, box_rh or antecedent on rows outside the classified
    subsample, which is 3,099 of 11,457. And it does not detect deliberate tampering,
    which is declared out of scope rather than half-attempted: any reference value this
    program writes can be rewritten by running this program again. The guard is built
    against STALENESS, which is what has actually gone wrong in rounds 7, 8 and 10.

    The full statement is docs/DESIGN_CACHE_SPEC.md, written before this repair because
    the unit had been repaired three times and each repair introduced the next defect.
    """
    dep = dep_dir or os.path.dirname(cache) or "."
    msgs, notes = [], []
    n_cache = n_dep = n_cmp = uncovered = 0
    try:
        msgs.extend(_check_freshness(cache, dep))

        # Every column that enters a fitted model, not only the response and the six
        # predictors. year and lonmonth were omitted from this list in the first version
        # and they were permuted undetected.
        missing = [c for c in ["response", *PREDICTORS, "year", "lonmonth"]
                   if c not in df.columns]
        if missing:
            raise _Refuse(f"cache: missing modelled column(s) {', '.join(missing)}")
        if df.columns.duplicated().any():
            dupcols = sorted(set(df.columns[df.columns.duplicated()]))
            raise _Refuse(f"cache: duplicate column label(s) {', '.join(dupcols)}")
        cl = _keyed(df, "cache")
        for c in PREDICTORS:
            _numeric(cl, c, "cache", require_numeric_dtype=True)
        resp = _exact_int(_numeric(cl, "response", "cache", require_numeric_dtype=True),
                          "response", "cache")
        if not np.isfinite(resp).all():
            raise _Refuse(f"cache: {int((~np.isfinite(resp)).sum())} non-finite "
                          f"response value(s)")

        # FIXED EFFECTS. Exact functions of the row's own key, so a disagreement is
        # certain rather than merely suspicious.
        want_year, want_lonmonth = derived_fixed_effects(cl["time_k"].dt.tz_localize(None),
                                                         cl["lon"])
        off_year = int((_numeric(cl, "year", "cache",
                                 require_numeric_dtype=True).values != want_year).sum())
        if off_year:
            msgs.append(f"year: {off_year} of {len(cl)} disagree with the row's own time")
        off_lm = int((cl["lonmonth"].astype(str).values != want_lonmonth).sum())
        if off_lm:
            msgs.append(f"lonmonth: {off_lm} of {len(cl)} disagree with the row's own "
                        f"time and longitude")

        # COHORT. An OUTER join in both directions, so a cached row matching no
        # deposited trough and a deposited trough missing from the cache are both
        # findings. An inner join, which the previous version used, silently drops the
        # very rows a truncation test is about. validate= asserts the one-to-one
        # property at runtime rather than trusting the duplicate checks above.
        tl = _keyed(_read_anchor(dep, "troughs_pooled.csv"), "troughs_pooled.csv",
                    wave_col="traj_id")
        n_cache, n_dep = len(cl), len(tl)
        if "response" not in tl.columns:
            raise _Refuse("troughs_pooled.csv: missing column(s) response")
        _exact_int(cl["wave_k"], "wave", "cache")
        _exact_int(tl["wave_k"], "traj_id", "troughs_pooled.csv")
        j = _merge_keyed(cl[KEY + ["response"]], tl[KEY + ["response"]],
                         "cohort join", how="outer", suffixes=("", "_dep"),
                         indicator=True, validate="one_to_one")
        orphan = int((j["_merge"] == "left_only").sum())
        absent = int((j["_merge"] == "right_only").sum())
        if orphan:
            msgs.append(f"{orphan} cached row(s) match no deposited trough")
        if absent:
            msgs.append(f"{absent} deposited trough(s) missing from the cache")
        both = j[j["_merge"] == "both"]
        # EXACT for the response. It is a count of convective systems, deposited at
        # three decimal places, so it round-trips exactly and there is no reason to
        # allow it the 1e-3 the four-decimal environmental fields need. Under the
        # tolerance a uniform shift of 0.0005 on every response was accepted and then
        # reported as "identical".
        a = _numeric(both, "response", "cache")
        b = _numeric(both, "response_dep", "troughs_pooled.csv")
        off = int((a.values != b.values).sum())
        if off:
            msgs.append(f"response: {off} of {len(both)} differ from the deposit")
        # NO LONGITUDE VALUE COMPARISON HERE, deliberately, and the reason is recorded
        # so it is not re-added as an improvement. A comparison at TOL could never fire:
        # the key rounds longitude to three decimals, so two values sharing a key differ
        # by strictly less than 1e-3 and always pass, while any larger difference has
        # already changed the key and been caught as an orphan. It would be a check that
        # cannot fail, which is worse than no check because it draws the eye of the next
        # reader. The residual it appears to cover is irreducible against this deposit
        # and is stated in docs/DESIGN_CACHE_SPEC.md.

        # VALUES. Every quantity the deposit also carries, on every shared row. No
        # dropna, because the previous version's dropna turned an injected NaN into a
        # row that was silently not compared.
        cs = _keyed(_read_anchor(dep, "cases_pooled_700.csv"), "cases_pooled_700.csv",
                    wave_col="traj_id")
        absent_cols = [d for _, d in COMPARED if d not in cs.columns]
        if absent_cols:
            raise _Refuse(f"cases_pooled_700.csv: missing column(s) "
                          f"{', '.join(absent_cols)}")
        # The classified subsample is a STRICT SUBSET of the eligible cohort, so a cases
        # row matching no cached row means the two tables describe different records.
        # The inner join below silently drops such a row, so an incoherent cases anchor
        # carrying 40 valid rows and 1 orphan returned success. Checked before the join
        # rather than inferred from it.
        stray = int((~cs.set_index(KEY).index.isin(cl.set_index(KEY).index)).sum())
        if stray:
            msgs.append(f"cases_pooled_700.csv: {stray} row(s) match no cached trough")
        cmp_ = _merge_keyed(cl[KEY + [c for c, _ in COMPARED]],
                            cs[KEY + [d for _, d in COMPARED]], "predictor join",
                            how="inner", suffixes=("", "_dep"), validate="one_to_one")
        if cmp_.empty:
            raise _Refuse("cases_pooled_700.csv shares no row with the cache, so no "
                          "predictor was compared")
        n_cmp = len(cmp_)
        uncovered = n_cache - n_cmp
        for col, dep_col in COMPARED:
            right = dep_col if dep_col != col else col + "_dep"
            a = _numeric(cmp_, col, "cache")
            b = _numeric(cmp_, right, "cases_pooled_700.csv")
            one_sided = int((a.isna() ^ b.isna()).sum())
            if one_sided:
                msgs.append(f"{col}: {one_sided} row(s) present on one side only")
            # An infinity on either side is refused rather than compared. inf minus inf
            # is NaN, which is not greater than the tolerance, so two matching
            # infinities passed and the guard then printed that they agreed within
            # 1e-3, which is not a statement any comparison had made.
            inf = int((np.isinf(a.values) | np.isinf(b.values)).sum())
            if inf:
                msgs.append(f"{col}: {inf} infinite value(s), which cannot be compared")
            pair = a.notna() & b.notna() & np.isfinite(a.values) & np.isfinite(b.values)
            off = int(((a[pair] - b[pair]).abs() > TOL).sum())
            if off:
                msgs.append(f"{col}: {off} of {int(pair.sum())} differ from the deposit")
            neither = int((a.isna() & b.isna()).sum())
            if neither:
                notes.append(f"{col}: {neither} absent on both sides, read as agreement")
    except _Refuse as e:
        msgs.append(str(e))

    if not msgs:
        print(f"cache accepted: {n_cache} rows in one-to-one correspondence with "
              f"{n_dep} deposited troughs, response EXACTLY equal, year and lonmonth "
              f"consistent with each row's own time and longitude; "
              f"{', '.join(c for c, _ in COMPARED)} agree within {TOL:g} on the {n_cmp} "
              f"rows the classified subsample covers. NOT CHECKED: {uncovered} row(s) "
              f"have no deposited predictor counterpart and rest on the response alone, "
              f"and amplitude, shear and tcwv have no independent source in the deposit "
              f"at all, so a stale value in those three is NOT detectable either. Within the "
              f"checked cohort and the checked columns this detects stale generations; "
              f"it does not detect deliberate tampering "
              f"(docs/DESIGN_CACHE_SPEC.md)."
              + ("".join(f" NOTE: {n}." for n in notes)), flush=True)
    return (not msgs), msgs


def plan_cache_reuse(cache_path, outdir):
    """Return an eligible-cohort design to reuse, or None meaning rebuild.

    Extracted from main() so the reuse DECISION is unit-testable without the two-hour
    build behind it. None is returned, and NOTHING is raised, for four cases: no --cache
    given, an absent cache (the ordinary first run [R-CACHE-01]), an unreadable cache
    [R-CACHE-02], and a cache the guard rejects. A DataFrame is returned only when the
    guard accepts. A cache that cannot be trusted is a rebuild, never an error and never
    a silent reuse.

    float_precision on the read (via read_design_csv) is REQUIRED, not a refinement:
    pandas' default C parser is not correctly rounded, so without it the cached and the
    freshly built path produce numerically different design tables, and
    control_model_design.csv would depend on whether a cache happened to exist.
    """
    if not cache_path or not os.path.exists(cache_path):
        return None
    try:
        df = read_design_csv(cache_path)
    except Exception as e:
        print(f"cache {cache_path} unreadable ({e.__class__.__name__}); "
              f"rebuilding the design", flush=True)
        return None
    # VALIDATE against the current deposit rather than trusting that the file exists.
    # run_canonical rebuilds the deposit and then reruns this driver from the cache, and
    # the fresh OUTPUT mtimes then satisfy the checker, so a stale design could ride
    # through a green run. That was the round-7 blocker, and rounds 8 and 10 each
    # defeated the repair for it. See docs/DESIGN_CACHE_SPEC.md.
    ok, msgs = validate_design_cache(df, cache_path, outdir)
    if not ok:
        print(f"cache {cache_path} REJECTED ({'; '.join(msgs)}); "
              f"rebuilding the design", flush=True)
        return None
    print(f"loaded cached eligible design, {len(df)} rows, from {cache_path}", flush=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csct", default="data/original/csct/csct_africa_cs245.nc")
    ap.add_argument("--cache", default=None,
                    help="CSV cache of the design matrix; built and written if absent")
    ap.add_argument("--outdir", default="deposit")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    pooled_years = parse_years(TIERS["pooled"])
    # The cache holds the ELIGIBLE cohort, so both deposited tables this script owns
    # (eligible_troughs.csv and control_model_design.csv) are derived from it below on
    # either path. The previous version cached the POST-filter design, could not
    # reproduce the eligible table from it, and therefore forced a full rebuild whenever
    # eligible_troughs.csv was missing. That branch is gone rather than repaired.
    df = plan_cache_reuse(a.cache, a.outdir)
    if df is None:
        df = build_design(pooled_years, a.csct)
        if a.cache:
            df.to_csv(a.cache, index=False)
            write_freshness_record(a.cache, a.outdir)
            print(f"wrote {a.cache} and {_freshness_path(a.cache)}", flush=True)

    # R4 source (implementation fold): the wave-unit estimand's cohort is every
    # eligible trough with only the missing-H rule applied, NOT the model's six-way
    # complete-case cohort, so this table is written from the eligible frame.
    df[["wave", "time", "lon", "response", "inflow_rh"]].to_csv(
        os.path.join(a.outdir, "eligible_troughs.csv"), index=False,
        float_format="%.6f")
    df = complete_case(df)
    # The published design table, consumed by scripts/within_between_model.py. It is
    # written on every run, cached or not, and it is no longer the cache itself.
    df.to_csv(os.path.join(a.outdir, "control_model_design.csv"), index=False)
    print(f"analyzable troughs (finite predictors): {len(df)}; wrote "
          f"{a.outdir}/eligible_troughs.csv and {a.outdir}/control_model_design.csv",
          flush=True)

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

    # R6 (REPAIR_SPEC.md): overdispersion sensitivities on the primary specification,
    # identical design matrix and fixed effects, cluster-robust on the wave, reported
    # as IRR per pooled SD beside the Poisson values
    prim_terms = ["inflow_rh", "antecedent", "amplitude", "shear"]
    rhs_prim = " + ".join(f"z_{c}" for c in prim_terms) + " + C(lonmonth) + C(year)"
    sens_rows = []
    pois = smf.glm(f"response ~ {rhs_prim}", data=d,
                   family=sm.families.Poisson()).fit(
        cov_type="cluster", cov_kwds={"groups": d["wave"].values})
    # a quasi-Poisson variant (scale="X2") was removed: under cluster-robust
    # covariance the scale never enters the standard errors, so its rows were
    # byte-identical to the Poisson-cluster rows and reported nothing; NB2 below is
    # the genuine dispersion sensitivity
    try:
        nb = smf.negativebinomial(f"response ~ {rhs_prim}", data=d).fit(
            cov_type="cluster", cov_kwds={"groups": d["wave"].values},
            maxiter=200, disp=0)
        nb_ok = getattr(nb.mle_retvals, "get", lambda *a: True)("converged", True) \
            if hasattr(nb, "mle_retvals") else True
    except Exception as e:                                    # report, never silent
        nb, nb_ok = None, False
        print(f"negative binomial fit failed: {e}", flush=True)
    for model_name, res, note in (("poisson_cluster", pois, "primary"),
                                  ("negbin2_cluster", nb,
                                   "NB2 log link" if nb_ok else "DID NOT CONVERGE")):
        if res is None:
            sens_rows.append(dict(model=model_name, predictor="inflow_rh",
                                  coef=np.nan, irr=np.nan, irr_lo=np.nan,
                                  irr_hi=np.nan, pvalue=np.nan, note=note))
            continue
        for c in prim_terms:
            ci_ = res.conf_int().loc[f"z_{c}"]
            sens_rows.append(dict(
                model=model_name, predictor=c, coef=float(res.params[f"z_{c}"]),
                irr=float(np.exp(res.params[f"z_{c}"])),
                irr_lo=float(np.exp(ci_[0])), irr_hi=float(np.exp(ci_[1])),
                pvalue=float(res.pvalues[f"z_{c}"]), note=note))
    pd.DataFrame(sens_rows).to_csv(
        os.path.join(a.outdir, "control_model_sensitivities.csv"), index=False,
        float_format="%.6f")
    disp = float(d["response"].var() / d["response"].mean())
    lines.append(f"\nOVERDISPERSION: response variance/mean = {disp:.2f}; "
                 "NB2 sensitivity in control_model_sensitivities.csv")

    summary = "\n".join(lines)
    print(summary)
    with open(os.path.join(a.outdir, "control_model_summary.txt"), "w") as f:
        f.write("Continuous-response control model for the moisture-conditioning claim.\n")
        f.write("Poisson GLM, cluster-robust SE on the wave; predictors standardized "
                "to pooled unit variance; longitude-month and year fixed effects.\n")
        f.write(summary + "\n")
    pd.DataFrame(rows).to_csv(
        os.path.join(a.outdir, "control_model_coeffs.csv"), index=False,
        float_format="%.6f")
    print(f"\nwrote {a.outdir}/control_model_summary.txt and control_model_coeffs.csv")


if __name__ == "__main__":
    main()
