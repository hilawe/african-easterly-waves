"""The three wave-level estimands (REPAIR_SPEC.md R4, implementation fold).

The single wave-level contrast of the original design let the 529 crossover waves sit
on both sides of the comparison. Three prespecified estimands replace it, each
capturing one named source of variation:

1. ``wave_unit_contrast`` -- one response and one environment value per physical wave
   from the FULL eligible-trough table (only the missing-H rule drops waves),
   wave-response terciles within longitude-month cells (minimum 10 waves per cell),
   top minus bottom tercile of the wave environment means, waves weighted equally.
2. ``paired_contrast`` -- for mixed waves, the within-wave difference of class means.
   Class presence comes from the UNFILTERED labels (a wave with an
   all-missing-H class is still mixed; it drops from the estimate as attrition, never
   by silent reclassification). Conditional on being a mixed wave.
3. ``between_contrast`` -- active-only against quiet-only waves (label presence,
   unfiltered) on cell-standardized environment values, with cells carried from the
   full eligible wave table, the same 10-wave cell minimum, and the standardization
   recomputed inside every bootstrap replicate.

Uncertainty everywhere: the wave-resampling bootstrap (percentile intervals, class
thresholds fixed at their point estimates, disclosed) AND the two-stage
season-then-wave bootstrap as the prespecified sensitivity, for all three estimands.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LON_EDGES = np.arange(-30.0, 41.0, 10.0)
MIN_WAVES_PER_CELL = 10
N_BOOT = 20_000


def _cell_of(mean_lon, month):
    lonbin = np.digitize(np.asarray(mean_lon, dtype=float), LON_EDGES)
    return np.array([f"L{int(b)}M{int(m)}" for b, m in zip(lonbin, month)])


def wave_table(df):
    """Per-wave table from a per-trough frame with traj_id, time, lon, response, H.

    The wave's response is the mean per-observation forward count, its environment
    value the mean of finite H over its observations (NaN when none), its cell the
    10-degree bin of its arithmetic mean longitude by the calendar month of its
    median observation time, its year the calendar year.
    """
    t = pd.to_datetime(df["time"])
    g = pd.DataFrame({
        "traj_id": df["traj_id"].values,
        "lon": pd.to_numeric(df["lon"]).values,
        "response": pd.to_numeric(df["response"]).values,
        "H": pd.to_numeric(df["H"]).values,
        "t": t.values,
    }).groupby("traj_id")
    out = pd.DataFrame({
        "resp": g["response"].mean(),
        "H": g["H"].mean(),          # nanmean semantics via pandas (skipna)
        "mean_lon": g["lon"].mean(),
        "t_med": g["t"].median(),
        "n_obs": g.size(),
    })
    out["month"] = out["t_med"].dt.month
    out["year"] = out["t_med"].dt.year
    out["cell"] = _cell_of(out["mean_lon"].values, out["month"].values)
    return out.reset_index()


def _tercile_labels(values):
    """Top/bottom tercile labels with the R3 boundary rule (equal-to-threshold joins
    the outer class); middle stays unlabeled."""
    lo, hi = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
    lab = np.full(values.size, "mid", dtype=object)
    lab[values <= lo] = "bottom"
    lab[values >= hi] = "top"
    return lab


def _both_cis(stat_by_idx, years, rng, n_boot):
    """(wave-resample CI, season-then-wave CI) for a statistic over wave indices."""
    n = years.size
    stats = np.empty(n_boot)
    for b in range(n_boot):
        stats[b] = stat_by_idx(rng.integers(0, n, size=n))
    ok = np.isfinite(stats)
    lo, hi = np.percentile(stats[ok], [2.5, 97.5])
    uyears = np.unique(years)
    by_year = {y: np.where(years == y)[0] for y in uyears}
    for b in range(n_boot):
        ys = rng.choice(uyears, size=uyears.size, replace=True)
        idx = np.concatenate([
            by_year[y][rng.integers(0, by_year[y].size, size=by_year[y].size)]
            for y in ys])
        stats[b] = stat_by_idx(idx)
    ok = np.isfinite(stats)
    slo, shi = np.percentile(stats[ok], [2.5, 97.5])
    return (float(lo), float(hi)), (float(slo), float(shi))


def wave_unit_contrast(waves, rng, n_boot=N_BOOT):
    """Estimand 1 on the full eligible wave table (missing-H drop only)."""
    w = waves.dropna(subset=["H"]).copy()
    dropped_h = int(len(waves) - len(w))
    cell_sizes = w.groupby("cell")["traj_id"].size()
    keep = cell_sizes[cell_sizes >= MIN_WAVES_PER_CELL].index
    dropped_cell = int(len(w) - w["cell"].isin(keep).sum())
    w = w[w["cell"].isin(keep)].reset_index(drop=True)
    lab = np.full(len(w), "mid", dtype=object)
    for cell, idx in w.groupby("cell").groups.items():
        idx = np.asarray(idx)
        lab[idx] = _tercile_labels(w["resp"].values[idx])
    top = lab == "top"
    bot = lab == "bottom"
    H = w["H"].values

    def stat(idx):
        t_, b_ = top[idx], bot[idx]
        if not t_.any() or not b_.any():
            return np.nan
        return H[idx][t_].mean() - H[idx][b_].mean()

    point = float(H[top].mean() - H[bot].mean())
    (lo, hi), (slo, shi) = _both_cis(stat, w["year"].values, rng, n_boot)
    return dict(estimand="wave_unit", diff=point, ci_lo=lo, ci_hi=hi,
                sens_season_lo=slo, sens_season_hi=shi,
                n_top=int(top.sum()), n_bottom=int(bot.sum()),
                n_waves=int(len(w)), n_dropped_missing_H=dropped_h,
                n_dropped_small_cell=dropped_cell)


def _per_wave_classes(cases, value_col):
    """Presence and class means with presence decided BEFORE any missing-H drop."""
    presence = (cases.groupby(["traj_id", "label"]).size().unstack(fill_value=0))
    means = (cases.dropna(subset=[value_col])
             .groupby(["traj_id", "label"])[value_col].mean().unstack())
    means = means.reindex(presence.index)
    t = pd.to_datetime(cases["time"])
    yr = (pd.DataFrame({"traj_id": cases["traj_id"].values, "t": t.values})
          .groupby("traj_id")["t"].median().dt.year)
    return presence, means, yr.reindex(presence.index)


def paired_contrast(cases, value_col="rh_m72", rng=None, n_boot=N_BOOT,
                    min_per_class=1):
    """Estimand 2. Mixed waves (unfiltered label presence); pairs whose class means
    are unavailable after the missing-H rule are attrition, not reclassification."""
    presence, means, years = _per_wave_classes(cases, value_col)
    mixed = ((presence.get("MCS-active", 0) >= min_per_class)
             & (presence.get("MCS-quiet", 0) >= min_per_class))
    d_all = (means["MCS-active"] - means["MCS-quiet"])[mixed]
    d = d_all.dropna()
    yrs = years.loc[d.index].values
    dv = d.values
    point = float(dv.mean())
    (lo, hi), (slo, shi) = _both_cis(lambda idx: dv[idx].mean(), yrs, rng, n_boot)
    return dict(estimand=f"paired_min{min_per_class}", diff=point, ci_lo=lo, ci_hi=hi,
                sens_season_lo=slo, sens_season_hi=shi, n_waves=int(dv.size),
                n_mixed_unusable=int(mixed.sum() - dv.size))


def between_contrast(cases, waves_full, value_col="rh_m72", rng=None, n_boot=N_BOOT):
    """Estimand 3. Active-only against quiet-only (unfiltered label presence),
    cell-standardized on cells from the FULL eligible wave table, the 10-wave cell
    minimum enforced, standardization recomputed inside every replicate."""
    presence, means, years = _per_wave_classes(cases, value_col)
    act = (presence.get("MCS-active", 0) > 0) & (presence.get("MCS-quiet", 0) == 0)
    qui = (presence.get("MCS-quiet", 0) > 0) & (presence.get("MCS-active", 0) == 0)
    H = pd.Series(np.nan, index=presence.index)
    H[act] = means.loc[act, "MCS-active"]
    H[qui] = means.loc[qui, "MCS-quiet"]
    cells = waves_full.set_index("traj_id")["cell"]
    tab = pd.DataFrame({"H": H, "act": act, "qui": qui,
                        "year": years}).join(cells, how="left")
    tab = tab[(tab["act"] | tab["qui"]) & tab["H"].notna() & tab["cell"].notna()]
    csize = tab.groupby("cell")["H"].size()
    tab = tab[tab["cell"].isin(csize[csize >= MIN_WAVES_PER_CELL].index)]
    tab = tab.reset_index()
    if tab.empty or tab["act"].all() or not tab["act"].any():
        return dict(estimand="between_only", diff=np.nan, ci_lo=np.nan,
                    ci_hi=np.nan, sens_season_lo=np.nan, sens_season_hi=np.nan,
                    n_active_only=int(tab["act"].sum()) if not tab.empty else 0,
                    n_quiet_only=int((~tab["act"]).sum()) if not tab.empty else 0)
    Hv = tab["H"].values
    is_a = tab["act"].values
    cell_codes = pd.Categorical(tab["cell"]).codes
    yrs = tab["year"].values

    def stat(idx):
        h, a, c = Hv[idx], is_a[idx], cell_codes[idx]
        if not a.any() or a.all():
            return np.nan
        # recompute the cell standardization inside the replicate
        dfr = pd.DataFrame({"h": h, "c": c})
        h_std = dfr["h"] - dfr.groupby("c")["h"].transform("mean")
        return h_std[a].mean() - h_std[~a].mean()

    point = float(stat(np.arange(Hv.size)))
    (lo, hi), (slo, shi) = _both_cis(stat, yrs, rng, n_boot)
    return dict(estimand="between_only", diff=point, ci_lo=lo, ci_hi=hi,
                sens_season_lo=slo, sens_season_hi=shi,
                n_active_only=int(is_a.sum()), n_quiet_only=int((~is_a).sum()))
