#!/usr/bin/env python
"""Merge every reportable output into one audited registry (implementation-review
fold of the R6 audit scope).

deposit/canonical_numbers.csv carries the deposit driver's statistics, but the R1
figure tests, the R4 wave estimands, the control-model ladder/panel/sensitivities,
and the lead-lag gradient live in their own files. This builder maps them all into
deposit/canonical_registry.csv under the same (tier, level, statistic, time_rel_h)
key schema, so the prose-to-canonical audit covers every number the manuscript can
cite. Rerun after any driver; the checker audits against the registry when present.
"""

import hashlib
import json
import os

import numpy as np
import pandas as pd

DEP = "deposit"


def rows_from_canonical():
    return pd.read_csv(f"{DEP}/canonical_numbers.csv")


def rows_from_wave_estimands():
    p = f"{DEP}/wave_estimands.csv"
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    out = []
    for _, r in d.iterrows():
        row = dict(tier="pooled", level=700.0,
                   statistic=f"we_{r['estimand']}", time_rel_h=-72.0,
                   unit="%", kind="contrast", diff=r["diff"],
                   ci_lo=r["ci_lo"], ci_hi=r["ci_hi"],
                   sens_season_lo=r.get("sens_season_lo"),
                   sens_season_hi=r.get("sens_season_hi"))
        # cohort counts ride on the same row so prose can cite them under one key
        for c in ("n_top", "n_bottom", "n_waves", "n_active_only", "n_quiet_only"):
            if c in r and pd.notna(r[c]):
                row[c] = r[c]
        out.append(row)
    return pd.DataFrame(out)


def rows_from_predictor_corr():
    p = f"{DEP}/control_model_predictor_corr.csv"
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p, index_col=0)
    out = []
    for a, b in (("inflow_rh", "shear"), ("inflow_rh", "antecedent"),
                 ("inflow_rh", "tcwv")):
        out.append(dict(tier="pooled", level=700.0, statistic=f"cm_corr_{a}_{b}",
                        time_rel_h=-72.0, unit="r", kind="model",
                        diff=float(d.loc[a, b])))
    return pd.DataFrame(out)


def rows_from_response_sensitivity():
    # the repaired response-definition sensitivity (7 variants of box/window/split);
    # the baseline row reproduces lagrangian_rh -72 h by construction
    p = f"{DEP}/response_sensitivity.csv"
    if not os.path.exists(p):
        return None
    out = []
    for _, r in pd.read_csv(p).iterrows():
        out.append(dict(tier="pooled", level=700.0,
                        statistic=f"respsens_{r['variant']}", time_rel_h=-72.0,
                        unit="%", kind="sensitivity", diff=r["diff"],
                        ci_lo=r["ci_lo"], ci_hi=r["ci_hi"],
                        significant=bool(r["significant"]),
                        n_quiet=r["n_quiet"], n_active=r["n_active"]))
    return pd.DataFrame(out)


def rows_from_tierule_sensitivity():
    # deduplication tie-rule sensitivity: the same pipeline rebuilt with the opposite
    # co-longest tie-break (largest original id instead of smallest). 614 of 2,972
    # merged components carry a tie, so this is a real degree of freedom rather than a
    # formality. Only the headline rows are registered, because the point of the
    # sensitivity in the paper is that the headline does not move; the full 83-row
    # comparison stays in deposit/tierule_sensitivity.csv.
    p = f"{DEP}/tierule_sensitivity.csv"
    if not os.path.exists(p):
        return None
    # -72 h only: that is the lead the paper quotes, and the source carries every lag,
    # so keying on the statistic alone produced duplicate registry keys (caught by the
    # registry's own duplicate gate).
    keep = {("lagrangian_rh", 700.0), ("lagrangian_rh_wave_level", 700.0)}
    out = []
    for _, r in pd.read_csv(p).iterrows():
        if (r["statistic"], float(r["level"])) not in keep:
            continue
        if float(r["time_rel_h"]) != -72.0:
            continue
        out.append(dict(tier="pooled", level=float(r["level"]),
                        statistic=f"tiesens_{r['statistic']}",
                        time_rel_h=float(r["time_rel_h"]),
                        unit="%", kind="sensitivity", diff=r["diff_alt"],
                        ci_lo=r["ci_lo_alt"], ci_hi=r["ci_hi_alt"],
                        significant=bool(r["significant_alt"])))
    return pd.DataFrame(out) if out else None


def rows_from_catalog_sensitivity():
    # catalog sensitivity: the pooled tier rebuilt with the response and the antecedent
    # count taken from the catalog that pools every satellite position, on the same
    # eligible troughs. The rows registered are those the paper would quote beside its
    # own, and the full comparison stays in deposit/catalog_sensitivity.csv.
    p = f"{DEP}/catalog_sensitivity.csv"
    if not os.path.exists(p):
        return None
    keep = {("lagrangian_rh", 700.0, -72.0), ("lagrangian_rh_wave_level", 700.0, -72.0),
            ("lagrangian_rh_epoch_1983_1993", 700.0, -72.0),
            ("lagrangian_rh_epoch_1994_2007", 700.0, -72.0),
            ("antecedent_convection", 700.0, -48.0)}
    out = []
    for _, r in pd.read_csv(p).iterrows():
        if (r["statistic"], float(r["level"]), float(r["time_rel_h"])) not in keep:
            continue
        out.append(dict(tier="pooled", level=float(r["level"]),
                        statistic=f"catsens_{r['statistic']}",
                        time_rel_h=float(r["time_rel_h"]),
                        unit=r["unit"], kind="sensitivity", diff=r["diff_alt"],
                        ci_lo=r["ci_lo_alt"], ci_hi=r["ci_hi_alt"],
                        significant=bool(r["significant_alt"])))
    return pd.DataFrame(out) if out else None


OBSERVED_RECORD = "scripts/met_observed_slots_record.json"
OBSERVED_SLOTS = "scripts/met_observed_slots.csv"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rows_from_observation_rule():
    """The counts behind the availability criterion, which the text states.

    The slot counts come from the record written with the calendar of slots, and the
    record must be the one of the calendar on disk, by hash. The share of the
    calendar-eligible troughs that the criterion removes is derived per tier from the
    deposit's own count rows, so it cannot disagree with them.
    """
    p = f"{DEP}/canonical_numbers.csv"
    if not (os.path.exists(p) and os.path.exists(OBSERVED_RECORD) and os.path.exists(OBSERVED_SLOTS)):
        return None
    d = pd.read_csv(p)
    if not (d.statistic == "n_window_unobserved").any():
        return None                       # a deposit built before the criterion states no such count
    rec = json.load(open(OBSERVED_RECORD))
    if rec["out_sha256"] != _sha256(OBSERVED_SLOTS):
        raise SystemExit(f"{OBSERVED_RECORD} is not the record of {OBSERVED_SLOTS}")
    counts = dict(obs_slots_total=rec["slots_of_the_seasons"], obs_slots_with_row=rec["slots_with_a_row"],
                  obs_slots_no_row=rec["slots_without_a_row"], obs_slots_partial=len(rec["slots_taken_as_partial"]),
                  obs_slots_passing=rec["slots_observed"], obs_slots_failing=rec["slots_not_observed"])
    if counts["obs_slots_with_row"] + counts["obs_slots_no_row"] != counts["obs_slots_total"] \
            or counts["obs_slots_passing"] + counts["obs_slots_failing"] != counts["obs_slots_total"] \
            or counts["obs_slots_no_row"] + counts["obs_slots_partial"] != counts["obs_slots_failing"]:
        raise SystemExit(f"the slot counts of {OBSERVED_RECORD} do not add up")
    out = [dict(tier="pooled", level=np.nan, statistic=k, time_rel_h=np.nan, unit="count", kind="point", diff=float(v))
           for k, v in counts.items()]
    out.append(dict(tier="pooled", level=np.nan, statistic="obs_partial_fraction", time_rel_h=np.nan, unit="fraction",
                    kind="config", diff=float(rec["partial_fraction"])))
    for tier in ("dev", "heldout", "pooled"):
        got = {}
        for stat in ("n_corridor_troughs", "n_window_unobserved"):
            m = (d.tier == tier) & (d.statistic == stat)
            if int(m.sum()) != 1:
                raise SystemExit(f"{p} states {stat} of tier {tier} {int(m.sum())} times")
            got[stat] = float(d["diff"][m].iloc[0])
        total = got["n_corridor_troughs"] + got["n_window_unobserved"]
        out.append(dict(tier=tier, level=np.nan, statistic="n_calendar_eligible", time_rel_h=np.nan, unit="count",
                        kind="point", diff=total))
        out.append(dict(tier=tier, level=np.nan, statistic="pct_window_unobserved", time_rel_h=np.nan, unit="%",
                        kind="point", diff=100.0 * got["n_window_unobserved"] / total))
    return pd.DataFrame(out)


SCALE_MEASURES = (("inflow RH at -72 h, 700 hPa", "rh", -72.0, "%"), ("fixed box RH at -24 h, 700 hPa", "box", -24.0, "%"),
                  ("antecedent count", "antecedent", -48.0, "systems"))
SCALE_LABELS = (("canonical", "canonical"), ("year-matched", "yearmatched"), ("era", "era"), ("every position pooled", "allpos"),
                ("every position pooled, year-matched", "allpos_yearmatched"))
# each sample with the labelings registered for it, the deposit's own classes first
SCALE_SAMPLES = (("whole sample", "", ("canonical", "yearmatched", "era", "allpos", "allpos_yearmatched")),
                 ("30 W to 0", "_west", ("canonical",)), ("0 to 10 E", "_middle", ("canonical",)), ("10 E to 40 E", "_east", ("canonical",)),
                 ("1983 to 1993", "_1983_1993", ("canonical", "allpos")), ("1994 to 2007", "_1994_2007", ("canonical", "allpos")),
                 ("1983 to 1997", "_1983_1997", ("canonical", "era", "allpos")), ("1998 to 2007", "_1998_2007", ("canonical", "era", "allpos")))
SCALE_QUANTITIES = ("total", "years", "cells", "waves", "passages", "years and cells", "waves and passages")
_H, _BOX, _BG = "inflow RH at -72 h, 700 hPa", "fixed box RH at -24 h, 700 hPa", "every eligible trough"
# the differences between two rows that the text quotes, each read from its own paired row of the record
SCALE_DIFFERENCES = (("canonical_less_allpos", "whole sample", (_H, "canonical"), (_H, "every position pooled")),
                     ("rh_less_box_canonical", "whole sample", (_H, "canonical"), (_BOX, "canonical")),
                     ("rh_less_box_yearmatched", "whole sample", (_H, "year-matched"), (_BOX, "year-matched")),
                     ("epochs_canonical", "1983 to 1993 less 1994 to 2007", (_H, "canonical"), (_H, "canonical")),
                     ("epochs_allpos", "1983 to 1993 less 1994 to 2007", (_H, "every position pooled"), (_H, "every position pooled")),
                     ("eras_canonical", "1983 to 1997 less 1998 to 2007", (_H, "canonical"), (_H, "canonical")),
                     ("eras_era", "1983 to 1997 less 1998 to 2007", (_H, "era"), (_H, "era")),
                     ("eras_allpos", "1983 to 1997 less 1998 to 2007", (_H, "every position pooled"), (_H, "every position pooled")))


def rows_from_scale_split():
    """The class contrasts split by scale, for the measures and samples the text quotes.

    A total carries the declared wave-cluster interval. A part carries the interval with
    the year, cell and wave means held at their sample values, which is the one the
    analysis declares for parts, and the declared interval of a part stays in
    deposit/scale_split.csv. The whole sample is registered under every labeling and the
    sectors and periods under those named for them. A difference between two rows comes
    from its own paired row of deposit/scale_split_record.json, never from two intervals.
    """
    p = f"{DEP}/scale_split.csv"
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    d = d[d.background == "every eligible trough"]
    out = []
    for measure, mslug, hour, unit in SCALE_MEASURES:
        for sample, sslug, wanted in SCALE_SAMPLES:
            for label, lslug in SCALE_LABELS:
                if lslug not in wanted or (sslug and mslug != "rh"):
                    continue
                for q in SCALE_QUANTITIES:
                    m = (d["sample"] == sample) & (d.measure == measure) & (d["labels"] == label) & (d.quantity == q)
                    if int(m.sum()) > 1:
                        raise SystemExit(f"{p} holds {sample}, {measure}, {label}, {q} {int(m.sum())} times")
                    if not m.any():
                        if label == "canonical":
                            raise SystemExit(f"{p} lacks {sample}, {measure}, {label}, {q}")
                        continue          # the labelings of a second catalog are absent when none was named
                    r = d[m].iloc[0]
                    lo, hi = (r["low"], r["high"]) if q == "total" else (r["held_low"], r["held_high"])
                    out.append(dict(tier="pooled", level=700.0, statistic=f"ss_{mslug}_{lslug}_{q.replace(' ', '_')}{sslug}",
                                    time_rel_h=hour, unit=unit, kind="split", n_quiet=r["n_quiet"], n_active=r["n_active"],
                                    diff=r["diff"], ci_lo=lo, ci_hi=hi, significant=bool(lo > 0 or hi < 0),
                                    n_boot=r["draws_used"]))
    q = f"{DEP}/scale_split_record.json"
    if not os.path.exists(q):
        raise SystemExit(f"{p} is present and {q} is not")
    pairs = json.load(open(q))["paired"]
    for name, sample, first, second in SCALE_DIFFERENCES:
        for quantity in SCALE_QUANTITIES:
            hit = [x for x in pairs if x["sample"] == sample and x["quantity"] == quantity
                   and tuple(x["first"]) == first + (_BG,) and tuple(x["second"]) == second + (_BG,)]
            if len(hit) > 1:
                raise SystemExit(f"{q} holds the difference {name}, {quantity} {len(hit)} times")
            if not hit:
                if "every position pooled" in (first[1], second[1]):
                    continue              # absent when no second catalog was named
                raise SystemExit(f"{q} lacks the difference {name}, {quantity}")
            x = hit[0]
            lo, hi = (x["low"], x["high"]) if quantity == "total" else (x["held_low"], x["held_high"])
            out.append(dict(tier="pooled", level=700.0, statistic=f"ssd_{name}_{quantity.replace(' ', '_')}", time_rel_h=-72.0,
                            unit="%", kind="split", diff=x["diff"], ci_lo=lo, ci_hi=hi, significant=bool(lo > 0 or hi < 0)))
    return pd.DataFrame(out)


def rows_from_yearstrat_sensitivity():
    # temporal-stratification sensitivity: the same eligible sample re-terciled within
    # year x month x longitude, so the classes are matched in year (the full-access
    # review's largest methodological finding; the canonical month x lon split leaves
    # a ~4-year mean-date gap between the classes)
    p = f"{DEP}/yearstrat_sensitivity.csv"
    if not os.path.exists(p):
        return None
    out = []
    for _, r in pd.read_csv(p).iterrows():
        out.append(dict(tier="pooled", level=700.0,
                        statistic=f"yearstrat_{r['split']}", time_rel_h=-72.0,
                        unit="%", kind="sensitivity", diff=r["diff"],
                        ci_lo=r["ci_lo"], ci_hi=r["ci_hi"],
                        significant=bool(r["significant"]),
                        n_quiet=r["n_quiet"], n_active=r["n_active"],
                        n_total=int(r["n_quiet"]) + int(r["n_active"]),
                        year_gap=r["year_gap"],
                        **{k: r[k] for k in ("n_cells", "n_cells_kept", "n_in_kept_cells",
                                             "median_cell", "smallest_cell") if k in r.index}))
    return pd.DataFrame(out)


def rows_from_validation_constants():
    # frozen validation results (tracker agreement, reproduction targets, alignment
    # diagnostics); a curated, VERSION-CONTROLLED table under scripts/, not a rerun
    # output, so it survives a deposit rebuild. The tracked copy is authoritative; a
    # deposit-local copy is a fallback only.
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (os.path.join(here, "..", "scripts", "validation_constants.csv"),
              f"{DEP}/validation_constants.csv"):
        if os.path.exists(p):
            return pd.read_csv(p)
    return None


def rows_from_control_model():
    out = []
    p = f"{DEP}/control_model_ladder.csv"
    if os.path.exists(p):
        for _, r in pd.read_csv(p).iterrows():
            slug = (r["step"].replace(" ", "_").replace("+", "plus")
                    .replace("(", "").replace(")", ""))
            out.append(dict(tier="pooled", level=700.0,
                            statistic=f"cm_ladder_{slug}", time_rel_h=-72.0,
                            unit="IRR/SD", kind="model", irr=r["irr"],
                            irr_lo=r["irr_lo"], irr_hi=r["irr_hi"],
                            pvalue=r["pvalue"]))
    # the moisture term of the two specifications on each tier, which the text quotes
    # beside the pooled value, with the troughs and waves each fit used
    p = f"{DEP}/control_model_coeffs.csv"
    if os.path.exists(p):
        for _, r in pd.read_csv(p).iterrows():
            spec = r["spec"].replace("+", "_plus_")
            out.append(dict(tier=r["tier"], level=700.0,
                            statistic=f"cm_coef_{spec}_{r['moisture']}", time_rel_h=-72.0,
                            unit="IRR/SD", kind="model", irr=r["irr"],
                            irr_lo=r["irr_lo"], irr_hi=r["irr_hi"],
                            pvalue=r["pvalue"], n_obs=r["n"], n_waves=r["waves"]))
    p = f"{DEP}/control_model_panel.csv"
    if os.path.exists(p):
        for _, r in pd.read_csv(p).iterrows():
            out.append(dict(tier="pooled", level=700.0,
                            statistic=f"cm_panel_{r['predictor']}",
                            time_rel_h=-72.0, unit="IRR/SD", kind="model",
                            irr=r["irr"], irr_lo=r["irr_lo"], irr_hi=r["irr_hi"],
                            pvalue=r["pvalue"]))
    p = f"{DEP}/control_model_sensitivities.csv"
    if os.path.exists(p):
        for _, r in pd.read_csv(p).iterrows():
            out.append(dict(tier="pooled", level=700.0,
                            statistic=f"cm_sens_{r['model']}_{r['predictor']}",
                            time_rel_h=-72.0, unit="IRR/SD", kind="model",
                            irr=r["irr"], irr_lo=r["irr_lo"], irr_hi=r["irr_hi"],
                            pvalue=r["pvalue"]))
    return pd.DataFrame(out) if out else None


def rows_from_within_between():
    # the common-sample hybrid decomposition, with one model, one cohort and a direct
    # test of between-wave against within-wave, which the exclusive/mixed cohort
    # summaries cannot provide
    p = f"{DEP}/within_between_model.csv"
    if not os.path.exists(p):
        return None
    out = []
    for _, r in pd.read_csv(p).iterrows():
        row = dict(tier="pooled", level=700.0,
                   statistic=f"wb_{r['component']}", time_rel_h=-72.0,
                   unit="IRR/SD", kind="model", pvalue=r["pvalue"],
                   n_waves=r["n_waves"], n_obs=r["n_obs"])
        if pd.notna(r.get("irr")):
            row.update(irr=r["irr"], irr_lo=r["irr_lo"], irr_hi=r["irr_hi"])
        if pd.notna(r.get("chi2")):
            row["diff"] = r["chi2"]
        out.append(row)
    return pd.DataFrame(out)


def rows_from_bootstrap_config():
    """One canonical row for the replicate count the canonical contrasts actually used.

    The manuscript states this number in prose. Without a row to tag it against, the
    claim was unverifiable and drifted from the code once already (the text
    said 20,000 while the driver ran 2,000). Derived from the n_boot column the deposit
    now records, and deliberately fails when producers disagree rather than picking one.
    """
    p = f"{DEP}/canonical_numbers.csv"
    if not os.path.exists(p):
        return None
    vals = set()
    # EVERY producer that records a count, not just the deposit: a disagreement between
    # them is exactly the drift this row exists to make impossible.
    for src in ("canonical_numbers.csv", "wave_estimands.csv",
                "response_sensitivity.csv", "leadlag_stats.csv"):
        q = f"{DEP}/{src}"
        if not os.path.exists(q):
            continue
        dd = pd.read_csv(q)
        if "n_boot" in dd.columns:
            vals |= set(dd["n_boot"].dropna().astype(int))
        elif "statistic" in dd.columns and "value" in dd.columns:
            hit = dd[dd["statistic"] == "leadlag_n_boot"]
            vals |= set(hit["value"].dropna().astype(int))
    d = pd.read_csv(p)
    if "n_boot" not in d.columns:
        return None
    vals = sorted(vals)
    if len(vals) != 1:
        raise SystemExit(f"producers used MORE THAN ONE replicate count {vals}; "
                         f"the manuscript cannot state a single number")
    return pd.DataFrame([dict(tier="config", level=np.nan,
                              statistic="bootstrap_replicates", time_rel_h=np.nan,
                              unit="count", kind="config", diff=float(vals[0]),
                              note="replicates in every canonical cluster bootstrap")])


def rows_from_fig3_descriptive():
    # Fig 3 carries no randomization inference (REPAIR_SPEC.md, 2026-07-24 note:
    # the R1 support assertion is unsatisfiable for the regional CT catalog), so
    # its registry rows are descriptive peaks only
    p = f"{DEP}/fig3_descriptive_stats.csv"
    if not os.path.exists(p):
        return None
    out = []
    for _, r in pd.read_csv(p).iterrows():
        out.append(dict(tier="pooled", level=np.nan,
                        statistic=f"fig3_desc_{r['subset']}", time_rel_h=np.nan,
                        unit="counts", kind="descriptive",
                        peak_count=r["peak_count"],
                        peak_rel_lon=r["peak_rel_lon"], n_events=r["n_events"]))
        # shape statistics on the self-normalized profile. The born-deep set is a
        # subset of all first detections, so only these are comparable between the
        # two curves; the raw peaks above scale with sample size.
        for col, unit in (("peak_frac_pct", "%"), ("share_core_pct", "%"),
                          ("share_west_pct", "%"), ("centroid_rel_lon", "deg")):
            if col in r.index:
                out.append(dict(tier="pooled", level=np.nan,
                                statistic=f"fig3_{col}_{r['subset']}",
                                time_rel_h=np.nan, unit=unit, kind="descriptive",
                                diff=r[col], n_events=r["n_events"]))
    # the difference between the two centroids with whole seasons resampled
    q = f"{DEP}/fig3_centroid_interval.csv"
    if os.path.exists(q):
        c = pd.read_csv(q)
        if len(c) != 1:
            raise SystemExit(f"{q} holds {len(c)} rows")
        c = c.iloc[0]
        out.append(dict(tier="pooled", level=np.nan, statistic="fig3_centroid_difference", time_rel_h=np.nan,
                        unit="deg", kind="descriptive", diff=c["diff"], ci_lo=c["ci_lo"], ci_hi=c["ci_hi"],
                        significant=bool(c["significant"]), n_boot=c["n_boot"], n_seasons=c["n_seasons"]))
    return pd.DataFrame(out)


def rows_from_dedup_counts():
    """The observations, trajectories and distinct time-position pairs before and after
    deduplication, and the two shares the text states in words."""
    p = f"{DEP}/dedup_counts.csv"
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p).set_index("stage")
    if sorted(d.index) != ["as_archived", "deduplicated"]:
        raise SystemExit(f"{p} does not hold exactly the two stages")
    out = []
    for stage, slug in (("as_archived", "archived"), ("deduplicated", "dedup")):
        for col in ("observations", "trajectories", "distinct_time_positions"):
            out.append(dict(tier="pooled", level=np.nan, statistic=f"n_{slug}_{col}", time_rel_h=np.nan, unit="count",
                            kind="point", diff=float(d.loc[stage, col])))
    for col in ("pct_archived_repeats", "pct_waves_of_trajectories"):
        vals = set(d[col])
        if len(vals) != 1:
            raise SystemExit(f"{p} states {col} with more than one value")
        out.append(dict(tier="pooled", level=np.nan, statistic=col, time_rel_h=np.nan, unit="%", kind="point",
                        diff=float(vals.pop())))
    return pd.DataFrame(out)


def rows_from_exceedance_sensitivity():
    """The moist-tail result at the percentiles other than the canonical one."""
    p = f"{DEP}/exceedance_sensitivity.csv"
    if not os.path.exists(p):
        return None
    out = []
    for _, r in pd.read_csv(p).iterrows():
        out.append(dict(tier="pooled", level=700.0, statistic=f"exceedsens_p{int(round(r['percentile']))}", time_rel_h=-72.0,
                        unit="points", kind="sensitivity", n_quiet=r["n_quiet"], n_active=r["n_active"],
                        mean_quiet=r["mean_quiet"], mean_active=r["mean_active"], diff=r["diff"], ci_lo=r["ci_lo"],
                        ci_hi=r["ci_hi"], significant=bool(r["significant"]), n_boot=r["n_boot"],
                        ratio=r["ratio"], threshold=r["threshold"]))
    return pd.DataFrame(out) if out else None


def rows_from_alignment_region():
    """The cross-reanalysis alignment on the regional 0.5 degree fields. The row of the
    global band is the script's own check against the stated constants and is not
    registered a second time, except for the number of troughs its median is taken over,
    which no other source states."""
    p = f"{DEP}/alignment_region.csv"
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    m, band = d[d.grid == "regional"], d[d.grid == "global band"]
    if len(m) != 1 or len(band) != 1:
        raise SystemExit(f"{p} holds {len(m)} rows for the regional grid and {len(band)} for the global band")
    r = m.iloc[0]
    out = [dict(tier="pooled", level=700.0, statistic=f"align_region_{col}", time_rel_h=np.nan, unit=unit, kind="point",
                diff=float(r[col]))
           for col, unit in (("peak_deg", "deg"), ("centroid_deg", "deg"), ("median_argmax_deg", "deg"),
                             ("grid_step_deg", "deg"), ("n_matched", "count"), ("n_window_outside_grid", "count"),
                             ("n_argmax", "count"))]
    out.append(dict(tier="pooled", level=700.0, statistic="align_band_n_argmax", time_rel_h=np.nan, unit="count", kind="point",
                    diff=float(band.iloc[0]["n_argmax"])))
    return pd.DataFrame(out)


def rows_from_fig_nulls():
    out = []
    for fig in ("fig2",):
        p = f"{DEP}/null_r1_{fig}_stats.csv"
        if not os.path.exists(p):
            continue
        for _, r in pd.read_csv(p).iterrows():
            row = dict(tier="pooled", level=np.nan,
                       statistic=f"r1_{fig}_{r['subset']}", time_rel_h=np.nan,
                       unit="counts", kind="null_test",
                       peak_excess=r["peak_excess"],
                       peak_rel_lon=r["peak_rel_lon"],
                       peak_ratio=r["peak_ratio"], p_value=r["p_value"],
                       simult_half_width=r["simult_half_width"])
            # n_draws was recorded by the producer but dropped here, so the prose's
            # "1,000 draws" had no row to tag against
            for c in ("n_troughs", "n_matched", "n_binned", "n_draws",
                      "n_troughs_region", "region_half_deg", "window_half_deg"):
                if c in r and pd.notna(r[c]):
                    row[c] = r[c]
            out.append(row)
    return pd.DataFrame(out) if out else None


def rows_from_leadlag():
    p = f"{DEP}/leadlag_stats.csv"
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    d.insert(0, "tier", "pooled")
    d.insert(1, "level", 700.0)
    d["time_rel_h"] = np.nan
    d = d.rename(columns={"value": "diff"})
    d["kind"] = "leadlag"
    return d


def main():
    parts = [rows_from_canonical()]
    for fn in (rows_from_wave_estimands, rows_from_control_model,
               rows_from_predictor_corr, rows_from_fig_nulls,
               rows_from_within_between,
               rows_from_bootstrap_config,
               rows_from_fig3_descriptive, rows_from_leadlag,
               rows_from_response_sensitivity, rows_from_yearstrat_sensitivity,
               rows_from_tierule_sensitivity, rows_from_catalog_sensitivity,
               rows_from_observation_rule, rows_from_scale_split,
               rows_from_dedup_counts, rows_from_exceedance_sensitivity,
               rows_from_alignment_region,
               rows_from_validation_constants):
        d = fn()
        if d is None:
            print(f"NOTE: {fn.__name__} input absent, skipped")
        else:
            parts.append(d)
    reg = pd.concat(parts, ignore_index=True, sort=False)
    key = reg[["tier", "level", "statistic", "time_rel_h"]].astype(str)
    dupmask = key.duplicated()
    if dupmask.any():   # empty-frame apply(axis=1) yields a DataFrame, so gate first
        dups = key[dupmask].apply(":".join, axis=1).tolist()
        raise SystemExit(f"duplicate registry keys: {dups[:5]}")
    out = f"{DEP}/canonical_registry.csv"
    reg.to_csv(out, index=False, float_format="%.6f")
    print(f"wrote {out} ({len(reg)} rows from {len(parts)} sources)")


if __name__ == "__main__":
    main()
