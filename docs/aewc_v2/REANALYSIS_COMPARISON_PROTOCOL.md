# The common reanalysis protocol, draft for agreement, 2026-09-25

STATUS: AGREED 2026-09-25 on the four decisions of section 9 (1979 to 2010 on both
reanalyses, Rule A applied separately under the same rules with the historical constants
a labeled ERA-Interim sensitivity, calendar-year initialization, observation-based
boundary definitions), with the one-year ERA5 retrieval and the bounded implementation
of section 8 authorized. The full-period retrieval and the public push are separate
approvals. The historical discrepancy investigation stays closed. It fixes, before
any comparison, the rules under which one fixed tracker implementation is run on
ERA-Interim and on ERA5, so that whatever differs between the two results is not a
difference of rules. The same rule is applied to both datasets in every item below, and
where a choice cannot be made the same on both, the item says so and stops.

The claim it serves, as revised and agreed on 2026-09-25: using
one fixed tracker implementation and a declared preprocessing and calibration protocol,
the paper measures how the resulting track statistics differ between ERA-Interim and
ERA5, with the archived record and QTrack as context. What the 1990 pilot demonstrated
and this protocol has to handle explicitly: tracks carry gaps between observations, a
season-end truncation changes what survives, and a membership rule read from the archive
is not a longitude box.

## 1. Inputs

- FIELDS. Zonal and meridional wind at 700 hPa, six-hourly, on both datasets, and
  nothing else. Version 1's optional environment layers are absent on both sides.
- GRID. Both datasets on version 1's buffered one-degree grid, 50 S to 50 N and 155 W to
  55 E (101 by 211), regridded to one degree before anything else, so that the
  decimation and the nine-point smoother act on the same geometry on both sides.
  ERA-Interim is on disk on this grid for 1979 to 2010 (64 files, 5.5 GB). ERA5 IS NOT:
  the trees on disk are a half-degree box stopping at 20 S, 45 N and 120 W for 1981 to
  2010, a 1.5 degree global tree for 1983 to 2007, and partial others, and the ERA5
  inventory says nothing has been retrieved on the buffered grid. The retrieval script
  `scripts/download_era5_v1port.py` was corrected 2026-09-25 to request the buffered
  grid, to validate every existing and returned file against the requested
  coordinates, calendar, variable, level and units before counting it, and to write a
  completion record per file. The earlier request's files stay untouched in
  `data/era5/v1port` as the record of that request.
- YEARS. The intersection both datasets can supply on that grid. ERA-Interim ends in
  2019 as a product and this project holds 1979 to 2010. The protocol's comparison
  years are 1979 to 2010, or 1981 to 2010 if 1979 and 1980 are not retrieved for ERA5,
  and the same years on both sides whichever is chosen.
- IDENTITY. Every input file digested, every run retaining its exported case file (not
  only its digest), a producer record in every output, one durable process per long
  run with a completion record.

## 2. Preprocessing, identical code path on both datasets

The port's own pipeline, `scripts/export_tracker_case.py` and `aew.v1port.pipeline`,
with no dataset-specific branch:

In the order the code executes it, since a review found the first draft's order wrong:

- Curvature vorticity from the winds on the one-degree grid.
- The anomaly against a climatology of the dataset's OWN six-hourly curvature vorticity
  over the protocol years, built by the same builder with the same fingerprint recipe on
  both sides (the fingerprints themselves differ, since the data differ, and the producer
  comparison expects that). The archive labels its climatology 1981 to 2010, and the
  pilot ran the port under 1979 to 2010, so the period is the protocol years of section
  1 on both sides and is stated in every artifact.
- Advection of the anomaly on the native one-degree grid.
- DECIMATION FIRST, with version 1's nominal parameter of 2.5 degrees, which on
  one-degree input is a stride of two and therefore an ACHIEVED coarse spacing of two
  degrees (the nominal parameter and the achieved spacing are both recorded), then the
  subset of both the coarse and the native grids to the tracking domain 35 S to 35 N and
  140 W to 40 E, as version 1's driver does and the archive's file attributes record.
- SMOOTHING INSIDE DETECTION, one nine-point pass on each cropped field at each
  timestep, followed by the southern-hemisphere sign reversal that makes cyclonic
  curvature positive in both hemispheres, applied in detection and in the calibration
  population alike.
- The zonal wind mask applied to the coarse fields only, as version 1's source does and
  as the port reproduces.

## 3. Calibration, the same rule on both datasets

Rule A, the agreed calibration rule, applied to BOTH reanalyses and not only to
ERA5: the 55th and 66th percentiles of each dataset's own smoothed 700 hPa curvature
vorticity anomaly, coarse and fine, over the declared population (every finite cell of
the tracking domain over the protocol years), with the exact linear-interpolation
estimator, no wind mask in the population, and no "90 percent for smoothing" adjustment.
ERA-Interim is therefore recalibrated under the rule rather than run on the archived
constants, so that the two sides differ in the data and not in how the thresholds were
made. The archived constants stay as a labeled ERA-Interim sensitivity, never as the
production rule. Each calibration artifact carries the four reproducibility items
(determinism, population accounting with the deficit enumerated, provenance with a
recomputable consistency block, descriptive comparison) and the adequacy reading (the
deficit by region and month), which the brief's C4 keeps as separate questions.

## 4. Region membership, the archive's rule on both datasets

The archive's own rule, reconstructed from its code on 2026-09-25 and implemented
in `scripts/season_metrics.py`: a track's first observation, each coordinate rounded to
the nearest quarter degree with ties away from zero as MATLAB rounds, tested against
the archive's seven region polygons in its priority order, the first containing polygon
naming the source region, else Other. The paper's product is the Africa region. Every
track of the whole tracking domain is retained with its region, and whole-domain totals
are reported beside the Africa-origin comparison.

## 5. Initialization, the same on both datasets

Each calendar year is run from January 1, as the archive's driver runs it, so that by
the reporting season the association carries the live state it would carry in the
archive. The run starts cold on January 1 and that is declared. No run starts inside a
reporting season. A track alive on December 31 is truncated by the year boundary on
both sides equally. A track whose last observation falls on December 31 is counted as
potentially censored, not as a demonstrated truncation, since nothing retained says
whether the association would have continued it (section 6), and that count is
reported per year.

## 6. Reporting seasons and tracks that cross their boundaries

- MEMBERSHIP. A track belongs to the June to September reporting season by its first
  observation, as the archive's files do. This is an operational definition.
- CROSSING IN, an exact predicate on retained observations: first observation before
  June 1 00Z and last observation at or after June 1 00Z. Such a track is not a season
  member and is not dropped silently: its count and its observations at or after June 1
  are reported as a separate line.
- CROSSING OUT, the same kind of predicate: first observation at or after June 1 00Z and
  before October 1 00Z, and last observation at or after October 1 00Z. Such a track is
  kept WHOLE, because the run continues to December 31, and its count is reported. The
  pilot's export stopped on September 30 and could not do this, which is why its
  duration percentiles carry an unmeasured truncation.
- WHAT "ALIVE" IS NOT. The retained outputs hold finished tracks' observations and
  nothing of the association's state, which finalization discards, so no predicate here
  reads that state, and a track whose observations straddle a boundary with a gap across
  it is classified by the observation predicates above and by nothing else.
- YEAR END, BOUNDARY-OBSERVED. A track whose last observation falls on December 31 is
  counted as boundary-observed and potentially censored, since nothing retained says
  whether the association would have continued it. The count is reported per year on
  both sides. It is not called a demonstrated truncation.
- BOUNDARY FIXTURES BEFORE IMPLEMENTATION: a track spanning a boundary with no
  observation on it, a track ending one timestep before a boundary, and a track ending
  on December 31, each with its expected classification, written before the predicates
  are coded.
- GAPS. Version 1's association carries a track across a missed timestep. Lifetime is
  reported as observations recorded and, apart, as duration in days between the first
  and the last observation, with the spacing block (fraction of one-timestep intervals,
  largest interval, tracks with a gap), on both sides.
- GENESIS AND LYSIS are the first and the last recorded observation.

## 7. The measurement and the reporting

The reviewed instrument, `scripts/season_metrics.py`, with two additions and nothing
else: the boundary-crossing lines of section 6, and a REANALYSIS-COMPARISON MODE. The
instrument today refuses two track files whose case identities differ, which is right
for an implementation comparison on one export and wrong here, since the identities
digest the exported fields and two reanalyses must differ. The mode retains both case
identities, checks that the two runs share the protocol settings (years, grid, nominal
and achieved spacing, tracker flags, calibration rule) from their producer records, and
labels the columns by dataset. The numerical functions are reused as they are. Run on
each dataset's finished tracks, with the
published archive's interannual spread as the scale where a scale is reported and the
archive and QTrack as context columns whose methodological differences are listed in
the same table. EVERY MEASURED CONTRAST IS REPORTED, including small and null
differences. The thresholds in the brief's section 7 are labeled "contrast considered
scientifically material" and serve the discussion of what the paper calls a change,
not what it reports.

## 7a. Implementation prerequisites, listed so no run makes them silently

- ONE SHARED ENTRY POINT. `scripts/export_tracker_case.py` today hardcodes the
  `eraint` file prefix, builds the climatology over every year on disk, labels the case
  `ERA-Int`, selects version 1's historical thresholds by that label, and carries an
  `--exclusive` switch that selects a tracker variant. A protocol run needs an entry
  point that takes the dataset, the agreed years, the calibration artifact and the fixed
  tracker flags (`exclusive` off, `absorb` off) explicitly, and records all of them in
  the retained case and the producer record. Until it exists, the protocol cannot be
  executed without edits that could defeat the common-rule requirement.
- ONE RETAINED PROTOCOL MANIFEST, read by both runs: the agreed years, the grid, the
  nominal and achieved spacing, the calibration artifacts, the tracker flags and the
  reporting predicates, so the comparison mode of section 7 checks both runs against
  the same object.
- THE RETRIEVAL SCRIPT'S AREA AND GRID set to the buffered one-degree grid before any
  ERA5 request, since it pins a different box and a half-degree spacing today.

## 8. One paired feasibility run, proposed for after the protocol is agreed

PURPOSE, corrected on Hilawe's review of 2026-09-25: a ONE-YEAR INPUT-READINESS PILOT,
not a paired tracking validation. One ERA5 year cannot establish readiness for
meaningful anomalies, calibrated thresholds or tracking under the intended climatology,
and those capabilities are marked UNTESTED here rather than required.

THE RUN. One calendar year, 1990, on ERA-Interim and on ERA5, port only, through the
shared entry point of section 7a. THE FEASIBILITY CLIMATOLOGY IS DECLARED APART. With
one retrieved ERA5 year, a climatology over that year alone is degenerate (each
calendar timestep's mean is its own value and the anomaly is zero), so the one-year run
does not exercise tracking on ERA5. It checks the inputs and the preprocessing through
the advection and decimation of section 2 on both sides, and TRACKING READINESS ON ERA5
IS MARKED UNTESTED until enough independent years are on the buffered grid to build the
protocol climatology. On ERA-Interim, which has its protocol years on disk, the run goes
through tracking under section 3's thresholds. The producer comparison of the two runs
expects differing climatology and threshold hashes and records them.

WHAT IT CHECKS, each a yes or no with the evidence named, and what it marks untested:

- INPUT READINESS. ERA5 for 1990 exists on the buffered one-degree grid, opens, carries
  exactly the requested coordinates, a gapless six-hourly full-calendar axis, the
  variable, the level and the units, under the same validator the ERA-Interim files
  pass, with a completion record per file and the actual size and elapsed time.
- COMPARABLE PROCESSING THROUGH THE ENTRY POINT. The shared entry point of section 7a
  runs on ERA-Interim 1990 and on ERA5 1990 through one code path up to the advection
  and decimation of section 2, and the two producer records differ only in the dataset
  and the input digests, with every protocol setting equal. On ERA5 the preprocessing
  arrays pass the same shape, mask and finiteness checks as the ERA-Interim ones.
- THE INSTRUMENT'S COMPARISON MODE AND BOUNDARY LINES, tested on fixtures, and run on
  the ERA-Interim 1990 output so every line of sections 6 and 7 is present, including
  the crossing lines, the spacing blocks and the potentially censored year-end count.
- UNTESTED, BY CONSTRUCTION: anomalies against the protocol climatology on ERA5,
  calibrated thresholds on ERA5, and tracking on ERA5. No calibration artifact is
  required or produced for ERA5 in this pilot, since none can exist before the
  protocol years are on the grid.

COST. The ERA5 retrieval for one year on the buffered grid (two variables, six-hourly,
101 by 211, on the order of 250 MB), which is queued at Copernicus with no throughput
guarantee. The port run is minutes per year and the instrument seconds. No Octave run.
The sizes are uncompressed payload arithmetic (two float32 fields, 1,460 timesteps,
101 by 211 cells, 249 MB per non-leap year, 7.97 GB for 1979 to 2010), not download
sizes.

STOPPING CONDITION. When every check has its answer, or when the retrieval has not
returned within a week, which is then the finding. Its results are a readiness table
and nothing else, and the full retrieval for the protocol years (about 8 GB) is priced
from the one-year request's actual behavior.

## 9. What Hilawe agrees or amends

1. The protocol years, 1979 to 2010 or 1981 to 2010, the same on both sides.
2. Rule A on both datasets, with ERA-Interim recalibrated and the archived constants a
   labeled sensitivity.
3. The initialization and the boundary rules of sections 5 and 6.
4. The feasibility run of section 8, and the one-year ERA5 retrieval it needs.
