# Data and methods

Draft methods text for the reproduced and extended African Easterly Wave (AEW) analyses.
Numbers quoted are the values produced by the Python reimplementation described here.
They match the original NCL results where a direct comparison was possible.

## Data

Two wind records support the wave analysis. The original study used ERA-Interim
daily-averaged meridional wind at 700 hPa (v700) at 0.75 degree resolution. ERA-Interim
was retired from the Copernicus archive, so the reproduction also uses ERA5, its
successor, downloaded at 6-hourly cadence on the full longitude circle for the
wavenumber-frequency filtering step. ERA5 resolves finer scales than ERA-Interim, so the
two are close rather than identical over the common period, a point quantified in the
wave-identification subsection below.

Convective systems come from the International Satellite Cloud Climatology Project (ISCCP)
Convective System (CS) and Convective Tracking (CT) databases (Machado et al. 1998). The
CS records give one entry per system observation with time, latitude, longitude, and
equivalent radius. The CT records group systems into tracked families with per-step
location, radius, minimum cloud-top temperature, and family lifetime. System size follows
the equivalent-circle radius R = sqrt(N A / pi), where N is the pixel count and A the
pixel area, so a family qualifies for the CT database at R >= 90 km (about 30 pixels at
the ISCCP DX sampling of roughly 30 km).

Two independent products support cross-checking. The African Easterly Wave Climatology
(AEWC, Belanger et al., NCEI dataset C00784) provides objectively tracked wave-trough
trajectories from curvature-vorticity anomalies, 6-hourly over 1948 to 2010, from four
reanalyses. The ERA-Interim member is used here to match the original winds. The Huang et
al. (2018) tropical mesoscale convective system dataset provides an independently tracked
infrared cloud-system record (Clouds Archive User Service brightness temperature, 233 K
threshold, area-overlap plus Kalman-filter tracking) over 1985 to 2008.

GridSat-B1 (Knapp et al. 2011) supplies the raw infrared window brightness temperature for
an in-house convective-tracking product, on a roughly 8 km, 3-hourly, global grid from
1980 to present.

## Wave identification and composite dates

The wave signal is the meridional wind at 700 hPa filtered for westward-propagating
disturbances in the tropical-depression band. Filtering follows the space-time
wavenumber-frequency method of Frank and Roundy (2006), retaining zonal wavenumbers -20 to
0 (westward) and periods of 2.5 to 10 days. The reimplementation applies a two-dimensional
Fourier transform in time and longitude, keeps the coefficients inside the wavenumber and
period window on both members of each Hermitian conjugate pair, and inverts to a real
field. A linear detrend and a 5 percent cosine taper precede the transform, matching the
reference filter. A synthetic-wave test set confirms that the filter retains a westward
wave in the band, removes an eastward wave of the same wavenumber and period, and removes
in-band waves outside the wavenumber or period limits.

The base series is the filtered v700 at a fixed basepoint, here 10 N and 0 E. Composite
dates are the local maxima of that series above a threshold of two standard deviations,
which selects southerly wave maxima. Applied to the original ERA-Interim filtered field,
this yields a standard deviation of 1.63068 m/s, a two-sigma threshold of 3.26136 m/s, and
272 composite dates, matching the published values to five decimal places.

Running the same filter on global ERA5 v700 reproduces the ERA-Interim base series without
the original wave file. Over 2000 to 2004 the two basepoint series correlate at r = 0.92
across 7,308 six-hourly samples, with standard deviations of 1.581 m/s (ERA5) and 1.556
m/s (ERA-Interim) and composite-date counts of 73 and 71. A temporal band-pass filter that
omits the zonal-wavenumber selection gives far fewer dates over the same basepoint, which
identifies the space-time (rather than time-only) filtering as the operative step.

## Longitude-lag and map composites

Two composite operators act on the composite dates. The longitude-lag (Hovmoller)
operator averages a gridded field over a latitude band and, for each lag from -6 to +6
days, averages the band-mean field over the base dates whose lagged date falls inside the
record. The latitude-longitude operator is the map analogue at a chosen lag. Both draw the
statistical significance from a Monte Carlo test in which the null population is the set of
dates with the same calendar day and hour in other years (Frank and Roundy null), so the
resampling preserves the seasonal cycle while breaking the wave phase relationship. A
composite value is retained where its two-sided percentile against the null exceeds the
chosen level.

Cloud-system frequency about the wave is binned rather than averaged. For each composite
date, every system within the analysis latitude band contributes to a longitude-lag grid
at its longitude and its lag relative to that date, summed over all base dates. The shaded
anomaly is the raw count minus its mean over the lag window at each longitude. The map form
bins systems into latitude-longitude cells within a narrow lag window.

## In-house convective tracking from GridSat-B1

An in-house tracker regenerates an ISCCP-like cloud-system record from GridSat-B1, so the
convection side of the analysis does not depend on a single external product. Brightness
temperature is first coarsened to about 0.28 degrees, near the ISCCP DX sampling, so that
the equivalent-radius size cut carries the same physical meaning as in the ISCCP record.
Systems are cold-cloud shields below 245 K containing at least one convective core below
220 K, sized by equivalent radius and kept at R >= 90 km. Systems are linked across
consecutive 3-hourly images by area overlap, with each previous system projected forward
by its own estimated motion before the overlap is computed. The projection step follows
the tracking-and-classifying approach of Nunez Ocasio and Moon (2024) and addresses the
tendency of pure overlap tracking to fragment fast-moving systems, which matters for
wave-embedded systems that propagate quickly westward.

The regenerated record reproduces the geography of the original ISCCP CS. Over twelve July
to September months (1985, 1988, 1991, 1994), the in-house and ISCCP count maps over
tropical Africa at the common 90 km cut correlate at r = 0.90 plus or minus 0.04 on the
full grid, and at 0.84 plus or minus 0.06 over cells where ISCCP records systems, which is
the more demanding metric because it excludes the shared empty ocean and desert cells. The
Huang record correlates with ISCCP at 0.84 and 0.75 on the same two metrics. Absolute
system counts differ across the three records by a factor of about two, consistent with the
inter-tracker spread reported by the Mesoscale Convective System Tracking Method
Intercomparison (Feng et al. 2025). The wave-relative anomaly used in the composites is
less sensitive to that spread than the absolute count. The in-house record underweights the
Ethiopian Highlands relative to ISCCP, a difference that should be examined against the
cold-core detection on coarsened terrain.

## Wave-following (trough-relative) composites

The fixed-basepoint composite keys on wave maxima at a single point. A wave-following
composite instead keys on the moving trough, using the AEWC curvature-vorticity trajectories.
For each trough observation over West Africa in July to September, every cloud system within
three hours of the observation is binned by its longitude relative to the trough and its
latitude, summed over all trough observations. The time window is half open so a system
midway between two consecutive 6-hourly trough observations is counted once.

The trough-relative signal is separated from the background cloud climatology by a matched
null that keeps the trough times and latitudes but randomizes the trough longitudes, so any
enhancement fixed in geography rather than tied to the trough averages out. Cloud-system
counts exceed this null by about six times its two-standard-deviation spread in and just
west of the trough, and fall to the null level beyond about ten degrees. The peak sits near
the trough and slightly to its west, the direction of propagation, consistent with the
established phase relationship between AEW troughs and deep convection.

Stratifying the trough population by curvature vorticity sharpens the picture. Troughs in
the upper amplitude tercile produce a narrower and higher convective excess at the trough
(peak about 102 systems in the 5 to 15 N band) than troughs in the lower tercile (about
76). Convective-system genesis, taken as the first observation of each CT family, is also
enhanced near the trough but is distributed more broadly in trough-relative longitude than
mature-system frequency, which concentrates in and just west of the trough.

## Reproducibility

The analysis is a single tested Python package (numpy, scipy, xarray, pandas, matplotlib,
cartopy). Each numerical operator has unit tests on synthetic inputs with known answers,
and the full suite covers the filter, the composite-date selection, both composite
operators, the binning routines, the tracker, and the readers. Every non-trivial numerical
component was checked line by line against the surviving original code by an
independent code review. Input datasets and their access paths are recorded so that each
figure can be regenerated from public sources, with the exception of the ISCCP CS/CT
records, which are the original derived product.

## Limitations

The ERA5 wave series is close to the ERA-Interim series but not identical, because the two
reanalyses differ in resolution. The in-house tracker matches the ISCCP spatial pattern but
not the absolute counts, and it underweights the Ethiopian Highlands. The AEWC trough
record ends in 2010 and stops at 40 E, which excludes the eastern genesis region that the
CT record shows to be active. The CT family genesis time is taken from the family time
coordinate rather than the day-of-month field, which is unreliable in the processed file.

## References (to be completed)

Belanger et al. (AEWC / NCEI C00784); Feng et al. (2023, PyFLEXTRKR; 2025, MCSMIP); Frank
and Roundy (2006); Huang et al. (2018); Knapp et al. (2011, GridSat-B1); Machado et al.
(1998, ISCCP CT); Nunez Ocasio and Moon (2024, TAMS); Wheeler and Kiladis (1999).
