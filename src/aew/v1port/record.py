"""Record output, ported from output_ews_netcdf_f.m.

Stage 7, the last of the port. It writes the tracks the tracker produced as one netCDF
file per region, in the CF contiguous ragged array layout version 1 used, so a file this
module writes and a file from the published record can be read by the same code.

WHAT THE LAYOUT IS. Three dimensions. `trajectory` counts the waves in the file, `sample`
is every observation of every wave laid end to end, and `count` gives each wave's length,
so wave n occupies the slice of `sample` starting at the sum of the counts before it. That
is CF's contiguous ragged array, and `count` carries the `sample_dimension` attribute that
names it as such.

THE STATISTICS THIS PORT CANNOT FILL. Version 1's file carries thirty of its thirty-nine
variables from generate_ew_stats_f.m, which composites satellite and radiation fields over
each trough: outgoing longwave radiation, Claus brightness temperature, and four SSM/I
products. None of that is ported, and version 2's source for it is an open question, so
those variables are written at the declared fill value of -999. A CF reader sees them as
missing, which is what they are. They are present rather than omitted because the point of
this stage is a file the same reader code can open.

THREE METADATA DEFECTS IN VERSION 1, all confirmed in the published record rather than
inferred from the source, by reading ERA-Int_ew_700hPa_2005_AFR.nc:

  1. `geospatial_lon_units` is "degrees_north". Longitude is not measured north.
  2. The `obs` dimension is defined, sized to the longest track in the file (53 in the 2005
     Africa file), and used by no variable at all.
  3. The six area-fraction variables disagree with each other: two carry `units = ""` and
     four carry no units attribute. They are all fractions.

`reproduce_v1_metadata=True`, the default, writes all three as version 1 wrote them, which
is what a comparison against the published record needs. False corrects them: longitude in
degrees_east, no unused dimension, and every fraction with `units = "1"`. It also drops the
satellite provenance strings from the seventeen composited variables, because naming an
instrument behind a variable this port fills entirely with the fill value claims more than
the file holds. No variable, value or dimension that carries data changes between the two.

THE TIME CONVERSION IS THE EASIEST THING HERE TO GET WRONG AND THE HARDEST TO NOTICE.
Version 1's track times are MATLAB serial datenums and it subtracts `datenum('01/01/1900')`
before writing, which is what turns 732313.25 into the 38351.25 the published 2005 file
holds. A port that writes its numbers through unchanged produces a file whose units
attribute reads correctly and whose values are eight centuries out, and nothing in the file
gives that away. So `days_since_1900` requires the caller to say what its numbers mean and
refuses to guess.
"""

import numpy as np

FILL = -999.0
TIME_EPOCH = "days since 1900-01-01 00:00:00"

# MATLAB `datenum('01/01/1900')`, which version 1 subtracts before writing a time. Its
# serial day count starts at a proleptic year zero, so the offset is large and forgetting
# it puts every timestamp eight centuries out while the units attribute still reads right.
MATLAB_DATENUM_1900 = 693962
# `datetime.date(1900, 1, 1).toordinal()`, for callers whose times are Python ordinals.
PYTHON_ORDINAL_1900 = 693596

REGIONS = ("NEP", "SEP", "CAM", "SAM", "NAL", "SAL", "AFR", "OTH")
REGION_NAMES = {
    "NEP": "Northeast Pacific", "SEP": "Southeast Pacific",
    "CAM": "Central America", "SAM": "South America",
    "NAL": "North Atlantic", "SAL": "South Atlantic",
    "AFR": "Africa", "OTH": "Other",
}
SOURCE_NAMES = {
    "CFS": "NCEP Climate Forecast System Reanalysis",
    "ERA-Int": "ERA-Interim Reanalysis",
    "ERA-40": "ERA-40 Reanalysis",
    "NCEP": "NCEP/NCAR Reanalysis",
}

# The track fields the tracker itself produces, as (variable name, track key).
TRACK_VARIABLES = (
    ("time", "time"),
    ("lat", "meanlat"),
    ("lon", "meanlon"),
    ("maxlat", "maxlat"),
    ("meanlonatmaxlat", "meanlon_maxlat"),
    ("minlat", "minlat"),
    ("meanlonatminlat", "meanlon_minlat"),
    ("wavelength", "wavelength"),
    ("meanrv", "meanrv"), ("maxrv", "maxrv"),
    ("minrv", "minrv"), ("stdrv", "stdrv"),
    ("meancrv", "meancurrv"), ("maxcrv", "maxcurrv"),
    ("mincrv", "mincurrv"), ("stdcrv", "stdcurrv"),
    ("meansrv", "meanshrrv"), ("maxsrv", "maxshrrv"),
    ("minsrv", "minshrrv"), ("stdsrv", "stdshrrv"),
)

# The composited fields generate_ew_stats_f.m fills and this port does not. Written at the
# fill value, in the order version 1 defines them, because variable order is part of what
# a comparison against the published record checks.
#
# THEY ARE STILL READ FROM THE TRACK IF PRESENT, under the names version 1 uses for them,
# so a later port of generate_ew_stats_f.m does not have to change this function to be
# written out. An earlier version hard-wired them to fill, which would have silently
# discarded exactly the data that stage exists to produce. Writing fill because a value is
# absent is right; writing fill over a value that is there is not.
COMPOSITE_VARIABLES = (
    "meanolr", "stdolr", "olr_area_fraction", "meanolra", "stdolra",
    "meanctb", "stdctb", "ctb_area_fraction",
    "meantpw", "stdtpw", "tpw_area_fraction",
    "meanrain", "stdrain", "rain_area_fraction",
    "meancloud", "stdcloud", "cloud_area_fraction",
)

# The track field each composited variable reads, taken from the write loop of
# output_ews_netcdf_f.m. The names are not guessable: `meanctb` reads `meanclaus`,
# `meantpw` reads `meanvapor`, and every area fraction reads a `_cover` field.
COMPOSITE_TRACK_KEYS = {
    "meanolr": "meanolr", "stdolr": "stdolr", "olr_area_fraction": "olr_cover",
    "meanolra": "meanolr_anom", "stdolra": "stdolr_anom",
    "meanctb": "meanclaus", "stdctb": "stdclaus", "ctb_area_fraction": "claus_cover",
    "meantpw": "meanvapor", "stdtpw": "stdvapor", "tpw_area_fraction": "vapor_cover",
    "meanrain": "meanrain", "stdrain": "stdrain", "rain_area_fraction": "rain_cover",
    "meancloud": "meancloud", "stdcloud": "stdcloud",
    "cloud_area_fraction": "cloud_cover",
}

FRACTION_VARIABLES = ("olr_area_fraction", "ctb_area_fraction", "tpw_area_fraction",
                      "rain_area_fraction", "cloud_area_fraction")

# Which fraction variables version 1 gave an empty units attribute, and which it gave none.
# Reproduced exactly because the inconsistency is one of the three recorded defects.
_V1_EMPTY_UNITS = ("olr_area_fraction", "ctb_area_fraction")

SAMPLE_VARIABLES = tuple(name for name, _ in TRACK_VARIABLES) + COMPOSITE_VARIABLES


def variable_names():
    """Every variable the file carries, in version 1's definition order."""
    return ("count", "trajectory") + SAMPLE_VARIABLES


def _global_attributes(year, level, reanalysis, region, reproduce_v1_metadata, created):
    lon_units = "degrees_north" if reproduce_v1_metadata else "degrees_east"
    return {
        "Conventions": "CF-1.6",
        "Metadata_Conventions": "Unidata Dataset Discovery v1.0",
        "featureType": "trajectory",
        "cdm_data_type": "Trajectory",
        "standard_name_vocabulary":
            "CF Standard Name Table (v26, 08 November 2013)",
        "title": "African Easterly Wave Climatology",
        "summary": (f"easterly wave trajectories for {level:03d} hPa originating from "
                    f"{REGION_NAMES[region]} for {year:04d}"),
        "source": SOURCE_NAMES.get(reanalysis, reanalysis),
        "id": f"{reanalysis}_ew_{level}hPa_{year}_{region}.nc",
        "naming_authority": "gov.noaa.ncdc",
        "time_coverage_start": f"{year:04d}-01-01T00:00:00Z",
        "time_coverage_end": f"{year:04d}-12-31T18:00:00Z",
        "time_coverage_resolution": "P6H",
        "time_coverage_duration": "P1Y",
        "geospatial_lat_min": np.float32(-35),
        "geospatial_lat_max": np.float32(35),
        "geospatial_lat_units": "degrees_north",
        "geospatial_lon_min": np.float32(-140),
        "geospatial_lon_max": np.float32(40),
        "geospatial_lon_units": lon_units,
        "institution": ("GATECH/GTRI > Georgia Institute of Technology, Georgia Tech "
                        "Research Institute, School of Earth & Atmospheric Sciences"),
        "creator_name": "James Belanger, Mark Jelinek, Judith Curry",
        "creator_email": "james.belanger@gatech.edu",
        "project": ("U.S. Department of Commerce NOAA Grant 3506G58; National Science "
                    "Foundation Grant 3506G42"),
        "processing_level": "Level 4",
        "keywords_vocabulary": ("NASA Global Change Master Directory (GCMD) Earth "
                                "Science Keywords, Version 8.0"),
        "keywords": "EARTH SCIENCE, ATMOSPHERE, ATMOSPHERIC PHENOMENA, HURRICANES",
        "references": ("Belanger, J.I, M. T. Jelinek, and  J. A. Curry, 2014: Revisiting "
                       "the tropical cyclone-easterly wave relationship on interannual "
                       "time scales, J. Climate."),
        "date_created": created,
        "license": "No constraints on data access or use",
        "metadata_link": "gov.noaa.ncdc:C00784",
    }


def _variable_attributes(name, reproduce_v1_metadata):
    """The per-variable attributes version 1 writes, for every variable in the file.

    Under `reproduce_v1_metadata=True` this is the complete set the published record
    carries, provenance strings included. Under False the satellite provenance is dropped
    from the composited variables, because this port does not fill them and naming the
    instrument behind an all-fill variable claims more than the file holds.
    """
    if name == "count":
        return {"long_name": "number of observations for the easterly wave",
                "sample_dimension": "sample"}
    if name == "trajectory":
        return {"long_name": "easterly wave trajectory", "cf_role": "trajectory_id"}
    if name == "time":
        return {"long_name": "time", "standard_name": "time", "units": TIME_EPOCH,
                "calendar": "gregorian", "axis": "T"}
    if name == "lat":
        return {"long_name": "wave trough centroid latitude",
                "standard_name": "latitude", "units": "degrees_north", "axis": "Y",
                "cell_methods": "latitude: mean (over the wave trough)"}
    if name == "lon":
        return {"long_name": "wave trough centroid longitude",
                "standard_name": "longitude", "units": "degrees_east", "axis": "X",
                "cell_methods": "longitude: mean (over the wave trough)"}
    attrs = {"long_name": _LONG_NAMES[name], "coordinates": "time lat lon"}
    if name in FRACTION_VARIABLES:
        if reproduce_v1_metadata:
            if name in _V1_EMPTY_UNITS:
                attrs["units"] = ""
        else:
            attrs["units"] = "1"
    else:
        units = _UNITS.get(name)
        if units is not None:
            attrs["units"] = units
        cell_methods = _CELL_METHODS.get(name)
        if cell_methods is not None:
            attrs["cell_methods"] = cell_methods
    comment = _COMMENTS.get(name)
    if comment is not None:
        attrs["comment"] = comment
    if reproduce_v1_metadata:
        attrs.update(_PROVENANCE.get(name, {}))
    return attrs


# Version 1 attaches a comment to most composited variables, naming the satellite
# provenance. Only wavelength's is carried here, because it describes a definition this
# port implements rather than data this file does not contain.
_COMMENTS = {
    "wavelength": ("wavelength defined as distance in wave direction where curvature "
                   "vorticity anomaly for wave trough equals 0 s-1."),
}


_LONG_NAMES = {
    "maxlat": "wave trough maximum latitude",
    "meanlonatmaxlat": "mean longitude of wave trough maximum latitude",
    "minlat": "wave trough minimum latitude",
    "meanlonatminlat": "mean longitude of wave trough minimum latitude",
    "wavelength": "horizontal wavelength",
    "meanrv": "wave trough mean relative vorticity",
    "maxrv": "wave trough maximum relative vorticity",
    "minrv": "wave trough minimum relative vorticity",
    "stdrv": "wave trough standard deviation relative vorticity",
    "meancrv": "wave trough mean curvature vorticity",
    "maxcrv": "wave trough maximum curvature vorticity",
    "mincrv": "wave trough minimum curvature vorticity",
    "stdcrv": "wave trough standard deviation curvature vorticity",
    "meansrv": "wave trough mean shear vorticity",
    "maxsrv": "wave trough maximum shear vorticity",
    "minsrv": "wave trough minimum shear vorticity",
    "stdsrv": "wave trough standard deviation shear vorticity",
    "meanolr": "wave trough mean outgoing longwave radiation",
    "stdolr": "wave trough standard deviation outgoing longwave radiation",
    "olr_area_fraction": "wave trough outgoing longwave radiation area fraction",
    "meanolra": "wave trough mean outgoing longwave radiation anomalies",
    "stdolra": "wave trough standard deviation outgoing longwave radiation anomalies",
    "meanctb": "wave trough mean Claus brightness temperature",
    "stdctb": "wave trough standard deviation Claus brightness temperature",
    "ctb_area_fraction": "wave trough Claus brightness temperature area fraction",
    "meantpw": "wave trough mean total precipitable water",
    "stdtpw": "wave trough standard deviation total precipitable water",
    "tpw_area_fraction": "wave trough total precipitable water area fraction",
    "meanrain": "wave trough mean rain rate",
    "stdrain": "wave trough standard deviation rain rate",
    "rain_area_fraction": "wave trough total rain rate area fraction",
    "meancloud": "wave trough mean total cloud liquid water",
    "stdcloud": "wave trough standard deviation cloud liquid water",
    "cloud_area_fraction": "wave trough total cloud liquid water area fraction",
}

_UNITS = {
    "maxlat": "degrees_north", "meanlonatmaxlat": "degrees_east",
    "minlat": "degrees_north", "meanlonatminlat": "degrees_east",
    "wavelength": "km",
    "meanrv": "s-1", "maxrv": "s-1", "minrv": "s-1", "stdrv": "s-1",
    "meancrv": "s-1", "maxcrv": "s-1", "mincrv": "s-1", "stdcrv": "s-1",
    "meansrv": "s-1", "maxsrv": "s-1", "minsrv": "s-1", "stdsrv": "s-1",
    "meanolr": "W m-2", "stdolr": "W m-2",
    "meanolra": "W m-2", "stdolra": "W m-2",
    "meanctb": "K", "stdctb": "K",
    "meantpw": "mm", "stdtpw": "mm",
    "meanrain": "mm hr-1", "stdrain": "mm hr-1",
    "meancloud": "mm", "stdcloud": "mm",
}

_CELL_METHODS = {
    "maxlat": "latitude: maximum (over the wave trough)",
    "meanlonatmaxlat": ("latitude: maximum (over the wave trough) longitude: mean "
                        "(across the wave trough at maximum latitude)"),
    "minlat": "latitude: minimum (over the wave trough)",
    "meanlonatminlat": ("latitude: minimum (over the wave trough) longitude: mean "
                        "(across the wave trough at minimum latitude)"),
    "meanrv": "area: mean (over the wave trough)",
    "maxrv": "area: maximum (over the wave trough)",
    "minrv": "area: minimum (over the wave trough)",
    "stdrv": "area: standard_deviation (over the wave trough)",
    "meancrv": "area: mean (over the wave trough)",
    "maxcrv": "area: maximum (over the wave trough)",
    "mincrv": "area: minimum (over the wave trough)",
    "stdcrv": "area: standard_deviation (over the wave trough)",
    "meansrv": "area: mean (over the wave trough)",
    "maxsrv": "area: maximum (over the wave trough)",
    "minsrv": "area: minimum (over the wave trough)",
    "stdsrv": "area: standard_deviation (over the wave trough)",
    "meanolr": "area: mean (over the wave trough)",
    "stdolr": "area: standard_deviation (over the wave trough)",
    "meanolra": "area: mean (over the wave trough)",
    "stdolra": "area: standard_deviation (over the wave trough)",
    "meanctb": "area: mean (over the wave trough)",
    "stdctb": "area: standard_deviation (over the wave trough)",
    "meantpw": "area: mean (over the wave trough)",
    "stdtpw": "area: standard_deviation (over the wave trough)",
    "meanrain": "area: mean (over the wave trough)",
    "stdrain": "area: standard_deviation (over the wave trough)",
    "meancloud": "area: mean (over the wave trough)",
    "stdcloud": "area: standard_deviation (over the wave trough)",
}


def days_since_1900(times, time_epoch):
    """Convert a track's time values to the days-since-1900 the file declares.

    THIS CONVERSION IS THE WRITER'S JOB, and getting it wrong is invisible. Version 1
    writes `single(ews(ewn).time - datenum('01/01/1900'))`, because its track times are
    MATLAB serial datenums, days counted from a proleptic year zero. `datenum('01/01/1900')`
    is 693962, and subtracting it is what turns 732313.25 into the 38351.25 the published
    2005 file actually holds. A port that writes the raw numbers produces a file declaring
    "days since 1900-01-01" and holding values eight hundred years out, which no reader can
    detect from the file alone.

    `time_epoch` says what the track's numbers mean, and there is NO DEFAULT, deliberately.
    A silent guess here is exactly the failure above.

        "1900"    already days since 1900-01-01; passed through
        "matlab"  MATLAB serial datenums; 693962 subtracted
        "python"  `datetime.date.toordinal` values, days from 0001-01-01
    """
    offsets = {"1900": 0.0, "matlab": float(MATLAB_DATENUM_1900),
               "python": float(PYTHON_ORDINAL_1900)}
    if time_epoch not in offsets:
        raise ValueError(
            f"time_epoch must be one of {sorted(offsets)}, not {time_epoch!r}. The file "
            f"declares {TIME_EPOCH!r}, so the writer has to be told what the track's time "
            f"numbers count from; guessing is how a record ends up with the right units "
            f"attribute and the wrong values.")
    return np.asarray(times, dtype=np.float64) - offsets[time_epoch]


# The provenance strings version 1 attaches to its composited variables: which satellite
# record each came from, the platform and instrument list, and how the overpasses were
# binned. GENERATED FROM THE PUBLISHED RECORD rather than transcribed from the MATLAB,
# because a review found the first draft had simply omitted all of them and the oracle test
# could not see it: that test compared only the attributes the port chose to emit, so
# anything left out was invisible to it. It now compares in both directions.
#
# These are written under `reproduce_v1_metadata=True` and omitted under False. An earlier
# draft omitted them always, on the argument that describing a satellite record this file
# does not contain would be misleading. That argument is not wrong, but it made the choice
# silently and put an undocumented divergence in the way of the one thing this stage is
# for, which is a file comparable to the published one. So it became a setting like the
# other three, and the misleading-attribute case is what False is for.
#
# THESE STRINGS ARE VERBATIM QUOTATIONS AND NO SPELLING SWEEP MAY TOUCH THEM. They are the
# published file's own bytes, so they keep the spelling they were deposited with, British
# or otherwise ("National Climatic Data Centre" appears here and stays). A US English pass
# over this file rewrote three of them on 2026-08-28 and the two-directional oracle test
# caught it immediately, which is what that test is for. Same shape as the manuscript's
# rule against find-and-replace across registry tag interiors: a sweep must run over prose,
# and quoted source text is not prose.
_PROVENANCE = {
    "meanolr": {
        "source": (
            "The Outgoing Longwave Radiation - Daily CDR used in this study was"
            " acquired from NOAA's National Climatic Data Center (http://www.nc"
            "dc.noaa.gov).  This CDR was originally developed by Hai-Tien Lee a"
            "nd colleagues for the NOAA's CDR Program."
        ),
        "references": (
            "Lee,H.-T., C.J. Schreck, K. R. Knapp, 2014: Generation of the Dail"
            "y OLR Climate Data Record.  2014 EUMETSTAT Meteorological Satellit"
            "e Conference, 22-26 September 2014, Geneva, Switzerland."
        ),
    },
    "stdolr": {
        "source": (
            "The Outgoing Longwave Radiation - Daily CDR used in this study was"
            " acquired from NOAA's National Climatic Data Center (http://www.nc"
            "dc.noaa.gov).  This CDR was originally developed by Hai-Tien Lee a"
            "nd colleagues for the NOAA's CDR Program."
        ),
        "references": (
            "Lee,H.-T., C.J. Schreck, K. R. Knapp, 2014: Generation of the Dail"
            "y OLR Climate Data Record.  2014 EUMETSTAT Meteorological Satellit"
            "e Conference, 22-26 September 2014, Geneva, Switzerland."
        ),
    },
    "olr_area_fraction": {
        "source": (
            "The Outgoing Longwave Radiation - Daily CDR used in this study was"
            " acquired from NOAA's National Climatic Data Center (http://www.nc"
            "dc.noaa.gov).  This CDR was originally developed by Hai-Tien Lee a"
            "nd colleagues for the NOAA's CDR Program."
        ),
        "references": (
            "Lee,H.-T., C.J. Schreck, K. R. Knapp, 2014: Generation of the Dail"
            "y OLR Climate Data Record.  2014 EUMETSTAT Meteorological Satellit"
            "e Conference, 22-26 September 2014, Geneva, Switzerland."
        ),
        "comment": (
            "Fractional area coverage of available OLR data for the wave trough"
            "."
        ),
    },
    "meanolra": {
        "source": (
            "The Outgoing Longwave Radiation - Daily CDR used in this study was"
            " acquired from NOAA's National Climatic Data Center (http://www.nc"
            "dc.noaa.gov).  This CDR was originally developed by Hai-Tien Lee a"
            "nd colleagues for the NOAA's CDR Program."
        ),
        "references": (
            "Lee,H.-T., C.J. Schreck, K. R. Knapp, 2014: Generation of the Dail"
            "y OLR Climate Data Record.  2014 EUMETSTAT Meteorological Satellit"
            "e Conference, 22-26 September 2014, Geneva, Switzerland."
        ),
        "comment": (
            "OLR anomalies calculated by removing daily OLR from the long-term "
            "daily mean for 1981-2010."
        ),
    },
    "stdolra": {
        "source": (
            "The Outgoing Longwave Radiation - Daily CDR used in this study was"
            " acquired from NOAA's National Climatic Data Center (http://www.nc"
            "dc.noaa.gov).  This CDR was originally developed by Hai-Tien Lee a"
            "nd colleagues for the NOAA's CDR Program."
        ),
        "references": (
            "Lee,H.-T., C.J. Schreck, K. R. Knapp, 2014: Generation of the Dail"
            "y OLR Climate Data Record.  2014 EUMETSTAT Meteorological Satellit"
            "e Conference, 22-26 September 2014, Geneva, Switzerland."
        ),
        "comment": (
            "OLR anomalies calculated by removing daily OLR from the long-term "
            "daily mean for 1981-2010."
        ),
    },
    "meanctb": {
        "references": (
            "Environmental Systems Science Centre (ESSC), [Robinson, G.J.] . Cl"
            "oud Archive User Service (CLAUS), [Internet]. NCAS British Atmosph"
            "eric Data Centre, 2002. Available from http://badc.nerc.ac.uk/view"
            "/badc.nerc.ac.uk__ATOM__dataent_claus."
        ),
    },
    "stdctb": {
        "references": (
            "Environmental Systems Science Centre (ESSC), [Robinson, G.J.] . Cl"
            "oud Archive User Service (CLAUS), [Internet]. NCAS British Atmosph"
            "eric Data Centre, 2002. Available from http://badc.nerc.ac.uk/view"
            "/badc.nerc.ac.uk__ATOM__dataent_claus."
        ),
    },
    "ctb_area_fraction": {
        "references": (
            "Environmental Systems Science Centre (ESSC), [Robinson, G.J.] . Cl"
            "oud Archive User Service (CLAUS), [Internet]. NCAS British Atmosph"
            "eric Data Centre, 2002. Available from http://badc.nerc.ac.uk/view"
            "/badc.nerc.ac.uk__ATOM__dataent_claus."
        ),
        "comment": (
            "Fractional area coverage of available Claus brightness temperature"
            " data for the wave trough."
        ),
    },
    "meantpw": {
        "references": (
            "SSM/I and SSMIS data are produced by Remote Sensing Systems and sp"
            "onsored by the NASA Earth Science MEaSUREs Program and are availab"
            "le at www.remss.com."
        ),
        "platform": "DMSP SSM/I",
        "instrument": (
            "F08:1987-1991, F10:1990-1997, F11:1991-2000, F13:1995-2009, F14:19"
            "97-2008, F15:1999-2006, F16:2003-2010, F17:2006-2010"
        ),
        "comment": (
            "Satellite overpass data rounded to nearest 6-hr and averaged for a"
            "ll overlapping times and locations."
        ),
    },
    "stdtpw": {
        "references": (
            "SSM/I and SSMIS data are produced by Remote Sensing Systems and sp"
            "onsored by the NASA Earth Science MEaSUREs Program and are availab"
            "le at www.remss.com."
        ),
        "platform": "DMSP SSM/I",
        "instrument": (
            "F08:1987-1991, F10:1990-1997, F11:1991-2000, F13:1995-2009, F14:19"
            "97-2008, F15:1999-2006, F16:2003-2010, F17:2006-2010"
        ),
        "comment": (
            "Satellite overpass data rounded to nearest 6-hr and averaged for a"
            "ll overlapping times and locations."
        ),
    },
    "tpw_area_fraction": {
        "references": (
            "SSM/I and SSMIS data are produced by Remote Sensing Systems and sp"
            "onsored by the NASA Earth Science MEaSUREs Program and are availab"
            "le at www.remss.com."
        ),
        "platform": "DMSP SSM/I",
        "instrument": (
            "F08:1987-1991, F10:1990-1997, F11:1991-2000, F13:1995-2009, F14:19"
            "97-2008, F15:1999-2006, F16:2003-2010, F17:2006-2010"
        ),
        "comment": (
            "Fractional area coverage of available SSM/I total precipitable wat"
            "er data for the wave trough. Satellite overpass data rounded to ne"
            "arest 6-hr and averaged for all overlapping times and locations."
        ),
    },
    "meanrain": {
        "references": (
            "SSM/I and SSMIS data are produced by Remote Sensing Systems and sp"
            "onsored by the NASA Earth Science MEaSUREs Program and are availab"
            "le at www.remss.com."
        ),
        "platform": "DMSP SSM/I",
        "instrument": (
            "F08:1987-1991, F10:1990-1997, F11:1991-2000, F13:1995-2009, F14:19"
            "97-2008, F15:1999-2006, F16:2003-2010, F17:2006-2010"
        ),
        "comment": (
            "Satellite overpass data rounded to nearest 6-hr and averaged for a"
            "ll overlapping times and locations."
        ),
    },
    "stdrain": {
        "references": (
            "SSM/I and SSMIS data are produced by Remote Sensing Systems and sp"
            "onsored by the NASA Earth Science MEaSUREs Program and are availab"
            "le at www.remss.com."
        ),
        "platform": "DMSP SSM/I",
        "instrument": (
            "F08:1987-1991, F10:1990-1997, F11:1991-2000, F13:1995-2009, F14:19"
            "97-2008, F15:1999-2006, F16:2003-2010, F17:2006-2010"
        ),
        "comment": (
            "Satellite overpass data rounded to nearest 6-hr and averaged for a"
            "ll overlapping times and locations."
        ),
    },
    "rain_area_fraction": {
        "references": (
            "SSM/I and SSMIS data are produced by Remote Sensing Systems and sp"
            "onsored by the NASA Earth Science MEaSUREs Program and are availab"
            "le at www.remss.com."
        ),
        "platform": "DMSP SSM/I",
        "instrument": (
            "F08:1987-1991, F10:1990-1997, F11:1991-2000, F13:1995-2009, F14:19"
            "97-2008, F15:1999-2006, F16:2003-2010, F17:2006-2010"
        ),
        "comment": (
            "Fractional area coverage of available SSM/I rain rate data for the"
            " wave trough. Satellite overpass data rounded to nearest 6-hr and "
            "averaged for all overlapping times and locations."
        ),
    },
    "meancloud": {
        "references": (
            "SSM/I and SSMIS data are produced by Remote Sensing Systems and sp"
            "onsored by the NASA Earth Science MEaSUREs Program and are availab"
            "le at www.remss.com."
        ),
        "platform": "DMSP SSM/I",
        "instrument": (
            "F08:1987-1991, F10:1990-1997, F11:1991-2000, F13:1995-2009, F14:19"
            "97-2008, F15:1999-2006, F16:2003-2010, F17:2006-2010"
        ),
        "comment": (
            "Satellite overpass data rounded to nearest 6-hr and averaged for a"
            "ll overlapping times and locations."
        ),
    },
    "stdcloud": {
        "references": (
            "SSM/I and SSMIS data are produced by Remote Sensing Systems and sp"
            "onsored by the NASA Earth Science MEaSUREs Program and are availab"
            "le at www.remss.com."
        ),
        "platform": "DMSP SSM/I",
        "instrument": (
            "F08:1987-1991, F10:1990-1997, F11:1991-2000, F13:1995-2009, F14:19"
            "97-2008, F15:1999-2006, F16:2003-2010, F17:2006-2010"
        ),
        "comment": (
            "Satellite overpass data rounded to nearest 6-hr and averaged for a"
            "ll overlapping times and locations."
        ),
    },
    "cloud_area_fraction": {
        "references": (
            "SSM/I and SSMIS data are produced by Remote Sensing Systems and sp"
            "onsored by the NASA Earth Science MEaSUREs Program and are availab"
            "le at www.remss.com."
        ),
        "platform": "DMSP SSM/I",
        "instrument": (
            "F08:1987-1991, F10:1990-1997, F11:1991-2000, F13:1995-2009, F14:19"
            "97-2008, F15:1999-2006, F16:2003-2010, F17:2006-2010"
        ),
        "comment": (
            "Fractional area coverage of available SSM/I rain rate data for the"
            " wave trough. Satellite overpass data rounded to nearest 6-hr and "
            "averaged for all overlapping times and locations."
        ),
    },
}


def ragged_arrays(tracks, time_epoch="1900"):
    """Lay the tracks end to end into the contiguous ragged arrays the file stores.

    Returns (counts, columns), where counts has one entry per track and columns maps each
    sample variable to a float32 array of length sum(counts). A track missing one of the
    tracker's own fields contributes fill for it, which is how a run that did not compute
    wavelength still produces a readable file.

    `time_epoch` is passed to `days_since_1900` for the time column only.
    """
    counts = np.array([len(t["time"]) for t in tracks], dtype=np.int32)
    total = int(counts.sum())
    columns = {name: np.full(total, FILL, dtype=np.float32)
               for name in SAMPLE_VARIABLES}
    offset = 0
    columns_of = tuple(TRACK_VARIABLES) + tuple(COMPOSITE_TRACK_KEYS.items())
    for track in tracks:
        n = len(track["time"])
        for name, key in columns_of:
            values = track.get(key)
            if values is None:
                continue
            values = np.asarray(values, dtype=np.float64)
            if values.size != n:
                raise ValueError(
                    f"track field {key!r} has {values.size} values for {n} observations")
            if name == "time":
                values = days_since_1900(values, time_epoch)
            columns[name][offset:offset + n] = values.astype(np.float32)
        offset += n
    return counts, columns


def write_region(path, tracks, year, level, reanalysis, region,
                 reproduce_v1_metadata=True, date_created=None, time_epoch="1900"):
    """Write one region's tracks as a version 1 style netCDF file.

    `tracks` are the association stage's output for this region only; filtering by region
    is the caller's job, as it is version 1's, and `write_year` does it.

    `time_epoch` says what the track's time numbers count from. See `days_since_1900`.

    `date_created` defaults to today, which is what version 1 writes and what the metadata
    convention asks for. Pass a fixed date when byte-identical output from identical input
    matters, for a regression test or a reproducible deposit. An earlier draft made the
    fixed date the default and did not label it a divergence, so the file quietly stopped
    recording when it was made.
    """
    import datetime
    import netCDF4 as nc

    if region not in REGION_NAMES:
        raise ValueError(f"unknown region {region!r}, expected one of {REGIONS}")
    if date_created is None:
        date_created = datetime.date.today().isoformat()
    counts, columns = ragged_arrays(tracks, time_epoch=time_epoch)
    total = int(counts.sum())

    with nc.Dataset(path, "w", format="NETCDF4") as ds:
        for key, value in _global_attributes(year, level, reanalysis, region,
                                             reproduce_v1_metadata,
                                             date_created).items():
            ds.setncattr(key, value)
        if reproduce_v1_metadata:
            # DEFECT 2, reproduced. Sized to the longest track and used by nothing.
            ds.createDimension("obs", int(counts.max()) if counts.size else 0)
        ds.createDimension("trajectory", len(counts))
        ds.createDimension("sample", total)

        for name in ("count", "trajectory"):
            var = ds.createVariable(name, "i4", ("trajectory",), zlib=True,
                                    shuffle=True, complevel=2)
            for key, value in _variable_attributes(name, reproduce_v1_metadata).items():
                var.setncattr(key, value)
        ds.variables["count"][:] = counts
        ds.variables["trajectory"][:] = np.arange(len(counts), dtype=np.int32)

        for name in SAMPLE_VARIABLES:
            var = ds.createVariable(name, "f4", ("sample",), zlib=True, shuffle=True,
                                    complevel=2, fill_value=np.float32(FILL))
            for key, value in _variable_attributes(name, reproduce_v1_metadata).items():
                var.setncattr(key, value)
            var[:] = columns[name]
    return path


def write_year(directory, tracks, year, level, reanalysis, region_of=None, **kwargs):
    """Write all eight region files for one year, as output_ews_netcdf_f.m does.

    `region_of` maps a track to its basin code, defaulting to reading a "region_name" key,
    which is what version 1 calls it. Note that a track ALSO carries `wave_points`, the
    per-observation trough masks; those are a different thing and are not read here.
    Region assignment itself belongs to generate_ew_stats_f.m, which is not ported, so it
    is the caller's to supply.

    FAITHFUL: the original's loop runs over all eight regions unconditionally and creates a
    file whether or not any wave belongs to that region, so a region with no waves gets a
    file with zero trajectories and zero samples. That is asserted from the MATLAB, not
    from the deposit: only the Africa files are held here, so no empty file has been read.
    """
    import os

    if region_of is None:
        def region_of(track):
            try:
                return track["region_name"]
            except KeyError:
                raise ValueError(
                    "this track has no 'region_name', so it cannot be filed. The tracker "
                    "does not assign one: version 1 sets `region` and `region_name` in "
                    "generate_ew_stats_f.m, which is not ported, so a track coming "
                    "straight out of finalize_tracks has no basin. Assign one to each "
                    "track, or pass region_of=... to say where it comes from. Note the "
                    "tracker's own 'wave_points' key holds the trough masks and is a "
                    "different thing entirely."
                ) from None
    grouped = {code: [] for code in REGIONS}
    for track in tracks:
        code = region_of(track)
        if code not in grouped:
            raise ValueError(f"unknown region {code!r}, expected one of {REGIONS}")
        grouped[code].append(track)
    return [write_region(
        os.path.join(directory, f"{reanalysis}_ew_{level}hPa_{year}_{code}.nc"),
        grouped[code], year=year, level=level, reanalysis=reanalysis, region=code,
        **kwargs) for code in REGIONS]
