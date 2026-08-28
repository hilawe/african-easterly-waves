"""Tests for the version 1 record writer.

THE MUTATION LIST, WRITTEN BEFORE THE ASSERTIONS EXISTED, executable as
tests/mutations_v1port_record.py.

    R1   a variable is dropped, or the definition order changes
    R2   the ragged layout is wrong: counts do not partition the sample dimension, or a
         track's observations land at the wrong offset
    R3   the trajectory identifiers are not zero-based and consecutive
    R4   the fill value is not the one version 1 declares, so uncomputed statistics read
         as data rather than as missing
    R5   a track field is written into the wrong variable
    R6   a track field the tracker did not compute is written as zero rather than fill
    R7   the count variable loses its sample_dimension attribute, which is what makes the
         file a contiguous ragged array rather than three unrelated arrays
    R8   the trajectory variable loses cf_role
    R9   the time units or calendar change, so the record's epoch moves
    R10  version 1's longitude-units defect is silently corrected under the faithful
         setting, or not corrected under the corrected one
    R11  the unused obs dimension is dropped under the faithful setting, or kept under
         the corrected one
    R12  the area-fraction units inconsistency is not reproduced faithfully
    R13  the summary, id or coverage attributes stop tracking year, level and region
    R14  a mismatched-length track field is written silently instead of refused
    R15  an unknown region is accepted
    R16  a year writes only the regions that have waves, not all eight
    R17  a track in an unrecognized region is dropped rather than refused
    R18  the basin key the writer reads is not the one the tracker writes
    R19  the time values are written without the epoch shift version 1 applies, so the
         units attribute is right and every timestamp is eight centuries out
    R20  the satellite provenance attributes are dropped under the faithful setting, or
         kept under the corrected one
    R21  date_created is a fixed placeholder rather than the day the file was written

THE PUBLISHED RECORD IS THE ORACLE HERE, not an analytic case. C00784's own files are in
data/aewc, so the port's structure and metadata are checked against a file version 1
actually wrote. Two long names were wrong in the first draft ("anomaly" for the published
"anomalies") and that check is what found them. Those tests skip when the record is
absent, so the suite still runs on a clone without the data; the structural tests below do
not depend on it.
"""

import numpy as np
import pytest

nc = pytest.importorskip("netCDF4")

from aew.v1port import record as R  # noqa: E402

PUBLISHED = "data/aewc/ERA-Int_ew_700hPa_2005_AFR.nc"

# WRITTEN OUT RATHER THAN READ FROM THE MODULE, on purpose. Three mutations survived the
# first run of the catalog because the tests that were supposed to pin the variable set,
# the fill value and the time epoch compared the written file against the module's own
# constants, so changing a constant moved both sides of the assertion together and the
# tests passed. The published record would have caught all three, but it is untracked data
# and the mutation checker stages only tracked files, so those tests skip exactly when they
# are needed. Duplicating the values here is the point: the copy has to be changed by hand,
# which is what makes it a check.
EXPECTED_VARIABLES = (
    "count", "trajectory", "time", "lat", "lon", "maxlat", "meanlonatmaxlat",
    "minlat", "meanlonatminlat", "wavelength",
    "meanrv", "maxrv", "minrv", "stdrv",
    "meancrv", "maxcrv", "mincrv", "stdcrv",
    "meansrv", "maxsrv", "minsrv", "stdsrv",
    "meanolr", "stdolr", "olr_area_fraction", "meanolra", "stdolra",
    "meanctb", "stdctb", "ctb_area_fraction",
    "meantpw", "stdtpw", "tpw_area_fraction",
    "meanrain", "stdrain", "rain_area_fraction",
    "meancloud", "stdcloud", "cloud_area_fraction",
)
EXPECTED_FILL = -999.0
EXPECTED_TIME_EPOCH = "days since 1900-01-01 00:00:00"


def track(n, start_lat=10.0, start_lon=0.0, with_wavelength=True):
    out = {
        "time": [40000.0 + 0.25 * i for i in range(n)],
        "meanlat": [start_lat + 0.1 * i for i in range(n)],
        "meanlon": [start_lon - 1.5 * i for i in range(n)],
        "maxlat": [start_lat + 1.0] * n,
        "minlat": [start_lat - 1.0] * n,
        "meanlon_maxlat": [start_lon] * n,
        "meanlon_minlat": [start_lon] * n,
    }
    if with_wavelength:
        out["wavelength"] = [2500.0 + i for i in range(n)]
    return out


@pytest.fixture
def written(tmp_path):
    def _write(tracks, **kwargs):
        kwargs.setdefault("year", 2005)
        kwargs.setdefault("level", 700)
        kwargs.setdefault("reanalysis", "ERA-Int")
        kwargs.setdefault("region", "AFR")
        path = tmp_path / "out.nc"
        R.write_region(str(path), tracks, **kwargs)
        return str(path)
    return _write


# --- the ragged layout ----------------------------------------------------------------

def test_counts_partition_the_sample_dimension():
    """R2."""
    tracks = [track(4), track(7), track(3)]
    counts, columns = R.ragged_arrays(tracks)
    assert list(counts) == [4, 7, 3]
    assert all(col.size == 14 for col in columns.values())


def test_each_track_lands_at_the_offset_its_counts_imply():
    """R2, the part a matching total would not catch. The second track's latitudes must
    begin exactly where the first track's end."""
    first, second = track(4, start_lat=10.0), track(7, start_lat=-20.0)
    _, columns = R.ragged_arrays([first, second])
    assert columns["lat"][:4] == pytest.approx(first["meanlat"])
    assert columns["lat"][4:11] == pytest.approx(second["meanlat"])


def test_a_field_the_tracker_did_not_compute_is_fill_not_zero():
    """R6. Writing zero would make an uncomputed wavelength read as a real measurement of
    zero kilometers."""
    _, columns = R.ragged_arrays([track(5, with_wavelength=False)])
    assert np.all(columns["wavelength"] == EXPECTED_FILL)
    assert not np.any(columns["wavelength"] == 0.0)


def test_the_composited_statistics_are_all_fill():
    _, columns = R.ragged_arrays([track(5)])
    for name in R.COMPOSITE_VARIABLES:
        assert np.all(columns[name] == EXPECTED_FILL), name


def test_a_track_field_of_the_wrong_length_is_refused():
    """R14. Silently truncating or padding would corrupt every later track's offset."""
    bad = track(5)
    bad["wavelength"] = [1.0, 2.0]
    with pytest.raises(ValueError, match="wavelength"):
        R.ragged_arrays([bad])


def test_no_tracks_gives_empty_arrays():
    counts, columns = R.ragged_arrays([])
    assert counts.size == 0
    assert all(col.size == 0 for col in columns.values())


# --- the written file -------------------------------------------------------------------

def test_every_variable_is_written_in_version_ones_order(written):
    """R1. Against the written-out list, not the module's own."""
    with nc.Dataset(written([track(4), track(6)])) as ds:
        assert list(ds.variables) == list(EXPECTED_VARIABLES)
        assert len(ds.variables) == 39


def test_the_dimensions_match_the_tracks(written):
    with nc.Dataset(written([track(4), track(6)])) as ds:
        assert len(ds.dimensions["trajectory"]) == 2
        assert len(ds.dimensions["sample"]) == 10


def test_trajectory_identifiers_are_zero_based_and_consecutive(written):
    """R3. The published file starts at zero, which the original gets by initializing its
    counter to -1 before the write loop."""
    with nc.Dataset(written([track(4), track(6), track(2)])) as ds:
        assert list(ds.variables["trajectory"][:]) == [0, 1, 2]
        assert list(ds.variables["count"][:]) == [4, 6, 2]


def test_uncomputed_statistics_read_back_as_missing(written):
    """R4. The fill value is what makes a written -999 mean absent rather than present."""
    with nc.Dataset(written([track(5)])) as ds:
        var = ds.variables["meanolr"]
        assert var._FillValue == pytest.approx(EXPECTED_FILL)
        assert np.ma.getmaskarray(var[:]).all()


def test_each_track_field_reaches_its_own_variable(written):
    """R5. Latitude and its maximum differ in this fixture, so a swap is visible."""
    t = track(4)
    with nc.Dataset(written([t])) as ds:
        read = {n: np.asarray(ds.variables[n][:]) for n in
                ("lat", "lon", "maxlat", "wavelength")}
    assert read["lat"] == pytest.approx(t["meanlat"], abs=1e-4)
    assert read["lon"] == pytest.approx(t["meanlon"], abs=1e-4)
    assert read["maxlat"] == pytest.approx(t["maxlat"], abs=1e-4)
    assert read["wavelength"] == pytest.approx(t["wavelength"], abs=1e-2)


def test_the_ragged_array_attributes_are_present(written):
    """R7 and R8. Without these two the file is three arrays a reader cannot relate."""
    with nc.Dataset(written([track(4)])) as ds:
        assert ds.variables["count"].sample_dimension == "sample"
        assert ds.variables["trajectory"].cf_role == "trajectory_id"


def test_the_time_epoch_is_the_records_own(written):
    """R9."""
    with nc.Dataset(written([track(4)])) as ds:
        assert ds.variables["time"].units == EXPECTED_TIME_EPOCH
        assert ds.variables["time"].calendar == "gregorian"


def test_the_identifying_attributes_track_the_arguments(written):
    """R13."""
    path = written([track(4)], year=1999, level=850, reanalysis="NCEP", region="NAL")
    with nc.Dataset(path) as ds:
        assert ds.id == "NCEP_ew_850hPa_1999_NAL.nc"
        assert "850 hPa" in ds.summary and "North Atlantic" in ds.summary
        assert "1999" in ds.summary
        assert ds.time_coverage_start == "1999-01-01T00:00:00Z"
        assert ds.time_coverage_end == "1999-12-31T18:00:00Z"
        assert ds.source == "NCEP/NCAR Reanalysis"


def test_an_unknown_region_is_refused(written):
    """R15."""
    with pytest.raises(ValueError, match="unknown region"):
        written([track(4)], region="XXX")


# --- version 1's metadata defects, reproduced and corrected -----------------------------

def test_the_longitude_units_defect_is_reproduced_by_default(written):
    """R10. Version 1 labels longitude degrees_north, and the published record carries it,
    so a comparison against that record needs the port to carry it too."""
    with nc.Dataset(written([track(4)])) as ds:
        assert ds.geospatial_lon_units == "degrees_north"


def test_the_longitude_units_defect_is_corrected_on_request(written):
    with nc.Dataset(written([track(4)], reproduce_v1_metadata=False)) as ds:
        assert ds.geospatial_lon_units == "degrees_east"
        assert ds.geospatial_lat_units == "degrees_north"


def test_the_unused_obs_dimension_is_reproduced_by_default(written):
    """R11. Sized to the longest track and referenced by no variable."""
    with nc.Dataset(written([track(4), track(9)])) as ds:
        assert len(ds.dimensions["obs"]) == 9
        assert all("obs" not in v.dimensions for v in ds.variables.values())


def test_the_unused_obs_dimension_is_dropped_on_request(written):
    with nc.Dataset(written([track(4)], reproduce_v1_metadata=False)) as ds:
        assert "obs" not in ds.dimensions


def test_the_area_fraction_units_inconsistency_is_reproduced(written):
    """R12. Two of the five carry an empty units string and three carry none. Reproduced
    variable by variable rather than uniformly, because a uniform empty string would be a
    different file from the published one."""
    with nc.Dataset(written([track(4)])) as ds:
        for name in R.FRACTION_VARIABLES:
            var = ds.variables[name]
            if name in R._V1_EMPTY_UNITS:
                assert var.units == "", name
            else:
                assert "units" not in var.ncattrs(), name


def test_the_area_fraction_units_are_uniform_when_corrected(written):
    with nc.Dataset(written([track(4)], reproduce_v1_metadata=False)) as ds:
        for name in R.FRACTION_VARIABLES:
            assert ds.variables[name].units == "1", name


def test_the_two_settings_differ_only_in_metadata(written):
    """Neither setting may change a value, a dimension that carries data, or a variable."""
    tracks = [track(4), track(6)]
    faithful = written(tracks)
    with nc.Dataset(faithful) as ds:
        values = {n: np.array(v[:]) for n, v in ds.variables.items()}
        names, samples = list(ds.variables), len(ds.dimensions["sample"])
    corrected = written(tracks, reproduce_v1_metadata=False)
    with nc.Dataset(corrected) as ds:
        assert list(ds.variables) == names
        assert len(ds.dimensions["sample"]) == samples
        for n, v in ds.variables.items():
            assert np.array_equal(np.array(v[:]), values[n], equal_nan=True), n


# --- the whole year, all eight regions ---------------------------------------------------

def test_a_year_writes_all_eight_regions_including_the_empty_ones(tmp_path):
    """R16. The original's loop is over the region list, not over the regions that have
    waves, so a region with nothing in it still gets a file."""
    tracks = [dict(track(4), region_name="AFR"), dict(track(6), region_name="NAL"),
              dict(track(3), region_name="AFR")]
    paths = R.write_year(str(tmp_path), tracks, 2005, 700, "ERA-Int")
    assert len(paths) == 8
    assert [__import__("os").path.basename(p) for p in paths] == [
        f"ERA-Int_ew_700hPa_2005_{code}.nc" for code in R.REGIONS]
    with nc.Dataset(paths[R.REGIONS.index("AFR")]) as ds:
        assert len(ds.dimensions["trajectory"]) == 2
        assert len(ds.dimensions["sample"]) == 7
    with nc.Dataset(paths[R.REGIONS.index("SEP")]) as ds:
        assert len(ds.dimensions["trajectory"]) == 0
        assert len(ds.dimensions["sample"]) == 0
        assert list(ds.variables) == list(EXPECTED_VARIABLES)


def test_a_track_in_an_unknown_region_is_refused(tmp_path):
    """R17. Silently dropping it would lose waves without saying so."""
    with pytest.raises(ValueError, match="unknown region"):
        R.write_year(str(tmp_path), [dict(track(4), region_name="ZZZ")], 2005, 700, "ERA-Int")


def test_the_region_accessor_can_be_supplied(tmp_path):
    paths = R.write_year(str(tmp_path), [track(4)], 2005, 700, "ERA-Int",
                         region_of=lambda t: "SAL")
    with nc.Dataset(paths[R.REGIONS.index("SAL")]) as ds:
        assert len(ds.dimensions["trajectory"]) == 1


# --- checked against the record version 1 actually wrote ---------------------------------

published = pytest.mark.skipif(
    not __import__("os").path.exists(PUBLISHED),
    reason="the published C00784 files are not in this clone")


@published
def test_the_variable_set_and_order_match_the_published_record():
    """R1, against the record rather than against this module's own table."""
    with nc.Dataset(PUBLISHED) as ds:
        assert list(ds.variables) == list(R.variable_names())


@published
def test_the_port_and_the_published_record_carry_THE_SAME_attributes():
    """R20, and the shape of this test is the finding rather than the content.

    The first version compared only the attributes the port chose to emit, so anything the
    port left out was invisible to it. It reported a clean match while the writer was
    omitting the satellite provenance from all seventeen composited variables. A one-sided
    comparison against an oracle is not a check, it is a check of the half you already
    thought about, so this walks BOTH directions: nothing missing, nothing extra, no value
    different.
    """
    missing, extra, wrong = [], [], []
    with nc.Dataset(PUBLISHED) as ds:
        for name in EXPECTED_VARIABLES:
            real = ds.variables[name]
            mine = R._variable_attributes(name, True)
            published_attrs = {k: str(getattr(real, k)) for k in real.ncattrs()
                               if k != "_FillValue"}
            extra += [f"{name}.{k}" for k in mine if k not in published_attrs]
            missing += [f"{name}.{k}" for k in published_attrs if k not in mine]
            wrong += [f"{name}.{k}" for k, v in mine.items()
                      if k in published_attrs and published_attrs[k] != str(v)]
    assert missing == [], f"the port omits attributes the record carries: {missing}"
    assert extra == [], f"the port invents attributes the record lacks: {extra}"
    assert wrong == [], f"values differ: {wrong}"


@published
def test_the_composited_variables_carry_their_satellite_provenance():
    """R20 stated positively, so the count cannot quietly drop to zero. Eighteen variables
    in the published file name a source, references, platform, instrument or binning
    comment; seventeen are composited fields and the eighteenth is wavelength."""
    keys = ("source", "references", "platform", "instrument", "comment")
    with nc.Dataset(PUBLISHED) as ds:
        published = {n for n in EXPECTED_VARIABLES
                     if any(k in ds.variables[n].ncattrs() for k in keys)}
    ported = {n for n in EXPECTED_VARIABLES
              if any(k in R._variable_attributes(n, True) for k in keys)}
    assert published == ported
    assert len(ported) == 18


def test_the_faithful_setting_carries_the_satellite_provenance():
    """R20, bound WITHOUT the published file, on purpose.

    The two-directional oracle test above is the real check, and it skips whenever the
    C00784 files are absent, which includes every run of the mutation checker because that
    stages only tracked files and the data is untracked. So a mutation deleting the
    provenance survived while the suite stayed green. These values are written out by hand
    for the same reason the variable list is.
    """
    tpw = R._variable_attributes("meantpw", True)
    assert tpw["platform"] == "DMSP SSM/I"
    assert tpw["instrument"].startswith("F08:1987-1991")
    assert "Remote Sensing Systems" in tpw["references"]
    olr = R._variable_attributes("meanolr", True)
    assert "Outgoing Longwave Radiation" in olr["source"]
    assert "Lee" in olr["references"]
    ctb = R._variable_attributes("meanctb", True)
    assert "references" in ctb
    carriers = [n for n in EXPECTED_VARIABLES
                if any(k in R._variable_attributes(n, True)
                       for k in ("source", "references", "platform", "instrument",
                                 "comment"))]
    assert len(carriers) == 18


def test_the_corrected_setting_drops_the_satellite_provenance():
    """The other half of R20. Under the corrected setting the provenance goes, because
    naming an instrument behind a variable this port fills entirely with the fill value
    claims more than the file holds."""
    keys = ("source", "references", "platform", "instrument")
    for name in R.COMPOSITE_VARIABLES:
        attrs = R._variable_attributes(name, False)
        assert not any(k in attrs for k in keys), name
    assert "comment" in R._variable_attributes("wavelength", False), \
        "wavelength's comment describes a definition this port implements, so it stays"


@published
def test_the_global_attributes_the_port_writes_match_the_published_record():
    with nc.Dataset(PUBLISHED) as ds:
        mine = R._global_attributes(2005, 700, "ERA-Int", "AFR", True,
                                    ds.date_created)
        mismatches = [f"{k}: {v!r} != {getattr(ds, k, '<absent>')!r}"
                      for k, v in mine.items()
                      if str(getattr(ds, k, "<absent>")) != str(v)]
    assert mismatches == []


@published
def test_the_published_record_carries_the_three_defects_this_port_reproduces():
    """The defects are recorded as observed in the shipped files, not inferred from the
    MATLAB, so the claim is bound to the files themselves."""
    with nc.Dataset(PUBLISHED) as ds:
        assert ds.geospatial_lon_units == "degrees_north"
        assert "obs" in ds.dimensions
        assert all("obs" not in v.dimensions for v in ds.variables.values())
        assert ds.variables["olr_area_fraction"].units == ""
        assert "units" not in ds.variables["tpw_area_fraction"].ncattrs()


@published
def test_a_port_written_file_reads_back_the_same_way_as_the_published_one(written):
    """The point of the whole stage: one reader, both files. Walk the published file's
    ragged layout and a port-written file's with the same code."""
    def waves(path):
        with nc.Dataset(path) as ds:
            counts = np.array(ds.variables["count"][:])
            lat = np.array(ds.variables["lat"][:])
            offsets = np.concatenate([[0], np.cumsum(counts)])
            return [lat[offsets[i]:offsets[i + 1]] for i in range(len(counts))]

    from_record = waves(PUBLISHED)
    assert len(from_record) == 485 and sum(len(w) for w in from_record) == 7470

    tracks = [track(4), track(9), track(2)]
    from_port = waves(written(tracks))
    assert [len(w) for w in from_port] == [4, 9, 2]
    assert from_port[1] == pytest.approx(tracks[1]["meanlat"], abs=1e-4)


# --- the time epoch ------------------------------------------------------------------

def test_a_matlab_datenum_becomes_the_value_the_published_file_holds():
    """R19. Version 1 writes `single(time - datenum('01/01/1900'))`. The published 2005
    file's first observation is 38351.25, and the MATLAB datenum for that instant is
    732313.25. Getting this wrong leaves the units attribute correct and every value eight
    centuries out, which nothing in the file reveals."""
    out = R.days_since_1900([732313.25, 732313.5], "matlab")
    assert out == pytest.approx([38351.25, 38351.5])
    assert R.MATLAB_DATENUM_1900 == 693962


def test_already_converted_values_pass_through():
    assert R.days_since_1900([38351.25], "1900") == pytest.approx([38351.25])


def test_a_python_ordinal_converts_too():
    import datetime
    ordinal = datetime.date(2005, 1, 1).toordinal() + 0.25
    assert R.days_since_1900([ordinal], "python") == pytest.approx([38351.25])


def test_an_unstated_epoch_is_refused_rather_than_guessed():
    """R19. The whole point: a writer that guesses produces a plausible wrong file."""
    with pytest.raises(ValueError, match="time_epoch"):
        R.days_since_1900([1.0], "unix")
    with pytest.raises(ValueError, match="time_epoch"):
        R.days_since_1900([1.0], None)


def test_the_written_time_column_carries_the_conversion(written):
    """The conversion has to reach the file, not just the helper."""
    matlab_track = dict(track(4))
    matlab_track["time"] = [732313.25 + 0.25 * i for i in range(4)]
    path = written([matlab_track], time_epoch="matlab")
    with nc.Dataset(path) as ds:
        values = np.asarray(ds.variables["time"][:])
    assert values == pytest.approx([38351.25, 38351.5, 38351.75, 38352.0], abs=1e-2)


# --- date_created ---------------------------------------------------------------------

def test_date_created_defaults_to_today(written):
    """R21. Version 1 writes `datestr(now)`, and the metadata convention asks for the real
    creation date. A first draft defaulted to a fixed placeholder for reproducible output,
    which is a legitimate thing to want and the wrong default."""
    import datetime
    with nc.Dataset(written([track(4)])) as ds:
        assert ds.date_created == datetime.date.today().isoformat()


def test_a_fixed_date_can_be_requested_for_reproducible_output(written):
    with nc.Dataset(written([track(4)], date_created="2014-03-01")) as ds:
        assert ds.date_created == "2014-03-01"
