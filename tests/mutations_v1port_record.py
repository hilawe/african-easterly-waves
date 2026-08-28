"""Mutations for the version 1 record writer.

The list was written before the assertions existed and is reproduced in the docstring of
tests/test_v1port_record.py. Run with the repository's mutation checker against
src/aew/v1port/record.py.
"""


def _sub(old, new):
    return lambda s: s.replace(old, new)


MUTATIONS = {
    # R1 a variable is dropped from the definition set
    "a_variable_is_dropped": _sub(
        '    "meancloud", "stdcloud", "cloud_area_fraction",\n)',
        '    "meancloud", "stdcloud",\n)'),
    # R1 the definition order changes
    "variable_order_changes": _sub(
        'SAMPLE_VARIABLES = tuple(name for name, _ in TRACK_VARIABLES) '
        "+ COMPOSITE_VARIABLES",
        "SAMPLE_VARIABLES = COMPOSITE_VARIABLES "
        "+ tuple(name for name, _ in TRACK_VARIABLES)"),
    # R2 every track is written at the start rather than at its own offset
    "tracks_all_written_at_offset_zero": _sub(
        "        offset += n",
        "        offset += 0"),
    # R2 the counts do not describe the sample dimension
    "counts_do_not_match_the_samples": _sub(
        '    counts = np.array([len(t["time"]) for t in tracks], dtype=np.int32)',
        '    counts = np.array([len(t["time"]) + 1 for t in tracks], dtype=np.int32)'),
    # R3 the trajectory identifiers are one-based
    "trajectory_ids_are_one_based": _sub(
        '        ds.variables["trajectory"][:] = np.arange(len(counts), dtype=np.int32)',
        '        ds.variables["trajectory"][:] = np.arange(1, len(counts) + 1, '
        "dtype=np.int32)"),
    # R4 the fill value moves off the one version 1 declares
    "fill_value_changes": _sub(
        "FILL = -999.0",
        "FILL = -9999.0"),
    # R5 a track field is written into the wrong variable
    "a_track_field_goes_to_the_wrong_variable": _sub(
        '    ("lat", "meanlat"),\n    ("lon", "meanlon"),',
        '    ("lat", "maxlat"),\n    ("lon", "meanlon"),'),
    # R6 an uncomputed field is written as zero rather than fill
    "uncomputed_fields_written_as_zero": _sub(
        "    columns = {name: np.full(total, FILL, dtype=np.float32)\n"
        "               for name in SAMPLE_VARIABLES}",
        "    columns = {name: np.zeros(total, dtype=np.float32)\n"
        "               for name in SAMPLE_VARIABLES}"),
    # R7 the count variable loses the attribute that makes the file a ragged array
    "count_loses_sample_dimension": _sub(
        '        return {"long_name": "number of observations for the easterly wave",\n'
        '                "sample_dimension": "sample"}',
        '        return {"long_name": "number of observations for the easterly wave"}'),
    # R8 the trajectory variable loses its CF role
    "trajectory_loses_cf_role": _sub(
        '        return {"long_name": "easterly wave trajectory", '
        '"cf_role": "trajectory_id"}',
        '        return {"long_name": "easterly wave trajectory"}'),
    # R9 the time epoch moves
    "time_epoch_moves": _sub(
        'TIME_EPOCH = "days since 1900-01-01 00:00:00"',
        'TIME_EPOCH = "days since 1970-01-01 00:00:00"'),
    # R10 the longitude-units defect is silently corrected under the faithful setting
    "longitude_units_defect_not_reproduced": _sub(
        '    lon_units = "degrees_north" if reproduce_v1_metadata else "degrees_east"',
        '    lon_units = "degrees_east"'),
    # R10 the corrected setting does not correct it
    "longitude_units_not_corrected_on_request": _sub(
        '    lon_units = "degrees_north" if reproduce_v1_metadata else "degrees_east"',
        '    lon_units = "degrees_north"'),
    # R11 the unused dimension is dropped even under the faithful setting
    "obs_dimension_never_written": _sub(
        '            ds.createDimension("obs", int(counts.max()) if counts.size else 0)',
        "            pass"),
    # R11 the unused dimension is kept even under the corrected setting
    "obs_dimension_always_written": _sub(
        "        if reproduce_v1_metadata:\n"
        "            # DEFECT 2, reproduced. Sized to the longest track and used by "
        "nothing.\n"
        '            ds.createDimension("obs", int(counts.max()) if counts.size else 0)',
        '        ds.createDimension("obs", int(counts.max()) if counts.size else 0)'),
    # R12 the fraction-units inconsistency is flattened to a uniform empty string
    "fraction_units_made_uniform": _sub(
        "            if name in _V1_EMPTY_UNITS:\n"
        '                attrs["units"] = ""',
        '            attrs["units"] = ""'),
    # R13 the summary stops tracking the region
    "summary_ignores_the_region": _sub(
        '                    f"{REGION_NAMES[region]} for {year:04d}"),',
        '                    f"Africa for {year:04d}"),'),
    # R13 the identifier stops tracking the level
    "id_ignores_the_level": _sub(
        '        "id": f"{reanalysis}_ew_{level}hPa_{year}_{region}.nc",',
        '        "id": f"{reanalysis}_ew_700hPa_{year}_{region}.nc",'),
    # R14 a mismatched-length field is written silently
    "length_mismatch_not_refused": _sub(
        "            if values.size != n:\n"
        "                raise ValueError(\n"
        '                    f"track field {key!r} has {values.size} values for {n} '
        'observations")',
        "            if values.size != n:\n                continue"),
    # R16 only the regions that have waves are written
    "empty_regions_skipped": _sub(
        "    return [write_region(\n"
        '        os.path.join(directory, f"{reanalysis}_ew_{level}hPa_{year}_{code}.nc"),\n'
        "        grouped[code], year=year, level=level, reanalysis=reanalysis, "
        "region=code,\n"
        "        **kwargs) for code in REGIONS]",
        "    return [write_region(\n"
        '        os.path.join(directory, f"{reanalysis}_ew_{level}hPa_{year}_{code}.nc"),\n'
        "        grouped[code], year=year, level=level, reanalysis=reanalysis, "
        "region=code,\n"
        "        **kwargs) for code in REGIONS if grouped[code]]"),
    # R17 a track in an unrecognized region is dropped instead of refused
    "unknown_region_track_dropped": _sub(
        "        if code not in grouped:\n"
        '            raise ValueError(f"unknown region {code!r}, expected one of '
        '{REGIONS}")\n'
        "        grouped[code].append(track)",
        "        if code in grouped:\n            grouped[code].append(track)"),
    # R22 the composited fields are hard-wired to fill, so a track that carries them has
    # its statistics silently discarded
    "composited_fields_never_read_from_the_track": _sub(
        "    columns_of = tuple(TRACK_VARIABLES) + tuple(COMPOSITE_TRACK_KEYS.items())",
        "    columns_of = tuple(TRACK_VARIABLES)"),
    # R22 a composited variable reads the wrong track field; the names are not guessable
    "composited_field_reads_the_wrong_key": _sub(
        '    "meanctb": "meanclaus", "stdctb": "stdclaus", '
        '"ctb_area_fraction": "claus_cover",',
        '    "meanctb": "meanctb", "stdctb": "stdclaus", '
        '"ctb_area_fraction": "claus_cover",'),
    # R23 a track with no basin raises a bare KeyError instead of saying what is wrong
    "missing_basin_gives_a_bare_keyerror": _sub(
        "            try:\n"
        '                return track["region_name"]\n'
        "            except KeyError:",
        '            return track["region_name"]\n'
        "            if False:"),
    # R19 the epoch shift is dropped, so the units attribute is right and the values are
    # eight centuries out
    "time_epoch_shift_dropped": _sub(
        '    offsets = {"1900": 0.0, "matlab": float(MATLAB_DATENUM_1900),\n'
        '               "python": float(PYTHON_ORDINAL_1900)}',
        '    offsets = {"1900": 0.0, "matlab": 0.0, "python": 0.0}'),
    # R19 the shift is the wrong way round
    "time_epoch_shift_inverted": _sub(
        "    return np.asarray(times, dtype=np.float64) - offsets[time_epoch]",
        "    return np.asarray(times, dtype=np.float64) + offsets[time_epoch]"),
    # R19 an unstated epoch is guessed instead of refused
    "unknown_time_epoch_guessed": _sub(
        "    if time_epoch not in offsets:",
        "    if False:"),
    # R19 the conversion never reaches the written column
    "conversion_not_applied_to_the_time_column": _sub(
        '            if name == "time":\n'
        "                values = days_since_1900(values, time_epoch)",
        "            pass"),
    # R20 the satellite provenance is dropped under the faithful setting
    "provenance_dropped_when_faithful": _sub(
        "    if reproduce_v1_metadata:\n"
        "        attrs.update(_PROVENANCE.get(name, {}))",
        "    if False:\n        attrs.update(_PROVENANCE.get(name, {}))"),
    # R20 the provenance is kept under the corrected setting too
    "provenance_kept_when_corrected": _sub(
        "    if reproduce_v1_metadata:\n"
        "        attrs.update(_PROVENANCE.get(name, {}))",
        "    attrs.update(_PROVENANCE.get(name, {}))"),
    # R21 date_created is a fixed placeholder rather than the day of writing
    "date_created_is_a_fixed_placeholder": _sub(
        "    if date_created is None:\n"
        "        date_created = datetime.date.today().isoformat()",
        '    if date_created is None:\n        date_created = "1970-01-01"'),
    # R15 an unknown region is accepted
    "unknown_region_accepted": _sub(
        "    if region not in REGION_NAMES:\n"
        '        raise ValueError(f"unknown region {region!r}, expected one of '
        '{REGIONS}")',
        "    pass"),
}
