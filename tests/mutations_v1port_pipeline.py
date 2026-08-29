"""Mutations for the whole tracker run end to end.

The list was written before the assertions existed and is reproduced in the docstring of
tests/test_v1port_pipeline.py. Run with the repository's mutation checker against
src/aew/v1port/pipeline.py.

These bind WIRING rather than a single function's behavior, which is a different kind of
defect and the kind independent checking found: two functions each faithful alone and
wrong as a pair. Most of the mutations below leave every module untouched and only change
how they are joined.
"""


def _sub(old, new):
    return lambda s: s.replace(old, new)


MUTATIONS = {
    # P1, WITHDRAWN AND NOT REPLACED. Swapping the two is measurably a no-op, and the
    # reason is a coincidence of two constants rather than anything about the order: a
    # track association can extend was matched at most two steps ago, two steps is half a
    # day, and the prune's recency clause keeps anything seen within half a day INCLUSIVE.
    # So the pre-association state of a track the association touches always passes the
    # prune too, and pruning after keeps a superset of what pruning before keeps, with the
    # sets equal. Measured over a fixture with a vanishing wave, a stationary wave and a
    # healthy one: identical output. `test_the_prunes_recency_window_is_what_makes_the_
    # order_moot` binds the relationship between the constants, so this becomes reachable
    # again the moment either changes.
    # P1 the prune is told the timestep had candidates when it did not
    "prune_told_the_step_had_candidates": _sub(
        "        tracks, states = prune_stale_tracks(tracks, states, step, "
        "float(times[step]),\n"
        "                                            waves)",
        "        tracks, states = prune_stale_tracks(tracks, states, step, "
        "float(times[step]),\n"
        "                                            [object()])"),
    # P2 the coarse fields are never decimated
    "coarse_fields_not_decimated": _sub(
        "    coarse = {name: clim.gaussian_decimate(field, native_deg, "
        "coarse_resolution_deg)\n",
        "    coarse = {name: field\n"),
    # P2 the coarse coordinates are smoothed like the data
    "coarse_coordinates_smoothed": _sub(
        "        clim.subsample_coordinates(longrid[0, :], stride),\n"
        "        clim.subsample_coordinates(latgrid[:, 0], stride))[::-1]",
        "        clim.gaussian_decimate(longrid, input_resolution, "
        "output_resolution)[0, :],\n"
        "        clim.gaussian_decimate(latgrid, input_resolution, "
        "output_resolution)[:, 0])[::-1]"),
    # P3 the native spacing is read from longitude rather than latitude
    "native_spacing_read_from_the_wrong_axis": _sub(
        "    native_deg = abs(float(latgrid[1, 0] - latgrid[0, 0]))",
        "    native_deg = abs(float(latgrid[0, 1] - latgrid[0, 0]))"),
    # P3 a request to refine is accepted
    "refining_is_accepted": _sub(
        "    if coarse_resolution_deg < native_deg:",
        "    if False:"),
    # P4 the domain subset is not applied to the fine grid
    "fine_grid_not_subset": _sub(
        "    def cut_fine(field):\n"
        "        return field[:, rows_f, :][:, :, cols_f]",
        "    def cut_fine(field):\n        return field"),
    # P4 the domain subset is not applied to the coarse grid
    "coarse_grid_not_subset": _sub(
        "    def cut_coarse(field):\n"
        "        return field[:, rows_c, :][:, :, cols_c]",
        "    def cut_coarse(field):\n        return field"),
    # P5 detection is handed the coarse grid where it expects the fine one
    "fine_and_coarse_swapped_at_detection": _sub(
        "            latgrid_f, longrid_f, anomaly_f[step],",
        "            latgrid_c, longrid_c, anomaly_c[step],"),
    # P6 the advection is computed without version 1's latitude flip
    "advection_flip_dropped": _sub(
        "    out[:, flip, :] = advection_of_vorticity(\n"
        "        lat_d[flip, :], longrid, u_d[:, flip, :], v_d[:, flip, :], "
        "anomaly_d[:, flip, :])",
        "    out[:, :, :] = advection_of_vorticity(\n"
        "        lat_d, longrid, u_d, v_d, anomaly_d)"),
    # P6 the input row order is taken on trust, so ascending input gets a sign-flipped
    # advection field and nothing says so
    "row_order_assumed_descending": _sub(
        "    if latitude_descends(latgrid):",
        "    if True:"),
    # P6 the orientation is applied but never undone, so the output rows are reversed
    "orientation_not_restored": _sub(
        "    return out[:, ::-1, :] if restore else out",
        "    return out"),
    # P6 the row-order test reads the wrong axis
    "row_order_read_from_the_wrong_axis": _sub(
        "    return float(latgrid[1, 0]) < float(latgrid[0, 0])",
        "    return float(latgrid[0, 1]) < float(latgrid[0, 0])"),
    # P7, WITHDRAWN, and the reason is a finding about version 1 rather than about the
    # test. Predicting from the RAW fine winds instead of the smoothed ones produced
    # identical tracks on a fixture built to separate them: a wind field whose raw median
    # over a trough is zero and whose smoothed median is about -9 m/s. The tracks were the
    # same length and travelled the same 21 degrees either way.
    #
    # The reason is the search polygon. It is inflated to a minimum of 75 square degrees,
    # about ten degrees across, while the speed ceiling lets a wave move at most 4.9 degrees
    # in six hours. So the polygon reaches further than any admissible wave can travel, and
    # a track matches its own wave whether or not the prediction points anywhere near it.
    # WITHIN VERSION 1'S OWN PARAMETERS THE WIND PREDICTION BARELY AFFECTS MATCHING; what it
    # affects is which of several candidates is nearest, and the polygon size dominates the
    # rest. That is worth knowing before anyone tunes the prediction expecting it to matter.
    # The smoothing is still applied, because version 1 applies it and it does change the
    # stored predictions, and `test_the_median_wind_reads_the_waves_own_cells` binds the
    # median itself.
    # P8 the median does not ignore missing values
    "median_does_not_ignore_missing": _sub(
        "            return float(np.nanmedian(values))",
        "            return float(np.median(values))"),
    # P9 the median is taken over the whole field rather than the wave's own cells
    "median_over_the_whole_field": _sub(
        "        values = field_2d[np.asarray(mask, dtype=bool)]",
        "        values = field_2d.ravel()"),
    # P10 the exclusivity flag never reaches the association step
    "exclusivity_not_passed_through": _sub(
        "                                        u_median, v_median, exclusive=exclusive)",
        "                                        u_median, v_median, exclusive=False)"),
    # P11 the final filter is skipped
    "final_filter_skipped": _sub(
        "    return finalize_tracks(tracks, total_steps=times.size)",
        "    return tracks"),
    # P11 the final filter is not told how long the run was, so its gate never fires
    "final_filter_not_told_the_run_length": _sub(
        "    return finalize_tracks(tracks, total_steps=times.size)",
        "    return finalize_tracks(tracks)"),
    # P12 an unknown reanalysis is given some other one's thresholds
    "unknown_reanalysis_gets_a_default": _sub(
        "    if key not in V1_THRESHOLDS:",
        '    if key not in V1_THRESHOLDS:\n        return V1_THRESHOLDS[("ERA-Int", 700)]\n'
        "    if False:"),
    # the epoch conversion, which sits between the record's float days and the calendar
    "days_converted_against_the_wrong_epoch": _sub(
        'RECORD_EPOCH = np.datetime64("1900-01-01T00:00:00", "ns")',
        'RECORD_EPOCH = np.datetime64("1970-01-01T00:00:00", "ns")'),
    # P13 the curvature is computed in the file's own row order, which is exactly the
    # defect the orientation branch was added to fix
    "curvature_row_order_taken_on_trust": _sub(
        "    if latitude_descends(latgrid):\n"
        "        oriented = curvature_from_winds(",
        "    if False:\n"
        "        oriented = curvature_from_winds("),
    # P13 the orientation test points the wrong way, so both storage orders are wrong
    "curvature_oriented_for_the_wrong_order": _sub(
        "    if latitude_descends(latgrid):\n"
        "        oriented = curvature_from_winds(",
        "    if not latitude_descends(latgrid):\n"
        "        oriented = curvature_from_winds("),
    # P13 the result is never flipped back, so the output rows are reversed
    "curvature_orientation_not_restored": _sub(
        "        return oriented[:, ::-1, :]",
        "        return oriented"),
    # P13 the grid is flipped and the winds are not
    "curvature_grid_flipped_without_the_winds": _sub(
        "        oriented = curvature_from_winds(latgrid[::-1, :], longrid,\n"
        "                                        u[:, ::-1, :], v[:, ::-1, :])",
        "        oriented = curvature_from_winds(latgrid[::-1, :], longrid, u, v)"),
    # P13 the orientation only runs for a single-timestep stack, an outside-chosen
    # mutation that defeated the single-timestep version of the row-order fixture
    "curvature_oriented_only_for_one_timestep": _sub(
        "    if latitude_descends(latgrid):\n"
        "        oriented = curvature_from_winds(",
        "    if latitude_descends(latgrid) and u.shape[0] == 1:\n"
        "        oriented = curvature_from_winds("),
    # P13 the same chooser's second try, gated on a small stack instead of exactly one
    "curvature_oriented_only_for_short_stacks": _sub(
        "    if latitude_descends(latgrid):\n"
        "        oriented = curvature_from_winds(",
        "    if latitude_descends(latgrid) and u.shape[0] <= 2:\n"
        "        oriented = curvature_from_winds("),
    # the stack-shape refusal is dropped, so a 2-D caller fails differently by row order
    "curvature_accepts_a_2d_field": _sub(
        "    if u.ndim != 3 or v.ndim != 3:",
        "    if False:"),
}
