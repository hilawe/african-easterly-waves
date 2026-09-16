"""Mutations for the version 1 trough detection port.

The list was written before the assertions existed and is reproduced in the docstring of
tests/test_v1port_detection.py. Run with the repository's mutation checker against
src/aew/v1port/detection.py.
"""


def _sub(old, new):
    return lambda s: s.replace(old, new)


MUTATIONS = {
    # K26 the absorption flag reaches the coarse merge and not the fine one, so the two
    # halves of the same merge disagree about which behaviour they are reproducing
    "absorb_flag_not_passed_to_the_fine_merge": _sub(
        "    return merge_contours(coarse, latgrid_fine, longrid_fine,\n"
        "                          curvature_f, fine_threshold, absorb=absorb)",
        "    return merge_contours(coarse, latgrid_fine, longrid_fine,\n"
        "                          curvature_f, fine_threshold)"),
    # D1 the trough level moved off zero
    "trough_level_not_zero": _sub(
        "TROUGH_LEVEL = 0.0",
        "TROUGH_LEVEL = 1e-11"),
    # D2 the wind mask tests magnitude, discarding strong easterlies
    "wind_mask_tests_magnitude": _sub(
        "    westerly = wind > MAX_ZONAL_WIND",
        "    westerly = np.abs(wind) > MAX_ZONAL_WIND"),
    # D3 the wind mask never applied
    "wind_mask_not_applied": _sub(
        "    westerly = wind > MAX_ZONAL_WIND",
        "    westerly = np.zeros_like(wind, dtype=bool)"),
    # D4 the wind mask also applied to the fine field, which the original comments out
    "wind_mask_applied_to_the_fine_field": _sub(
        "    curvature_f = np.where(curvature_f < fine_threshold, np.nan, curvature_f)",
        "    curvature_f = np.where(curvature_f < fine_threshold, np.nan, curvature_f)\n"
        "    curvature_f = np.where(westerly, np.nan, curvature_f)"),
    # D5 the curvature mask never applied
    "curvature_mask_not_applied": _sub(
        "    weak_c = curvature_c < coarse_threshold",
        "    weak_c = np.zeros_like(curvature_c, dtype=bool)"),
    # D6 the fine field gated by the coarse threshold
    "fine_field_uses_the_coarse_threshold": _sub(
        "    curvature_f = np.where(curvature_f < fine_threshold, np.nan, curvature_f)",
        "    curvature_f = np.where(curvature_f < coarse_threshold, np.nan, curvature_f)"),
    # D7 the hemispheric sign flip skipped
    "no_southern_sign_flip": _sub(
        "    return southern_hemisphere_sign(smooth9(field)[np.newaxis, ...], lat_values)[0]",
        "    return smooth9(field)"),
    # D8 the fields not smoothed before contouring
    "no_smoothing": _sub(
        "    advection_c = smooth9(np.asarray(advection_anomaly_coarse, dtype=float))",
        "    advection_c = np.asarray(advection_anomaly_coarse, dtype=float)"),
    # D9 only the coarse merge runs
    "only_one_merge_pass": _sub(
        "    return merge_contours(coarse, latgrid_fine, longrid_fine,\n"
        "                          curvature_f, fine_threshold, absorb=absorb)",
        "    return coarse"),
    # D10 the merge passes run in the wrong order
    "merge_passes_reversed": _sub(
        "    coarse = merge_contours(candidates, latgrid_coarse, longrid_coarse,\n"
        "                            curvature_c, coarse_threshold, absorb=absorb)",
        "    coarse = merge_contours(candidates, latgrid_fine, longrid_fine,\n"
        "                            curvature_f, fine_threshold, absorb=absorb)"),
    # D11 the candidate centre taken from an endpoint rather than the mean
    "candidate_centre_is_an_endpoint": _sub(
        '                   "lat_mean": float(np.mean(lats)),\n'
        '                   "lon_mean": float(np.mean(lons))}',
        '                   "lat_mean": float(lats[0]),\n'
        '                   "lon_mean": float(lons[0])}'),
    # D12 short contours admitted, the shape the original's ambiguous header search lets in
    "admits_degenerate_contours": _sub(
        "        if seg.ndim != 2 or seg.shape[0] < 2:\n            continue",
        "        if seg.ndim != 2:\n            continue"),
}
