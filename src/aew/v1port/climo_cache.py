"""A climatology cache that refuses to serve values another code version produced.

WHY THIS IS A MODULE AND NOT A HELPER IN ONE SCRIPT. It began inside
`scripts/diagnose_worked_timestep.py`, and a second consumer,
`scripts/export_tracker_case.py`, then read the same cache files with no check at all.
The oracle comparison that consumer feeds was therefore resting on a climatology nothing
had verified, which is the defect the fingerprint exists to prevent, reintroduced by
copying the cache path instead of the guard. Both now go through here.
"""
import hashlib
import os

import numpy as np

from aew.v1port import load as L


def climatology_fingerprint(years, directory, prefix):
    """A digest of everything that determines the cached values.

    The years, the input files' size and modification time, and the SOURCE of every
    module in the chain that computes them, because the first version of this cache
    checked years alone and would have silently served a climatology computed by the
    defective curvature code the diagnostics exist to investigate. File metadata, not
    contents, since hashing gigabytes per run would defeat the cache, so a replaced input
    with an identical size and modification time slips this. `short_steps` is not stored
    either, so the loaded dict carries it empty rather than claiming to be byte-identical
    to a fresh build.
    """
    import aew.v1port.climatology
    import aew.v1port.climo_cache
    import aew.v1port.load
    import aew.v1port.pipeline
    import aew.v1port.vorticity
    h = hashlib.sha256()
    h.update(repr((sorted(years), os.path.abspath(directory), prefix)).encode())
    for year in sorted(years):
        for var in ("u700", "v700"):
            path = os.path.join(directory, f"{prefix}_{var}_{year}_6h_region.nc")
            try:
                st = os.stat(path)
                h.update(repr((os.path.basename(path), st.st_size,
                               st.st_mtime_ns)).encode())
            except OSError:
                h.update(repr((os.path.basename(path), "absent")).encode())
    # THIS MODULE IS IN THE FINGERPRINT TOO. It writes the file and defines the schema, so
    # a change to how the cache is built or read must invalidate what is already on disk.
    # Leaving it out was a hole a review found: the guard did not guard itself.
    for module in (aew.v1port.vorticity, aew.v1port.climatology, aew.v1port.load,
                   aew.v1port.pipeline, aew.v1port.climo_cache):
        with open(module.__file__, "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()


def refuse_a_gappy_year_list(years):
    """Refuse a year list with a hole in it.

    `load.available_years` reports the years whose files are BOTH present, so a single
    missing wind file drops that year silently. With a cache already on disk the
    fingerprint catches it, because the year list is part of the digest. With no cache it
    does not: the build simply runs over 29 years and the result is recorded as a 30-year
    climatology. The provenance of every number downstream then rests on directory
    contents that nothing checked.

    A gap is refused rather than the span being hardcoded, so extending the record to
    more years needs no edit here, while losing one in the middle stops the run.
    """
    years = sorted(int(y) for y in years)
    if not years:
        raise SystemExit("no years found. Check the data directory and prefix.")
    missing = sorted(set(range(years[0], years[-1] + 1)) - set(years))
    if missing:
        raise SystemExit(
            f"years {years[0]} to {years[-1]} are present except for "
            f"{', '.join(str(y) for y in missing)}. A climatology built over a gapped "
            f"record would be recorded as though it covered the whole span, so this is "
            f"refused. Restore the missing files, or pass the years you mean explicitly.")
    return years


def load_or_build(years, directory, prefix, cache, verbose=True):
    """The climatology dict, from a cache file when one matches the fingerprint.

    REFUSES rather than rebuilds when a cache exists and does not match, including when
    it carries no fingerprint at all. Rebuilding silently would hide the fact that a
    recorded result was produced against different values, and an absent fingerprint is
    the least trustworthy case of all, not the most.
    """
    years = refuse_a_gappy_year_list(years)
    fingerprint = climatology_fingerprint(years, directory, prefix)
    # `np.savez_compressed` APPENDS `.npz` WHEN THE NAME LACKS IT, so a caller passing
    # `climo` wrote `climo.npz` and then looked for `climo`, found nothing, and rebuilt
    # and overwrote on every run without ever reading the fingerprint it had just
    # written. The guard was a no-op for exactly the callers most likely to get the name
    # slightly wrong. Resolve the name the writer will actually use, once, and use it for
    # both the read and the write.
    if cache:
        # os.fspath so a pathlib.Path works, which numpy and os.path both accept and
        # which `.endswith` did not.
        cache = os.fspath(cache)
        if os.path.isdir(cache):
            raise SystemExit(
                f"{cache} is a directory. Naming one here would silently write a sibling "
                f"{os.path.basename(cache)}.npz beside it rather than fail, so it is "
                f"refused instead. Give the cache file's own path.")
        if not cache.endswith(".npz"):
            cache = cache + ".npz"
    if cache and os.path.exists(cache):
        with np.load(cache, allow_pickle=False) as z:
            if "fingerprint" not in z:
                raise SystemExit(
                    f"{cache} carries no fingerprint, so nothing establishes which code "
                    f"or inputs produced it. Delete it and let it rebuild.")
            if str(z["fingerprint"]) != fingerprint:
                raise SystemExit(
                    f"{cache} was built over different years, inputs, or code than this "
                    f"run would use (the curvature chain's source is part of the "
                    f"fingerprint). Delete it and let it rebuild.")
            keys = [tuple(int(x) for x in row) for row in z["keys"]]
            climatology = {"keys": keys, "mean": z["mean"], "counts": z["counts"],
                           "short_steps": []}
        if verbose:
            print(f"climatology read from {cache}", flush=True)
        return climatology
    if verbose:
        print(f"building climatology over {len(years)} years "
              f"({years[0]}-{years[-1]})", flush=True)
    climatology = L.build_climatology(years, directory, prefix)
    if cache:
        np.savez_compressed(
            cache, years=np.asarray(years), fingerprint=np.asarray(fingerprint),
            keys=np.asarray(climatology["keys"], dtype=int),
            mean=climatology["mean"], counts=climatology["counts"])
        if verbose:
            print(f"climatology cached at {cache}", flush=True)
    return climatology
