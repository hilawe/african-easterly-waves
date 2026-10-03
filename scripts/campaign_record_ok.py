#!/usr/bin/env python3
"""Whether a completion record at a canonical campaign path is a record OF THAT RUN.

The campaign driver skips a run whose record exists and counts it as present at the
end. A review built the pair that defeats a pathname test: an empty file, or another
run's record, copied to the expected path, reads as a completed run to a restart and
is never rerun. So the driver asks this script instead of the file system. A record
passes when it parses, names the dataset and year of the path it sits at, and, when a
tracks file sits beside it, names that file's digest. Exit 0 when it passes, 1 with a
reason otherwise. The binding check goes much further; this is the one predicate the
driver and the check share for "is this a record of this run".

THE DRIVER ALSO NAMES ITS MANIFEST. A review seeded records made under two different
manifests, and a campaign run under a third skipped both and reported itself complete
(2026-10-03). From the command line the expected manifest digest is therefore required,
and a record passes only when it was made under that manifest. The binding check and the
threshold driver test manifest identity themselves and call problems() without it.

    python3 scripts/campaign_record_ok.py --manifest-sha256 <hex digest> <record path>
"""
import hashlib
import json
import os
import re
import sys


def problems(record_path, manifest_sha256=None):
    name = os.path.basename(record_path)
    run = os.path.basename(os.path.dirname(record_path))
    if not (name.startswith("tracking_") and name.endswith(".json")):
        return [f"{name} is not a tracking record's name"]
    dataset, year = run.rsplit("_", 1) if "_" in run else (run, "")
    if not year.isdigit():
        return [f"{run} does not end in a year"]
    try:
        record = json.load(open(record_path))
        d = record["dataset_specific"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return [f"unreadable or not a record: {exc}"]
    out = []
    if d.get("dataset") != dataset or not isinstance(d.get("year"), int) or d.get("year") != int(year):
        out.append(f"record names {d.get('dataset')} {d.get('year')}, the path says {dataset} {year}")
    tracks = os.path.join(os.path.dirname(record_path), "tracker_port.mat")
    if os.path.exists(tracks):
        digest = hashlib.sha256(open(tracks, "rb").read()).hexdigest()
        if d.get("tracks_sha256") != digest:
            out.append("the tracks file beside the record does not have the digest the record names")
    if manifest_sha256 is not None:
        settings = record.get("protocol_settings") if isinstance(record, dict) else None
        named = settings.get("manifest_sha256") if isinstance(settings, dict) else None
        if named != manifest_sha256:
            out.append(f"the record was made under manifest {str(named)[:12]}, this campaign's is {manifest_sha256[:12]}")
    return out


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 3 or argv[0] != "--manifest-sha256" or not re.fullmatch(r"[0-9a-f]{64}", argv[1]):
        print("usage: campaign_record_ok.py --manifest-sha256 <hex digest> <record path>")
        return 2
    out = problems(argv[2], manifest_sha256=argv[1])
    for p in out:
        print(p)
    return 0 if not out else 1


if __name__ == "__main__":
    sys.exit(main())
