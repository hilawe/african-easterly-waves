"""The cloud inputs of the pipeline, named in one place and checked against their recorded hashes.

The pipeline counts its classes from the detections of one satellite position, reads the
calendar of slots that position observed, and keeps the catalog that pools every position for a
sensitivity run. scripts/cloud_inputs.json names each file with its SHA-256. A script takes its
default from here and calls ``checked`` on the path it was given, so a run cannot start on a
file other than the one recorded without saying so.
"""
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "cloud_inputs.json")


def _manifest():
    with open(MANIFEST) as f:
        return json.load(f)


def path_of(name):
    """The recorded path of an input, relative to the repository root."""
    return _manifest()[name]["path"]


CLASS_CATALOG = path_of("class_catalog")
POOLED_CATALOG = path_of("pooled_catalog")
OBSERVED_SLOTS = path_of("observed_slots")
TRACKING_FILE = path_of("tracking_file")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def checked(path):
    """``path``, after its hash is compared with the record when it is a recorded input.

    A recorded input whose bytes differ from the record stops the run. A path the record does
    not name is returned with a printed line that says so, with its hash, so a run on another
    file is visible in its log.
    """
    if not os.path.exists(path):
        raise SystemExit(f"REFUSED: the input {path} does not exist")
    got = sha256(path)
    for name, entry in _manifest().items():
        if os.path.abspath(entry["path"]) == os.path.abspath(path) or os.path.realpath(entry["path"]) == os.path.realpath(path):
            if entry["sha256"] != got:
                raise SystemExit(f"REFUSED: {path} is recorded as {name} with another SHA-256 than the file has")
            print(f"input {name}: {path} (hash as recorded)", flush=True)
            return path
    print(f"input NOT among the recorded ones: {path} sha256 {got}", flush=True)
    return path
