"""Named, frozen configurations, so a run cannot silently mix incompatible choices.

WHY THIS EXISTS. A science-direction review found two canonical entry points able to
recreate already-disproven configurations without saying so: a validation script defaulting
to the unbuffered 0.75 degree tree, and a download script defaulting to the domain and grid
that tree was first built at. Both defaults have since been corrected, but correcting
defaults one at a time is not the same as making the intended configuration NAMEABLE. A
profile is the thing a result can cite.

THE DISTINCTION THAT MATTERS MOST is between reproducing version 1 and repairing it. The
port carries both behaviours behind flags (`absorb`, `exclusive`), which is the right shape,
and the risk is not that a flag is wrong but that a RUN mixes them and a number is later
read as though it came from one. Profiles are frozen dataclasses for that reason: a caller
picks one, and every setting travels with the answer.

WHAT A PROFILE DOES NOT DO. It does not verify that the data on disk matches what it names.
`describe()` returns the settings for recording alongside a result, and the caller is still
responsible for pointing at the right tree. Making it check would need the data, which is
not something a configuration object should reach for.
"""
from dataclasses import asdict, dataclass, field
from typing import Optional, Tuple


@dataclass(frozen=True)
class Profile:
    """One named configuration of the whole chain."""

    name: str
    purpose: str
    directory: str
    prefix: str
    native_resolution: float
    coarse_resolution: float
    domain_lat: Tuple[float, float]
    domain_lon: Tuple[float, float]
    climatology_years: Tuple[int, int]
    coarse_threshold: Optional[float]
    fine_threshold: Optional[float]
    threshold_source: str
    absorb: bool
    exclusive: bool
    matcher: str
    notes: str = ""

    def describe(self):
        """The settings as a plain dict, for recording beside a result."""
        return asdict(self)


# THE ARCHIVED ERA-INTERIM PAIR, from find_ews_f.m's own hardcoded table. Not recomputed,
# because recomputing them is what the threshold work is for and a baseline that used a
# recomputed number would be measuring two things at once.
V1_ERAINT_700 = (7.16e-7, 2.80e-6)

FAITHFUL_ERAINT = Profile(
    name="faithful-eraint-700",
    purpose="reproduce version 1 as published, so a later correction can be attributed",
    directory="data/eraint/v1port_buffered",
    prefix="eraint",
    native_resolution=1.0,
    coarse_resolution=2.5,
    domain_lat=(-35.0, 35.0),
    domain_lon=(-140.0, 40.0),
    climatology_years=(1981, 2010),
    coarse_threshold=V1_ERAINT_700[0],
    fine_threshold=V1_ERAINT_700[1],
    threshold_source="find_ews_f.m's hardcoded table, not recomputed",
    absorb=False,
    exclusive=False,
    matcher="maximum-cardinality",
    notes=(
        "The one-degree buffered tree, NOT data/eraint/v1port, which is the unbuffered "
        "0.75 degree retrieval this project has measured to be the wrong configuration: "
        "decimate_f takes the FLOOR of the resolution ratio, so 0.75 degree input gives a "
        "2.25 degree tracking mesh where version 1's one degree gives 2.0. "
        "coarse_resolution is 2.5 because that is the PARAMETER version 1 passes; the mesh "
        "that results from it is 2.0. absorb=False and exclusive=False are version 1's own "
        "behaviour, defects included, which is the whole point of this profile."
    ),
)

REPAIRED_ERAINT = Profile(
    name="repaired-eraint-700",
    purpose="version 1's chain with the duplication repair on, for science rather than fidelity",
    directory="data/eraint/v1port_buffered",
    prefix="eraint",
    native_resolution=1.0,
    coarse_resolution=2.5,
    domain_lat=(-35.0, 35.0),
    domain_lon=(-140.0, 40.0),
    climatology_years=(1981, 2010),
    coarse_threshold=V1_ERAINT_700[0],
    fine_threshold=V1_ERAINT_700[1],
    threshold_source="find_ews_f.m's hardcoded table, not recomputed",
    absorb=False,
    exclusive=True,
    matcher="maximum-cardinality",
    notes=(
        "Differs from the faithful profile in ONE setting, deliberately. A track-level "
        "comparison of this profile against version 1 is not a fidelity measure and will "
        "score lower BY DESIGN, because the repair diverges from the behaviour being "
        "reproduced. Score it on distinct waves instead."
    ),
)

# ERA5 IS DELIBERATELY ABSENT AND THAT IS NOT AN OVERSIGHT. Its thresholds are the subject
# of an open gate: the machine-readable artifact held superseded whole-domain values, and
# the same recipe does not reproduce version 1's published ERA-Interim COARSE threshold on
# any sample tried, and the archive's OWN documented sample is the worst of them (37.5
# percent low on the tracking domain, against a 10 percent criterion, where a tuned African
# box was 20.6). That documented sample also breaks the FINE threshold, 19.7 percent low
# where the box gave 5.6, so both thresholds are now open. Naming an ERA5 profile now would
# freeze a number the project has not earned. Add one when the gate closes, and record in
# `threshold_source` which artifact it came from.
#
# THE COARSE GAP WIDENED WHEN THE DECIMATION WAS FIXED, from 15.9 percent to 20.6, on a
# rerun differing from its predecessor in the source fingerprint alone (same region, years,
# seed, stride, input file hashes, percentiles) and leaving the fine threshold bit-identical
# as it must, since the fine threshold never gates a decimated field. The decimation shift
# was therefore NOT what the coarse discrepancy was measuring, even though chasing that
# discrepancy is how the shift was found. The gap is NOT in an unrecorded sample either,
# because the archive's documented sample has since been run and is worse. It is in some
# transformation not yet identified.

PROFILES = {p.name: p for p in (FAITHFUL_ERAINT, REPAIRED_ERAINT)}


def get(name):
    """A profile by name, refusing an unknown one rather than falling back."""
    if name not in PROFILES:
        raise SystemExit(
            f"unknown profile {name!r}. Known: {', '.join(sorted(PROFILES))}. "
            f"There is deliberately no default: a result that does not say which "
            f"configuration produced it cannot be compared with another.")
    return PROFILES[name]
