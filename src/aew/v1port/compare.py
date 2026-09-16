"""Compare a port-produced record against version 1's published one.

THIS IS THE INSTRUMENT THE WHOLE PORT IS BUILT TOWARD. Version 1's published record is
cannot be executed on a test case, and repeated independent checking found twenty-three places the
port's reading of its source was wrong. Every check in the repository up to now compares
the port against that same reading. This one compares it against what version 1 actually
produced: NCEI C00784, in `data/aewc` for 1983 to 2007.

WHAT A DIFFERENCE MEANS, which is the part worth getting right before any numbers arrive.
A perfect match is NOT the expected result and would itself be suspicious, because three
divergences are already recorded and two of them are deliberate. Each has a signature, and
telling them apart is what this module is for:

  REGION TRUNCATION, deliberate. Version 1's connected-region search is a recursive flood
  fill wrapped in a catch that silently returns a partial region at MATLAB's recursion
  limit. The port computes the whole region. Signature: differences confined to LARGE
  features, and in the direction of the port finding a bigger trough than version 1 did.

  CONTOUR PARSING, deliberate. Version 1 finds contour headers by matching the level in a
  row that holds longitude, and the level is zero, so on this domain the search can match
  vertices on the prime meridian as well as real headers. Signature: version 1 holding
  candidates the port does not, concentrated NEAR ZERO LONGITUDE.

  THE CONVEX HULL'S STARTING VERTEX, not deliberate and not established. Closing the hull
  doubles its first vertex in the inflation's mean, and neither MATLAB nor scipy documents
  where its cycle starts. If they disagree, every search polygon is offset by up to about a
  degree. Signature: tracks that follow each other for a while and then DRIFT APART
  GRADUALLY, rather than agreeing exactly and then breaking at one step. This is the one no
  review can settle and the reason the comparison is worth its retrieval.

ANYTHING ELSE IS A PORTING DEFECT, and the point of naming the three in advance is that a
difference which fits none of these patterns cannot be explained away after the fact.

WHAT COUNTS AS THE SAME TRACK. There is no identifier in common, so tracks are matched
geographically: two tracks correspond when they overlap in time and their positions at the
shared timesteps stay within a tolerance. That is a judgment, so the tolerance is a
parameter and the unmatched counts are reported on both sides rather than folded into a
single score.
"""

import numpy as np

from .geometry import great_circle_distance


def _positions(track):
    return (np.asarray(track["time"], dtype=float),
            np.asarray(track["meanlat"], dtype=float),
            np.asarray(track["meanlon"], dtype=float))


def separation_km(a, b):
    """Distance between two tracks at each timestep they share, and the shared times.

    Returns (times, distances). Empty when they never coexist.
    """
    ta, la_a, lo_a = _positions(a)
    tb, la_b, lo_b = _positions(b)
    # The record stores time to single precision, so exact equality is too strict; a
    # sixth of a day is well inside the six-hour spacing and well outside the rounding.
    tolerance = 1.0 / 6.0
    shared_t, distances = [], []
    for i, t in enumerate(ta):
        j = int(np.argmin(np.abs(tb - t)))
        if abs(tb[j] - t) > tolerance:
            continue
        shared_t.append(float(t))
        distances.append(float(great_circle_distance(la_a[i], lo_a[i],
                                                     la_b[j], lo_b[j], "km")))
    return np.asarray(shared_t), np.asarray(distances)


def match_tracks(port_tracks, published_tracks, tolerance_km=500.0,
                 min_shared_steps=3):
    """Pair port tracks with published ones on position, greedily and best-first.

    GREEDY AND BEST-FIRST RATHER THAN OPTIMAL, deliberately. A global assignment would pair
    more tracks, and it would do so by accepting worse pairs to improve a total. What this
    comparison needs is confidence that a pair really is the same wave, so each pair is
    taken in order of how well it agrees and nothing is paired twice.

    Returns a dict with `pairs` (port index, published index, median separation, shared
    steps), `port_unmatched` and `published_unmatched`.
    """
    candidates = []
    for i, a in enumerate(port_tracks):
        for j, b in enumerate(published_tracks):
            shared, distances = separation_km(a, b)
            if shared.size < min_shared_steps:
                continue
            median = float(np.median(distances))
            if median <= tolerance_km:
                candidates.append((median, shared.size, i, j))
    candidates.sort(key=lambda c: (c[0], -c[1]))

    used_port, used_published, pairs = set(), set(), []
    for median, shared, i, j in candidates:
        if i in used_port or j in used_published:
            continue
        used_port.add(i)
        used_published.add(j)
        pairs.append({"port": i, "published": j,
                      "median_separation_km": median, "shared_steps": int(shared)})
    return {"pairs": pairs,
            "port_unmatched": [i for i in range(len(port_tracks)) if i not in used_port],
            "published_unmatched": [j for j in range(len(published_tracks))
                                    if j not in used_published]}


def divergence_signature(port_tracks, published_tracks, matching):
    """Classify what the differences look like against the three recorded signatures.

    Reports the evidence, not a verdict. Each field answers one question, and a reader has
    to decide what the combination means:

      `gradual_drift_fraction`   share of matched pairs whose separation GROWS over their
                                 shared life rather than staying flat. The hull's starting
                                 vertex would produce a systematic polygon offset, which
                                 shows up as tracks that agree at first and slowly part.
      `meridian_enrichment`      how much MORE concentrated near zero longitude the
                                 unmatched published tracks are than the published tracks
                                 as a whole. The contour-parsing ambiguity would put
                                 version 1's phantom candidates there, so an enrichment
                                 above one is the signature and ONE IS NO SIGNAL AT ALL.
                                 A first version reported the bare share and read 24.7
                                 percent as evidence; a quarter of all the published
                                 tracks sit near the meridian anyway, so it was reporting
                                 the base rate.
      `port_tracks_longer`       share of matched pairs where the port's track has more
                                 observations. Region truncation would let version 1 lose a
                                 wave the port keeps following. ITS NULL IS ONE HALF, not
                                 zero: absent any effect either track is longer about as
                                 often, so only a value well away from 0.5 says anything.
    """
    growing, longer = 0, 0
    for pair in matching["pairs"]:
        a, b = port_tracks[pair["port"]], published_tracks[pair["published"]]
        shared, distances = separation_km(a, b)
        if shared.size >= 4:
            half = shared.size // 2
            if np.median(distances[half:]) > np.median(distances[:half]) * 1.5:
                growing += 1
        if len(a["time"]) > len(b["time"]):
            longer += 1
    pairs = max(len(matching["pairs"]), 1)

    def near_meridian(indices):
        count = 0
        for j in indices:
            lon = np.asarray(published_tracks[j]["meanlon"], dtype=float)
            if lon.size and abs(float(np.mean(lon))) <= 10.0:
                count += 1
        return count

    unmatched_indices = matching["published_unmatched"]
    unmatched = max(len(unmatched_indices), 1)
    all_indices = range(len(published_tracks))
    baseline_n = max(len(published_tracks), 1)
    unmatched_share = near_meridian(unmatched_indices) / unmatched
    baseline_share = near_meridian(all_indices) / baseline_n
    # AGAINST THE BASE RATE, not in absolute terms. On the first real comparison a quarter
    # of the unmatched tracks sat near the meridian and so did a quarter of ALL the
    # published tracks, so the bare share was reporting the geography of the record rather
    # than anything about the difference.
    enrichment = (unmatched_share / baseline_share) if baseline_share else float("nan")

    return {"gradual_drift_fraction": growing / pairs,
            "port_tracks_longer_fraction": longer / pairs,
            "unmatched_near_meridian_fraction": unmatched_share,
            "near_meridian_base_rate": baseline_share,
            "meridian_enrichment": enrichment,
            "matched_pairs": len(matching["pairs"]),
            "port_unmatched": len(matching["port_unmatched"]),
            "published_unmatched": len(matching["published_unmatched"])}


def compare(port_tracks, published_tracks, tolerance_km=500.0):
    """The whole comparison: counts, matching quality, and the divergence signatures."""
    matching = match_tracks(port_tracks, published_tracks, tolerance_km=tolerance_km)
    separations = [p["median_separation_km"] for p in matching["pairs"]]
    return {
        "port_tracks": len(port_tracks),
        "published_tracks": len(published_tracks),
        "matched": len(matching["pairs"]),
        "match_rate": len(matching["pairs"]) / max(len(published_tracks), 1),
        "median_separation_km": float(np.median(separations)) if separations else
                                float("nan"),
        "signatures": divergence_signature(port_tracks, published_tracks, matching),
    }
