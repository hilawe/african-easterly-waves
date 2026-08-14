"""Prose-to-canonical number audit and untagged-numeral lint (REPAIR_SPEC.md R6).

A numeric claim in the manuscript sources is tagged with an invisible HTML comment
immediately after the printed value:

    +2.99<!--n:pooled:700:lagrangian_rh:-72:diff:2-->

The tag names the canonical row (tier, level, statistic, time_rel_h), the field
(diff, ci_lo, ci_hi, or any canonical column), and the printed precision (an integer
decimal count, prefixed ``u`` for an unsigned rendering; signed with a leading + is
the default). The audit recomputes the expected string from
deposit/canonical_numbers.csv and errors when the printed value disagrees, when a key
is missing, or when two tags for one key/field print conflicting values, so text,
captions, and canonical values cannot drift apart again. The builders strip the tags
(``strip_tags``) so nothing reaches the PDF.

The lint pass flags numerals that carry no tag and match no allowlist pattern (years,
dates and hours, pressure levels and temperature thresholds, degrees and grid steps,
section/figure/table references, citation years, small design constants defined in the
methods). It runs in warn mode until the prose reconciliation pass tags the manuscript,
then becomes blocking (--strict-lint).
"""

import re

import pandas as pd

TAG_RE = re.compile(r"<!--n:([^:>]+):([^:>]*):([^:>]+):([^:>]*):([^:>]+):(u?\d)-->")
NUM_BEFORE_RE = re.compile(r"([+-]?\d+(?:[.,]\d+)*)\s*$")


def strip_tags(text):
    """Remove audit tags (builders call this so tags never render)."""
    return TAG_RE.sub("", text)


def _fmt(value, spec):
    unsigned = spec.startswith("u")
    nd = int(spec[-1])
    s = f"{value:+.{nd}f}" if not unsigned else f"{value:.{nd}f}"
    return s


def load_canonical(path):
    """Keyed rows plus duplicate-key errors (a silent overwrite would let two
    conflicting canonical rows coexist unnoticed; implementation fold)."""
    df = pd.read_csv(path)
    rows, dups = {}, []
    for _, r in df.iterrows():
        level = "" if pd.isna(r.get("level")) else f"{float(r['level']):g}"
        trel = "" if pd.isna(r.get("time_rel_h")) else f"{float(r['time_rel_h']):g}"
        key = (r["tier"], level, r["statistic"], trel)
        if key in rows:
            dups.append("duplicate canonical key " + ":".join(str(k) for k in key))
        rows[key] = r
    return rows, dups


def audit(text, canonical):
    """Check every tagged claim; returns a list of error strings."""
    errors = []
    seen = {}
    for m in TAG_RE.finditer(text):
        tier, level, stat, trel, field, spec = m.groups()
        key = (tier, level, stat, trel)
        before = text[max(0, m.start() - 40):m.start()]
        nm = NUM_BEFORE_RE.search(before)
        if not nm:
            errors.append(f"tag {m.group(0)} has no number immediately before it")
            continue
        printed = nm.group(1).replace(",", "")
        row = canonical.get(key)
        if row is None:
            errors.append(f"tag {':'.join(key)} matches no canonical row")
            continue
        if field not in row or pd.isna(row[field]):
            errors.append(f"tag {':'.join(key)} field {field!r} absent in canonical")
            continue
        expected = _fmt(float(row[field]), spec)
        if printed != expected:
            errors.append(f"STALE {':'.join(key)}:{field}: printed {printed!r}, "
                          f"canonical {expected!r}")
        k2 = (key, field)
        if k2 in seen and seen[k2] != printed:
            errors.append(f"CONFLICT {':'.join(key)}:{field}: printed both "
                          f"{seen[k2]!r} and {printed!r}")
        seen[k2] = printed
    return errors


# numerals that need no tag: structure references, dates and times, instrument and
# design constants defined in the methods, coordinates, and citation years
ALLOW_RES = [re.compile(p) for p in (
    r"\b(19|20)\d{2}\b",                                  # years, citation years
    r"\bFig(?:ure)?s?\.?\s*S?\d+[a-c]?\b",                # figure references
    r"\bsections?\s*\d[a-z]?(?:\s*(?:and|to|,)\s*\d[a-z]?)?\b",  # section references
    r"\bTables?\s*S?\d+\b",
    r"(?m)^#{1,4}\s*\d+\.",                               # numbered section headings
    r"\b\d+(?:\.\d+)?\s*(?:hPa|K\b|km\b|UTC|GHz)",        # levels, thresholds, units
    r"\b\d+(?:\.\d+)?\s+(?:and|to)\s+\d+(?:\.\d+)?\s*hPa\b",   # paired/ranged levels
    r"\b(?:June|July|August|September|October)\s+\d{1,2}\b",   # calendar dates
    r"\b\d+(?:\.\d+)?[-\s]?(?:h\b|hourly|day\b|days?\b|months?\b|seasons?\b)",
    r"(?<![\d.])[+-](?:24|36|48|60|72)\s*h\b",            # signed passage-relative lags
    r"\b\d+(?:\.\d+)?[-\s]?(?:degrees?|deg\b)",           # widths and grid steps
    r"\b\d+(?:\.\d+)?\s*(?:N|S|E|W)\b",                   # coordinates
    r"\b\d+(?:\.\d+)?\s+to\s+\d+(?:\.\d+)?\s*(?:N|S|E|W)\b",       # coordinate ranges
    r"[+-]?\d+(?:\.\d+)?\s+to\s+[+-]?\d+(?:\.\d+)?[-\s](?:h\b|days?\b|degrees?\b)",
    r"\b95\s*percent\b",                                  # the confidence level
    # HYPHENATED adjectival use only ("a 20,000-replicate bootstrap"). The spaced form
    # ("20,000 replicates") is a bare count and must be tagged, because that is exactly
    # the claim that silently drifted from the code in round 6.
    r"\b\d{1,3}(?:,\d{3})*-(?:observation|wave|parcel|season|replicate|draw)s?\b",
    r"\b\d{1,3}(?:,\d{3})*\s(?:observation|wave|parcel|season)s?\b",
    r"\b\d+th\b",                                          # percentile ordinals
    r"\bpart\s+[IVX\d]+\b",
    # Patterns below cover forms that occur mainly in FIGURE CAPTIONS. The lint used to
    # stop at the first table heading and so never reached them (round 6); they
    # are structural or design constants, not estimands.
    r"\b\d+(?:\.\d+)?[-\s]\d+(?:\.\d+)?\s*(?:N|S|E|W)\b",   # "5-15 N" band form
    r"\b\d{2}(?:,\s*\d{2})*(?:,?\s*and\s*\d{2})?\s*UTC\b",   # "00, 06, 12, and 18 UTC"
    r"(?<![\d.])-(?:24|36|48|60|72)(?:,\s*(?:and\s*)?-?(?:24|36|48|60|72))*\s*h\b",
    # replicate and draw counts are NOT allowlisted: they are exactly the class of
    # claim that drifted from the code in round 6 (manuscript said 20,000, the driver
    # ran 2,000) and an allowlist would let it recur unnoticed. Tag them.
)]

# no letter, digit, or hyphen immediately before (skips B1, CS-245, C00784, S1 and
# digit-internal restarts); comma only as a thousands separator so a trailing
# "2007," does not swallow its comma
NUMERAL_RE = re.compile(r"(?<![A-Za-z0-9-])[+-]?\d+(?:,\d{3})*(?:\.\d+)?")


def lint_untagged(text, skip_headings=("## References",)):
    """Numerals with no tag and no allowlist cover; returns (line_no, token) pairs.

    Only the reference list is excluded. "## Table" used to truncate here too, which
    silently ended the lint at the first table and left EVERY figure caption unchecked,
    since the captions sit after it. That is how "about two" and an untagged 1,000-draw
    caption reached the manuscript (round 6). Table design constants are covered
    by the allowlist instead of by skipping the rest of the document.
    """
    for h in skip_headings:
        i = text.find(h)
        if i >= 0:
            text = text[:i]
    tag_spans = [(m.start(), m.end()) for m in TAG_RE.finditer(text)]
    allow_spans = []
    for rx in ALLOW_RES:
        allow_spans += [(m.start(), m.end()) for m in rx.finditer(text)]
    out = []
    for m in NUMERAL_RE.finditer(text):
        s, e = m.span()
        if any(a <= s and e <= b for a, b in tag_spans):   # inside a tag comment
            continue
        if any(a <= s and e <= b for a, b in allow_spans):
            continue
        if TAG_RE.match(text, e):                          # the number owns a tag
            continue
        line = text.count("\n", 0, s) + 1
        out.append((line, m.group(0)))
    return out
