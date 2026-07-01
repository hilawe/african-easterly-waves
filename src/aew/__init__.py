"""aew: reproducible African Easterly Wave / convective-system analysis.

Python port of the PhD NCL code. Faithful reimplementations of the original
Carl Schreck NCL library functions (composite_dates, bin_sum, filwgts_lanczos +
wgt_runave) that the legacy scripts depended on.

See docs/REFACTOR_PLAN.md, docs/FIGURE_SCRIPT_MAP.md, docs/VALIDATION_TARGETS.md.
"""

from . import binning, composites, events, filtering, core, tracks

__all__ = ["binning", "composites", "events", "filtering", "core", "tracks"]
__version__ = "0.0.1"
