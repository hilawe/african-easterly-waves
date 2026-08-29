"""aew: reproducible African Easterly Wave / convective-system analysis.

Python port of the PhD NCL code. Faithful reimplementations of the original
Carl Schreck NCL library functions (composite_dates, bin_sum, filwgts_lanczos +
wgt_runave) that the legacy scripts depended on.

The validation targets, the figure-to-script map and the port plan live in the
project documentation.
"""

from . import binning, composites, events, filtering, core, tracks, waves

__all__ = ["binning", "composites", "events", "filtering", "core", "tracks"]
__version__ = "0.0.1"
