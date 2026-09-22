"""The shared exact comparator, and the bypass it was found to have.

THE DEFECT THIS FILE EXISTS FOR. `require_identical` once delegated its VERDICT to
`describe_difference`, an explainer that walked latitudes over `range(len(x[1]))`. Two runs
whose TIME arrays agreed while one carried an EXTRA LATITUDE were reported IDENTICAL,
because the extra entry lay past the range walked, and the same pair in the other order
raised IndexError rather than returning a verdict. A control gate that passes when the runs
differ is worse than no gate, because it is read as evidence.

The repair is not a longer loop. It is that the verdict is the structural comparison and the
explainer only says what changed, so a case the explainer misses can no longer become a pass.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import exact_tracks as X  # noqa: E402


def track(times, lats, lons):
    return {"time": list(times), "meanlat": list(lats), "meanlon": list(lons)}


BASE = [track([1.0, 2.0], [3.0, 4.0], [5.0, 6.0]), track([7.0], [8.0], [9.0])]


def test_identical_runs_pass():
    assert X.require_identical(BASE, [dict(t) for t in BASE], "a", "b") is True
    assert X.describe_difference(BASE, [dict(t) for t in BASE], "a", "b") is None


@pytest.mark.parametrize("other, why", [
    # THE REGRESSION. An extra latitude with the time arrays untouched, both orders.
    ([track([1.0, 2.0], [3.0, 4.0, 99.0], [5.0, 6.0]), BASE[1]], "latitudes"),
    ([track([1.0, 2.0], [3.0, 4.0], [5.0, 6.0, 99.0]), BASE[1]], "longitudes"),
    # A track that is ragged on ONE side only, which must be named rather than indexed into.
    ([track([1.0, 2.0], [3.0], [5.0, 6.0]), BASE[1]], "ragged"),
    # The differences the first version did catch, kept so the repair did not lose them.
    ([BASE[0]], "finished tracks"),
    ([track([1.0, 2.0], [3.0, 4.0 + 1e-12], [5.0, 6.0]), BASE[1]], "observation 1"),
    ([track([1.0, 2.0 + 1e-9], [3.0, 4.0], [5.0, 6.0]), BASE[1]], "time"),
    ([track([1.0], [3.0], [5.0]), BASE[1]], "observations"),
])
def test_a_difference_is_refused_and_named(other, why):
    with pytest.raises(SystemExit) as raised:
        X.require_identical(BASE, other, "baseline", "control")
    message = str(raised.value)
    assert "REFUSED" in message
    assert why in message
    # and it must refuse in the other order too, where the old code raised IndexError
    with pytest.raises(SystemExit):
        X.require_identical(other, BASE, "control", "baseline")


def test_the_explainer_never_returns_none_when_the_structures_differ():
    """Silence must not be readable as agreement, whatever the explainer fails to localize."""
    for other in ([track([1.0, 2.0], [3.0, 4.0, 99.0], [5.0, 6.0]), BASE[1]],
                  [track([1.0, 2.0], [3.0, 4.0], [5.0, 6.0, 99.0]), BASE[1]]):
        assert X.canonical(BASE) != X.canonical(other)
        assert X.describe_difference(BASE, other, "a", "b") is not None
        assert X.describe_difference(other, BASE, "b", "a") is not None


def test_the_verdict_does_not_depend_on_the_explainer():
    """Even an explainer that says nothing must not turn a difference into a pass."""
    original = X.describe_difference
    X.describe_difference = lambda *a, **k: None
    try:
        with pytest.raises(SystemExit):
            X.require_identical(BASE, [BASE[0]], "baseline", "control")
    finally:
        X.describe_difference = original


def test_holds_exactly_is_exact():
    t, la, lo = [1.0, 2.0], [3.0, 4.0], [5.0, 6.0]
    assert X.holds_exactly(BASE, t, la, lo)
    assert not X.holds_exactly(BASE, t, [3.0, 4.0 + 1e-12], lo)
    assert not X.holds_exactly(BASE, [1.0, 2.0 + 1e-12], la, lo)
    assert not X.holds_exactly(BASE, [1.0], [3.0], [5.0])
