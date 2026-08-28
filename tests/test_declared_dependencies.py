"""Every import in the package resolves to a dependency the package declares.

WHY THIS EXISTS. The public face is exported from an allowlist and installed with
`pip install -e .`, so it gets the CORE dependencies and nothing else. In this working
tree matplotlib is installed, because the plotting extra is, and an import of it therefore
succeeds here and would fail there. That is not hypothetical: the association stage used
`matplotlib.path` for its point-in-polygon test, and a core installation would have seeded
tracks successfully and then failed part-way through its first matching pass. The leak scan
cannot see this, because nothing is leaking; the export gate catches it only once the
export runs.

WHAT IT CHECKS. Every `import` and `from ... import` in src/aew, INCLUDING the ones inside
function bodies, names either the standard library, another module of this package, a core
dependency, or a module that is allowed to need an extra. Lazy imports matter most, since
those are exactly the ones that survive a module-level smoke test and fail in use.

WHAT IT DOES NOT CHECK. That the declared versions are right, that an extra is installed
where it is needed, or that a dependency is actually reachable at run time. It reads the
source; it does not install anything.
"""

import ast
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
PACKAGE = REPO / "src" / "aew"
PYPROJECT = REPO / "pyproject.toml"

# Modules permitted to import an optional extra, with the extra they belong to. Anything
# not listed here must run on the core dependencies alone.
EXTRA_USERS = {
    "plotting.py": {"matplotlib", "cartopy"},
    "data/era5.py": {"cdsapi"},
}


def declared_core():
    """The core dependency names from pyproject, read rather than duplicated here."""
    text = PYPROJECT.read_text(encoding="utf-8")
    start = text.index("dependencies = [")
    body = text[start:text.index("]", start)]
    names = set()
    for line in body.splitlines()[1:]:
        line = line.strip()
        if not line.startswith('"'):
            continue
        spec = line.strip('",')
        for sep in (">=", "==", "<=", "~=", ">", "<", "["):
            spec = spec.split(sep)[0]
        names.add(spec.strip())
    return names


def imports_in(path):
    """Every top-level module name imported anywhere in the file, with its line."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:                       # relative, inside this package
                continue
            names = [(node.module or "").split(".")[0]]
        else:
            continue
        found.extend((name, node.lineno) for name in names if name)
    return found


def package_files():
    return sorted(PACKAGE.rglob("*.py"))


def test_the_core_dependency_list_was_actually_found():
    """Guards the parser itself. If pyproject is reformatted so the list stops being
    found, an empty set would make every import below look undeclared, or a set containing
    everything would make the whole test vacuous."""
    core = declared_core()
    assert "numpy" in core and "scipy" in core
    assert 3 <= len(core) <= 40


def test_there_are_files_to_check():
    """A glob that matches nothing passes every assertion in the loop below."""
    assert len(package_files()) > 5


@pytest.mark.parametrize("path", package_files(), ids=lambda p: str(p.name))
def test_every_import_is_declared_or_allowed(path):
    core = declared_core() | set(sys.stdlib_module_names) | {"aew"}
    relative = path.relative_to(PACKAGE).as_posix()
    allowed = core | EXTRA_USERS.get(relative, set())
    offenders = [f"line {line}: {name}" for name, line in imports_in(path)
                 if name not in allowed]
    assert offenders == [], (
        f"{relative} imports something the package does not declare as a core dependency. "
        f"Either add it to pyproject's `dependencies`, or list this file in EXTRA_USERS "
        f"if the import belongs to an optional extra: {offenders}")


def test_the_tracker_runs_on_the_core_dependencies_alone():
    """The port is the part that has to work in a bare install, so it is named explicitly
    rather than left to the sweep above. No module of it may appear in EXTRA_USERS."""
    port = sorted((PACKAGE / "v1port").glob("*.py"))
    assert len(port) >= 6, "the port's modules were not found"
    core = declared_core() | set(sys.stdlib_module_names) | {"aew"}
    offenders = []
    for path in port:
        relative = path.relative_to(PACKAGE).as_posix()
        assert relative not in EXTRA_USERS, f"{relative} must not depend on an extra"
        offenders += [f"{relative}:{line} {name}" for name, line in imports_in(path)
                      if name not in core]
    assert offenders == []
