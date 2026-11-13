"""
No concrete model or dataset name may drive CI — anywhere, not just in generic/.

tests/unit/generic/test_no_hardcoded_names.py guards one directory. This module
guards the whole machinery a contributor's new model or dataset has to pass
through: the GitLab pipeline, the Makefile, the ci/ helpers, the fixture script,
and every test outside the two sanctioned homes for names.

The rule it enforces: a name may appear in ci/policy.yml (the CI-owned record of
known-broken components), in tests/unit/specific/ (component tests, where naming
is the point), and in prose. Everywhere else, behaviour must come from
auto-discovery — so adding a model or a dataset changes no pipeline file, no
Makefile target and no test.

CODE VERSUS PROSE
-----------------
Only executable content is scanned. Python is parsed with `ast`, so comments are
invisible and docstrings are skipped: "FDataDataset -> sample_fdata.parquet"
documents the naming rule and keeps working when a dataset is added, whereas
`if name == "FDataDataset":` does not. YAML and the Makefile are scanned with
their comment lines stripped, for the same reason.

WHAT IT LOOKS FOR
-----------------
Everything comes from discovery, so this module needs no maintenance:
  * model class names            (MODEL_CLASSES)
  * dataset class names          (DATASET_CLASSES)
  * dataset ids                  (DATASET_ID on each discovered dataset)
  * sample fixture file names    (fixture_name_for on each discovered dataset)

Surfaces and exemptions live in ci/policy.yml -> name_scan, not here, so
widening or narrowing the scan is a reviewable policy change.
"""

import ast
import fnmatch
import re
from pathlib import Path

import pytest

import torch  # noqa: F401

from tests.conftest import (
    DATASET_CLASSES,
    MODEL_CLASSES,
    fixture_name_for,
    load_policy,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

_POLICY = load_policy().get("name_scan") or {}
_SURFACES = list(_POLICY.get("surfaces") or [])
_EXEMPT = dict(_POLICY.get("exempt") or {})


# ---------------------------------------------------------------------------
# The names to hunt for — all derived, none typed
# ---------------------------------------------------------------------------


def _forbidden_names() -> set:
    names = set(MODEL_CLASSES) | set(DATASET_CLASSES)
    for cls in DATASET_CLASSES.values():
        dataset_id = getattr(cls, "DATASET_ID", "")
        if dataset_id:
            names.add(dataset_id)
        names.add(fixture_name_for(cls))
    return {n for n in names if n}


_FORBIDDEN = sorted(_forbidden_names(), key=len, reverse=True)
_PATTERNS = [(n, re.compile(rf"(?<![\w-]){re.escape(n)}(?![\w-])")) for n in _FORBIDDEN]


# ---------------------------------------------------------------------------
# Which files are scanned
# ---------------------------------------------------------------------------


def _is_exempt(rel_path: str) -> bool:
    return any(fnmatch.fnmatch(rel_path, pattern) for pattern in _EXEMPT)


def _scanned_files() -> list:
    files = []
    for surface in _SURFACES:
        for path in sorted(_PROJECT_ROOT.glob(surface)):
            if not path.is_file():
                continue
            rel = path.relative_to(_PROJECT_ROOT).as_posix()
            if _is_exempt(rel) or "__pycache__" in rel:
                continue
            files.append(path)
    return files


_SCANNED = _scanned_files()


# ---------------------------------------------------------------------------
# Extracting the executable content of a file
# ---------------------------------------------------------------------------


def _docstring_nodes(tree: ast.AST) -> set:
    """ids of the Constant nodes that are module/class/function docstrings."""
    marked = set()
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        body = getattr(node, "body", None) or []
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            marked.add(id(body[0].value))
    return marked


def _python_code_strings(path: Path) -> list:
    """
    (line_no, text) for every piece of EXECUTABLE content in a Python file:
    identifiers, attributes, imports, definitions and non-docstring literals.

    Comments never appear (ast drops them) and docstrings are filtered out, so
    prose about a dataset is free while code keyed on its name is not.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:  # pragma: no cover - sanity job catches this first
        return []

    docstrings = _docstring_nodes(tree)
    found = []

    for node in ast.walk(tree):
        line = getattr(node, "lineno", None)
        if line is None:
            continue
        if isinstance(node, ast.Name):
            found.append((line, node.id))
        elif isinstance(node, ast.Attribute):
            found.append((line, node.attr))
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            found.append((line, node.name))
        elif isinstance(node, ast.alias):
            found.append((line, node.name))
            if node.asname:
                found.append((line, node.asname))
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.append((line, node.module))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in docstrings:
                found.append((line, node.value))
    return found


_COMMENT = re.compile(r"(?<!\$)#.*$")


def _plain_code_lines(path: Path) -> list:
    """(line_no, text) for YAML/Makefile lines with comments stripped."""
    lines = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = _COMMENT.sub("", raw).strip()
        if stripped:
            lines.append((line_no, stripped))
    return lines


def _offences(path: Path) -> list:
    """(line_no, name, excerpt) for every concrete name in executable content."""
    pieces = (
        _python_code_strings(path) if path.suffix == ".py" else _plain_code_lines(path)
    )
    hits = []
    for line_no, text in pieces:
        for name, pattern in _PATTERNS:
            if pattern.search(text):
                hits.append((line_no, name, text[:80]))
                break
    return hits


# ---------------------------------------------------------------------------
# The scan
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    _SCANNED,
    ids=[p.relative_to(_PROJECT_ROOT).as_posix() for p in _SCANNED],
)
def test_ci_surface_is_name_agnostic(path):
    """
    PASS: the file drives everything from discovery — a new model or dataset
          needs no edit here.
    FAIL: it keys behaviour on a concrete name, so the next model added to the
          repo is handled differently from the ones someone typed out.
    """
    rel = path.relative_to(_PROJECT_ROOT).as_posix()
    offences = _offences(path)

    assert not offences, (
        f"FAIL [{rel}] uses {len(offences)} concrete model/dataset name(s) in code:\n"
        + "\n".join(
            f"    {rel}:{ln}  '{name}'  ->  {text}" for ln, name, text in offences
        )
        + "\n\n  → CI and test behaviour must come from auto-discovery "
        "(MODEL_CLASSES / DATASET_CLASSES / get_train_config), so adding a model\n"
        "    or dataset needs no pipeline, Makefile or test edit.\n"
        "  → A test that genuinely targets one component belongs in "
        "tests/unit/specific/.\n"
        "  → A statement ABOUT one component (known-broken, out of scope) belongs "
        "in ci/policy.yml.\n"
        "  → Retired material belongs in archive/ (see archive/README.md).\n"
        "  → Prose is fine: comments and docstrings are not scanned."
    )


def test_the_scan_covers_the_real_ci_surface():
    """
    PASS: the scan is pointed at the pipeline, the Makefile, the ci/ helpers and
          the test tree.
    FAIL: a surface was dropped from ci/policy.yml -> name_scan.surfaces, which
          would let hardcoding return unnoticed in the file that was removed.
    """
    assert _SURFACES, "ci/policy.yml -> name_scan.surfaces is empty"
    assert _FORBIDDEN, (
        "discovery produced no names, so every scan below passes vacuously — "
        "check that the ML dependencies are installed"
    )

    scanned = {p.relative_to(_PROJECT_ROOT).as_posix() for p in _SCANNED}
    for required in (".gitlab-ci.yml", "Makefile"):
        assert required in scanned, (
            f"{required} is not scanned — the pipeline definition itself is where "
            f"per-model jobs would reappear first."
        )
    assert any(p.startswith("ci/") for p in scanned), "no ci/ helper is scanned"
    assert any(p.startswith("tests/") for p in scanned), "no test module is scanned"


@pytest.mark.parametrize("pattern", sorted(_EXEMPT), ids=sorted(_EXEMPT))
def test_every_exemption_is_justified_and_used(pattern):
    """
    PASS: the exemption carries a reason and still matches something in the repo.
    FAIL: an unexplained exemption, or one left behind after its files moved —
          either way the allowlist stops describing reality.
    """
    entry = _EXEMPT[pattern]
    reason = entry.get("reason") if isinstance(entry, dict) else entry
    assert reason and len(str(reason).split()) >= 5, (
        f"FAIL exemption {pattern!r} has no usable reason. Every path allowed to "
        f"name a component must say why it is allowed."
    )

    # Globbed from the pattern itself rather than by walking the repo: an
    # rglob("*") here also descends .venv/ and .git/, which costs seconds.
    # pathlib's "**" matches directories only, so a "dir/**" exemption needs the
    # "/*" suffix to reach the files inside it.
    candidates = list(_PROJECT_ROOT.glob(pattern))
    if pattern.endswith("**"):
        candidates += list(_PROJECT_ROOT.glob(pattern + "/*"))
    matches = [p for p in candidates if p.is_file()]
    assert matches, (
        f"FAIL exemption {pattern!r} matches no file. Delete it, or fix the "
        f"pattern — a stale allowlist entry hides the next offender that lands "
        f"on that path."
    )
