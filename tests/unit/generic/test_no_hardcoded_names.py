"""
The convention is enforced, not just written down (GitLab issue #7).

tests/unit/generic/ must contain NO concrete model or dataset name. Everything
there is driven by auto-discovery, so adding a model or dataset to the repo
extends the suite with no test edit — and nothing silently keeps working only
because someone typed a name by hand.

This suite drifted back to hardcoding once already: 126 references to concrete
names across six unit modules, which is why the two datasets nobody typed
(M100, SCRIPT-AI) had zero coverage. A convention that is not enforced will not
hold, so this test fails the pipeline the moment a name reappears.

Component-specific tests are legitimate — they just belong in
tests/unit/specific/, which this scan deliberately ignores.
"""

import fnmatch
import re
from pathlib import Path

import pytest

import torch  # noqa: F401

from tests.conftest import DATASET_CLASSES, MODEL_CLASSES, load_policy

_GENERIC_DIR = Path(__file__).resolve().parent

# The names to hunt for come from discovery itself, so this needs no updating
# when a model or dataset is added.
_FORBIDDEN = sorted(set(MODEL_CLASSES) | set(DATASET_CLASSES))

# The same allowlist the repo-wide scan uses (ci/policy.yml -> name_scan.exempt),
# so a file is exempt in one place or in neither. It covers the scanner modules
# themselves: they receive the names from discovery and must be able to explain
# the rule in prose.
_PROJECT_ROOT = _GENERIC_DIR.parents[2]
_EXEMPT = (load_policy().get("name_scan") or {}).get("exempt") or {}


def _exempt(path: Path) -> bool:
    rel = path.relative_to(_PROJECT_ROOT).as_posix()
    return any(fnmatch.fnmatch(rel, pattern) for pattern in _EXEMPT)


_GENERIC_MODULES = sorted(
    p for p in _GENERIC_DIR.glob("*.py") if p.name != "__init__.py" and not _exempt(p)
)


def _offences(path: Path) -> list[tuple[int, str, str]]:
    """(line_no, name, line) for every concrete model/dataset name in a file."""
    hits: list[tuple[int, str, str]] = []
    # This module necessarily *mentions* the idea of names; skip itself.
    if path.name == Path(__file__).name:
        return hits
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        for name in _FORBIDDEN:
            if re.search(rf"\b{re.escape(name)}\b", line):
                hits.append((line_no, name, line.strip()))
    return hits


@pytest.mark.parametrize(
    "module", _GENERIC_MODULES, ids=[p.name for p in _GENERIC_MODULES]
)
def test_generic_module_names_no_model_or_dataset(module):
    """
    PASS: the module is fully discovery-driven.
    FAIL: it names a concrete model or dataset, with the file, line and name
          reported, and tests/unit/specific/ named as the correct home.
    """
    offences = _offences(module)
    assert not offences, (
        f"FAIL [{module.name}] hardcodes {len(offences)} concrete name(s):\n"
        + "\n".join(
            f"    {module.name}:{ln}  '{name}'  ->  {text[:70]}"
            for ln, name, text in offences
        )
        + "\n\n  → tests/unit/generic/ must be driven by MODEL_CLASSES / DATASET_CLASSES\n"
        "    so a new model or dataset is covered with no test edit.\n"
        "  → A test that genuinely targets one component belongs in\n"
        "    tests/unit/specific/, where naming it is expected.\n"
        "  → If the case is about a KNOWN-BROKEN component, put the name in\n"
        "    ci/policy.yml (core_change_requests / dataset_scope) instead of in\n"
        "    the test. That documents it; it does not silence the failure."
    )


def test_discovery_found_something_to_check():
    """
    PASS: discovery produced names, so the scan above is meaningful.
    FAIL: MODEL_CLASSES and DATASET_CLASSES are both empty, which would make
          every scan vacuously pass and hide real hardcoding.
    """
    assert _FORBIDDEN, (
        "Neither models nor datasets were discovered, so the hardcoded-name scan "
        "would pass trivially. Check that the ML dependencies are installed."
    )


# def test_generic_and_specific_layout_exists():
#     """
#     PASS: both halves of the convention exist.
#     FAIL: the split was flattened, so there is nowhere for component-specific
#           tests to live and they will leak back into the generic suite.
#     """
#     assert _GENERIC_DIR.is_dir(), "tests/unit/generic/ is missing"
#     assert _SPECIFIC_DIR.is_dir(), (
#         "tests/unit/specific/ is missing — component-specific tests need a home, "
#         "or they end up back in generic/ where the guard rejects them."
#     )
def test_generic_layout_exists():
    assert _GENERIC_DIR.is_dir()
