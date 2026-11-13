"""
Core change requests: things that must change in seanergys_modelzoo/, which CI
must NOT change itself.

The modelzoo core belongs to the model and dataset authors. This test layer
never edits it. When a test cannot be written correctly because of something in
core, the request is recorded in ci/policy.yml -> core_change_requests, published
in docs/ci-cd.md, and handed to the owner. CI reports the truth about the code;
it does not reshape the code to make itself pass.

The register is kept honest from both ends:

  * a malformed entry fails, so a request cannot be vague about what is wanted,
    why CI cannot do it, or who owns it;
  * a request whose fix has LANDED fails, so the entry must be deleted. An entry
    is therefore either open and accurate, or gone.

Since the xfail lists were removed (2026-09-13) this register is the ONLY place a
known-broken component is recorded. It documents the defect and its owner — it
does not suppress the failing test, which stays red until the fix lands.

Nothing here imports a concrete model or dataset by name — the targets come from
the policy file.
"""

from pathlib import Path

import pytest

import torch  # noqa: F401

from tests.conftest import DATASET_CLASSES, MODEL_DATASET_CONFIGS, load_policy

# parents[3]: tests/unit/generic/<file> -> repo root. This was
# parent.parent.parent (= tests/), so every source_contains check found no file
# and reported SKIP — the register's expiry conditions never actually ran.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_REQUESTS = load_policy().get("core_change_requests") or {}
_IDS = sorted(_REQUESTS)

_REQUIRED_FIELDS = (
    "file",
    "symbol",
    "owner",
    "request",
    "why",
    "why_ci_cannot",
    "evidence",
)

# Each check answers one question: "is this request still open?"
# Returning None means the target is not present on this branch, so the request
# cannot be evaluated here — reported as a skip, never as a silent pass.
_CHECKS = (
    "dataset_missing_dataset_id",
    "dataset_not_consumed",
    "source_contains",
    "source_missing",
)


def _is_open(entry: dict):
    kind = entry.get("check")
    target = entry.get("target")
    if not isinstance(target, str):
        return None

    if kind == "dataset_missing_dataset_id":
        dataset_cls = DATASET_CLASSES.get(target)
        if dataset_cls is None:
            return None
        return not getattr(dataset_cls, "DATASET_ID", "")

    if kind == "dataset_not_consumed":
        if target not in DATASET_CLASSES:
            return None
        consumed = {cls.__name__ for _m, cls, _p, _d in MODEL_DATASET_CONFIGS.values()}
        return target not in consumed

    if kind == "source_contains":
        path = _REPO_ROOT / str(target)
        if not path.exists():
            return None
        return str(entry.get("needle")) in path.read_text(encoding="utf-8")

    if kind == "source_missing":
        # The mirror of source_contains, for a request whose fix ADDS something.
        # Open while <target> still lacks <needle>; closed once it appears.
        path = _REPO_ROOT / str(target)
        if not path.exists():
            return None
        return str(entry.get("needle")) not in path.read_text(encoding="utf-8")

    raise AssertionError(f"unknown check kind {kind!r}; expected one of {_CHECKS}")


@pytest.mark.parametrize("request_id", _IDS, ids=_IDS)
def test_request_is_well_formed(request_id):
    """
    PASS: the entry says what must change, why, why CI cannot do it, who owns
          it, and what evidence backs it.
    FAIL: a field is missing — an under-specified request cannot be acted on by
          the owner and will be ignored.
    """
    entry = _REQUESTS[request_id]

    missing = [f for f in _REQUIRED_FIELDS if not str(entry.get(f, "")).strip()]
    assert not missing, (
        f"FAIL [{request_id}] core_change_requests entry is missing: {', '.join(missing)}.\n"
        f"  → Every request must be actionable on its own; see ci/policy.yml."
    )
    assert entry.get("check") in _CHECKS, (
        f"FAIL [{request_id}] check must be one of {_CHECKS}, got {entry.get('check')!r}.\n"
        f"  → Without a machine-checkable condition the entry cannot expire and will rot."
    )
    assert str(entry.get("file", "")).startswith("seanergys_modelzoo/"), (
        f"FAIL [{request_id}] file {entry.get('file')!r} is not under seanergys_modelzoo/.\n"
        f"  → This register is only for changes CI must not make itself. Anything in\n"
        f"    tests/, ci/ or scripts/ is ours to fix directly."
    )


@pytest.mark.parametrize("request_id", _IDS, ids=_IDS)
def test_request_is_still_open(request_id):
    """
    PASS: the requested core change has NOT landed yet, so the entry is accurate.
    SKIP: the target does not exist on this branch (e.g. it lives on another
          branch not merged here) — reported, never silently passed.
    FAIL: the owner has landed the fix. Delete the entry so the register stops
          claiming something untrue about the code.
    """
    entry = _REQUESTS[request_id]
    still_open = _is_open(entry)

    if still_open is None:
        pytest.skip(
            f"SKIP [{request_id}] target {entry.get('target')!r} is not present on this "
            f"branch, so the request cannot be evaluated here.\n"
            f"  → {entry.get('note', 'Re-check after merging the branch that carries it.')}"
        )

    assert still_open, (
        f"FAIL [{request_id}] this core change request appears to be DONE.\n"
        f"  → Requested: {entry.get('request')}\n"
        f"  → Owner: {entry.get('owner')}\n"
        f"  → Delete CR entry '{request_id}' from ci/policy.yml and run `make docs-ci`.\n"
        f"  → Leaving it in place would publish a false claim about "
        f"{entry.get('file')} in docs/ci-cd.md."
    )
