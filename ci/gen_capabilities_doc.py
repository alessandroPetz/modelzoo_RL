#!/usr/bin/env python3
"""
Generate the CI/CD capabilities & rules reference (docs/ci-cd.md).

The tables in docs/ci-cd.md are AUTO-GENERATED from three sources of truth so they
can never silently drift from what the pipeline actually does:

  1. .gitlab-ci.yml        — jobs, stages, trigger rules, allow_failure (the "what")
  2. the code (discovery)  — model×dataset capability matrix + test inventory
  3. ci/policy.yml         — the engineering rationale behind each rule (the "why",
                             which cannot be introspected and is edited by hand)

Only the regions between ``<!-- BEGIN:GENERATED <block> -->`` and
``<!-- END:GENERATED <block> -->`` markers are rewritten; hand-written prose outside
the markers is preserved. Generated regions contain NO timestamps, so regenerating
an unchanged repo is a no-op (this is what lets the docs:autogen CI job commit only
real changes without looping).

Usage:
    python ci/gen_capabilities_doc.py            # write/update docs/ci-cd.md
    python ci/gen_capabilities_doc.py --check    # exit 1 if the doc is stale
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Discovery instantiates models, which log at INFO. Silence that noise so the
# generator's own output stays clean (important for the docs:check job log).
logging.disable(logging.INFO)

try:
    import yaml
except ImportError:  # pragma: no cover - yaml is a dev dependency
    print("PyYAML is required: pip install pyyaml", file=sys.stderr)
    sys.exit(2)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_GITLAB_CI = _REPO_ROOT / ".gitlab-ci.yml"
_POLICY = _REPO_ROOT / "ci" / "policy.yml"
_DOC = _REPO_ROOT / "docs" / "ci-cd.md"
_CONFTEST = _REPO_ROOT / "tests" / "conftest.py"

# Keys in .gitlab-ci.yml that are configuration, not jobs.
_NON_JOB_KEYS = {"stages", "variables", "workflow", "include", "default"}


# ---------------------------------------------------------------------------
# .gitlab-ci.yml introspection
# ---------------------------------------------------------------------------


def _load_ci() -> Dict[str, Any]:
    with open(_GITLAB_CI, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _anchor_rules(ci: Dict[str, Any]) -> Dict[str, list]:
    """Map each `.rules_*` hidden-job anchor to its resolved `rules` list."""
    out: Dict[str, list] = {}
    for key, val in ci.items():
        if key.startswith(".rules_") and isinstance(val, dict) and "rules" in val:
            out[key.lstrip(".")] = val["rules"]
    return out


def _positive_rules(rules: list) -> list:
    """Rules that can cause the job to run (i.e. not `when: never`)."""
    return [r for r in rules if not (isinstance(r, dict) and r.get("when") == "never")]


def _trigger_label(job: Dict[str, Any], anchor_rules: Dict[str, list]) -> str:
    """Human-readable trigger for a job, matched back to a known rule anchor."""
    rules = job.get("rules")
    if rules is None:
        return "always"
    # Match against a known anchor by value.
    for name, arules in anchor_rules.items():
        if rules == arules:
            return name
    positive = _positive_rules(rules)
    # A positive rule gated on the default branch → runs post-merge on main.
    for rule in positive:
        if isinstance(rule, dict) and "CI_DEFAULT_BRANCH" in str(rule.get("if", "")):
            return "default-branch"
    # Default branch is explicitly excluded but the job runs on file changes → MRs.
    excludes_default = any(
        isinstance(r, dict)
        and r.get("when") == "never"
        and "CI_DEFAULT_BRANCH" in str(r.get("if", ""))
        for r in rules
    )
    if excludes_default and any(
        "changes" in r for r in positive if isinstance(r, dict)
    ):
        return "merge-request"
    return "custom"


def _outcome(job: Dict[str, Any]) -> str:
    return "Warns" if job.get("allow_failure") else "Blocks"


# Setup/housekeeping command prefixes that are not the "point" of a job.
_NOISE_PREFIXES = (
    "echo",
    "ls ",
    "ls\t",
    "mkdir",
    "chmod",
    "cd ",
    "export ",
    "if ",
    "fi",
    "pip install",
    "pip3 install",
    "apt-get",
    "poetry install",
    "poetry config",
    "git config",
    "git add",
    "git commit",
    "test -",
    "find ",
)


def _script_summary(job: Dict[str, Any]) -> str:
    """One-line-ish summary of what a job runs (first meaningful, non-setup command)."""
    script = job.get("script") or []
    if isinstance(script, str):
        script = [script]
    cmds = [" ".join(s.split()) for s in script if isinstance(s, str) and s.strip()]

    def is_noise(c: str) -> bool:
        cl = c.lstrip("-").strip()
        return cl.startswith(_NOISE_PREFIXES) or cl.startswith(("|", "fi", "else"))

    non_noise = [c for c in cmds if not is_noise(c)]
    preferred = (
        "pytest",
        "ci/validate_yaml",
        "ci/verify_wheel",
        "gen_capabilities_doc",
        "coverage combine",
        "coverage xml",
        "poetry build",
        "poetry export",
        "ruff check",
        "ruff format",
        "mypy ",
    )
    # 1. real-work line that is not setup noise
    for c in non_noise:
        if any(k in c for k in preferred):
            return _clip(c)
    # 2. any line (incl. block scalars) with a real-work keyword
    for c in cmds:
        if any(k in c for k in preferred):
            return _clip(c)
    # 3. first non-noise line, else first line
    if non_noise:
        return _clip(non_noise[0])
    return _clip(cmds[0]) if cmds else "—"


def _clip(s: str, n: int = 90) -> str:
    s = s.replace("|", "\\|")
    return s if len(s) <= n else s[: n - 1] + "…"


def _collect_jobs(ci: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
    anchor_rules = _anchor_rules(ci)
    stages: List[str] = ci.get("stages", [])
    jobs: List[Dict[str, Any]] = []
    for name, val in ci.items():
        if name.startswith(".") or name in _NON_JOB_KEYS:
            continue
        if not isinstance(val, dict) or "stage" not in val:
            continue
        jobs.append(
            {
                "name": name,
                "stage": val.get("stage", ""),
                "trigger": _trigger_label(val, anchor_rules),
                "outcome": _outcome(val),
                "runs": _script_summary(val),
            }
        )
    # Order jobs by stage order, then by name.
    stage_index = {s: i for i, s in enumerate(stages)}
    jobs.sort(key=lambda j: (stage_index.get(j["stage"], 99), j["name"]))
    return jobs, stages


# ---------------------------------------------------------------------------
# Code discovery (reuse the exact discovery the tests use)
# ---------------------------------------------------------------------------


def _load_conftest() -> Optional[Any]:
    """Import tests/conftest.py to reuse MODEL_CLASSES/DATASET_CLASSES/pairs."""
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    spec = importlib.util.spec_from_file_location("_seanergys_conftest", _CONFTEST)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:  # pragma: no cover - discovery needs ML deps installed
        print(f"WARNING: could not import conftest for discovery: {e}", file=sys.stderr)
        return None
    return module


def _model_task(model_cls) -> str:
    try:
        inst = model_cls(model_hyperparameters={"n_jobs": 1})
        task = getattr(inst, "task_type", None)
        if task is not None:
            # Prefer the readable enum name (e.g. REGRESSION) over its int value.
            name = getattr(task, "name", None)
            if name:
                return str(name).replace("_", " ").title()
            return str(getattr(task, "value", task))
    except Exception:
        pass
    return "—"


def _capabilities(conftest) -> Tuple[List[dict], List[dict]]:
    """Return (capability rows [model×dataset], dataset rows)."""
    if conftest is None:
        return [], []

    models = getattr(conftest, "MODEL_CLASSES", {})
    datasets = getattr(conftest, "DATASET_CLASSES", {})
    pairs = getattr(conftest, "MODEL_DATASET_PAIRS", [])
    tested_ids = {(m.__name__, d) for m, d in pairs}

    cap_rows: List[dict] = []
    for model_name in sorted(models):
        model_cls = models[model_name]
        task = _model_task(model_cls)
        try:
            supported = (
                model_cls(model_hyperparameters={"n_jobs": 1}).supported_datasets or []
            )
        except Exception:
            supported = []
        if not supported:
            cap_rows.append(
                {
                    "model": model_name,
                    "task": task,
                    "dataset": "—",
                    "dataset_id": "—",
                    "tested": "no",
                }
            )
            continue
        for ds_cls in supported:
            ds_id = getattr(ds_cls, "DATASET_ID", None)
            # DATASET_ID may be a pydantic field default rather than a plain attr.
            if ds_id is None and hasattr(ds_cls, "model_fields"):
                fld = ds_cls.model_fields.get("DATASET_ID")
                ds_id = getattr(fld, "default", None) if fld else None
            ds_id = ds_id or "—"
            # No "xfail" state any more: a pair is either exercised or it is not.
            # A pair that IS exercised and fails shows up as a red pipeline, not
            # as a tolerated cell in this table.
            tested = "yes" if (model_name, ds_id) in tested_ids else "no"
            cap_rows.append(
                {
                    "model": model_name,
                    "task": task,
                    "dataset": ds_cls.__name__,
                    "dataset_id": ds_id,
                    "tested": tested,
                }
            )

    ds_rows: List[dict] = []
    for ds_name in sorted(datasets):
        ds_cls = datasets[ds_name]
        ds_id = getattr(ds_cls, "DATASET_ID", None)
        if ds_id is None and hasattr(ds_cls, "model_fields"):
            fld = ds_cls.model_fields.get("DATASET_ID")
            ds_id = getattr(fld, "default", None) if fld else None
        has_dummy = "yes" if "is_dummy" in getattr(ds_cls, "model_fields", {}) else "no"

        # A dataset is "covered" when some model consumes it, i.e. it appears in
        # a (model, dataset) pair. One that is discovered but reachable from no
        # model contributes zero executed cases, so it must be recorded in
        # ci/policy.yml -> dataset_scope; tests/unit/test_discovery_contract.py
        # fails the unit job otherwise.
        scope = (_load_policy().get("dataset_scope") or {}).get(ds_name, {})
        covered = "yes" if any(r["dataset"] == ds_name for r in cap_rows) else "no"
        ds_rows.append(
            {
                "dataset": ds_name,
                "dataset_id": ds_id or "—",
                "is_dummy": has_dummy,
                "covered": covered,
                "missing": scope.get("missing", "") or "—",
                "reason": scope.get("reason", "")
                or (
                    "—"
                    if covered == "yes"
                    else "**undocumented — add a dataset_scope entry to ci/policy.yml**"
                ),
                "owner_action": scope.get("owner_action", "") or "—",
            }
        )
    return cap_rows, ds_rows


def _test_inventory() -> List[dict]:
    """List test files per category with their module docstring first line."""
    rows: List[dict] = []
    for category in ("unit", "smoke", "integration"):
        cat_dir = _REPO_ROOT / "tests" / category
        if not cat_dir.is_dir():
            continue
        for f in sorted(cat_dir.glob("test_*.py")):
            rows.append(
                {
                    "category": category,
                    "file": f.name,
                    "purpose": _first_docstring_line(f),
                }
            )
    return rows


def _first_docstring_line(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return "—"
    m = re.search(r'"""(.*?)"""', text, re.DOTALL)
    if not m:
        return "—"
    for line in m.group(1).strip().splitlines():
        if line.strip():
            return _clip(line.strip(), 80)
    return "—"


# ---------------------------------------------------------------------------
# Test-skip inventory
# A skip is the one place a green run can silently hide an untested path, so every
# skip in the suite is inventoried here — where it is, why it skips, and how to
# make it actually run. Scanned from the source (AST) so the table never goes stale.
# ---------------------------------------------------------------------------


def _is_pytest_skip(func: ast.expr) -> bool:
    """True for a `pytest.skip` attribute access."""
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "skip"
        and isinstance(func.value, ast.Name)
        and func.value.id == "pytest"
    )


def _is_zenodo_marker(dec: ast.expr) -> bool:
    """True for a `@pytest.mark.zenodo` decorator."""
    return (
        isinstance(dec, ast.Attribute)
        and dec.attr == "zenodo"
        and isinstance(dec.value, ast.Attribute)
        and dec.value.attr == "mark"
    )


def _flatten_str(node: ast.expr) -> str:
    """Best-effort reconstruction of a string/f-string/concatenation literal."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(
            str(v.value) if isinstance(v, ast.Constant) else "{…}" for v in node.values
        )
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _flatten_str(node.left) + _flatten_str(node.right)
    return ""


def _skip_reason_solution(call: ast.Call) -> Tuple[str, str]:
    """Split a skip message into (reason, how-to-remove).

    Skip messages in this suite embed the fix after a `→` arrow; use that when
    present. CI_SKIP_ZENODO gates get the standard "runs in test:zenodo" answer.
    """
    raw = _flatten_str(call.args[0]) if call.args else ""
    text = " ".join(raw.split())  # collapse newlines/indentation
    # Strip a leading "SKIP [X]" / "SKIP:" label so the reason reads cleanly.
    text = re.sub(r"^SKIP\s*(\[[^\]]*\])?\s*:?\s*", "", text).strip()
    if "→" in text:
        reason, _, sol = text.partition("→")
        return _clip(reason.strip(" -—"), 80), _clip(sol.strip(), 90)
    if "CI_SKIP_ZENODO" in text:
        return (
            _clip(text, 80),
            "Runs for real in the `test:zenodo` job (no CI_SKIP_ZENODO); "
            "locally run with network.",
        )
    # Some messages embed the fix after an em-dash ("… — implement dummy support").
    if " — " in text:
        reason, _, sol = text.partition(" — ")
        return _clip(reason.strip(), 80), _clip(sol.strip().capitalize(), 90)
    return _clip(text, 90), "—"


def _scan_test_skips() -> List[dict]:
    """Inventory every skip construct in tests/: `pytest.skip(...)` calls and
    `@pytest.mark.zenodo` markers, plus curated `skip_notes` from policy.yml for
    implicit skips (e.g. empty-parametrize) that are not literal skip() calls."""
    rows: List[dict] = []
    tests_dir = _REPO_ROOT / "tests"
    for f in sorted(tests_dir.rglob("test_*.py")):
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        rel = f.relative_to(_REPO_ROOT).as_posix()
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for dec in fn.decorator_list:
                if _is_zenodo_marker(dec):
                    rows.append(
                        {
                            "file": rel,
                            "loc": f"`{fn.name}` (L{dec.lineno})",
                            "reason": "Needs a live Zenodo download; deselected by "
                            '`-m "not zenodo"` and skipped when CI_SKIP_ZENODO=1.',
                            "solution": "Runs for real in the `test:zenodo` job.",
                        }
                    )
            for node in ast.walk(fn):
                if isinstance(node, ast.Call) and _is_pytest_skip(node.func):
                    reason, solution = _skip_reason_solution(node)
                    rows.append(
                        {
                            "file": rel,
                            "loc": f"`{fn.name}` (L{node.lineno})",
                            "reason": reason,
                            "solution": solution,
                        }
                    )
    for note in _load_policy().get("skip_notes") or []:
        rows.append(
            {
                "file": note.get("file", "—"),
                "loc": note.get("location", "—"),
                "reason": note.get("reason", "—"),
                "solution": note.get("solution", "—"),
            }
        )
    rows.sort(key=lambda r: (r["file"], r["loc"]))
    return rows


# ---------------------------------------------------------------------------
# ci/policy.yml (rationale)
# ---------------------------------------------------------------------------


def _load_policy() -> Dict[str, Any]:
    if not _POLICY.exists():
        return {}
    with open(_POLICY, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# Markdown table + block rendering
# ---------------------------------------------------------------------------


def _table(headers: List[str], rows: List[List[str]]) -> str:
    if not rows:
        rows = [["_(none discovered)_"] + [""] * (len(headers) - 1)]
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _build_blocks() -> Dict[str, str]:
    ci = _load_ci()
    jobs, stages = _collect_jobs(ci)
    policy = _load_policy()
    conftest = _load_conftest()
    cap_rows, ds_rows = _capabilities(conftest)
    inventory = _test_inventory()

    rule_desc = policy.get("rules", {})
    job_rationale = policy.get("jobs", {})

    blocks: Dict[str, str] = {}

    # Pipeline stages.
    blocks["pipeline-stages"] = "**Stage order:** " + " → ".join(
        f"`{s}`" for s in stages
    )

    # Jobs table.
    job_rows = []
    for j in jobs:
        trig = j["trigger"]
        trig_txt = f"`{trig}`"
        job_rows.append(
            [j["stage"], f"`{j['name']}`", trig_txt, j["outcome"], j["runs"]]
        )
    blocks["jobs"] = _table(["Stage", "Job", "Trigger", "On failure", "Runs"], job_rows)

    # Trigger legend (from policy).
    legend_rows = []
    for name in sorted({j["trigger"] for j in jobs}):
        desc = rule_desc.get(name, {}).get("description", "")
        legend_rows.append([f"`{name}`", desc or "—"])
    blocks["triggers"] = _table(["Trigger", "When the job runs"], legend_rows)

    # Pass/fail rationale (why blocks vs warns).
    rationale_rows = []
    for j in jobs:
        why = job_rationale.get(j["name"], {}).get("rationale", "")
        rationale_rows.append([f"`{j['name']}`", j["outcome"], why or "—"])
    blocks["rationale"] = _table(["Job", "On failure", "Why"], rationale_rows)

    # Capability matrix.
    cap_tbl = [
        [r["model"], r["task"], r["dataset"], f"`{r['dataset_id']}`", r["tested"]]
        for r in cap_rows
    ]
    blocks["capabilities"] = _table(
        ["Model", "Task", "Supported dataset", "DATASET_ID", "Tested in CI"], cap_tbl
    )

    # Dataset table.
    ds_tbl = [
        [r["dataset"], f"`{r['dataset_id']}`", r["is_dummy"], r["covered"]]
        for r in ds_rows
    ]
    blocks["datasets"] = _table(
        ["Dataset", "DATASET_ID", "is_dummy mode", "Exercised by CI"], ds_tbl
    )

    # Datasets discovered but not exercised — the coverage gaps, and who closes
    # each one. Empty table = every discovered dataset is genuinely tested.
    gap_tbl = [
        [r["dataset"], r["missing"], r["reason"], r["owner_action"]]
        for r in ds_rows
        if r["covered"] == "no"
    ]
    blocks["dataset-gaps"] = _table(
        [
            "Dataset",
            "What is missing",
            "Why it is not tested",
            "Who closes it, and how",
        ],
        gap_tbl,
    )

    # Core change requests — what must change in seanergys_modelzoo/, and who owns it.
    cr = _load_policy().get("core_change_requests") or {}
    cr_tbl = [
        [
            f"`{rid}`",
            # The GitLab issue carrying the full write-up: evidence, reproduction,
            # the proposed patch and acceptance criteria. The register keeps the
            # machine-checkable condition; the issue keeps the conversation.
            f"#{e['issue']}" if e.get("issue") else "—",
            f"`{e.get('symbol', '—')}`",
            e.get("owner", "—"),
            e.get("request", "—"),
            e.get("why_ci_cannot", "—"),
        ]
        for rid, e in sorted(cr.items())
    ]
    blocks["core-change-requests"] = _table(
        ["ID", "Tracker", "Symbol", "Owner", "What must change", "Why CI cannot do it"],
        cr_tbl,
    )

    # Test inventory.
    inv_tbl = [[r["category"], f"`{r['file']}`", r["purpose"]] for r in inventory]
    blocks["test-inventory"] = _table(["Layer", "File", "Checks"], inv_tbl)

    # Test-skip inventory (every place a test can be silently skipped).
    skip_tbl = [
        [f"`{r['file']}`", r["loc"], r["reason"], r["solution"]]
        for r in _scan_test_skips()
    ]
    blocks["test-skips"] = _table(
        ["File", "Test (line)", "Reason", "How to remove the skip"], skip_tbl
    )

    return blocks


# ---------------------------------------------------------------------------
# Marker replacement
# ---------------------------------------------------------------------------

_BEGIN = "<!-- BEGIN:GENERATED {name} -->"
_END = "<!-- END:GENERATED {name} -->"


def _render(existing: str, blocks: Dict[str, str]) -> str:
    """Replace each generated region in `existing`; return the new document."""
    result = existing
    for name, content in blocks.items():
        begin = _BEGIN.format(name=name)
        end = _END.format(name=name)
        pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.DOTALL)
        replacement = f"{begin}\n{content}\n{end}"
        if pattern.search(result):
            result = pattern.sub(lambda _m: replacement, result)
        else:
            print(
                f"WARNING: marker '{name}' not found in {_DOC.name}; block not written.",
                file=sys.stderr,
            )
    return result


def _scaffold(blocks: Dict[str, str]) -> str:
    """Create docs/ci-cd.md from scratch (prose + empty marker regions)."""
    names = list(blocks.keys())
    empty = {n: f"{_BEGIN.format(name=n)}\n{_END.format(name=n)}" for n in names}
    return _DOC_TEMPLATE.format(**empty)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="Exit 1 if docs are stale."
    )
    args = parser.parse_args()

    blocks = _build_blocks()

    if _DOC.exists():
        existing = _DOC.read_text(encoding="utf-8")
    else:
        existing = _scaffold(blocks)

    new = _render(existing, blocks)

    if args.check:
        if not _DOC.exists() or _DOC.read_text(encoding="utf-8") != new:
            print(
                f"✗ {_DOC.relative_to(_REPO_ROOT)} is STALE. "
                "Run `make docs-ci` (or `python ci/gen_capabilities_doc.py`) and commit.",
                file=sys.stderr,
            )
            return 1
        print(f"✓ {_DOC.relative_to(_REPO_ROOT)} is up to date.")
        return 0

    _DOC.parent.mkdir(parents=True, exist_ok=True)
    changed = not _DOC.exists() or _DOC.read_text(encoding="utf-8") != new
    _DOC.write_text(new, encoding="utf-8")
    print(f"{'✓ Updated' if changed else '• Unchanged'} {_DOC.relative_to(_REPO_ROOT)}")
    return 0


# The one-time scaffold used only when docs/ci-cd.md does not yet exist.
# After creation the prose is hand-editable; only marker regions are regenerated.
_DOC_TEMPLATE = """# CI/CD Capabilities & Rules Reference

> **This is the single source of truth for testing & CI in this repo.**
> The tables between the `<!-- BEGIN/END:GENERATED -->` markers are **auto-generated**
> by `ci/gen_capabilities_doc.py` from `.gitlab-ci.yml` (jobs/rules), the model/dataset
> code (capability matrix), and `ci/policy.yml` (rationale) — **do not edit them by hand**.
> Regenerate with `make docs-ci`. On the default branch the `docs:autogen` job regenerates
> and commits this file automatically, so the tables never go stale. Prose sections are
> hand-written and preserved across regeneration. For how the auto-generation works, see
> [`how-ci-cd-docs-are-generated.md`](how-ci-cd-docs-are-generated.md).
>
> **See also:** [`future_pydantic_tests.md`](future_pydantic_tests.md) (planned Pydantic
> validation tests) and [`MODELZOO_ARCHITECTURE_NOTES.md`](MODELZOO_ARCHITECTURE_NOTES.md)
> (packaging / serving notes).

## Purpose

This document is the single reference for **what the CI/CD pipeline tests, and
under which rules**, so it can be shared with colleagues. To request a change to
any rule, gate, or capability, open an issue with the **CI Change Request**
template (`.gitlab/issue_templates/CI_Change_Request.md`).

## Pipeline stages

{pipeline-stages}

## Jobs, triggers, and gates

Every job, the condition under which it runs, and whether a failure blocks the
pipeline or only warns.

{jobs}

### Trigger legend

{triggers}

## Pass / fail engineering decisions

Why each job blocks the pipeline or is allowed to fail. The *what* (above) is
auto-derived from `.gitlab-ci.yml`; the *why* below is maintained in
`ci/policy.yml`.

{rationale}

## Model × dataset capability matrix

Which models are tested against which datasets. Discovered automatically from each
model's `supported_datasets` and the auto-discovery in `tests/conftest.py` — adding
a model or dataset updates this table on the next regeneration.

{capabilities}

## Datasets

{datasets}

## Test inventory

Every test file grouped by layer, with the check it performs (from its docstring).

{test-inventory}

## Test-skip inventory

Every skip in the suite — where it is, why it skips, and how to make it run for real.
A skipped test is the one place a green pipeline can hide an untested path.

{test-skips}
"""


if __name__ == "__main__":
    sys.exit(main())
