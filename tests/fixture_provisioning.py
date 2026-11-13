"""
Automatic fixture provisioning — the test suite fetches its own data.

WHAT THIS CLOSES
----------------
Fixture handling used to be a Makefile concern: `make test-integration` depended
on the `ensure-fixtures` target, which downloaded anything missing. Anyone who
ran pytest directly — a developer with `.venv/bin/pytest tests/`, an IDE runner,
or the GitLab jobs, which invoke `poetry run pytest ...` and never go through
make — got no such help. A missing fixture then surfaced as a SKIP in the
discovery contract or a FileNotFoundError deep inside the integration suite.

So the check now lives in the test session itself (tests/conftest.py calls
`ensure_sample_fixtures()` from `pytest_sessionstart`): before a single test
runs, every discovered dataset is checked for its sample fixture, and anything
missing is downloaded as a small `is_dummy`-sized chunk. Present fixtures are
never re-fetched, so the normal case costs a handful of `stat()` calls.

NOTHING HERE NAMES A DATASET
----------------------------
The list of fixtures to check comes from dataset auto-discovery in
`scripts/create_sample_data.py` (`missing_fixtures()`), which scans
`seanergys_modelzoo/datasets/` for concrete `SeanergysDataset` subclasses and
derives each file name from the class name. Add a dataset and its fixture is
checked — and fetched — with no edit here.

MODES
-----
`MODELZOO_FIXTURES` (or the older `CI_SKIP_ZENODO`) decides what may happen:

  auto   (default)  download only what is missing
  never             never download; report the gap and let tests skip
  force             re-download every fixture, even if present

`CI_SKIP_ZENODO=1` implies `never`: that variable's contract in this repo is
"this run does not touch the network", and fixture provisioning must honour it
rather than quietly opening a socket. CI keeps the fixtures committed, so the
gate there is a presence check, not a download.

The download itself is delegated to `scripts/create_sample_data.py` in a
subprocess, so there is exactly one implementation of "fetch a sample" and this
module cannot drift from `make sample-data`.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _PROJECT_ROOT / "scripts" / "create_sample_data.py"
_FIXTURE_DIR = _PROJECT_ROOT / "tests" / "fixtures" / "sample_data"

# A sample download is a few tens of MB over HTTP range requests; generous, but
# bounded, so a hung mirror fails the session instead of stalling it forever.
_DOWNLOAD_TIMEOUT_S = int(os.environ.get("MODELZOO_FIXTURE_TIMEOUT", "1800"))

MODE_AUTO = "auto"
MODE_NEVER = "never"
MODE_FORCE = "force"


@dataclass
class ProvisioningReport:
    """What the session did about fixtures, so a test can assert on it."""

    mode: str = MODE_AUTO
    reason: str = ""
    missing_before: Dict[str, str] = field(default_factory=dict)
    missing_after: Dict[str, str] = field(default_factory=dict)
    downloaded: List[str] = field(default_factory=list)
    attempted: bool = False
    succeeded: bool = False
    output: str = ""
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        """True when no samplable dataset is left without a fixture."""
        return not self.missing_after

    def summary(self) -> str:
        if self.ok and not self.attempted:
            return f"fixtures: all present in {_FIXTURE_DIR.name}/ (mode={self.mode})"
        if self.ok:
            return (
                f"fixtures: downloaded {', '.join(self.downloaded)} "
                f"in {self.seconds:.0f}s (mode={self.mode})"
            )
        if self.mode == MODE_NEVER:
            return (
                f"fixtures: MISSING {', '.join(sorted(self.missing_after))} — "
                f"downloading disabled ({self.reason}); data tests will skip"
            )
        return (
            f"fixtures: MISSING {', '.join(sorted(self.missing_after))} after a "
            f"download attempt ({self.seconds:.0f}s); data tests will skip"
        )


def _resolve_mode() -> tuple:
    """(mode, reason) from the environment. CI_SKIP_ZENODO=1 wins over auto."""
    requested = (os.environ.get("MODELZOO_FIXTURES") or "").strip().lower()
    if requested in (MODE_NEVER, MODE_FORCE, MODE_AUTO):
        if requested == MODE_FORCE:
            return MODE_FORCE, "MODELZOO_FIXTURES=force"
        if requested == MODE_NEVER:
            return MODE_NEVER, "MODELZOO_FIXTURES=never"
    if os.environ.get("CI_SKIP_ZENODO") == "1":
        return MODE_NEVER, "CI_SKIP_ZENODO=1 (this run must not touch the network)"
    if requested == MODE_AUTO:
        return MODE_AUTO, "MODELZOO_FIXTURES=auto"
    return MODE_AUTO, "default"


def _load_sample_script():
    """
    Import scripts/create_sample_data.py as a module (it is not a package).

    Only its offline helpers are used here — `missing_fixtures()` discovers
    datasets and stats files; it imports no model and opens no socket.
    """
    spec = importlib.util.spec_from_file_location("_create_sample_data", _SCRIPT)
    if spec is None or spec.loader is None:  # pragma: no cover - unreachable in repo
        raise ImportError(f"cannot load {_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def missing_fixture_paths() -> Dict[str, str]:
    """
    {DatasetClassName: expected fixture path} for every samplable dataset whose
    fixture is absent or empty. Offline and cheap; {} means nothing to do.

    Returns {} if the ML dependencies are not installed — a dependency-less
    environment has no dataset classes to check, and that is reported by the
    discovery tests, not by turning fixture provisioning into an import error.
    """
    try:
        module = _load_sample_script()
        return {
            name: str(path)
            for name, path in module.missing_fixtures(_FIXTURE_DIR).items()
        }
    except Exception:
        return {}


def _run_download(force: bool) -> tuple:
    """Run scripts/create_sample_data.py in a subprocess. (succeeded, output)."""
    cmd = [sys.executable, str(_SCRIPT)]
    if force:
        cmd.append("--force")

    env = dict(os.environ)
    # The child does the downloading, so the "no network" flag must not follow it
    # in — we only reach this point after deciding a download is allowed.
    env.pop("CI_SKIP_ZENODO", None)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_PROJECT_ROOT), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)

    try:
        completed = subprocess.run(
            cmd,
            cwd=str(_PROJECT_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=_DOWNLOAD_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return False, f"timed out after {_DOWNLOAD_TIMEOUT_S}s"
    except Exception as exc:  # pragma: no cover - defensive
        return False, f"{type(exc).__name__}: {exc}"

    return completed.returncode == 0, (completed.stdout or "") + (
        completed.stderr or ""
    )


def ensure_sample_fixtures(verbose: bool = True) -> ProvisioningReport:
    """
    Make sure every discovered dataset has its sample fixture, downloading the
    missing ones when the mode allows it.

    Never raises: a failed or forbidden download is recorded in the report and
    the affected tests skip with a precise message. Turning it into a session
    error would make an offline machine unable to run the unit suite at all.
    """
    mode, reason = _resolve_mode()
    report = ProvisioningReport(mode=mode, reason=reason)
    _FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    report.missing_before = missing_fixture_paths()

    if mode == MODE_FORCE:
        needs_download = True
    else:
        needs_download = bool(report.missing_before)

    if not needs_download or mode == MODE_NEVER:
        report.missing_after = dict(report.missing_before)
        if verbose and (report.missing_after or mode == MODE_FORCE):
            print(f"\n[fixtures] {report.summary()}", file=sys.stderr)
        return report

    if verbose:
        wanted = ", ".join(sorted(report.missing_before)) or "all (force)"
        print(
            f"\n[fixtures] missing: {wanted} — fetching a small sample via "
            f"scripts/create_sample_data.py (set MODELZOO_FIXTURES=never or "
            f"CI_SKIP_ZENODO=1 to disable)",
            file=sys.stderr,
        )

    started = time.monotonic()
    report.attempted = True
    report.succeeded, report.output = _run_download(force=(mode == MODE_FORCE))
    report.seconds = time.monotonic() - started

    report.missing_after = missing_fixture_paths()
    report.downloaded = sorted(set(report.missing_before) - set(report.missing_after))
    if mode == MODE_FORCE and report.succeeded and not report.downloaded:
        # Nothing was missing, so no name appears in the difference; the refresh
        # still happened and the report should say so rather than "nothing".
        report.downloaded = ["<all fixtures refreshed>"]

    if verbose:
        print(f"[fixtures] {report.summary()}", file=sys.stderr)
        if not report.ok and report.output:
            tail = "\n".join(report.output.strip().splitlines()[-12:])
            print(f"[fixtures] downloader output (tail):\n{tail}", file=sys.stderr)

    return report
