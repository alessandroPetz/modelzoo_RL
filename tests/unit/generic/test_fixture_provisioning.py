"""
The suite must provision its own data — automatically, and for any dataset.

A test run should never be blocked on "someone forgot to run `make sample-data`".
tests/fixture_provisioning.py checks every discovered dataset for its sample
parquet at session start and downloads the missing ones as a small
is_dummy-sized chunk; tests/conftest.py calls it from pytest_sessionstart, so
the behaviour reaches every runner — make, a bare `pytest`, an IDE, CI.

This module pins that behaviour down:

  1. the missing-fixture check is derived from dataset discovery, not a list
  2. downloading is skipped when the environment forbids it, and attempted
     otherwise — never the other way round
  3. a successful download really does clear the gap (proved with a stand-in
     downloader, so the check needs no network)
  4. the Makefile and the pytest hook ask the SAME question, so `make
     ensure-fixtures` and a plain `pytest` can never disagree
  5. whatever is still missing at the end of the session is explained

Nothing here names a dataset: the expected fixture names come from
tests.conftest.fixture_name_for() applied to DATASET_CLASSES.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

import torch  # noqa: F401

from tests.conftest import DATASET_CLASSES, fixture_name_for
from tests import fixture_provisioning as fp

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# 1. What gets checked comes from discovery
# ---------------------------------------------------------------------------


def test_missing_check_is_derived_from_discovered_datasets():
    """
    PASS: every dataset the provisioner reports as missing is a discovered
          dataset, and the path it expects follows the shared naming rule.
    FAIL: the provisioner invented a dataset or a file name of its own, which
          would make it drift from the fixtures the tests actually read.
    """
    missing = fp.missing_fixture_paths()

    unknown = sorted(set(missing) - set(DATASET_CLASSES))
    assert not unknown, (
        f"FAIL the fixture provisioner reports {unknown} as missing, but no such "
        f"dataset was discovered.\n"
        f"  → It must derive its list from the dataset classes in "
        f"seanergys_modelzoo/datasets/, never from a hand-written list."
    )

    for name, path in missing.items():
        expected = fixture_name_for(DATASET_CLASSES[name])
        assert Path(path).name == expected, (
            f"FAIL [{name}] provisioner expects {Path(path).name!r} but the tests "
            f"read {expected!r}.\n"
            f"  → Both must come from tests.conftest.fixture_name_for()."
        )


def test_every_discovered_dataset_is_either_checked_or_unsamplable():
    """
    PASS: each discovered dataset either has its fixture on disk, is reported as
          missing (so provisioning will fetch it), or is a source that
          create_sample_data.py cannot slice into a single parquet.
    FAIL: a dataset falls through all three, meaning nobody would ever notice its
          fixture is absent.
    """
    fixture_dir = _PROJECT_ROOT / "tests" / "fixtures" / "sample_data"
    missing = fp.missing_fixture_paths()

    script = fp._load_sample_script()
    samplable = set(script._samplable_dataset_classes())

    for name, cls in DATASET_CLASSES.items():
        on_disk = (fixture_dir / fixture_name_for(cls)).exists()
        assert on_disk or name in missing or name not in samplable, (
            f"FAIL [{name}] has no fixture, is not reported missing, and is not "
            f"declared unsamplable.\n"
            f"  → It would silently contribute no data-driven test case."
        )


# ---------------------------------------------------------------------------
# 2. The mode decides whether the network may be touched
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "env,expected",
    [
        ({"CI_SKIP_ZENODO": "1"}, fp.MODE_NEVER),
        ({"MODELZOO_FIXTURES": "never"}, fp.MODE_NEVER),
        ({"MODELZOO_FIXTURES": "force"}, fp.MODE_FORCE),
        ({"MODELZOO_FIXTURES": "auto"}, fp.MODE_AUTO),
        ({}, fp.MODE_AUTO),
    ],
    ids=["ci-skip-zenodo", "never", "force", "auto", "default"],
)
def test_mode_resolution(monkeypatch, env, expected):
    """
    PASS: the environment maps to the documented mode.
    FAIL: an offline run could start downloading, or a developer's run would
          refuse to fetch what it needs.
    """
    for key in ("CI_SKIP_ZENODO", "MODELZOO_FIXTURES"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    mode, reason = fp._resolve_mode()
    assert mode == expected, f"expected {expected}, got {mode} ({reason})"
    assert reason, "every mode decision must carry a reason for the log line"


def test_no_download_when_downloading_is_forbidden(monkeypatch, tmp_path):
    """
    PASS: with CI_SKIP_ZENODO=1 the provisioner reports the gap and does not run
          the downloader.
    FAIL: an offline/CI run would open a socket — exactly what that flag exists
          to prevent.
    """
    monkeypatch.setenv("CI_SKIP_ZENODO", "1")
    _install_stub_script(monkeypatch, tmp_path, missing={"Some": "x.parquet"})

    def _must_not_run(force):  # pragma: no cover - the assertion is that it isn't
        raise AssertionError("downloader ran while CI_SKIP_ZENODO=1")

    monkeypatch.setattr(fp, "_run_download", _must_not_run)

    report = fp.ensure_sample_fixtures(verbose=False)

    assert report.mode == fp.MODE_NEVER
    assert not report.attempted
    assert report.missing_after, "the gap must still be reported, not hidden"
    assert "download" in report.summary().lower()


# ---------------------------------------------------------------------------
# 3. A successful download clears the gap (no network needed to check this)
# ---------------------------------------------------------------------------


def _install_stub_script(monkeypatch, tmp_path, missing: dict):
    """
    Replace the downloader script with an in-memory stand-in whose notion of
    "missing" is a mutable dict, so the success and failure paths can be driven
    deterministically and offline.
    """

    state = dict(missing)

    class _Stub:
        @staticmethod
        def missing_fixtures(out_dir):
            return {name: Path(out_dir) / path for name, path in state.items()}

        @staticmethod
        def _samplable_dataset_classes():
            return {}

    monkeypatch.setattr(fp, "_FIXTURE_DIR", tmp_path)
    monkeypatch.setattr(fp, "_load_sample_script", lambda: _Stub)
    return state


def test_successful_download_clears_the_gap(monkeypatch, tmp_path):
    """
    PASS: a missing fixture triggers the downloader, and once it succeeds the
          report says the gap is closed and names what it fetched.
    FAIL: the provisioner would report success without the data, or success
          without saying what it did.
    """
    monkeypatch.delenv("CI_SKIP_ZENODO", raising=False)
    monkeypatch.delenv("MODELZOO_FIXTURES", raising=False)
    state = _install_stub_script(monkeypatch, tmp_path, missing={"Alpha": "a.parquet"})

    def _fake_download(force):
        state.clear()  # the download "created" the file
        return True, "ok"

    monkeypatch.setattr(fp, "_run_download", _fake_download)

    report = fp.ensure_sample_fixtures(verbose=False)

    assert report.attempted and report.succeeded
    assert report.ok, report.summary()
    assert report.downloaded == ["Alpha"]


def test_failed_download_is_reported_not_raised(monkeypatch, tmp_path):
    """
    PASS: a download that fails (Zenodo down, no network) leaves a report saying
          so, and the session continues — data tests skip with a reason.
    FAIL: a flaky external service would turn the whole run red, including the
          unit tests that need no data at all.
    """
    monkeypatch.delenv("CI_SKIP_ZENODO", raising=False)
    monkeypatch.delenv("MODELZOO_FIXTURES", raising=False)
    _install_stub_script(monkeypatch, tmp_path, missing={"Alpha": "a.parquet"})
    monkeypatch.setattr(fp, "_run_download", lambda force: (False, "HTTP 504"))

    report = fp.ensure_sample_fixtures(verbose=False)

    assert report.attempted and not report.succeeded
    assert not report.ok
    assert "Alpha" in report.summary()


# ---------------------------------------------------------------------------
# 4. make and pytest must ask the same question
# ---------------------------------------------------------------------------


def test_make_and_pytest_agree_on_what_is_missing():
    """
    PASS: `create_sample_data.py --list-missing` (what `make ensure-fixtures`
          runs) returns exactly what the pytest hook computes.
    FAIL: the two paths have drifted, so `make test` and a bare `pytest` would
          disagree about whether the suite has its data.
    """
    result = subprocess.run(
        [
            sys.executable,
            str(_PROJECT_ROOT / "scripts" / "create_sample_data.py"),
            "--list-missing",
        ],
        cwd=str(_PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "PYTHONPATH": str(_PROJECT_ROOT)},
    )
    assert result.returncode == 0, (
        f"--list-missing failed ({result.returncode}):\n{result.stderr[-1500:]}"
    )

    from_cli = sorted(
        line.strip() for line in result.stdout.splitlines() if line.strip()
    )
    from_api = sorted(fp.missing_fixture_paths())

    assert from_cli == from_api, (
        f"FAIL make and pytest disagree about missing fixtures:\n"
        f"  --list-missing : {from_cli}\n"
        f"  pytest hook    : {from_api}\n"
        f"  → Both must call create_sample_data.missing_fixtures()."
    )


# ---------------------------------------------------------------------------
# 5. Whatever is still missing is explained
# ---------------------------------------------------------------------------


def test_no_undeclared_fixture_gap_when_downloading_is_disabled(fixture_provisioning):
    """
    A fixture that cannot be fetched must be committed — or declared.

    On a developer's machine a missing fixture is downloaded and the run
    continues. On a runner, downloading is off (MODELZOO_FIXTURES=never in
    .gitlab-ci.yml, because pulling a multi-hundred-MB Zenodo file OOM-kills the
    job), so the same situation means the repo simply cannot exercise that
    dataset offline. Letting that pass as a SKIP is how a dataset ends up
    "in CI" while contributing zero executed cases.

    PASS: every samplable dataset has its fixture, or the dataset is recorded in
          ci/policy.yml -> dataset_scope with a reason.
    SKIP: a download was attempted and failed — an external outage is not the
          contributor's fault and must not turn unrelated runs red.
    FAIL: downloading is disabled, a fixture is missing, and nothing declares it.
    """
    report = fixture_provisioning
    assert report is not None, "no provisioning report — the conftest hook is missing"

    if report.ok:
        return

    if report.attempted and not report.succeeded:
        pytest.skip(
            f"SKIP the fixture download failed, so this cannot be judged here: "
            f"{report.summary()}\n"
            f"  → Re-run when the source is reachable, or commit the fixture."
        )

    from tests.conftest import load_dataset_scope

    scope = load_dataset_scope()
    undeclared = sorted(set(report.missing_after) - set(scope))
    assert not undeclared, (
        f"FAIL no fixture for {', '.join(undeclared)} and downloading is disabled "
        f"({report.reason}).\n"
        f"  → The fixtures are committed in tests/fixtures/sample_data/. Run\n"
        f"    `make sample-data` locally and commit the new file, so every runner\n"
        f"    can exercise the dataset offline.\n"
        f"  → If the dataset genuinely cannot be sampled, record it under\n"
        f"    'dataset_scope' in ci/policy.yml with the reason and the owner's fix."
    )


def test_session_report_explains_any_remaining_gap(fixture_provisioning):
    """
    PASS: either every fixture is present, or the session report says why not
          (downloading disabled, or the download failed) — so a skipped data
          test is never a mystery.
    FAIL: fixtures are missing and the session has no record of having tried,
          which is the silent-hole case this mechanism exists to remove.
    """
    report = fixture_provisioning
    assert report is not None, (
        "pytest_sessionstart did not record a provisioning report — the hook in "
        "tests/conftest.py is not wired up."
    )

    if report.ok:
        return

    assert report.mode == fp.MODE_NEVER or report.attempted, (
        f"FAIL fixtures are missing ({sorted(report.missing_after)}) but nothing "
        f"tried to fetch them and no mode forbids it.\n  → {report.summary()}"
    )
    assert report.reason, "a 'never' decision must name the environment that caused it"
