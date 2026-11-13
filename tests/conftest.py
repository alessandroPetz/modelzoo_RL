"""
Pytest configuration and shared fixtures.

Auto-discovery at collection time:
  MODEL_CLASSES   — all concrete SeanergysModel subclasses in models/tasks/
  DATASET_CLASSES — all concrete SeanergysDataset subclasses in datasets/

Both dicts are used to parametrize integration and smoke tests.
Adding a new model or dataset to the repo automatically adds it to the test suite.
"""

import os
import signal
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Type

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def pytest_configure(config):
    """Ensure project root is on Python path."""
    root = str(_PROJECT_ROOT)
    if root not in os.environ.get("PYTHONPATH", ""):
        os.environ.setdefault("PYTHONPATH", root)


# Filled in by pytest_sessionstart below; read by
# tests/unit/generic/test_fixture_provisioning.py and by the
# `fixture_provisioning` session fixture.
FIXTURE_REPORT = None


def pytest_sessionstart(session):
    """
    Check — and if needed fetch — the sample fixtures before any test runs.

    The suite provisions its own data: every discovered dataset is checked for
    its sample parquet, and anything missing is downloaded as a small
    is_dummy-sized chunk. Present fixtures are never re-fetched, so the usual
    cost is a few stat() calls.

    Deliberately at session start rather than behind a per-test fixture: every
    layer (the discovery contract in unit, the data smoke tests, the whole
    integration suite) reads the same files, and a new test written by someone
    else must inherit the behaviour without having to request a fixture by name.

    Never fails the session. If downloading is disabled (CI_SKIP_ZENODO=1,
    MODELZOO_FIXTURES=never) or the download fails, the affected tests skip with
    a precise message and tests/unit/generic/test_fixture_provisioning.py
    reports the gap.
    """
    global FIXTURE_REPORT
    from tests.fixture_provisioning import ensure_sample_fixtures

    FIXTURE_REPORT = ensure_sample_fixtures()


# ---------------------------------------------------------------------------
# Auto-discovery helpers
# ---------------------------------------------------------------------------


# Seconds a model may take to instantiate or to assemble one train config.
# Both are supposed to be pure in-memory work; anything slower is doing I/O.
COLLECTION_TIME_BUDGET_S = int(os.environ.get("MODELZOO_COLLECT_TIMEOUT", "20"))

# Models excluded from discovery because they blew the budget above. Recorded so
# a test can report them instead of the suite silently shrinking.
SLOW_MODELS: Dict[str, str] = {}


class _CollectionTimeout(Exception):
    pass


@contextmanager
def _time_budget(model_name: str, what: str):
    """
    Abort a model's discovery step if it exceeds COLLECTION_TIME_BUDGET_S.

    Discovery must never depend on the network. Constructing a model and asking
    it for a train config are meant to be in-memory operations: build the
    estimator, fill in a params object. A model that instead builds a dataset in
    its constructor turns `pytest --collect-only` into a multi-hundred-MB
    download, which is how a full run came to hang for minutes.

    We cannot fix that from the test layer — it lives in the model, and the core
    belongs to its authors (see core_change_requests in ci/policy.yml). What we
    can do is refuse to hang: the offending model is dropped from discovery and
    recorded in SLOW_MODELS, and tests/unit/generic/test_discovery_contract.py
    turns that into a visible failure rather than a suite that quietly shrank.

    SIGALRM is Unix-only; on any platform without it the guard is a no-op and
    behaviour is exactly as before.
    """
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def _fire(_signum, _frame):
        raise _CollectionTimeout(
            f"{model_name}: {what} exceeded {COLLECTION_TIME_BUDGET_S}s — "
            f"it is doing I/O during discovery."
        )

    previous = signal.signal(signal.SIGALRM, _fire)
    signal.alarm(COLLECTION_TIME_BUDGET_S)
    try:
        yield
    except _CollectionTimeout as exc:
        SLOW_MODELS[model_name] = str(exc)
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def _collect_model_classes() -> Dict[str, Type]:
    """
    Discover all concrete SeanergysModel subclasses in models/tasks/.
    Only scans tasks/ so base/abstract classes are never included.
    Returns empty dict if deps (torch, sklearn) are not installed.
    """
    try:
        from ci.utils import retrieve_instances_from_file
        from seanergys_modelzoo.models.common.seanergys_model import SeanergysModel
    except ImportError:
        return {}

    classes: Dict[str, Type] = {}
    for py_file in (_PROJECT_ROOT / "seanergys_modelzoo" / "models" / "tasks").rglob(
        "*.py"
    ):
        if py_file.name == "__init__.py":
            continue
        try:
            # retrieve_instances_from_file INSTANTIATES each class it finds, so a
            # model whose constructor loads a dataset would download here — at
            # import of this conftest, before a single test runs. Budget it.
            with _time_budget(py_file.stem, "class discovery"):
                classes.update(retrieve_instances_from_file(py_file, SeanergysModel))
        except Exception:
            continue
    return classes


def _collect_dataset_classes() -> Dict[str, Type]:
    """
    Discover all concrete SeanergysDataset subclasses in datasets/.
    Excludes abstract base classes (SeanergysDataset, SeanergysParquetDataset).
    Returns empty dict if deps (torch) are not installed.
    """
    try:
        from ci.utils import retrieve_instances_from_file
        from seanergys_modelzoo.datasets.common.seanergys_dataset import (
            SeanergysDataset,
        )
    except ImportError:
        return {}

    _BASE_CLASSES = {"SeanergysDataset", "SeanergysParquetDataset"}
    classes: Dict[str, Type] = {}

    for py_file in (_PROJECT_ROOT / "seanergys_modelzoo" / "datasets").rglob("*.py"):
        if py_file.name == "__init__.py":
            continue
        try:
            objs = retrieve_instances_from_file(py_file, SeanergysDataset)
            classes.update(
                {k: v for k, v in objs.items() if v.__name__ not in _BASE_CLASSES}
            )
        except Exception:
            continue
    return classes


# Collected once at session start. Used to parametrize tests.
# Any model/dataset added to the repo is automatically included.
MODEL_CLASSES = _collect_model_classes()
DATASET_CLASSES = _collect_dataset_classes()


def _collect_model_dataset_pairs() -> list:
    """
    Build (model_cls, dataset_id) pairs by reading each model's supported_datasets.
    Uses a lightweight instantiation (no heavy hyperparams) — same pattern as
    test_model_instantiation in the smoke suite.

    Each pair becomes one parametrized test case in integration tests, so a failure
    is reported as e.g. test_full_pipeline[MACK-f-data] instead of test_full_pipeline[MACK].
    """
    pairs = []
    for model_cls in MODEL_CLASSES.values():
        try:
            instance = model_cls(model_hyperparameters={"n_jobs": 1})
            for dataset_cls in instance.supported_datasets or []:
                dataset_id = dataset_cls.DATASET_ID
                if dataset_id:
                    pairs.append((model_cls, dataset_id))
        except Exception:
            continue
    return pairs


MODEL_DATASET_PAIRS = _collect_model_dataset_pairs()


# ---------------------------------------------------------------------------
# Shared discovery helpers
#
# These live here, once, so the unit, smoke and integration layers all read the
# SAME contract. They deliberately hold no knowledge of any concrete model or
# dataset: everything is derived from what a model author already has to
# implement (get_train_config / supported_datasets) plus ci/policy.yml.
# ---------------------------------------------------------------------------

_POLICY_PATH = _PROJECT_ROOT / "ci" / "policy.yml"


def load_policy() -> Dict:
    """Read ci/policy.yml. Returns {} if absent or unparseable."""
    if not _POLICY_PATH.exists():
        return {}
    try:
        import yaml

        return yaml.safe_load(_POLICY_PATH.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def load_dataset_scope() -> Dict[str, Dict]:
    """
    ci/policy.yml -> dataset_scope: { "DatasetClassName": {...} }.

    Records a discovered dataset that is deliberately NOT exercised by the
    discovery-driven tests, with the reason. Anything discovered and neither
    reachable from a model nor listed here is a coverage hole, and
    tests/unit/test_discovery_contract.py fails on it.
    """
    return {
        str(k): (v or {}) for k, v in (load_policy().get("dataset_scope") or {}).items()
    }


def fixture_name_for(dataset_cls) -> str:
    """
    The one definition of the sample-fixture naming rule.

    FDataDataset -> sample_fdata.parquet, PM100Dataset -> sample_pm100.parquet.
    tests/smoke, tests/integration and scripts/create_sample_data.py all import
    this instead of re-deriving it, so the rule cannot drift.
    """
    name = dataset_cls.__name__.lower().replace("dataset", "")
    return f"sample_{name}.parquet"


def collect_model_dataset_configs() -> Dict:
    """
    Every (model, dataset) training configuration the repo can offer, derived
    only from the model-author contract.

    For each discovered model, ask it about EVERY dataset in its
    supported_datasets (not just its default), then call
    get_train_config(dataset_name=<DATASET_ID>).

    Returns {"<Model>-<dataset_id>": (model_cls, dataset_cls, ds_params, dl_params)}.

    Safe at collection time: get_train_config() only assembles a model and a
    params object — it does not instantiate the dataset and does not touch the
    network. (Measured with outbound sockets blocked: ~0s for every model.)
    """
    configs: Dict = {}
    for model_name, model_cls in MODEL_CLASSES.items():
        try:
            with _time_budget(model_name, "instantiation"):
                instance = model_cls(model_hyperparameters={"n_jobs": 1})
            dataset_ids = [
                ds.DATASET_ID
                for ds in (instance.supported_datasets or [])
                if getattr(ds, "DATASET_ID", "")
            ]
        except Exception:
            continue

        for dataset_id in dataset_ids:
            try:
                with _time_budget(model_name, f"get_train_config({dataset_id})"):
                    config = model_cls.get_train_config(dataset_name=dataset_id)
            except Exception:
                continue
            for _entry, (_model, dataset_cls, (ds_params, dl_params)) in config.items():
                configs[f"{model_name}-{dataset_id}"] = (
                    model_cls,
                    dataset_cls,
                    ds_params,
                    dl_params,
                )
    return configs


MODEL_DATASET_CONFIGS = collect_model_dataset_configs()


def offline_params(ds_params, fixture_path) -> Dict:
    """
    Turn a model author's dataset params into offline params.

    Only the TRANSPORT is overridden — read this local file instead of the
    remote source. The features, filters and transforms stay exactly as the
    model author wrote them, which is what makes this a real test of their
    config rather than of something a test invented.
    """
    params = ds_params.model_dump()
    params["data_path"] = str(fixture_path)
    params["is_dummy"] = True
    for remote_flag in ("use_zenodo_url", "use_github_url"):
        if remote_flag in params:
            params[remote_flag] = False
    # File selection is meaningless for a single local fixture and would make
    # the reader look for months that the fixture does not contain.
    params.pop("files", None)
    return params


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def project_root() -> Path:
    return _PROJECT_ROOT


@pytest.fixture(scope="session")
def ci_max_samples() -> int:
    """Max samples for CI runs. Override via env CI_MAX_SAMPLES."""
    return int(os.environ.get("CI_MAX_SAMPLES", "50"))


@pytest.fixture(scope="session")
def sample_data_dir(project_root) -> Path:
    """Path to tests/fixtures/sample_data/. Create with `make sample-data`."""
    d = project_root / "tests" / "fixtures" / "sample_data"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture(scope="session")
def fixture_provisioning():
    """
    The session's fixture-provisioning report (see tests/fixture_provisioning.py).

    Lets a test say WHY a fixture is absent — downloading disabled, or a failed
    download — instead of reporting a bare "file not found".
    """
    return FIXTURE_REPORT
