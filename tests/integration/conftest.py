"""
Fixtures and helpers for the integration pipeline tests.

The integration suite runs the full lifecycle (config → load → train → predict →
save → load) for every discovered (model, dataset) pair, offline, against the
committed sample fixtures in tests/fixtures/sample_data/.

Nothing here hardcodes a model or dataset name. Pairs come from each model's
supported_datasets (see tests/conftest.MODEL_DATASET_PAIRS), and every pair is
expected to work: there is no xfail list and no tolerated failure. A pair that is
broken pending an author fix is recorded in ci/policy.yml -> core_change_requests
and stays RED until the fix lands.
"""

from __future__ import annotations

import contextlib
import traceback
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _integration_data_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "fixtures" / "sample_data"


@pytest.fixture(scope="session")
def integration_data_dir() -> Path:
    """Path to tests/fixtures/sample_data. Run `make sample-data` if a fixture is missing."""
    d = _integration_data_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def temp_model_dir(tmp_path) -> Path:
    """Temporary directory for saving models during a test."""
    return tmp_path


# ---------------------------------------------------------------------------
# Fixture path convention (mirrors scripts/create_sample_data.py)
# ---------------------------------------------------------------------------


def fixture_path_for(dataset_cls, data_dir: Path) -> Path:
    """
    FDataDataset → sample_fdata.parquet, PM100Dataset → sample_pm100.parquet.

    Delegates to tests.conftest.fixture_name_for so the rule has exactly one
    definition; it used to be written out separately here, in
    tests/smoke/test_smoke_data.py and in scripts/create_sample_data.py.
    """
    from tests.conftest import fixture_name_for

    return data_dir / fixture_name_for(dataset_cls)


# ---------------------------------------------------------------------------
# Precise stage reporting
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def run_stage(stage: str, model_name: str, dataset_id: str):
    """
    Wrap one pipeline stage. On any exception, re-raise as an AssertionError whose
    message names the exact stage, model, and dataset, and includes the original
    traceback (so the failing file:line is right there in the pytest report).
    """
    try:
        yield
    except Exception as e:  # noqa: BLE001 — we deliberately annotate and re-raise
        tb = traceback.format_exc()
        raise AssertionError(
            "\n"
            + "=" * 72
            + "\nPIPELINE FAILURE"
            + f"\n  stage   : {stage}"
            + f"\n  model   : {model_name}"
            + f"\n  dataset : {dataset_id}"
            + f"\n  error   : {type(e).__name__}: {e}\n"
            + "-" * 72
            + f"\n{tb}"
            + "=" * 72
        ) from e


# ---------------------------------------------------------------------------
# Offline pair loading from committed fixtures
# ---------------------------------------------------------------------------


def load_pair_from_fixture(model_cls, dataset_id: str, data_dir: Path):
    """
    Build (model_instance, dataset, loader) for one (model, dataset_id) pair using
    get_train_config(dataset_id) and the committed fixture for that dataset.

    Raises (never skips) so the caller can turn problems into precise failures:
      - FileNotFoundError  if the fixture parquet is not committed
      - ValueError         if the fixture loads 0 rows
    """
    from seanergys_modelzoo.dataloader.seanergys_dataloader import SeanergysDataloader

    config = model_cls.get_train_config(dataset_id)
    if dataset_id not in config:
        raise KeyError(
            f"{model_cls.__name__}.get_train_config({dataset_id!r}) returned no entry "
            f"for {dataset_id!r} (keys: {list(config)})"
        )

    model_instance, dataset_cls, (ds_params, dl_params) = config[dataset_id]

    fixture = fixture_path_for(dataset_cls, data_dir)
    if not fixture.exists():
        raise FileNotFoundError(
            f"missing fixture {fixture.name} for {dataset_cls.__name__}. "
            f"Run `make sample-data` and commit "
            f"tests/fixtures/sample_data/{fixture.name}."
        )

    # Point the dataset at the local fixture and drop the date/range filters, which
    # are sized for the full Zenodo dataset and would zero-out a small sample.
    # Datasets are constructed from their params (to_dict preserves the transform
    # callables) and load their data in model_post_init — no explicit load_data().
    local_params = ds_params.model_copy(update={"data_path": fixture, "filters": None})
    dataset = dataset_cls(**local_params.to_dict())

    if len(dataset) == 0:
        raise ValueError(
            f"fixture {fixture.name} loaded 0 rows for {dataset_cls.__name__}; "
            f"regenerate it with `make sample-data`."
        )

    loader = SeanergysDataloader.from_config(
        data_loader_config=dl_params, dataset=dataset
    )
    return model_instance, dataset, loader
