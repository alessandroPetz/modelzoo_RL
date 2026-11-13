"""
Smoke tests: model discovery, instantiation, and train-config structure.

All parametrized tests are driven by MODEL_CLASSES collected at session start
in conftest.py — no model names are hardcoded here. Adding a new model to
models/tasks/ automatically includes it in every test below.

Sections
--------
1. Imports     — core utilities can be imported.
2. Discovery   — at least one model class found.
3. Contract    — each model instantiates and exposes the required interface.
4. Train config — get_train_config() returns a valid structure (no data loaded).
"""

import pytest

import torch  # noqa: F401
import sklearn  # noqa: F401

from tests.conftest import MODEL_CLASSES  # noqa: E402

_ABSTRACT = {"SeanergysModel", "SeanergysSklearnModel"}
_CONCRETE = [(k, v) for k, v in MODEL_CLASSES.items() if v.__name__ not in _ABSTRACT]
_CONCRETE_IDS = [k for k, v in MODEL_CLASSES.items() if v.__name__ not in _ABSTRACT]

# No xfail wiring (removed 2026-09-13). A model that cannot be built or used is a
# FAILING test, not an "expected failure" — the breakage belongs to its author and
# must stay red until they fix it. The engineering detail lives in ci/policy.yml
# -> core_change_requests, which names the owner and the evidence.
_CONCRETE_CASES = [pytest.param(name, cls, id=name) for name, cls in _CONCRETE]


# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------


def test_logger_imports():
    """SeanergysLogger can be imported and instantiated."""
    from seanergys_modelzoo.logger.seanergys_logger import SeanergysLogger

    assert SeanergysLogger() is not None


def test_ci_utils_imports():
    """ci.utils discovery helpers can be imported."""
    from ci.utils import import_module_from_file, retrieve_instances_from_file

    assert callable(import_module_from_file)
    assert callable(retrieve_instances_from_file)


# ---------------------------------------------------------------------------
# 2. Discovery
# ---------------------------------------------------------------------------


def test_model_discovery():
    """
    PASS: at least one SeanergysModel subclass found in models/tasks/.
    FAIL: no model classes found — check your model file location and base class.
    """
    assert len(MODEL_CLASSES) >= 1, (
        "No SeanergysModel subclasses found in seanergys_modelzoo/models/tasks/.\n"
        "Make sure your model file:\n"
        "  - lives in seanergys_modelzoo/models/tasks/<task>/<name>/\n"
        "  - subclasses SeanergysModel\n"
        "  - is not named __init__.py"
    )


# ---------------------------------------------------------------------------
# 3. Contract: instantiation + required interface
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_name,model_cls", _CONCRETE_CASES)
def test_model_instantiation(model_name, model_cls):
    """
    Each concrete model instantiates with minimal hyperparams and exposes
    train / predict / save / load.

    PASS: model builds without error, model_name set, required methods present.
    FAIL: constructor raises, or a required method is missing.
    """
    model = model_cls(model_hyperparameters={"n_estimators": 2, "n_jobs": 1})
    assert model is not None
    assert hasattr(model, "model_name"), f"{model_name}: model_name attribute missing"
    for method in ("train", "predict", "save", "load"):
        assert hasattr(model, method), (
            f"{model_name}: missing required method '{method}'"
        )


# ---------------------------------------------------------------------------
# 4. Train config: structure check (no data loading)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_name,model_cls", _CONCRETE_CASES)
def test_model_get_train_config(model_name, model_cls):
    """
    get_train_config() returns a valid dict without loading any data.

    Expected shape per entry:
      { entry_name: (model_instance, DatasetClass, (SeanergysDatasetParams, SeanergysDataloaderParams)) }

    PASS: config is a non-empty dict with the correct tuple structure per entry.
    SKIP: get_train_config() returns None (model author still needs to implement it).
    FAIL: config is wrong type, empty, or any entry has wrong shape/types.
    """
    from seanergys_modelzoo.datasets.common.seanergys_dataset import SeanergysDataset
    from seanergys_modelzoo.models.common.seanergys_configurator import (
        SeanergysDataloaderParams,
        SeanergysDatasetParams,
    )

    config = model_cls.get_train_config(get_dummy_data=True)

    if config is None:
        pytest.skip(
            f"SKIP [{model_name}] get_train_config() returned None.\n"
            f"  → Implement get_train_config() in {model_name} to enable this test."
        )

    assert isinstance(config, dict), (
        f"FAIL [{model_name}] get_train_config() must return a dict, got {type(config)}"
    )
    assert len(config) > 0, (
        f"FAIL [{model_name}] get_train_config() returned an empty dict"
    )

    for entry_name, entry in config.items():
        assert len(entry) == 3, (
            f"FAIL [{model_name}/{entry_name}] entry must be "
            f"(model_instance, DatasetClass, (ds_params, dl_params)), got len={len(entry)}"
        )
        model_instance, dataset_cls, params = entry
        assert len(params) == 2, (
            f"FAIL [{model_name}/{entry_name}] third element must be "
            f"(SeanergysDatasetParams, SeanergysDataloaderParams)"
        )
        ds_params, dl_params = params

        assert issubclass(dataset_cls, SeanergysDataset), (
            f"FAIL [{model_name}/{entry_name}] DatasetClass must subclass SeanergysDataset, "
            f"got {dataset_cls}"
        )
        assert isinstance(ds_params, SeanergysDatasetParams), (
            f"FAIL [{model_name}/{entry_name}] ds_params must be SeanergysDatasetParams, "
            f"got {type(ds_params)}"
        )
        assert isinstance(dl_params, SeanergysDataloaderParams), (
            f"FAIL [{model_name}/{entry_name}] dl_params must be SeanergysDataloaderParams, "
            f"got {type(dl_params)}"
        )
        for method in ("train", "predict", "save", "load"):
            assert hasattr(model_instance, method), (
                f"FAIL [{model_name}/{entry_name}] model instance missing method '{method}'"
            )
