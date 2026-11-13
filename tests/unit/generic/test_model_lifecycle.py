"""
Model lifecycle, once, for every discovered model (GitLab issue #3).

The same six checks used to be hand-written three times over, once per model
name, so a fourth model got none of them. Here each is written once and
parametrized over MODEL_CLASSES, which means a model added to the repo is
covered with no edit to this file — and a model removed takes its cases with it.

Nothing in this module names a concrete model. That is enforced by
tests/unit/generic/test_no_hardcoded_names.py.

Capability, not identity, decides what runs: a check that needs an optional
method skips when the model does not implement it, rather than listing the
models that do.
"""

import sys

import pytest

import torch  # noqa: F401

from seanergys_modelzoo.models.common.seanergys_model import SeanergysModelTask
from tests.conftest import MODEL_CLASSES


def _models():
    """
    One parametrized case per discovered model.

    No xfail wiring (removed 2026-09-13): a model that cannot satisfy the generic
    lifecycle contract FAILS. Marking it "expected to fail" kept the pipeline
    green while shipped code was broken, which is the one signal this suite must
    never send. The owner and the evidence live in ci/policy.yml ->
    core_change_requests.
    """
    return [pytest.param(cls, id=name) for name, cls in MODEL_CLASSES.items()]


_MODEL_CASES = _models()

_MODEL_CASES_NO_BUILD = list(MODEL_CLASSES.values())
_MODEL_IDS_NO_BUILD = list(MODEL_CLASSES.keys())

pytestmark = pytest.mark.skipif(not MODEL_CLASSES, reason="no models discovered")


def _build(model_cls):
    """Instantiate a model the cheap way, as the smoke suite does."""
    return model_cls(model_hyperparameters={"n_jobs": 1})


@pytest.mark.parametrize("model_cls", _MODEL_CASES)
def test_model_declares_a_task_type(model_cls):
    """
    PASS: the model exposes a task_type from the shared enum, so the capability
          matrix in docs/ci-cd.md can classify it.
    FAIL: task_type is missing or not a SeanergysModelTask.
    """
    model = _build(model_cls)
    assert isinstance(model.task_type, SeanergysModelTask), (
        f"FAIL [{model_cls.__name__}] task_type is {model.task_type!r}; "
        f"expected a SeanergysModelTask member."
    )


@pytest.mark.parametrize("model_cls", _MODEL_CASES)
def test_model_declares_supported_datasets(model_cls):
    """
    PASS: the model lists at least one dataset class it supports.
    FAIL: it lists none, so it can never be paired with data and contributes no
          integration coverage.
    """
    model = _build(model_cls)
    assert model.supported_datasets, (
        f"FAIL [{model_cls.__name__}] declares no supported_datasets, so no "
        f"(model, dataset) pair can be built for it."
    )


@pytest.mark.parametrize("model_cls", _MODEL_CASES)
def test_save_load_roundtrip(model_cls, tmp_path):
    """
    PASS: save() writes the model and load() returns an instance of the same class.
    FAIL: the persistence contract is broken, so a trained model cannot be reused.
    """
    model = _build(model_cls)
    target = tmp_path / "model"

    assert model.save(target) is True, (
        f"FAIL [{model_cls.__name__}] save() returned False for {target}."
    )
    restored = model_cls.load(target)
    assert isinstance(restored, model_cls), (
        f"FAIL [{model_cls.__name__}] load() returned {type(restored).__name__}."
    )


@pytest.mark.parametrize("model_cls", _MODEL_CASES_NO_BUILD, ids=_MODEL_IDS_NO_BUILD)
def test_load_missing_path_raises(model_cls, tmp_path):
    """
    PASS: loading a path that does not exist raises rather than returning a
          silently empty model.
    FAIL: a typo'd path would yield an untrained model that looks valid.
    """
    with pytest.raises(Exception):
        model_cls.load(tmp_path / "does" / "not" / "exist")


@pytest.mark.parametrize("model_cls", _MODEL_CASES)
def test_save_returns_false_on_write_error(model_cls, tmp_path, monkeypatch):
    """
    PASS: when persistence raises, save() reports False instead of propagating.
    SKIP: the model's module does not use joblib, so there is nothing to patch.
    FAIL: a failed save looks like a successful one.

    Reaches the model's OWN module via sys.modules[model_cls.__module__] rather
    than importing a list of module paths by hand — that is what keeps this
    generic.
    """
    module = sys.modules.get(model_cls.__module__)
    if module is None or not hasattr(module, "joblib"):
        pytest.skip(f"{model_cls.__name__} does not persist via joblib")

    def _boom(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(module.joblib, "dump", _boom)
    model = _build(model_cls)
    assert model.save(tmp_path / "model") is False, (
        f"FAIL [{model_cls.__name__}] save() must return False when the write fails."
    )


@pytest.mark.parametrize("model_cls", _MODEL_CASES)
def test_is_trained_starts_false(model_cls):
    """
    PASS: a freshly built model reports is_trained False.
    FAIL: an untrained model claims to be trained, so predict() would run on an
          unfitted estimator.
    """
    model = _build(model_cls)
    assert model.is_trained is False, (
        f"FAIL [{model_cls.__name__}] is_trained is True before any training."
    )
