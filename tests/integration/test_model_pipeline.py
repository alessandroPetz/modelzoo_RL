"""
Integration tests: the full pipeline for every (model, dataset) pair.

For each pair discovered from supported_datasets (tests/conftest.MODEL_DATASET_PAIRS)
this exercises the complete lifecycle and checks as many aspects as possible:

    config → load → dataloader → train → predict → save → reload → compare → metadata

Two variants run the same stages:
  * test_full_pipeline_offline — uses the committed sample fixtures (no network).
    This is the one that runs in CI. It FAILS (not skips) on any real error, and the
    failure names the exact stage + model + dataset (see run_stage).
  * test_full_pipeline_zenodo  — @pytest.mark.zenodo; uses is_dummy=True against real
    Zenodo downloads. Skipped with CI_SKIP_ZENODO=1.

Nothing here hardcodes a model or dataset name. Adding a model and declaring its
supported_datasets automatically adds its pairs. There is no xfail list: every pair
must complete the lifecycle, and one that cannot is a failing test. A break pending
an author fix is recorded in ci/policy.yml -> core_change_requests, which names the
owner and the evidence, and stays red until they land it.
"""

from __future__ import annotations

import os

import pytest

from tests.conftest import MODEL_DATASET_PAIRS
from tests.integration.conftest import (
    load_pair_from_fixture,
    run_stage,
)


def _plain_params():
    """One case per discovered (model, dataset) pair. No xfail — see below."""
    return [pytest.param(m, d, id=f"{m.__name__}-{d}") for m, d in MODEL_DATASET_PAIRS]


# Every pair runs plainly (xfail wiring removed 2026-09-13). A pair that cannot
# complete the lifecycle is a FAILING test: this is the end-to-end gate, and a
# green gate over a model that does not run is worse than no gate. The reason a
# given pair is broken, its owner and the evidence live in ci/policy.yml ->
# core_change_requests, and are published in docs/ci-cd.md.
_PLAIN_PARAMS = _plain_params()
_PIPELINE_PARAMS = _PLAIN_PARAMS


# ---------------------------------------------------------------------------
# Structural contract (fast, no data)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_cls,dataset_id", _PLAIN_PARAMS)
def test_get_train_config_structure(model_cls, dataset_id):
    """get_train_config(dataset_id) returns the documented structure and interface."""
    from seanergys_modelzoo.datasets.common.seanergys_dataset import SeanergysDataset
    from seanergys_modelzoo.models.common.seanergys_configurator import (
        SeanergysDataloaderParams,
        SeanergysDatasetParams,
    )

    name = model_cls.__name__
    with run_stage("get_train_config", name, dataset_id):
        config = model_cls.get_train_config(dataset_id)
        assert isinstance(config, dict), "get_train_config must return a dict"
        assert dataset_id in config, (
            f"no entry for {dataset_id!r} (keys: {list(config)})"
        )

        model_instance, dataset_cls, params = config[dataset_id]
        assert len(params) == 2, (
            "params must be (SeanergysDatasetParams, SeanergysDataloaderParams)"
        )
        ds_params, dl_params = params

        for method in ("train", "predict", "save", "load"):
            assert hasattr(model_instance, method), f"model missing method: {method}"
        assert issubclass(dataset_cls, SeanergysDataset), (
            "DatasetClass must subclass SeanergysDataset"
        )
        assert isinstance(ds_params, SeanergysDatasetParams)
        assert isinstance(dl_params, SeanergysDataloaderParams)


# ---------------------------------------------------------------------------
# Full offline pipeline (runs in CI, fails loudly, checks many aspects)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_cls,dataset_id", _PIPELINE_PARAMS)
def test_full_pipeline_offline(model_cls, dataset_id, integration_data_dir, tmp_path):
    """Complete lifecycle for one pair against committed fixtures. Fails on any error."""
    import numpy as np

    name = model_cls.__name__

    # --- load (fixture-backed) -------------------------------------------------
    with run_stage("load-fixture", name, dataset_id):
        model_instance, dataset, loader = load_pair_from_fixture(
            model_cls, dataset_id, integration_data_dir
        )

    # --- dataset contract ------------------------------------------------------
    with run_stage("dataset-contract", name, dataset_id):
        assert len(dataset) > 0, "dataset is empty"
        item = dataset[0]
        assert isinstance(item, tuple) and len(item) == 2, "dataset[0] must be (x, y)"

    # --- dataloader yields a batch --------------------------------------------
    with run_stage("dataloader", name, dataset_id):
        batch = next(iter(loader))
        assert batch is not None, "dataloader yielded nothing"

    # --- train -----------------------------------------------------------------
    with run_stage("train", name, dataset_id):
        history = model_instance.train(loader)
        assert model_instance.is_trained, "is_trained must be True after train()"
        assert history is not None, "train() must return a history object"

    # --- predict ---------------------------------------------------------------
    with run_stage("predict", name, dataset_id):
        predictions = model_instance.predict(loader)
        assert predictions is not None, "predict() returned None"
        assert len(predictions) == len(dataset), (
            f"len(predictions)={len(predictions)} != len(dataset)={len(dataset)}"
        )
        arr = np.asarray(predictions)
        if np.issubdtype(arr.dtype, np.number):
            assert np.isfinite(arr).all(), "predictions contain NaN/inf"

    # --- save ------------------------------------------------------------------
    save_path = tmp_path / name / dataset_id
    save_path.mkdir(parents=True)
    model_file = save_path / "model"
    with run_stage("save", name, dataset_id):
        saved = model_instance.save(str(model_file))
        assert saved, "save() must return True"
        # At least one artifact must exist on disk (model file or companions).
        produced = list(save_path.glob("model*"))
        assert produced, f"save() wrote no files under {save_path}"

    # --- reload + compare ------------------------------------------------------
    with run_stage("load", name, dataset_id):
        loaded = type(model_instance).load(str(model_file))
        assert loaded is not None, "load() returned None"
        assert loaded.is_trained, "reloaded model must be trained"

    with run_stage("predict-after-reload", name, dataset_id):
        pred_after = np.asarray(loaded.predict(loader))
        pred_before = np.asarray(predictions)
        assert len(pred_before) == len(pred_after), (
            "prediction length changed after reload"
        )
        if np.issubdtype(pred_before.dtype, np.number) and np.issubdtype(
            pred_after.dtype, np.number
        ):
            assert np.allclose(pred_before, pred_after, equal_nan=True), (
                "predictions changed after save/reload"
            )
        else:
            assert list(pred_before) == list(pred_after), (
                "predictions changed after save/reload"
            )

    # --- metadata --------------------------------------------------------------
    with run_stage("metadata", name, dataset_id):
        assert getattr(model_instance, "metadata", None) is not None, "metadata missing"
        assert getattr(model_instance.metadata, "name", None), "metadata.name not set"


# ---------------------------------------------------------------------------
# Zenodo variant (real is_dummy download; skipped with CI_SKIP_ZENODO=1)
# ---------------------------------------------------------------------------


@pytest.mark.zenodo
@pytest.mark.parametrize("model_cls,dataset_id", _PIPELINE_PARAMS)
def test_full_pipeline_zenodo(model_cls, dataset_id, tmp_path):
    """Same lifecycle against a real is_dummy=True dataset from Zenodo."""
    if os.environ.get("CI_SKIP_ZENODO") == "1":
        pytest.skip("CI_SKIP_ZENODO=1 — Zenodo tests disabled for this run.")

    from seanergys_modelzoo.dataloader.seanergys_dataloader import SeanergysDataloader

    name = model_cls.__name__
    config = model_cls.get_train_config(dataset_id)
    model_instance, dataset_cls, (ds_params, dl_params) = config[dataset_id]
    ds_params.is_dummy = True

    with run_stage("load-zenodo", name, dataset_id):
        dataset = dataset_cls(**ds_params.to_dict())

    if len(dataset) == 0:
        pytest.skip(
            f"{name}/{dataset_id}: is_dummy=True returned 0 rows — "
            f"implement dummy support in {dataset_cls.__name__}."
        )

    loader = SeanergysDataloader.from_config(
        data_loader_config=dl_params, dataset=dataset
    )

    with run_stage("train", name, dataset_id):
        model_instance.train(loader)
        assert model_instance.is_trained

    with run_stage("predict", name, dataset_id):
        predictions = model_instance.predict(loader)
        assert predictions is not None and len(predictions) > 0

    save_path = tmp_path / name / dataset_id
    save_path.mkdir(parents=True)
    with run_stage("save-load", name, dataset_id):
        assert model_instance.save(str(save_path / "model"))
        reloaded = type(model_instance).load(str(save_path / "model"))
        assert reloaded is not None and reloaded.is_trained
