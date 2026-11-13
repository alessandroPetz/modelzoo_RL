"""
The shared parquet-dataset contract, run against every discovered parquet
dataset (GitLab issue #4).

`test_parquet_datasets.py` used to write one synthetic frame and then run the
same contract by hand for each dataset — the helpers were the same function with
a different class. Worse, the inherited base helpers (get_feature_statistics,
__repr__, get_stats, get_metadata) were exercised through ONE subclass only, so a
base-class regression that surfaced in another subclass was invisible.

Here the contract is written once and parametrized over every discovered
SeanergysParquetDataset subclass. The fixture frame is built from the columns the
MODEL's own config asks for, so no column name is invented by the test — the old
synthetic frame used `n_nodes` / `power_consumption`, which exist in neither
F-DATA nor PM100.

No concrete dataset name appears here; the guard in test_no_hardcoded_names.py
enforces that.
"""

import numpy as np
import pandas as pd
import pytest

import torch  # noqa: F401
import pyarrow  # noqa: F401

from seanergys_modelzoo.datasets.common.seanergys_parquet_dataset import (
    SeanergysParquetDataset,
)
from tests.conftest import MODEL_DATASET_CONFIGS


def _parquet_cases():
    """
    One case per (model, dataset) config whose dataset is a parquet dataset.

    Driving off the model config rather than off DATASET_CLASSES is what gives
    us real column names: the config states the input/output features, so the
    synthetic frame matches the schema the dataset will actually project.
    """
    cases = []
    for key, (_model, dataset_cls, ds_params, _dl) in sorted(
        MODEL_DATASET_CONFIGS.items()
    ):
        if not issubclass(dataset_cls, SeanergysParquetDataset):
            continue
        # This suite builds its OWN frame from the columns the config declares,
        # so it checks the base-class contract rather than the real fixture: a
        # pair that cannot load its committed sample can still satisfy it. The
        # fixture-backed check lives in test_discovery_contract.py.
        cases.append(pytest.param(key, id=key))
    return cases


_CASES = _parquet_cases()

pytestmark = pytest.mark.skipif(
    not _CASES, reason="no parquet (model, dataset) configs discovered"
)


def _synthetic_frame(ds_params, rows: int = 6) -> pd.DataFrame:
    """
    Build a frame carrying exactly the columns this config declares. Numeric, so
    every dataset's tensor conversion has something valid to work with.
    """
    columns = list(ds_params.input_features) + list(ds_params.output_features)
    return pd.DataFrame(
        {c: np.linspace(1.0, float(rows), rows) for c in dict.fromkeys(columns)}
    )


def _build(config_key, tmp_path):
    _model, dataset_cls, ds_params, _dl = MODEL_DATASET_CONFIGS[config_key]
    path = tmp_path / "sample.parquet"
    _synthetic_frame(ds_params).to_parquet(path)

    params = ds_params.model_dump()
    params.update(data_path=str(path), is_dummy=False)
    for remote in ("use_zenodo_url", "use_github_url"):
        params.pop(remote, None)
    # A transform expects a fitted model that does not exist at unit-test time,
    # and filters reference columns outside this minimal frame.
    params.update(transform=None, target_transform=None, filters=None, columns=None)
    params.pop("preprocessing_functions", None)
    params.pop("files", None)
    return dataset_cls(**params), ds_params


@pytest.mark.parametrize("config_key", _CASES)
def test_dataset_loads_and_len_matches_stats(config_key, tmp_path):
    """
    PASS: the dataset loads the frame and len() agrees with its own stats.
    FAIL: the two disagree, so downstream batching would be wrong.
    """
    dataset, _ = _build(config_key, tmp_path)
    assert len(dataset) == 6
    assert dataset.stats["n_samples"] == len(dataset)


@pytest.mark.parametrize("config_key", _CASES)
def test_getitem_returns_input_and_output(config_key, tmp_path):
    """
    PASS: dataset[0] yields one value per declared input and output feature.
    FAIL: the sample shape does not match what the config declared.
    """
    dataset, ds_params = _build(config_key, tmp_path)
    x, y = dataset[0]
    assert len(x) == len(ds_params.input_features)
    assert len(y) == len(ds_params.output_features)


@pytest.mark.parametrize("config_key", _CASES)
def test_getitem_out_of_range_raises(config_key, tmp_path):
    """
    PASS: an index past the end raises IndexError.
    FAIL: it returns data or a confusing error, hiding a bad sampler.
    """
    dataset, _ = _build(config_key, tmp_path)
    with pytest.raises(IndexError):
        dataset[999]


@pytest.mark.parametrize("config_key", _CASES)
def test_inherited_base_helpers_work(config_key, tmp_path):
    """
    The base-class contract, through EVERY subclass rather than just one — the
    gap that let a regression hide in whichever subclass nobody named.

    PASS: get_feature_statistics, get_stats, get_metadata and __repr__ all work.
    FAIL: a base helper breaks for this subclass.
    """
    dataset, ds_params = _build(config_key, tmp_path)

    stats = dataset.get_feature_statistics()
    for feature in list(ds_params.input_features) + list(ds_params.output_features):
        assert feature in stats, f"get_feature_statistics() omitted {feature!r}"
        assert set(stats[feature]) >= {"mean", "std", "min", "max"}

    assert isinstance(dataset.get_stats(), dict)
    assert "stats" in dataset.get_metadata()
    assert dataset.__class__.__name__ in repr(dataset)


@pytest.mark.parametrize("config_key", _CASES)
def test_missing_data_path_raises(config_key, tmp_path):
    """
    PASS: constructing with no source raises instead of yielding an empty dataset.
    FAIL: a misconfigured run trains on nothing and looks fine.
    """
    _model, dataset_cls, ds_params, _dl = MODEL_DATASET_CONFIGS[config_key]
    params = ds_params.model_dump()
    params.update(data_path=None, transform=None, target_transform=None, filters=None)
    for remote in ("use_zenodo_url", "use_github_url"):
        params[remote] = False
    with pytest.raises(Exception):
        dataset_cls(**params)
