"""Unit tests for concrete parquet datasets (PM100, F-DATA) using tiny local
parquet fixtures written to tmp_path — no Zenodo access required.

These also exercise the inherited base helpers (__len__, get_feature_statistics,
__repr__, get_stats, get_metadata).
"""

import numpy as np
import pytest

import torch  # noqa: F401
import pyarrow  # noqa: F401
import pandas as pd

from seanergys_modelzoo.datasets.f_data import FDataDataset  # noqa: E402
from seanergys_modelzoo.datasets.pm100 import PM100Dataset  # noqa: E402


def _write_parquet(path, rows=5):
    df = pd.DataFrame(
        {
            "n_nodes": np.arange(1, rows + 1, dtype="int64"),
            "duration": np.linspace(10.0, 50.0, rows),
            "power_consumption": np.linspace(100.0, 200.0, rows),
            "partition": ["a", "b"] * (rows // 2) + ["a"] * (rows % 2),
        }
    )
    df.to_parquet(path)
    return df


@pytest.fixture
def parquet_file(tmp_path):
    p = tmp_path / "sample.parquet"
    _write_parquet(p)
    return p


# ---------------------------------------------------------------------------
# PM100Dataset
# ---------------------------------------------------------------------------


def _pm100(path):
    return PM100Dataset(
        data_path=str(path),
        input_features=["n_nodes", "duration"],
        output_features=["power_consumption"],
    )


def test_pm100_loads_and_len(parquet_file):
    ds = _pm100(parquet_file)
    assert len(ds) == 5
    assert ds.stats["n_samples"] == 5


def test_pm100_getitem(parquet_file):
    ds = _pm100(parquet_file)
    x, y = ds[0]
    assert len(x) == 2
    assert len(y) == 1


def test_pm100_getitem_tensor(parquet_file):

    ds = _pm100(parquet_file)
    x, y = ds.__getitem__(0, return_tensor=True, dtype=torch.float32)
    assert isinstance(x, torch.Tensor)


def test_pm100_getitem_out_of_range(parquet_file):
    ds = _pm100(parquet_file)
    with pytest.raises(IndexError):
        ds[999]


def test_pm100_missing_path_raises():
    with pytest.raises(ValueError, match="data_path"):
        PM100Dataset(
            input_features=["n_nodes"],
            output_features=["power_consumption"],
            use_zenodo_url=False,
        )


def test_pm100_metadata_enriched(parquet_file):
    ds = _pm100(parquet_file)
    assert ds.metadata["dataset_name"] == "PM100"


def test_pm100_download_path(tmp_path, parquet_file):
    out = tmp_path / "out"
    ds = PM100Dataset(
        data_path=str(parquet_file),
        input_features=["n_nodes"],
        output_features=["power_consumption"],
        download_path=str(out),
    )
    assert (out / "job_table.parquet").exists()
    assert len(ds) == 5


# ---------------------------------------------------------------------------
# Inherited base helpers (via PM100)
# ---------------------------------------------------------------------------


def test_feature_statistics(parquet_file):
    ds = _pm100(parquet_file)
    stats = ds.get_feature_statistics()
    assert set(stats.keys()) == {"n_nodes", "duration", "power_consumption"}
    assert set(stats["n_nodes"].keys()) == {"mean", "std", "min", "max", "median"}


def test_repr(parquet_file):
    ds = _pm100(parquet_file)
    text = repr(ds)
    assert "PM100Dataset" in text
    assert "n_samples=5" in text


def test_get_stats_and_metadata(parquet_file):
    ds = _pm100(parquet_file)
    assert ds.get_stats()["n_samples"] == 5
    meta = ds.get_metadata()
    assert "stats" in meta


# ---------------------------------------------------------------------------
# FDataDataset (local file branch)
# ---------------------------------------------------------------------------


def _fdata(path):
    return FDataDataset(
        data_path=str(path),
        input_features=["n_nodes", "duration"],
        output_features=["power_consumption"],
    )


def test_fdata_loads_local_file(parquet_file):
    ds = _fdata(parquet_file)
    assert len(ds) == 5


def test_fdata_getitem(parquet_file):
    ds = _fdata(parquet_file)
    x, y = ds[0]
    assert len(x) == 2 and len(y) == 1


def test_fdata_getitem_out_of_range(parquet_file):
    ds = _fdata(parquet_file)
    with pytest.raises(IndexError):
        ds[999]


def test_fdata_loads_local_directory(tmp_path):
    d = tmp_path / "fdata_dir"
    d.mkdir()
    _write_parquet(d / "21_03.parquet")
    _write_parquet(d / "21_04.parquet")
    ds = FDataDataset(
        data_path=str(d),
        input_features=["n_nodes", "duration"],
        output_features=["power_consumption"],
    )
    assert len(ds) == 10


def test_fdata_empty_directory_raises(tmp_path):
    d = tmp_path / "empty_dir"
    d.mkdir()
    with pytest.raises(Exception):
        FDataDataset(
            data_path=str(d),
            input_features=["n_nodes"],
            output_features=["power_consumption"],
        )


def test_fdata_missing_path_raises():
    with pytest.raises(ValueError, match="data_path"):
        FDataDataset(
            input_features=["n_nodes"],
            output_features=["power_consumption"],
            use_zenodo_url=False,
        )


def test_fdata_statistics_by_partition(parquet_file):
    ds = FDataDataset(
        data_path=str(parquet_file),
        input_features=["n_nodes", "duration"],
        output_features=["power_consumption"],
        columns=["n_nodes", "duration", "power_consumption", "partition"],
    )
    stats = ds.get_statistics_by_partition()
    assert "a" in stats and "b" in stats
    assert "n_jobs" in stats["a"]
    assert "avg_duration" in stats["a"]


def test_fdata_statistics_by_partition_missing_column(tmp_path):
    p = tmp_path / "nopartition.parquet"
    pd.DataFrame({"n_nodes": [1, 2], "power_consumption": [1.0, 2.0]}).to_parquet(p)
    ds = FDataDataset(
        data_path=str(p),
        input_features=["n_nodes"],
        output_features=["power_consumption"],
    )
    with pytest.raises(ValueError, match="partition"):
        ds.get_statistics_by_partition()
