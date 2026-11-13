"""Additional unit tests targeting branches not reachable through the happy
paths: transform/tensor variants, preprocessing/download hooks, mocked network
loads (F-DATA Zenodo merge, M100 download+extract, model get_test_config), and
save error paths.
"""

import io
import tarfile

import numpy as np
import pytest

import torch  # noqa: F401
import pyarrow  # noqa: F401
import cachetools  # noqa: F401
import pandas as pd


from seanergys_modelzoo.datasets.common.seanergys_parquet_dataset import (  # noqa: E402
    SeanergysParquetDataset,
)
from seanergys_modelzoo.datasets.f_data import FDataDataset  # noqa: E402
from seanergys_modelzoo.datasets.pm100 import PM100Dataset  # noqa: E402
import seanergys_modelzoo.datasets.f_data as fdata_mod  # noqa: E402
import seanergys_modelzoo.datasets.m100 as m100_mod  # noqa: E402
from seanergys_modelzoo.datasets.m100 import M100Dataset  # noqa: E402
from tests.conftest import MODEL_DATASET_PAIRS  # noqa: E402

# ---------------------------------------------------------------------------
# Constants read from the CLASS, never retyped (GitLab issue #6).
# Changing M100Dataset's record map upstream requires no edit here.
# ---------------------------------------------------------------------------
FDATA_FILES = list(FDataDataset.model_fields["AVAILABLE_FILES"].default)
_RECORDS = list(M100Dataset.model_fields["AVAILABLE_FILES_PER_RECORD"].default)
AGG_RECORD = next(iter(_RECORDS[-1]))
_RAW = [
    (rid, f)
    for record in _RECORDS
    for rid, files in record.items()
    if rid != AGG_RECORD
    for f in files
]
# A representative raw (record_id, month) pair — the last raw month available.
RAW_RECORD, RAW_MONTH = _RAW[-1]

_PAIR_IDS = [f"{m.__name__}-{d}" for m, d in MODEL_DATASET_PAIRS]


def _basic_parquet(path, rows=3):
    pd.DataFrame(
        {
            "n_nodes": np.arange(1, rows + 1, dtype="int64"),
            "duration": np.linspace(10.0, 50.0, rows),
            "power_consumption": np.linspace(100.0, 200.0, rows),
            "partition": ["a", "b"] * (rows // 2) + ["a"] * (rows % 2),
        }
    ).to_parquet(path)


@pytest.fixture
def parquet_file(tmp_path):
    p = tmp_path / "sample.parquet"
    _basic_parquet(p)
    return p


# ---------------------------------------------------------------------------
# Base SeanergysParquetDataset __getitem__/__len__/get_feature_statistics
# (dead-but-reachable: concrete subclasses override these, base is exercised
#  directly via model_construct to avoid the auto-load validator)
# ---------------------------------------------------------------------------


def test_base_parquet_len_zero_without_df():
    ds = SeanergysParquetDataset.model_construct(
        input_features=["a"], output_features=["y"]
    )
    assert len(ds) == 0


def test_base_parquet_getitem_without_df_raises():
    ds = SeanergysParquetDataset.model_construct(
        input_features=["a"], output_features=["y"]
    )
    with pytest.raises(RuntimeError):
        ds[0]


def test_base_parquet_getitem_with_df_and_transforms():
    ds = SeanergysParquetDataset.model_construct(
        input_features=["a", "b"],
        output_features=["y"],
        transform=lambda t: t * 2,
        target_transform=lambda t: t + 1,
    )
    ds._df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "y": [5.0, 6.0]})
    assert len(ds) == 2
    x, y = ds[0]
    assert isinstance(x, torch.Tensor)
    assert torch.allclose(x, torch.tensor([2.0, 6.0]))  # transform doubled
    assert torch.allclose(y, torch.tensor([6.0]))  # target +1


def test_base_parquet_feature_statistics_without_df_raises():
    ds = SeanergysParquetDataset.model_construct(
        input_features=["a"], output_features=["y"]
    )
    with pytest.raises(RuntimeError):
        ds.get_feature_statistics()


# ---------------------------------------------------------------------------
# F-DATA: transforms + tensor, preprocessing, download_path, mocked Zenodo
# ---------------------------------------------------------------------------


def test_fdata_getitem_transforms_and_tensor(parquet_file):
    ds = FDataDataset(
        data_path=str(parquet_file),
        input_features=["n_nodes", "duration"],
        output_features=["power_consumption"],
        transform=lambda a: a,
        target_transform=lambda a: a,
    )
    x, y = ds.__getitem__(0, return_tensor=True, dtype=torch.float32)
    assert isinstance(x, torch.Tensor)


def test_fdata_preprocessing_functions(parquet_file):
    seen = {}

    def _pp(df):
        seen["called"] = True
        return df

    FDataDataset(
        data_path=str(parquet_file),
        input_features=["n_nodes"],
        output_features=["power_consumption"],
        preprocessing_functions=[_pp],
    )
    assert seen.get("called") is True


def test_fdata_download_path(tmp_path, parquet_file):
    out = tmp_path / "out"
    FDataDataset(
        data_path=str(parquet_file),
        input_features=["n_nodes"],
        output_features=["power_consumption"],
        download_path=str(out),
    )
    assert (out / "f_data.parquet").exists()


def test_fdata_max_rows_caps_read(tmp_path):
    """max_rows reads only the first N rows via iter_batches (the memory-safe path
    is_dummy uses so a single-row-group file is never fully materialised)."""
    p = tmp_path / "many.parquet"
    _basic_parquet(p, rows=10)
    ds = FDataDataset(
        data_path=str(p),
        input_features=["n_nodes", "duration"],
        output_features=["power_consumption"],
        max_rows=4,
    )
    assert len(ds) == 4
    # Only the requested columns were materialised.
    assert set(ds._df.columns) == {"n_nodes", "duration", "power_consumption"}


def test_fdata_is_dummy_sets_row_cap(monkeypatch):
    """is_dummy must configure the small row cap (not a non-pushable filter)."""
    # Stub the load so this stays offline; we only assert the derived parameters.
    monkeypatch.setattr(
        FDataDataset, "load_data", lambda self: setattr(self, "_df", pd.DataFrame())
    )
    ds = FDataDataset(
        use_zenodo_url=True,
        is_dummy=True,
        input_features=["n_nodes"],
        output_features=["power_consumption"],
    )
    assert ds.max_rows == 200
    assert ds.filters is None


def test_fdata_zenodo_merge(monkeypatch):
    fake = pd.DataFrame(
        {"n_nodes": [1, 2], "duration": [1.0, 2.0], "power_consumption": [3.0, 4.0]}
    )
    monkeypatch.setattr(fdata_mod.pd, "read_parquet", lambda *a, **k: fake.copy())
    ds = FDataDataset(
        use_zenodo_url=True,
        files=FDATA_FILES[:2],
        input_features=["n_nodes", "duration"],
        output_features=["power_consumption"],
    )
    assert len(ds) == 4  # two mocked files concatenated


# ---------------------------------------------------------------------------
# PM100: dummy mode (local) + preprocessing
# ---------------------------------------------------------------------------


def test_pm100_dummy_local_filter(tmp_path):
    p = tmp_path / "pm.parquet"
    pd.DataFrame(
        {
            "n_nodes": [1, 2],
            "power_consumption": [1.0, 2.0],
            "submit_time": pd.to_datetime(["2020-04-01", "2020-06-01"], utc=True),
        }
    ).to_parquet(p)
    ds = PM100Dataset(
        data_path=str(p),
        input_features=["n_nodes"],
        output_features=["power_consumption"],
        columns=["n_nodes", "power_consumption", "submit_time"],
        is_dummy=True,
    )
    # dummy filter keeps submit_time <= 2020-05-01 → 1 row
    assert len(ds) == 1


def test_pm100_preprocessing_functions(parquet_file):
    seen = {}
    PM100Dataset(
        data_path=str(parquet_file),
        input_features=["n_nodes"],
        output_features=["power_consumption"],
        preprocessing_functions=[lambda df: (seen.setdefault("c", 1), df)[1]],
    )
    assert seen.get("c") == 1


# ---------------------------------------------------------------------------
# M100: nan-fill across heterogeneous schema, node filtering, download+extract
# ---------------------------------------------------------------------------


def test_m100_nan_fill_for_missing_column(tmp_path):
    d = tmp_path / AGG_RECORD / "0"
    d.mkdir(parents=True)
    pd.DataFrame({"timestamp": [0, 1], "p0": [1.0, 2.0], "p1": [3.0, 4.0]}).to_parquet(
        d / "0.parquet"
    )
    pd.DataFrame({"timestamp": [0, 1], "p0": [5.0, 6.0]}).to_parquet(d / "1.parquet")
    ds = M100Dataset(aggregated=True, nodes="0:1", dataset_path=str(tmp_path))
    rows = [ds[i] for i in range(len(ds))]
    assert any(np.isnan(r).any() for r in rows)


def test_m100_node_filter_skips_extra_files(tmp_path):
    d = tmp_path / AGG_RECORD / "0"
    d.mkdir(parents=True)
    for nd in (0, 1, 2):
        pd.DataFrame({"timestamp": [0, 1], "p0": [1.0, 2.0]}).to_parquet(
            d / f"{nd}.parquet"
        )
    ds = M100Dataset(aggregated=True, nodes="0:1", dataset_path=str(tmp_path))
    assert len(ds) == 4  # nodes 0,1 only (node 2 skipped)


def test_m100_plugin_filter_skips(tmp_path):
    for plugin in ("ipmi", "other"):
        base = (
            tmp_path
            / RAW_RECORD
            / RAW_MONTH
            / "year_month=22-02"
            / f"plugin={plugin}"
            / "metric=power"
        )
        base.mkdir(parents=True)
        pd.DataFrame({"timestamp": [0, 1], "p0": [1.0, 2.0]}).to_parquet(
            base / "d.parquet"
        )
    ds = M100Dataset(
        yy_mm=RAW_MONTH, plugins=["ipmi"], metrics=["power"], dataset_path=str(tmp_path)
    )
    assert len(ds) == 2  # only the ipmi plugin file


def test_m100_download_and_extract(tmp_path, monkeypatch):
    node_pq = tmp_path / "0.parquet"
    pd.DataFrame({"timestamp": [0, 1], "p0": [1.0, 2.0]}).to_parquet(node_pq)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as t:
        t.add(node_pq, arcname="0.parquet")
    tar_bytes = buf.getvalue()

    class _FakeResp:
        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size=8192):
            yield tar_bytes

    monkeypatch.setattr(m100_mod.requests, "get", lambda url, stream=True: _FakeResp())
    ds = M100Dataset(aggregated=True, nodes="0", dataset_path=str(tmp_path / "data"))
    assert len(ds) == 2


# ---------------------------------------------------------------------------
# Model get_test_config valid branches (mocked dataset loads)
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_dataset_loads(monkeypatch):
    """Stub the network load of every dataset a discovered model actually uses,
    so the get_test_config valid branch is reachable offline for any (model,
    dataset) pair — no per-dataset hardcoding.

    NB: the stub targets are taken from each model's ``supported_datasets`` (the
    canonical dataset classes the model imports and instantiates), NOT from the
    discovery's DATASET_CLASSES — those are imported under mangled module names
    and are DIFFERENT class objects, so patching them would miss and the real
    network load would run.
    """

    def _stub(self):
        self._df = pd.DataFrame({"x": [1, 2]})

    seen = set()
    for model_cls, _ in MODEL_DATASET_PAIRS:
        for ds_cls in model_cls().supported_datasets or []:
            if ds_cls not in seen:
                monkeypatch.setattr(ds_cls, "load_data", _stub, raising=False)
                seen.add(ds_cls)


@pytest.mark.parametrize("model_cls,dataset_id", MODEL_DATASET_PAIRS, ids=_PAIR_IDS)
def test_get_test_config_valid_branch(model_cls, dataset_id, stub_dataset_loads):
    """get_test_config builds a config for every supported (model, dataset) pair
    (dataset loads stubbed). Discovery-driven — new pairs are covered automatically."""
    cfg = model_cls.get_test_config(model_cls(), dataset_id)
    assert dataset_id in cfg


# ---------------------------------------------------------------------------
# Model save error paths
# ---------------------------------------------------------------------------


def test_mcbound_save_error_returns_false(tmp_path, monkeypatch):
    import seanergys_modelzoo.models.tasks.performance_prediction.mcbound.mcbound_model as m  # noqa: E501
    from seanergys_modelzoo.models.tasks.performance_prediction.mcbound.mcbound_model import (  # noqa: E501
        MCBound,
    )

    def _boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(m.joblib, "dump", _boom)
    assert MCBound().save(tmp_path / "m") is False


def test_mack_save_error_returns_false(tmp_path, monkeypatch):
    import seanergys_modelzoo.models.tasks.performance_prediction.mack.mack_model as m
    from seanergys_modelzoo.models.tasks.performance_prediction.mack.mack_model import (
        MACK,
    )

    monkeypatch.setattr(
        m.joblib, "dump", lambda *a, **k: (_ for _ in ()).throw(OSError("x"))
    )
    assert MACK().save(tmp_path / "m") is False


def test_jpcp_save_error_returns_false(tmp_path, monkeypatch):
    import seanergys_modelzoo.models.tasks.power_consumption_prediction.jpcp.jpcp_model as m  # noqa: E501
    from seanergys_modelzoo.models.tasks.power_consumption_prediction.jpcp.jpcp_model import (  # noqa: E501
        JPCP,
    )

    monkeypatch.setattr(
        m.joblib, "dump", lambda *a, **k: (_ for _ in ()).throw(OSError("x"))
    )
    assert JPCP().save(tmp_path / "m") is False
