"""Unit tests for M100Dataset.

The dataset is exercised entirely offline by pre-creating the local
``dataset_path/<record_id>/<file>`` parquet layout that ``build_index`` expects,
so no Zenodo download is triggered (download only happens when the path is
absent). The Zenodo download path and the ``__main__`` demo block are not
covered here.
"""

import io
import tarfile

import numpy as np
import pytest

import pyarrow  # noqa: F401
import cachetools  # noqa: F401
import pandas as pd

from seanergys_modelzoo.datasets.m100 import M100Dataset  # noqa: E402

# ---------------------------------------------------------------------------
# Constants read from the CLASS, never retyped (GitLab issue #6).
# Changing M100Dataset's record map upstream requires no edit here.
# ---------------------------------------------------------------------------
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


def _write_node_parquet(path, n=3):
    pd.DataFrame(
        {
            "timestamp": list(range(n)),
            "p0": np.arange(n, dtype=float),
            "p1": np.arange(n, dtype=float) + 10.0,
        }
    ).to_parquet(path)


def _agg_layout(root, rack=0, nodes=(0, 1), n=3):
    d = root / AGG_RECORD / str(rack)
    d.mkdir(parents=True)
    for nd in nodes:
        _write_node_parquet(d / f"{nd}.parquet", n)
    return root


# ---------------------------------------------------------------------------
# Aggregated mode — full index + access
# ---------------------------------------------------------------------------


def test_m100_aggregated_len_and_getitem(tmp_path):
    _agg_layout(tmp_path, nodes=(0, 1), n=3)
    ds = M100Dataset(aggregated=True, nodes="0:1", dataset_path=str(tmp_path))
    assert len(ds) == 6  # 2 node files x 3 rows
    x = ds[0]
    assert isinstance(x, np.ndarray)
    assert x.shape == (2,)  # p0, p1 (timestamp discarded)
    # last row reachable
    assert ds[5].shape == (2,)


def test_m100_dummy_mode(tmp_path):
    _agg_layout(tmp_path, nodes=(0, 1), n=2)
    ds = M100Dataset(is_dummy=True, dataset_path=str(tmp_path))
    assert len(ds) >= 1
    assert isinstance(ds[0], np.ndarray)


def test_m100_row_group_cache_hit(tmp_path):
    """Second access to the same row group exercises the LRU cache branch."""
    _agg_layout(tmp_path, nodes=(0,), n=4)
    ds = M100Dataset(aggregated=True, nodes="0", dataset_path=str(tmp_path))
    first = ds[0]
    again = ds[0]
    assert np.array_equal(first, again)


# ---------------------------------------------------------------------------
# Raw mode
# ---------------------------------------------------------------------------


def test_m100_raw_mode(tmp_path):
    base = (
        tmp_path
        / RAW_RECORD
        / RAW_MONTH
        / "year_month=22-02"
        / "plugin=ipmi"
        / "metric=power"
    )
    base.mkdir(parents=True)
    _write_node_parquet(base / "data.parquet", n=3)
    ds = M100Dataset(
        yy_mm=RAW_MONTH,
        plugins=["ipmi"],
        metrics=["power"],
        dataset_path=str(tmp_path),
    )
    assert len(ds) == 3


def test_m100_raw_mode_metric_filtered_out(tmp_path):
    base = (
        tmp_path
        / RAW_RECORD
        / RAW_MONTH
        / "year_month=22-02"
        / "plugin=ipmi"
        / "metric=temp"
    )
    base.mkdir(parents=True)
    _write_node_parquet(base / "data.parquet", n=3)
    # metric "power" requested but only "temp" present → no files → ValueError
    with pytest.raises(ValueError, match="No parquet"):
        M100Dataset(
            yy_mm=RAW_MONTH,
            plugins=["ipmi"],
            metrics=["power"],
            dataset_path=str(tmp_path),
        )


# ---------------------------------------------------------------------------
# Validation error branches
# ---------------------------------------------------------------------------


def test_m100_aggregated_requires_nodes(tmp_path):
    with pytest.raises(ValueError, match="Aggregated"):
        M100Dataset(aggregated=True, dataset_path=str(tmp_path))


def test_m100_invalid_node_ids(tmp_path):
    with pytest.raises(ValueError, match="Invalid node"):
        M100Dataset(aggregated=True, nodes="5000", dataset_path=str(tmp_path))


def test_m100_raw_requires_yymm_and_plugins(tmp_path):
    with pytest.raises(ValueError, match="Raw data"):
        M100Dataset(aggregated=False, dataset_path=str(tmp_path))


def test_m100_no_files_found(tmp_path):
    (tmp_path / AGG_RECORD / "0").mkdir(parents=True)  # exists but empty
    with pytest.raises(ValueError, match="No parquet"):
        M100Dataset(aggregated=True, nodes="0", dataset_path=str(tmp_path))


def test_m100_getitem_out_of_range(tmp_path):
    _agg_layout(tmp_path, nodes=(0,), n=2)
    ds = M100Dataset(aggregated=True, nodes="0", dataset_path=str(tmp_path))
    with pytest.raises(IndexError):
        ds[9999]


# ---------------------------------------------------------------------------
# _parse_nodes additional branches (via model_construct)
# ---------------------------------------------------------------------------


def test_m100_parse_nodes_string_in_list():
    assert M100Dataset.model_construct(nodes=[1, "7"])._parse_nodes() == [1, 7]


def test_m100_parse_nodes_invalid_type_raises():
    with pytest.raises(TypeError):
        M100Dataset.model_construct(nodes=1.5)._parse_nodes()


# ---------------------------------------------------------------------------
# _safe_extract
# ---------------------------------------------------------------------------


def test_m100_safe_extract_rejects_traversal(tmp_path):
    ds = M100Dataset.model_construct()
    tar_path = tmp_path / "evil.tar"
    with tarfile.open(tar_path, "w") as t:
        info = tarfile.TarInfo(name="../evil.txt")
        data = b"x"
        info.size = len(data)
        t.addfile(info, io.BytesIO(data))
    with tarfile.open(tar_path) as t:
        with pytest.raises(Exception, match="Unsafe"):
            ds._safe_extract(t, tmp_path / "dest")


def test_m100_safe_extract_ok(tmp_path):
    ds = M100Dataset.model_construct()
    src = tmp_path / "a.txt"
    src.write_text("hi")
    tar_path = tmp_path / "ok.tar"
    with tarfile.open(tar_path, "w") as t:
        t.add(src, arcname="a.txt")
    dest = tmp_path / "out"
    dest.mkdir()
    with tarfile.open(tar_path) as t:
        ds._safe_extract(t, dest)
    assert (dest / "a.txt").exists()
