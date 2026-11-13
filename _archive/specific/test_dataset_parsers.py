"""Unit tests for dataset file/node-spec parsing logic.

These parsers (M100Dataset, FDataDataset) are pure functions over class-level
file tables. They are exercised via ``model_construct`` so instantiation does
not trigger ``validate_model`` (which would download from Zenodo).
"""

import pytest

import torch  # noqa: F401
import pyarrow  # noqa: F401
import cachetools  # noqa: F401

from seanergys_modelzoo.datasets.f_data import FDataDataset  # noqa: E402
from seanergys_modelzoo.datasets.m100 import M100Dataset  # noqa: E402

# ---------------------------------------------------------------------------
# Constants read from the CLASSES, never retyped (GitLab issue #6).
#
# Zenodo record ids and monthly file ids used to be typed out here. When one
# changed upstream these tests failed with a mismatch that read like a code bug
# rather than a stale test. Deriving them means changing AVAILABLE_FILES or the
# record map needs no edit to this file.
# ---------------------------------------------------------------------------

# F-DATA monthly file ids, e.g. ("21_03", "21_04", ...).
FDATA_FILES = list(FDataDataset.model_fields["AVAILABLE_FILES"].default)

# M100 record map: [{record_id: [file, ...]}, ...]. The LAST entry is the
# 15-minute aggregated record; every earlier entry is raw monthly data.
M100_RECORDS = list(M100Dataset.model_fields["AVAILABLE_FILES_PER_RECORD"].default)
AGG_RECORD, AGG_FILES = next(iter(M100_RECORDS[-1].items()))

# Flatten the raw (non-aggregated) records into an ordered (record_id, file) list.
M100_RAW = [
    (rid, f)
    for record in M100_RECORDS
    for rid, files in record.items()
    if rid != AGG_RECORD
    for f in files
]
NODES_PER_RACK = 20


# ---------------------------------------------------------------------------
# M100Dataset._parse_nodes
# ---------------------------------------------------------------------------


def test_parse_nodes_none():
    assert M100Dataset.model_construct(nodes=None)._parse_nodes() is None


def test_parse_nodes_single_int():
    assert M100Dataset.model_construct(nodes=7)._parse_nodes() == [7]


def test_parse_nodes_range_string():
    assert M100Dataset.model_construct(nodes="5:10")._parse_nodes() == [
        5,
        6,
        7,
        8,
        9,
        10,
    ]


def test_parse_nodes_single_string():
    assert M100Dataset.model_construct(nodes="3")._parse_nodes() == [3]


def test_parse_nodes_mixed_list():
    assert M100Dataset.model_construct(nodes=[1, "3:5", 10])._parse_nodes() == [
        1,
        3,
        4,
        5,
        10,
    ]


def test_parse_nodes_all():
    nodes = M100Dataset.model_construct(nodes="all")._parse_nodes()
    assert nodes == list(range(1000))


def test_get_total_nodes():
    expected = len(AGG_FILES) * NODES_PER_RACK
    assert M100Dataset.model_construct()._get_total_nodes() == expected


# ---------------------------------------------------------------------------
# M100Dataset._build_file_list
# ---------------------------------------------------------------------------


def test_build_file_list_aggregated_maps_nodes_to_racks():
    ds = M100Dataset.model_construct()
    ds._nodes = [0, NODES_PER_RACK + 5, 2 * NODES_PER_RACK]  # racks 0, 1, 2
    files = ds._build_file_list(None, aggregated=True)
    assert files == [(AGG_RECORD, "0"), (AGG_RECORD, "1"), (AGG_RECORD, "2")]


def test_build_file_list_aggregated_fallback_all_racks():
    ds = M100Dataset.model_construct()
    ds._nodes = []
    files = ds._build_file_list(None, aggregated=True)
    assert len(files) == len(AGG_FILES)
    assert all(rid == AGG_RECORD for rid, _ in files)


def test_build_file_list_raw_single():
    ds = M100Dataset.model_construct()
    ds._nodes = []
    rid, month = M100_RAW[-1]
    assert ds._build_file_list(month, aggregated=False) == [(rid, month)]


def test_build_file_list_raw_range():
    ds = M100Dataset.model_construct()
    ds._nodes = []
    first, third = M100_RAW[0][1], M100_RAW[2][1]
    files = ds._build_file_list(f"{first}:{third}", aggregated=False)
    assert files == M100_RAW[:3]


def test_build_file_list_raw_list():
    ds = M100Dataset.model_construct()
    ds._nodes = []
    picked = [M100_RAW[0], M100_RAW[10]]
    files = ds._build_file_list([f for _, f in picked], aggregated=False)
    assert files == picked


def test_build_file_list_raw_all():
    ds = M100Dataset.model_construct()
    ds._nodes = []
    files = ds._build_file_list(None, aggregated=False)
    # all non-aggregated months
    assert M100_RAW[-1] in files
    assert all(rid != AGG_RECORD for rid, _ in files)


def test_build_file_list_invalid_type_raises():
    ds = M100Dataset.model_construct()
    ds._nodes = []
    with pytest.raises(TypeError):
        ds._build_file_list(123, aggregated=False)


# ---------------------------------------------------------------------------
# FDataDataset._parse_files_parameter
# ---------------------------------------------------------------------------


def test_fdata_parse_files_none_returns_all():
    ds = FDataDataset.model_construct()
    assert ds._parse_files_parameter(None) == ds.AVAILABLE_FILES


def test_fdata_parse_files_all_string():
    ds = FDataDataset.model_construct()
    assert ds._parse_files_parameter("all") == ds.AVAILABLE_FILES


def test_fdata_parse_files_single():
    ds = FDataDataset.model_construct()
    assert ds._parse_files_parameter(FDATA_FILES[0]) == [FDATA_FILES[0]]


def test_fdata_parse_files_single_strips_extension():
    ds = FDataDataset.model_construct()
    assert ds._parse_files_parameter(f"{FDATA_FILES[0]}.parquet") == [FDATA_FILES[0]]


def test_fdata_parse_files_range():
    ds = FDataDataset.model_construct()
    spec = f"{FDATA_FILES[0]}:{FDATA_FILES[2]}"
    assert ds._parse_files_parameter(spec) == FDATA_FILES[:3]


def test_fdata_parse_files_list():
    ds = FDataDataset.model_construct()
    assert ds._parse_files_parameter(FDATA_FILES[:2]) == FDATA_FILES[:2]


def test_fdata_parse_files_invalid_single_raises():
    ds = FDataDataset.model_construct()
    with pytest.raises(ValueError):
        ds._parse_files_parameter("99_99")


def test_fdata_parse_files_invalid_range_raises():
    ds = FDataDataset.model_construct()
    with pytest.raises(ValueError):
        ds._parse_files_parameter("21_03:99_99")


def test_fdata_parse_files_invalid_list_raises():
    ds = FDataDataset.model_construct()
    with pytest.raises(ValueError):
        ds._parse_files_parameter([FDATA_FILES[0], "nope"])


def test_fdata_parse_files_invalid_type_raises():
    ds = FDataDataset.model_construct()
    with pytest.raises(TypeError):
        ds._parse_files_parameter(123)
