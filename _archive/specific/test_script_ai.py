"""Unit tests for SCRIPTAIDataset using a local features.csv + script files.

The remote GitHub loading path requires network and is not exercised here.
"""

import pytest

import torch  # noqa: F401
import pandas as pd

from seanergys_modelzoo.datasets.script_ai import SCRIPTAIDataset  # noqa: E402


def _layout(root, jids=("j1", "j2"), targets=(1.0, 2.0)):
    pd.DataFrame({"jid": list(jids), "target": list(targets)}).to_csv(
        root / "features.csv", index=False
    )
    for jid in jids:
        (root / jid).write_text(f"#!/bin/bash\n# job {jid}\necho hello\n")
    return root


@pytest.fixture
def script_dir(tmp_path):
    return _layout(tmp_path)


def _dataset(root, **kwargs):
    return SCRIPTAIDataset(data_path=str(root), output_features=["target"], **kwargs)


def test_loads_local_features_and_scripts(script_dir):
    ds = _dataset(script_dir)
    assert len(ds) == 2
    assert ds.stats["n_samples"] == 2


def test_getitem_returns_script_and_target(script_dir):
    ds = _dataset(script_dir)
    script, target = ds[0]
    assert "echo hello" in script
    assert len(target) == 1


def test_getitem_with_transforms(script_dir):
    ds = _dataset(
        script_dir,
        transform=lambda s: s.upper(),
        target_transform=lambda y: y * 10,
    )
    script, target = ds[0]
    assert script == script.upper()


def test_getitem_out_of_range(script_dir):
    ds = _dataset(script_dir)
    with pytest.raises(IndexError):
        ds[999]


def test_get_stats_includes_feature_stats(script_dir):
    ds = _dataset(script_dir)
    stats = ds.get_stats()
    assert "feature_stats" in stats
    assert "valid_scripts" in stats
    assert stats["valid_scripts"] == 2


def test_summary_string(script_dir):
    ds = _dataset(script_dir)
    text = ds.summary()
    assert "SCRIPT-AI Dataset Summary" in text
    assert "Total Samples:  2" in text


def test_missing_data_path_raises():
    with pytest.raises(ValueError, match="data_path"):
        SCRIPTAIDataset(output_features=["target"], use_github_url=False)


def test_download_path_persists_features(tmp_path, script_dir):
    out = tmp_path / "saved"
    ds = SCRIPTAIDataset(
        data_path=str(script_dir),
        output_features=["target"],
        download_path=str(out),
    )
    assert (out / "features.csv").exists()
    assert len(ds) == 2


def test_metadata_enriched(script_dir):
    ds = _dataset(script_dir)
    assert ds.metadata["dataset_name"] == "SCRIPT-AI"
    assert ds.metadata["valid_scripts"] == 2
