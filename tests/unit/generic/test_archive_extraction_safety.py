"""
Tar path-traversal rejection, for every dataset that extracts archives (#6).

This was one dataset's test. It is not one dataset's property: any dataset that
unpacks a downloaded archive must refuse members that escape the target
directory, or a malicious or corrupted archive overwrites files outside it.
Written once here, it applies automatically to the next dataset that grows an
extraction step.

Datasets are selected by CAPABILITY — "exposes _safe_extract" — not by name, so
this module stays free of concrete names (enforced by test_no_hardcoded_names).
"""

import tarfile

import pytest

import torch  # noqa: F401

from tests.conftest import DATASET_CLASSES

_EXTRACTORS = {
    name: cls for name, cls in DATASET_CLASSES.items() if hasattr(cls, "_safe_extract")
}

pytestmark = pytest.mark.skipif(
    not _EXTRACTORS, reason="no discovered dataset exposes _safe_extract"
)


def _tar_with_member(tmp_path, member_name: str):
    """Build a one-member tar whose entry is stored under `member_name`."""
    payload = tmp_path / "payload.txt"
    payload.write_text("data")
    archive = tmp_path / "archive.tar"
    with tarfile.open(archive, "w") as tar:
        tar.add(payload, arcname=member_name)
    return archive


@pytest.mark.parametrize(
    "dataset_cls", list(_EXTRACTORS.values()), ids=list(_EXTRACTORS)
)
def test_extraction_rejects_path_traversal(dataset_cls, tmp_path):
    """
    PASS: a member whose path escapes the destination is refused.
    FAIL: the archive can write outside the target directory — a path-traversal
          write triggered by downloading a dataset.
    """
    destination = tmp_path / "dest"
    destination.mkdir()
    archive = _tar_with_member(tmp_path, "../escaped.txt")

    dataset = dataset_cls.model_construct()
    with tarfile.open(archive) as tar:
        with pytest.raises(Exception):
            dataset._safe_extract(tar, destination)

    assert not (tmp_path / "escaped.txt").exists(), (
        f"FAIL [{dataset_cls.__name__}] _safe_extract() wrote outside the "
        f"destination directory."
    )


@pytest.mark.parametrize(
    "dataset_cls", list(_EXTRACTORS.values()), ids=list(_EXTRACTORS)
)
def test_extraction_accepts_a_normal_member(dataset_cls, tmp_path):
    """
    PASS: an ordinary member still extracts, so the guard above is a real check
          and not a blanket refusal that would make the dataset unusable.
    FAIL: legitimate archives are rejected too.
    """
    destination = tmp_path / "dest"
    destination.mkdir()
    archive = _tar_with_member(tmp_path, "inside.txt")

    dataset = dataset_cls.model_construct()
    with tarfile.open(archive) as tar:
        dataset._safe_extract(tar, destination)

    assert (destination / "inside.txt").exists(), (
        f"FAIL [{dataset_cls.__name__}] _safe_extract() refused a legitimate member."
    )
