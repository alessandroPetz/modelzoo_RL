"""Unit tests for SeanergysDataloader (type guard + from_config)."""

import numpy as np
import pytest

import torch  # noqa: F401

from seanergys_modelzoo.dataloader.seanergys_dataloader import (  # noqa: E402
    SeanergysDataloader,
)
from seanergys_modelzoo.datasets.common.seanergys_dataset import (  # noqa: E402
    SeanergysDataset,
)
from seanergys_modelzoo.models.common.seanergys_configurator import (  # noqa: E402
    SeanergysDataloaderParams,
)


class _TinyDataset(SeanergysDataset):
    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self._X = np.arange(12, dtype=np.float32).reshape(6, 2)

    def __getitem__(self, idx: int):
        return self._X[idx], np.float32(idx)

    def __len__(self) -> int:
        return 6


def test_dataloader_rejects_non_seanergys_dataset():
    from torch.utils.data import TensorDataset

    plain = TensorDataset(torch.zeros(4, 2))
    with pytest.raises(ValueError, match="SeanergysDataset"):
        SeanergysDataloader(plain)


def test_dataloader_accepts_seanergys_dataset():
    loader = SeanergysDataloader(_TinyDataset(), batch_size=2)
    batches = list(loader)
    assert len(batches) == 3


def test_dataloader_from_config():
    params = SeanergysDataloaderParams(batch_size=3, shuffle=False, num_workers=0)
    loader = SeanergysDataloader.from_config(_TinyDataset(), params)
    assert loader.batch_size == 3
    assert len(list(loader)) == 2
