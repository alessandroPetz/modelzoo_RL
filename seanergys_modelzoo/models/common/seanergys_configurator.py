from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class SeanergysComponentsParams(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SeanergysComponentsParams":
        return cls(**data)


class SeanergysModelParams(SeanergysComponentsParams):
    pass


class SeanergysDatasetParams(SeanergysComponentsParams):
    data_path: Optional[Path] = Field(default=None)
    transform: Optional[Callable] = Field(default=None)
    target_transform: Optional[Callable] = Field(default=None)
    metadata: Optional[Any] = Field(default=None)

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        for key, value in self.model_dump().items():
            if value is not None:
                result[key] = value
        for key in self.model_fields_set:
            if key not in result:
                result[key] = getattr(self, key)
        extra = self.model_extra or {}
        result.update(extra)
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SeanergysDatasetParams":
        return cls(**data)


class SeanergysDataloaderParams(SeanergysComponentsParams):
    batch_size: Optional[int] = Field(default=1)
    shuffle: Optional[bool] = Field(default=None)
    sampler: Optional[Any] = Field(default=None)
    batch_sampler: Optional[Any] = Field(default=None)
    num_workers: int = Field(default=0)
    collate_fn: Optional[Callable] = Field(default=None)
    pin_memory: bool = Field(default=False)
    drop_last: bool = Field(default=False)
    timeout: float = Field(default=0)
    worker_init_fn: Optional[Callable] = Field(default=None)
    multiprocessing_context: Optional[Any] = Field(default=None)
    generator: Optional[Any] = Field(default=None)
    prefetch_factor: Optional[int] = Field(default=None)
    persistent_workers: bool = Field(default=False)
    pin_memory_device: str = Field(default="")
    in_order: bool = Field(default=True)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SeanergysDataloaderParams":
        return cls(**data)
