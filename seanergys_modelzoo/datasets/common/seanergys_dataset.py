from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, List, Optional, Union
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict, model_validator
from torch.utils.data import Dataset

from seanergys_modelzoo.logger.seanergys_logger import SeanergysLogger


class SeanergysDataset(BaseModel, Dataset, ABC):
    """
    Base dataset class for Seanergy datasets.

    This class provides a foundation for creating datasets with support for
    various data sources, preprocessing, caching, and monitoring capabilities.
    """

    # Pydantic configuration
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    # Every concrete dataset must set this to the key used in get_train_config().
    # Example: DATASET_ID = "f-data" or DATASET_ID = "pm100"
    DATASET_ID: ClassVar[str] = ""

    # Define fields with proper Pydantic syntax
    data_path: Optional[Union[str, Path]] = Field(
        default=None, description="Path to data source"
    )
    transform: Optional[Any] = Field(
        default=None, description="Transformations to apply to input data"
    )
    target_transform: Optional[Any] = Field(
        default=None, description="Transformations to apply to targets"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional metadata about the dataset"
    )
    preprocessing_functions: Optional[List[Any]] = Field(
        default_factory=list, description="Preprocessing functions to apply"
    )
    is_dummy: Optional[bool] = Field(
        default=False,
        description="Whether to load the dataset in dummy mode. Defaults to False",
    )
    logger: SeanergysLogger = Field(
        default_factory=SeanergysLogger, description="Logger instance"
    )
    stats: Dict[str, Any] = Field(
        default_factory=lambda: {
            "n_samples": 0,
            "n_features": 0,
            "load_time": 0.0,
        },
        description="Dataset statistics",
    )

    def model_post_init(self, __context: Any) -> None:
        """
        Pydantic v2 method called after model initialization.
        Replaces __init__ for post-initialization logic.
        """
        # Initialize logger if not provided
        if self.logger is None:
            self.logger = SeanergysLogger()

        if self.metadata is None:
            self.metadata = {
                "dataset_name": "SeanergysDataset",
            }

        self.data_name = self.__class__.__name__
        # Log activities
        self.logger.info(f"Initialized {self.data_name}")

    @model_validator(mode="after")
    def validate_model(self) -> "SeanergysDataset":
        """
        Implement to validate the model
        """
        return self

    @abstractmethod
    def __getitem__(self, idx: int) -> Any:
        """
        Retrieve a single item from the dataset.

        Subclasses may return either a ``(data, target)`` tuple for supervised
        datasets or a bare feature vector for unsupervised ones, so the return
        type is intentionally ``Any``.

        Args:
            idx: Index of the item to retrieve

        Returns:
            The sample for the given index (e.g. a ``(data, target)`` tuple or a
            feature vector).
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Return dataset statistics."""
        return self.stats.copy()

    def get_metadata(self) -> Dict[str, Any]:
        """Return dataset metadata."""
        return {
            **(self.metadata or {}),
            "stats": self.stats,
        }
