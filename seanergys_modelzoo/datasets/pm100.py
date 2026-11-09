import time
from pathlib import Path
from typing import ClassVar, Optional, Tuple, Union

import pandas as pd
import torch
from pydantic import Field, model_validator

from seanergys_modelzoo.datasets.common.seanergys_parquet_dataset import (
    SeanergysParquetDataset,
)


class PM100Dataset(SeanergysParquetDataset):
    """
    Dataset class for PM100: A Job Power Consumption Dataset from HPC System.

    PM100 contains 231,116 jobs from Marconi100 supercomputer with power consumption
    recorded at Node, CPU, and Memory levels. Each job has 32 features including
    job characteristics and power consumption time series.

    Dataset: https://zenodo.org/records/10127767
    Paper: https://doi.org/10.1145/3624062.3624263

    Key Features:
    - job_id: Unique job identifier
    - user: User who submitted the job
    - account: Account associated with the job
    - partition: Partition where job was executed
    - nodes: List of allocated nodes
    - n_nodes: Number of nodes allocated
    - n_tasks: Number of tasks in the job
    - cpus_per_task: CPUs allocated per task
    - time_limit: Maximum time limit (seconds)
    - job_state: Exit state of the job
    - elapsed_time: Actual execution time (seconds)
    - submit_time: Job submission timestamp
    - start_time: Job start timestamp
    - end_time: Job end timestamp
    - power_consumption: List of power consumption values (watts) at 20s intervals
    - And more...
    """

    DATASET_ID: ClassVar[str] = "pm100"

    # Direct download URL for the PM100 dataset
    ZENODO_URL: str = "https://zenodo.org/records/10127767/files/job_table.parquet"

    # Pydantic fields specific to PM100Dataset
    use_zenodo_url: bool = Field(
        default=False,
        description="If True and data_path is None, load the dataset directly from Zenodo.",
    )
    download_path: Optional[Union[str, Path]] = Field(
        default=None,
        description="If provided, save the loaded dataset as a parquet file in this directory.",
    )

    @model_validator(mode="after")
    def validate_model(self) -> "PM100Dataset":
        """Resolve data_path from Zenodo URL if needed, enrich metadata, and trigger data loading."""
        # Resolve data_path
        if self.data_path is None and self.use_zenodo_url:
            self.logger.info(f"Using Zenodo URL for PM100 dataset: {self.ZENODO_URL}")
            self.data_path = self.ZENODO_URL
        elif self.data_path is None:
            raise ValueError(
                "data_path must be provided or use_zenodo_url must be True."
            )

        if self.columns is None:
            self.columns = self.input_features + self.output_features
        # else:
        #     self.columns = list(set(self.columns+self.input_features+self.output_features))

        # Enrich metadata with PM100-specific info
        assert self.metadata is not None
        self.metadata.update(
            {
                "dataset_name": "PM100",
                "dataset_version": "v3",
                "source": "Marconi100 HPC System (CINECA, Italy)",
                "time_period": "May-October 2020",
                "doi": "10.5281/zenodo.10127767",
                "paper_doi": "10.1145/3624062.3624263",
            }
        )

        # If dummy swap parameters
        if self.is_dummy:
            if not (self.filters):
                self.filters = []
            self.filters += [
                [("submit_time", "<=", pd.Timestamp("2020-05-01", tz="UTC"))]  # type: ignore
            ]

        self.load_data()
        return self

    def load_data(self) -> None:
        """
        Load data from Parquet file (local or remote) and apply filters.
        """
        start_time = time.time()

        try:
            data_path_str = str(self.data_path)

            if data_path_str.startswith(("http://", "https://", "s3://", "gs://")):
                self.logger.info(
                    f"Loading {self.__class__.__name__} from remote source: {data_path_str}"
                )
            else:
                self.logger.info(
                    f"Loading {self.__class__.__name__} from local path: {data_path_str}"
                )

            self._df = pd.read_parquet(
                data_path_str, columns=self.columns, filters=self.filters
            )
            self.logger.info(
                f"Loaded {self.__class__.__name__} with shape: {self._df.shape}"
            )

            # Apply preprocessing: each entry is (columns, function, output_feature_name)
            for fn in self.preprocessing_functions or []:
                self._df = fn(self._df)

            # Optionally persist to disk
            if self.download_path:
                download_path = Path(self.download_path)
                download_path.mkdir(parents=True, exist_ok=True)
                out_file = download_path / "job_table.parquet"
                self._df.to_parquet(out_file, index=False)
                self.logger.info(f"Saved {self.__class__.__name__} file to: {out_file}")

            # Update statistics
            self.stats.update(
                {
                    "n_samples": len(self._df),
                    "n_input_features": len(self.input_features),
                    "n_output_features": len(self.output_features),
                    "n_features": len(self.input_features),
                    "load_time": time.time() - start_time,
                }
            )

            self.logger.info(
                f"{self.__class__.__name__} ready: {self.stats['n_samples']} samples, "
                f"{self.stats['n_input_features']} input features, "
                f"{self.stats['n_output_features']} output features "
                f"(loaded in {self.stats['load_time']:.2f}s)"
            )

        except Exception as e:
            self.logger.error(f"Failed to load {self.__class__.__name__} file: {e}")
            raise

    def __getitem__(
        self, idx: int, return_tensor: bool = False, dtype: Optional[torch.dtype] = None
    ) -> Tuple:
        """
        Retrieve a single sample from the dataset.

        Args:
            idx: Index of the sample to retrieve
            return_tensor: If True, return torch tensors instead of numpy arrays
            dtype: Tensor dtype (only used when return_tensor=True)

        Returns:
            Tuple of (input_data, output_data)
        """
        if self._df is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")
        if idx >= len(self):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self)}"
            )

        input_data = self._df[self.input_features].iloc[idx].values
        output_data = self._df[self.output_features].iloc[idx].values

        if self.transform is not None:
            input_data = self.transform(input_data)

        if self.target_transform is not None:
            output_data = self.target_transform(output_data)

        if return_tensor:
            input_data = torch.tensor(input_data, dtype=dtype)
            output_data = torch.tensor(output_data, dtype=dtype)

        return input_data, output_data
