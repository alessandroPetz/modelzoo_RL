from typing import ClassVar, Dict, List, Optional, Tuple, Union
from pathlib import Path
import time

import pandas as pd
import torch
from pydantic import Field, model_validator

from seanergys_modelzoo.datasets.common.seanergys_parquet_dataset import (
    SeanergysParquetDataset,
)


class FDataDataset(SeanergysParquetDataset):
    """
    Dataset class for F-DATA: A Fugaku Workload Dataset from Supercomputer Fugaku.

    F-DATA contains ~24 million jobs from Fugaku supercomputer with extensive features
    including exit codes, duration, power consumption, performance metrics (#flops,
    memory bandwidth, operational intensity, memory/compute bound labels).

    Dataset: https://zenodo.org/records/11467483
    Paper: https://doi.org/10.1038/s41597-025-05633-1

    The dataset is split across 38 monthly parquet files (YY_MM.parquet) covering
    March 2021 to April 2024.

    Key Features:
    - job_id: Unique job identifier
    - user_encoded: Anonymized user ID
    - account_encoded: Anonymized account ID
    - partition: Partition where job was executed
    - exit_code: Job exit code
    - n_nodes: Number of nodes allocated
    - duration: Job duration (seconds)
    - power_consumption: Average power consumption (watts)
    - flops: Floating point operations per second
    - memory_bandwidth: Memory bandwidth (GB/s)
    - operational_intensity: Ratio of computation to memory access
    - mem_compute_bound: Memory/compute bound classification
    - And more...
    """

    # =====================================================================
    # SUGGESTION — Mohsen Seyedkazemi Ardebili (CI & test maintainer), 2026-09-08
    # Mohsen suggests changing the docstring above, to improve the quality of
    # the code. Not applied here: this file is owned by the dataset/model
    # authors. Tracked in GitLab issue #10.
    #
    # WHY: the "Key Features" list above does not match the schema that the
    # published F-DATA parquet files actually carry, and it is the source of a
    # real CI failure (see the SUGGESTION blocks in mack_model.py and
    # mcbound_model.py). Measured on tests/fixtures/sample_data/sample_fdata.parquet:
    #
    #     documented name        actual published column
    #     ---------------------  -----------------------
    #     job_id                 jid
    #     user_encoded           usr
    #     n_nodes                nnuma / nnumr
    #     power_consumption      avgpcon / minpcon / maxpcon / econ
    #     memory_bandwidth       mbwidth
    #     operational_intensity  opint
    #     mem_compute_bound      pclass      <-- values e.g. 'memory-bound'
    #
    # `mem_compute_bound` exists in NO F-DATA file. Because validate_model()
    # sets `columns = input_features + output_features` and _read_parquet()
    # projects those columns, any model that asks for a documented-but-absent
    # name fails at dataset construction with
    #     "No match for FieldRef.Name(mem_compute_bound)".
    #
    # PROPOSED: replace the list with the names the files really publish, and
    # add a note that the reader projects these names verbatim, so an alias
    # invented in the docstring becomes a runtime error rather than a typo.
    # =====================================================================

    DATASET_ID: ClassVar[str] = "f-data"

    # Base URL for F-DATA files on Zenodo
    ZENODO_BASE_URL: str = r"https://zenodo.org/records/11467483/files"

    # Available monthly files
    AVAILABLE_FILES: List[str] = [
        # 2021
        "21_03",
        "21_04",
        "21_05",
        "21_06",
        "21_07",
        "21_08",
        "21_09",
        "21_10",
        "21_11",
        "21_12",
        # 2022
        "22_01",
        "22_02",
        "22_03",
        "22_04",
        "22_05",
        "22_06",
        "22_07",
        "22_08",
        "22_09",
        "22_10",
        "22_11",
        "22_12",
        # 2023
        "23_01",
        "23_02",
        "23_03",
        "23_04",
        "23_05",
        "23_06",
        "23_07",
        "23_08",
        "23_09",
        "23_10",
        "23_11",
        "23_12",
        # 2024
        "24_01",
        "24_02",
        "24_03",
        "24_04",
    ]

    # Pydantic fields specific to FDataDataset
    files: Optional[Union[str, List[str]]] = Field(
        default=None,
        description=(
            "Monthly file(s) to load. Can be: "
            "single file ('21_03'), list (['21_03', '21_04']), "
            "range ('21_03:21_06'), or 'all' / None for all files."
        ),
    )
    use_zenodo_url: bool = Field(
        default=False,
        description="If True and data_path is None, load files from Zenodo.",
    )
    download_path: Optional[Union[str, Path]] = Field(
        default=None,
        description="If provided, save the merged dataset to this directory after loading.",
    )
    max_rows: Optional[int] = Field(
        default=None,
        description=(
            "If set, read at most this many rows PER FILE, decoding only what is "
            "needed via pyarrow.iter_batches (memory-safe). F-DATA monthly files are a "
            "single large row group, so `filters` cannot push down at read time — the "
            "whole row group would be materialised (~GBs). max_rows stops decoding early, "
            "bounding memory to a few hundred rows. is_dummy sets this automatically."
        ),
    )

    # Private — not a Pydantic field
    _file_list: Optional[List[str]] = None

    @model_validator(mode="after")
    def validate_model(self) -> "FDataDataset":
        """Validate parameters, resolve file list, and trigger data loading."""

        if self.columns is None:
            self.columns = self.input_features + self.output_features
        else:
            self.columns = list(
                set(self.columns + self.input_features + self.output_features)
            )

        # Enrich metadata with F-DATA specific info
        assert self.metadata is not None
        self.metadata.update(
            {
                "dataset_name": "F-DATA",
                "dataset_version": "1.0",
                "source": "Supercomputer Fugaku (RIKEN, Japan)",
                "time_period": "March 2021 - April 2024",
                "n_jobs": "~24 million",
                "doi": "10.5281/zenodo.11467483",
                "paper_doi": "10.1038/s41597-025-05633-1",
                "files_loaded": self._file_list if self._file_list else "local",
            }
        )

        # If dummy swap parameters. Use a small row cap (not a filter): F-DATA
        # monthly files are a single row group, so filters can't push down and the
        # whole group (~2 GB) would be read. max_rows stops decoding after a few
        # hundred rows, keeping the dummy download tiny and OOM-safe.
        if self.is_dummy:
            self.files = "21_04"
            self.filters = None
            if self.max_rows is None:
                self.max_rows = 200

        # Resolve file list
        if self.data_path is None and self.use_zenodo_url:
            self._file_list = self._parse_files_parameter(self.files)
        elif self.data_path is None:
            raise ValueError(
                "data_path must be provided or use_zenodo_url must be True."
            )
        else:
            self._file_list = None

        self.load_data()
        return self

    def _parse_files_parameter(
        self, files: Optional[Union[str, List[str]]]
    ) -> List[str]:
        """
        Parse the files parameter to get list of file names to load.

        Args:
            files: File specification (None, "all", single file, list, or range)

        Returns:
            List of file names (without .parquet extension)
        """
        if files is None or files == "all":
            return self.AVAILABLE_FILES.copy()

        if isinstance(files, str):
            if ":" in files:
                start, end = files.split(":")
                start = start.replace(".parquet", "")
                end = end.replace(".parquet", "")
                try:
                    start_idx = self.AVAILABLE_FILES.index(start)
                    end_idx = self.AVAILABLE_FILES.index(end)
                    return self.AVAILABLE_FILES[start_idx : end_idx + 1]
                except ValueError as e:
                    raise ValueError(
                        f"Invalid file range: {files}. "
                        f"Both {start} and {end} must be in available files."
                    ) from e
            else:
                file_name = files.replace(".parquet", "")
                if file_name not in self.AVAILABLE_FILES:
                    raise ValueError(
                        f"File {file_name} not found in available files. "
                        f"Available: {self.AVAILABLE_FILES}"
                    )
                return [file_name]

        elif isinstance(files, list):
            file_list = [f.replace(".parquet", "") for f in files]
            invalid_files = [f for f in file_list if f not in self.AVAILABLE_FILES]
            if invalid_files:
                raise ValueError(
                    f"Invalid files: {invalid_files}. Available: {self.AVAILABLE_FILES}"
                )
            return file_list

        else:
            raise TypeError(f"files must be str, list, or None, got {type(files)}")

    def _read_parquet(self, source: Union[str, Path]) -> pd.DataFrame:
        """
        Read one Parquet source (local path or remote URL) honouring ``columns``,
        ``filters`` and ``max_rows``.

        When ``max_rows`` is set, only the first ``max_rows`` rows are decoded via
        ``pyarrow.ParquetFile.iter_batches`` and the read stops early — so a single
        large row group is never fully materialised. Remote URLs are opened through
        fsspec (HTTP range requests), so only the needed byte ranges are fetched.
        When ``max_rows`` is None, this is a plain ``pd.read_parquet`` with column
        projection and predicate pushdown (unchanged behaviour).
        """
        if self.max_rows is None:
            return pd.read_parquet(source, columns=self.columns, filters=self.filters)

        import pyarrow as pa
        import pyarrow.parquet as pq

        source_str = str(source)
        handle = None
        try:
            if source_str.startswith(("http://", "https://")):
                import fsspec

                handle = fsspec.open(source_str).open()
                parquet_file = pq.ParquetFile(handle)
            else:
                parquet_file = pq.ParquetFile(source_str)

            batches = []
            collected = 0
            batch_size = min(self.max_rows, 256)
            for batch in parquet_file.iter_batches(
                batch_size=batch_size, columns=self.columns
            ):
                batches.append(batch)
                collected += batch.num_rows
                if collected >= self.max_rows:
                    break

            if not batches:
                return pd.DataFrame(columns=self.columns)

            table = pa.Table.from_batches(batches).slice(0, self.max_rows)
            return table.to_pandas()
        finally:
            if handle is not None:
                handle.close()

    def load_data(self) -> None:
        """
        Load data from Parquet file(s) (local or remote) and merge them.
        """
        start_time = time.time()

        try:
            if self.use_zenodo_url and self._file_list:
                self.logger.info(
                    f"Loading {len(self._file_list)} F-DATA file(s) from Zenodo: "
                    f"{self._file_list[0]} to {self._file_list[-1]}"
                )

                dfs = []
                for file_name in self._file_list:
                    file_url = f"{self.ZENODO_BASE_URL}/{file_name}.parquet"
                    self.logger.info(f"Downloading {file_name}.parquet...")
                    try:
                        df_chunk = self._read_parquet(file_url)
                        dfs.append(df_chunk)
                        self.logger.info(
                            f"  Loaded {file_name}.parquet: {len(df_chunk)} rows"
                        )
                    except Exception as e:
                        self.logger.error(f"  Failed to load {file_name}.parquet: {e}")
                        raise

                self.logger.info("Merging all dataframes...")
                self._df = pd.concat(dfs, ignore_index=True)
                self.logger.info(f"Merged F-DATA with shape: {self._df.shape}")

            else:
                assert self.data_path is not None
                data_path = Path(self.data_path)

                if data_path.is_dir():
                    self.logger.info(f"Loading F-DATA from directory: {data_path}")
                    parquet_files = sorted(data_path.glob("*.parquet"))
                    parquet_files = [
                        f
                        for f in parquet_files
                        if self._file_list is None or f.stem in self._file_list
                    ]
                    if not parquet_files:
                        raise ValueError(f"No parquet files found in {data_path}")

                    self.logger.info(f"Found {len(parquet_files)} parquet file(s)")
                    dfs = []
                    for file_path in parquet_files:
                        self.logger.info(f"Loading {file_path.name}...")
                        df_chunk = self._read_parquet(file_path)
                        dfs.append(df_chunk)
                        self.logger.info(f"  Loaded: {len(df_chunk)} rows")

                    self._df = pd.concat(dfs, ignore_index=True)
                    self.logger.info(f"Merged F-DATA with shape: {self._df.shape}")

                else:
                    self.logger.info(f"Loading F-DATA from local file: {data_path}")
                    self._df = self._read_parquet(data_path)
                    self.logger.info(f"Loaded F-DATA with shape: {self._df.shape}")

            # Apply preprocessing: each entry is (columns, function, output_feature_name)
            for fn in self.preprocessing_functions or []:
                self._df = fn(self._df)

            # Optionally persist merged data to disk
            if self.download_path:
                download_path = Path(self.download_path)
                download_path.mkdir(parents=True, exist_ok=True)
                out_file = download_path / "f_data.parquet"
                self.logger.info(f"Saving merged F-DATA to: {out_file}")
                self._df.to_parquet(out_file, index=False)
                self.logger.info(f"Saved F-DATA to: {out_file}")

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
            self.logger.error(f"Failed to load {self.__class__.__name__} file(s): {e}")
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
        if idx >= len(self):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self)}"
            )

        assert self._df is not None
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

    def get_statistics_by_partition(self) -> Dict[str, Dict[str, float]]:
        """
        Compute job statistics grouped by partition.

        Returns:
            Dictionary mapping partition name to job statistics.
        """
        if self._df is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")
        if "partition" not in self._df.columns:
            raise ValueError("'partition' column not available in dataset.")

        partition_stats = {}
        for partition in self._df["partition"].unique():
            partition_jobs = self._df[self._df["partition"] == partition]
            partition_stats[str(partition)] = {"n_jobs": float(len(partition_jobs))}

            for col, key in [("duration", "avg_duration"), ("n_nodes", "avg_nodes")]:
                if col in partition_jobs.columns:
                    partition_stats[partition][key] = float(partition_jobs[col].mean())

            if "power_consumption" in partition_jobs.columns:
                partition_stats[partition]["avg_power"] = float(
                    partition_jobs["power_consumption"].mean()
                )

        return partition_stats

    @classmethod
    def get_available_files(cls) -> List[str]:
        """Return a copy of all available monthly file identifiers."""
        return cls.AVAILABLE_FILES.copy()

    @classmethod
    def get_date_range(cls) -> Tuple[str, str]:
        """
        Return the date range covered by the dataset.

        Returns:
            Tuple of (start_date, end_date) in YY_MM format.
        """
        return (cls.AVAILABLE_FILES[0], cls.AVAILABLE_FILES[-1])
