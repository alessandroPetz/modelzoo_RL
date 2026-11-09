import tarfile
from pathlib import Path
from typing import Optional, Union, List, Tuple, Any, Dict
import requests
import pyarrow.parquet as pq
import numpy as np
import bisect
from cachetools import LRUCache
from pydantic import Field, model_validator, PrivateAttr

from seanergys_modelzoo.datasets.common.seanergys_dataset import SeanergysDataset


class M100Dataset(SeanergysDataset):
    """
    Dataset class for the Marconi 100 (M100) supercomputer at Cineca, Italy.

    This class provides efficient, lazy access to large telemetry datasets stored
    as Parquet files on Zenodo.

    Supports:
    - Raw monthly data (filtered by yy_mm, plugins, metrics)
    - Aggregated 15-minute data (filtered by nodes or racks)

    Features:
    - Automatic download and extraction of dataset archives
    - Lazy loading of Parquet row groups (no full dataset in memory)
    - Global schema alignment across heterogeneous files
    - LRU caching for Parquet files and row groups

    Data is accessed row-by-row via indexing without loading the full dataset.
    """

    ZENODO_BASE_URL: str = r"https://zenodo.org/records/"

    AVAILABLE_FILES_PER_RECORD: List[dict] = [
        {
            "7588815": [
                "20-03",
                "20-04",
                "20-05",
                "20-06",
                "20-07",
                "20-08",
                "20-09",
                "20-10",
                "20-11",
                "20-12",
            ]
        },
        {"7589131": ["21-01", "21-02", "21-03", "21-04", "21-05", "21-06"]},
        {"7589320": ["21-07", "21-08", "21-09"]},
        {"7589630": ["21-10", "21-11", "21-12"]},
        {"7589942": ["22-01", "22-02"]},
        {"7590061": ["22-03"]},
        {"7590308": ["22-04"]},
        {"7590547": ["22-05"]},
        {"7590555": ["22-06"]},
        {"7590565": ["22-07"]},
        {"7590574": ["22-08"]},
        {"7590583": ["22-09"]},
        {"7541722": [str(i) for i in range(50)]},  # 15-min aggregated
    ]

    # User input (Pydantic Fields)
    yy_mm: Optional[Union[str, List[str]]] = Field(
        default=None,
        description="YY-MM for raw data; ignored for aggregated. Can be single (e.g., '20-03'), list, range, or 'all' for all files.",
    )
    aggregated: bool = Field(
        default=False, description="Use 15-min aggregated data if True"
    )
    nodes: Optional[Union[int, List[int], str]] = None
    plugins: Optional[List[str]] = None
    metrics: Optional[List[str]] = None
    dataset_path: Optional[Union[str, Path]] = Field(default="./data")

    is_dummy: bool = Field(
        default=False,
        description="If True, loads a minimal subset of data for testing/debugging.",
    )

    # Internal states
    _metadata: Dict[str, Any] = PrivateAttr(default_factory=dict)
    _nodes: List[int] = PrivateAttr(default_factory=list)
    _dataset_path: Path = PrivateAttr()
    _file_list: List[Tuple[str, str]] = PrivateAttr(default_factory=list)

    _chunks: List[Tuple[Path, int, int, int]] = PrivateAttr(default_factory=list)
    _chunk_starts: List[int] = PrivateAttr(default_factory=list)

    _col_names: List[str] = PrivateAttr(default_factory=list)

    _total_rows: int = PrivateAttr(default=0)

    # LRU Cache for files
    _file_cache: LRUCache[str, pq.ParquetFile] = PrivateAttr(
        default_factory=lambda: LRUCache(maxsize=4)
    )

    _rg_cache: LRUCache[Tuple[str, int], Dict[str, np.ndarray]] = PrivateAttr(
        default_factory=lambda: LRUCache(maxsize=16)
    )

    @model_validator(mode="after")
    def validate_model(self) -> "M100Dataset":
        """
        Validate and initialize the dataset after model creation.

        This method:
        - Validates required parameters depending on mode (raw vs aggregated)
        - Normalizes and validates node selections (for aggregated mode)
        - Builds the list of dataset files to load
        - Populates dataset metadata
        - Triggers dataset indexing via `build_index()`

        Returns:
            M100Dataset: The validated and initialized dataset instance.

        Raises:
            ValueError: If required parameters are missing or invalid.
        """

        self._dataset_path = (
            Path(self.dataset_path) if self.dataset_path else Path("./data")
        )
        self._metadata = dict(getattr(self, "metadata", {}) or {})

        if self.is_dummy:
            self.logger.warning("M100 running in dummy mode")

            self.aggregated = True
            self.yy_mm = None
            self.plugins = None
            self.metrics = None

            self._nodes = list(range(20))
            self._file_list = self._build_file_list(None, aggregated=True)

            self.build_index()

            return self

        # normalize nodes
        parsed_nodes = self._parse_nodes() or []
        self._nodes = sorted(set(parsed_nodes))

        # Check for aggregated data
        if self.aggregated:
            self.yy_mm = None
            self.plugins = None
            self.metrics = None

            if len(self._nodes) == 0:
                raise ValueError("Aggregated data requires `nodes`, range, or 'all'")

            if self.yy_mm or self.plugins or self.metrics:
                self.logger.info(
                    "Filters (yy_mm/plugins/metrics) are ignored for aggregated data"
                )

            max_node = self._get_total_nodes() - 1

            invalid = [n for n in self._nodes if n < 0 or n > max_node]
            if invalid:
                raise ValueError(f"Invalid node IDs: {invalid}")
        else:
            if not self.yy_mm or not self.plugins:
                raise ValueError("Raw data requires `yy_mm` and `plugins`")
            if self._nodes:
                self.logger.info("Node filters are ignored for raw data")

        self._file_list = self._build_file_list(self.yy_mm, aggregated=self.aggregated)
        self._metadata.update(
            {
                "dataset_name": "M100",
                "dataset_version": "1.0",
                "source": "Marconi 100 - Cineca, Italy",
                "records_loaded": [rid for rid, _ in self._file_list],
                "files_loaded": [fname for _, fname in self._file_list],
            }
        )
        self.build_index()
        return self

    def _build_file_list(
        self, yy_mm: Optional[Union[str, List[str]]], aggregated: bool
    ) -> List[Tuple[str, str]]:
        """
        Convert user input into a list of dataset files to load.

        For raw data:
            - Accepts a single month (e.g., "22-02")
            - A list of months
            - A range ("22-01:22-03")
            - "all" for all available months

        For aggregated data:
            - Maps selected nodes to rack-level files

        Args:
            yy_mm (str | list[str] | None): Time selection for raw data.
            aggregated (bool): Whether aggregated mode is enabled.

        Returns:
            List[Tuple[str, str]]: List of (record_id, file_name) pairs.

        Raises:
            TypeError: If input format is invalid.
        """
        if aggregated:
            record_id = "7541722"
            if self._nodes:
                # Map nodes → racks
                racks = sorted(set(n // 20 for n in self._nodes))
                files = [str(r) for r in racks]
            else:
                # fallback: all racks
                files = self.AVAILABLE_FILES_PER_RECORD[-1][record_id]
            return [(record_id, f) for f in files]

        # Raw data path (yy_mm can be single, list, range, or all)
        file_mapping = {}
        for record_dict in self.AVAILABLE_FILES_PER_RECORD:
            for rid, rec_files in record_dict.items():
                if rid == "7541722":
                    continue  # skip aggregated
                for f in rec_files:
                    file_mapping[f] = rid

        # Handle 'all'
        if yy_mm is None or yy_mm == "all":
            return [(rid, f) for f, rid in file_mapping.items()]

        # Range, single, or list
        if isinstance(yy_mm, str):
            if ":" in yy_mm:
                start, end = yy_mm.split(":")
                ordered = [
                    (rid, f)
                    for record_dict in self.AVAILABLE_FILES_PER_RECORD
                    for rid, rec_files in record_dict.items()
                    if rid != "7541722"
                    for f in rec_files
                ]
                start_idx = next(i for i, (_, f) in enumerate(ordered) if f == start)
                end_idx = next(i for i, (_, f) in enumerate(ordered) if f == end)
                return ordered[start_idx : end_idx + 1]
            else:
                return [(file_mapping[yy_mm], yy_mm)]
        if isinstance(yy_mm, list):
            return [(file_mapping[f], f) for f in yy_mm]

        raise TypeError("yy_mm must be str, list, or None")

    def _parse_nodes(self) -> Optional[List[int]]:
        """
        Normalize the `nodes` parameter into a list of node indices.

        Supported formats:
            - Single integer (e.g., 5)
            - String range (e.g., "5:10")
            - List of integers and/or ranges (e.g., [1, "3:5"])
            - "all" to select all nodes

        Returns:
            List[int] | None: Sorted list of unique node indices, or None if not set.

        Raises:
            TypeError: If the input format is invalid.
        """
        if self.nodes is None:
            return None

        total_nodes = self._get_total_nodes()

        # "all"
        if self.nodes == "all":
            return list(range(total_nodes))

        # single int
        if isinstance(self.nodes, int):
            return [self.nodes]

        # range "start:end"
        if isinstance(self.nodes, str):
            if ":" in self.nodes:
                start, end = map(int, self.nodes.split(":"))
                return list(range(start, end + 1))
            else:
                return [int(self.nodes)]

        # list
        if isinstance(self.nodes, list):
            result = []
            for item in self.nodes:
                if isinstance(item, int):
                    result.append(item)
                elif isinstance(item, str):
                    if ":" in item:
                        start, end = map(int, item.split(":"))
                        result.extend(range(start, end + 1))
                    else:
                        result.append(int(item))
            return sorted(set(result))

        raise TypeError("nodes must be int, list, range string, or 'all'")

    def _get_total_nodes(self) -> int:
        """
        Returns the total number of nodes in the aggregated dataset.
        """
        record_id = "7541722"
        total_racks = len(self.AVAILABLE_FILES_PER_RECORD[-1][record_id])
        nodes_per_rack = 20
        return total_racks * nodes_per_rack

    def _safe_extract(self, tar: tarfile.TarFile, path: Path) -> None:
        """
        Safely extract a tar archive to a destination directory.

        Prevents path traversal attacks by ensuring all extracted files
        remain within the target directory.

        Args:
            tar (tarfile.TarFile): Open tar archive.
            path (Path): Destination directory.

        Raises:
            Exception: If a file attempts to escape the target directory.
        """
        for member in tar.getmembers():
            member_path = path / member.name
            if not str(member_path.resolve()).startswith(str(path.resolve())):
                raise Exception("Unsafe tar file")
        tar.extractall(path)

    def _download_and_extract(self, rid: str, fname: str) -> None:
        """
        Download and extract a dataset archive from Zenodo.

        The archive is downloaded as a `.tar` file, extracted into the dataset
        directory, and then deleted to save disk space.

        Args:
            rid (str): Zenodo record ID.
            fname (str): File name within the record.

        Raises:
            requests.HTTPError: If the download fails.
            Exception: If extraction fails or is unsafe.
        """
        dest_dir = self._dataset_path / rid / fname
        dest_dir.mkdir(parents=True, exist_ok=True)

        tarball_url = f"{self.ZENODO_BASE_URL}{rid}/files/{fname}.tar"
        tarball_path = dest_dir.parent / f"{fname}.tar"

        if not tarball_path.exists():
            self.logger.info(f"Downloading {tarball_url}...")
            resp = requests.get(tarball_url, stream=True)
            resp.raise_for_status()
            with open(tarball_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            self.logger.info(f"Downloaded {tarball_path}")

        # Extract tarball
        self.logger.info(f"Extracting {tarball_path}...")

        with tarfile.open(tarball_path, "r:") as tar:
            self._safe_extract(tar, dest_dir)
        self.logger.info(f"Extracted to {dest_dir}")

        # Delete tarball to save space
        tarball_path.unlink()
        self.logger.info(f"Deleted tarball {tarball_path}")

    def _get_file(self, path: Path) -> pq.ParquetFile:
        path_str = str(path.resolve())

        if path_str in self._file_cache:
            return self._file_cache[path_str]

        pf = pq.ParquetFile(path_str)
        self._file_cache[path_str] = pf
        return pf

    def _get_row_group_data(self, path: Path, rg: int) -> Dict[str, np.ndarray]:

        key: Tuple[str, int] = (str(path.resolve()), rg)

        if key in self._rg_cache:
            return self._rg_cache[key]

        pf = self._get_file(path)

        table = pf.read_row_group(rg)
        available_cols = set(table.schema.names)

        arrays: Dict[str, np.ndarray] = {}

        for col in self._col_names:
            if col in available_cols:
                arrays[col] = table[col].to_numpy()
            else:
                arrays[col] = np.full(table.num_rows, np.nan, dtype=np.float64)

        self._rg_cache[key] = arrays
        return arrays

    def build_index(self) -> None:
        """
        Scan dataset files and build an index for lazy access.

        This method:
        - Downloads missing dataset files if necessary
        - Collects all relevant Parquet files based on filters
        - Builds a global schema across all files
        - Splits data into row-group chunks
        - Computes total dataset size

        The resulting index enables efficient random access via `__getitem__`
        without loading the full dataset into memory.

        Raises:
            ValueError: If no files match the provided filters.
        """
        self.logger.debug("Scanning parquet files...")

        files: List[Path] = []

        base_root = Path(self._dataset_path)
        for rid, fname in self._file_list:
            base_path = base_root / rid / fname

            if not base_path.exists():
                self._download_and_extract(rid, fname)

            if rid == "7541722":
                parquet_files = base_path.glob("*.parquet")

                for f in parquet_files:
                    node = int(f.stem)

                    if self._nodes and node not in self._nodes:
                        continue
                    files.append(f.resolve())
            else:
                parquet_files = base_path.glob(
                    "year_month=*/plugin=*/metric=*/*.parquet"
                )

                for f in parquet_files:
                    parts = f.parts
                    plugin = next(
                        p.split("=")[1] for p in parts if p.startswith("plugin=")
                    )
                    metric = next(
                        p.split("=")[1] for p in parts if p.startswith("metric=")
                    )

                    if self.plugins and plugin not in self.plugins:
                        continue
                    if self.metrics and metric not in self.metrics:
                        continue

                    files.append(f.resolve())
        self.logger.debug(f"Found {len(files)} parquet files")

        if not files:
            raise ValueError("No parquet files found with given filters")

        self.logger.debug("Building global schema...")
        all_columns = set()

        for f in files:
            pf = self._get_file(f)
            all_columns.update(pf.schema_arrow.names)
        all_columns.discard("timestamp")

        all_columns = {c for c in all_columns if not c.startswith("__index_level_")}

        self._col_names = sorted(all_columns)

        self.logger.debug(f"Global schema size = {len(self._col_names)}")
        self.logger.debug(f"Sample columns = {self._col_names[:10]}")

        self._chunks = []
        self._chunk_starts = []
        total = 0

        for f in files:
            pf = self._get_file(f)

            for rg in range(pf.num_row_groups):
                n = pf.metadata.row_group(rg).num_rows

                self._chunks.append((f.resolve(), rg, total, total + n))
                self._chunk_starts.append(total)

                total += n

        self._total_rows = total

        self.logger.debug(f"Chunks Size = {len(self._chunks)}. Below are first 5:")
        for c in self._chunks[:5]:
            self.logger.debug(c)

    def __getitem__(self, idx: int) -> np.ndarray:
        """
        Retrieve a single row from the dataset by global index.

        The method locates the corresponding Parquet file and row group,
        loads the necessary data, and returns a feature vector.

        Args:
            idx (int): Global row index.
            return_tensor (bool, optional): If True, return a PyTorch tensor.
            dtype (str, optional): Torch dtype name (e.g., "float32").

        Returns:
            np.ndarray | torch.Tensor: Feature vector for the selected row.

        Raises:
            RuntimeError: If the dataset is empty.
            IndexError: If the index is out of range.
            ValueError: If an invalid dtype is provided.
        """

        if self._total_rows == 0:
            raise RuntimeError("Dataset is empty. Call build_index() first.")

        if idx < 0 or idx >= len(self):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self)}"
            )

        if not self._chunk_starts:
            raise RuntimeError("Index not built. Call build_index() first.")

        pos = bisect.bisect_right(self._chunk_starts, idx) - 1
        path, rg, start, end = self._chunks[pos]

        local_idx = idx - start
        arrays = self._get_row_group_data(path, rg)

        values = [arrays[c][local_idx] for c in self._col_names]
        input_data = np.asarray(values, dtype=np.float64)

        return input_data

    def __len__(self) -> int:
        """
        Return the total number of rows in the dataset.

        Returns:
            int: Total number of rows.
        """
        return self._total_rows


if __name__ == "__main__":
    # python -m seanergys_modelzoo.datasets.m100
    dataset = M100Dataset(
        yy_mm="22-02",
        dataset_path="./data",
        plugins=["ipmi_pub"],
        metrics=["total_power"],
        aggregated=True,
        nodes="0:19",
    )

    # dataset = M100Dataset(
    #     is_dummy = True
    # )

    print(f"Number of chunks: {len(dataset._chunks)}")

    print(f"Dataset loaded with {len(dataset)} rows.")

    for i in range(min(5, len(dataset))):
        x = dataset[i]
        print(f"Row {i}: Item shape = {x.shape}, Item Type = {type(x)}, Item = {x}")
