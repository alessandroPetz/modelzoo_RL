from enum import Enum
import os
from pathlib import Path
import joblib
from typing import Any, Dict, Optional, Tuple, Type, Union
from seanergys_modelzoo.dataloader.seanergys_dataloader import SeanergysDataloader
from seanergys_modelzoo.datasets.common.seanergys_dataset import SeanergysDataset
from seanergys_modelzoo.datasets.f_data import FDataDataset
from seanergys_modelzoo.models.common.seanergys_configurator import (
    SeanergysDataloaderParams,
    SeanergysDatasetParams,
)
from seanergys_modelzoo.models.common.seanergys_model import (
    SeanergysModel,
    SeanergysModelTask,
)
from seanergys_modelzoo.models.common.seanergys_model_metadata import (
    SeanergysModelMetadata,
)
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
import numpy as np
from pydantic import model_validator, Field
import itertools

from seanergys_modelzoo.models.common.sklearn_seanergys_model import (
    SeanergysSklearnModel,
)


class _DenseLabelXGBClassifier(XGBClassifier):
    """XGBClassifier that accepts labels with gaps (e.g. [0, 1, 3])."""

    def fit(self, X, y, **kwargs):
        y = np.asarray(y).ravel()
        self.label_ids_, dense = np.unique(y, return_inverse=True)
        return super().fit(X, dense, **kwargs)

    def predict(self, X, **kwargs):
        return self.label_ids_[super().predict(X, **kwargs)]


class Embedding(Enum):
    INT = 0  # Input features are mapped to categorical
    SB = 1  # Input features are encoded with sentence_transformers
    NONE = 2  # Input features are used as is


class MACK(SeanergysSklearnModel):
    """
    Class for the MACK framework
    """

    kmeans_flops: Optional[KMeans] = Field(
        default=None, description="The fitted kmeans to cluster the flops values"
    )
    kmeans_memory_bw: Optional[KMeans] = Field(
        default=None,
        description="The fitted kmeans to cluster the memory bandwidth values",
    )
    k_flops: Optional[int] = Field(
        default=3, description="The k value for the kmeans_flops"
    )
    k_memory_bw: Optional[int] = Field(
        default=3, description="The k value for the kmeans_memory_bw"
    )
    embedder: Optional[Any] = Field(
        default=None,
        description="The list containing the embedder instances to encode the input labels.",
    )
    embedding_type: Optional[Embedding] = Field(
        default=Embedding.NONE, description="The embedding strategy to use on the data."
    )
    embedding_weights: Optional[str] = Field(
        default="google/embeddinggemma-300m",
        description="The weights to use for the SentenceTransformers, if embedding_type == Embdedding.SB",
    )

    @model_validator(mode="before")
    @classmethod
    def _inject_defaults(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Lock in the estimator class and task type before Pydantic validation,
        while still allowing model_hyperparameters to be customised by the caller.
        """
        values.setdefault("model_class", _DenseLabelXGBClassifier)
        values.setdefault("task_type", SeanergysModelTask.CLASSIFICATION)
        values.setdefault("supported_datasets", [FDataDataset])
        values.setdefault(
            "metadata",
            SeanergysModelMetadata(name="MACK"),
        )
        return values

    @model_validator(mode="after")
    def validate_model(self) -> "MACK":
        """
        Implement to validate the model
        """
        if self.kmeans_flops is None:
            self.kmeans_flops = KMeans(self.k_flops)
        if self.kmeans_memory_bw is None:
            self.kmeans_memory_bw = KMeans(self.k_memory_bw)
        return self

    def transform_embeddings(self, features: np.ndarray) -> np.ndarray:
        """
        Concatenate feature values into strings and encode with SentenceTransformer.

        Args:
            features: 2D array of shape (n_samples, n_features)
            encoder_weights: str containing the name of the encoder's weights, defaults to all-MiniLM-L6-v2

        Returns:
            Embedding array of shape (n_samples, embedding_dim)
        """
        if self.embedding_type == Embedding.NONE:
            return features
        elif self.embedding_type == Embedding.SB:
            self.embedder = SentenceTransformer(self.embedding_weights)
            sentences = [" ".join(str(val) for val in row) for row in features]
            return self.embedder.encode(sentences)
        else:
            self.embedder = [LabelEncoder() for _ in range(features.shape[1])]
            return np.column_stack(
                [le.fit_transform(features[:, i]) for i, le in enumerate(self.embedder)]
            )

    def embedding_parsing(self, embedding: np.ndarray) -> np.ndarray:
        """
        Parses the precomputed embedding vector to a suitable format for the model
        """

        arr = np.zeros((384,))
        arr[:] = np.array(embedding[0])
        return arr

    def transform_kmeans(
        self, flops: np.ndarray, memory_bw: np.ndarray, fit: bool = False
    ) -> list:
        """
        Fit KMeans on flops and memory bandwidth arrays separately,
        order labels by ascending centroid values, and return combined labels.

        Args:
            flops: Array of FLOP values
            memory_bw: Array of memory bandwidth values
            k_flops: Number of clusters for flops KMeans
            k_memory_bw: Number of clusters for memory bandwidth KMeans

        Returns:
            List of aggregated labels combining flops and memory bandwidth labels
        """

        assert self.kmeans_flops is not None, "kmeans_flops not initialised"
        assert self.k_flops is not None
        flops_labels = (
            self.kmeans_flops.fit_predict(flops.reshape(-1, 1))
            if fit
            else self.kmeans_flops.predict(flops.reshape(-1, 1))
        )
        sorted_centroid_indices_flops = np.argsort(
            self.kmeans_flops.cluster_centers_.flatten()
        )
        label_mapping_flops = np.empty_like(sorted_centroid_indices_flops)
        label_mapping_flops[sorted_centroid_indices_flops] = np.arange(self.k_flops)
        flops_labels = label_mapping_flops[flops_labels]

        assert self.kmeans_memory_bw is not None, "kmeans_memory_bw not initialised"
        assert self.k_memory_bw is not None
        memory_bw_labels = (
            self.kmeans_memory_bw.fit_predict(memory_bw.reshape(-1, 1))
            if fit
            else self.kmeans_memory_bw.predict(memory_bw.reshape(-1, 1))
        )
        sorted_centroid_indices_memory_bw = np.argsort(
            self.kmeans_memory_bw.cluster_centers_.flatten()
        )
        label_mapping_memory_bw = np.empty_like(sorted_centroid_indices_memory_bw)
        label_mapping_memory_bw[sorted_centroid_indices_memory_bw] = np.arange(
            self.k_memory_bw
        )
        memory_bw_labels = label_mapping_memory_bw[memory_bw_labels]

        combined_labels = list(zip(flops_labels.tolist(), memory_bw_labels.tolist()))

        # Create label mapping
        label_mapping = {
            label_key: i
            for i, label_key in enumerate(
                itertools.product(
                    (j for j in range(self.k_flops)),
                    (k for k in range(self.k_memory_bw)),
                )
            )
        }

        return [label_mapping[label_key] for label_key in combined_labels]

    def parse_job_script(self, script: str):
        out = ""
        for e in script.split("\n"):
            if (e.startswith("#")) or (e == "\n") or (e == " "):
                continue
            idx = -1
            if "#" in e:
                idx = e.index("#")
            out += e[:idx].replace("\n", "")
        return out

    # SUGGESTION (Mohsen Seyedkazemi Ardebili, 2026-09-08): same three problems as
    # in get_train_config below — mem_compute_bound / memory_bw do not exist, and
    # df["pclass"] is overwritten. See the full SUGGESTION block on get_train_config.
    @classmethod
    def run_dummy_validation(cls):
        """
        Run dummy validation for MACK on FData
        """

        model_fdata = cls(
            embedding_type=Embedding.SB,
            k_flops=2,
            k_memory_bw=2,
            model_hyperparameters={"n_jobs": -1},
        )
        model_fdata.logger.info("Validation on FData")

        # Define pclass creation
        def preprocess_fdata(df):
            df["pclass"] = model_fdata.transform_kmeans(
                df.flops.values, df.memory_bw.values, fit=True
            )
            return df

        fdata = FDataDataset(
            use_zenodo_url=True,
            files="21_04",
            transform=model_fdata.embedding_parsing,
            preprocessing_functions=[preprocess_fdata],
            filters=[[("adt", "<=", "2021-04-02")]],
            input_features=["embedding"],
            output_features=["pclass"],
        )
        data_loader_fdata = SeanergysDataloader(dataset=fdata, batch_size=1)
        model_fdata.train(train_data_loader=data_loader_fdata)

    # =====================================================================
    # SUGGESTION — Mohsen Seyedkazemi Ardebili (CI & test maintainer), 2026-09-08
    # Mohsen suggests changing the three points below, to improve the quality of
    # the code. NOT APPLIED: this model is owned by its authors, and picking the
    # target column is a modelling decision, not a test fix. Tracked in GitLab
    # issue #10. Until it is fixed, test_full_pipeline_offline[MACK-f-data] FAILS
    # in CI on purpose — the failure is left visible rather than masked with xfail.
    #
    # Reproduce:
    #   CI_SKIP_ZENODO=1 PYTHONPATH=. pytest tests/integration/ -q \
    #       -k "full_pipeline_offline and MACK"
    #
    # 1) output_features=["mem_compute_bound"] — that column exists in NO F-DATA
    #    file. The dataset projects columns = input_features + output_features at
    #    read time, so the pair dies at construction, before training:
    #        ValidationError: No match for FieldRef.Name(mem_compute_bound)
    #    The published column holding the memory/compute-bound label is `pclass`
    #    (observed values e.g. 'memory-bound'). SUGGESTED: output_features=["pclass"].
    #
    # 2) preprocess_fdata() reads `df.memory_bw`, which does not exist either.
    #    F-DATA's memory-bandwidth column is `mbwidth`.
    #    SUGGESTED: df.mbwidth.values.
    #
    # 3) preprocess_fdata() writes its KMeans result into df["pclass"], i.e. it
    #    OVERWRITES the real label column with a (flops_class, mbw_class) tuple.
    #    A tuple target also breaks the dataloader: torch's default_collate
    #    rejects the resulting object array ("found object"). SUGGESTED: write the
    #    derived class to its own column (e.g. `kmeans_pclass`), encode `pclass`
    #    to integer ids for XGBoost, and widen `columns` to
    #    ["embedding", "pclass", "flops", "mbwidth"] so preprocessing still gets
    #    the columns it needs.
    #
    # A working patch for all three was measured green end-to-end (train ->
    # predict -> save -> load) against the committed fixtures; it is attached to
    # issue #10 for the authors to accept, adapt, or reject.
    # =====================================================================
    @classmethod
    def get_train_config(
        cls, dataset_name: str = "f-data", get_dummy_data: bool = False
    ) -> Dict[
        str,
        Tuple[
            "MACK",
            Type[SeanergysDataset],
            Tuple[SeanergysDatasetParams, SeanergysDataloaderParams],
        ],
    ]:
        """
        Get training configuration for MACK on FDATA.

        """
        if dataset_name == "f-data":
            model_fdata = cls(  # type: ignore[call-arg]
                embedding_type=Embedding.SB,
                k_flops=2,
                k_memory_bw=2,
                model_hyperparameters={"n_jobs": -1},
            )

            def preprocess_fdata(df):
                df["pclass"] = model_fdata.transform_kmeans(
                    df.flops.values, df.mbwidth.values, fit=True
                )
                return df

            ds_params = SeanergysDatasetParams(
                use_zenodo_url=True,
                files=["23_12", "24_01"],
                is_dummy=get_dummy_data,
                transform=model_fdata.embedding_parsing,
                preprocessing_functions=[preprocess_fdata],
                filters=[
                    ("adt", ">=", "2023-12-01"),
                    ("adt", "<=", "2024-01-31"),
                ],
                input_features=["embedding"],
                columns=["flops", "mbwidth"],
                output_features=["pclass"],
            )
            return {
                "f-data": (
                    model_fdata,
                    FDataDataset,
                    (ds_params, SeanergysDataloaderParams(batch_size=1)),
                )
            }
        else:
            raise Exception("Invalid dataset name. Datasets allowed: 'f-data'")

    # SUGGESTION (Mohsen Seyedkazemi Ardebili, 2026-09-08): same three problems as
    # in get_train_config above. Note also that this testing config refits the
    # KMeans on the test data (fit=True), which leaks test statistics into the
    # clustering — Mohsen suggests fit=False here once the columns are corrected.
    @classmethod
    def get_test_config(
        cls,
        model: SeanergysModel,
        dataset_name: str = "f-data",
        get_dummy_data: bool = False,
    ) -> Dict[str, Tuple["MACK", SeanergysDataloader]]:
        """
        Get testing configuration for MACK on FDATA. It requires the model and the dataset_name

        """
        assert isinstance(model, MACK)
        if dataset_name == "f-data":
            model.logger.info("Validation on FData")

            # Define pclass creation
            def preprocess_fdata(df):
                df["pclass"] = model.transform_kmeans(
                    df.flops.values, df.memory_bw.values, fit=True
                )
                return df

            fdata = FDataDataset(
                use_zenodo_url=True,
                files=["24_02"],
                is_dummy=get_dummy_data,
                transform=model.embedding_parsing,
                preprocessing_functions=[preprocess_fdata],
                filters=[
                    ("adt", ">=", "2023-12-01"),
                    ("adt", "<=", "2024-02-01"),
                ],
                input_features=["embedding"],
                output_features=["mem_compute_bound"],
            )
            data_loader_fdata = SeanergysDataloader(dataset=fdata, batch_size=1)
            return {"f-data": (model, data_loader_fdata)}
        else:
            raise Exception("Invalid dataset name. Datasets allowed: 'f-data'")

    def save(
        self, path: Union[str, Path] = "./saved_model", compress: int = 3, **kwargs
    ) -> bool:
        """
        Persist the estimator, kmeans, metadata, and supplementary info to disk.

        Args:
            path:     Destination path for the joblib model file.
            compress: joblib compression level (0–9).

        Returns:
            True on success, False on failure.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        try:
            # Save the kmeans
            joblib.dump(
                self.kmeans_flops,
                f"{os.path.join(str(path), 'kmeans_flops.joblib')}",
                compress=compress,
            )
            self.logger.info(f"Kmeans flops saved to {path}")

            joblib.dump(
                self.kmeans_memory_bw,
                f"{os.path.join(str(path), 'kmeans_memory_bw.joblib')}",
                compress=compress,
            )
            self.logger.info(f"Kmeans memory_bw saved to {path}")

            super().save(path=path, compress=compress)
            return True

        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            return False

    @classmethod
    def load(cls, path: Union[str, Path] = "./saved_model", **kwargs) -> "MACK":
        """
        Restore a MACK model from disk.

        Args:
            path: Path to the joblib model file produced by save().

        Returns:
            A fully restored MACK instance.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        try:
            obj = super().load(path=path, **kwargs)
            assert isinstance(obj, cls), (
                f"Expected {cls.__name__}, got {type(obj).__name__}"
            )
            obj.kmeans_flops = joblib.load(
                f"{os.path.join(str(path), 'kmeans_flops.joblib')}"
            )
            obj.kmeans_memory_bw = joblib.load(
                f"{os.path.join(str(path), 'kmeans_memory_bw.joblib')}"
            )
            return obj

        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}") from e
