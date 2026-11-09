import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type, Union
from sklearn.ensemble import RandomForestClassifier
from pydantic import model_validator, Field
from enum import Enum
from sklearn.preprocessing import LabelEncoder
from seanergys_modelzoo.models.common.seanergys_configurator import (
    SeanergysDataloaderParams,
    SeanergysDatasetParams,
)
from sentence_transformers import SentenceTransformer
import joblib
import numpy as np

from seanergys_modelzoo.dataloader.seanergys_dataloader import SeanergysDataloader
from seanergys_modelzoo.datasets.common.seanergys_dataset import SeanergysDataset
from seanergys_modelzoo.datasets.f_data import FDataDataset
from seanergys_modelzoo.models.common.seanergys_model import (
    SeanergysModel,
    SeanergysModelTask,
)
from seanergys_modelzoo.models.common.seanergys_model_metadata import (
    SeanergysModelMetadata,
)
from seanergys_modelzoo.models.common.sklearn_seanergys_model import (
    SeanergysSklearnModel,
)


class Embedding(Enum):
    INT = 0  # Input features are mapped to categorical
    SB = 1  # Input features are encoded with sentence_transformers
    NONE = 2  # Input features are used as is


class MCBound(SeanergysSklearnModel):
    """
    Memory-Consumption Bound classifier.

    A RandomForestClassifier-backed classification model. All
    SeanergysSklearnModel capabilities (training, prediction,
    save/load, feature importance, probability prediction) are
    inherited without modification.

    Example usage:
        model = MCBound(model_hyperparameters={"n_estimators": 150, "max_depth": 8})
        model.train(train_loader, val_loader, validation_metrics=[accuracy_score])
        predictions = model.predict(test_loader)
        probabilities = model.predict(test_loader, return_proba=True)
        model.save("models/mcbound.joblib")
    """

    embedder: Optional[Any] = Field(
        default=None,
        description="The list containing the embedder instances to encode the input labels.",
    )
    embedding_type: Optional[Embedding] = Field(
        default=Embedding.NONE, description="The embedding strategy to use on the data."
    )
    embedding_weights: Optional[str] = Field(
        default="all-MiniLM-L6-v2",
        description="The weights to use for the SentenceTransformers, if embedding_type == Embdedding.SB",
    )

    @model_validator(mode="before")
    @classmethod
    def _inject_defaults(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Lock in the estimator class and task type before Pydantic validation,
        while still allowing model_hyperparameters to be customised by the caller.
        """
        values.setdefault("model_class", RandomForestClassifier)
        values.setdefault("task_type", SeanergysModelTask.CLASSIFICATION)
        values.setdefault("supported_datasets", [FDataDataset])
        values.setdefault(
            "metadata",
            SeanergysModelMetadata(name="MCBound"),
        )
        return values

    @model_validator(mode="after")
    def validate_model(self) -> "MCBound":
        """
        Implement to validate the model
        """
        return self

    def transform_embeddings(
        self, features: np.ndarray, fit: bool = False
    ) -> np.ndarray:
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
                [
                    le.fit_transform(features[:, i])
                    if fit
                    else le.transform(features[:, i])
                    for i, le in enumerate(self.embedder)
                ]
            )

    def embedding_parsing(self, embedding: np.ndarray) -> np.ndarray:
        """
        Parses the precomputed embedding vector to a suitable format for the model
        """

        arr = np.zeros((384,))
        arr[:] = np.array(embedding[0])
        return arr

    # =====================================================================
    # SUGGESTION — Mohsen Seyedkazemi Ardebili (CI & test maintainer), 2026-09-08
    # Mohsen suggests changing the target column below, to improve the quality of
    # the code. NOT APPLIED: this model is owned by its authors, and picking the
    # target is a modelling decision. Tracked in GitLab issue #10. Until it is
    # fixed, test_full_pipeline_offline[MCBound-f-data] FAILS in CI on purpose —
    # the failure is left visible rather than masked with xfail.
    #
    # Reproduce:
    #   CI_SKIP_ZENODO=1 PYTHONPATH=. pytest tests/integration/ -q \
    #       -k "full_pipeline_offline and MCBound"
    #
    # output_features=["mem_compute_bound"] — that column exists in NO F-DATA
    # file. The dataset projects columns = input_features + output_features at
    # read time, so the pair dies at construction, before training:
    #     ValidationError: No match for FieldRef.Name(mem_compute_bound)
    # The published column holding the memory/compute-bound label is `pclass`.
    # SUGGESTED: output_features=["pclass"].
    #
    # `pclass` ships as a STRING ('memory-bound', ...), so it also needs encoding:
    # RandomForestClassifier needs a numeric target, and torch's default_collate
    # rejects the object array a string target produces ("found object"). Mohsen
    # suggests a preprocessing_functions entry that label-encodes `pclass` with an
    # encoder fitted once and persisted in save()/load(), so this training config
    # and get_test_config below agree on the class ids.
    #
    # A working patch was measured green end-to-end (train -> predict -> save ->
    # load) against the committed fixtures; it is attached to issue #10 for the
    # authors to accept, adapt, or reject.
    # =====================================================================
    @classmethod
    def get_train_config(
        cls, dataset_name: str = "f-data", get_dummy_data: bool = False
    ) -> Dict[
        str,
        Tuple[
            "MCBound",
            Type[SeanergysDataset],
            Tuple[SeanergysDatasetParams, SeanergysDataloaderParams],
        ],
    ]:
        """
        Get training configuration for MCBound on FDATA.

        """
        if dataset_name == "f-data":
            model_fdata = cls(  # type: ignore[call-arg]
                embedding_type=Embedding.NONE, model_hyperparameters={"n_jobs": -1}
            )

            ds_params = SeanergysDatasetParams(
                use_zenodo_url=True,
                is_dummy=get_dummy_data,
                files=["23_12", "24_01"],
                transform=model_fdata.embedding_parsing,
                target_transform=lambda pclass: 0 if pclass == "memory-bound" else 1,
                filters=[
                    ("adt", ">=", "2023-12-01"),
                    ("adt", "<=", "2024-01-31"),
                ],
                input_features=["embedding"],
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

    # SUGGESTION (Mohsen Seyedkazemi Ardebili, 2026-09-08): same target-column
    # problem as in get_train_config above. See that SUGGESTION block.
    @classmethod
    def get_test_config(
        cls,
        model: SeanergysModel,
        dataset_name: str = "f-data",
        get_dummy_data: bool = False,
    ) -> Dict[str, Tuple["MCBound", SeanergysDataloader]]:
        """
        Get testing configuration for MCBound on FDATA.

        """
        assert isinstance(model, MCBound)
        if dataset_name == "f-data":
            model.logger.info("Testing on FData")
            fdata = FDataDataset(
                use_zenodo_url=True,
                is_dummy=get_dummy_data,
                files=["24_02"],
                transform=model.embedding_parsing,
                target_transform=lambda pclass: 0 if pclass == "memory-bound" else 1,
                filters=[
                    ("adt", ">=", "2023-02-01"),
                    ("adt", "<=", "2024-02-28"),
                ],
                input_features=["embedding"],
                output_features=["pclass"],
            )
            data_loader_fdata = SeanergysDataloader(dataset=fdata, batch_size=1)
            return {"f-data": (model, data_loader_fdata)}
        else:
            raise Exception("Invalid dataset name. Datasets allowed: 'f-data'")

    def save(
        self, path: Union[str, Path] = "./saved_model", compress: int = 3, **kwargs
    ) -> bool:
        """
        Persist the estimator, embedder, metadata, and supplementary info to disk.

        Args:
            path:     Destination path for the joblib model file.
            compress: joblib compression level (0–9).

        Returns:
            True on success, False on failure.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        try:
            # Save the label encoder
            joblib.dump(
                self.embedder,
                f"{os.path.join(str(path), 'embedder.joblib')}",
                compress=compress,
            )
            self.logger.info(f"Embedder saved to {path}")
            super().save(path=path, compress=compress)
            return True

        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            return False

    @classmethod
    def load(cls, path: Union[str, Path] = "./saved_model", **kwargs) -> "MCBound":
        """
        Restore a MCBound model from disk.

        Args:
            path: Path to the joblib model file produced by save().

        Returns:
            A fully restored MCBound instance.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        try:
            obj = super().load(path=path, **kwargs)
            assert isinstance(obj, cls)
            obj.embedder = joblib.load(f"{os.path.join(str(path), 'embedder.joblib')}")
            return obj

        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}") from e
