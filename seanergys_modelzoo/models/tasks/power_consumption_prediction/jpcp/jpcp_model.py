import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type, Union
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from pydantic import Field, model_validator
from sentence_transformers import SentenceTransformer
import joblib
from enum import Enum
import numpy as np
import pandas as pd

from seanergys_modelzoo.dataloader.seanergys_dataloader import SeanergysDataloader
from seanergys_modelzoo.datasets.common.seanergys_dataset import SeanergysDataset
from seanergys_modelzoo.datasets.f_data import FDataDataset
from seanergys_modelzoo.datasets.pm100 import PM100Dataset
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
from seanergys_modelzoo.models.common.sklearn_seanergys_model import (
    SeanergysSklearnModel,
)


class Embedding(Enum):
    INT = 0  # Input features are mapped to categorical
    SB = 1  # Input features are encoded with sentence_transformers
    NONE = 2  # Input features are used as is


class JPCP(SeanergysSklearnModel):
    """
    Job Power Consumption Predictor.

    A RandomForestRegressor-backed regression model that predicts the power
    consumption of HPC jobs. All SeanergysSklearnModel capabilities
    (training, prediction, save/load, feature importance) are inherited
    without modification.

    Example usage:
        model = JPCP(model_hyperparameters={"n_estimators": 200, "max_depth": 10})
        model.train(train_loader, val_loader, validation_metrics=[mean_absolute_error])
        predictions = model.predict(test_loader)
        model.save("models/jpcp.joblib")
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
        values.setdefault("model_class", RandomForestRegressor)
        values.setdefault("task_type", SeanergysModelTask.REGRESSION)
        values.setdefault("supported_datasets", [PM100Dataset, FDataDataset])
        values.setdefault(
            "metadata",
            SeanergysModelMetadata(name="JPCP"),
        )

        return values

    @model_validator(mode="after")
    def validate_model(self) -> "JPCP":
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

    @classmethod
    def get_train_config(
        cls, dataset_name: str = "f-data", get_dummy_data: bool = False
    ) -> Dict[
        str,
        Tuple[
            "JPCP",
            Type[SeanergysDataset],
            Tuple[SeanergysDatasetParams, SeanergysDataloaderParams],
        ],
    ]:
        """
        Get training configuration for JPCP.

        """
        config_dict: Dict = {}
        if dataset_name == "f-data":
            model_fdata = cls(  # type: ignore[call-arg]
                embedding_type=Embedding.NONE, model_hyperparameters={"n_jobs": -1}
            )

            model_fdata.logger.info("Training on FData")
            ds_params = SeanergysDatasetParams(
                use_zenodo_url=True,
                is_dummy=get_dummy_data,
                files=["23_12", "24_01"],
                transform=model_fdata.embedding_parsing,
                target_transform=lambda output_features: (
                    output_features[0] / output_features[1]
                ),
                filters=[
                    ("adt", ">=", "2023-12-01"),
                    ("adt", "<=", "2024-01-31"),
                ],
                input_features=["embedding"],
                output_features=["avgpcon", "nnuma"],
            )
            return {
                "f-data": (
                    model_fdata,
                    FDataDataset,
                    (ds_params, SeanergysDataloaderParams(batch_size=1)),
                )
            }

        if dataset_name != "pm100":
            raise Exception("Invalid dataset name. Datasets allowed: 'f-data', 'pm100'")

        model_pm100 = cls(  # type: ignore[call-arg]
            embedding_type=Embedding.INT, model_hyperparameters={"n_jobs": -1}
        )
        model_pm100.logger.info("Training on PM100")

        # FIXED by Mohsen (2026-03-23) — original used model_pm100.transform_embeddings as
        # preprocessing_functions, but transform_embeddings expects a numpy array while the
        # preprocessing API passes a full DataFrame and expects a DataFrame back. It also had
        # a typo: embeding_type (missing 'd') so embedding_type defaulted to Embedding.NONE.
        #
        # Correct approach: a DataFrame→DataFrame function that label-encodes the categorical
        # input columns and adds _cat versions, then stores the encoders on the model instance
        # so they are included in the save/load cycle.
        def _preprocess_pm100(df: pd.DataFrame) -> pd.DataFrame:
            from sklearn.preprocessing import LabelEncoder

            df = df.copy()
            encoders = []
            for col in ["num_nodes_req", "user_id"]:
                le = LabelEncoder()
                df[f"{col}_cat"] = le.fit_transform(df[col].astype(str))
                encoders.append(le)
            model_pm100.embedder = encoders
            return df

        pm100_config = SeanergysDatasetParams(
            use_zenodo_url=True,
            transform=None,
            target_transform=lambda output_features: (
                np.mean(output_features[0]) / output_features[1]
            ),
            preprocessing_functions=[_preprocess_pm100],
            filters=[
                [
                    ("submit_time", ">=", pd.Timestamp("2020-05-01", tz="UTC")),
                    ("submit_time", "<=", pd.Timestamp("2020-09-01", tz="UTC")),
                ]
            ],
            columns=[
                "num_nodes_req",
                "user_id",
                "node_power_consumption",
                "num_nodes_alloc",
            ],
            input_features=["num_nodes_req_cat", "user_id_cat"],
            output_features=["node_power_consumption", "num_nodes_alloc"],
        )
        data_loader_pm100 = SeanergysDataloaderParams(batch_size=1)
        # FIXED by Mohsen (2026-03-23) — Francesco's original line was:
        #   config_dict["pm100"] = (model_pm100, PM100Dataset(pm100_config, data_loader_pm100))
        #
        # Two bugs in that line:
        #   1. PM100Dataset is a Pydantic BaseModel — it does NOT accept positional
        #      constructor arguments. Calling PM100Dataset(pm100_config, data_loader_pm100)
        #      raises: BaseModel.__init__() takes 1 positional argument but 3 were given
        #
        #   2. The tuple contract for get_train_config() is:
        #        (model_instance, DatasetClass, (ds_params, dl_params))
        #      Francesco passed an already-instantiated dataset object instead of the class
        #      and its params separately. The CI pipeline (and create_sample_data.py) both
        #      unpack the tuple as (model, dataset_cls, (ds_params, dl_params)) — passing
        #      an instance here breaks all downstream tooling.
        #
        # Correct form: pass the CLASS and the params tuple, not an instantiated object.
        config_dict["pm100"] = (
            model_pm100,
            PM100Dataset,
            (pm100_config, data_loader_pm100),
        )
        return config_dict

    @classmethod
    def get_test_config(
        cls,
        model: SeanergysModel,
        dataset_name: str = "f-data",
        get_dummy_data: bool = False,
    ) -> Dict[str, Tuple["JPCP", SeanergysDataloader]]:
        """
        Get testing configuration for JPCP. It requires the model and the dataset_name

        """
        assert isinstance(model, JPCP)
        if dataset_name == "f-data":
            model.logger.info("Testing on FData")

            fdata = FDataDataset(
                use_zenodo_url=True,
                is_dummy=get_dummy_data,
                files=["24_02"],
                transform=model.embedding_parsing,
                target_transform=lambda output_features: (
                    output_features[0] / output_features[1]
                ),
                filters=[
                    ("adt", ">=", "2024-02-01"),
                    ("adt", "<=", "2024-02-28"),
                ],
                input_features=["embedding"],
                output_features=["avgpcon", "nnuma"],
            )
            return {"f-data": (model, SeanergysDataloader(dataset=fdata, batch_size=1))}
        elif dataset_name == "pm100":
            # Load the fdata config
            model.logger.info("Testing on PM100")

            pm100 = PM100Dataset(
                use_zenodo_url=True,
                is_dummy=get_dummy_data,
                transform=None,
                target_transform=lambda output_features: (
                    np.mean(output_features[0]) / output_features[1]
                ),
                preprocessing_functions=[model.transform_embeddings],
                filters=[
                    ("submit_time", ">=", pd.Timestamp("2020-09-01", tz="UTC")),
                    ("submit_time", "<=", pd.Timestamp("2020-10-01", tz="UTC")),
                ],
                columns=[
                    "num_nodes_req",
                    "user_id",
                    "node_power_consumption",
                    "num_nodes_alloc",
                ],
                input_features=["num_nodes_req_cat", "user_id_cat"],
                output_features=["node_power_consumption", "num_nodes_alloc"],
            )
            return {"pm100": (model, SeanergysDataloader(dataset=pm100, batch_size=1))}
        else:
            raise Exception("Invalid dataset name. Datasets allowed: 'f-data', 'pm100'")

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
    def load(cls, path: Union[str, Path] = "./saved_model", **kwargs) -> "JPCP":
        """
        Restore a JPCP model from disk.

        Args:
            path: Path to the joblib model file produced by save().

        Returns:
            A fully restored JPCP instance.
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
