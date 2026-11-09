from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Type, Union

import joblib
import numpy as np
import torch
import torch.nn as nn
from pydantic import Field, model_validator
from sklearn.metrics import roc_auc_score

from seanergys_modelzoo.models.common.seanergys_model import (
    SeanergysModel,
    SeanergysModelTask,
)
from seanergys_modelzoo.models.common.seanergys_model_metadata import (
    SeanergysModelMetadata,
)

from seanergys_modelzoo.dataloader.seanergys_dataloader import SeanergysDataloader
from seanergys_modelzoo.datasets.common.seanergys_dataset import SeanergysDataset
from seanergys_modelzoo.datasets.m100 import M100Dataset
from seanergys_modelzoo.models.common.seanergys_configurator import (
    SeanergysDataloaderParams,
    SeanergysDatasetParams,
)


class RUAD(SeanergysModel):
    """
    RUAD model
    """

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    learning_rate: float = 0.01
    sequence_length: int = 20

    # Default node used only by the ModelZoo CI.
    # AIProduction can instantiate RUAD with the actual node_id.
    DEFAULT_CI_NODE_ID: ClassVar[int] = 36

    node_id: int = Field(
        DEFAULT_CI_NODE_ID,
        description=(
            "Single M100 node used for training. "
            "Feature dimensionality is inferred from the data "
            "associated with this node."
        ),
    )

    model: Optional[nn.Module] = None

    _input_dim: Optional[int] = None

    _x_min = None
    _x_max = None
    _epochs = 3

    @model_validator(mode="before")
    @classmethod
    def _inject_defaults(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Lock in the estimator class and task type before Pydantic validation,
        while still allowing model_hyperparameters to be customised by the caller.
        """
        values.setdefault("model_class", RUADNetwork)
        values.setdefault("task_type", SeanergysModelTask.CLASSIFICATION)
        values.setdefault("supported_datasets", [M100Dataset])
        values.setdefault(
            "metadata",
            SeanergysModelMetadata(name="RUAD"),
        )

        return values

    @model_validator(mode="after")
    def validate_model(self) -> "RUAD":
        """
        Implement to validate the model
        """
        if self.node_id < 0 or self.node_id >= 980:
            raise ValueError("node_id must be in the range [0, 979]")

        return self

    def build_model(self) -> None:
        if self._input_dim is None:
            raise RuntimeError("Cannot build RUAD model: input dimension is unknown.")

        try:
            self.model = RUADNetwork(input_dim=self._input_dim).to(self.device)

        except Exception as e:
            self.logger.error(f"Error building RUAD model: {e}")
            raise

        self.logger.info(
            f"Built RUAD model for node_id={self.node_id} with input_dim={self._input_dim}"
        )

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)

        dataset = M100Dataset(
            aggregated=True,
            nodes=self.node_id,
            is_dummy=False,
        )

        sample = np.asarray(dataset[0], dtype=np.float32)

        self._input_dim = sample.shape[0] - 1

        self.build_model()

    def train(
        self,
        train_data_loader,
        val_data_loader=None,
        training_loss=None,
        validation_metrics=None,
        **kwargs,
    ) -> Dict[str, Any]:

        if self.model is None:
            raise RuntimeError("Model has not been built.")

        train_dataset = RUADSequenceDataset(
            base_dataset=train_data_loader.dataset,
            sequence_length=self.sequence_length,
            split="train",
        )

        self._x_min = train_dataset._x_min
        self._x_max = train_dataset._x_max

        train_data_loader = SeanergysDataloader(
            dataset=train_dataset, batch_size=1, shuffle=True
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        loss_fn = torch.nn.MSELoss()

        self.model.train()

        for epoch in range(self._epochs):
            epoch_loss = 0.0

            for x, y, _ in train_data_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                pred = self.model(x)

                loss = loss_fn(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            epoch_loss = epoch_loss / len(train_data_loader)

            self.logger.info(f"Epoch {epoch}: loss={epoch_loss}")

        self.is_trained = True
        return {"loss": [epoch_loss]}

    def predict(
        self,
        data_loader: SeanergysDataloader,
        **kwargs,
    ) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not built")

        self.model.eval()
        preds = []

        with torch.no_grad():
            for batch in data_loader:
                # safely extract x
                if isinstance(batch, (tuple, list)):
                    x = batch[0]
                else:
                    x = batch

                x = x.to(self.device)

                pred = self.model(x)

                preds.append(pred.cpu().numpy())

        return np.concatenate(preds)

    def evaluate(
        self,
        data_loader: SeanergysDataloader,
        metrics: Optional[List[Callable]] = None,
        **kwargs,
    ) -> List[Dict[str, float]]:

        if self.model is None:
            raise RuntimeError("Model not built")

        self.model.eval()

        score_batches: List[np.ndarray] = []
        label_batches: List[np.ndarray] = []

        with torch.no_grad():
            for x, y, label in data_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                pred = self.model(x)

                score = torch.mean((pred - y) ** 2, dim=1)

                score_batches.append(score.cpu().numpy())
                label_batches.append(label.cpu().numpy())

            scores = np.concatenate(score_batches)
            labels = np.concatenate(label_batches)

        auc = roc_auc_score(labels, scores)

        return [
            {
                "roc_auc": float(auc),
            }
        ]

    @classmethod
    def get_train_config(
        cls,
        dataset_name: str = "M100",
        get_dummy_data: bool = False,
    ) -> Dict[
        str,
        Tuple[
            "RUAD",
            Type[SeanergysDataset],
            Tuple[SeanergysDatasetParams, SeanergysDataloaderParams],
        ],
    ]:
        """
        Get the training configuration for RUAD on M100.

        The ModelZoo CI uses DEFAULT_CI_NODE_ID as a representative node.
        For real training, AIProduction can instantiate RUAD directly with
        the required node_id and override the dataset configuration accordingly.
        """

        if dataset_name != "M100":
            raise ValueError("RUAD only supports M100")

        node_id = cls.DEFAULT_CI_NODE_ID

        model_m100 = cls(
            node_id=node_id,
            task_type=SeanergysModelTask.CLASSIFICATION,
        )

        ds_params = SeanergysDatasetParams(
            aggregated=True,
            nodes=node_id,
            is_dummy=get_dummy_data,
        )

        return {
            "M100": (
                model_m100,
                M100Dataset,
                (
                    ds_params,
                    SeanergysDataloaderParams(batch_size=1),
                ),
            )
        }

    @classmethod
    def get_test_config(
        cls,
        model: SeanergysModel,
        dataset_name: str = "M100",
        get_dummy_data: bool = False,
    ) -> Dict[str, Tuple["RUAD", SeanergysDataloader]]:

        if dataset_name != "M100":
            raise ValueError("RUAD only supports M100")

        if not isinstance(model, RUAD):
            raise TypeError("RUAD.get_test_config requires a RUAD model")

        if model.model is None:
            raise RuntimeError("Model not initialized. Train or build first.")

        node_id = model.node_id

        model.logger.info(f"Testing RUAD on M100 node {node_id}")

        base_dataset = M100Dataset(
            aggregated=True,
            nodes=node_id,
            is_dummy=get_dummy_data,
        )

        test_dataset = RUADSequenceDataset(
            base_dataset=base_dataset,
            sequence_length=model.sequence_length,
            split="test",
            x_min=model._x_min,
            x_max=model._x_max,
        )

        test_data_loader_m100 = SeanergysDataloader(
            dataset=test_dataset,
            batch_size=1,
            shuffle=False,
        )

        return {
            "M100": (
                model,
                test_data_loader_m100,
            )
        }

    def save(
        self,
        path: Union[str, Path],
        **kwargs: Any,
    ) -> bool:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self.model is None:
            raise RuntimeError("Cannot save uninitialized model")

        try:
            torch.save(self.model.state_dict(), path / f"ruad_{self.node_id}.pt")

            joblib.dump(
                {
                    "node_id": self.node_id,
                    "input_dim": self._input_dim,
                    "x_min": self._x_min,
                    "x_max": self._x_max,
                    "sequence_length": self.sequence_length,
                },
                path / f"config_{self.node_id}.joblib",
            )
            return True
        except Exception as e:
            self.logger.error(f"Save failed: {e}")
            return False

    @classmethod
    def load(cls, path: Union[str, Path] = "./saved_model", **kwargs) -> "RUAD":
        path = Path(path)
        node_id = kwargs.pop("node_id", None)

        if node_id is None:
            configs = list(path.glob("config_*.joblib"))

            if not configs:
                raise FileNotFoundError(f"No RUAD config found in: {path}")

            if len(configs) > 1:
                raise ValueError(
                    "'node_id' is required when multiple RUAD configs exist."
                )

            config = joblib.load(configs[0])
            node_id = config["node_id"]
        config_path = path / f"config_{node_id}.joblib"
        model_path = path / f"ruad_{node_id}.pt"

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        config = joblib.load(config_path)

        obj = cls(
            node_id=config["node_id"],
            task_type=SeanergysModelTask.CLASSIFICATION,
        )

        input_dim = config.get("input_dim")

        if input_dim is None:
            raise ValueError("Saved configuration missing input_dim")

        obj._input_dim = input_dim

        obj._x_min = config["x_min"]
        obj._x_max = config["x_max"]

        obj.sequence_length = config.get("sequence_length", 20)

        obj.build_model()

        if obj.model is None:
            raise RuntimeError("Failed to build RUAD model")

        obj.model.load_state_dict(
            torch.load(
                model_path,
                map_location=obj.device,
            )
        )

        obj.is_trained = True

        return obj


class RUADEncoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()

        self.lstm1 = nn.LSTM(input_size=input_dim, hidden_size=16, batch_first=True)

        self.lstm2 = nn.LSTM(input_size=16, hidden_size=8, batch_first=True)

    def forward(self, x):
        bsz = x.size(0)
        device = x.device

        h0 = torch.zeros(1, bsz, 16, device=device)
        c0 = torch.zeros(1, bsz, 16, device=device)
        x, _ = self.lstm1(x, (h0, c0))

        h0 = torch.zeros(1, bsz, 8, device=device)
        c0 = torch.zeros(1, bsz, 8, device=device)
        x, _ = self.lstm2(x, (h0, c0))

        return x  # (B, T, 8)


class RUADNetwork(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()

        self.encoder = RUADEncoder(input_dim=input_dim)

        self.fc1 = nn.Linear(8, 16)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(16, input_dim)

    def forward(self, x):
        x = self.encoder(x)

        x = x[:, -1, :]

        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)

        return x


class RUADSequenceDataset(SeanergysDataset):
    def __init__(
        self,
        base_dataset,
        sequence_length: int,
        split: str,
        label_col: str = "value",
        x_min=None,
        x_max=None,
    ):
        super().__init__()

        self.sequence_length = sequence_length
        self.split = split

        self.label_col = label_col

        self._x_min = x_min
        self._x_max = x_max

        # LOAD RAW DATA
        raw: np.ndarray = np.array(
            [
                np.array(base_dataset[i], dtype=np.float32)
                for i in range(len(base_dataset))
            ],
            dtype=np.float32,
        )

        raw = np.nan_to_num(
            raw,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        col_names = base_dataset._col_names
        label_idx = col_names.index(label_col)

        X = np.delete(raw, label_idx, axis=1)
        y_raw = raw[:, label_idx]

        # BINARY LABELS (NAGIOS)
        y = (y_raw != 0).astype(np.float32)

        # STRICT SPLIT (NO OVERLAP)
        split_idx = int(len(X) * 0.8)

        if split == "train":
            X = X[:split_idx]
            y = y[:split_idx]

        elif split == "test":
            X = X[split_idx:]
            y = y[split_idx:]

        else:
            raise ValueError("split must be 'train' or 'test'")

        # NORMALIZATION (fit only on train)
        if split == "train":
            self._x_min = X.min(axis=0)
            self._x_max = X.max(axis=0)

        X = (X - self._x_min) / (self._x_max - self._x_min + 1e-8)

        # SLIDING WINDOW BUILD
        X_seq = []
        y_seq = []
        labels = []

        for i in range(len(X) - sequence_length):
            X_seq.append(X[i : i + sequence_length])
            y_seq.append(X[i + sequence_length])
            labels.append(y[i + sequence_length])

        self.X_seq = torch.tensor(
            np.array(X_seq),
            dtype=torch.float32,
        )
        self.y_seq = torch.tensor(
            np.array(y_seq),
            dtype=torch.float32,
        )
        self.labels = torch.tensor(
            np.array(labels),
            dtype=torch.float32,
        )

    def __len__(self):
        return len(self.X_seq)

    def __getitem__(self, idx):
        return self.X_seq[idx], self.y_seq[idx], self.labels[idx]
