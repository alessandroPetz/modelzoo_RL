"""
PyTorch neural network pipeline for model training and inference.

Provides a unified wrapper for building, training, validating, and
serializing neural network models (MLP, CNN) using PyTorch, with
support for both regression and classification tasks.

Ensures compatibility with scikit-learn interfaces and Model Zoo
integration.
"""

import gc
import json
import logging
import os
from typing import Optional, Tuple, cast

import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import accuracy_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, TensorDataset

from ..set_seed import set_seed
from .nn_models import CNNTorch, MLPTorchModel, MLPTorchModelWithDropout

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

gc.collect()
set_seed(42)


class NNTorch(BaseEstimator, RegressorMixin):
    """
    Wrapper class for building, training, validating, and saving/loading
    PyTorch neural network models (MLP or CNN).
    """

    def __init__(
        self,
        task_type: str = "classification",
        nn_key: str = "MLP",
        nn_params: Optional[dict] = None,
        n_epochs: int = 10,
        batch_size: int = 100,
        learning_rate: float = 0.001,
        cost_function: torch.nn.Module = torch.nn.MSELoss(),
        metrics: list = ["accuracy"],
        verbose: bool = True,
        is_log_target: bool = False,
    ):
        """
        Initialize the neural network wrapper.

        Args:
            task_type (str): Task type ("classification" or "regression")
            nn_key (str): Model type identifier ("MLP", "CNN", "MLP_Dropout")
            nn_params (dict): Parameters defining the neural network architecture
            n_epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            learning_rate (float): Learning rate for optimizer
            cost_function (torch.nn.Module): Loss function
            metrics (list): List of evaluation metrics
            verbose (bool): Whether to print training progress
            is_log_target (bool): Whether the target is log-transformed
        """
        self.task_type = task_type  # either "classification" or "regression"
        self.val_accuracies: list[float] = []
        self.nn_params = nn_params
        self.nn_key = nn_key
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.cost_function = cost_function
        self.verbose = verbose
        self.metrics = metrics
        self.nn_model = self.create_model()
        self.is_log_target = is_log_target
        # Add history dictionary to track training metrics
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

    def create_model(self) -> torch.nn.Module:
        """
        Create the neural network model based on the selected architecture

        Returns:
            torch.nn.Module: Instantiated PyTorch model
        """
        if self.nn_params is None:
            raise ValueError("nn_params must not be None")

        nn_params = self.nn_params

        if self.nn_key == "MLP":
            return MLPTorchModel(
                input_shape=nn_params["input_shape"],
                output_shape=nn_params["output_shape"],
                in_nn=nn_params["in_nn"],
                activations=nn_params["activations"],
            )

        elif self.nn_key == "CNN":
            return CNNTorch(
                input_shape=nn_params["input_shape"],
                output_shape=nn_params["output_shape"],
                filters=nn_params["filters"],
                in_nn=nn_params["in_nn"],
                activations=nn_params["activations"],
            )
        elif self.nn_key == "MLP_Dropout":
            return MLPTorchModelWithDropout(
                input_shape=nn_params["input_shape"],
                output_shape=nn_params["output_shape"],
                in_nn=nn_params["in_nn"],
                activations=nn_params["activations"],
                dropout=nn_params.get("dropout", 0.1),
            )
        else:
            raise ValueError(f"Unsupported nn_key: {self.nn_key}")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        continue_training: bool = True,
        dataset: Optional[Dataset] = None,
        model_path: Optional[str] = None,
    ) -> "NNTorch":
        """
        Train the neural network model.

        Args:
            X (np.ndarray): Training input data of shape (n_samples, n_features)
            y (np.ndarray): Target values
            validation_data (Optional[Tuple[np.ndarray, np.ndarray]]): Validation dataset
            continue_training (bool): If False, reinitialize the model before training
            dataset (Optional[Dataset]): Optional dataset (unused in current implementation)
            model_path (Optional[str]): Optional path to save the trained model

        Returns:
            NNTorch: Trained model instance.
        """
        if self.task_type == "regression":
            y = y.reshape(-1, 1)

        if self.verbose:
            print(self.nn_model)

        if not continue_training:
            self.nn_model = self.create_model()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.is_available():
            print("Using GPU")
        else:
            print("Using CPU")

        self.nn_model.to(device)

        # Prepare dataset
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        y_t = torch.tensor(y, dtype=torch.float32, device=device)
        dataset = TensorDataset(X_t, y_t)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer_choice = AdamW(
            self.nn_model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4,  # L2 regularization
        )

        scheduler = ReduceLROnPlateau(
            optimizer_choice,
            mode="min",
            factor=0.5,  # reduce LR by 50%
            patience=3,  # wait 3 epochs without improvement
            threshold=0.001,
            min_lr=1e-5,
        )

        log.info("Optimizer initialized.")

        val_dataloader = None
        if validation_data is not None:
            x_val = torch.tensor(validation_data[0], dtype=torch.float32, device=device)
            y_val = torch.tensor(validation_data[1], dtype=torch.float32, device=device)
            if self.task_type == "regression" and y_val.ndim == 1:
                y_val = y_val.unsqueeze(1)
            val_dataloader = DataLoader(
                TensorDataset(x_val, y_val),
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
            )

        best_val_loss = float("inf")
        best_state_dict = None
        patience = 10
        patience_counter = 0
        for epoch in range(self.n_epochs):
            self.nn_model.train()
            train_loss = 0.0
            num_samples = 0

            if self.task_type == "classification":
                correct_preds = 0
            else:
                squared_error_sum = 0.0

            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer_choice.zero_grad()
                outputs = self.nn_model(inputs)

                loss = self.cost_function(outputs, targets)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.nn_model.parameters(), max_norm=1.0)

                optimizer_choice.step()

                train_loss += loss.item() * targets.size(0)
                num_samples += targets.size(0)

                if self.task_type == "classification":
                    _, preds = torch.max(outputs, 1)
                    correct_preds += (preds == targets).sum().item()
                else:
                    if self.is_log_target:
                        outputs_real = torch.expm1(outputs)
                        targets_real = torch.expm1(targets)
                    else:
                        outputs_real = outputs
                        targets_real = targets
                    squared_error_sum += (
                        ((outputs_real - targets_real) ** 2).sum().item()
                    )

            train_loss /= num_samples
            if self.task_type == "classification":
                train_metric = correct_preds / num_samples
            else:
                train_metric = np.sqrt(squared_error_sum / num_samples)

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_metric)

            if validation_data is not None:
                assert val_dataloader is not None
                val_loss, val_metric = self.validate_model(
                    self.nn_model, val_dataloader, self.cost_function, device
                )
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_metric)
                self.val_accuracies.append(val_metric)
                scheduler.step(val_loss)

                # -------------------------
                # EARLY STOPPING
                # -------------------------
                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    best_state_dict = {
                        k: v.detach().cpu().clone()
                        for k, v in self.nn_model.state_dict().items()
                    }
                    patience_counter = 0
                else:
                    patience_counter += 1

                # Track learning rate
                current_lr = optimizer_choice.param_groups[0]["lr"]
                if "lr" not in self.history:
                    self.history["lr"] = []
                self.history["lr"].append(current_lr)
            if validation_data is not None and patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

            if self.verbose:
                msg = f"Epoch {epoch + 1}/{self.n_epochs} "
                msg += f"Train Loss: {train_loss:.4f} Train Metric: {train_metric:.4f}"
                if validation_data is not None:
                    msg += f" | Val Loss: {val_loss:.4f} Val Metric: {val_metric:.4f}"
                    msg += f" | LR: {current_lr:.6f}"
                print(msg)
        # Load best model state if early stopping was triggered
        # -------------------------
        # Restore best model
        # -------------------------
        if best_state_dict is not None:
            self.nn_model.load_state_dict(best_state_dict)
            print(f"Restored best model with val_loss={best_val_loss:.4f}")

        return self

    def validate_model(
        self,
        nn_model: torch.nn.Module,
        val_dataloader: DataLoader,
        cost_function: torch.nn.Module,
        device: torch.device,
    ) -> Tuple[float, float]:
        """
        Evaluate the model on a validation dataset.

        Args:
            nn_model (torch.nn.Module): Trained model
            val_dataloader (DataLoader): Validation data loader
            cost_function (torch.nn.Module): Loss function
            device (torch.device): Execution device

        Returns:
            Tuple[float, float]:
                - validation loss
                - validation metric (accuracy or RMSE)
        """
        nn_model.eval()
        val_loss = 0.0
        num_samples = 0

        if self.task_type == "classification":
            correct_preds = 0
        else:
            squared_error_sum = 0.0

        with torch.no_grad():
            for inputs, targets in val_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = nn_model(inputs)

                batch_loss = cost_function(outputs, targets).item()
                val_loss += batch_loss * targets.size(0)
                num_samples += targets.size(0)

                if self.task_type == "classification":
                    _, preds = torch.max(outputs, 1)
                    correct_preds += (preds == targets).sum().item()
                else:
                    if self.is_log_target:
                        outputs_real = torch.expm1(outputs)
                        targets_real = torch.expm1(targets)
                    else:
                        outputs_real = outputs
                        targets_real = targets

                    squared_error_sum += (
                        ((outputs_real - targets_real) ** 2).sum().item()
                    )

        val_loss /= num_samples
        if self.task_type == "classification":
            val_metric = correct_preds / num_samples
        else:
            val_metric = np.sqrt(squared_error_sum / num_samples)

        return val_loss, val_metric

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate model performance.

        Args:
            X (np.ndarray): Input features.
            y (np.ndarray): True labels or values.

        Returns:
            float: Accuracy score (classification) or metric based on predictions.
        """
        # Compute and return the score
        return accuracy_score(y, self.predict(X))

    def get_params(self, deep: bool = True):
        return {
            "task_type": self.task_type,
            "nn_key": self.nn_key,
            "nn_params": self.nn_params,
            "n_epochs": self.n_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "cost_function": self.cost_function,
            "metrics": self.metrics,
            "verbose": self.verbose,
        }

    def set_params(self, **params: dict) -> "NNTorch":
        """
        Set model parameters (scikit-learn compatibility)

        Args:
            **params (dict): Parameters to set

        Returns:
            NNTorch: Updated model instance
        """
        for parameter, value in params.items():
            setattr(self, parameter, value)
        return self

    def save(self, model_path: str, model_name: str) -> None:
        """
        Save model weights and configuration to disk

        Args:
            model_path (str): Directory to save the model
            model_name (str): Filename (without extension)
        """
        if len(model_name) > 100:
            model_name = model_name[:100]

        torch.save(
            self.nn_model.state_dict(), os.path.join(model_path, model_name + ".pt")
        )
        # save additional params in a json
        params = {
            "nn_params": self.nn_params,
            "n_epochs": self.n_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "cost_function": str(self.cost_function),
            "verbose": self.verbose,
            "metrics": self.metrics,
        }
        with open(os.path.join(model_path, model_name + "_params.json"), "w") as f:
            json.dump(params, f)

    def load(self, model_path: str, model_name: str) -> torch.nn.Module:
        """
        Load model weights and configuration from disk.

        Args:
            model_path (str): Directory containing the model.
            model_name (str): Filename (without extension).

        Returns:
            NNTorch: Loaded model instance.
        """
        # Shorten the model name if it's too long
        if len(model_name) > 100:
            model_name = model_name[:100]
        self.nn_model.load_state_dict(
            torch.load(os.path.join(model_path, model_name + ".pt"))
        )
        # load additional params from a json
        with open(os.path.join(model_path, model_name + "_params.json"), "r") as f:
            params = json.load(f)
        self.nn_params = params["nn_params"]
        self.n_epochs = params["n_epochs"]
        self.batch_size = params["batch_size"]
        self.learning_rate = params["learning_rate"]
        if params["cost_function"] == "MSELoss":
            self.cost_function = torch.nn.MSELoss()
        # add other cost functions as needed
        self.verbose = params["verbose"]
        self.metrics = params["metrics"]

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions with the trained model.

        Args:
            - X (np.ndarray): Input features of shape (n_samples, n_features).

        Returns:
            - np.ndarray: Predicted labels (classification) or values (regression).
        """
        self.nn_model.eval()
        device = next(self.nn_model.parameters()).device

        # Convert to tensor and move to device
        X_t = torch.tensor(X, dtype=torch.float32).to(device)

        with torch.no_grad():
            outputs = self.nn_model(X_t)
            if self.task_type == "classification":
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                _, predicted_classes = torch.max(probabilities, 1)
                return predicted_classes.cpu().numpy()

            elif self.task_type == "regression":
                outputs = outputs.squeeze()

                if self.is_log_target:
                    outputs = torch.expm1(outputs)

                return outputs.cpu().numpy()

            raise ValueError(f"Unsupported task_type: {self.task_type}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Generate class probability predictions (classification only).

        Args:
            X (np.ndarray): Input features.

        Returns:
            np.ndarray: Class probabilities.
        """
        self.nn_model.eval()  # Set the model to evaluation mode

        # Get the device that the model is on
        device = next(self.nn_model.parameters()).device

        # Convert input to tensor if needed
        if not isinstance(X, torch.Tensor):
            X_t = torch.tensor(X, dtype=torch.float32).to(device)
        else:
            X_t = cast(torch.Tensor, X)

        # Get predictions without calculating gradients
        with torch.no_grad():
            outputs = self.nn_model(X_t)

            # Apply softmax to convert logits to probabilities
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

            # Convert to numpy array
            probabilities_np = probabilities.cpu().numpy()

        return probabilities_np
