"""
Duration prediction model based on an ensemble of neural networks.

Provides training, inference, and serialization logic for runtime
prediction on HPC workloads, using an ensemble of candidate models
selected by validation performance.
"""

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

import joblib
import logging
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from seanergys_modelzoo.models.tasks.duration_prediction.duration_prediction_bull.preprocessing.preprocessing import (
    build_preprocessor,
)

from .base_model import BaseModel
from .pytorch_nn_pipeline import NNTorch


def _train_candidate(args):
    """
    Train a single candidate model.

    This function:
    - Instantiates a NNTorch model with given parameters
    - Trains it on (X_train, y_train) with validation
    - Computes validation RMSE

    Args:
        args (Tuple):
            i (int): Index of the model
            params (dict): Parameters to initialize NNTorch
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training targets
            X_val (np.ndarray): Validation features
            y_val (np.ndarray): Validation targets

    Returns:
        Dict:
            {
                "model": trained NNTorch model,
                "rmse": validation RMSE,
                "params": parameters used
            }
    """
    (i, params, X_train, y_train, X_val, y_val) = args

    torch.set_num_threads(1)

    print(f"[Training] Model {i + 1}")

    model = NNTorch(**params)

    model.fit(X_train, y_train, validation_data=(X_val, y_val))

    # validation prediction
    y_pred = model.predict(X_val)

    rmse = np.sqrt(np.mean((y_pred - y_val) ** 2))

    return {"model": model, "rmse": rmse, "params": params}


class DurationModel(BaseModel):
    """
    Duration prediction model using an ensemble of neural networks.

    This class:
    - Trains multiple candidate neural networks (NNTorch)
    - Selects the top-K models based on validation RMSE
    - Retrains selected models on train+validation data
    - Performs weighted ensemble prediction

    Key features:
        - Parallel training (CPU) or sequential (GPU)
        - RMSE-based model selection
        - Weighted ensemble
        - Uncertainty estimation via ensemble variance

    Attributes:
        name (str): Model name.
        version (str): Model version.
        task_type (str): Task type ("regression").
        models (List[NNTorch]): Trained ensemble models.
        metadata (Dict): Training metadata and metrics.
    """

    name: str = "duration_prediction"
    version: str = "v1"
    task_type: str = "regression"

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.models: List[NNTorch] = []
        self.metadata: Dict = {}
        self.preprocessor = None  # DurationPreprocessor()
        self.is_experiment = self.config.get("is_experiment", False)
        self.logger = logging.getLogger("DurationModel")

    # ----------------------------------------------------------
    # ---------------------- TRAINING
    # ----------------------------------------------------------

    def _parallel_train(self, param_grid, X_train, y_train, X_val, y_val):
        """
        Train multiple candidate models in parallel.

        Args:
            param_grid (List[Dict]): List of parameter configurations
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training targets
            X_val (np.ndarray): Validation features
            y_val (np.ndarray): Validation targets

        Returns:
            List[Dict]: List of results from '_train_candidate', each containing:
                - model
                - rmse
                - params
        """
        n_jobs = min(len(param_grid), self.config.get("n_jobs", 2))

        tasks = [
            (i, params, X_train, y_train, X_val, y_val)
            for i, params in enumerate(param_grid)
        ]

        results = []
        errors = 0

        print(f"[Parallel Training] {len(tasks)} models | workers={n_jobs}")

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(_train_candidate, t) for t in tasks]

            for f in as_completed(futures):
                try:
                    results.append(f.result())
                except Exception as e:
                    errors += 1
                    print(f"[Worker ERROR] {e}")

        if errors:
            print(f"[Parallel Training] {errors} failures")

        return results

    def _sequential_train(self, param_grid, X_train, y_train, X_val, y_val):
        """
        Train candidate models sequentially (no parallelism).

        Args:
            param_grid (List[Dict]): List of parameter configurations
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training targets
            X_val (np.ndarray): Validation features
            y_val (np.ndarray): Validation targets

        Returns:
            List[Dict]: List of results from '_train_candidate', each containing:
                - model: trained NNTorch model
                - rmse: validation RMSE
                - params: configuration used
        """
        results = []

        for i, params in enumerate(param_grid):
            results.append(
                _train_candidate((i, params, X_train, y_train, X_val, y_val))
            )

        return results

    def _get_member_predictions(self, X):
        """
        Compute predictions from all ensemble members.

        Returns:
            np.ndarray of shape (n_models, n_samples)
        """
        if self.preprocessor is not None:
            X = self.preprocessor.transform(X)

        preds = np.asarray(
            [model.predict(X) for model in self.models],
            dtype=np.float32,
        )

        return preds

    def _get_weights(self):
        """
        Compute ensemble weights from validation RMSE.

        Returns:
            np.ndarray of shape (n_models,)
        """
        rmse = np.asarray(self.val_rmse, dtype=np.float32)

        weights = 1.0 / (rmse + 1e-8)
        weights /= weights.sum()

        return weights

    def fit(self, X, y, validation_data=None):
        """
        Train the duration prediction ensemble.

        The training pipeline follows these steps:
            1. Preprocess input data (feature engineering + encoding)
            2. Split data into train/validation
            3. Generate candidate model configurations
            4. Train all candidates (parallel on CPU, sequential on GPU)
            5. Select top-K models based on validation RMSE
            6. Retrain selected models on full train + validation data
            7. Store training metadata for reproducibility

        Args:
            X (pd.DataFrame or np.ndarray): Input features.
            y (np.ndarray): Target values.
            validation_data (Tuple[np.ndarray, np.ndarray], optional): Tuple (X_val, y_val).

        Returns:
            DurationModel: Fitted model instance.
        """

        start_time = time.time()
        y = np.asarray(y, dtype=np.float32)
        y = y / 60.0  # Convert to minutes

        if self.preprocessor is not None:
            if not hasattr(X, "columns"):
                raise ValueError(
                    "X must be a pandas DataFrame (required for preprocessing)"
                )

            categorical = self.config.get("categorical_features", [])
            numeric = self.config.get("numeric_features", [])

            self.preprocessor = build_preprocessor(
                categorical=categorical,
                numeric=numeric,
                mode=self.config.get("preprocess_mode", "full"),
                cat_strategy=self.config.get("cat_strategy", "onehot"),
            )

            X = self.preprocessor.fit_transform(X)

        if validation_data is not None:
            X_train, y_train = X, y
            X_val, y_val = validation_data

            y_val = np.asarray(y_val, dtype=np.float32)
            y_val = y_val / 60.0
            if self.preprocessor is not None:
                X_val = self.preprocessor.transform(X_val)
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y,
                test_size=self.config.get("val_size", 0.1),
                random_state=self.config.get("random_state", 42),
            )

        # -----------------------------------------
        # 1. Build candidate configurations
        # -----------------------------------------
        param_grid = self._get_param_grid(X_train.shape[1])

        use_gpu = torch.cuda.is_available()

        if self.is_experiment:
            results = self._parallel_train(param_grid, X_train, y_train, X_val, y_val)

        else:
            # auto mode
            if use_gpu:
                self.logger.info("GPU detected: using sequential GPU mode")
                results = self._sequential_train(
                    param_grid, X_train, y_train, X_val, y_val
                )
            else:
                self.logger.info("No GPU detected: using CPU mode")
                results = self._sequential_train(
                    param_grid, X_train, y_train, X_val, y_val
                )

        # -----------------------------------------
        # 2. Select best models (ensemble)
        # -----------------------------------------
        if not results:
            raise RuntimeError("All candidate models failed during training.")
        results_sorted = sorted(results, key=lambda x: x["rmse"])

        top_k = self.config.get("top_k", 5)

        top_results = results_sorted[:top_k]

        # Keep RMSE for weighting
        self.val_rmse = [r["rmse"] for r in top_results]

        # R2 from validation phase
        self.val_r2 = []
        for r in top_results:
            y_val_pred = r["model"].predict(X_val)
            self.val_r2.append(r2_score(y_val, y_val_pred))

        # Recreate models from scratch
        X_trval = np.vstack([X_train, X_val])
        y_trval = np.concatenate([y_train, y_val])

        fresh_models = []

        for r in top_results:
            new_model = NNTorch(**r["params"])

            new_model.fit(X_trval, y_trval)

            fresh_models.append(new_model)

        self.models = fresh_models
        # -----------------------------------------
        # 3. Metadata logging (AIDAS requirement)
        # -----------------------------------------
        self.metadata = {
            "training_time_sec": time.time() - start_time,
            "n_candidates": len(results),
            "n_selected": len(self.models),
            "best_rmse": results_sorted[0]["rmse"],
            "ensemble_rmse": self.val_rmse,
            "ensemble_r2": self.val_r2,
            "config": self.config,
            "model_name": self.name,
            "version": self.version,
            "timestamp": time.time(),
            "target_unit": "minutes",
            "walltime_unit": "minutes",
        }

        print(f"[Training Done] Best RMSE: {self.metadata['best_rmse']:.4f}")

        return self

    # -----------------------------------------
    # Inference
    # -----------------------------------------
    def predict(self, X):
        """
        Predict target values using a weighted ensemble.

        Each model prediction is weighted by inverse validation RMSE:
            weight_i = 1 / (rmse_i + epsilon)

        Args:
            X (np.ndarray): Input features.

        Returns:
            np.ndarray: Weighted ensemble prediction. Predicted job durations in minutes.

        """

        if not self.models:
            raise RuntimeError("Model not trained")

        preds = self._get_member_predictions(X)
        weights = self._get_weights()
        return np.sum(preds * weights[:, None], axis=0)

    def predict_full(self, X):
        """
        Generate full ensemble prediction output.

        Returns detailed information for each prediction:
            - ensemble mean prediction
            - prediction uncertainty (std across models)
            - individual model predictions
            - best model prediction

        Args:
            X (np.ndarray): Input features.

        Returns:
            Dict:
                {
                    "y_pred": weighted ensemble prediction,
                    "uncertainty": standard deviation across models,
                    "members": all model predictions (n_models, n_samples),
                    "y_pred_best": prediction from best model,
                    "best_model_idx": index of best model
                }
        """

        preds = self._get_member_predictions(X)
        weights = self._get_weights()

        mean_pred = np.sum(preds * weights[:, None], axis=0)
        std_pred = preds.std(axis=0)

        best_idx = np.argmin(self.val_rmse)
        best_pred = preds[best_idx]

        return {
            "y_pred": mean_pred,
            "uncertainty": std_pred,
            "members": preds,
            "y_pred_best": best_pred,
            "best_model_idx": best_idx,
        }

    # -----------------------------------------
    # Confidence
    # -----------------------------------------
    def predict_with_confidence(self, X):
        """
        Predict values with associated uncertainty.

        Args:
            X (np.ndarray): Input features.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - mean prediction (ensemble)
                - uncertainty (standard deviation across models)
        """

        preds = self._get_member_predictions(X)
        weights = self._get_weights()

        mean_pred = np.sum(preds * weights[:, None], axis=0)
        std_pred = preds.std(axis=0)

        return mean_pred, std_pred

    # -----------------------------------------
    # Save / Load
    # -----------------------------------------
    def save(self, path: str | Path, **kwargs) -> bool:
        """
        Save the model to disk.

        Args:
            path (str | Path): Destination directory.

        Returns:
            bool: True if the model was successfully saved.
        """
        try:
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)

            joblib.dump(self, path / "model.joblib")

            return True

        except Exception as e:
            self.logger.error(f"Failed to save model to '{path}': {e}")
            return False

    @classmethod
    def load(cls, path: str | Path, **kwargs):
        """
        Load a saved model from disk.

        Args:
            path (str | Path): Path to the saved model directory.

        Returns:
            DurationModel: Loaded model instance.
        """
        path = Path(path)

        return joblib.load(path / "model.joblib")

    # -----------------------------------------
    # Param grid
    # -----------------------------------------
    def _get_param_grid(self, input_dim: int) -> List[Dict]:
        """
        Define a fixed set of candidate model configurations.

        Generates a reproducible set of neural network architectures and
        training parameters used for model selection.

        Args:
            input_dim (int): Number of input features.

        Returns:
            List[Dict]: List of parameter dictionaries for NNTorch.
        """

        base = {
            "task_type": "regression",
            "metrics": "rmse",
            "verbose": False,
            "is_log_target": False,
            "batch_size": 64,
            "learning_rate": 0.001,
        }

        # -----------------------------------------
        # Manual configs
        # -----------------------------------------
        manual_configs = [
            dict(
                layers=[100, 200, 100],
                act=["tanh", "tanh", "relu"],
                epochs=50,
                loss=nn.MSELoss(),
            ),
            dict(
                layers=[120, 240, 120],
                act=["tanh", "tanh", "relu"],
                epochs=50,
                loss=nn.MSELoss(),
            ),
            dict(
                layers=[80, 160, 80],
                act=["tanh", "tanh", "relu"],
                epochs=50,
                loss=nn.MSELoss(),
            ),
            dict(
                layers=[100, 300, 100],
                act=["tanh", "relu", "relu"],
                epochs=50,
                loss=nn.MSELoss(),
            ),
            dict(
                layers=[150, 200, 150],
                act=["tanh", "tanh", "relu"],
                epochs=50,
                loss=nn.MSELoss(),
            ),
            # dict(layers=[100,200,100], act=["tanh","tanh","relu"], epochs=50, loss=nn.HuberLoss(delta=5.0)),
            # dict(layers=[120,240,120], act=["tanh","tanh","relu"], epochs=50, loss=nn.HuberLoss(delta=5.0)),
            dict(
                layers=[80, 160, 80],
                act=["tanh", "tanh", "relu"],
                epochs=20,
                loss=nn.HuberLoss(delta=5.0),
            ),
            dict(
                layers=[128, 128, 64],
                act=["relu", "relu", "relu"],
                epochs=50,
                loss=nn.HuberLoss(delta=1.0),
            ),
            dict(
                layers=[256, 128, 64],
                act=["relu", "relu", "relu"],
                epochs=50,
                loss=nn.HuberLoss(delta=1.0),
            ),
        ]

        configs = []

        for cfg in manual_configs:
            configs.append(
                {
                    **base,
                    "nn_key": "MLP",
                    "n_epochs": cfg["epochs"],
                    "cost_function": cfg["loss"],
                    "nn_params": {
                        "in_nn": cfg["layers"],
                        "activations": cfg["act"],
                        "input_shape": input_dim,
                        "output_shape": 1,
                    },
                }
            )

        # -----------------------------------------
        # Deduplicate
        # -----------------------------------------
        def _freeze(c):
            nn = c["nn_params"]
            return (
                tuple(nn["in_nn"]),
                tuple(nn["activations"]),
                c["n_epochs"],
                type(c["cost_function"]).__name__,
            )

        seen = set()
        final_configs = []

        for c in configs:
            key = _freeze(c)
            if key not in seen:
                seen.add(key)
                final_configs.append(c)

        print(f"[Model Grid] {len(final_configs)} configs")

        return final_configs
