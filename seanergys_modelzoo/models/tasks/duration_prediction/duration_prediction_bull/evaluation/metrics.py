"""
Evaluation utilities for regression models.

This module provides:
- Standard metric computation for regression tasks (MAE, RMSE, R2, etc.)

Designed for lightweight, reusable evaluation across training,
experiments, and Model Zoo validation pipelines.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict


def create_metrics(y_true: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    """
    Compute common regression evaluation metrics.

    This function calculates standard regression metrics to assess
    prediction performance, including both error-based and relative metrics.

    Metrics computed:
        - MAE  : Mean Absolute Error
        - MSE  : Mean Squared Error
        - RMSE : Root Mean Squared Error
        - R2   : Coefficient of determination
        - MAPE : Mean Absolute Percentage Error (in %)
        - UPP  : Under-Prediction Percentage (percentage of predictions below true values)

    Args:
        y_true (np.ndarray): Ground truth target values of shape (n_samples,)
        preds (np.ndarray): Predicted target values of shape (n_samples,)

    Returns:
        Dict[str, float]: Dictionary containing all computed metrics:
            {
                "MAE": float,
                "MSE": float,
                "RMSE": float,
                "R2": float,
                "MAPE": float,
                "UPP": float
            }
    """
    mae = mean_absolute_error(y_true, preds)
    mse = mean_squared_error(y_true, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, preds)
    mape = np.mean(np.abs((y_true - preds) / np.clip(np.abs(y_true), 1e-8, None))) * 100
    upp = np.mean(preds < y_true) * 100

    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2, "MAPE": mape, "UPP": upp}
