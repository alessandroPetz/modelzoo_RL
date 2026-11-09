"""
Abstract base class for machine learning models.

Defines the interface for all models in the Model Zoo, ensuring
consistent training, prediction, and serialization methods.
"""

from abc import ABC, abstractmethod

import numpy as np


class BaseModel(ABC):
    """
    Abstract base class for all models in the Model Zoo.

    Defines the standard interface for training, inference, and
    serialization, ensuring consistency across different model types.
    """

    name: str
    version: str
    task_type: str  # "regression" | "classification"

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the model on input features and targets.

        Args:
            X (np.ndarray): Input feature matrix.
            y (np.ndarray): Target values.
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions from input features.

        Args:
            X (np.ndarray): Input feature matrix.

        Returns:
            np.ndarray: Predicted values.
        """
        pass

    @abstractmethod
    def save(self, path: str) -> bool:
        """
        Save the model to disk.

        Args:
            path (str): Destination path.

        Returns:
            bool: True if the model was successfully saved.
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str):
        """
        Load a model from disk.

        Args:
            path (str): Path to the saved model.

        Returns:
            BaseModel: Loaded model instance.
        """
        pass
