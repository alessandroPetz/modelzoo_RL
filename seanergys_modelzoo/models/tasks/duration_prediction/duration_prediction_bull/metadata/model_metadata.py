"""
Metadata container for duration prediction models.

Provides a structured and serializable representation of model
configuration, validation performance, and feature schema for
reproducibility and inference consistency.
"""

from typing import Dict, List, Any


class DurationModelMetadata:
    """
    Container for metadata associated with a trained DurationModel.

    This class stores all information required to:
    - Reproduce the model configuration
    - Track validation performance
    - Rebuild the feature pipeline at inference time

    Args:
        config (Dict[str, Any]): Model configuration used during training
        val_rmse (List[float]): Validation RMSE scores for ensemble members
        feature_schema (Any): Description of input features
        emb_cols (List[str]): Names of embedding feature columns
        num_cols (List[str]): Names of structured (non-embedding) feature columns

    Notes:
        - Intended for serialization (e.g., JSON) alongside trained models.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        val_rmse: List[float],
        feature_schema: Any,
        emb_cols: List[str],
        num_cols: List[str],
    ):
        self.config = config
        self.val_rmse = val_rmse
        self.feature_schema = feature_schema
        self.emb_cols = emb_cols
        self.num_cols = num_cols

    def to_dict(self):
        """
        Convert metadata to a serializable dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of the metadata.
        """
        return {
            "config": self.config,
            "val_rmse": self.val_rmse,
            "feature_schema": self.feature_schema,
            "emb_cols": self.emb_cols,
            "num_cols": self.num_cols,
        }
