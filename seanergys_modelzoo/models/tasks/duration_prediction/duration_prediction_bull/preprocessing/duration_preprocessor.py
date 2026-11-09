import numpy as np
import pandas as pd


class DurationPreprocessor:
    """
    Preprocessor for DurationModel.

    - Handles feature engineering
    - Ensures numeric dtype
    - Keeps schema consistent between train & inference
    """

    def __init__(self):
        self.emb_cols = None
        self.num_cols = None
        self.fitted = False

    def fit(self, df: pd.DataFrame):
        df = df.copy()

        # --- embeddings ---
        self.emb_cols = [c for c in df.columns if c.startswith("emb_")]
        if not self.emb_cols:
            raise ValueError("No emb_* columns found")

        # --- numeric base features ---
        base_num = ["cnumr", "nnumr", "elpl"]
        for col in base_num:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # engineered features (defined but computed later)
        self.num_cols = base_num + [
            "inv_elpl",
            "cnumr_per_node",
            "elpl_per_node",
            "elpl_per_core",
            "cores_per_node",
            "total_compute",
            "node_compute",
        ]

        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")

        df = df.copy()

        # --- base transforms ---
        df["elpl"] = df["elpl"] / 60.0

        eps = 1e-6

        df["elpl"] = np.log1p(df["elpl"])
        df["cnumr"] = np.log1p(df["cnumr"])
        df["nnumr"] = np.log1p(df["nnumr"])

        # --- engineered ---
        df["inv_elpl"] = np.log1p(1.0 / (np.expm1(df["elpl"]) + eps))

        df["cnumr_per_node"] = np.log1p(
            np.expm1(df["cnumr"]) / (np.expm1(df["nnumr"]) + eps)
        )

        df["elpl_per_node"] = np.log1p(
            np.expm1(df["elpl"]) / (np.expm1(df["nnumr"]) + eps)
        )

        df["elpl_per_core"] = np.log1p(
            np.expm1(df["elpl"]) / (np.expm1(df["cnumr"]) + eps)
        )

        df["cores_per_node"] = np.log1p(
            np.expm1(df["cnumr"]) / (np.expm1(df["nnumr"]) + eps)
        )

        df["total_compute"] = np.log1p(np.expm1(df["cnumr"]) * np.expm1(df["elpl"]))

        df["node_compute"] = np.log1p(np.expm1(df["nnumr"]) * np.expm1(df["elpl"]))

        # --- build arrays ---
        X_emb = df[self.emb_cols].values.astype(np.float32)
        X_num = df[self.num_cols].values.astype(np.float32)

        X = np.hstack([X_emb, X_num]).astype(np.float32)

        # safety check
        if X.dtype == object:
            raise ValueError("X is object dtype → preprocessing bug")

        return X
