"""
Utilities for preparing duration prediction datasets.

Responsibilities:
- Expand Fugaku embeddings into emb_* columns
- Create duration prediction targets
- Convert raw parquet data into model-ready DataFrames
"""

from __future__ import annotations

import pandas as pd


def expand_embeddings(
    df: pd.DataFrame, embedding_column: str = "embedding"
) -> pd.DataFrame:
    """
    Expand an embedding column into individual emb_* feature columns.

    Args:
        df (pd.DataFrame):
            Input dataframe containing embedding vectors.
        embedding_column (str):
            Name of the column containing embedding vectors.

    Returns:
        pd.DataFrame:
            Dataframe where the embedding column has been replaced
            by emb_0 ... emb_n feature columns.
    """
    df = df.copy()

    if embedding_column not in df.columns:
        raise ValueError(f"Missing embedding column: {embedding_column}")

    emb_df = pd.DataFrame(
        df[embedding_column].tolist(),
        index=df.index,
    )

    emb_df.columns = [f"emb_{i}" for i in range(emb_df.shape[1])]

    df = pd.concat(
        [df.drop(columns=[embedding_column]), emb_df],
        axis=1,
    )

    return df


def create_duration_target(
    df: pd.DataFrame,
    source_column: str = "duration",
    target_column: str = "duration_job",
) -> pd.DataFrame:
    """
    Create the target column used for duration prediction.

    Args:
        df (pd.DataFrame):
            Input dataframe.
        source_column (str):
            Name of the column containing the raw duration values.
        target_column (str):
            Name of the target column to create.

    Returns:
        pd.DataFrame:
            Dataframe containing the duration prediction target column.
    """
    df = df.copy()

    if source_column not in df.columns:
        raise ValueError(f"Missing source column: {source_column}")

    df[target_column] = df[source_column].astype(float)

    return df


def prepare_duration_dataframe(
    df: pd.DataFrame,
    embedding_column: str = "embedding",
) -> pd.DataFrame:
    """
    Prepare a raw F-DATA dataframe for duration prediction.

    The preparation pipeline expands embedding vectors into
    individual features and creates the duration prediction target.

    Args:
        df (pd.DataFrame):
            Raw input dataframe.
        embedding_column (str):
            Name of the column containing embedding vectors.

    Returns:
        pd.DataFrame:
            Prepared dataframe ready for preprocessing and training.
    """
    df = expand_embeddings(
        df,
        embedding_column=embedding_column,
    )

    df = create_duration_target(df)

    return df
