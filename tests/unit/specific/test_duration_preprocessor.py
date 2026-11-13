import numpy as np
import pandas as pd
import pytest

from seanergys_modelzoo.models.tasks.duration_prediction.duration_prediction_bull.preprocessing.duration_preprocessor import (
    DurationPreprocessor,
)


def test_missing_embedding_columns_raises():
    df = pd.DataFrame(
        {
            "cnumr": [64],
            "nnumr": [4],
            "elpl": [3600],
        }
    )

    prep = DurationPreprocessor()

    with pytest.raises(ValueError):
        prep.fit(df)


def test_missing_required_column_raises():
    df = pd.DataFrame(
        {
            "emb_0": [0.1],
            "cnumr": [64],
            "nnumr": [4],
        }
    )

    prep = DurationPreprocessor()

    with pytest.raises(ValueError):
        prep.fit(df)


def test_elpl_is_converted_to_minutes():
    df = pd.DataFrame(
        {
            "emb_0": [0.1],
            "cnumr": [64],
            "nnumr": [4],
            "elpl": [3600.0],  # seconds
        }
    )

    prep = DurationPreprocessor()
    prep.fit(df)

    X = prep.transform(df)

    # first numerical columns are:
    # cnumr, nnumr, elpl
    elpl_idx = len(prep.emb_cols) + 2

    expected = np.log1p(60.0)

    assert np.isclose(X[0, elpl_idx], expected)


def test_transform_before_fit_raises():
    prep = DurationPreprocessor()

    with pytest.raises(RuntimeError):
        prep.transform(np.zeros((1, 3)))
