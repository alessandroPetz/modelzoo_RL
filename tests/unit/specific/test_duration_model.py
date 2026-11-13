from seanergys_modelzoo.models.tasks.duration_prediction.duration_prediction_bull.models.duration_model import (
    DurationModel,
)

from seanergys_modelzoo.models.tasks.duration_prediction.duration_prediction_bull.models.duration_prediction_bull import (
    DurationPredictionBull,
)

import pytest
import numpy as np


def test_metadata_contains_units():
    model = DurationModel()

    model.metadata = {
        "target_unit": "minutes",
        "walltime_unit": "minutes",
    }

    assert model.metadata["target_unit"] == "minutes"
    assert model.metadata["walltime_unit"] == "minutes"


def test_get_weights_sum_to_one():
    model = DurationModel()
    model.val_rmse = [1.0, 2.0, 4.0]

    weights = model._get_weights()

    assert np.isclose(weights.sum(), 1.0)


def test_get_weights_favor_best_model():
    model = DurationModel()
    model.val_rmse = [1.0, 10.0]

    weights = model._get_weights()

    assert weights[0] > weights[1]


def test_predict_without_training_raises():
    model = DurationModel()

    with pytest.raises(RuntimeError):
        model.predict(np.zeros((1, 10)))


class DummyModel:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.full(X.shape[0], self.value)


def test_weighted_ensemble_prediction():
    model = DurationModel()

    model.models = [
        DummyModel(10.0),
        DummyModel(20.0),
    ]

    model.val_rmse = [1.0, 2.0]

    X = np.zeros((5, 3))

    pred = model.predict(X)

    expected = ((1.0 / 1.0) * 10.0 + (1.0 / 2.0) * 20.0) / ((1.0 / 1.0) + (1.0 / 2.0))

    assert np.allclose(pred, expected)


def test_predict_full_contains_required_keys():
    model = DurationModel()

    model.models = [DummyModel(10.0)]
    model.val_rmse = [1.0]

    X = np.zeros((2, 3))

    result = model.predict_full(X)

    assert "y_pred" in result
    assert "uncertainty" in result
    assert "members" in result
    assert "y_pred_best" in result
    assert "best_model_idx" in result


def test_save_and_load(tmp_path):
    model = DurationModel()

    assert model.save(tmp_path)

    loaded = DurationModel.load(tmp_path)

    assert isinstance(loaded, DurationModel)


def test_flatten_input_sample():
    """Duration input samples should be flattened into a numeric vector."""
    embedding = np.arange(384, dtype=np.float32)
    sample = np.array([embedding, 8, 2, 3600], dtype=object)

    result = DurationPredictionBull.flatten_input_sample(sample)

    assert result.shape == (387,)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result[:384], embedding)
    np.testing.assert_array_equal(result[-3:], [8, 2, 3600])


def test_duration_train_config_uses_raw_fdata_features():
    """Training config should use the raw F-DATA duration features."""
    config = DurationPredictionBull.get_train_config(
        dataset_name="f-data",
        get_dummy_data=True,
    )

    _, _, (ds_params, _) = config["f-data"]

    assert ds_params.input_features == [
        "embedding",
        "cnumr",
        "nnumr",
        "elpl",
    ]
    assert ds_params.output_features == ["duration"]
    assert ds_params.transform is not None
