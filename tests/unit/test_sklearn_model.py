"""Unit tests for SeanergysSklearnModel.

Exercises the full train/predict/evaluate/save/load lifecycle of the generic
scikit-learn wrapper using a tiny in-memory dataset and trivial estimators, so
no network access or data files are required.
"""

import numpy as np
import pytest

import torch  # noqa: F401
import sklearn  # noqa: F401

from sklearn.linear_model import (  # noqa: E402
    LinearRegression,
    LogisticRegression,
    SGDClassifier,
    SGDRegressor,
)
from sklearn.neighbors import KNeighborsRegressor  # noqa: E402

from seanergys_modelzoo.dataloader.seanergys_dataloader import (  # noqa: E402
    SeanergysDataloader,
)
from seanergys_modelzoo.datasets.common.seanergys_dataset import (  # noqa: E402
    SeanergysDataset,
)
from seanergys_modelzoo.models.common.seanergys_model import (  # noqa: E402
    SeanergysModelTask,
)
from seanergys_modelzoo.models.common.sklearn_seanergys_model import (  # noqa: E402
    SeanergysSklearnModel,
)


class _TinyDataset(SeanergysDataset):
    """Deterministic in-memory dataset: ``n`` rows of ``n_features``."""

    n: int = 12
    n_features: int = 3
    classification: bool = True

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        rng = np.random.RandomState(0)
        self._X = rng.randn(self.n, self.n_features).astype(np.float32)
        if self.classification:
            self._y = (self._X.sum(axis=1) > 0).astype(np.int64)
        else:
            self._y = self._X.sum(axis=1).astype(np.float32)

    def __getitem__(self, idx: int):
        return self._X[idx], self._y[idx]

    def __len__(self) -> int:
        return self.n


def _loader(classification: bool = True, batch_size: int = 4) -> SeanergysDataloader:
    return SeanergysDataloader(
        _TinyDataset(classification=classification), batch_size=batch_size
    )


def _mse(y_true, y_pred):
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


# ---------------------------------------------------------------------------
# build_model / construction
# ---------------------------------------------------------------------------


def test_build_model_creates_estimator():
    model = SeanergysSklearnModel(
        model_class=LinearRegression,
        task_type=SeanergysModelTask.REGRESSION,
    )
    assert model.estimator is not None
    assert model.model_name == "SeanergysSklearnModel"
    assert model.is_trained is False


def test_build_model_error_path_reraises():
    class _Boom:
        def __init__(self, **kwargs):
            raise RuntimeError("cannot construct")

    with pytest.raises(RuntimeError, match="cannot construct"):
        SeanergysSklearnModel(
            model_class=_Boom,
            task_type=SeanergysModelTask.REGRESSION,
        )


# ---------------------------------------------------------------------------
# helpers: _to_numpy, _extract_data_from_loader
# ---------------------------------------------------------------------------


def test_to_numpy_variants():

    arr = np.array([1, 2, 3])
    assert SeanergysSklearnModel._to_numpy(arr) is arr
    assert np.array_equal(
        SeanergysSklearnModel._to_numpy(torch.tensor([1, 2, 3])), np.array([1, 2, 3])
    )
    assert np.array_equal(
        SeanergysSklearnModel._to_numpy([1, 2, 3]), np.array([1, 2, 3])
    )


def test_extract_data_from_loader_shapes():
    model = SeanergysSklearnModel(
        model_class=LinearRegression, task_type=SeanergysModelTask.REGRESSION
    )
    X, y = model._extract_data_from_loader(_loader(classification=False))
    assert X.shape == (12, 3)
    assert y.shape == (12,)


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


def test_train_partial_fit_path():
    """SGDRegressor exposes partial_fit → mini-batch path."""
    model = SeanergysSklearnModel(
        model_class=SGDRegressor,
        task_type=SeanergysModelTask.REGRESSION,
        model_hyperparameters={"random_state": 0},
    )
    history = model.train(_loader(classification=False), training_loss=_mse)
    assert model.is_trained is True
    assert history["training"]["training_loss"] is not None
    assert history["training"]["training_time"] >= 0
    assert getattr(model.metadata, "training_time", None) is not None


def test_train_bulk_fit_path():
    """LinearRegression has no partial_fit → bulk fit path."""
    model = SeanergysSklearnModel(
        model_class=LinearRegression, task_type=SeanergysModelTask.REGRESSION
    )
    history = model.train(_loader(classification=False), training_loss=_mse)
    assert model.is_trained is True
    assert history["training"]["training_loss"] is not None


def test_train_with_validation_metrics():
    model = SeanergysSklearnModel(
        model_class=LinearRegression, task_type=SeanergysModelTask.REGRESSION
    )
    history = model.train(
        _loader(classification=False),
        val_data_loader=_loader(classification=False),
        validation_metrics=[_mse],
    )
    assert history["validation"]["validation_metrics"] is not None
    assert "_mse" in history["validation"]["validation_metrics"]
    assert getattr(model.metadata, "validation_time", None) is not None


def test_train_with_validation_loader_but_no_metrics():
    """val_data_loader provided without metrics → validation skipped, no error."""
    model = SeanergysSklearnModel(
        model_class=LinearRegression, task_type=SeanergysModelTask.REGRESSION
    )
    history = model.train(
        _loader(classification=False),
        val_data_loader=_loader(classification=False),
        validation_metrics=None,
    )
    assert history["validation"]["validation_metrics"] is None


def test_train_error_path_reraises():
    """A training_loss that explodes surfaces through the except block."""
    model = SeanergysSklearnModel(
        model_class=LinearRegression, task_type=SeanergysModelTask.REGRESSION
    )

    def _bad_loss(y_true, y_pred):
        raise ValueError("loss boom")

    with pytest.raises(ValueError, match="loss boom"):
        model.train(_loader(classification=False), training_loss=_bad_loss)


# ---------------------------------------------------------------------------
# predict / evaluate
# ---------------------------------------------------------------------------


def test_predict_with_loader():
    model = SeanergysSklearnModel(
        model_class=LinearRegression, task_type=SeanergysModelTask.REGRESSION
    )
    model.train(_loader(classification=False))
    preds = model.predict(_loader(classification=False))
    assert isinstance(preds, np.ndarray)
    assert len(preds) == 12


def test_predict_with_array_input():
    model = SeanergysSklearnModel(
        model_class=LinearRegression, task_type=SeanergysModelTask.REGRESSION
    )
    model.train(_loader(classification=False))
    preds = model.predict(np.random.RandomState(1).randn(5, 3))
    assert len(preds) == 5


def test_predict_proba_classification():
    model = SeanergysSklearnModel(
        model_class=LogisticRegression,
        task_type=SeanergysModelTask.CLASSIFICATION,
        model_hyperparameters={"max_iter": 200},
    )
    model.train(_loader(classification=True))
    proba = model.predict(_loader(classification=True), return_proba=True)
    assert proba.ndim == 2


def test_predict_proba_fallback_without_predict_proba():
    """SGDClassifier(loss='hinge') has no predict_proba → falls back to predict."""
    model = SeanergysSklearnModel(
        model_class=SGDClassifier,
        task_type=SeanergysModelTask.CLASSIFICATION,
        model_hyperparameters={"loss": "hinge", "random_state": 0},
    )
    model.train(_loader(classification=True), classes=np.array([0, 1]))
    preds = model.predict(_loader(classification=True), return_proba=True)
    assert preds.ndim == 1


def test_predict_before_training_raises_and_warns():
    model = SeanergysSklearnModel(
        model_class=LinearRegression, task_type=SeanergysModelTask.REGRESSION
    )
    with pytest.raises(Exception):
        model.predict(np.random.RandomState(2).randn(4, 3))


def test_evaluate_returns_metric_list():
    model = SeanergysSklearnModel(
        model_class=LinearRegression, task_type=SeanergysModelTask.REGRESSION
    )
    model.train(_loader(classification=False))
    results = model.evaluate(_loader(classification=False), metrics=[_mse])
    assert len(results) == 1
    assert isinstance(results[0], float)


def test_evaluate_without_metrics_returns_empty():
    model = SeanergysSklearnModel(
        model_class=LinearRegression, task_type=SeanergysModelTask.REGRESSION
    )
    model.train(_loader(classification=False))
    assert model.evaluate(_loader(classification=False)) == []


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------


def test_save_load_roundtrip(tmp_path):
    model = SeanergysSklearnModel(
        model_class=LinearRegression, task_type=SeanergysModelTask.REGRESSION
    )
    model.train(_loader(classification=False))
    before = model.predict(np.arange(6).reshape(2, 3).astype(float))

    save_dir = tmp_path / "model"
    assert model.save(save_dir) is True
    assert (tmp_path / "model_info.json").exists()
    assert (tmp_path / "model_metadata.json").exists()

    loaded = SeanergysSklearnModel.load(save_dir)
    assert loaded.is_trained is True
    assert loaded.task_type == SeanergysModelTask.REGRESSION
    after = loaded.predict(np.arange(6).reshape(2, 3).astype(float))
    assert np.allclose(before, after)


def test_save_returns_false_on_error(tmp_path, monkeypatch):
    model = SeanergysSklearnModel(
        model_class=LinearRegression, task_type=SeanergysModelTask.REGRESSION
    )
    model.train(_loader(classification=False))

    import seanergys_modelzoo.models.common.sklearn_seanergys_model as mod

    def _boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(mod.joblib, "dump", _boom)
    assert model.save(tmp_path / "model") is False


def test_load_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        SeanergysSklearnModel.load(tmp_path / "does_not_exist")


def test_load_missing_info_raises(tmp_path):
    model = SeanergysSklearnModel(
        model_class=LinearRegression, task_type=SeanergysModelTask.REGRESSION
    )
    model.train(_loader(classification=False))
    save_dir = tmp_path / "model"
    model.save(save_dir)
    (tmp_path / "model_info.json").unlink()
    with pytest.raises(RuntimeError):
        SeanergysSklearnModel.load(save_dir)


# ---------------------------------------------------------------------------
# feature importance / summary / metadata
# ---------------------------------------------------------------------------


def test_feature_importance_from_coef():
    model = SeanergysSklearnModel(
        model_class=LinearRegression, task_type=SeanergysModelTask.REGRESSION
    )
    model.train(_loader(classification=False))
    importance = model.get_feature_importance()
    assert importance is not None
    assert np.all(importance >= 0)  # abs(coef_)


def test_feature_importance_none_when_unsupported():
    model = SeanergysSklearnModel(
        model_class=KNeighborsRegressor, task_type=SeanergysModelTask.REGRESSION
    )
    model.train(_loader(classification=False))
    assert model.get_feature_importance() is None


def test_summary_includes_estimator_name():
    model = SeanergysSklearnModel(
        model_class=LinearRegression, task_type=SeanergysModelTask.REGRESSION
    )
    model.train(_loader(classification=False))
    text = model.summary()
    assert "LinearRegression" in text
    assert "Estimator Parameters" in text


def test_set_metadata_updates_hyperparameters():
    model = SeanergysSklearnModel(
        model_class=LinearRegression, task_type=SeanergysModelTask.REGRESSION
    )
    model._set_metadata("foo", 123)
    assert model.metadata.hyperparameters["foo"] == 123


def test_get_metadata_returns_dict():
    model = SeanergysSklearnModel(
        model_class=LinearRegression, task_type=SeanergysModelTask.REGRESSION
    )
    assert isinstance(model.get_metadata(), dict)
