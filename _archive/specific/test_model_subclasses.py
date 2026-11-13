"""Unit tests for the concrete model subclasses (MACK, MCBound, JPCP).

These cover the offline-reachable surface: construction/default injection,
embedding transforms, config builders (which only assemble params, no data
load), MACK's KMeans label transform, and the save/load round-trip on an
untrained estimator. The ``get_test_config`` valid branches and the SB
(SentenceTransformer) embedding path require network/large downloads and are
intentionally not exercised here.
"""

import numpy as np
import pytest

import torch  # noqa: F401
import sklearn  # noqa: F401
import xgboost  # noqa: F401
import sentence_transformers  # noqa: F401

from seanergys_modelzoo.models.common.seanergys_model import (  # noqa: E402
    SeanergysModelTask,
)
from seanergys_modelzoo.models.tasks.performance_prediction.mack.mack_model import (  # noqa: E402
    MACK,
    Embedding as MackEmbedding,
)
from seanergys_modelzoo.models.tasks.performance_prediction.mcbound.mcbound_model import (  # noqa: E402
    MCBound,
    Embedding as McbEmbedding,
)
from seanergys_modelzoo.models.tasks.power_consumption_prediction.jpcp.jpcp_model import (  # noqa: E402
    JPCP,
    Embedding as JpcpEmbedding,
)
from seanergys_modelzoo.models.common.seanergys_model import (  # noqa: E402
    SeanergysModel,
)
from tests.conftest import MODEL_CLASSES, MODEL_DATASET_PAIRS  # noqa: E402

_PAIR_IDS = [f"{m.__name__}-{d}" for m, d in MODEL_DATASET_PAIRS]
_MODEL_IDS = list(MODEL_CLASSES.keys())
_MODELS = list(MODEL_CLASSES.values())


# ---------------------------------------------------------------------------
# Construction / default injection
# ---------------------------------------------------------------------------


def test_mack_defaults():
    m = MACK()
    assert m.task_type == SeanergysModelTask.CLASSIFICATION
    assert m.kmeans_flops is not None
    assert m.kmeans_memory_bw is not None


def test_mcbound_defaults():
    m = MCBound()
    assert m.task_type == SeanergysModelTask.CLASSIFICATION


def test_jpcp_defaults():
    m = JPCP()
    assert m.task_type == SeanergysModelTask.REGRESSION


# ---------------------------------------------------------------------------
# transform_embeddings / embedding_parsing
# ---------------------------------------------------------------------------


def test_embedding_none_passthrough():
    feats = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert np.array_equal(MACK().transform_embeddings(feats), feats)
    assert np.array_equal(
        MCBound(embedding_type=McbEmbedding.NONE).transform_embeddings(feats), feats
    )
    assert np.array_equal(
        JPCP(embedding_type=JpcpEmbedding.NONE).transform_embeddings(feats), feats
    )


def test_embedding_int_label_encodes():
    feats = np.array([["a", "x"], ["b", "x"], ["a", "y"]])
    out = MACK(embedding_type=MackEmbedding.INT).transform_embeddings(feats)
    assert out.shape == (3, 2)
    # MCBound/JPCP require fit=True for the encoder to be fitted first
    out_mcb = MCBound(embedding_type=McbEmbedding.INT).transform_embeddings(
        feats, fit=True
    )
    assert out_mcb.shape == (3, 2)
    out_jpcp = JPCP(embedding_type=JpcpEmbedding.INT).transform_embeddings(
        feats, fit=True
    )
    assert out_jpcp.shape == (3, 2)


def test_embedding_parsing_returns_384_vector():
    arr = MACK().embedding_parsing([np.arange(384)])
    assert arr.shape == (384,)


# ---------------------------------------------------------------------------
# MACK-specific: transform_kmeans, parse_job_script
# ---------------------------------------------------------------------------


def test_mack_transform_kmeans_fit():
    m = MACK(k_flops=2, k_memory_bw=2)
    flops = np.array([1.0, 1.1, 10.0, 10.2])
    memory_bw = np.array([2.0, 2.1, 20.0, 20.5])
    labels = m.transform_kmeans(flops, memory_bw, fit=True)
    assert len(labels) == 4
    assert all(isinstance(t, tuple) and len(t) == 2 for t in labels)


def test_mack_parse_job_script():
    script = "#SBATCH stuff\nmodule load gcc\nrun --flag # comment\n"
    out = MACK().parse_job_script(script)
    # Comment lines (starting with '#') and inline '# comment' trailers are stripped.
    assert "#SBATCH" not in out
    assert "comment" not in out
    # NOTE: characterizes a known off-by-one — for lines without '#', idx stays -1
    # so e[:-1] drops the final character ("gcc" -> "gc") and lines are concatenated
    # without separators. Captured here so a future fix is a deliberate change.
    assert out == "module load gcrun --flag "


# ---------------------------------------------------------------------------
# get_train_config structure (params only — no data load).
# Discovery-driven over every supported (model, dataset) pair, so a new model or
# dataset is covered automatically — no per-model hardcoded case to add.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_cls,dataset_id", MODEL_DATASET_PAIRS, ids=_PAIR_IDS)
def test_get_train_config_structure(model_cls, dataset_id):
    """Every supported (model, dataset) pair builds a well-formed train config."""
    cfg = model_cls.get_train_config(dataset_id)
    assert dataset_id in cfg
    model, ds_cls, (ds_params, dl_params) = cfg[dataset_id]
    assert isinstance(model, model_cls)
    assert isinstance(model, SeanergysModel)
    assert isinstance(ds_cls, type)
    # The pair was discovered from ds_cls.DATASET_ID, so the config must resolve
    # back to the same dataset id.
    assert ds_cls.DATASET_ID == dataset_id
    assert dl_params.batch_size >= 1
    assert ds_params.input_features, "config must declare input_features"


@pytest.mark.parametrize("model_cls", _MODELS, ids=_MODEL_IDS)
def test_get_train_config_invalid_dataset_raises(model_cls):
    """Every model's train-config builder rejects an unknown dataset id.

    (JPCP previously mapped any non-"f-data" id to its PM100 branch instead of
    raising; that was fixed so all three validate consistently.)
    """
    with pytest.raises(Exception):
        model_cls.get_train_config("definitely-not-a-real-dataset-id")


# ---------------------------------------------------------------------------
# get_test_config invalid branches (valid branches need Zenodo) — discovery-driven.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_cls", _MODELS, ids=_MODEL_IDS)
def test_get_test_config_invalid_dataset_raises(model_cls):
    """An unknown dataset id is rejected by every model's test-config builder."""
    with pytest.raises(Exception):
        model_cls.get_test_config(model_cls(), "nope")


# ---------------------------------------------------------------------------
# save / load round-trip (untrained estimator)
# ---------------------------------------------------------------------------


def test_mcbound_save_load_roundtrip(tmp_path):
    m = MCBound()
    save_dir = tmp_path / "mcb"
    assert m.save(save_dir) is True
    loaded = MCBound.load(save_dir)
    assert isinstance(loaded, MCBound)
    assert loaded.is_trained is True


def test_jpcp_save_load_roundtrip(tmp_path):
    m = JPCP()
    save_dir = tmp_path / "jpcp"
    assert m.save(save_dir) is True
    loaded = JPCP.load(save_dir)
    assert isinstance(loaded, JPCP)


def test_mack_save_load_roundtrip(tmp_path):
    m = MACK()
    save_dir = tmp_path / "mack"
    assert m.save(save_dir) is True
    loaded = MACK.load(save_dir)
    assert isinstance(loaded, MACK)
    assert loaded.kmeans_flops is not None


def test_load_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        MCBound.load(tmp_path / "nope")
    with pytest.raises(FileNotFoundError):
        MACK.load(tmp_path / "nope")
    with pytest.raises(FileNotFoundError):
        JPCP.load(tmp_path / "nope")
