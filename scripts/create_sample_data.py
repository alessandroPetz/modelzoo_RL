#!/usr/bin/env python3
"""
Create small sample fixture files for local testing and CI.

WHY THIS FILE EXISTS
--------------------
Francesco's original CI scripts ran validation directly against Zenodo —
every test run downloaded the full dataset. That made CI slow, fragile
(network-dependent), and impossible to run offline.

This script solves that by downloading a tiny sample ONCE (during CI setup or
locally with `make sample-data`) and saving it as a local parquet file.
The integration tests then use those local files — no network needed during
the actual test run.

HOW IT WORKS
------------
Instead of hardcoding dataset names or requiring get_dummy_params(), this script
uses the models as the entry point:

  1. Auto-discover every SeanergysModel subclass in models/tasks/
  2. Call get_train_config() on each — this already defines which dataset class
     to use, which input/output features, and which filters
  3. Override ds_params with is_dummy=True to get a small filtered subset
  4. Instantiate the dataset — the model_validator fires load_data() automatically
  5. Save the loaded DataFrame to tests/fixtures/sample_data/

Datasets are deduplicated — if MACK and MCBound both use FDataDataset, it is
only downloaded once.

When a developer adds a new model with get_train_config(), its dataset fixture
is created automatically — no changes needed here.

LIMITATION
----------
If a dataset is added without any model (no get_train_config() pointing to it),
this script will not create a fixture for it. In that case the dataset needs a
self-describing way to produce test params (there is no get_dummy_params() today);
see the "Design decisions & known limitations" section of docs/ci-cd.md.

Usage:
    python scripts/create_sample_data.py
    SAMPLE_MAX_ROWS=50 python scripts/create_sample_data.py
    python scripts/create_sample_data.py --max-rows 200

Output:
    tests/fixtures/sample_data/sample_<datasetclassname_lower>.parquet
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _max_rows() -> int:
    return int(
        os.environ.get("SAMPLE_MAX_ROWS", os.environ.get("CI_MAX_SAMPLES", "100"))
    )


def _output_dir() -> Path:
    d = PROJECT_ROOT / "tests" / "fixtures" / "sample_data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _discover_model_classes() -> dict:
    """
    Auto-discover all concrete SeanergysModel subclasses in models/tasks/.
    Same mechanism as tests/conftest.py — no model names hardcoded.
    """
    try:
        from ci.utils import retrieve_instances_from_file
        from seanergys_modelzoo.models.common.seanergys_model import SeanergysModel
    except ImportError as e:
        print(f"Cannot import project packages: {e}", file=sys.stderr)
        sys.exit(1)

    classes = {}
    for py_file in (PROJECT_ROOT / "seanergys_modelzoo" / "models" / "tasks").rglob(
        "*.py"
    ):
        if py_file.name == "__init__.py":
            continue
        try:
            classes.update(retrieve_instances_from_file(py_file, SeanergysModel))
        except Exception:
            continue
    return classes


def _collect_dataset_configs_from_models(model_classes: dict) -> dict:
    """
    Call get_train_config() on every model and collect unique datasets.

    For each dataset, accumulates the union of raw columns required by all
    models that use it. This ensures the fixture file has every column any
    model might need.

    Returns a dict of {dataset_class_name: (dataset_cls, set[str])} where
    the set contains all required raw column names.
    """
    seen: dict = {}  # {name: (dataset_cls, set_of_columns)}

    for model_name, model_cls in model_classes.items():
        try:
            config = model_cls.get_train_config()
        except Exception as e:
            print(f"  WARN  {model_name}.get_train_config() failed: {e}")
            continue

        for entry_name, (_, dataset_cls, (ds_params, _dl)) in config.items():
            key = dataset_cls.__name__

            # Prefer explicit columns list; fall back to input + output features
            if getattr(ds_params, "columns", None):
                cols = set(ds_params.columns)
            else:
                cols = set(
                    getattr(ds_params, "input_features", [])
                    + getattr(ds_params, "output_features", [])
                )

            if key not in seen:
                seen[key] = (dataset_cls, cols)
                print(f"  found {key} (via {model_name}/{entry_name})")
            else:
                # Accumulate columns from additional models using the same dataset
                seen[key][1].update(cols)

    return seen


def fixture_name_for(dataset_cls) -> str:
    """FDataDataset -> sample_fdata.parquet. One definition of the naming rule."""
    name = dataset_cls.__name__.lower().replace("dataset", "")
    return f"sample_{name}.parquet"


def _samplable_dataset_classes() -> dict:
    """
    Dataset classes that this script knows how to sample, found WITHOUT importing
    or instantiating any model.

    Deliberately dataset-only: discovering models instantiates them, and a model
    whose constructor builds a dataset would reach the network just to answer
    "which fixtures am I missing?". Presence checking must stay free and offline,
    so it never triggers a download it was run to avoid.

    Returns {dataset_class_name: dataset_cls}.
    """
    try:
        from ci.utils import retrieve_instances_from_file
        from seanergys_modelzoo.datasets.common.seanergys_dataset import (
            SeanergysDataset,
        )
    except ImportError as e:
        print(f"Cannot import project packages: {e}", file=sys.stderr)
        return {}

    _BASE_CLASSES = {"SeanergysDataset", "SeanergysParquetDataset"}
    found = {}
    for py_file in (PROJECT_ROOT / "seanergys_modelzoo" / "datasets").rglob("*.py"):
        if py_file.name == "__init__.py":
            continue
        try:
            classes = retrieve_instances_from_file(py_file, SeanergysDataset)
        except Exception:
            continue
        for name, cls in classes.items():
            if cls.__name__ in _BASE_CLASSES:
                continue
            if _is_samplable(cls):
                found.setdefault(name, cls)
    return found


def _is_samplable(dataset_cls) -> bool:
    """
    True when create_sample() has a branch that can actually fetch this dataset.

    Must mirror create_sample() exactly. A dataset that merely declares
    ZENODO_BASE_URL is NOT enough: M100Dataset has one but ships per-record tar
    archives keyed by AVAILABLE_FILES_PER_RECORD, so there is no single parquet
    to slice. Listing it as "missing" would make `make ensure-fixtures` try, and
    fail, to download it on every run.
    """
    fields = dataset_cls.model_fields
    multi_file = "ZENODO_BASE_URL" in fields and "AVAILABLE_FILES" in fields
    single_file = "ZENODO_URL" in fields
    return multi_file or single_file


def missing_fixtures(out_dir: Path) -> dict:
    """
    {dataset_class_name: expected_path} for every samplable dataset whose fixture
    file is absent or empty. Empty dict means nothing needs downloading.
    """
    missing = {}
    for name, cls in _samplable_dataset_classes().items():
        path = out_dir / fixture_name_for(cls)
        if not path.exists() or path.stat().st_size == 0:
            missing[name] = path
    return missing


def _collect_dataset_configs_directly() -> dict:
    """
    Discover datasets directly by scanning DATASET_CLASSES for any class that
    has a ZENODO_URL or ZENODO_BASE_URL Pydantic field.

    This covers datasets that exist without a model pointing to them.
    Returns a dict of {dataset_class_name: (dataset_cls, None)}.
    """
    try:
        from ci.utils import retrieve_instances_from_file
        from seanergys_modelzoo.datasets.common.seanergys_dataset import (
            SeanergysDataset,
        )
    except ImportError as e:
        print(f"Cannot import project packages: {e}", file=sys.stderr)
        return {}

    _BASE_CLASSES = {"SeanergysDataset", "SeanergysParquetDataset"}
    seen = {}

    for py_file in (PROJECT_ROOT / "seanergys_modelzoo" / "datasets").rglob("*.py"):
        if py_file.name == "__init__.py":
            continue
        try:
            classes = retrieve_instances_from_file(py_file, SeanergysDataset)
        except Exception:
            continue
        for name, cls in classes.items():
            if cls.__name__ in _BASE_CLASSES:
                continue
            if not _is_samplable(cls):
                # Not an error: the dataset simply is not a single-parquet source
                # this script can slice (see _is_samplable). Say so once, rather
                # than letting it fail later inside create_sample().
                print(f"  skip  {cls.__name__} (no single-parquet source to sample)")
                continue
            if name not in seen:
                seen[name] = (cls, None)
                print(f"  found {cls.__name__} (via direct dataset discovery)")

    return seen


def _collect_dataset_configs(model_classes: dict) -> dict:
    """
    Collect unique datasets from both model get_train_config() and direct
    dataset discovery (for datasets with ZENODO_URL/ZENODO_BASE_URL fields).

    Model-based discovery takes priority (deduplicated by dataset class name).
    """
    seen = _collect_dataset_configs_from_models(model_classes)

    for key, value in _collect_dataset_configs_directly().items():
        if key not in seen:
            seen[key] = value

    return seen


def create_sample(dataset_cls, required_cols, max_rows: int, out_dir: Path) -> Path:
    """
    Download a small raw sample from Zenodo and save to parquet.

    WHY WE BYPASS THE DATASET CLASS ENTIRELY
    -----------------------------------------
    Going through the dataset class has several problems:

    1. is_dummy=True in FDataDataset only adds date filters — it still hits Zenodo.
    2. Column filtering: a model's required columns may not all exist in every
       monthly file, so a projected read can fail on the newest one.
    3. Transforms require fitted models (embedding_parsing, etc.) which cannot
       run at sample creation time — the model has not been trained yet.

    SOLUTION: download raw parquet directly with pandas (all columns, no filters).
    For multi-file datasets, try files from newest to oldest until a file that
    contains all required columns is found. Take head(max_rows) and save all
    columns — the integration tests select only the columns they need.

    Args:
        required_cols: set of column names that must be present in the fixture.
                       None means any file is acceptable.
    """
    import pandas as pd

    # ZENODO_BASE_URL and ZENODO_URL are Pydantic fields (not plain class attrs),
    # so we read their default values from model_fields rather than the class directly.
    fields = dataset_cls.model_fields

    # --- Multi-file datasets (e.g. FDataDataset: 38 monthly parquet files) ---
    if "ZENODO_BASE_URL" in fields and "AVAILABLE_FILES" in fields:
        base_url = fields["ZENODO_BASE_URL"].default
        available_files = fields["AVAILABLE_FILES"].default

        # Try files from newest to oldest until one contains all required columns.
        df = None
        for file_name in reversed(available_files):
            url = f"{base_url}/{file_name}.parquet"
            print(f"         trying {file_name}.parquet ...")
            candidate = pd.read_parquet(url)
            if required_cols and not required_cols.issubset(set(candidate.columns)):
                missing = required_cols - set(candidate.columns)
                print(f"           missing columns {missing}, trying older file...")
                continue
            df = candidate
            print("           OK — has all required columns.")
            break

        if df is None:
            raise ValueError(
                f"{dataset_cls.__name__}: no file in AVAILABLE_FILES contains all "
                f"required columns: {required_cols}"
            )

    # --- Single-file datasets (e.g. PM100Dataset: one parquet URL) ---
    elif "ZENODO_URL" in fields:
        url = fields["ZENODO_URL"].default
        print("         downloading from Zenodo (no column filter)...")
        df = pd.read_parquet(url)

    else:
        raise ValueError(
            f"{dataset_cls.__name__} has no ZENODO_BASE_URL or ZENODO_URL Pydantic field. "
            "Cannot determine download URL. Add one of these fields to the dataset class."
        )

    sample = df.head(max_rows)

    # Consistent naming: FDataDataset → sample_fdata.parquet
    out_path = out_dir / fixture_name_for(dataset_cls)
    sample.to_parquet(out_path, index=False)

    return out_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Create sample fixture parquet files using model get_train_config()."
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Max rows per sample (default: SAMPLE_MAX_ROWS or CI_MAX_SAMPLES or 100).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=os.environ.get("FORCE_SAMPLE_DATA") == "1",
        help=(
            "Re-download fixtures that already exist. Without this, an existing "
            "fixture is kept and never re-fetched (FORCE_SAMPLE_DATA=1 also works)."
        ),
    )
    parser.add_argument(
        "--list-missing",
        action="store_true",
        help=(
            "Print the dataset classes whose fixture is absent, one per line, and "
            "exit. Never downloads and never imports a model, so it is safe to call "
            "before deciding whether a download is needed."
        ),
    )
    args = parser.parse_args()

    max_rows = args.max_rows or _max_rows()
    out_dir = _output_dir()

    # Presence check only — offline, no model import, no network.
    if args.list_missing:
        for name in sorted(missing_fixtures(out_dir)):
            print(name)
        return

    if args.force:
        print("--force: existing fixtures will be re-downloaded.\n")
    else:
        missing = missing_fixtures(out_dir)
        if not missing:
            print(
                f"✓ Every fixture is already present in {out_dir}.\n"
                f"  Nothing to download. Use --force (or FORCE_SAMPLE_DATA=1) to refresh."
            )
            return
        print(f"Missing fixtures: {', '.join(sorted(missing))}\n")

    print("Discovering models...")
    model_classes = _discover_model_classes()
    if not model_classes:
        print("No model classes found.")
        sys.exit(1)
    print(f"Found {len(model_classes)} model(s): {', '.join(model_classes)}\n")

    print("Collecting datasets from get_train_config()...")
    dataset_configs = _collect_dataset_configs(model_classes)
    if not dataset_configs:
        print("No datasets found via get_train_config().")
        sys.exit(1)
    print(f"\nWill create samples for: {', '.join(dataset_configs)}")
    print(f"Output dir : {out_dir}")
    print(f"Max rows   : {max_rows}\n")

    errors = []
    skipped = []
    for dataset_name, (dataset_cls, required_cols) in dataset_configs.items():
        existing = out_dir / fixture_name_for(dataset_cls)
        if existing.exists() and existing.stat().st_size > 0 and not args.force:
            print(f"  SKIP  {dataset_name} → {existing.name} already present")
            skipped.append(dataset_name)
            continue
        try:
            print(f"  ...   {dataset_name}")
            out_path = create_sample(
                dataset_cls, required_cols, max_rows=max_rows, out_dir=out_dir
            )
            print(f"  OK    {dataset_name} → {out_path.name}")
        except Exception as e:
            print(f"  ERR   {dataset_name}: {e}")
            errors.append(dataset_name)

    print()
    if errors:
        print(
            f"WARNING: {len(errors)} dataset(s) could not be sampled and were skipped: "
            f"{', '.join(errors)}\n"
            "Other fixtures were created successfully. Fix the errors above and re-run "
            "`make sample-data` to fill in the missing ones.",
            file=sys.stderr,
        )
    created = len(dataset_configs) - len(errors) - len(skipped)
    summary = f"Done. {created} fixture(s) created"
    if skipped:
        summary += f", {len(skipped)} already present ({', '.join(skipped)})"
    if errors:
        summary += f", {len(errors)} failed"
    print(summary + ".")

    # A fixture that is still missing after this run is a hard failure: the
    # caller asked for fixtures and did not get them, and a downstream test would
    # otherwise fail later with a much less obvious message.
    still_missing = missing_fixtures(out_dir)
    if still_missing:
        print(
            f"✗ Still missing after this run: {', '.join(sorted(still_missing))}",
            file=sys.stderr,
        )
        sys.exit(1)

    # An explicit --force that could not refresh anything is also a failure: the
    # developer asked for fresh data and the old files are still on disk. Without
    # --force, errors are tolerated as long as nothing is missing.
    if errors and args.force:
        print(
            f"✗ --force requested but {len(errors)} dataset(s) failed to refresh: "
            f"{', '.join(errors)} (existing fixtures left untouched)",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
