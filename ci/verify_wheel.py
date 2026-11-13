#!/usr/bin/env python3
"""
Verify the *installed* seanergys_modelzoo wheel imports correctly.

Run this in the test-wheel CI stage with ``PYTHONPATH`` cleared, so it exercises
the distributed package (site-packages), not the source tree.

Rather than hardcoding a fixed list of modules/classes (which drifts as the
package grows), this walks every submodule of the installed package and imports
it, then confirms the abstract base classes are importable and are real classes.

The wheel ships only its core runtime deps; a submodule that fails to import
solely because an optional ML dep (pandas, numpy, sklearn, …) is absent is
recorded as skipped, not a failure. Any other import error fails the job.

Exit 0 if the wheel is import-clean, 1 otherwise.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys

PACKAGE = "seanergys_modelzoo"

# Abstract base classes every downstream consumer relies on. Import paths are
# stable public API; the concrete models/datasets are validated by walking.
_CORE_CLASSES = [
    ("seanergys_modelzoo.logger.seanergys_logger", "SeanergysLogger"),
    ("seanergys_modelzoo.models.common.seanergys_model", "SeanergysModel"),
    ("seanergys_modelzoo.datasets.common.seanergys_dataset", "SeanergysDataset"),
]

# The wheel ships only its core runtime deps (pydantic, torch, requests, urllib3).
# The heavy ML stack lives in the `ci` dependency group and is NOT part of the
# distributed wheel (it is slated to become optional extras). A submodule that
# fails to import ONLY because one of these optional packages is absent is
# expected for the lean wheel — we record it as skipped, not a failure. Any other
# import error (syntax, a broken internal import, a missing CORE dep) still fails.
_OPTIONAL_ML_MODULES = {
    "numpy",
    "pandas",
    "pyarrow",
    "sklearn",
    "joblib",
    "xgboost",
    "scipy",
    "sentence_transformers",
    "transformers",
    "cachetools",
}


def _import_all_submodules(package_name: str):
    """
    Import every submodule of ``package_name``.

    Returns (failures, skipped):
      failures — real import errors [(module, error), ...]
      skipped  — modules that need an optional ML dep absent from the lean wheel
                 [(module, missing_top_level_package), ...]
    """
    failures: list[tuple[str, str]] = []
    skipped: list[tuple[str, str]] = []
    try:
        pkg = importlib.import_module(package_name)
    except Exception as e:  # noqa: BLE001 — report any import failure
        return [(package_name, f"{type(e).__name__}: {e}")], skipped

    for mod in pkgutil.walk_packages(pkg.__path__, prefix=f"{package_name}."):
        try:
            importlib.import_module(mod.name)
        except ModuleNotFoundError as e:
            missing_top = (e.name or "").split(".")[0]
            if missing_top in _OPTIONAL_ML_MODULES:
                skipped.append((mod.name, missing_top))
            else:
                failures.append((mod.name, f"ModuleNotFoundError: {e}"))
        except Exception as e:  # noqa: BLE001
            failures.append((mod.name, f"{type(e).__name__}: {e}"))
    return failures, skipped


def _check_core_classes() -> list[str]:
    """Confirm each core base class imports and is a class; return error strings."""
    errors: list[str] = []
    for module_path, class_name in _CORE_CLASSES:
        try:
            module = importlib.import_module(module_path)
            obj = getattr(module, class_name)
            if not isinstance(obj, type):
                errors.append(f"{module_path}.{class_name} is not a class")
        except Exception as e:  # noqa: BLE001
            errors.append(f"{module_path}.{class_name}: {type(e).__name__}: {e}")
    return errors


def main() -> int:
    failures, skipped = _import_all_submodules(PACKAGE)
    if failures:
        print("✗ Wheel import failures:")
        for name, err in failures:
            print(f"    {name}: {err}")
        return 1

    class_errors = _check_core_classes()
    if class_errors:
        print("✗ Core class checks failed:")
        for err in class_errors:
            print(f"    {err}")
        return 1

    if skipped:
        print(
            f"• {len(skipped)} submodule(s) skipped — need an optional ML dep not "
            f"shipped in the lean wheel (installed via the `ci` group / future extras):"
        )
        for name, dep in skipped:
            print(f"    {name} (needs {dep})")

    n_ok = sum(
        1
        for _ in pkgutil.walk_packages(
            importlib.import_module(PACKAGE).__path__, prefix=f"{PACKAGE}."
        )
    ) - len(skipped)
    print(
        f"✓ {PACKAGE}: core imports clean ({n_ok} submodule(s) verified); core classes OK"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
