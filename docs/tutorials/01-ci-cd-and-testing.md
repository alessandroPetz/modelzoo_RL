# Tutorial 1: CI/CD and testing in the Model Zoo

## 1. What CI/CD means

**CI** means Continuous Integration. When somebody pushes code or opens a merge request,
GitLab automatically checks the change. The purpose is to find problems before the change
is merged.

**CD** means Continuous Delivery or Continuous Deployment. It normally includes packaging
or releasing software after CI succeeds. In this repository, the pipeline builds and tests
the Python wheel. It does not automatically publish the wheel to a package registry.

The pipeline definition is at the repository root:

```text
.gitlab-ci.yml
```

GitLab reads this file and creates the pipeline jobs.

## 2. Important words

| Word | Simple meaning |
|---|---|
| Pipeline | The complete set of automatic checks for one change. |
| Stage | A group of related jobs, such as `test` or `build`. |
| Job | One command or group of commands run by GitLab. |
| Gate | A check that must pass. If it fails, the pipeline fails. |
| Warn-only job | A useful check whose failure is visible but does not fail the pipeline. |
| Artifact | A file kept by GitLab after a job, such as coverage XML or a wheel. |
| Merge request (MR) | A request to review and merge a branch. |
| Default branch | The main shared branch after a merge. |

In `.gitlab-ci.yml`, gate status is controlled by `allow_failure`:

```yaml
allow_failure: false  # hard gate: failure blocks the pipeline
allow_failure: true   # warning: failure is visible but does not block
```

This value is also the source used by the automatic documentation. There is no second,
manually maintained list of gate statuses.

## 3. What happens after a push

The stages run in this order:

```text
change pushed
    |
    v
sanity -> validate -> lint -> test -> coverage -> docs -> build -> test-wheel
    |
    v
pipeline result: passed, passed with warnings, or failed
```

GitLab can run independent jobs in the same stage at the same time. A later stage normally
starts after the required jobs in earlier stages have completed.

The workflow rules near the top of `.gitlab-ci.yml` avoid duplicate pipelines. If a branch
has an open merge request, the merge-request pipeline is used. On the default branch, the
correctness checks always run. On other branches, `rules:changes` starts only the jobs whose
inputs may have changed.

## 4. Pipeline stages and jobs

The following is a beginner-friendly description. For the always-current table, see
[`../ci-cd.md`](../ci-cd.md).

### Stage 1: `sanity`

These are cheap checks that run on every pipeline.

| Job | What it checks | Current result on failure |
|---|---|---|
| `sanity:python-syntax` | Compiles Python files to find syntax errors. | Blocks |
| `sanity:check-structure` | Confirms important files and directories exist. | Blocks |
| `sanity:check-secrets` | Runs `ci/check_secrets.sh` to detect possible committed secrets. | Blocks |

### Stage 2: `validate`

| Job | What it checks | Current result on failure |
|---|---|---|
| `validate:yaml-json` | Uses `ci/validate_yaml.py` to parse YAML and JSON files. | Blocks |

This catches an invalid `.gitlab-ci.yml` or policy file early.

### Stage 3: `lint`

| Job | Tool | What it checks | Current result on failure |
|---|---|---|---|
| `lint:python` | Ruff | Common Python mistakes and style rules. | Warns |
| `lint:format` | Ruff | Whether files follow the required format. | Warns |
| `lint:mypy` | mypy | Type consistency. | Warns |

These jobs currently give quality warnings. They do not replace the hard correctness tests.

### Stage 4: `test`

| Job | Location or tool | What it checks | Current result on failure |
|---|---|---|---|
| `test:unit` | `tests/unit/` | Small components in isolation. | Blocks |
| `test:smoke-models` | `tests/smoke/test_smoke_models.py` | Model discovery, construction, and training configuration. | Blocks |
| `test:smoke-data` | `tests/smoke/test_smoke_data.py` | Dataset discovery, contracts, and local fixture loading. | Blocks |
| `test:integration` | `tests/integration/` | Full offline model and dataset lifecycle. | Blocks |
| `test:zenodo` | Zenodo-marked integration tests | Real small downloads and a real lifecycle. | Warns |
| `test:audit` | `pip-audit` | Known vulnerabilities in dependencies. | Warns |

The offline tests use files under `tests/fixtures/sample_data/`. Only `test:zenodo` should
contact Zenodo. This separation keeps the main correctness tests stable when an external
service is unavailable.

### Stage 5: `coverage`

Each main test job writes a separate coverage data file. `coverage:report` combines the
available files and creates:

```text
coverage/cobertura-coverage.xml
```

GitLab stores it as a coverage artifact. The job uses optional dependencies, so it can
combine the results that exist even when a test job was correctly skipped by its change
rules.

### Stage 6: `docs`

| Job | Where it runs | What it does | Current result on failure |
|---|---|---|---|
| `docs:check` | Branches and merge requests | Reports if `docs/ci-cd.md` is stale. | Warns |
| `docs:autogen` | Default branch | Regenerates and commits changed tables. | Warns |

These jobs are explained fully in
[Tutorial 2](02-automatic-documentation.md).

### Stage 7: `build`

`build:wheel` runs `poetry build --format wheel`. The resulting `.whl` file is kept in
`dist/` as an artifact. A package that cannot be built is not ready to deliver, so this job
blocks.

### Stage 8: `test-wheel`

`test:wheel` installs the built wheel and runs `ci/verify_wheel.py`. It clears `PYTHONPATH`
so Python imports the installed wheel, not the source directory by accident. The verifier
walks the package modules dynamically. This job blocks because consumers need a package
that installs and imports correctly.

## 5. The three test levels

### Unit tests: one component at a time

Location:

```text
tests/unit/
```

They check configurators, dataloaders, dataset parsing, metadata, logger behaviour, model
base classes, concrete models, saving and loading, and error cases. They should be fast and
should normally avoid network access.

`tests/unit/` has three homes, chosen by what a test is actually testing:

| Home | What goes there | May name a concrete model/dataset? |
|---|---|---|
| `tests/unit/` (root) | Shared/common framework code with nothing to discover — one configurator, one dataloader, one logger, one metadata class, the generic sklearn wrapper, `ci/utils.py`. Plain unit tests, not parametrized over discovery. | No |
| `tests/unit/generic/` | Contract tests parametrized over `MODEL_CLASSES`/`DATASET_CLASSES` (see §6). Adding a model or dataset to the repo extends these automatically, with no test edit. | No |
| `tests/unit/specific/` | Tests that deliberately target one concrete model or dataset (e.g. one component's preprocessing, parsing quirks, or save/load error paths that a generic contract cannot reach). Naming the component here is the point. | Yes |

The root and `generic/` homes look similar (neither names a component) but differ in *why*:
root tests have no discovery loop to parametrize over — there is exactly one `SeanergysLogger`
— while `generic/` tests are defined by iterating discovery output, so the suite grows with the
repo. A test belongs in `specific/` the moment it imports or names one concrete model/dataset
class, a `DATASET_ID`, or a fixture filename; `tests/unit/generic/test_no_hardcoded_names.py`
and `tests/unit/generic/test_no_hardcoded_names_repo.py` enforce this split (see below).

### Smoke tests: can the main objects be found and created?

Locations:

```text
tests/smoke/test_smoke_models.py
tests/smoke/test_smoke_data.py
```

Important model smoke functions include:

- `test_model_discovery()` — at least one model was found.
- `test_model_instantiation()` — every discovered model can be constructed.
- `test_model_get_train_config()` — the training configuration has the expected shape.

Important dataset smoke functions include:

- `test_dataset_discovery()` — dataset classes were found.
- `test_dataset_contract_via_model_config()` — a dataset follows the common interface.
- `test_dataset_fixture_has_data()` — committed test data is present and usable.
- the `is_dummy` and Zenodo tests — small real-data loading works where supported.

### Integration tests: does the complete workflow work?

Main file:

```text
tests/integration/test_model_pipeline.py
```

`test_get_train_config_structure()` checks the structural contract without loading data.

`test_full_pipeline_offline()` then checks this lifecycle for every discovered pair:

```text
get config
  -> load fixture
  -> check dataset
  -> create dataloader
  -> train
  -> predict
  -> save
  -> reload
  -> predict again
  -> compare predictions
  -> check metadata
```

`test_full_pipeline_zenodo()` checks a smaller version using a real `is_dummy=True` Zenodo
download. It is selected with the `zenodo` pytest marker.

## 6. How models and datasets enter CI automatically

The central discovery file is:

```text
tests/conftest.py
```

Its important items are:

| Name | Meaning |
|---|---|
| `_collect_model_classes()` | Scans `seanergys_modelzoo/models/tasks/` for concrete model subclasses. |
| `_collect_dataset_classes()` | Scans `seanergys_modelzoo/datasets/` for concrete dataset subclasses. |
| `MODEL_CLASSES` | All discovered model classes. |
| `DATASET_CLASSES` | All discovered dataset classes. |
| `_collect_model_dataset_pairs()` | Reads each model's `supported_datasets`. |
| `MODEL_DATASET_PAIRS` | Every `(model class, dataset ID)` combination to test. |

The smoke and integration tests use these collections as pytest parameters. Therefore,
contributors do not maintain a separate hardcoded list in the pipeline.

This is enforced rather than merely encouraged. `tests/unit/generic/test_no_hardcoded_names.py`
and `tests/unit/generic/test_no_hardcoded_names_repo.py` scan the pipeline file, the Makefile,
the `ci/` helpers, the scripts and the tests for the names discovery produces — model and
dataset class names, `DATASET_ID` values, and the `sample_*.parquet` fixture names — and fail
the unit job if one appears in executable code. Comments and docstrings are not scanned, so
explaining a rule in prose stays free. A name is allowed only in `tests/unit/specific/`
(tests that deliberately target one component), in `ci/policy.yml` (records about a component),
and in `archive/` (retired material). The list of scanned paths and the allowlist live in
`ci/policy.yml → name_scan`.

A related guarantee: `test_every_supported_pair_is_exercised_or_declared` requires every model
to produce a working configuration for **every** dataset in its `supported_datasets`, not just
its default one. A model that advertises three datasets and only works with one now fails,
instead of quietly contributing a single test case.

For a new model to participate correctly:

1. Put its module below `seanergys_modelzoo/models/tasks/`.
2. Subclass the appropriate common model class.
3. Set `supported_datasets`.
4. Implement `get_train_config(dataset_id)`.
5. Make the returned dataset parameters work with small local fixtures and, where relevant,
   `is_dummy=True`.
6. Run the local checks and regenerate the CI documentation.

## 7. Currently discovered support

The current model–dataset combinations are:

| Model | Dataset | Dataset ID | CI state |
|---|---|---|---|
| JPCP | PM100Dataset | `pm100` | Tested; must pass |
| JPCP | FDataDataset | `f-data` | Tested; must pass |
| MACK | FDataDataset | `f-data` | Tested; **currently failing** (issue #10) |
| MCBound | FDataDataset | `f-data` | Tested; **currently failing** (issue #10) |

The discovered datasets also include `M100Dataset` and `SCRIPTAIDataset`, but they are not
currently connected to a model through a non-empty `DATASET_ID` pair. Unit tests still
cover these dataset classes.

The live capability table in [`../ci-cd.md`](../ci-cd.md) is authoritative. This tutorial
is a teaching snapshot and may change less often.

### There are no expected failures

This repository has **no `xfail`**. The `integration_xfail`, `model_xfail` and
`config_load_xfail` lists were removed on 2026-09-13, and no test carries
`pytest.mark.xfail`.

A strict `xfail` runs the test, lets it fail, reports `XFAIL`, and keeps the exit code at
`0` — so the pipeline goes green over code that does not work. The entries were accurate and
documented, and the pipeline still reported "pass" while three (model, dataset) pairs could
not load their own data and one model could not be constructed at all.

So today: a model that cannot be built, a pair that cannot load its fixture, and a pair that
cannot finish the lifecycle are all plain **failures**, and `main` stays red until the model
authors fix them. What each failure is, who owns it and what evidence backs it lives in
`ci/policy.yml → core_change_requests`, which is published in
[`../ci-cd.md`](../ci-cd.md#core-change-requests). Recording a request documents the defect —
it never makes the test pass.

## 8. Files colleagues should know

| Location | Purpose |
|---|---|
| `.gitlab-ci.yml` | Pipeline stages, jobs, commands, trigger rules, and hard/warn status. |
| `pyproject.toml` | Python dependencies and tool configuration. |
| `poetry.lock` | Exact resolved dependency versions. |
| `Makefile` | Short local commands that mirror common CI actions. |
| `tests/conftest.py` | Automatic model and dataset discovery. |
| `tests/unit/` | Unit tests of shared/common framework code (no model or dataset named). |
| `tests/unit/generic/` | Discovery-parametrized contract tests (auto-extend to new models/datasets). |
| `tests/unit/specific/` | Tests that deliberately target one concrete model or dataset. |
| `tests/smoke/` | Discovery and construction tests. |
| `tests/integration/` | Full lifecycle tests. |
| `tests/fixtures/` | Small committed data used by offline tests. |
| `ci/policy.yml` | Human explanations, expected failures, audit exceptions, and skip notes. |
| `scripts/ensure_tools.sh` | Machine toolchain check and bootstrap (uv, Python, poetry). |
| `ci/gen_capabilities_doc.py` | Automatic CI documentation generator. |
| `ci/validate_yaml.py` | YAML and JSON validation. |
| `ci/verify_wheel.py` | Installed-wheel verification. |
| `ci/check_secrets.sh` | Basic secret scan. |
| `docs/ci-cd.md` | Generated current reference plus human-written guidance. |

## 9. Commands to run locally

First-time setup: there is none. The only things your machine needs are `curl`, `git` and
`make`. Every command below installs whatever it is missing before it runs — `uv`, a
supported Python (3.12–3.14), `poetry`, the `.venv/` virtual environment, and all locked
dependencies. Nothing is installed system-wide and no `sudo` is used; the tools land in
`~/.local/bin` and the packages in `.venv/`.

So on a fresh clone this is enough:

```bash
make test
```

If you prefer to do the setup as an explicit step, or want to inspect it:

```bash
make bootstrap      # toolchain + .venv + all dependencies, no tests
make doctor         # report what is present, change nothing
make tools-check    # check the toolchain and FAIL instead of installing
```

The setup is guarded, so it is cheap to repeat: `poetry install` re-runs only when
`pyproject.toml` or `poetry.lock` actually changed, and the toolchain step is a handful of
`command -v` checks when everything is already there.

On a machine where installing tools is not allowed, set `MODELZOO_NO_AUTO_INSTALL=1` and the
setup step will only report what is missing:

```bash
MODELZOO_NO_AUTO_INSTALL=1 make test
```

The integration fixtures are committed to the repository, so the offline tests work
immediately. If one is ever missing, you do not have to do anything: the test session checks
every discovered dataset at start-up and downloads the missing sample itself
(`tests/fixture_provisioning.py`), whether you run `make test` or `pytest` directly. Existing
fixtures are never overwritten.

```bash
make sample-data                       # refresh what is missing, explicitly
MODELZOO_FIXTURES=force make test      # re-download every fixture first
MODELZOO_FIXTURES=never make test      # never touch the network (what CI does)
```

CI runs with `MODELZOO_FIXTURES=never`, because downloading a full Zenodo file on a runner
exhausts its memory. There, a missing fixture fails the unit job with a message telling you to
run `make sample-data` locally and commit the result.

During normal development:

```bash
make format          # fix formatting
make pre-push        # fast local gate
make test-all        # unit + smoke + integration
make docs-ci         # regenerate CI tables after relevant changes
make docs-ci-check   # confirm generated tables are current
```

To simulate the complete GitLab pipeline locally:

```bash
make ci
```

Useful focused commands:

```bash
make test-unit
make test-smoke-models
make test-smoke-data
make test-integration
make test-wheel
```

## 10. Normal merge-request checklist

0. Nothing to set up — any `make` target below prepares the environment itself.
1. Make the code or CI change.
2. Add or update tests.
3. Run `make format` and `make pre-push`.
4. Run `make test-all` for model, dataset, or pipeline behaviour changes.
5. If CI structure, policy, tests, models, or datasets changed, run `make docs-ci`.
6. Review `git diff`, especially the generated `docs/ci-cd.md` table changes.
7. Run `make docs-ci-check`.
8. Commit the implementation, tests, and generated documentation together.
9. Open the merge request and review every blocking failure and warning.

Continue with [Tutorial 2](02-automatic-documentation.md) to understand how the generated
tables and automatic gate status work.
