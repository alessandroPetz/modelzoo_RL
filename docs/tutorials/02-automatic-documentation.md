# Tutorial 2: automatic CI/CD documentation

## 1. The problem this system solves

A manually written CI document can become incorrect. For example, somebody can change a
job from warning-only to blocking in `.gitlab-ci.yml` but forget to update a table in the
documentation.

This repository reduces that problem by generating the factual tables from the files that
CI and the tests really use.

The generated output is:

```text
docs/ci-cd.md
```

Only its marked table regions are automatic. Explanatory prose outside those regions is
still maintained by people.

## 2. The complete data flow

```text
.gitlab-ci.yml
  jobs, stages, rules, commands, allow_failure
                         |
tests/conftest.py        |
  models, datasets, supported pairs
                         |----> ci/gen_capabilities_doc.py
test source files        |             |
  inventory, skips, markers            v
                         |       generated Markdown tables
ci/policy.yml            |             |
  human explanations, CRs,             v
  audit exceptions, skip notes    docs/ci-cd.md
```

There are three kinds of input:

1. `.gitlab-ci.yml` supplies **what the pipeline does**.
2. Code and tests supply **what the project supports and tests**.
3. `ci/policy.yml` supplies **why the team made a decision**.

The distinction between “what” and “why” is important. A program can read
`allow_failure: false`, but it cannot know the team's engineering reason for making that
job a hard gate.

## 3. Generated tables

The generator currently builds these named blocks:

| Block | Contents | Main source |
|---|---|---|
| `pipeline-stages` | Ordered CI stages. | `.gitlab-ci.yml` |
| `jobs` | Job, stage, trigger, command summary, and failure result. | `.gitlab-ci.yml` |
| `triggers` | Plain-language trigger descriptions. | CI rules + `ci/policy.yml` |
| `rationale` | Gate result and the reason for it. | CI gate status + policy rationale |
| `capabilities` | Model–dataset combinations and their CI state (`yes` / `no` — there is no `xfail` state). | Discovery |
| `datasets` | Discovered datasets and dummy-mode support. | Dataset classes |
| `test-inventory` | Test files and their purpose. | Test files and module docstrings |
| `test-skips` | Where tests can skip and how to remove the skip. | Test AST + policy notes |

## 4. How marked regions protect human text

An automatic region looks like this:

```markdown
<!-- BEGIN:GENERATED jobs -->
| Stage | Job | Trigger | On failure | Runs |
| ... generated rows ... |
<!-- END:GENERATED jobs -->
```

The generator replaces the text from the matching `BEGIN` marker through the matching
`END` marker. It leaves everything outside the markers unchanged.

This allows one document to contain:

- automatically maintained facts in tables;
- human-written explanations and developer instructions.

Never manually edit text inside a generated region. The next generator run will replace
it.

## 5. Important generator functions

The implementation is in `ci/gen_capabilities_doc.py`.

### Reading the CI pipeline

| Function | Basic purpose |
|---|---|
| `_load_ci()` | Reads `.gitlab-ci.yml` with PyYAML. YAML anchors are resolved during loading. |
| `_anchor_rules()` | Collects reusable hidden rule definitions such as `.rules_unit`. |
| `_trigger_label()` | Gives a job's resolved rule a readable name. |
| `_outcome()` | Converts `allow_failure` into `Warns` or `Blocks`. |
| `_script_summary()` | Selects a useful command from a job for the jobs table. |
| `_collect_jobs()` | Builds and orders the complete job list. |

The small `_outcome()` decision is the key to automatic gate status:

```python
return "Warns" if job.get("allow_failure") else "Blocks"
```

### Reading models, datasets, and tests

| Function | Basic purpose |
|---|---|
| `_load_conftest()` | Imports `tests/conftest.py` so docs reuse the same discovery as pytest. |
| `_model_task()` | Reads a model's task type for the capability table. |
| `_capabilities()` | Builds model–dataset and dataset rows. |
| `_test_inventory()` | Finds `test_*.py` files in unit, smoke, and integration directories. |
| `_first_docstring_line()` | Uses the first module-docstring line as a short test description. |
| `_scan_test_skips()` | Parses test source and finds skip calls and Zenodo markers. |

The generator reuses `MODEL_CLASSES`, `DATASET_CLASSES`, and `MODEL_DATASET_PAIRS` from
`tests/conftest.py`. This matters because the documentation and pytest see the same set of
models and datasets.

### Reading policy and writing Markdown

| Function | Basic purpose |
|---|---|
| `_load_policy()` | Reads `ci/policy.yml`. |
| `_table()` | Converts headings and rows into a Markdown table. |
| `_build_blocks()` | Combines all sources and constructs every named block. |
| `_render()` | Replaces only the matching generated regions. |
| `_scaffold()` | Creates an initial document if `docs/ci-cd.md` does not exist. |
| `main()` | Handles normal write mode and `--check` mode. |

## 6. Automatic gate-status example

Suppose this job is warning-only:

```yaml
example:quality:
  stage: lint
  script:
    - run-quality-check
  allow_failure: true
```

The generated table says:

```text
example:quality | Warns
```

Now change only this line:

```yaml
allow_failure: false
```

Run:

```bash
make docs-ci
```

The generator reads the new value, `_outcome()` returns `Blocks`, and the generated jobs
and rationale tables change automatically:

```text
example:quality | Blocks
```

Use this to review the exact result:

```bash
git diff docs/ci-cd.md
```

No person should manually replace `Warns` with `Blocks` in the generated table.

## 7. What is not automatic

The gate result is automatic, but the human reason is not. If a job changes from warning to
blocking, update its rationale under `jobs:` in `ci/policy.yml`.

For example:

```yaml
jobs:
  example:quality:
    rationale: "This check now blocks because ..."
```

The generator can prove that the YAML says “Blocks.” It cannot prove that the sentence
explaining the decision is true. Reviewers must check that semantic relationship.

The following also remain human responsibilities:

- prose outside generated markers;
- explanations under `rules:` and `jobs:` in `ci/policy.yml`;
- deciding whether an expected failure is justified;
- deciding whether a vulnerability may temporarily be ignored;
- adding `skip_notes` for implicit skips that source scanning cannot detect.

## 8. Write mode and check mode

### Write mode

```bash
make docs-ci
```

This runs the generator normally and writes the new marked regions to `docs/ci-cd.md`.

Equivalent direct command:

```bash
.venv/bin/python ci/gen_capabilities_doc.py
```

### Check mode

```bash
make docs-ci-check
```

This generates the expected document in memory and compares it with the committed file:

- same content: print that the document is current and exit with code `0`;
- different content: print that the document is stale and exit with code `1`.

Check mode does not repair the file. Run `make docs-ci` to repair it.

## 9. How GitLab keeps the document current

### On a branch or merge request: `docs:check`

When relevant source, test, CI, or policy files change, GitLab runs:

```bash
poetry run python ci/gen_capabilities_doc.py --check
```

This tells the author and reviewers if the committed generated tables are stale. The job is
currently `allow_failure: true`, so it warns but does not block merging.

### On the default branch: `docs:autogen`

When `DOCS_BOT_TOKEN` is configured, this job:

1. installs the required environment;
2. runs the generator in write mode;
3. checks `git diff -- docs/ci-cd.md`;
4. does nothing when there is no difference;
5. otherwise commits only `docs/ci-cd.md`;
6. pushes the commit to the default branch.

The bot commit uses `[skip ci]` and the push uses `-o ci.skip`. This prevents the automatic
documentation commit from starting another pipeline.

The generated blocks contain no timestamp. Therefore, the same repository state always
produces the same output, and an unchanged run creates no commit.

### Required token

`DOCS_BOT_TOKEN` must be a masked GitLab CI/CD variable containing a Project Access Token
with `write_repository`. Its user must be allowed to push to the protected default branch.
Without this variable, `docs:autogen` does not run.

## 10. How different changes reach the document

| Change | Edit this source | Generated result |
|---|---|---|
| Add, remove, or rename a CI job | `.gitlab-ci.yml` | Jobs and rationale rows change. |
| Change `allow_failure` | `.gitlab-ci.yml` | `Blocks`/`Warns` changes automatically. |
| Change a job command | `.gitlab-ci.yml` | Command summary changes. |
| Change a job trigger | `.gitlab-ci.yml` | Trigger column changes. |
| Explain a gate decision | `ci/policy.yml` | “Why” text changes. |
| Add a model | Model file under `models/tasks/` | Model and capability rows appear after discovery. |
| Change supported datasets | Model's `supported_datasets` | Pair rows change. |
| Add a dataset | Dataset module | Dataset inventory changes. |
| Add a test file | `tests/unit`, `tests/smoke`, or `tests/integration` | Test inventory changes. |
| Add a literal pytest skip | Test source | Skip inventory changes. |
| Record a defect owed by a model author | `core_change_requests` in policy | It is published with owner + evidence; the failing test stays red. |
| Add an audit exception | `audit_ignore` in policy | Policy and audit behaviour use the same entry. |

## 11. Safe workflow for a CI change

Example: changing whether a test job blocks.

1. Edit `allow_failure` in `.gitlab-ci.yml`.
2. Update the matching explanation under `jobs:` in `ci/policy.yml`.
3. Run `make docs-ci`.
4. Run `git diff -- .gitlab-ci.yml ci/policy.yml docs/ci-cd.md`.
5. Confirm the gate status and explanation agree.
6. Run `make docs-ci-check`.
7. Commit all three files together.

Example: adding a model.

1. Add the model below `seanergys_modelzoo/models/tasks/`.
2. Declare `supported_datasets` and implement `get_train_config()`.
3. Run smoke and integration tests.
4. Run `make docs-ci`.
5. Confirm the new model–dataset rows appear in `docs/ci-cd.md`.
6. Commit the model, tests, and generated document together.

## 12. Troubleshooting

### `make docs-ci-check` says the file is stale

Run:

```bash
make docs-ci
git diff docs/ci-cd.md
```

Review and commit the generated changes.

### A generated table has the wrong gate status

Check `allow_failure` on that job in `.gitlab-ci.yml`. The generated table follows the
pipeline YAML, not the rationale text.

### The gate status is correct but the reason is wrong

Update the matching `jobs.<job-name>.rationale` entry in `ci/policy.yml`, then regenerate.

### A model or dataset is missing

Check discovery first:

- the model is a concrete subclass below `models/tasks/`;
- the dataset is a concrete subclass below `datasets/`;
- imports succeed with the CI dependency group installed;
- the model's `supported_datasets` is populated;
- the dataset has a non-empty `DATASET_ID` when it should form an integration pair.

### `docs:autogen` does not run or cannot push

Check that `DOCS_BOT_TOKEN` exists, has `write_repository`, is available on the protected
branch, and belongs to a user allowed to push there. The job is warn-only, so inspect its
log even when the overall pipeline is green.

## 13. Final mental model

Remember these four points:

1. `.gitlab-ci.yml` owns pipeline behaviour and gate status.
2. Code and `tests/conftest.py` own discovered capabilities.
3. `ci/policy.yml` owns human explanations and approved exceptions.
4. `ci/gen_capabilities_doc.py` combines them and replaces only marked tables.

That is how the documentation follows the real CI/CD system without requiring somebody to
rewrite every table by hand.
