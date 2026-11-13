# `archive/` — retired CI and test material

This directory is where a CI, test or helper file goes when it is **superseded but
worth keeping for reference**. Nothing here is imported, collected by pytest, run
by the Makefile, or referenced by `.gitlab-ci.yml`; `ci/policy.yml → name_scan`
exempts it from the hardcoded-name scan for exactly that reason.

## Why it exists

The test suite's central promise is that a developer adds a model or a dataset
and CI exercises it **without anyone editing a test**. That only holds while no
pipeline file, Makefile target, or test keys its behaviour on a concrete model or
dataset name. Two guards enforce it:

| Guard | Scope |
|---|---|
| `tests/unit/generic/test_no_hardcoded_names.py` | `tests/unit/generic/` — the discovery-driven suite |
| `tests/unit/generic/test_no_hardcoded_names_repo.py` | `.gitlab-ci.yml`, `Makefile`, `ci/**/*.py`, `scripts/**/*.py`, `tests/**/*.py` |

When a file cannot pass the scan and cannot reasonably be made
discovery-driven, there are four destinations, in order of preference:

1. **Rewrite it against auto-discovery** — `MODEL_CLASSES` / `DATASET_CLASSES`,
   `supported_datasets`, `get_train_config()`. This is almost always possible.
2. **`tests/unit/specific/`** — if it genuinely tests one component's internals.
   Naming the component there is the point, and the scan skips that directory.
3. **`ci/policy.yml`** — if the file is making a *statement about* a component
   (known-broken pair, out-of-scope dataset, a change owed by the core authors)
   rather than testing it. Policy is the CI-owned home for names.
4. **`archive/`** — only if it is dead: superseded by a generic replacement, or
   written against an API that no longer exists.

## Audit — 2026-09-13

Every surface listed above was scanned for the names produced by discovery
(model classes, dataset classes, `DATASET_ID` values, and the derived
`sample_*.parquet` fixture names). **Executable content only**: Python is parsed
with `ast`, so comments are invisible and docstrings are skipped; YAML and the
Makefile are scanned with comment lines stripped. Prose that explains the naming
rule is documentation and keeps working when a dataset is added; `if name ==
"…":` does not.

Result: **one file was archived** (`example_model_train.py`, below); every CI and
test surface was already name-agnostic in code.

| Surface | Verdict |
|---|---|
| `.gitlab-ci.yml` | Clean. No per-model or per-dataset job; every test job runs a directory or a marker. |
| `Makefile` | Clean. Fixture targets go through `scripts/create_sample_data.py`, which discovers datasets itself. |
| `ci/*.py` | Clean. `utils.py` discovers by base class; `gen_capabilities_doc.py` builds the model×dataset matrix from discovery. |
| `scripts/create_sample_data.py` | Clean in code; names appear only in the module docstring as examples of the naming rule. |
| `tests/` outside `unit/specific/` | Clean. Everything is parametrized over `MODEL_CLASSES` / `DATASET_CLASSES` / `MODEL_DATASET_CONFIGS`. |
| `tests/unit/specific/` | **Kept, not archived** — see below. |
| `scripts/example_data_access.py` | **Kept, exempted** — a user-facing example; naming the dataset is its content. |
| `scripts/example_model_train.py` | **Archived here** — broken against the current model API. |

### Why `tests/unit/specific/` was kept

These six modules do name components, and that is their purpose: they cover
offline branches — file/node-spec parsers, tar-extraction safety, preprocessing
and download hooks, save/load error paths — that no generic contract can reach,
because reaching them requires knowing the component's internals.

Archiving them was measured before being rejected. On `main` @ `2fa549c`,
`pytest tests/ -m "not zenodo"`:

| Suite | Result | Statement coverage of `seanergys_modelzoo` |
|---|---|---|
| Full | 269 passed | **82%** |
| Without `tests/unit/specific/` | 157 passed | **61%** |

Removing them would delete 112 tests and 21 percentage points of coverage — a
real loss of verification to satisfy a convention whose purpose is to *increase*
verification. They stay, in the directory that exists for them, exempted in
policy with that reason recorded.

### `example_model_train.py` — archived

`scripts/example_model_train.py` was the one file that genuinely qualified: it is
written against an API the models no longer have, so it cannot run at all.

```python
train_config_dict = JPCP.get_train_config("f-data", get_dummy_data=True)
model_obj, dataloader = train_config_dict          # line 7
```

`get_train_config()` returns a **dict** keyed by dataset id, whose value is a
3-tuple `(model, dataset_cls, (ds_params, dl_params))` — it does not return a
`(model, dataloader)` pair, and it never builds a dataloader. Measured on
`main` @ `2fa549c`:

```
$ python -c "...JPCP.get_train_config('f-data', get_dummy_data=True)..."
type: dict   keys: ['f-data']   value tuple len: 3
unpack raises: ValueError not enough values to unpack (expected 2, got 1)
```

The next line has the same problem: `get_test_config()` also returns a dict, and
the example passes it straight to `evaluate()` as if it were a dataloader.

It is archived rather than rewritten because a corrected version cannot be
verified end to end right now — the example trains on live data, and
`zenodo.org` is returning `HTTP 504` from this network (checked 2026-09-13,
record 10127767). Shipping an unverified "fix" for a user-facing example would
just replace one broken example with an unproven one. The working, exercised
equivalent of what it demonstrates lives in
`tests/integration/test_model_pipeline.py`, which runs config → load → train →
predict → save → load for **every** discovered (model, dataset) pair against the
committed fixtures.

`scripts/example_data_access.py` was left in place: it constructs the datasets
directly and matches the current API, so it needs only network to run.

## Restoring something from here

Nothing is deleted by archiving, so recovering a file is `git mv` back and a run
of the suite. If it still names a component, it must land in
`tests/unit/specific/` and a matching exemption must exist in
`ci/policy.yml → name_scan.exempt`, with a reason — otherwise the scan will
reject it, which is the intended outcome.
