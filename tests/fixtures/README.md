# Test Fixtures

## Sample Data (for local and CI tests)

Small parquet files for dataset load and integration tests, created by
`scripts/create_sample_data.py`. The script auto-discovers models via
`get_train_config()` and downloads a raw subset from Zenodo for each dataset it
finds. **No dataset names are hardcoded** — adding a model that references a new
dataset causes that dataset's fixture to be created on the next run.

These files are **committed**, so the offline tests work on a fresh clone with no
download at all.

## Fixtures are automatic

You do not normally run anything by hand. Every target that needs data
(`make test-integration`, `make test-pipeline`, `make test-all`) depends on
`ensure-fixtures`, which:

1. asks the script which fixtures are **missing** (`--list-missing`);
2. downloads **only those**, leaving every existing fixture untouched;
3. does nothing at all when they are all present.

The presence check discovers *datasets only* and never imports a model. That is
deliberate: a model whose constructor builds a dataset would reach the network
just to answer "what am I missing?" — i.e. trigger the download the check exists
to avoid.

An existing fixture is **never** silently overwritten. Refreshing is an explicit
request:

```bash
make sample-data          # download only what is missing (safe, idempotent)
make sample-data-force    # re-download everything, overwriting what is there
FORCE_SAMPLE_DATA=1 make sample-data     # same as sample-data-force

SAMPLE_MAX_ROWS=50 make sample-data      # limit rows per fixture (default: 100)
CI_SKIP_ZENODO=1 make test-integration   # never touch the network; missing fixture = clear error
```

### Output

```
tests/fixtures/sample_data/
├── sample_fdata.parquet    # F-DATA subset (~200 rows, all columns)
└── sample_pm100.parquet    # PM100 subset (~100 rows, all columns)
```

### Which datasets get a fixture

Only datasets the script can actually slice into one parquet: those declaring
`ZENODO_URL` (single file), or `ZENODO_BASE_URL` **and** `AVAILABLE_FILES`
(monthly files). Anything else is skipped with a printed reason rather than
retried on every run — `M100Dataset` ships per-record tar archives keyed by
`AVAILABLE_FILES_PER_RECORD`, and `SCRIPTAIDataset` is served from GitHub and
needs one script file per row, so neither has a single parquet to sample.

### Usage in tests

- `tests/integration/test_model_pipeline.py` — full train → predict → save → load
  per discovered (model, dataset) pair, offline, against these fixtures
- `tests/smoke/test_smoke_data.py` — fixture load and dataset contract checks

### Manual script

```bash
python scripts/create_sample_data.py                 # missing only
python scripts/create_sample_data.py --force         # refresh everything
python scripts/create_sample_data.py --list-missing  # print what is absent, download nothing
python scripts/create_sample_data.py --max-rows 50
```

## See also

- `docs/ci-cd.md` — CI/CD reference + developer guide (running tests, adding models)
- `Makefile` — `make help` for all targets
