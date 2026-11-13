# seanergys-modelzoo Makefile
# Run `make help` for usage

.PHONY: help bootstrap tools tools-check env venv install install-ci install-hooks \
        sample-data sample-data-force ensure-fixtures \
        lock deps-check doctor \
        test test-unit test-smoke-models test-smoke-data test-integration test-pipeline test-all test-wheel \
        lint format format-check typecheck audit check-secrets pre-push ci \
        docs-ci docs-ci-check \
        coverage clean clean-venv clean-fixtures clean-tools clean-all

VENV ?= .venv
PYTHON ?= $(VENV)/bin/python
PYTEST ?= $(VENV)/bin/pytest
export PYTHONPATH := $(shell pwd):$(PYTHONPATH)

# Everything here shares one .venv, so serialise even under `make -j`.
.NOTPARALLEL:

# uv and poetry install themselves into ~/.local/bin. Put it on PATH for every
# recipe so a tool installed by `make tools` is usable in the same make run,
# even if the user's shell rc does not export it yet.
export PATH := $(HOME)/.local/bin:$(PATH)

# Poetry consults the system keyring for package-repository credentials. Over SSH,
# in a container, or on any session without an unlocked DBus secret service, that
# lookup raises DBusErrorResponse / ItemNotFoundException and aborts the install
# half-way ("Cannot install httpx"). Every dependency here comes from public PyPI,
# so the null backend is the correct choice and makes the install deterministic
# across desktop, headless, and CI. Override it if you ever add a private index.
PYTHON_KEYRING_BACKEND ?= keyring.backends.null.Keyring
export PYTHON_KEYRING_BACKEND

# Never block on a prompt in an automated run.
POETRY_NO_INTERACTION ?= 1
export POETRY_NO_INTERACTION

# Stamp file written after a successful `poetry install`. It lives inside .venv/
# so deleting the venv also invalidates it. It depends on pyproject.toml and
# poetry.lock, so editing either triggers a re-sync on the next make command.
DEPS_STAMP := $(VENV)/.deps-installed

# Interpreter chosen by scripts/ensure_tools.sh (system python3 when it is in the
# supported range, otherwise a uv-managed one). Read at recipe time, not parse
# time, because `make tools` may create the file during this same run.
VENV_PYTHON = $$(cat .make/python 2>/dev/null || echo python3)

# Committed fixtures used by the offline integration tests.
FIXTURES_DIR := tests/fixtures/sample_data

# -----------------------------------------------------------------------------
# Help
# -----------------------------------------------------------------------------
help:
	@echo ""
	@echo "seanergys-modelzoo — available commands"
	@echo "========================================"
	@echo ""
	@echo "┌─────────────────────────────────────────────────────────┐"
	@echo "│  BEFORE YOU PUSH — run these steps every time:         │"
	@echo "│                                                         │"
	@echo "│  1. make format       Auto-fix code style              │"
	@echo "│  2. make pre-push     Lint + secrets + fast tests      │"
	@echo "│  3. make ci           FULL local mirror of GitLab CI   │"
	@echo "│                       (run this last — if it's green,  │"
	@echo "│                        the GitLab pipeline will be too)│"
	@echo "│                                                         │"
	@echo "│  Fresh clone?  make bootstrap  (or just: make test)    │"
	@echo "│  One-time setup: make install-hooks                    │"
	@echo "│  (runs pre-push automatically on every git push)       │"
	@echo "└─────────────────────────────────────────────────────────┘"
	@echo ""
	@echo "Setup — you do NOT have to run these by hand."
	@echo "Every test/lint/docs target depends on 'env', so 'make test' on a fresh"
	@echo "clone installs the toolchain, creates .venv and syncs dependencies itself."
	@echo ""
	@echo "  make bootstrap       One-shot setup: toolchain + .venv + all dependencies"
	@echo "  make doctor          Report toolchain/.venv/fixture state, change nothing"
	@echo "  make tools           Ensure uv, Python 3.12-3.14 and poetry exist (installs if not)"
	@echo "  make tools-check     Same checks, never installs — fails if anything is missing"
	@echo "  make env             Ensure .venv exists and matches poetry.lock (dev + ci)"
	@echo "  make venv            Create the .venv virtual environment only"
	@echo "  make install         .venv + dev dependencies only (tooling; tests need dev+ci)"
	@echo "  make install-ci      .venv + all deps including the ML libs (same as 'make env')"
	@echo "  make install-hooks   Install git pre-push hook (auto-runs make pre-push)"
	@echo "  make sample-data       Download any MISSING fixture (keeps existing ones)"
	@echo "  make sample-data-force Re-download every fixture, overwriting what is there"
	@echo "  make ensure-fixtures   Fetch missing fixtures only if needed (auto-run by tests)"
	@echo ""
	@echo "Dependencies (poetry.lock <-> pyproject.toml):"
	@echo "  make deps-check      Verify poetry.lock is in sync (runs before every test)"
	@echo "  make lock            Regenerate poetry.lock after editing pyproject.toml"
	@echo ""
	@echo "Fixtures are automatic, from both sides: any target that needs data runs"
	@echo "ensure-fixtures first, AND the test session itself checks every discovered"
	@echo "dataset at start-up and fetches what is missing (tests/fixture_provisioning.py),"
	@echo "so a bare 'pytest' behaves like 'make test'. Only missing fixtures are"
	@echo "downloaded; nothing is overwritten unless you ask with sample-data-force."
	@echo ""
	@echo "Before every push:"
	@echo "  make format          Auto-fix code formatting (run this first)"
	@echo "  make format-check    Check formatting without modifying files"
	@echo "  make pre-push        Full pre-push gate: format-check + lint + secrets + tests"
	@echo ""
	@echo "Tests:"
	@echo "  make test                Unit + smoke tests (fast, no data loading)"
	@echo "  make test-unit           Unit tests + terminal coverage — tests/unit/"
	@echo "  make test-smoke-models   Model smoke tests (discovery, instantiation, no data)"
	@echo "  make test-smoke-data     Data smoke tests (fixture loading, no Zenodo)"
	@echo "  make test-integration    Full integration tests — tests/integration/"
	@echo "  make test-all            Everything — unit + smoke + integration"
	@echo "  make test-wheel          Build wheel + verify it installs and imports"
	@echo ""
	@echo "Code quality:"
	@echo "  make lint            Ruff lint check"
	@echo "  make typecheck       Mypy type check (informational, non-blocking)"
	@echo "  make coverage        Unit tests + XML/HTML/terminal coverage report"
	@echo "  make audit           Dependency vulnerability scan (pip-audit)"
	@echo "  make check-secrets   Scan for secrets in code"
	@echo ""
	@echo "Docs (auto-generated CI capability tables):"
	@echo "  make docs-ci         Regenerate docs/ci-cd.md from .gitlab-ci.yml + code + policy"
	@echo "  make docs-ci-check   Fail if docs/ci-cd.md is stale (used by make ci)"
	@echo ""
	@echo "CI simulation:"
	@echo "  make ci              Run everything exactly as GitLab CI does"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean           Remove caches and build artifacts"
	@echo "  make clean-tools     Forget the cached interpreter choice (.make/)"
	@echo "  make clean-venv      Remove .venv/ (any make target recreates it)"
	@echo "  make clean-fixtures  Remove fixture parquet files (forces re-download)"
	@echo "  make clean-all       clean + clean-tools + clean-venv + clean-fixtures"
	@echo ""
	@echo "Options:"
	@echo "  CI_SKIP_ZENODO=1 make test-integration     Never touch the network"
	@echo "  MODELZOO_FIXTURES=never make test          Never auto-download a fixture"
	@echo "  MODELZOO_FIXTURES=force make test          Refresh every fixture first"
	@echo "  SAMPLE_MAX_ROWS=50 make sample-data        Limit rows per fixture (default: 100)"
	@echo "  FORCE_SAMPLE_DATA=1 make sample-data       Refresh fixtures already on disk"
	@echo "  MODELZOO_NO_AUTO_INSTALL=1 make test       Never install machine tools, only report"
	@echo "  PYTHON_BIN=/usr/bin/python3.13 make venv   Force the interpreter used for .venv"
	@echo ""

# -----------------------------------------------------------------------------
# Setup
#
# Three layers, each guarded so it only does work when something is actually
# missing. Every test / lint / docs target depends on `env`, so a fresh clone
# needs no manual setup — `make test` bootstraps the machine, builds .venv and
# installs the locked dependencies on its own.
#
#   tools  → curl/git/make present; uv, a supported Python, and poetry installed
#            into ~/.local/bin if absent            (scripts/ensure_tools.sh)
#   $(VENV) → the virtual environment itself
#   env    → .venv synced to poetry.lock (dev + ci groups), lock verified
# -----------------------------------------------------------------------------

# One-shot setup for a fresh machine or a fresh clone.
bootstrap: env
	@echo ""
	@echo "✓ Bootstrap complete — .venv/ is ready with dev + ci dependencies."
	@echo ""
	@echo "  Optional next steps:"
	@echo "    make install-hooks   run the pre-push gate automatically on git push"
	@echo "    make sample-data     fetch any missing Zenodo test fixture (most are committed)"
	@echo ""
	@echo "  Then: make test"

# Machine toolchain: curl, git, make, uv, Python 3.12–3.14, poetry.
# Installs the missing user-level tools (uv, poetry, a uv-managed Python) into
# ~/.local/bin — no sudo. Set MODELZOO_NO_AUTO_INSTALL=1 to only report.
tools:
	@bash scripts/ensure_tools.sh

# Same checks, never installs anything. Non-zero exit if the toolchain is
# incomplete — use it in environments where installing is not allowed.
tools-check:
	@bash scripts/ensure_tools.sh --check

# Report the toolchain + environment state without changing anything.
doctor: tools-check
	@echo "→ Virtual environment"
	@if [ -x "$(PYTHON)" ]; then \
	    echo "  ✓ $(VENV)/ — $$($(PYTHON) --version 2>&1)"; \
	else \
	    echo "  ✗ $(VENV)/ missing — run 'make bootstrap'"; \
	fi
	@if [ -f "$(DEPS_STAMP)" ]; then \
	    echo "  ✓ dependencies synced to poetry.lock"; \
	else \
	    echo "  ! dependencies not synced — the next make target will install them"; \
	fi
	@echo "→ Test fixtures"
	@if [ -n "$$(ls -A $(FIXTURES_DIR) 2>/dev/null)" ]; then \
	    echo "  ✓ $(FIXTURES_DIR)/ ($$(ls $(FIXTURES_DIR) | wc -l) files)"; \
	else \
	    echo "  ! $(FIXTURES_DIR)/ empty — run 'make sample-data'"; \
	fi

# Real file target — Make skips creation if .venv/ already exists.
$(VENV): | tools
	@echo "→ Creating virtual environment in $(VENV)/..."
	@$(VENV_PYTHON) -m venv $(VENV)
	@$(VENV)/bin/python -m pip install --upgrade pip --quiet
	@echo "✓ Virtual environment ready — activate with: source $(VENV)/bin/activate"

venv: $(VENV)

# Re-runs `poetry install` only when pyproject.toml or poetry.lock changed.
$(DEPS_STAMP): pyproject.toml poetry.lock | $(VENV)
	@echo "→ Syncing $(VENV)/ with poetry.lock (dev + ci groups)..."
	poetry config virtualenvs.in-project true
	poetry install --with dev,ci
	@touch $@
	@echo "✓ Dependencies in sync"

# The single prerequisite every other target uses: toolchain present, lock in
# sync with the manifest, .venv built and populated.
env: tools deps-check $(DEPS_STAMP)

# dev group only (tooling: pytest, ruff, mypy). NOT enough to run the tests —
# they import torch/sklearn through the package — so this drops the stamp and
# the next test target re-installs the full dev+ci set.
install: | tools $(VENV)
	poetry config virtualenvs.in-project true
	poetry install --with dev
	@rm -f $(DEPS_STAMP)

# dev + ci groups (the full ML stack). This is what the tests need.
install-ci: env
	@echo "✓ dev + ci dependencies installed"

# -----------------------------------------------------------------------------
# Dependencies — keep poetry.lock in sync with pyproject.toml
#
# GitLab CI runs `poetry install`, which under Poetry 2.x ABORTS if the lock is
# stale ("pyproject.toml changed significantly since poetry.lock was last
# generated"). That fails EVERY test job before a single test runs. deps-check
# catches that here in ~1 s so you never push a stale lock. If it fails, run
# `make lock` and commit both pyproject.toml and poetry.lock together.
# -----------------------------------------------------------------------------
deps-check: | tools
	@echo "→ Verifying poetry.lock is in sync with pyproject.toml..."
	@poetry check --lock || { \
	    echo ""; \
	    echo "✗ poetry.lock is OUT OF SYNC with pyproject.toml."; \
	    echo "  Run 'make lock' and commit both files, then retry."; \
	    exit 1; \
	}
	@echo "✓ poetry.lock in sync with pyproject.toml"

lock: | tools
	@echo "→ Regenerating poetry.lock from pyproject.toml..."
	poetry lock
	@echo "✓ poetry.lock regenerated — commit it ALONGSIDE pyproject.toml"

# -----------------------------------------------------------------------------
# Sample data (needed for integration and dataset load tests)
# -----------------------------------------------------------------------------
# Download any fixture that is missing. A fixture already on disk is kept and
# never re-fetched — pass --force (or use `make sample-data-force`) to refresh.
sample-data: env
	@echo "→ Ensuring sample fixtures (only missing ones are downloaded)..."
	$(PYTHON) scripts/create_sample_data.py $(ARGS)

# Explicit refresh: re-download every fixture even if it is already present.
# This is the "flag" escape hatch — nothing else ever overwrites a fixture.
sample-data-force: env
	@echo "→ Re-downloading ALL sample fixtures from Zenodo (--force)..."
	$(PYTHON) scripts/create_sample_data.py --force $(ARGS)

# Automatic fixture guard used by every data-dependent test target.
#
# This is the make-side half of fixture provisioning. The other half lives in
# tests/fixture_provisioning.py, called from pytest_sessionstart, so the same
# check runs for anyone who invokes pytest directly (an IDE, a bare
# `.venv/bin/pytest`, the GitLab jobs). Both ask create_sample_data.py the same
# question — tests/unit/generic/test_fixture_provisioning.py fails if they drift.
#
# Checks fixtures PER DATASET, not "is the directory empty" — a half-populated
# directory (one dataset added, its fixture never generated) used to satisfy the
# old check and the tests then failed with a confusing error much later.
#
# The presence check runs `--list-missing`, which discovers datasets only. It
# deliberately does not import models: a model whose constructor builds a dataset
# would reach the network just to answer "what am I missing?", i.e. trigger the
# very download this target exists to avoid.
#
# CI_SKIP_ZENODO=1 turns a missing fixture into a clear failure instead of a
# silent network call.
ensure-fixtures: env
	@mkdir -p $(FIXTURES_DIR); \
	missing="$$($(PYTHON) scripts/create_sample_data.py --list-missing 2>/dev/null)"; \
	if [ -z "$$missing" ]; then \
	    echo "✓ fixtures present in $(FIXTURES_DIR)/ — nothing to download"; \
	else \
	    if [ "$$CI_SKIP_ZENODO" = "1" ]; then \
	        echo "✗ Missing fixture(s): $$(echo $$missing | tr '\n' ' ')"; \
	        echo "  CI_SKIP_ZENODO=1 forbids downloading."; \
	        echo "  Run 'make sample-data' (needs network) or restore the committed fixtures."; \
	        exit 1; \
	    fi; \
	    echo "→ Missing fixture(s): $$(echo $$missing | tr '\n' ' ') — fetching from Zenodo..."; \
	    $(MAKE) --no-print-directory sample-data; \
	fi

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

# Fast: unit + smoke tests (no network, no data loading — run before every push)
test: test-unit test-smoke-models test-smoke-data
	@echo ""
	@echo "✓ All fast tests passed"

test-unit: env
	@echo "→ Running unit tests..."
	mkdir -p coverage
	$(PYTEST) tests/unit/ -v \
	    --cov=seanergys_modelzoo \
	    --cov-report=xml:coverage/cobertura-coverage.xml \
	    --cov-report=term-missing

test-smoke-models: env
	@echo "→ Running model smoke tests (discovery, instantiation, no data loading)..."
	$(PYTEST) tests/smoke/test_smoke_models.py -v

test-smoke-data: env
	@echo "→ Running data smoke tests (fixtures, no Zenodo download)..."
	@echo "  (Run 'make sample-data' first for local fixtures)"
	CI_SKIP_ZENODO=1 $(PYTEST) tests/smoke/test_smoke_data.py -v -m "not zenodo"

# Full pipeline: train → predict → save → load for each discovered model
test-pipeline: env ensure-fixtures
	@echo "→ Running model pipeline integration tests..."
	@echo "  (Run 'make sample-data' first)"
	$(PYTEST) tests/integration/test_model_pipeline.py -v -s

# All integration tests
test-integration: env ensure-fixtures
	@echo "→ Running all integration tests..."
	@echo "  (Run 'make sample-data' first)"
	$(PYTEST) tests/integration/ -v -s

# Everything
test-all: ensure-fixtures test-unit test-smoke-models test-smoke-data test-integration
	@echo ""
	@echo "✓ All tests passed"

# Build the wheel and verify it installs + imports with the source tree off the
# path — mirrors GitLab's build:wheel + test:wheel stages. Uses --no-deps so it
# reuses the already-installed ML stack instead of re-downloading torch, then
# uninstalls the wheel to leave .venv clean.
test-wheel: env
	@echo "→ Building wheel (mirrors build:wheel)..."
	poetry build --format wheel
	@ls -lh dist/*.whl
	@echo "→ Verifying the built wheel installs and imports (mirrors test:wheel)..."
	$(PYTHON) -m pip install -q --force-reinstall --no-deps dist/*.whl
	PYTHONPATH= $(PYTHON) ci/verify_wheel.py
	$(PYTHON) -m pip uninstall -y -q seanergys-modelzoo
	@echo "✓ Wheel builds, installs, and imports cleanly"

# -----------------------------------------------------------------------------
# Code quality
# -----------------------------------------------------------------------------
lint: env
	@echo "→ Running ruff lint..."
	$(PYTHON) -m ruff check seanergys_modelzoo ci tests

format: env
	@echo "→ Running ruff format..."
	$(PYTHON) -m ruff format seanergys_modelzoo ci tests

format-check: env
	@echo "→ Checking code formatting (no changes made)..."
	$(PYTHON) -m ruff format --check seanergys_modelzoo ci tests

typecheck: env
	@echo "→ Running mypy..."
	$(PYTHON) -m pip install -q mypy types-PyYAML 2>/dev/null || true
	-$(PYTHON) -m mypy seanergys_modelzoo ci tests --ignore-missing-imports --explicit-package-bases

coverage: env
	@echo "→ Running unit tests with coverage report..."
	mkdir -p coverage
	$(PYTEST) tests/unit/ -v \
	    --cov=seanergys_modelzoo \
	    --cov-report=xml:coverage/cobertura-coverage.xml \
	    --cov-report=html:coverage/html \
	    --cov-report=term-missing
	@echo "✓ HTML report: coverage/html/index.html"

audit: env
	@echo "→ Running dependency vulnerability scan..."
	$(PYTHON) -m pip install -q pip-audit poetry-plugin-export 2>/dev/null || true
	poetry export -f requirements.txt --without-hashes --with dev,ci -o /tmp/poetry-audit-reqs.txt
	$(PYTHON) -m pip_audit -r /tmp/poetry-audit-reqs.txt

check-secrets:
	@chmod +x ci/check_secrets.sh 2>/dev/null || true
	@./ci/check_secrets.sh .

# -----------------------------------------------------------------------------
# Docs — auto-generated CI capability & rules tables (docs/ci-cd.md)
# The tables are regenerated from .gitlab-ci.yml + the code + ci/policy.yml, so
# they never drift. On the default branch the docs:autogen CI job commits any
# regenerated changes automatically.
# -----------------------------------------------------------------------------
docs-ci: env
	@echo "→ Regenerating docs/ci-cd.md from pipeline + code + policy..."
	$(PYTHON) ci/gen_capabilities_doc.py

docs-ci-check: env
	@echo "→ Checking docs/ci-cd.md is up to date..."
	$(PYTHON) ci/gen_capabilities_doc.py --check

# -----------------------------------------------------------------------------
# Pre-push gate — fast checks that must pass before git push
# -----------------------------------------------------------------------------

# Installs a git hook so 'git push' runs pre-push automatically
install-hooks:
	@echo "→ Installing git pre-push hook..."
	@printf '#!/bin/sh\nexec make pre-push\n' > .git/hooks/pre-push
	@chmod +x .git/hooks/pre-push
	@echo "✓ Hook installed — 'make pre-push' will run automatically on every git push"

pre-push: format-check check-secrets lint test
	@echo ""
	@echo "→ Checking CI docs are current (non-blocking)..."
	-$(PYTHON) ci/gen_capabilities_doc.py --check
	@echo ""
	@echo "✓ Pre-push checks passed — safe to push"

# -----------------------------------------------------------------------------
# CI simulation — mirrors GitLab pipeline exactly
# -----------------------------------------------------------------------------
# Full local mirror of the GitLab pipeline:
#   deps → sanity → validate → lint → test → audit → build → test-wheel
# Hard-fails on anything GitLab marks allow_failure:false (deps, sanity,
# validate, tests, wheel). Steps GitLab marks allow_failure:true (lint, audit,
# secrets) are prefixed with '-' so they report but don't abort — exactly like
# the pipeline. If this is green, the GitLab run will be green too.
ci: env
	@echo ""
	@echo "════════════════════════════════════════"
	@echo " Simulating GitLab CI pipeline locally"
	@echo "════════════════════════════════════════"
	@echo ""
	@echo "── Stage 0: deps (sync .venv to poetry.lock) ──"
	poetry config virtualenvs.in-project true
	poetry install --with dev,ci
	@echo ""
	@echo "── Stage 1: sanity ──"
	find seanergys_modelzoo ci tests -name "*.py" -print0 | xargs -0 $(PYTHON) -m py_compile && echo "✓ syntax OK"
	test -f pyproject.toml && test -d seanergys_modelzoo && test -d ci && test -d tests && echo "✓ structure OK"
	-@chmod +x ci/check_secrets.sh 2>/dev/null || true
	-./ci/check_secrets.sh .
	@echo ""
	@echo "── Stage 2: validate ──"
	$(PYTHON) ci/validate_yaml.py && echo "✓ YAML OK"
	@echo ""
	@echo "── Stage 3: lint (allow_failure) ──"
	-$(PYTHON) -m ruff check seanergys_modelzoo ci tests
	-$(PYTHON) -m ruff format --check seanergys_modelzoo ci tests
	-$(PYTHON) -m mypy seanergys_modelzoo ci tests --ignore-missing-imports --explicit-package-bases
	@echo ""
	@echo "── Stage 4: test (each layer once; combined coverage) ──"
	mkdir -p coverage
	rm -f coverage/.coverage
	COVERAGE_FILE=coverage/.coverage $(PYTEST) tests/unit/ -v \
	    --cov=seanergys_modelzoo --cov-append --cov-report=
	COVERAGE_FILE=coverage/.coverage $(PYTEST) tests/smoke/test_smoke_models.py -v \
	    --cov=seanergys_modelzoo --cov-append --cov-report=
	CI_SKIP_ZENODO=1 COVERAGE_FILE=coverage/.coverage $(PYTEST) tests/smoke/test_smoke_data.py -v -m "not zenodo" \
	    --cov=seanergys_modelzoo --cov-append --cov-report=
	@echo "  (integration needs fixtures — run 'make sample-data' if this fails)"
	CI_SKIP_ZENODO=1 COVERAGE_FILE=coverage/.coverage $(PYTEST) tests/integration/ -v -m "not zenodo" \
	    --cov=seanergys_modelzoo --cov-append --cov-report=
	@echo ""
	@echo "── Stage 5: coverage (combined report) ──"
	COVERAGE_FILE=coverage/.coverage $(PYTHON) -m coverage xml -o coverage/cobertura-coverage.xml
	COVERAGE_FILE=coverage/.coverage $(PYTHON) -m coverage report
	@echo ""
	@echo "── Stage 6: docs (capability tables current?) ──"
	$(PYTHON) ci/gen_capabilities_doc.py --check
	@echo ""
	@echo "── Stage 4b: audit (allow_failure) ──"
	-$(MAKE) audit
	@echo ""
	@echo "── Stage 7 + 8: build wheel + test wheel ──"
	$(MAKE) test-wheel
	@echo ""
	@echo "✓ CI simulation complete — safe to push"

# -----------------------------------------------------------------------------
# Clean
# -----------------------------------------------------------------------------
clean:
	rm -rf .pytest_cache .ruff_cache build/ dist/ *.egg-info coverage/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Cleaned caches and build artifacts"

# Forget the cached interpreter choice; the next target re-detects it.
clean-tools:
	rm -rf .make
	@echo "✓ Cleared .make/ (toolchain detection cache)"

clean-venv:
	rm -rf $(VENV)
	@echo "✓ Removed $(VENV)/ — run 'make venv' or 'make install' to recreate"

clean-fixtures:
	rm -rf tests/fixtures/sample_data/
	@echo "✓ Cleared fixture files (run 'make sample-data' to recreate)"

clean-all: clean clean-tools clean-venv clean-fixtures
	@echo "✓ Cleaned everything"
