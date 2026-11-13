# Making the test suite model- and dataset-agnostic — work report

**Date:** 2026-09-08  ·  **Author:** Mohsen Seyedkazemi Ardebili (CI & test maintainer)
**Branch:** `main` @ `12a4d61`  ·  **Epic:** #2  ·  **Scope:** CI, tests, scripts, Makefile only

> **Ground rule for all of this work:** `seanergys_modelzoo/` is owned by the model and
> dataset authors. Nothing in it was modified.
> `git diff --name-only 59fb712..HEAD -- seanergys_modelzoo/` returns **0 files**.
> Every change core needs is recorded in `ci/policy.yml → core_change_requests`, published
> in `docs/ci-cd.md`, and handed to its owner.

---

## 1. The problem in one paragraph

Auto-discovery already found every model and dataset — `tests/conftest.py` builds
`MODEL_CLASSES`, `DATASET_CLASSES` and `MODEL_DATASET_PAIRS`, and the smoke and integration
layers were driven entirely from them. `tests/unit/` did not follow: it named `MACK`,
`MCBound`, `JPCP`, `FDataDataset`, `PM100Dataset`, `M100Dataset` and `SCRIPTAIDataset`
**123 times across six unit modules** (126 counting the two stray references in smoke and integration). The consequence was not cosmetic. A new model entered
smoke and integration as a parameter and received **zero unit coverage**, and the two
datasets nobody had typed by hand — `M100Dataset` and `SCRIPTAIDataset` — were invisible to
every discovery-driven test. The datasets that *were* covered were exactly the two whose
names someone had written into a test.

Auto-discovery without agnostic assertions gives a green pipeline that is silently narrower
than the codebase.

---

## 2. What was built

### 2.1 The `generic` / `specific` split (#2)

```
tests/unit/generic/    discovery-driven only. Zero concrete names. CI-enforced.
tests/unit/specific/   per-component tests, owned by the component author.
```

Six existing modules moved to `specific/` unchanged — they legitimately exercise one
component's own parsing and storage behaviour, and naming a component there is expected.

### 2.2 New generic suites

| Module | Issue | What it guarantees |
|---|---|---|
| `generic/test_model_lifecycle.py` | #3 | `task_type`, `supported_datasets`, save/load round-trip, load-missing-path, save-on-error, `is_trained` — written **once**, parametrized over `MODEL_CLASSES` |
| `generic/test_parquet_dataset_contract.py` | #4 | the shared parquet contract per discovered `(model, dataset)` config, **including the inherited base helpers through every subclass** |
| `generic/test_discovery_contract.py` | #5 (partial) | a discovered dataset that no model consumes **fails the build** unless `ci/policy.yml` records why |
| `generic/test_no_hardcoded_names.py` | #7 | scans `generic/` and fails on any concrete name, with file, line and the offending name |
| `generic/test_archive_extraction_safety.py` | #6 | tar path-traversal rejection for **every** dataset exposing `_safe_extract` |
| `generic/test_core_change_requests.py` | — | keeps the core-request register honest in both directions |

Three design choices did the heavy lifting:

- **Capability, not identity.** A check that needs an optional method skips when the model
  does not implement it, instead of listing the models that do.
- **The model's own contract is the source of truth.** Fixtures and parameters are derived
  from `supported_datasets` + `get_train_config()`, which authors already implement. The
  test layer never asks core to grow new members.
- **`sys.modules[model_cls.__module__]`** reaches a model's own module for monkeypatching,
  rather than importing a hand-written list of module paths.

### 2.3 The guard that makes it stick (#7)

`test_no_hardcoded_names.py` builds its forbidden list **from discovery itself**, so it needs
no maintenance. Verified red-green: dropping a `test_mack_defaults` into `generic/` produces

```
FAIL [test_tmp_offender.py] hardcodes 2 concrete name(s):
    test_tmp_offender.py:3  'MACK'  ->  from seanergys_modelzoo.models.tasks...
  → A test that genuinely targets one component belongs in tests/unit/specific/
```

Removing the file goes green again. The suite drifted back to hardcoding once already; a
convention that is not enforced does not hold.

### 2.4 Literals derived from class constants (#6)

Zenodo record ids (`7541722`, `7589942`, `7588815`, `7589131`) and monthly file ids
(`21_03`, `22-02`, …) were typed out **28 times** while the classes already held them. They
are now read from `AVAILABLE_FILES` and `AVAILABLE_FILES_PER_RECORD`, and assertions express
the *relationship* rather than the value — a range built from `FDATA_FILES[0]:FDATA_FILES[2]`
must equal `FDATA_FILES[:3]`; total nodes is `len(AGG_FILES) * NODES_PER_RACK`, not `1000`.

### 2.5 Fixtures made automatic (#9, and the fixture half of #8)

`ensure-fixtures` used to check *"is the directory empty"*. A half-populated directory — one
dataset added, its fixture never generated — satisfied that check, so the download never
fired and tests failed much later with a confusing error. It now checks **per dataset** and
downloads only what is missing; `make test-all` depends on it.

An existing fixture is **never** silently overwritten. Refreshing is explicit:
`make sample-data-force` or `FORCE_SAMPLE_DATA=1`.

---

## 3. What we found

These were measured, not inferred. Each is now a numbered entry in
`ci/policy.yml → core_change_requests`, rendered as a table in `docs/ci-cd.md`.

### 3.1 Discovery downloaded 273 MB and hung the whole suite

`RUAD.build_model()` constructs an `M100Dataset` inside the model validator, so **merely
creating the model loads data**. Every discovery-driven test constructs each model, so
`pytest` reached Zenodo before a single test ran. A clean run pulled **273 MB** and had not
finished after two minutes — with `CI_SKIP_ZENODO=1` set, which cannot prevent it.

Fixed from the test side by a per-model time budget (`MODELZOO_COLLECT_TIMEOUT`, default
20s) around instantiation and config assembly. Measured: **>2 min / 273 MB → 1.4 s / 12 KB**.
The real fix is CR-4, and it belongs to the RUAD author.

### 3.2 A documented CI rationale was false

`ci/policy.yml → skip_notes` claimed three tests parametrized to zero cases because
`get_train_config()` instantiates datasets that connect to Zenodo at collection time. Tested
with outbound sockets blocked: `get_train_config()` returns in **~0 s** for JPCP, MACK and
MCBound and never opens a socket. The empty parameter sets came from a defensive early
return in the test helper itself. Entry deleted, helper fixed.

*(Caveat, stated plainly: the claim later became true for a different reason — RUAD, added
after that note was written, does exactly what it described. See 3.1.)*

### 3.3 Test fixtures used column names that exist in neither dataset

The synthetic frame in the old unit tests used `n_nodes`, `duration`, `power_consumption`,
`partition`. Against the committed fixtures, real F-DATA publishes `nnumr` / `nnuma` /
`avgpcon` / `duration`, and real PM100 publishes `num_nodes_req` / `node_power_consumption` /
`partition`. Only `duration` and `partition` are real, each in only one of the two.

The generic contract now builds its frame from the columns the model's config declares, so
the test cannot invent a schema.

### 3.4 Three model configs name columns that do not exist

| Pair | Names | Reality | Request |
|---|---|---|---|
| MACK-f-data, MCBound-f-data | `mem_compute_bound` | no F-DATA file publishes it; the real column is `pclass` | CR-5 |
| JPCP-pm100 | `user_id_cat`, `num_nodes_req_cat` | created by the model's **own** preprocessing, but columns are projected at read time, so the read fails before preprocessing runs | CR-6 |

### 3.5 A public dataset method fails on real data

`FDataDataset.get_statistics_by_partition()` reads `partition`, `n_nodes` and
`power_consumption`. Against the committed fixture it raises
`ValueError: 'partition' column not available in dataset.` — raw F-DATA has no partition
column at all. Dead code that looks like a working feature. CR-3.

### 3.6 Two datasets were invisible to discovery

`M100Dataset` and `SCRIPTAIDataset` inherit the empty `DATASET_ID` default, so they enter no
`(model, dataset)` pair, no fixture can be built for them, and they contribute zero executed
cases. CR-1 and CR-2. Both are now recorded in `dataset_scope`, so the gap is **published in
`docs/ci-cd.md`** instead of hiding as a permanent skip.

---

## 4. Core change requests raised

The CI/test layer never edits core. Each request carries what must change, why, why CI
cannot do it, the owner, and the evidence — plus a machine-checkable condition, so
`generic/test_core_change_requests.py` **fails when the fix lands** and forces the entry out.
A request is therefore either open and accurate, or deleted.

| ID | Symbol | Owner | Ask |
|---|---|---|---|
| CR-1 | `M100Dataset` | M100 author | set `DATASET_ID`; have a model list it |
| CR-2 | `SCRIPTAIDataset` | SCRIPT-AI author | set `DATASET_ID`; have a model list it |
| CR-3 | `FDataDataset.get_statistics_by_partition` | F-DATA author | rewrite against real columns, or remove |
| CR-4 | `RUAD.build_model` | RUAD author | stop constructing a dataset at model construction |
| CR-5 | `MACK` / `MCBound` `get_train_config` | model authors | target `pclass`, not `mem_compute_bound` |
| CR-6 | `JPCP.get_train_config` (pm100) | JPCP author | declare raw source columns, not preprocessing outputs |

---

## 5. Verification

All run on `main` @ `12a4d61`:

| Gate | Result |
|---|---|
| `CI_SKIP_ZENODO=1 pytest tests/unit tests/smoke tests/integration -m "not zenodo"` | **269 passed, 4 skipped, 11 xfailed, 0 failed** |
| `tests/unit/generic` alone | 84 passed, 4 skipped, 6 xfailed |
| `make lint` | All checks passed |
| `ruff format --check tests ci` | 36 files already formatted |
| `make docs-ci-check` | `docs/ci-cd.md` up to date |
| concrete names in `tests/unit/generic/` | **0** |
| core files changed | **0** |

Behavioural proof, not just counts: dropping a `BrandNewDataset` into
`seanergys_modelzoo/datasets/` and editing **no test** fails `test:unit` with
*"is discovered but never exercised"* and *"does not set DATASET_ID"*; removing it goes green.

---

## 6. Issue status

| Issue | State | Note |
|---|---|---|
| #2 tracking | closed | |
| #3 model lifecycle | closed | |
| #4 parquet datasets | closed | |
| #6 literals | closed | |
| #7 CI guard | closed | |
| #9 Makefile | closed | |
| **#5 datasets self-describe** | **open** | see §7 |
| **#8 CI fixture generation** | **open** | see §7 |

---

## 7. Why #5 and #8 are still open, and the options

Both remain open for the **same structural reason**: their remaining requirements can only be
satisfied by changing `seanergys_modelzoo/`, which the CI/test maintainer does not own.
Closing either today would record something untrue.

### 7.1 Issue #5 — "Datasets must self-describe"

**What it asks.** Add an introspectable contract to `SeanergysDataset` and every concrete
dataset: `DATASET_NAME`, `DOWNLOAD_FILENAME`, `SAMPLE_SCHEMA`, `get_dummy_params()`.

**Done (from the test side):**
- no network during pytest collection — enforced by the per-model time budget (§3.1)
- the false `skip_notes` entry deleted, and the empty-parameter-set skips are gone
- `M100Dataset` / `SCRIPTAIDataset` documented in `dataset_scope` with the owner's fix

**Not done, and not doable here:** every dataset declaring those members, and generic tests
building fixtures from `SAMPLE_SCHEMA`. Those are **CR-1** and **CR-2**.

**Worth recording:** this was implemented once, exactly as the issue specifies, and then
deliberately discarded. Adding those members to core inverts ownership — it makes CI depend
on every model author adopting new core members, so the guarantee only holds once everyone
complies. The replacement derives what it needs from `supported_datasets` +
`get_train_config()`, which authors already implement, and **publishes** any gap rather than
patching around it. That is why the issue's acceptance criteria are partly unmet by design.

**Options:**

| Option | What happens | Cost |
|---|---|---|
| **A. Re-scope #5 to CR-1/CR-2 and close it (recommended)** | The self-description goal is met by the discovery-contract test; the two concrete gaps are already tracked as core requests with named owners | Needs Francesco to agree the members are no longer required |
| B. Keep #5 open until the dataset authors land CR-1/CR-2 | Issue closes when the code genuinely satisfies it | Stays open on someone else's schedule |
| C. Implement the members in core | Satisfies the issue literally | Breaks the ownership rule; rejected once already |

### 7.2 Issue #8 — "CI Fixture Generation"

Three requirements; two are met:

| Requirement | Status |
|---|---|
| Generic for any `SeanergysDataset`, no hard-coded names | **met** — discovery-driven; `_is_samplable()` decides by declared fields |
| Not re-downloaded if present, unless a flag requests it | **met** (`d176ac5`) — `--force` / `FORCE_SAMPLE_DATA=1` / `make sample-data-force` |
| **"The fixture should always be the dummy version of the SeanergysDataset classes"** | **not met** |

**Why the third is not met.** `scripts/create_sample_data.py` deliberately bypasses the
dataset class, and documents why: `is_dummy=True` still hits Zenodo, a model's required
columns may be absent from the newest monthly file, and transforms need a **fitted** model
that does not exist at fixture-creation time. So the script downloads raw parquet directly
and slices `head(max_rows)`. The fixture is a small subset — but it is not produced *via*
`is_dummy`, which is what the requirement asks for.

**Options:**

| Option | What happens | Cost |
|---|---|---|
| **A. Re-scope the third requirement and close (recommended)** | Amend to "the fixture must be a bounded subset" — which is met — and record the three technical reasons in the issue | Needs Francesco's agreement; the wording changes, the behaviour does not |
| B. Route fixture creation through `is_dummy` | Satisfies the wording literally | Needs core changes so `is_dummy` yields a fixture offline and without a fitted model — i.e. CR-1/CR-2-class work, plus a way to defer transforms |
| C. Leave open as a known limitation | Honest, no work | An open issue nobody can act on |

**Both are decisions for @francesco.antici**, since he wrote the acceptance criteria.

---

## 8. Open decision: xfail vs. a red pipeline

**This needs a call, and it is a conflict I introduced.**

Issue **#10** — *"Integration tests must not hide failures behind xfail"* — is **closed**, and
argues that `xfail(strict=True)` keeps the job exit code at `0` while pairs are never
exercised end to end. Commit `8b960ea` acted on that: it restored core to the authors'
version and left `integration_xfail` **empty**, with the message *"keep the failure visible"*.

In this work I **re-populated** `integration_xfail` with three entries (MACK-f-data,
MCBound-f-data, JPCP-pm100) and added `model_xfail` (RUAD) and `config_load_xfail`
(JPCP-pm100). That makes the suite report **0 failed** — and it contradicts the decision
recorded in #10 and in `8b960ea`.

The argument for the current state: the breakages are real, they are core-owned (CR-4/5/6),
every entry is `strict=True` so a fix forces it out, and each is published in
`docs/ci-cd.md`. The argument against: a green pipeline over known-broken pairs is exactly
what #10 rejected.

| Option | Result |
|---|---|
| **A. Revert to visible failures** — drop the entries I added | `main` goes red: 3 integration + 2 smoke failures, all genuine and core-owned. Honest signal, matches #10 |
| B. Keep the strict-xfail entries | Pipeline green; breakage documented but not in the exit code |
| C. Split the difference | Keep `model_xfail`/`config_load_xfail` (unit-layer, component cannot even be constructed), drop `integration_xfail` so the end-to-end gate stays honest |

**Recommendation: C**, then A once CR-4/5/6 land. The unit layer should not be red because a
model cannot be constructed — that is already reported by its own contract test — but the
integration gate is the one #10 was about, and it should tell the truth.

---

## 9. Follow-ups

1. **Decide §8** — this is the only item that changes `main`'s pipeline colour.
2. **Route CR-1 … CR-6 to their owners.** They are in `ci/policy.yml` and rendered in
   `docs/ci-cd.md`; they are not yet GitLab issues.
3. **Agree the re-scoping of #5 and #8** with @francesco.antici, or leave both open.
4. **GitLab API token is read-only** — every write returns
   `{"error":"insufficient_scope"} HTTP 403`. Closing issues and posting comments needs a
   legacy PAT with `api` write scope.
