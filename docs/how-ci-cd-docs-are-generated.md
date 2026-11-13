# How the CI/CD docs are auto-generated — the human manual

**This file is the manual for the CI/CD documentation system.** It is written and
maintained *entirely by hand*. It tells you, in plain terms:

- **which document is written by a human and which is generated automatically** — so you
  never waste time editing the wrong one ([Who owns what](#which-files-are-human-written-and-which-are-auto-generated));
- **your role as a human** — what you are responsible for keeping correct
  ([Your role](#your-role-as-a-human--what-you-must-do));
- **when to update `ci/policy.yml`** and the other sources
  ([When to update policy.yml](#when-should-i-update-cipolicyyml));
- **how the auto-generation actually works** under the hood (the rest of this file).

Its companion, [`ci-cd.md`](ci-cd.md), is the **reference**: its tables are *auto-generated*
from the pipeline, and it also carries a hand-written *Developer guide* (how to run tests,
add a model/dataset). Rule of thumb: **`ci-cd.md` = what the CI does; this file = how the
system works and what the human must do.**

> The sections below (Who owns what / Your role / When to update `policy.yml` / Will my
> change show up?) are written as direct answers to real questions asked while building this
> system — read them first if you just want to *use* the system rather than understand its
> internals.

## Which files are human-written and which are auto-generated?

There is exactly **one** auto-generated artifact — the *tables* inside `docs/ci-cd.md`.
Everything else is written by a human. Nothing is fully auto-generated end to end; a human
always owns the inputs.

| File / part | Written by | Do you edit it? |
|---|---|---|
| `docs/ci-cd.md` — the **tables** (between `<!-- BEGIN/END:GENERATED -->` markers) | **Automatic** (the generator) | **No** — never edit by hand; run `make docs-ci` |
| `docs/ci-cd.md` — the **prose** (Purpose, Developer guide) | Human | Yes |
| `docs/how-ci-cd-docs-are-generated.md` (this file) | Human | Yes |
| `docs/future_pydantic_tests.md`, `docs/MODELZOO_ARCHITECTURE_NOTES.md` | Human | Yes |
| `.gitlab-ci.yml` | Human | Yes — it is the pipeline; the tables read *from* it |
| `ci/policy.yml` | Human | Yes — the "why" + policy lists (see below) |
| `ci/gen_capabilities_doc.py` | Human | Only when adding/altering a table |
| The model/dataset/test **code** | Human | Yes — discovery reads it to build the capability tables |

So a human never edits the generated tables directly. Instead a human edits a **source**
(the YAML, `policy.yml`, or the code) and the generator re-derives the tables from it.

## Your role as a human — what you must do

The generator handles the *what* (jobs, rules, capability matrix, skips). **You** are
responsible for four things it cannot know on its own:

1. **Edit the source, not the output.** Change `.gitlab-ci.yml` (the pipeline), the code
   (models/datasets/tests), or `ci/policy.yml` (the rationale) — never the generated tables
   in `ci-cd.md`.
2. **Keep the "why" in `ci/policy.yml` truthful.** When you add/change a job or a known
   limitation, write or update its rationale there (see [when](#when-should-i-update-cipolicyyml)).
3. **Regenerate and commit** the doc after such a change: `make docs-ci`, then include the
   updated `ci-cd.md` in the same commit. (If you forget, `docs:check` warns you on the MR,
   and `docs:autogen` fixes it on `main` — but do it yourself so reviewers see the change.)
4. **Keep the hand-written prose current** — the Developer guide in `ci-cd.md` and this
   manual are yours to maintain; the generator will not touch them.

**Pre-merge checklist** (also covered by `make pre-push` / `make docs-ci-check`):

- [ ] Changed `.gitlab-ci.yml` or added/removed a model/dataset/test? → ran `make docs-ci`.
- [ ] Changed *why* a job blocks/warns, a core change request, an ignored CVE, or a skip? → edited
      the matching `ci/policy.yml` section.
- [ ] `make docs-ci-check` passes (the doc is not stale).
- [ ] The regenerated `ci-cd.md` is staged in the same commit.

## The idea in one line

A generator script reads three sources of truth and rewrites **only** the table regions of
`ci-cd.md`, leaving the hand-written prose untouched — so the "what the CI does" tables can
never drift from the actual pipeline, while the "how to" guide stays human-authored.

## The three sources of truth

`ci/gen_capabilities_doc.py` builds every table from these — nothing is typed into the doc
by hand:

| Source | Supplies | The "what" vs "why" |
|---|---|---|
| `.gitlab-ci.yml` | Jobs, stages, trigger rules, `allow_failure` | the **what** — parsed with PyYAML, anchors resolved |
| The code (via `tests/conftest.py`) | Model×dataset capability matrix, dataset list, test inventory | the **what** — reuses the exact auto-discovery the tests use |
| `ci/policy.yml` | Human rationale that cannot be introspected | the **why** — edited by hand |

Because the *what* comes from the YAML and the code, changing a job or adding a model
updates the tables on the next regeneration with zero manual edits. The *why* is the one
place a human writes prose, and it lives in `ci/policy.yml`.

## The role of `ci/policy.yml`

`ci/policy.yml` is the **human-written "why"** of the system — the one file you edit to
change anything that cannot be read automatically from the pipeline or the code. The
generator merges it into the tables, and the CI reads parts of it directly. It has five
sections:

| Section | What you put there | Where it shows up |
|---|---|---|
| `rules:` | One-line description of each trigger anchor (`rules_unit`, `rules_zenodo`, …) | Trigger legend table |
| `jobs:` | Per-job reason it **blocks vs warns** | Pass/fail table's "Why" column |
| `core_change_requests:` | Changes owed by the model/dataset authors + owner, evidence, expiry check | "Core change requests" section; a test fails when a request's fix lands so the entry must go |
| `audit_ignore:` | CVEs with no upstream fix yet + reason | `test:audit` (`--ignore-vuln`, via `ci/audit_ignore_flags.py`) |
| `skip_notes:` | Implicit skips the AST scan can't see (e.g. empty-parametrize) | Test-skip inventory |

The split is deliberate: the **"what"** (jobs, rules, `allow_failure`) is read from
`.gitlab-ci.yml` and the code so it can't drift, while the **"why"** and these curated lists
live in `policy.yml` — the single place a human maintains. Several sections
(`core_change_requests`, `dataset_scope`, `audit_ignore`, `name_scan`) are consumed by the CI
**and** the docs, so editing one line changes both the behaviour and its documentation at
once. Note what is deliberately absent: there is no list that marks a failing test as
acceptable — `xfail` was removed on 2026-09-13.

## When should I update `ci/policy.yml`?

Update it whenever a change would otherwise leave the docs (or the CI behaviour) telling a
half-truth. Concretely:

| When this happens… | Edit this `policy.yml` section | Why |
|---|---|---|
| You **add a new job**, or change *why* a job blocks vs warns | `jobs:` | The Pass/fail "Why" column is blank / wrong until you do |
| You **add a new trigger rule anchor** (`.rules_*`) in `.gitlab-ci.yml` | `rules:` | The Trigger legend needs its one-line description |
| A **(model, dataset) pair is known-broken** because of core code | `core_change_requests:` | Publishes the defect, its owner and the evidence. The test stays RED — nothing marks it acceptable. Remove the entry when fixed |
| `pip-audit` flags a **CVE that has no upstream fix yet** | `audit_ignore:` | Keeps `test:audit` green for it; remove once a patched version ships |
| A test **skips implicitly** in a way the AST scan can't see (e.g. an empty parametrize) | `skip_notes:` | So it still appears in the Test-skip inventory |

You do **not** touch `policy.yml` for: renaming a job, changing its command, adding a model
or dataset, or adding/removing a `pytest.skip(...)` — those are read automatically from the
YAML or the code. After any `policy.yml` edit, run `make docs-ci` and commit the result.

## How the generator protects hand-written prose

Every generated table sits between a matched pair of HTML-comment markers:

```
<!-- BEGIN:GENERATED jobs -->
| Stage | Job | ... |         ← this region is overwritten on every run
<!-- END:GENERATED jobs -->
```

The generator replaces **only** the text between each `BEGIN`/`END` pair. Anything outside
the markers — the Purpose section, the whole **Developer guide**, this sentence — is never
touched. That is what lets one file hold both auto-generated tables and a human guide.

The generated block names currently are: `pipeline-stages`, `jobs`, `triggers`, `rationale`,
`capabilities`, `datasets`, `test-inventory`, `test-skips`.

## Running it

```bash
make docs-ci          # regenerate docs/ci-cd.md in place
make docs-ci-check    # exit non-zero if the doc is stale (used in CI)
```

Under the hood both call `python ci/gen_capabilities_doc.py` (the second with `--check`).
The generated regions contain **no timestamps**, so regenerating an unchanged repo is a
no-op — this is what makes the auto-commit job loop-safe (see below).

## Will my change show up in the doc?

Yes — if you change a CI YAML file or the **structural** parts of your tests, the tables
update to match. Two things to keep in mind:

1. **It shows the new *state*, not a highlighted diff.** The generator rewrites the tables to
   describe how things are *now*; it does not annotate "you changed X to Y". To *see* exactly
   what your edit did to the doc, regenerate and then `git diff docs/ci-cd.md` — that shows
   the precise lines that changed.
2. **It updates only when the generator runs** — `make docs-ci` locally, or the `docs:autogen`
   job automatically on `main` after merge. On a merge request, `docs:check` *warns* if you
   changed the YAML/tests but forgot to regenerate.

| If you change… | Reflected? | Which table |
|---|---|---|
| A job in `.gitlab-ci.yml` (rename, add/remove, its command, `allow_failure`, `changes:` paths) | **Yes** | Jobs / Triggers / Pass-fail |
| A model or dataset class (add/remove, or its `supported_datasets`) | **Yes** (auto-discovered) | Capability matrix, Datasets |
| Add/remove a **test file** | **Yes** | Test inventory (file + its first docstring line) |
| Add/remove/change a `pytest.skip(...)` or `@pytest.mark.zenodo` | **Yes** (AST-scanned) | Test-skip inventory |
| The *reason* a job blocks or warns | **Only if you edit `ci/policy.yml`** (not read from the YAML) | Pass/fail "Why" column |
| The **logic inside a test** (an assertion, a value, a comment) | **No** — test bodies aren't read (except skip calls) | — |

**Concrete example:** flip a job's `allow_failure: true` → `false` in `.gitlab-ci.yml`, run
`make docs-ci`, and that job's row in the Pass/fail table flips from **"Warns"** to
**"Blocks"** — and `git diff docs/ci-cd.md` shows just that one line changing.

**The workflow:**

```bash
# 1. edit .gitlab-ci.yml, or add/remove a test / model / skip
make docs-ci               # 2. regenerate the tables
git diff docs/ci-cd.md     # 3. see exactly what changed in the doc
```

## The two CI jobs that keep it fresh

| Job | Runs on | Does | Blocks? |
|---|---|---|---|
| `docs:check` | merge requests | `gen_capabilities_doc.py --check` — warns if the committed doc is stale | No (warn-only) |
| `docs:autogen` | the default branch | regenerates and **commits** the doc back if it changed | No (warn-only) |

So the workflow is: on an MR, `docs:check` reminds you to run `make docs-ci` if you forgot;
after merge, `docs:autogen` regenerates on `main` and commits any remaining difference, so
the shared doc is always current even if a contributor forgets.

**Loop-safety & auth for `docs:autogen`:** the bot commit uses `[skip ci]` (and
`git push -o ci.skip`) so it does not retrigger the pipeline, and because generated regions
have no timestamps, an unchanged repo produces no commit. The push needs a masked CI/CD
variable `DOCS_BOT_TOKEN` — a Project Access Token with `write_repository`, whose user may
push to the protected default branch. That token is the one manual setup step; everything
else is code.

## How to change what the docs say

Pick the source that owns the fact — never edit `ci-cd.md`'s tables directly:

- **A job, stage, trigger, or `allow_failure` changed** → edit `.gitlab-ci.yml`, then
  `make docs-ci`. The Jobs / Triggers / Pass-fail tables update themselves.
- **A model or dataset was added/removed** → nothing to edit; discovery picks it up. Just
  `make docs-ci` and the capability matrix, dataset list, and test inventory refresh.
- **The rationale (the "why"), a core change request, an ignored CVE, or a skip note changed** → edit
  `ci/policy.yml` (see [The role of `ci/policy.yml`](#the-role-of-cipolicyyml) for its five
  sections), then `make docs-ci`.
- **A `pytest.skip(...)` or `@pytest.mark.zenodo` was added/removed** → nothing to edit; the
  generator re-scans the test source (AST) and updates the Test-skip inventory.

## How to add a new generated table

1. Add a builder in `_build_blocks()` in `ci/gen_capabilities_doc.py` that puts a rendered
   table (use the `_table(headers, rows)` helper) under a new key, e.g.
   `blocks["my-block"] = _table([...], rows)`.
2. Add the marker pair where you want it to appear in `ci-cd.md`:
   `<!-- BEGIN:GENERATED my-block -->` / `<!-- END:GENERATED my-block -->` (the generator
   warns if a block has no matching marker).
3. Add the same `{my-block}` placeholder to `_DOC_TEMPLATE` so a from-scratch scaffold stays
   consistent.
4. `make docs-ci` to fill it, and `make docs-ci-check` to confirm it's stable.

## Requesting a change without touching code

Colleagues who want a CI rule/gate/capability changed open an issue with the **CI Change
Request** template (`.gitlab/issue_templates/CI_Change_Request.md`). The reviewer edits the
owning source (`.gitlab-ci.yml` or `ci/policy.yml`), regenerates, and the tables follow.

## Files involved

| File | Role |
|---|---|
| `ci/gen_capabilities_doc.py` | The generator (parses YAML, discovers code, scans skips, renders tables) |
| `ci/policy.yml` | Hand-written rationale + core change requests, ignore/skip lists, name-scan policy |
| `ci/audit_ignore_flags.py` | Emits `--ignore-vuln` flags for `test:audit` from `policy.yml` |
| `docs/ci-cd.md` | The output: auto-generated tables + hand-written Developer guide |
| `.gitlab-ci.yml` | The pipeline itself + the `docs:check` / `docs:autogen` jobs |
| `Makefile` | `docs-ci`, `docs-ci-check` targets |
