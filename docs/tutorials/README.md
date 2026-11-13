# CI/CD tutorials: start here

These tutorials explain this repository's CI/CD system in basic language. They are for
colleagues who want to understand what happens after a change is pushed, which tests run,
and how the CI reference document stays up to date.

## Suggested reading order

1. [CI/CD and testing tutorial](01-ci-cd-and-testing.md) — start here. It explains the
   pipeline, gates, stages, tests, supported models and datasets, and the normal developer
   workflow.
2. [Automatic documentation tutorial](02-automatic-documentation.md) — explains where the
   generated information comes from, the main generator functions, and how a change to a
   CI gate automatically appears in the documentation.

## Tutorial versus reference

These files are learning material. They explain the ideas step by step.

The live reference is [`../ci-cd.md`](../ci-cd.md). Its tables list the current jobs,
triggers, gate status, supported model–dataset pairs, and tests. Those tables are generated
from the repository and should not be edited by hand.

The detailed maintainer manual is
[`../how-ci-cd-docs-are-generated.md`](../how-ci-cd-docs-are-generated.md). Read it when you
need to change the generator itself or investigate a documentation problem.

## The most important rule

Change the source, then regenerate the documentation:

```bash
make docs-ci
make docs-ci-check
```

Do not manually change a table between `BEGIN:GENERATED` and `END:GENERATED` markers in
`docs/ci-cd.md`.
