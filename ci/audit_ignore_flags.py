#!/usr/bin/env python3
"""Print ``--ignore-vuln <ID>`` flags for pip-audit, sourced from the
``audit_ignore`` section of ci/policy.yml.

Keeping the ignore list in policy.yml (with a rationale per entry) makes it the
single source of truth: the ``test:audit`` job consumes these flags and the
generated docs/ci-cd.md documents *why* each advisory is ignored, so the two can
never drift. An advisory is ignored ONLY because it has no fixed release yet;
remove its entry as soon as a patched version exists and the audit will flag the
upgrade again.

Fails open: if PyYAML is unavailable or the file is missing, it prints nothing,
so pip-audit reports every vulnerability (safe default).
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - fail open
    sys.exit(0)

_POLICY = Path(__file__).resolve().parent / "policy.yml"


def main() -> int:
    if not _POLICY.exists():
        return 0
    data = yaml.safe_load(_POLICY.read_text(encoding="utf-8")) or {}
    ignore = data.get("audit_ignore") or {}
    flags = " ".join(f"--ignore-vuln {vid}" for vid in ignore)
    if flags:
        print(flags)
    return 0


if __name__ == "__main__":
    sys.exit(main())
