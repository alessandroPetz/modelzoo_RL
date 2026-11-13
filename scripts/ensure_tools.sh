#!/usr/bin/env bash
#
# ensure_tools.sh — make sure the machine has the toolchain this repo needs.
#
# Checked (and, where possible, installed automatically into ~/.local/bin, no sudo):
#
#   curl, git, make   system prerequisites — reported only, need a package manager
#   uv                fast installer + Python version manager  (curl | sh)
#   python 3.12–3.14  interpreter used to build .venv          (system, or `uv python`)
#   poetry            dependency resolver / lock / build        (`uv tool install poetry`)
#
# The Python range mirrors `requires-python = ">=3.12,<3.15"` in pyproject.toml.
#
# Usage:
#   scripts/ensure_tools.sh            # check, install what is missing
#   scripts/ensure_tools.sh --check    # check only, never install, exit 1 if incomplete
#
# Environment:
#   MODELZOO_NO_AUTO_INSTALL=1   never install anything, just report (same as --check)
#   PYTHON_BIN=/path/to/python3  force a specific interpreter for the venv
#
# Side effect: writes the resolved interpreter path to .make/python so the Makefile
# can build .venv with the right Python without re-running the search.

set -euo pipefail

# Mirrors requires-python = ">=3.12,<3.15" in pyproject.toml. Update both together.
PY_MIN_MINOR=12
PY_MAX_MINOR_EXCL=15

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_MARKER="${REPO_ROOT}/.make/python"

CHECK_ONLY=0
[[ "${1:-}" == "--check" ]] && CHECK_ONLY=1
[[ "${MODELZOO_NO_AUTO_INSTALL:-0}" == "1" ]] && CHECK_ONLY=1

# ~/.local/bin is where both installers land; make it visible for the rest of this run.
export PATH="${HOME}/.local/bin:${PATH}"

MISSING=0
INSTALLED=()

info()  { printf '  %s\n' "$*"; }
ok()    { printf '  \033[32m✓\033[0m %s\n' "$*"; }
warn()  { printf '  \033[33m!\033[0m %s\n' "$*"; }
fail()  { printf '  \033[31m✗\033[0m %s\n' "$*"; MISSING=1; }

# ---------------------------------------------------------------------------
# 1. System prerequisites — cannot be installed without a package manager/sudo
# ---------------------------------------------------------------------------
check_system_tool() {
    local tool="$1" apt_pkg="$2"
    if command -v "$tool" >/dev/null 2>&1; then
        ok "$tool"
    else
        fail "$tool is missing — install it with your package manager, e.g."
        info "    sudo apt-get install -y ${apt_pkg}     # Debian/Ubuntu"
        info "    sudo dnf install -y ${apt_pkg}         # Fedora/RHEL"
    fi
}

echo "→ System prerequisites"
check_system_tool curl curl
check_system_tool git git
check_system_tool make make

if [[ $MISSING -eq 1 ]]; then
    echo ""
    echo "✗ Missing system prerequisites — install them and re-run 'make bootstrap'."
    exit 1
fi

# ---------------------------------------------------------------------------
# 2. uv — needed both as a fast installer and as the Python-version fallback
# ---------------------------------------------------------------------------
echo "→ uv"
if command -v uv >/dev/null 2>&1; then
    ok "uv $(uv --version 2>/dev/null | awk '{print $2}')"
elif [[ $CHECK_ONLY -eq 1 ]]; then
    fail "uv is not installed (run 'make bootstrap', or: curl -LsSf https://astral.sh/uv/install.sh | sh)"
else
    info "installing uv into ~/.local/bin ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
    hash -r
    if command -v uv >/dev/null 2>&1; then
        ok "uv $(uv --version 2>/dev/null | awk '{print $2}') installed"
        INSTALLED+=("uv")
    else
        fail "uv installation did not put 'uv' on PATH — add ~/.local/bin to your PATH"
    fi
fi

# ---------------------------------------------------------------------------
# 3. Python interpreter in the supported range
# ---------------------------------------------------------------------------
python_in_range() {
    local py="$1" ver maj min
    ver="$("$py" -c 'import sys; print("%d %d" % sys.version_info[:2])' 2>/dev/null)" || return 1
    maj="${ver%% *}"; min="${ver##* }"
    [[ "$maj" -eq 3 && "$min" -ge $PY_MIN_MINOR && "$min" -lt $PY_MAX_MINOR_EXCL ]]
}

echo "→ Python (>=3.${PY_MIN_MINOR}, <3.${PY_MAX_MINOR_EXCL})"
RESOLVED_PYTHON=""
for candidate in "${PYTHON_BIN:-}" python3 python3.14 python3.13 python3.12 python; do
    [[ -z "$candidate" ]] && continue
    if command -v "$candidate" >/dev/null 2>&1 && python_in_range "$candidate"; then
        RESOLVED_PYTHON="$(command -v "$candidate")"
        break
    fi
done

if [[ -n "$RESOLVED_PYTHON" ]]; then
    ok "$("$RESOLVED_PYTHON" --version 2>&1) at ${RESOLVED_PYTHON}"
elif [[ $CHECK_ONLY -eq 1 ]]; then
    fail "no Python 3.${PY_MIN_MINOR}–3.$((PY_MAX_MINOR_EXCL - 1)) found (run 'make bootstrap' to fetch one with uv)"
elif command -v uv >/dev/null 2>&1; then
    info "no suitable system Python — fetching a managed 3.13 with uv ..."
    uv python install 3.13 >/dev/null
    RESOLVED_PYTHON="$(uv python find 3.13 2>/dev/null || true)"
    if [[ -n "$RESOLVED_PYTHON" ]] && python_in_range "$RESOLVED_PYTHON"; then
        ok "$("$RESOLVED_PYTHON" --version 2>&1) at ${RESOLVED_PYTHON} (uv-managed)"
        INSTALLED+=("python 3.13 (uv-managed)")
    else
        fail "uv could not provide a usable Python 3.13"
        RESOLVED_PYTHON=""
    fi
else
    fail "no suitable Python and no uv to fetch one"
fi

if [[ -n "$RESOLVED_PYTHON" ]]; then
    mkdir -p "$(dirname "$PYTHON_MARKER")"
    printf '%s\n' "$RESOLVED_PYTHON" > "$PYTHON_MARKER"
fi

# ---------------------------------------------------------------------------
# 4. Poetry — resolves pyproject.toml, owns poetry.lock, builds the wheel
# ---------------------------------------------------------------------------
echo "→ Poetry"
if command -v poetry >/dev/null 2>&1; then
    ok "$(poetry --version 2>&1)"
elif [[ $CHECK_ONLY -eq 1 ]]; then
    fail "poetry is not installed (run 'make bootstrap', or: uv tool install poetry)"
elif command -v uv >/dev/null 2>&1; then
    info "installing poetry with uv ..."
    uv tool install poetry >/dev/null
    hash -r
    if command -v poetry >/dev/null 2>&1; then
        ok "$(poetry --version 2>&1) installed"
        INSTALLED+=("poetry")
    else
        fail "poetry installed but not on PATH — add ~/.local/bin to your PATH"
    fi
else
    fail "cannot install poetry without uv"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
if [[ $MISSING -eq 1 ]]; then
    if [[ $CHECK_ONLY -eq 1 ]]; then
        echo "✗ Toolchain incomplete. Run 'make bootstrap' to install the missing pieces."
    else
        echo "✗ Toolchain incomplete — see the errors above."
    fi
    exit 1
fi

if [[ ${#INSTALLED[@]} -gt 0 ]]; then
    echo "✓ Toolchain ready (installed: ${INSTALLED[*]})"
    echo ""
    echo "  The new tools live in ~/.local/bin. If a fresh shell cannot find them, add"
    echo "  this line to your ~/.bashrc (or ~/.zshrc) and reopen the terminal:"
    echo ""
    echo "      export PATH=\"\$HOME/.local/bin:\$PATH\""
else
    echo "✓ Toolchain ready"
fi
