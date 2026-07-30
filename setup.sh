#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv}"

command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
  echo "Python 3.12 is required. Set PYTHON_BIN to its executable." >&2
  exit 1
}

"$PYTHON_BIN" -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -r "$PROJECT_ROOT/requirements-dev.txt"
"$VENV_DIR/bin/python" "$PROJECT_ROOT/scripts/verify_portable_layout.py"

echo "Environment ready: $VENV_DIR"
