#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
# Keep the environment outside a syncing folder. iCloud sets a hidden flag on
# files it syncs, and Python skips a hidden .pth, which silently disables an
# editable install. $PROJECT_ROOT/.venv may be a symlink to this.
VENV_DIR="${VENV_DIR:-$HOME/.venvs/scriptorium}"

command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
  echo "Python 3.12 is required. Set PYTHON_BIN to its executable." >&2
  exit 1
}

command -v rg >/dev/null 2>&1 || {
  echo "ripgrep is required by the read-only retrieval shell." >&2
  echo "Install it first (Ubuntu: sudo apt-get install ripgrep)." >&2
  exit 1
}

# Re-running against an existing environment must not rebind it to a different
# interpreter: the packages inside it hold compiled extensions built for the
# one it was created with, and loading those under another build segfaults.
if [[ -f "$VENV_DIR/pyvenv.cfg" ]]; then
  # Compare the interpreters themselves, not how each was spelled: a symlink
  # such as /opt/homebrew/opt/python@3.12/bin/python3.12 is the same build as
  # the Cellar path recorded in pyvenv.cfg.
  existing="$(python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' \
    "$(sed -n 's/^executable = //p' "$VENV_DIR/pyvenv.cfg")")"
  wanted="$("$PYTHON_BIN" -c 'import os,sys; print(os.path.realpath(sys.executable))')"
  if [[ "$existing" != "$wanted" ]]; then
    echo "$VENV_DIR was built with $existing, but PYTHON_BIN is $wanted." >&2
    echo "Delete $VENV_DIR to rebuild, or set PYTHON_BIN to that interpreter." >&2
    exit 1
  fi
else
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -r "$PROJECT_ROOT/requirements-dev.txt"
# The layout check lives with the tests that define it, so a clone with a
# missing or duplicated directory fails here rather than mid-experiment.
"$VENV_DIR/bin/python" -m pytest -q "$PROJECT_ROOT/code/tests/test_portable_layout.py"

if [[ ! -e "$PROJECT_ROOT/.venv" ]]; then
  ln -s "$VENV_DIR" "$PROJECT_ROOT/.venv"
fi

echo "Environment ready: $VENV_DIR"
