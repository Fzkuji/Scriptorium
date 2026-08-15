#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=opencode_bridge.env
source "$SCRIPT_DIR/opencode_bridge.env"

INSTALL_DIR="${SCRIPTORIUM_OPENCODE_BRIDGE_HOME:-$HOME/.local/share/scriptorium/opencode-bridge}"

command -v git >/dev/null 2>&1 || {
  echo "git is required to install the OpenCode bridge." >&2
  exit 1
}
command -v node >/dev/null 2>&1 || {
  echo "Node.js 18 or newer is required for the OpenCode bridge." >&2
  exit 1
}

NODE_MAJOR="$(node -p 'process.versions.node.split(".")[0]')"
if (( NODE_MAJOR < 18 )); then
  echo "Node.js 18 or newer is required; found $(node --version)." >&2
  exit 1
fi

if [[ -e "$INSTALL_DIR" && ! -d "$INSTALL_DIR/.git" ]]; then
  echo "$INSTALL_DIR exists but is not a bridge Git checkout." >&2
  exit 1
fi

if [[ ! -d "$INSTALL_DIR/.git" ]]; then
  mkdir -p "$(dirname "$INSTALL_DIR")"
  git clone "$OPENCODE_BRIDGE_REPOSITORY" "$INSTALL_DIR"
fi

ACTUAL_REMOTE="$(git -C "$INSTALL_DIR" remote get-url origin)"
if [[ "$ACTUAL_REMOTE" != "$OPENCODE_BRIDGE_REPOSITORY" ]]; then
  echo "Unexpected bridge origin: $ACTUAL_REMOTE" >&2
  echo "Expected: $OPENCODE_BRIDGE_REPOSITORY" >&2
  exit 1
fi

git -C "$INSTALL_DIR" fetch --quiet origin "$OPENCODE_BRIDGE_COMMIT"
git -C "$INSTALL_DIR" checkout --quiet --detach "$OPENCODE_BRIDGE_COMMIT"

ACTUAL_COMMIT="$(git -C "$INSTALL_DIR" rev-parse HEAD)"
if [[ "$ACTUAL_COMMIT" != "$OPENCODE_BRIDGE_COMMIT" ]]; then
  echo "Bridge commit verification failed: $ACTUAL_COMMIT" >&2
  exit 1
fi

echo "OpenCode bridge installed at $INSTALL_DIR"
echo "Version $OPENCODE_BRIDGE_VERSION, commit $ACTUAL_COMMIT"
echo "Start it with: $SCRIPT_DIR/run_opencode_bridge.sh"
