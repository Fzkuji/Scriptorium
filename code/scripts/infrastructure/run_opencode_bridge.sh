#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=opencode_bridge.env
source "$SCRIPT_DIR/opencode_bridge.env"

INSTALL_DIR="${SCRIPTORIUM_OPENCODE_BRIDGE_HOME:-$HOME/.local/share/scriptorium/opencode-bridge}"
CONFIG_PATH="${SCRIPTORIUM_OPENCODE_BRIDGE_CONFIG:-$INSTALL_DIR/config.json}"

[[ -f "$INSTALL_DIR/server.js" ]] || {
  echo "OpenCode bridge is not installed at $INSTALL_DIR." >&2
  echo "Run $SCRIPT_DIR/install_opencode_bridge.sh first." >&2
  exit 1
}
[[ -f "$CONFIG_PATH" ]] || {
  echo "Bridge config not found: $CONFIG_PATH" >&2
  exit 1
}

# Local bridge traffic must not be routed through a corporate/system proxy.
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
export no_proxy="127.0.0.1,localhost${no_proxy:+,$no_proxy}"

# Keep this process in the foreground so systemd/launchd/tmux can supervise it.
exec node "$INSTALL_DIR/server.js" --config "$CONFIG_PATH"
