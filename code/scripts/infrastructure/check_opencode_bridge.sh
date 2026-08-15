#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=opencode_bridge.env
source "$SCRIPT_DIR/opencode_bridge.env"

BRIDGE_URL="${SCRIPTORIUM_OPENCODE_BRIDGE_URL:-http://127.0.0.1:$OPENCODE_BRIDGE_DEFAULT_PORT}"
command -v curl >/dev/null 2>&1 || {
  echo "curl is required for the bridge health check." >&2
  exit 1
}

# --noproxy is intentional: without it a proxy can return its own 502 and make
# a stopped local bridge look reachable.
curl --fail --silent --show-error --noproxy '*' "$BRIDGE_URL/health" >/dev/null
echo "OpenCode bridge healthy: $BRIDGE_URL"
