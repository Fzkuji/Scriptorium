#!/usr/bin/env bash
# Install the memory protocol into Codex.
#
# Codex has no plugin marketplace: hooks live in one global ~/.codex/hooks.json
# with absolute commands, and MCP servers in ~/.codex/config.toml. This adds
# our entries to both, leaving anything already there alone.
#
#   ./install-codex.sh              install
#   ./install-codex.sh --uninstall  remove

set -euo pipefail

PLUGIN_DIR="$(cd "$(dirname "$0")" && pwd)"
CODEX_DIR="${CODEX_HOME:-$HOME/.codex}"
HOOKS_JSON="${CODEX_DIR}/hooks.json"
CONFIG_TOML="${CODEX_DIR}/config.toml"
WORKSPACE="${SCRIPTORIUM_WORKSPACE:-$HOME/.scriptorium/memory}"
MARKER="scriptorium"

uninstall=0
[ "${1:-}" = "--uninstall" ] && uninstall=1

command -v python3 >/dev/null || { echo "python3 is required" >&2; exit 1; }
mkdir -p "$CODEX_DIR"

# hooks.json: add or drop our SessionStart entry, matched by the command text
# so a reinstall replaces rather than duplicates it.
python3 - "$HOOKS_JSON" "$PLUGIN_DIR/hooks/session-start" "$uninstall" <<'PY'
import json, os, sys

path, script, removing = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
try:
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
except (FileNotFoundError, json.JSONDecodeError):
    data = {}

hooks = data.setdefault("hooks", {})
entries = hooks.setdefault("SessionStart", [])
command = f"bash '{script}'"

# Drop any previous entry of ours, wherever it sits in the list.
kept = [
    entry for entry in entries
    if not any(script in h.get("command", "") for h in entry.get("hooks", []))
]

if not removing:
    kept.append({"hooks": [{"type": "command", "command": command, "timeout": 15}]})

if kept:
    hooks["SessionStart"] = kept
else:
    hooks.pop("SessionStart", None)
if not hooks:
    data.pop("hooks", None)

os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, "w", encoding="utf-8") as handle:
    json.dump(data, handle, indent=1)
    handle.write("\n")
print(("removed from " if removing else "wrote ") + path)
PY

# config.toml: append the MCP server block, or strip it back out. Editing TOML
# textually keeps every comment and ordering the user has in there.
python3 - "$CONFIG_TOML" "$WORKSPACE" "$uninstall" "$MARKER" <<'PY'
import os, sys

path, workspace, removing, marker = sys.argv[1], sys.argv[2], sys.argv[3] == "1", sys.argv[4]
text = ""
if os.path.exists(path):
    with open(path, encoding="utf-8") as handle:
        text = handle.read()

header = f"[mcp_servers.{marker}]"
lines = text.splitlines()

# Cut the existing block: its header through to the next top-level table.
start = next((i for i, line in enumerate(lines) if line.strip() == header), None)
if start is not None:
    end = start + 1
    while end < len(lines) and not lines[end].lstrip().startswith("["):
        end += 1
    while end > start + 1 and not lines[end - 1].strip():
        end -= 1
    del lines[start:end]
    while start < len(lines) and not lines[start].strip():
        del lines[start]

if not removing:
    if lines and lines[-1].strip():
        lines.append("")
    lines += [
        header,
        'command = "scriptorium"',
        f'args = ["mcp", "--workspace", "{workspace}"]',
    ]

with open(path, "w", encoding="utf-8") as handle:
    handle.write("\n".join(lines).rstrip("\n") + "\n")
print(("removed from " if removing else "wrote ") + path)
PY

if [ "$uninstall" = "1" ]; then
    echo "Scriptorium removed from Codex. The workspace at ${WORKSPACE} is untouched."
else
    echo
    echo "Scriptorium installed for Codex."
    echo "  memory:   ${WORKSPACE}"
    echo "  protocol: injected at session start"
    echo "Start a new Codex session to pick it up."
fi
