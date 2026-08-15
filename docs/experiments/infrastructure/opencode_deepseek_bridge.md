# OpenCode Go / DeepSeek provider setup

Scriptorium's Writer, Manager, verification, and LongMemEval query processes
use the Claude Agent SDK. OpenCode Go exposes an OpenAI-compatible upstream,
so it needs a local Anthropic-compatible translation bridge:

```text
Scriptorium / Claude Agent SDK
        -> http://127.0.0.1:8787
        -> OpenCode Go API
        -> DeepSeek model
```

This bridge is **not** built into OpenCode and is not a Python dependency. It
is the third-party MIT-licensed project
[`superheroYu/deepseek-v4-opencode-claude-code-bridge`](https://github.com/superheroYu/deepseek-v4-opencode-claude-code-bridge).
Scriptorium pins version `0.2.1` at commit
`e1212d246d05720a1d85853575dd3687b0067a94` in
`code/scripts/infrastructure/opencode_bridge.env`.

## Install

Node.js 18 or newer, Git, and curl are required. On either WSL/Linux or macOS:

```bash
bash code/scripts/infrastructure/install_opencode_bridge.sh
```

The default installation is outside the repository at
`~/.local/share/scriptorium/opencode-bridge`. Override it consistently with:

```bash
export SCRIPTORIUM_OPENCODE_BRIDGE_HOME=/absolute/path/to/opencode-bridge
```

The installer checks out the exact pinned commit. Re-running it verifies the
origin and restores that commit rather than silently using a newer bridge.

## Start and verify

Run the bridge in the foreground:

```bash
bash code/scripts/infrastructure/run_opencode_bridge.sh
```

In another terminal, verify it before starting an experiment:

```bash
bash code/scripts/infrastructure/check_opencode_bridge.sh
```

The health check deliberately bypasses HTTP proxies for localhost. A plain
`curl` can otherwise receive a proxy-generated 502 and misreport bridge state.

For unattended operation, supervise the foreground wrapper with `tmux`, a
systemd user service, or a macOS LaunchAgent. On WSL, `nohup` alone is not a
guarantee: Windows may stop the WSL VM after its owning session ends. The
upstream bridge checkout also contains optional systemd and LaunchAgent
installers, but their lifecycle remains machine-local.

## Credentials and runner configuration

Do not put an API key in the bridge's `config.json`. The bridge receives the
key from the SDK request header and forwards it to OpenCode Go. Keep the key in
a permission-restricted file outside Git, and point the Scriptorium run config
at that file.

The relevant experiment values are:

```json
{
  "provider_name": "opencode-go",
  "gateway_base_url": "http://127.0.0.1:8787",
  "model": "deepseek-v4-flash",
  "api_key_file": "/absolute/path/to/provider-api-key.txt"
}
```

Some runners use `base_url` rather than `gateway_base_url`; the value is the
same. Configuration must contain the key-file path, never the key text.

If a provider offers the Anthropic protocol and Claude Agent SDK tool semantics
directly, configure that provider URL without this bridge. An OpenAI-compatible
endpoint alone is not sufficient for these agent runners.

## Optional installation during repository setup

Python dependencies remain in `requirements*.txt`. To install the separate
Node bridge during normal repository setup, opt in explicitly:

```bash
INSTALL_OPENCODE_BRIDGE=1 ./setup.sh
```

Setup installs and pins the bridge but does not start a persistent service and
does not read a credential. Always run the health check before an experiment.

## macOS and WSL parity

Both systems use the same bridge source, commit, port, model identifier, and
request semantics. Only the supervisor and filesystem paths differ. Record the
bridge commit, provider URL, model, repository commit, and dataset hash with
each formal run so cross-machine results remain attributable.
