# Local PackyAPI -> Claude Agent SDK configuration.
# The secret itself stays in provider-api-key.txt and is never printed.

$keyFile = Join-Path $PSScriptRoot "provider-api-key.txt"
if (-not (Test-Path -LiteralPath $keyFile)) {
    throw "Missing API key file: $keyFile"
}

$apiKey = (Get-Content -LiteralPath $keyFile -Raw).Trim()
if ([string]::IsNullOrWhiteSpace($apiKey)) {
    throw "API key file is empty: $keyFile"
}

# Variables consumed by Claude Code / Claude Agent SDK.
$env:ANTHROPIC_BASE_URL = "https://www.packyapi.ai"
$env:ANTHROPIC_AUTH_TOKEN = $apiKey
$env:ANTHROPIC_MODEL = "deepseek-v4-flash"

# Do not echo $apiKey or $env:ANTHROPIC_AUTH_TOKEN.
Write-Host "Packy CC configuration loaded (key hidden)."
Write-Host "ANTHROPIC_BASE_URL=$env:ANTHROPIC_BASE_URL"
Write-Host "ANTHROPIC_MODEL=$env:ANTHROPIC_MODEL"
