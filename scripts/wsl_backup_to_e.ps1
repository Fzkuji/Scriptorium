<#
.SYNOPSIS
Safely exports a WSL 2 distribution to drive E: without deleting or unregistering
the original distribution.

.DESCRIPTION
This is phase 1 of a cautious WSL migration. It only creates and verifies a VHDX
backup. It never runs `wsl --unregister`, never deletes the original ext4.vhdx,
and never changes the default WSL distribution.

Run this script from Windows PowerShell after closing Codex, Claude, VS Code,
Cursor, Docker Desktop, and all WSL terminals.
#>

[CmdletBinding()]
param(
    [Parameter()]
    [string]$DistroName,

    [Parameter()]
    [ValidateNotNullOrEmpty()]
    [string]$DestinationRoot = 'E:\WSL-Migration',

    [Parameter()]
    [ValidateRange(5, 500)]
    [int]$MinimumFreeGB = 30
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Stop-WithMessage {
    param([Parameter(Mandatory)][string]$Message)
    Write-Host "`nStopped: $Message" -ForegroundColor Red
    Write-Host 'The original WSL distribution on C: was not deleted or unregistered.' -ForegroundColor Yellow
    exit 1
}

function Get-WslDistributions {
    $raw = & wsl.exe --list --quiet 2>&1
    if ($LASTEXITCODE -ne 0) {
        Stop-WithMessage "Could not read the WSL distribution list. Details: $($raw -join ' ')"
    }

    @(
        $raw |
            ForEach-Object { ($_ -replace "`0", '').Trim() } |
            Where-Object { $_ -and $_ -notmatch '^docker-desktop(-data)?$' }
    )
}

Write-Host 'WSL safe backup - Phase 1 (export only, no deletion)' -ForegroundColor Cyan
Write-Host 'Close Codex, Claude, VS Code, Cursor, Docker Desktop, and all WSL terminals first.'
Write-Host 'This script never unregisters a distribution or deletes the original virtual disk.' -ForegroundColor Green

$wslCommand = Get-Command wsl.exe -ErrorAction SilentlyContinue
if (-not $wslCommand) {
    Stop-WithMessage 'wsl.exe was not found on this system.'
}

$distributions = Get-WslDistributions
if ($distributions.Count -eq 0) {
    Stop-WithMessage 'No WSL distribution was found for backup.'
}

if ([string]::IsNullOrWhiteSpace($DistroName)) {
    if ($distributions.Count -eq 1) {
        $DistroName = $distributions[0]
    }
    else {
        Write-Host "`nDetected WSL distributions:"
        for ($index = 0; $index -lt $distributions.Count; $index++) {
            Write-Host ("  [{0}] {1}" -f ($index + 1), $distributions[$index])
        }

        $selectionText = Read-Host 'Enter the number of the distribution to back up'
        $selection = 0
        if (-not [int]::TryParse($selectionText, [ref]$selection) -or
            $selection -lt 1 -or
            $selection -gt $distributions.Count) {
            Stop-WithMessage 'The selected number is invalid.'
        }
        $DistroName = $distributions[$selection - 1]
    }
}
elseif ($DistroName -notin $distributions) {
    Stop-WithMessage "No WSL distribution named '$DistroName' was found."
}

$destinationFullPath = [System.IO.Path]::GetFullPath($DestinationRoot)
$destinationRootPath = [System.IO.Path]::GetPathRoot($destinationFullPath)
$destinationDrive = [System.IO.DriveInfo]::new($destinationRootPath)

if (-not $destinationDrive.IsReady) {
    Stop-WithMessage "The destination drive $destinationRootPath is not ready."
}

if ($destinationDrive.DriveFormat -ne 'NTFS') {
    Stop-WithMessage "The destination drive must use NTFS; its format is $($destinationDrive.DriveFormat)."
}

$freeGB = [math]::Round($destinationDrive.AvailableFreeSpace / 1GB, 2)
if ($freeGB -lt $MinimumFreeGB) {
    Stop-WithMessage "The destination drive has only $freeGB GB free; at least $MinimumFreeGB GB is required."
}

$safeDistroName = $DistroName -replace '[^a-zA-Z0-9._-]', '_'
$timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$backupDirectory = Join-Path $destinationFullPath "$safeDistroName-$timestamp"
$vhdPath = Join-Path $backupDirectory 'ext4.vhdx'
$logPath = Join-Path $backupDirectory 'backup.log'
$manifestPath = Join-Path $backupDirectory 'manifest.json'

if (Test-Path -LiteralPath $backupDirectory) {
    Stop-WithMessage "The destination directory already exists: $backupDirectory"
}

New-Item -ItemType Directory -Path $backupDirectory -Force | Out-Null
Start-Transcript -LiteralPath $logPath -Force | Out-Null

try {
    Write-Host "`nDistribution: $DistroName"
    Write-Host "Destination: $backupDirectory"
    Write-Host "Free space on destination drive: $freeGB GB"

    $confirmation = Read-Host "Type BACKUP to create the non-destructive migration backup"
    if ($confirmation -cne 'BACKUP') {
        throw 'BACKUP was not entered; the operation was cancelled.'
    }

    Write-Host "`nShutting down WSL..." -ForegroundColor Cyan
    & wsl.exe --shutdown
    if ($LASTEXITCODE -ne 0) {
        throw "wsl --shutdown failed with exit code $LASTEXITCODE"
    }

    Start-Sleep -Seconds 2

    Write-Host 'Creating a VHDX backup with the official WSL export command...' -ForegroundColor Cyan
    & wsl.exe --export $DistroName $vhdPath --vhd
    if ($LASTEXITCODE -ne 0) {
        throw "WSL export failed with exit code $LASTEXITCODE"
    }

    if (-not (Test-Path -LiteralPath $vhdPath -PathType Leaf)) {
        throw 'The export command ended without creating a VHDX file.'
    }

    $vhdFile = Get-Item -LiteralPath $vhdPath
    if ($vhdFile.Length -lt 100MB) {
        throw "The generated VHDX is unexpectedly small: $([math]::Round($vhdFile.Length / 1MB, 2)) MB"
    }

    Write-Host 'Calculating the SHA-256 checksum...' -ForegroundColor Cyan
    $hash = Get-FileHash -LiteralPath $vhdPath -Algorithm SHA256

    $manifest = [ordered]@{
        schema_version       = 1
        created_at           = (Get-Date).ToString('o')
        source_distribution  = $DistroName
        backup_file          = $vhdPath
        backup_size_bytes    = $vhdFile.Length
        backup_size_gb       = [math]::Round($vhdFile.Length / 1GB, 3)
        sha256               = $hash.Hash
        original_unregistered = $false
        original_deleted      = $false
        next_step             = 'Reopen Codex and verify this backup before any cleanup.'
    }

    $manifest | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath $manifestPath -Encoding UTF8

    Write-Host "`nBackup created successfully." -ForegroundColor Green
    Write-Host "File: $vhdPath"
    Write-Host "Size: $($manifest.backup_size_gb) GB"
    Write-Host "SHA-256: $($manifest.sha256)"
    Write-Host "Manifest: $manifestPath"
    Write-Host "Log: $logPath"
    Write-Host "`nReopen Codex to verify the backup. Do not delete the original WSL distribution yet." -ForegroundColor Yellow
}
catch {
    Write-Host "`nBackup did not complete: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host 'The original WSL distribution on C: was not deleted or unregistered.' -ForegroundColor Yellow
    Write-Host "Log: $logPath"
    exit 1
}
finally {
    try {
        Stop-Transcript | Out-Null
    }
    catch {
        # Transcript may not have started; there is nothing to stop.
    }
}
