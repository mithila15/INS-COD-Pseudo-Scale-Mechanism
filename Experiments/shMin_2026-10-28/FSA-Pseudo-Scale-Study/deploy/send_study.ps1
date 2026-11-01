# Run in Windows PowerShell. Pull committed source and transfer it; never run experiments.
[CmdletBinding()]
param(
    [string]$Repo = (Join-Path $PSScriptRoot '../../../..'),
    [string]$Server = 'master@192.168.1.200'
)

$ErrorActionPreference = 'Stop'
function Invoke-GitChecked {
    & git @args
    if ($LASTEXITCODE -ne 0) {
        throw "Git failed (exit $LASTEXITCODE). Transfer stopped."
    }
}

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw 'Git is unavailable. Open Windows PowerShell with Git for Windows installed.'
}
$Repo = (Resolve-Path -LiteralPath $Repo).Path
$Relative = 'Experiments/shMin_2026-10-28/FSA-Pseudo-Scale-Study'
$Origin = (Invoke-GitChecked -C $Repo remote get-url origin).Trim()
if ($Origin -notin @('https://github.com/sh-min/INS_COD.git', 'https://github.com/sh-min/INS_COD',
                    'git@github.com:sh-min/INS_COD.git')) {
    throw "Unexpected repository origin: $Origin"
}
if ((Invoke-GitChecked -C $Repo branch --show-current).Trim() -ne 'main') {
    throw 'Switch to the intended main checkout before transferring.'
}
if (Invoke-GitChecked -C $Repo status --porcelain) {
    throw 'This checkout has local changes. Preserve them before pulling; nothing was reset or stashed.'
}
$ScpCommand = Get-Command scp.exe -ErrorAction SilentlyContinue
if ($ScpCommand) {
    $Scp = $ScpCommand.Source
} else {
    $Scp = Join-Path $env:WINDIR 'System32/OpenSSH/scp.exe'
    if (-not (Test-Path -LiteralPath $Scp -PathType Leaf)) {
        throw 'OpenSSH Client is unavailable. Enable the Windows OpenSSH Client optional feature first.'
    }
}

Invoke-GitChecked -C $Repo pull --ff-only origin main
$Revision = (Invoke-GitChecked -C $Repo rev-parse HEAD).Trim()
$Published = (Invoke-GitChecked -C $Repo rev-parse refs/remotes/origin/main).Trim()
if ($Revision -ne $Published) {
    throw 'Local main contains unpublished commits. Transfer stopped; only the published origin/main may be sent.'
}
$ShortRevision = $Revision.Substring(0, 12)
$ZipName = "ins-cod-study-$ShortRevision.zip"
$Zip = Join-Path (Split-Path -Parent $Repo) $ZipName
# model.py imports the archived sibling FSA-ZoomNeXt; preserve that source layout too.
# Committed source only: no local datasets, virtual environments or checkpoints.
Invoke-GitChecked -C $Repo archive --format=zip --output $Zip HEAD -- $Relative 'Experiments/FSA-ZoomNeXt' 'docs/MIGRATION.md'
$Digest = (Get-FileHash -LiteralPath $Zip -Algorithm SHA256).Hash.ToLowerInvariant()
$Checksum = "$Zip.sha256"
[System.IO.File]::WriteAllText($Checksum, "$Digest  $ZipName`n", [System.Text.Encoding]::ASCII)

& $Scp -o ConnectTimeout=15 $Zip $Checksum "${Server}:/home/master/"
if ($LASTEXITCODE -ne 0) {
    throw "SCP failed (exit $LASTEXITCODE). Local archive remains at $Zip"
}
Write-Host "Transferred commit: $Revision"
Write-Host "Local archive: $Zip"
Write-Host "Server archive: /home/master/$ZipName"
Write-Host "Version for server installation: $ShortRevision"
Write-Host 'Active source destination after install_study.py: /home/master/mithila/shMin'
Write-Host 'Transfer only: extraction, evaluation and training have not started.'
