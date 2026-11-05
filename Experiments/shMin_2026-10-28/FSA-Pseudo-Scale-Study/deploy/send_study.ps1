# Run in Windows PowerShell. Pull committed source and transfer it; never run experiments.
[CmdletBinding()]
param(
    [string]$Repo = (Join-Path $PSScriptRoot '../../../..'),
    [string]$Server = 'master@192.168.1.200',
    [string]$Branch = 'main'
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
Invoke-GitChecked check-ref-format --branch $Branch | Out-Null
if ((Invoke-GitChecked -C $Repo branch --show-current).Trim() -ne $Branch) {
    throw "Switch to the intended $Branch checkout before transferring."
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

Invoke-GitChecked -C $Repo pull --ff-only origin $Branch
$Revision = (Invoke-GitChecked -C $Repo rev-parse HEAD).Trim()
$Published = (Invoke-GitChecked -C $Repo rev-parse "refs/remotes/origin/$Branch").Trim()
if ($Revision -ne $Published) {
    throw "Local branch contains unpublished commits. Transfer stopped; only published origin/$Branch may be sent."
}
$ShortRevision = $Revision.Substring(0, 12)
$ZipName = "ins-cod-study-$ShortRevision.zip"
$Zip = Join-Path (Split-Path -Parent $Repo) $ZipName
# model.py imports the archived sibling FSA-ZoomNeXt; preserve that source layout too.
# Committed source only: no local datasets, virtual environments or checkpoints.
# git archive performs working-tree EOL conversion; a Windows autocrlf=true
# default otherwise exports Bash launchers with CRLF even from LF Git blobs.
Invoke-GitChecked -c core.autocrlf=false -c core.eol=lf -C $Repo archive --format=zip --output $Zip HEAD -- $Relative 'Experiments/FSA-ZoomNeXt' 'docs/MIGRATION.md'
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
Write-Host "Transferred branch: $Branch"
Write-Host 'Choose the source destination with install_study.py --work; use a separate directory for research branches.'
Write-Host 'Transfer only: extraction, evaluation and training have not started.'
