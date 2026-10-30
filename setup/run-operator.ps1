param([string]$Video, [string]$Evidence, [string]$Out, [int]$Port = 8766)
$ErrorActionPreference = 'Stop'
$toolRoot = if ($env:AI_CLUB_HOME) { $env:AI_CLUB_HOME } else { Join-Path $env:USERPROFILE 'ai-club-tools' }
$pythonPath = Join-Path $toolRoot 'envs/data-windows/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Run pwsh -File setup/bootstrap.ps1 first.' }
$arguments = @('-m', 'replay.operator', '--port', "$Port")
if ($Video) { $arguments += @('--video', $Video) }
if ($Evidence) { $arguments += @('--evidence', $Evidence) }
if ($Out) { $arguments += @('--out', $Out) }
Push-Location (Split-Path $PSScriptRoot -Parent)
try {
    & $pythonPath @arguments
    if ($LASTEXITCODE -ne 0) { throw 'Operator workspace stopped with an error.' }
} finally { Pop-Location }
