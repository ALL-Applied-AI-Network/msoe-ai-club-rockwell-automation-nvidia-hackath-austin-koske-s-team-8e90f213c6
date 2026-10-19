param([string]$Data, [string]$Out, [int]$Port = 8765, [string]$RecipePlan, [string]$References, [string]$Config)
$ErrorActionPreference = 'Stop'
$toolRoot = if ($env:AI_CLUB_HOME) { $env:AI_CLUB_HOME } else { Join-Path $env:USERPROFILE 'ai-club-tools' }
$pythonPath = Join-Path $toolRoot 'envs/data-windows/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Run pwsh -File setup/bootstrap.ps1 first.' }
$arguments = @('-m', 'replay', 'serve', '--port', "$Port")
if ($Data) { $arguments += @('--data', $Data) }
if ($Out) { $arguments += @('--out', $Out) }
if ($RecipePlan) { $arguments += @('--recipe-plan', $RecipePlan) }
if ($References) { $arguments += @('--references', $References) }
if ($Config) { $arguments += @('--config', $Config) }
Push-Location (Split-Path $PSScriptRoot -Parent)
try {
    & $pythonPath @arguments
    if ($LASTEXITCODE -ne 0) { throw 'Replay server stopped with an error.' }
} finally { Pop-Location }
