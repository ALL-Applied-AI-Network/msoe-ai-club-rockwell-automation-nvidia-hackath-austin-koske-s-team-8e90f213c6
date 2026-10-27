param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
$setupRoot = $PSScriptRoot
$toolRoot = if ($env:AI_CLUB_HOME) { $env:AI_CLUB_HOME } else { Join-Path $env:USERPROFILE 'ai-club-tools' }
$environmentPath = Join-Path $toolRoot 'envs\data-windows'
if (-not (Test-Path (Join-Path $environmentPath 'Scripts\python.exe'))) {
    & $Python -m venv $environmentPath
    if ($LASTEXITCODE -ne 0) { throw 'Create the data environment using Python 3.12.' }
}
$environmentPython = Join-Path $environmentPath 'Scripts\python.exe'
& $environmentPython -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else "This starter uses Python 3.12. Pass -Python with its executable path and use a fresh AI_CLUB_HOME.")'
if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 is required for the recorded package set.' }
& $environmentPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'pip installation failed.' }
$requirementPath = Join-Path $setupRoot 'requirements\data-windows.lock.txt'
if (-not (Test-Path -LiteralPath $requirementPath)) { $requirementPath = Join-Path $setupRoot 'requirements\data.in' }
& $environmentPython -m pip install -r $requirementPath
if ($LASTEXITCODE -ne 0) { throw 'Data dependencies failed to install.' }
& $environmentPython (Join-Path $setupRoot 'doctor.py') --profile data
if ($LASTEXITCODE -ne 0) { throw 'Data environment check failed.' }
Write-Host "Ready. Use: $environmentPython"
