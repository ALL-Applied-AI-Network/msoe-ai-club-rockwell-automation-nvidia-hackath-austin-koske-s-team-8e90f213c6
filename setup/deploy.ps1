param([ValidateSet('Spark','Rosie')][string]$Target = 'Spark', [switch]$IncludeTwin)
$ErrorActionPreference = 'Stop'
$hostAlias = if ($Target -eq 'Spark') { 'msoe-spark' } else { 'msoe-bastion' }
$repositoryRoot = Split-Path $PSScriptRoot -Parent
$archivePath = Join-Path ([System.IO.Path]::GetTempPath()) ('ai-club-setup-' + [guid]::NewGuid().ToString('N') + '.tar.gz')
Push-Location $repositoryRoot
try {
    $packagePaths = @('setup')
    if ($IncludeTwin) {
        $packagePaths += @('datasets/mixing_twin/README.md', 'datasets/mixing_twin/mixing',
                          'datasets/mixing_twin/scripts/render_isaac.py', 'datasets/mixing_twin/assets')
    }
    & tar --exclude=config.env --exclude=__pycache__ --exclude=results --exclude=logs --exclude='*.local' -czf $archivePath @packagePaths
    if ($LASTEXITCODE -ne 0) { throw 'Could not package setup.' }
    & ssh $hostAlias 'mkdir -p ~/ai-club-onboarding'
    if ($LASTEXITCODE -ne 0) { throw 'SSH connection failed. Check your own SSH configuration.' }
    & scp $archivePath "${hostAlias}:ai-club-onboarding/setup-upload.tar.gz"
    if ($LASTEXITCODE -ne 0) { throw 'Upload failed.' }
    & ssh $hostAlias 'tar -xzf ~/ai-club-onboarding/setup-upload.tar.gz -C ~/ai-club-onboarding'
    if ($LASTEXITCODE -ne 0) { throw 'Remote extraction failed.' }
    Write-Host "Ready on $hostAlias in ~/ai-club-onboarding."
} finally {
    Pop-Location
    if (Test-Path -LiteralPath $archivePath) { Remove-Item -LiteralPath $archivePath }
}
