param(
    [Parameter(Mandatory=$true)][string]$Assets,
    [int]$Port = 8788,
    [switch]$Build
)
$ErrorActionPreference = 'Stop'
$assetsPath = (Resolve-Path $Assets).Path
if (Test-Path $assetsPath -PathType Leaf) {
    if ([System.IO.Path]::GetExtension($assetsPath) -ne '.zip') { throw 'Assets must be the extracted model folder or its ZIP' }
    $extracted = "$assetsPath.extracted"
    if (-not (Test-Path $extracted)) { Expand-Archive -LiteralPath $assetsPath -DestinationPath $extracted }
    $assetsPath = $extracted
}
if (-not (Test-Path (Join-Path $assetsPath 'models/liquid-best.pt'))) {
    $candidates = @(Get-ChildItem $assetsPath -Directory | Where-Object { Test-Path (Join-Path $_.FullName 'models/liquid-best.pt') })
    if ($candidates.Count -ne 1) { throw 'Choose the folder containing models/liquid-best.pt and models/vial-best.pt' }
    $assetsPath = $candidates[0].FullName
}
$pins = @{
    'models/liquid-best.pt' = '6714fe32191d8eba230aa87246f38a25c2906c619d60d7ba30c9e60205352eef'
    'models/vial-best.pt' = '0bfac07a256eaf190416b632ad1f094a6b534309df1bb11a8c175e9d811b56bc'
}
foreach ($file in $pins.Keys) {
    $actual = (Get-FileHash (Join-Path $assetsPath $file) -Algorithm SHA256).Hash.ToLower()
    if ($actual -ne $pins[$file]) { throw "Model hash mismatch: $file" }
}
$env:LINEWATCH_ASSETS_DIR = $assetsPath.Replace('\', '/')
$env:LINEWATCH_PORT = "$Port"
Push-Location (Split-Path $PSScriptRoot -Parent)
try {
    if ($Build) { docker compose up -d --build } else { docker compose up -d }
    if ($LASTEXITCODE -ne 0) { throw 'Docker Compose failed; inspect docker compose logs' }
    Write-Host "Linewatch: http://localhost:$Port"
    Write-Host 'Reference is a preview example, not a certified empty vial. No recipe approval is inferred.'
} finally { Pop-Location }
