# Live line view on Windows: the API with the models on the GPU, the station's USB cameras, and the UI.
#   pwsh -File setup/run-line-live.ps1 -Station C:\path\station.json
# Open http://127.0.0.1:8794 yourself (the LINE page is the landing page); Ctrl+C stops. See docs/line-view.md ("Live cameras").
# WSL and Docker cannot see Windows webcams, so this runs native Windows Python.
param(
    [string]$Station = "",
    [string]$Python = "$env:USERPROFILE\ai-club-tools\envs\vision-windows\Scripts\python.exe",
    [string]$Assets = "$env:USERPROFILE\ai-club-tools\results\temporal-cnn-team-release-20261026-v1\temporal-cnn-assets-v1",
    [string]$Runs = "",
    [string]$Data = "$env:LOCALAPPDATA\linewatch-live\data",
    [string]$Device = "cuda:0",
    [int]$Port = 8794
)
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
$Site = "$env:LOCALAPPDATA\linewatch-live\site"
$Dist = Join-Path $Repo "client\dist"

if (-not (Test-Path $Python)) { throw "No Python at $Python (needs torch with CUDA, OpenCV, ultralytics, scikit-image)" }
if (-not (Test-Path "$Assets\models")) { throw "No model release at $Assets" }
if (-not (Test-Path "$Dist\index.html")) { throw "Build the UI first: cd client; npm ci; npm run build" }
# The API packages go in a side folder so the model environment itself is not changed.
if (-not (Test-Path "$Site\fastapi")) {
    & $Python -m pip install --quiet --target $Site fastapi "uvicorn[standard]" pydantic-settings imageio-ffmpeg
}
New-Item -ItemType Directory -Force -Path $Data | Out-Null

$env:PYTHONPATH = "$Site;$Repo\api;$Repo"
$env:APP__MODEL_ASSETS = $Assets
$env:APP__DEVICE = $Device
$env:APP__DATA_DIR = $Data
$env:APP__CLIENT_DIST = $Dist
$env:YOLO_CONFIG_DIR = "$Data\yolo"
if ($Station) { $env:APP__LINE_STATION = (Resolve-Path $Station).Path }
if ($Runs) { $env:APP__LINE_RUNS = (Resolve-Path $Runs).Path }

Write-Host "Linewatch live line view on http://127.0.0.1:$Port  (LINE page; device $Device, data $Data). Ctrl+C stops."
Set-Location "$Repo\api"
& $Python -m uvicorn main:app --host 127.0.0.1 --port $Port
