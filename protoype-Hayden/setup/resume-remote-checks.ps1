# Refresh deployment and environment reports when campus SSH access is available.
param()
$ErrorActionPreference = 'Stop'
$setupRoot = $PSScriptRoot
$repositoryRoot = Split-Path $setupRoot -Parent
Push-Location $repositoryRoot
try {
    foreach ($hostAlias in @('msoe-spark','msoe-bastion')) {
        & ssh -o BatchMode=yes -o ConnectTimeout=10 $hostAlias 'echo SSH_OK'
        if ($LASTEXITCODE -ne 0) { throw "Restore access to $hostAlias before resuming." }
    }
    & (Join-Path $setupRoot 'deploy.ps1') -Target Spark -IncludeTwin
    & (Join-Path $setupRoot 'deploy.ps1') -Target Rosie
    & ssh msoe-bastion 'cd ~/ai-club-onboarding; bash setup/bootstrap.sh rosie > ~/ai-club-tools/logs/bootstrap-final.log 2>&1; result=$?; tail -20 ~/ai-club-tools/logs/bootstrap-final.log; exit $result'
    if ($LASTEXITCODE -ne 0) { throw 'Rosie bootstrap needs review; see the remote log.' }
    & ssh msoe-bastion 'cd ~/ai-club-onboarding; bash setup/rosie/run.sh --data setup/doctor.py --profile data --out ~/ai-club-tools/results/rosie-data.json; bash setup/rosie/submit.sh smoke'
    if ($LASTEXITCODE -ne 0) { throw 'Rosie data check or GPU submission failed.' }
    & ssh msoe-spark '~/.local/bin/uv pip freeze --python ~/ai-club-tools/envs/data/bin/python > ~/ai-club-tools/data-linux-aarch64.lock.txt; ~/.local/bin/uv pip freeze --python ~/ai-club-tools/envs/cosmos/bin/python > ~/ai-club-tools/cosmos-linux-aarch64.lock.txt; ~/.local/bin/uv pip freeze --python ~/ai-club-tools/envs/isaac-6.0.0.1/bin/python > ~/ai-club-tools/isaac-linux-aarch64.lock.txt'
    if ($LASTEXITCODE -ne 0) { throw 'Could not record Spark package versions.' }
    foreach ($profileName in @('data','cosmos','isaac')) {
        & scp "msoe-spark:ai-club-tools/$profileName-linux-aarch64.lock.txt" (Join-Path $setupRoot "requirements/$profileName-linux-aarch64.lock.txt")
        if ($LASTEXITCODE -ne 0) { throw "Could not collect the $profileName package list." }
    }
    $reports = @{
        'spark-data.json' = 'spark-data.json'
        'spark-cosmos-env.json' = 'spark-cosmos-env.json'
        'isaac-smoke/result.json' = 'isaac-smoke.json'
        'isaac-smoke/isaac-smoke.png' = 'isaac-smoke.png'
        'vial-render-first/render_status.json' = 'vial-render.json'
        'vial-render-first/s0000_e000003_v006.png' = 'vial-render.png'
    }
    foreach ($remoteReport in $reports.Keys) {
        & scp "msoe-spark:ai-club-tools/results/$remoteReport" (Join-Path $setupRoot ('verification/' + $reports[$remoteReport]))
        if ($LASTEXITCODE -ne 0) { throw "Could not collect $remoteReport." }
    }
    & scp 'msoe-bastion:ai-club-tools/results/rosie-data.json' (Join-Path $setupRoot 'verification/rosie-data.json')
    if ($LASTEXITCODE -ne 0) { throw 'Could not collect the Rosie data report.' }
    Write-Host 'Deployment, data checks and collection completed. Review the printed GPU job ID; the existing Cosmos reports are preserved.'
} finally {
    Pop-Location
}
