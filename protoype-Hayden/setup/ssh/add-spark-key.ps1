param([string]$HostName = 'hackathon8@spark8.ai.msoe.edu')
$ErrorActionPreference = 'Stop'
$keyPath = Join-Path $env:USERPROFILE '.ssh\ai_club_spark8_ed25519'
if (-not (Test-Path -LiteralPath $keyPath)) {
    & ssh-keygen -t ed25519 -f $keyPath -N '' -C 'ai-club-spark8'
    if ($LASTEXITCODE -ne 0) { throw 'Key generation failed. Generate an Ed25519 key with ssh-keygen and rerun.' }
}
$publicKey = (Get-Content -LiteralPath ($keyPath + '.pub') -Raw).Trim()
if ($publicKey -notmatch '^ssh-ed25519 [A-Za-z0-9+/=]+(?: [A-Za-z0-9@._-]+)?$') { throw 'Unexpected public key format.' }
$remoteCommand = "umask 077; mkdir -p ~/.ssh; touch ~/.ssh/authorized_keys; grep -qxF '$publicKey' ~/.ssh/authorized_keys || printf '%s\n' '$publicKey' >> ~/.ssh/authorized_keys; chmod 700 ~/.ssh; chmod 600 ~/.ssh/authorized_keys"
& ssh $HostName $remoteCommand
if ($LASTEXITCODE -ne 0) { throw 'Public key installation failed.' }
& ssh -o BatchMode=yes -o IdentitiesOnly=yes -i $keyPath $HostName 'echo KEY_AUTH_OK'
if ($LASTEXITCODE -ne 0) { throw 'The new key did not authenticate.' }
Write-Host 'Key installed. Add the Spark host block from setup/ssh/config.example to your own SSH configuration.'
