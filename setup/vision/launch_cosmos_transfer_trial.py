"""Bounded launcher with preserved output, PID/exit evidence and GPU lock wrapper."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main(a):
    marker=a.root/f'launch-{a.variant}-{a.attempt}.json';log=a.root/f'inference-{a.variant}-{a.attempt}.log'
    if marker.exists() or log.exists():raise ValueError('Preserve previous attempts; select a fresh attempt')
    info=dict(status='starting',variant=a.variant,attempt=a.attempt,started_unix=time.time(),
        input_specification_sha256=sha(a.root/'inputs-v1'/f'transfer-{a.variant}.json'),
        environment_sha256=sha(a.root/'environment-freeze.txt'),
        download_manifest_sha256=sha(a.root/'downloads-completed.json'),
        wrapper_sha256=sha(a.root/'run_cosmos_transfer_trial.sh'),
        launcher_sha256=sha(Path(__file__)),training=False)
    with log.open('w') as f:
        process=subprocess.Popen(['bash',str(a.root/'run_cosmos_transfer_trial.sh'),str(a.root),a.variant,a.attempt],
            stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        info.update(status='running',pid=process.pid);marker.write_text(json.dumps(info,indent=2)+'\n')
        try:code=process.wait(timeout=7200)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            code=124
    video=a.root/f'outputs-{a.variant}-{a.attempt}'/f'vial-{a.variant}.mp4'
    text=log.read_text(errors='replace')
    produced=video.is_file() and video.stat().st_size>0
    guardrails=all(s in text for s in ['Passed guardrail on prompt','Passed guardrail on generated video']) and 'Unexpected error' not in text
    info.update(status='completed' if code==0 and produced else 'failed',exit_code=code,
        output_video_present=produced,guardrail_success_logged_without_error=guardrails,
        elapsed_s=time.time()-info['started_unix'],
        log_sha256=sha(log));marker.write_text(json.dumps(info,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--variant',choices=['neutral','cool-light','warm-light'],default='neutral')
    p.add_argument('--attempt',default='v1');main(p.parse_args())
