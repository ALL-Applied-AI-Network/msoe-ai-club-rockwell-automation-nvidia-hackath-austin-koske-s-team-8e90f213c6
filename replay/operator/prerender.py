"""Pinned input-only video cache and a pauseable server-authoritative 1x clock."""
import hashlib
import json
import math
from pathlib import Path
import time


class PlaybackClock:
    def __init__(self,stream_id,duration,now=time.monotonic):
        if not isinstance(stream_id,str) or not stream_id or not math.isfinite(duration) or duration<=0:
            raise ValueError('Playback requires a stream identity and finite positive duration')
        self.stream_id=stream_id;self.duration=duration;self.now=now
        self.anchor=now();self.offset=0.;self.paused=False

    def seconds(self):return min(self.duration,self.offset+(0. if self.paused else max(0.,self.now()-self.anchor)))

    def pause(self,paused):
        if paused!=self.paused:
            self.offset=self.seconds();self.anchor=self.now();self.paused=paused

    def snapshot(self):return dict(stream_id=self.stream_id,seconds=self.seconds(),duration_s=self.duration,paused=self.paused,speed=1.)


class PrerenderBundle:
    def __init__(self,root):
        self.root=Path(root).resolve();self.manifest=json.loads((self.root/'manifest.json').read_text())
        m=self.manifest
        if (m.get('schema_version')!=1 or m.get('kind')!='prerendered_isaac_input_only' or m.get('cached_predictions') is not False
            or m.get('fps')!=30 or (m.get('width'),m.get('height'))!=(640,480) or not m.get('samples')
            or not math.isfinite(m['duration_s']) or m['duration_s']<=0):raise ValueError('Expected a completed 30 fps input-only Isaac cache')
        self.video=self.file(m['video']);self.duration=m['duration_s']
        if self.sha(self.video)!=m['video_sha256']:raise ValueError('Cached video hash differs')
        previous=-1.
        for s in m['samples']:
            if not previous<s['seconds']<self.duration:raise ValueError('Cached sample clock is not increasing')
            if self.sha(self.file(s['png']))!=s['png_sha256']:raise ValueError('Cached sample hash differs')
            previous=s['seconds']

    def file(self,name):
        path=(self.root/name).resolve()
        if not path.is_relative_to(self.root):raise ValueError('Cache path leaves its pinned directory')
        return path

    @staticmethod
    def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
