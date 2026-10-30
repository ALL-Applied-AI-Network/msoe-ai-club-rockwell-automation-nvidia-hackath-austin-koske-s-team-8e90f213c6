"""Source-neutral local frame boundary. No device, URL or industrial connection."""
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from urllib.request import Request, urlopen


FIELDS=frozenset(('schema_version','source_id','stream_id','kind','sequence','pts','time_base',
                  'width','height','rgb_sha256','segment_id','telemetry'))


@dataclass(frozen=True)
class IncomingFrame:
    metadata: dict
    rgb: object

    @classmethod
    def decode(cls,metadata,encoded,expected_source,expected_kind):
        import cv2
        import numpy as np
        if not isinstance(metadata,dict) or set(metadata)!=FIELDS or metadata['schema_version']!=1:
            raise ValueError('Expected frame-input/v1 fields')
        if metadata['source_id']!=expected_source or metadata['kind']!=expected_kind:
            raise ValueError('Frame does not belong to this configured source')
        for key in ('stream_id','source_id'):
            if not isinstance(metadata[key],str) or not 1<=len(metadata[key])<=100:
                raise ValueError('Explicit source/stream identity required')
        if metadata['segment_id'] is not None and (not isinstance(metadata['segment_id'],str) or not 1<=len(metadata['segment_id'])<=100):
            raise ValueError('Invalid optional segment identity')
        if type(metadata['sequence']) is not int or metadata['sequence']<0 or type(metadata['pts']) is not int:
            raise ValueError('Integer sequence and native PTS required')
        tb=Fraction(metadata['time_base']);seconds=float(metadata['pts']*tb)
        if tb<=0 or not math.isfinite(seconds) or seconds<0:raise ValueError('Invalid source clock')
        if (metadata['width'],metadata['height'])!=(640,480):
            raise ValueError('Current measurement profile requires 640x480; do not resize silently')
        # This contract currently accepts PNG with validated dimensions before decode.
        import struct
        if len(encoded)<24 or encoded[:8]!=b'\x89PNG\r\n\x1a\n' or encoded[12:16]!=b'IHDR':
            raise ValueError('Expected a PNG frame')
        if struct.unpack('>II',encoded[16:24])!=(640,480):raise ValueError('PNG dimensions differ from frame metadata')
        bgr=cv2.imdecode(np.frombuffer(encoded,dtype=np.uint8),cv2.IMREAD_COLOR)
        if bgr is None or bgr.shape!=(480,640,3):raise ValueError('Invalid PNG pixels')
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        if hashlib.sha256(rgb.tobytes()).hexdigest()!=metadata['rgb_sha256']:raise ValueError('Decoded RGB hash differs')
        telemetry=metadata['telemetry']
        if expected_kind!='isaac' and telemetry is not None:
            raise ValueError('Ordinary frame input has no qualified phase/meter telemetry')
        if telemetry is not None:
            keys={'phase_key','metered_ml','dispense_complete','recipe_key'}
            if not isinstance(telemetry,dict) or set(telemetry)!=keys:raise ValueError('Invalid explicitly simulated telemetry')
            if (not isinstance(telemetry['recipe_key'],str) or telemetry['phase_key'] not in ('1','2','3')
                or type(telemetry['dispense_complete']) is not bool or len(telemetry['metered_ml'])!=3
                or any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in telemetry['metered_ml'])):
                raise ValueError('Invalid explicitly simulated telemetry')
        return cls(dict(metadata),rgb)

    @property
    def seconds(self):return float(self.metadata['pts']*Fraction(self.metadata['time_base']))


class FramePublisher:
    """Small connector interface for a future approved local camera/CV producer.

    Read the token from a local file; never place it in a URL or console output.
    A POST acknowledges a completed inference, providing bounded backpressure.
    """
    def __init__(self,url,token_file,generation=None):
        from urllib.parse import urlsplit
        parsed=urlsplit(url)
        if parsed.scheme!='http' or parsed.hostname not in ('127.0.0.1','localhost') or parsed.path not in ('','/'):
            raise ValueError('Only the configured local station endpoint is supported')
        self.url=url.rstrip('/');self.token_file=token_file;self.generation=generation

    def publish(self,metadata,png):
        from pathlib import Path
        token=json.loads(Path(self.token_file).read_text())['token']
        headers={'Content-Type':'image/png','X-Frame-Metadata':json.dumps(metadata,separators=(',',':')),'X-Frame-Token':token}
        if self.generation is not None:headers['X-Source-Generation']=str(self.generation)
        request=Request(self.url+'/api/frames',data=png,headers=headers,method='POST')
        with urlopen(request,timeout=180) as response:return json.load(response)
