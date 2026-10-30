"""Source evidence must reject pixel or clock differences before writing a PNG."""
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from materialize_station_evidence import decode_original


class MaterializationTests(unittest.TestCase):
    def test_changed_pixels_or_clock_are_not_substituted(self):
        pixels=np.zeros((2,2,3),np.uint8)
        for mismatch in ('pixels','clock'):
            with self.subTest(mismatch=mismatch),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);(root/'fill').mkdir()
                frame=SimpleNamespace(pts=1,time_base='1/30',to_ndarray=lambda **kwargs:pixels)
                container=SimpleNamespace(streams=SimpleNamespace(video=[None]),decode=lambda stream:iter([frame]))
                source=dict(pts=2 if mismatch=='clock' else 1,time_base='1/30',
                    decoded_rgb_sha256='f'*64 if mismatch=='pixels' else hashlib.sha256(pixels.tobytes()).hexdigest())
                wanted={0:dict(observation_id='TEST',source=source)};assets={}
                with self.assertRaisesRegex(ValueError,'no substitution permitted'):
                    decode_original(container,wanted,root,assets)
                self.assertEqual(assets,{})
                self.assertEqual(list((root/'fill').iterdir()),[])


if __name__=='__main__':unittest.main()
