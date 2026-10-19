from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import cv2

from replay.config import Config
from replay.geometry import VialGeometry,encode_mask,decode_mask
from replay.measurement import Extractor
from replay.proposals import CachedImageDetector
from replay.segmentation import SamplingSegmenter,crop_input,INPUT_SIZE
from replay.tests.test_geometry import FIXTURE_BOXES


class StubSegmenter:
    def __init__(self,mask,box=None):
        self.mask,self.box=mask,box or FIXTURE_BOXES[1]['box']
    def propose(self,rgb,vial):
        return [] if self.mask is None else [dict(class_id=1,box=self.box,confidence=.9,candidate_mask=encode_mask(self.mask))]


class SegmentationChecks(unittest.TestCase):
    def test_semantic_fragments_are_not_instances_and_unsafe_pixels_stay_excluded(self):
        class Net:
            def setInput(self,value): pass
            def forward(self):
                logits=np.full((1,1,384,128),-10,np.float32)
                logits[:,:,170:185,45:85]=10  # upper label, outside physical domain
                logits[:,:,275:330,40:90]=10  # liquid sampling interior
                return logits
        segmenter=SamplingSegmenter.__new__(SamplingSegmenter);segmenter.net=Net()
        rgb=np.full((480,640,3),150,np.uint8);rgb[231:253,264:338]=20
        rgb[300:390,288:330]=(40,120,70)
        proposals=segmenter.propose(rgb,FIXTURE_BOXES[0]['box'])
        self.assertEqual(len(proposals),1)
        self.assertEqual(len(proposals[0]['raw_components']),2)
        extractor=Extractor(Config());extractor.geometry.segmenter=segmenter
        with patch.object(VialGeometry,'detect',return_value=FIXTURE_BOXES):
            for _ in range(3): row=extractor.measure(rgb)
        self.assertTrue(row['measurement_valid'])
        raw=decode_mask(proposals[0]['candidate_mask']).astype(bool)
        final=decode_mask(row['geometry']['mask']).astype(bool)
        self.assertFalse(np.any(final & ~raw))
        self.assertFalse(final[:int(65+.58*365)].any())
        stages=row['geometry']['mask_stages']
        previous=raw
        for name in ('physical_interior','hsv','opened','largest_component','eroded'):
            current=decode_mask(stages[name]).astype(bool)
            self.assertFalse(np.any(current & ~previous),name)
            previous=current
        self.assertFalse(np.any(final & ~previous))
        # Semantic correction cannot rescue a model that only predicts the base.
        base=np.zeros((480,640),np.uint8);base[410:423,315:330]=1
        extractor.geometry.segmenter=StubSegmenter(base,[315,410,330,423])
        with patch.object(VialGeometry,'detect',return_value=FIXTURE_BOXES):
            bad=extractor.measure(rgb)
        self.assertFalse(bad['measurement_valid'])
        self.assertIn('liquid_interior_too_small',bad['quality_flags'])

    def test_projection_matches_explicit_resize_at_offset_and_clipped_edges(self):
        logits=np.full((384,128),-4,np.float32);logits[130:350,30:95]=4
        class Net:
            def setInput(self,value): pass
            def forward(self): return logits[None,None]
        model=SamplingSegmenter.__new__(SamplingSegmenter);model.net=Net()
        rgb=np.full((480,640,3),100,np.uint8)
        for vial in ([278,65,340,430],[0,0,30,200],[610,300,640,480]):
            _,box=crop_input(rgb,vial);x0,y0,x1,y1=box
            expected=np.zeros((480,640),np.uint8)
            expected[y0:y1,x0:x1]=(cv2.resize(logits,(x1-x0,y1-y0),interpolation=cv2.INTER_LINEAR)>0)
            np.testing.assert_array_equal(decode_mask(model.propose(rgb,vial)[0]['candidate_mask']),expected)

    def test_correction_preserves_instance_ambiguity_and_coherent_area_floor(self):
        rgb=np.full((480,640,3),150,np.uint8);rgb[231:253,264:338]=20
        rgb[300:390,288:330]=(40,120,70)
        mask=np.zeros((480,640),np.uint8)
        mask[310:325,290:300]=1;mask[365:380,315:325]=1
        extractor=Extractor(Config());extractor.geometry.segmenter=StubSegmenter(mask,[290,310,325,380])
        with patch.object(VialGeometry,'detect',return_value=FIXTURE_BOXES):
            for _ in range(3): row=extractor.measure(rgb)
        self.assertFalse(row['measurement_valid'])
        self.assertIn('insufficient_coherent_colored_liquid',row['quality_flags'])
        # Separate detector instances are still ambiguous: no blanket union of
        # boxes or independently proposed objects, only a vial-conditioned map.
        geometry=VialGeometry(Config())
        boxes=FIXTURE_BOXES+[dict(class_id=1,box=[290,310,325,330],confidence=.9)]
        with patch.object(geometry,'detect',return_value=boxes): row=geometry.update(rgb)
        self.assertIn('liquid_missing_or_ambiguous_inside_vial',row['quality_flags'])

    def test_segmentation_cannot_fall_back_to_old_liquid_box_or_bypass_qa(self):
        rgb=np.full((480,640,3),150,np.uint8);rgb[231:253,264:338]=20;rgb[340:390,288:330]=(40,120,70)
        mask=np.zeros((480,640),np.uint8);mask[340:390,288:330]=1
        extractor=Extractor(Config());extractor.geometry.segmenter=StubSegmenter(mask)
        with patch.object(VialGeometry,'detect',return_value=FIXTURE_BOXES):
            for _ in range(3): row=extractor.measure(rgb)
            self.assertTrue(row['measurement_valid'])
            self.assertFalse(np.any(decode_mask(row['geometry']['mask']) & ~mask.astype(bool)))
            extractor.geometry.segmenter=StubSegmenter(None)
            missing=extractor.measure(rgb)
            self.assertFalse(missing['measurement_valid'])
            self.assertIsNone(missing['lab'])
            self.assertIn('liquid_missing_or_ambiguous_inside_vial',missing['quality_flags'])
            extractor.geometry.segmenter=StubSegmenter(mask)
            neutral=rgb.copy();neutral[340:390,288:330]=80
            rejected=extractor.measure(neutral)
            self.assertFalse(rejected['measurement_valid'])
            self.assertEqual(rejected['geometry']['mask_pixels'],0)
            extractor.geometry.segmenter=StubSegmenter(mask,[400,340,440,390])
            rejected=extractor.measure(rgb)
            self.assertFalse(rejected['measurement_valid'])
            self.assertIsNone(rejected['geometry']['mask'])

    def test_crop_and_binary_model_output_map_back_to_source_pixels(self):
        rgb=np.full((480,640,3),[200,80,30],np.uint8)
        image,region=crop_input(rgb,[278,65,340,430])
        self.assertEqual(image.shape,(INPUT_SIZE[1],INPUT_SIZE[0],3))
        class Net:
            def setInput(self,value):
                np.testing.assert_allclose(value[0,:,0,0],[200/255,80/255,30/255],atol=1e-7)
            def forward(self):
                logits=np.full((1,1,384,128),-10,np.float32);logits[:,:,240:340,40:90]=10
                return logits
        segmenter=SamplingSegmenter.__new__(SamplingSegmenter);segmenter.net=Net()
        proposals=segmenter.propose(rgb,[278,65,340,430])
        self.assertEqual(len(proposals),1)
        mask=decode_mask(proposals[0]['candidate_mask'])
        x0,y0,x1,y1=region
        self.assertFalse(mask[:y0].any());self.assertFalse(mask[y1:].any())
        self.assertFalse(mask[:,:x0].any());self.assertFalse(mask[:,x1:].any())
        self.assertGreater(mask.sum(),60)
        _,edge=crop_input(rgb,[0,0,30,200]);self.assertEqual(edge[:2],[0,0])

    def test_model_pairing_and_hash_are_required(self):
        for changes in (dict(liquid_segmentation_model_path='x'),dict(image_proposal_cache_path='x'),
                        dict(liquid_segmentation_model_path='x',liquid_segmentation_model_sha256='1'*64)):
            with self.assertRaises(ValueError): Config(**changes)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model';path.write_bytes(b'not the pinned model')
            with self.assertRaises(ValueError): SamplingSegmenter(path,'1'*64)

    def test_cached_proposals_bind_recording_model_frame_and_exact_pixels(self):
        rgb=np.full((480,640,3),50,np.uint8);model='1'*64
        payload=dict(manifest=dict(run_id=5,video_sha256='video',geometry_model_sha256=model,config={}),
                     features=[dict(frame_index=0,source_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                                    geometry=dict(model_sha256=model,proposals=FIXTURE_BOXES),
                                    lab=['poisoned'],fill_detection={'vial_id':'evaluation-poison'})],annotations='ignored')
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'cache.json';path.write_text(json.dumps(payload));sha=hashlib.sha256(path.read_bytes()).hexdigest()
            cache=CachedImageDetector(path,sha,model,'video',5)
            cache.prepare_frame(0);self.assertEqual(cache.detect(rgb),FIXTURE_BOXES)
            with self.assertRaises(ValueError): cache.detect(rgb)
            with self.assertRaises(ValueError): cache.prepare_frame(6)
            cache.prepare_frame(0)
            with self.assertRaises(ValueError): cache.detect(rgb+1)
            for args in ((path,'0'*64,model,'video',5),(path,sha,model,'different',5),(path,sha,'2'*64,'video',5)):
                with self.assertRaises(ValueError): CachedImageDetector(*args)


if __name__=='__main__': unittest.main()
