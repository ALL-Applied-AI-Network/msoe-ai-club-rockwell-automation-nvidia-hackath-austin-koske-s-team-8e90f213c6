"""Toy metric fixtures; no real annotations or approved labels are created."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from annotation_review.score_sam_temporal_review import score


@unittest.skipUnless(importlib.util.find_spec('numpy') and importlib.util.find_spec('PIL'), 'CPU scoring extras required')
class ComparedMaskScoringTests(unittest.TestCase):
    def fixture(self,root):
        import numpy as np
        from PIL import Image
        comparison=root/'comparison';probe=root/'probe';packet=root/'packet'
        for p in [comparison,probe,packet]:p.mkdir()
        def write(p,v):p.write_text(json.dumps(v))
        def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
        arms=['legacy_sam','synthetic_sam','temporal'];tasks=[];history=[]
        for i,status in enumerate(['empty','pool']):
            image=probe/f'source{i}.png';Image.new('RGB',(640,480),'black').save(image)
            task=dict(sample_id=f'toy{i}',case_id='toy',frame_index=i,pts=i,time_base='1/30',
                image=image.name,image_sha256=sha(image),source_rgb_sha256='toy-rgb',source_video_sha256='toy-video',drafts=[])
            for arm in arms:
                mask=np.zeros((480,640),np.uint8)
                if arm=='legacy_sam':mask[0,0]=255  # One true positive and one absent-frame alarm.
                if arm=='temporal' and i==1:mask[0,1]=255  # Present, but wrong boundary.
                path=comparison/f'{arm}-{i}.png';Image.fromarray(mask).save(path)
                task['drafts'].append(dict(mode=arm,model_sha256='toy-model',mask=path.name,mask_sha256=sha(path),
                    presence_probability=None,proposal_available=bool(mask.any())))
            tasks.append(task)
            keys=['sample_id','image_sha256','source_rgb_sha256','source_video_sha256','frame_index','pts','time_base']
            history.append(dict(kind='frame',seq=i+1,reviewer='Unit fixture only',source={k:task[k] for k in keys},status=status,
                notes='Toy fixture, not a team annotation.',draft_model_sha256=None,training_eligible=False,
                mask=dict(width=640,height=480,counts=[307200] if i==0 else [0,1,307199])))
        write(comparison/'plan.json',dict(arms=arms,source_video_sha256='toy-video'))
        write(comparison/'completed.json',dict(artifacts={}))
        write(comparison/'review-drafts.json',dict(tasks=tasks))
        (comparison/'pass-1.jsonl').write_text('')
        write(packet/'manifest.json',dict(packet_id='toy-packet'))
        review=root/'review.json';write(review,dict(schema='visible-pool-human-review/v1',packet_id='toy-packet',
            source_video_sha256='toy-video',source_run=4,demonstration=False,training_eligible=False,team_approved=False,history=history))
        return SimpleNamespace(comparison=comparison,source_probe=probe,packet=packet,review=review,out=root/'result')

    def test_missing_masks_count_as_missed_and_presence_is_separate_from_iou(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=self.fixture(Path(tmp))
            with contextlib.redirect_stdout(io.StringIO()):score(a)
            result=json.loads((a.out/'comparison.json').read_text())
            self.assertEqual(result['scored'],2)
            self.assertEqual(result['models']['legacy_sam']['proposal_precision'],.5)
            self.assertEqual(result['models']['synthetic_sam']['fn'],1)
            self.assertEqual(result['models']['synthetic_sam']['mean_positive_mask_iou'],0.)
            self.assertEqual(result['models']['temporal']['proposal_recall'],1.)
            self.assertEqual(result['models']['temporal']['mean_positive_mask_iou'],0.)
            self.assertFalse(result['team_approved'])

    def test_demonstrations_cannot_be_scored_as_real_truth(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=self.fixture(Path(tmp));v=json.loads(a.review.read_text());v['demonstration']=True;a.review.write_text(json.dumps(v))
            with self.assertRaisesRegex(ValueError,'non-demonstration'):score(a)
            self.assertFalse(a.out.exists())

    def test_optional_refinement_uses_same_labels_and_source_clock(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=self.fixture(Path(tmp));a.refinement=Path(tmp)/'refinement';a.refinement.mkdir()
            p=a.refinement/'plan.json';p.write_text(json.dumps(dict(parent_plan_sha256=hashlib.sha256((a.comparison/'plan.json').read_bytes()).hexdigest())))
            (a.refinement/'completed.json').write_text(json.dumps(dict(artifacts={})))
            rows=[dict(source_frame=i,source_rgb_sha256='toy-rgb',pts=i,time_base='1/30',proposal_available=bool(i),
                raw_union=None if not i else dict(origin=[0,0],size=[1,1],runs=[[0,1]])) for i in range(2)]
            (a.refinement/'pass-1.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            (a.comparison/'pass-1.jsonl').write_text('\n'.join(json.dumps(dict(arms=dict(temporal=dict(source_frame=i)))) for i in range(2)))
            with contextlib.redirect_stdout(io.StringIO()):score(a)
            result=json.loads((a.out/'comparison.json').read_text())['models']['temporal_guided_sam']
            self.assertEqual(result['mean_positive_mask_iou'],1.)
            self.assertEqual(result['tp'],1);self.assertEqual(result['tn'],1)
