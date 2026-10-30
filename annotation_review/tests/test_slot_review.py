"""Synthetic in-memory fixtures only; no test decision is an eligible team label."""
import copy
import hashlib
import json
import unittest

from annotation_review.slot_review import score,validate_history,validate_packet


class SlotReviewTests(unittest.TestCase):
    def fixtures(self):
        task=dict(task_id='fixture/G01',slot='G01',source_sha256='test-source',box_xyxy=[1,2,49,50],draft={'proposed_class':'no_vial'})
        packet=dict(training_eligible=False,tasks=[task])
        packet['packet_id']=hashlib.sha256(json.dumps(packet,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        event=dict(seq=1,task_id=task['task_id'],slot=task['slot'],source_sha256=task['source_sha256'],box_xyxy=task['box_xyxy'],
            reviewer='TEST FIXTURE',timestamp='2026-10-23T00:00:00Z',label='uncertain',mapping='uncertain',visibility='uncertain')
        review=dict(schema='dataset1-slot-human-review/v1',packet_id=packet['packet_id'],training_eligible=False,
            demonstration=True,eligible_for_team_review=False,history=[event])
        return packet,review

    def test_tampered_packet_rejected(self):
        packet,_=self.fixtures();packet['tasks'][0]['draft']['proposed_class']='pass'
        with self.assertRaises(ValueError):validate_packet(packet)

    def test_demo_cannot_provide_accuracy_truth(self):
        packet,review=self.fixtures()
        with self.assertRaisesRegex(ValueError,'Demonstration'):score(packet,review)

    def test_changed_target_crop_rejected(self):
        packet,review=self.fixtures();review['history'][0]['box_xyxy']=[2,2,50,50]
        with self.assertRaises(ValueError):validate_history(packet,review)

    def test_source_mismatch_rejected(self):
        packet,review=self.fixtures();review['history'][0]['source_sha256']='different'
        with self.assertRaises(ValueError):validate_history(packet,review)

    def test_occlusion_cannot_be_definite_empty(self):
        packet,review=self.fixtures();review['history'][0].update(visibility='occluded',label='no_vial')
        with self.assertRaises(ValueError):validate_history(packet,review)

    def test_unreviewed_is_unavailable_not_correct(self):
        packet,review=self.fixtures();review.update(history=[],demonstration=False,eligible_for_team_review=True)
        metrics=score(packet,review)
        self.assertIsNone(metrics['accuracy']);self.assertEqual(metrics['scored_targets'],0)
        self.assertEqual(metrics['excluded']['unreviewed'],1)

    def test_both_fixtures_and_supported_defect_classes_are_scored(self):
        packet,review=self.fixtures();template=packet['tasks'][0];tasks=[];history=[]
        classes=['pass','missing_cap','crooked_cap','damaged_vial','no_vial']
        for prefix in ('G','S'):
            for i in range(1,31):
                slot=f'{prefix}{i:02d}';label='crooked_cap' if i==1 else 'pass'
                task=dict(template,task_id='TEST/'+slot,slot=slot,draft={'proposed_class':label});tasks.append(task)
                history.append(dict(seq=len(history)+1,task_id=task['task_id'],slot=slot,source_sha256=task['source_sha256'],box_xyxy=task['box_xyxy'],
                    reviewer='SYNTHETIC TEST',timestamp='2026-10-23T00:00:00Z',label=label,mapping='confirmed',visibility='visible'))
        packet.update(tasks=tasks,model_classes=classes);packet.pop('packet_id')
        packet['packet_id']=hashlib.sha256(json.dumps(packet,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        review.update(packet_id=packet['packet_id'],history=history,demonstration=False,eligible_for_team_review=True)
        result=score(packet,review)
        self.assertEqual(result['unsupported_reviewed_defects'],0)
        self.assertEqual(result['fully_scored_30_slot_scenes'],2)
        self.assertEqual(result['fully_scored_60_slot_scenes'],1)
        self.assertEqual(result['all_60_correct_scenes'],1)

    def test_wrong_defect_name_is_not_automatically_an_escaped_defect(self):
        packet,review=self.fixtures()
        packet['tasks'][0]['draft'].update(proposed_class='missing_cap',occupied_defect_warning=True,defect_score=.8)
        packet['model_classes']=['pass','missing_cap','crooked_cap','damaged_vial','no_vial'];packet.pop('packet_id')
        packet['packet_id']=hashlib.sha256(json.dumps(packet,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        review.update(packet_id=packet['packet_id'],demonstration=False,eligible_for_team_review=True)
        review['history'][0].update(label='crooked_cap',mapping='confirmed',visibility='visible')
        result=score(packet,review)
        self.assertEqual(result['per_class']['crooked_cap']['missed'],1)
        self.assertEqual(result['occupied_defect_diagnostic']['per_class']['crooked_cap']['escaped_defects'],0)


if __name__=='__main__':unittest.main()
