"""Process continuity and the human gate; no fabricated model accuracy tests."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import uuid

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'setup/vision'))
from twin_contract import SharedPath, default_plan, episode_states
from replay.operator.twin_process import fill_states
from replay.operator.twin import TwinSession
from replay.operator.store import Conflict


class OperatorTwinChecks(unittest.TestCase):
    def test_fill_matches_existing_schedule_and_continues_real_inventory(self):
        plan=default_plan();path=SharedPath(2.,plan['prior_fraction'])
        actual=list(fill_states(plan,path))
        expected=list(episode_states(plan,plan['episodes'][2]))
        self.assertEqual(len(actual),37)
        for a,b in zip(actual,expected):
            for key in ('actual_components_ml','metered_ml','path_components_ml','dispense_complete'):
                self.assertEqual(a[key],b[key])
        before=list(path.inventory);receipt=path.flush(0,8.)
        for i in range(3):
            self.assertAlmostEqual(before[i]+(8 if i==0 else 0),receipt['residual_components_ml'][i]+receipt['discharged_components_ml'][i])
        after=list(path.inventory);recovery=list(fill_states(plan,path))
        self.assertEqual(recovery[0]['path_components_ml'],after)
        self.assertNotEqual(after,plan['initial_fraction'])
        self.assertEqual(recovery[0]['actual_components_ml'],[0.,0.,0.])
        for i in range(3):
            self.assertAlmostEqual(after[i]+recovery[-1]['metered_ml'][i],
                                   recovery[-1]['actual_components_ml'][i]+path.inventory[i])

    def session(self,temp):
        session=TwinSession(Path(temp)/'run');session.connected=True;session.step='ready_reference'
        self.addCleanup(session.journal.close);return session

    def command(self,session,name,**extra):
        return dict(command=name,actor='Test operator',request_id=uuid.uuid4().hex,
                    expected_step=session.step,review_id=session.review_id,**extra)

    def test_workflow_blocks_intervention_before_ack_and_receipt(self):
        temp = self.enterContext(tempfile.TemporaryDirectory())
        s=self.session(temp)
        with self.assertRaises(Conflict):s.command(self.command(s,'flush',note='test',confirmed=True))
        s.command(self.command(s,'reference'));self.assertEqual(s.commands.get()['role'],'reference')
        with self.assertRaises(Conflict):s.command(self.command(s,'test'))
        s.step='awaiting_response';s.review='requested';s.review_id='review-a'
        with self.assertRaises(Conflict):s.command(self.command(s,'flush',note='test',confirmed=True))
        with self.assertRaises(Conflict):s.command(self.command(s,'acknowledge')|{'review_id':'old-review'})
        s.command(self.command(s,'acknowledge'));self.assertEqual(s.review,'acknowledged')
        with self.assertRaises(ValueError):s.command(self.command(s,'flush',note='test',confirmed=False))
        s.command(self.command(s,'flush',note='Only simulate the requested intervention',confirmed=True))
        self.assertEqual(s.step,'flushing');self.assertEqual(s.commands.get()['type'],'flush')
        self.assertFalse(any(e['kind']=='simulated_flush_completed' for e in s.events))
        with self.assertRaises(Conflict):s.command(self.command(s,'recovery'))
        s.step='ready_recovery';s.command(self.command(s,'recovery'));s.commands.get()
        s.step='recovery_complete';s.command(self.command(s,'close',note='No physical quality outcome verified',confirmed=True))
        self.assertEqual(s.review,'closed')
        self.assertFalse(s.events[-1]['payload']['product_release_authorized'])
        rows=[json.loads(line) for line in (s.out/'operator-events.jsonl').read_text().splitlines()]
        self.assertEqual(rows,s.events)

    def test_duplicate_request_has_one_effect_and_reset_preserves_history(self):
        temp = self.enterContext(tempfile.TemporaryDirectory())
        s=self.session(temp);body=self.command(s,'reference')
        s.command(body);self.assertTrue(s.command(body)['duplicate']);self.assertEqual(s.commands.qsize(),1)
        with self.assertRaises(Conflict):s.command(body|{'actor':'different'})
        s.step='complete';s.review='closed';count=len(s.events)
        s.command(self.command(s,'reset'));self.assertEqual(s.cycle,2);self.assertEqual(s.step,'ready_reference')
        self.assertGreater(len(s.events),count)
        state=s.snapshot();self.assertEqual(state['hardware_commands_sent'],0)
        self.assertEqual(state['operational_recipe_assessment'],'unavailable')
        state['events'].clear();self.assertTrue(s.events)

    def test_disconnected_or_failed_simulation_never_accepts_controls(self):
        temp = self.enterContext(tempfile.TemporaryDirectory())
        s=self.session(temp);s.connected=False
        with self.assertRaises(Conflict):s.command(self.command(s,'reference'))
        s.connected=True;s.error='Lost renderer'
        with self.assertRaises(Conflict):s.command(self.command(s,'reference'))


if __name__=='__main__':unittest.main()
