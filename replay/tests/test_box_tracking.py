from copy import deepcopy
import unittest

import numpy as np

from replay.box_tracking import StationTracker, TrackingSettings


class StationChecks(unittest.TestCase):
    def setUp(self):
        self.rgb = np.full((480,640,3),100,np.uint8)
        self.vial = dict(class_id=0,box=[274,57,340,420],confidence=.9)
        self.liquid = dict(class_id=1,box=[283,382,337,417],confidence=.8)
        self.proposals = [self.vial,self.liquid]

    def stable(self, tracker, start=0):
        return [tracker.update(self.rgb,self.proposals,f) for f in range(start,start+18,6)][-1]

    def test_confirmation_raw_pixels_and_no_mutation(self):
        t=StationTracker();original=deepcopy(self.proposals)
        first=t.update(self.rgb,self.proposals,0)
        self.assertEqual(first['liquids'][0]['association'],'track_not_confirmed')
        t.update(self.rgb,self.proposals,6);r=t.update(self.rgb,self.proposals,12)
        self.assertEqual(r['liquids'][0]['association'],'candidate_link')
        self.assertEqual(r['detector_proposals'],original);self.assertEqual(self.proposals,original)
        self.assertFalse(r['liquids'][0]['measurement_authorized'])

    def test_missing_detection_retains_only_identity_not_observation(self):
        t=StationTracker();old=self.stable(t)['vials'][0]['track_id']
        r=t.update(self.rgb,[self.liquid],18)
        self.assertEqual(r['vials'],[]);self.assertIsNone(r['liquids'][0]['vial_track_id'])
        r=t.update(self.rgb,self.proposals,24)
        self.assertEqual(r['vials'][0]['track_id'],old)
        self.assertEqual(r['liquids'][0]['association'],'track_not_confirmed')
        for f in (30,36,42,48):t.update(self.rgb,[],f)
        r=t.update(self.rgb,self.proposals,54)
        self.assertNotEqual(r['vials'][0]['track_id'],old)

    def test_same_position_replacement_is_split_after_motion_and_settle(self):
        t=StationTracker();old=self.stable(t)['vials'][0]['track_id']
        for f,value in ((18,200),(24,100),(30,200)):
            r=t.update(np.full_like(self.rgb,value),self.proposals,f)
            self.assertIsNone(r['liquids'][0]['vial_track_id'])
        self.assertFalse(any(e['kind']=='carrier_transition_confirmed' for e in t.update(self.rgb,self.proposals,36)['events']))
        for f in (42,48,54,60,66):r=t.update(self.rgb,self.proposals,f)
        self.assertNotEqual(r['vials'][0]['track_id'],old)
        self.assertEqual(r['liquids'][0]['association'],'candidate_link')

    def test_ambiguous_liquid_vials_and_containment_fail_closed(self):
        t=StationTracker();self.stable(t)
        r=t.update(self.rgb,[*self.proposals,deepcopy(self.liquid)],18)
        self.assertTrue(all(l['association']=='multiple_liquid_candidates' for l in r['liquids']))
        r=t.update(self.rgb,[*self.proposals,deepcopy(self.vial)],24)
        self.assertEqual(r['liquids'][0]['association'],'ambiguous_active_vial')
        r=self.stable(StationTracker())
        self.assertEqual(r['liquids'][0]['association'],'candidate_link')
        t=StationTracker();self.stable(t)
        wide=dict(self.liquid,box=[250,382,360,417]);r=t.update(self.rgb,[self.vial,wide],18)
        self.assertEqual(r['liquids'][0]['association'],'outside_vial')
        self.assertEqual(r['liquids'][0]['box'],wide['box'])

    def test_single_motion_spike_blocks_links_without_false_identity_timeout(self):
        t=StationTracker();old=self.stable(t)['vials'][0]['track_id']
        # One changed sample followed by its same image: only one motion spike.
        lighter=np.full_like(self.rgb,120)
        r=t.update(lighter,self.proposals,18)
        self.assertIsNone(r['liquids'][0]['vial_track_id'])
        events=[]
        for f in (24,30,36,42,48):
            r=t.update(lighter,self.proposals,f);events.extend(r['events'])
        self.assertEqual(r['vials'][0]['track_id'],old)
        self.assertEqual(r['liquids'][0]['association'],'candidate_link')
        self.assertFalse(any(e['kind']=='station_track_retired' for e in events))

    def test_neighbor_vial_does_not_supply_station_liquid(self):
        t=StationTracker();self.stable(t)
        neighbor=dict(self.vial,box=[360,57,420,420])
        neighbor_liquid=dict(self.liquid,box=[368,382,415,417])
        r=t.update(self.rgb,[neighbor,neighbor_liquid],18)
        self.assertEqual(r['vials'][0]['track_status'],'off_station')
        self.assertEqual(r['liquids'][0]['association'],'no_fresh_station_vial')
        self.assertIsNone(r['liquids'][0]['vial_track_id'])

    def test_gap_restart_monotonicity_and_session_identity(self):
        t=StationTracker(session_id='a');old=self.stable(t)['vials'][0]['track_id']
        r=t.update(self.rgb,self.proposals,30)
        self.assertTrue(r['input_gap']);self.assertNotEqual(r['vials'][0]['track_id'],old)
        self.assertIsNone(r['liquids'][0]['vial_track_id'])
        with self.assertRaises(ValueError):t.update(self.rgb,self.proposals,30)
        restarted=StationTracker(session_id='b');r=self.stable(restarted)
        self.assertNotEqual(r['vials'][0]['track_id'],old)
        self.assertNotEqual(self.stable(StationTracker())['vials'][0]['track_id'],
                            self.stable(StationTracker())['vials'][0]['track_id'])

    def test_prefix_causality_and_no_liquid_history(self):
        def run(last,changed=False):
            t=StationTracker(session_id='prefix-test');out=[]
            for f in range(0,last,6):
                rgb=self.rgb if not changed or f<36 else np.full_like(self.rgb,230)
                out.append(t.update(rgb,self.proposals,f))
            return out
        self.assertEqual(run(30),run(90)[:5]);self.assertEqual(run(30),run(90,True)[:5])
        with self.assertRaises(ValueError):TrackingSettings(containment_minimum=.8)


if __name__=='__main__':unittest.main()
