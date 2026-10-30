"""Mass balance, action effects and source/truth boundary tests."""
import copy
import socket
import unittest

from twin_contract import SharedPath, default_plan, episode_states, validate_frame, digest, send, receive, FRAME_BYTES


class TwinContractTests(unittest.TestCase):
    def test_nonfinite_inventory_and_corrupt_clock_are_rejected(self):
        for capacity,fraction in [(float('nan'),(1.,0.,0.)),(2.,(float('nan'),0.,1.)),(-1.,(1.,0.,0.))]:
            with self.assertRaises(ValueError):SharedPath(capacity,fraction)
        pixels=bytes(FRAME_BYTES)
        h=dict(schema_version=1,type='frame',session_id='s',episode_id='e',frame_index=1,
            sim_time_s=.3,phase_key='1',recipe_key='r',metered_ml=[0.,0.,0.],
            dispense_complete=False,rgb_sha256=digest(pixels),shape=[480,640,3])
        with self.assertRaises(ValueError):validate_frame(h,pixels)

    def test_mass_conservation_across_recipe_and_flush(self):
        path=SharedPath(2., (0.,1.,0.))
        initial=list(path.inventory); delivered=[0.,0.,0.]; inlet=[0.,0.,0.]
        for tank, volume in [(0,.01),(0,8.),(1,4.),(2,10.)]:
            result=path.dispense(tank,volume); inlet[tank]+=volume
            delivered=[a+b for a,b in zip(delivered,result)]
            self.assertAlmostEqual(sum(result),volume)
        for i in range(3):self.assertAlmostEqual(initial[i]+inlet[i],path.inventory[i]+delivered[i])
        before=list(path.inventory); receipt=path.flush(0,8.)
        self.assertLess(receipt['residual_components_ml'][1],before[1])
        self.assertGreater(receipt['residual_components_ml'][1],0.)

    def test_intervention_changes_actual_state_not_meter_or_recipe(self):
        p=default_plan(); a=list(episode_states(p,p['episodes'][2]));b=list(episode_states(p,p['episodes'][3]))
        self.assertEqual([s['metered_ml'] for s in a],[s['metered_ml'] for s in b])
        self.assertNotEqual(a[-1]['actual_components_ml'],b[-1]['actual_components_ml'])
        for s in a:self.assertAlmostEqual(sum(s['actual_components_ml']),sum(s['metered_ml']))

    def test_transport_and_hidden_truth_rejection(self):
        pixels=bytes(FRAME_BYTES)
        h=dict(schema_version=1,type='frame',session_id='s',episode_id='e',frame_index=0,
            sim_time_s=0.,phase_key='1',recipe_key='r',metered_ml=[0.,0.,0.],
            dispense_complete=False,rgb_sha256=digest(pixels),shape=[480,640,3])
        validate_frame(h,pixels)
        corrupt=dict(h,actual_components_ml=[1.,0.,0.])
        with self.assertRaises(ValueError):validate_frame(corrupt,pixels)
        with self.assertRaises(ValueError):validate_frame(h,pixels,previous=h)
        with self.assertRaises(ValueError):validate_frame(dict(h,frame_index=2,sim_time_s=.4),pixels,previous=h)
        with self.assertRaises(ValueError):validate_frame(h,pixels[:-1])
        left,right=socket.socketpair()
        with left,right:
            send(left,{'type':'ack','frame_index':0})
            self.assertEqual(receive(right),({'type':'ack','frame_index':0},b''))


if __name__=='__main__':unittest.main()
