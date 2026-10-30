"""Diagnostic color reference cannot bypass missing QA or phase boundaries."""
import unittest

from live_twin_observer import ProvisionalComparison


class ProvisionalComparisonTests(unittest.TestCase):
    def sample(self,phase='1',valid=True,lab=(50.,0.,0.)):
        return dict(phase_key=phase,metered_ml=[5.,0.,0.]),dict(measurement=dict(measurement_valid=valid,lab=list(lab)))

    def test_missing_reference_and_failed_qa_stay_unavailable(self):
        c=ProvisionalComparison();h,r=self.sample()
        self.assertEqual(c.update(h,r,False)['status'],'unavailable')
        c.update(h,r,True)
        h,bad=self.sample(valid=False)
        self.assertEqual(c.update(h,bad,False)['status'],'unavailable')
        self.assertFalse(c.warned)

    def test_three_qualified_deviations_and_phase_reset(self):
        c=ProvisionalComparison()
        for phase in ['1','2']:
            h,r=self.sample(phase);c.update(h,r,True)
        c.reset()
        h,r=self.sample(lab=(90.,0.,0.))
        self.assertFalse(c.update(h,r,False)['warning'])
        self.assertFalse(c.update(h,r,False)['warning'])
        h,r=self.sample('2',lab=(90.,0.,0.))
        self.assertFalse(c.update(h,r,False)['warning'])
        self.assertFalse(c.update(h,r,False)['warning'])
        self.assertTrue(c.update(h,r,False)['warning'])
        c.reset();self.assertFalse(c.warned)
        self.assertEqual(len(c.reference),2)

    def test_missing_sample_breaks_consecutive_evidence(self):
        c=ProvisionalComparison();h,r=self.sample();c.update(h,r,True);c.reset()
        h,changed=self.sample(lab=(90.,0.,0.))
        c.update(h,changed,False);c.update(h,changed,False)
        h,bad=self.sample(valid=False);c.update(h,bad,False)
        h,changed=self.sample(lab=(90.,0.,0.))
        self.assertFalse(c.update(h,changed,False)['warning'])


if __name__=='__main__':unittest.main()
