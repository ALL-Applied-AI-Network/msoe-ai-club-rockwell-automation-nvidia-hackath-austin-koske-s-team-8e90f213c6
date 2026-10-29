import unittest

from review_box_tracking import VialTracker,associate_liquid


class BoxTrackingChecks(unittest.TestCase):
    def test_liquid_association_does_not_choose_between_ambiguous_vials(self):
        liquid=dict(box=[20,40,60,90])
        vial=dict(box=[10,10,70,100],track_id=7)
        self.assertEqual(associate_liquid(liquid,[vial])['vial_track_id'],7)
        self.assertEqual(associate_liquid(liquid,[vial,dict(vial,track_id=8)])['association'],'multiple_vials')
        self.assertIsNone(associate_liquid(liquid,[])['vial_track_id'])
        self.assertEqual(associate_liquid(dict(box=[200,200,230,250]),[vial])['association'],'no_vial')

    def test_current_boxes_are_not_replaced_with_predictions_or_stale_tracks(self):
        tracker=VialTracker(stride=6)
        vial=dict(class_id=0,box=[10,10,70,100],confidence=.9)
        liquid=dict(class_id=1,box=[20,40,60,90],confidence=.9)
        vials,liquids=tracker.update([vial,liquid],(480,640))
        identity=vials[0]['track_id'];self.assertIsNotNone(identity)
        self.assertEqual(liquids[0]['vial_track_id'],identity)
        shifted=dict(vial,box=[11,10,71,100])
        vials,_=tracker.update([shifted,liquid],(480,640))
        self.assertEqual(vials[0]['box'],shifted['box'])
        self.assertEqual(vials[0]['track_id'],identity)
        missing,orphaned=tracker.update([liquid],(480,640))
        self.assertEqual(missing,[])
        self.assertEqual(orphaned[0]['association'],'no_vial')


if __name__=='__main__':unittest.main()
