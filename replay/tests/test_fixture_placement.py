"""Simulated placement context only; no team labels or real signal approvals."""
import unittest
from replay.operator.fixture_placement import PlacementTracker,assess,expected_positions


class FixturePlacementTests(unittest.TestCase):
    def predictions(self,fixture='G',count=0):
        return [dict(slot=f'{fixture}{i:02d}',proposed_class='pass' if i<=count else 'no_vial',
            mapping_confirmed=True,visibility='confirmed') for i in range(1,31)]

    def context(self,fixture='G',count=12):
        return dict(fixture=fixture,position_order=[f'{fixture}{i:02d}' for i in range(1,31)],placement_count=count,
            order_complete=False,order_id='SYNTHETIC-ORDER',reviewer='SYNTHETIC TEST ONLY',evidence_id='SYNTHETIC-CONTEXT')

    def test_trailing_empty_positions_do_not_warn(self):
        result=assess(self.predictions(count=12),self.context())
        self.assertEqual(result['review_requests'],0)
        self.assertEqual(result['targets'][12]['finding'],'empty_not_yet_expected')

    def test_hole_inside_expected_pattern_warns(self):
        predictions=self.predictions(count=12);predictions[4]['proposed_class']='no_vial'
        result=assess(predictions,self.context())
        self.assertEqual(result['review_requests'],1);self.assertEqual(result['targets'][4]['finding'],'missed_placement_candidate')

    def test_unknown_count_is_not_inferred_from_predictions(self):
        result=assess(self.predictions(count=12))
        self.assertIsNone(result['expected_count']);self.assertEqual(result['review_requests'],0)

    def test_complete_order_requires_all_positions(self):
        context=self.context(count=30);context['order_complete']=True
        result=assess(self.predictions(count=29),context)
        self.assertEqual(result['targets'][29]['finding'],'missed_placement_candidate')

    def test_contradictory_context_is_rejected(self):
        context=self.context();context['order_complete']=True
        with self.assertRaises(ValueError):expected_positions('G',context)

    def test_occluded_or_unreviewed_target_cannot_assert_missing_placement(self):
        for field,value in [('visibility','occluded'),('mapping_confirmed',False)]:
            predictions=self.predictions();predictions[0][field]=value
            result=assess(predictions,self.context(count=1))
            self.assertFalse(result['targets'][0]['review_requested']);self.assertEqual(result['targets'][0]['finding'],'unavailable')

    def test_defects_are_reported_in_both_fixtures_without_product_release(self):
        for fixture in ('G','S'):
            for label in ('missing_cap','crooked_cap','damaged_vial'):
                predictions=self.predictions(fixture,count=1);predictions[0]['proposed_class']=label
                result=assess(predictions,self.context(fixture,count=1))
                self.assertEqual(result['targets'][0]['finding'],'visible_defect_candidate')
                self.assertFalse(result['qualified_product_decision_available']);self.assertIsNone(result['hardware_command'])

    def test_supplied_fill_order_controls_expectation(self):
        context=self.context(count=1);context['position_order'].reverse()
        result=assess(self.predictions(),context)
        self.assertEqual(result['targets'][29]['finding'],'missed_placement_candidate')
        self.assertEqual(result['targets'][0]['finding'],'empty_not_yet_expected')

    def test_high_recall_warning_survives_pass_classification_without_flagging_trailing_empties(self):
        predictions=self.predictions(count=1)
        for row in predictions:row['occupied_defect_warning']=True
        result=assess(predictions,self.context(count=1))
        self.assertEqual(result['review_requests'],1)
        self.assertEqual(result['targets'][0]['finding'],'visible_defect_candidate')
        self.assertEqual(result['targets'][1]['finding'],'empty_not_yet_expected')

    def test_source_changes_and_rewinds_clear_context(self):
        for new_source,new_frame in [('OTHER',11),('TEST',9)]:
            tracker=PlacementTracker();tracker.set_context('G',self.context(count=1));tracker.ingest('TEST',10,self.predictions())
            result=tracker.ingest(new_source,new_frame,self.predictions())
            self.assertIsNone(result['expected_count']);self.assertEqual(result['review_requests'],0)

    def test_duplicate_or_cross_fixture_targets_are_rejected(self):
        predictions=self.predictions();predictions[-1]['slot']='S30'
        with self.assertRaises(ValueError):assess(predictions,self.context())

    def test_new_order_clears_old_completion_and_count_cannot_silently_rewind(self):
        tracker=PlacementTracker();tracker.set_context('G',self.context(count=30))
        with self.assertRaises(ValueError):tracker.set_context('G',self.context(count=0))
        tracker.new_order('G','NEXT-SYNTHETIC-ORDER')
        result=tracker.ingest('TEST',1,self.predictions())
        self.assertIsNone(result['expected_count']);self.assertEqual(result['review_requests'],0)


if __name__=='__main__':unittest.main()
