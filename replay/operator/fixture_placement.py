"""Opt-in final-fixture placement assessment. No sensors, actuation or app changes.

Predicted occupancy never supplies its own expected placement count. Fixture
numbering/order must be supplied and attributed; a geometric proposal is not an
approved map. No filling-vial identity is needed for standalone slot inspection.
"""
from copy import deepcopy

CLASSES={'pass','missing_cap','crooked_cap','damaged_vial','no_vial'}
DEFECTS=CLASSES-{'pass','no_vial'}


def expected_positions(fixture,context):
    if fixture not in ('G','S'):raise ValueError('Unknown fixture')
    if not context:return None,['placement_context_unavailable']
    if context.get('fixture')!=fixture:raise ValueError('Context belongs to another fixture')
    if not context.get('reviewer') or not context.get('evidence_id') or not context.get('order_id'):
        return None,['placement_context_unattributed']
    order=context.get('position_order')
    if not isinstance(order,list) or len(order)!=30 or set(order)!={f'{fixture}{i:02d}' for i in range(1,31)}:
        raise ValueError('Expected30 unique positions in explicitly supplied fill order')
    count=context.get('placement_count');complete=context.get('order_complete')
    if count is not None and (type(count) is not int or not 0<=count<=30):raise ValueError('Invalid placement count')
    if complete is not None and type(complete) is not bool:raise ValueError('Order complete must be explicit boolean')
    if complete is True:
        if count is not None and count!=30:raise ValueError('Count contradicts order-complete input')
        count=30
    if count is None:return None,['placement_progress_unavailable']
    return set(order[:count]),[]


def assess(predictions,context=None):
    if not predictions:raise ValueError('No target observations')
    fixture=predictions[0]['slot'][0]
    if len(predictions)!=30 or {p['slot'] for p in predictions}!={f'{fixture}{i:02d}' for i in range(1,31)}:
        raise ValueError('Need30 unique observations from one fixture')
    expected,reasons=expected_positions(fixture,context);rows=[]
    for prediction in predictions:
        label=prediction['proposed_class']
        if label not in CLASSES:raise ValueError('Unknown proposal class')
        slot=prediction['slot'];mapped=prediction.get('mapping_confirmed') is True
        visible=prediction.get('visibility')=='confirmed'
        row=dict(slot=slot,proposed_class=label,expected_occupied=None if expected is None else slot in expected,
            finding='unavailable',review_requested=False,reasons=list(reasons),quality_decision=None,hardware_command=None)
        if not mapped:row['reasons'].append('target_mapping_unreviewed')
        if not visible:row['reasons'].append('target_visibility_unconfirmed')
        if mapped and visible:
            if label in DEFECTS or (label!='no_vial' and prediction.get('occupied_defect_warning') is True):
                row.update(finding='visible_defect_candidate',review_requested=True)
                if label=='pass':row['reasons'].append('frozen_high_recall_score_warned_despite_pass_argmax')
                # Report defects in S too; do not interpret S routing as a release.
            elif label=='no_vial':
                if expected is None:row['finding']='empty_progress_unknown'
                elif slot in expected:row.update(finding='missed_placement_candidate',review_requested=True)
                else:row['finding']='empty_not_yet_expected'
            elif expected is not None and slot not in expected:
                row.update(finding='occupancy_context_conflict',review_requested=True)
            else:row['finding']='visually_acceptable_proposal'
        rows.append(row)
    return dict(schema='fixture-placement-assessment/v1',fixture=fixture,targets=rows,
        expected_count=None if expected is None else len(expected),context=deepcopy(context),
        review_requests=sum(r['review_requested'] for r in rows),physical_vial_association_required=False,
        qualified_product_decision_available=False,hardware_command=None)


class PlacementTracker:
    """Keep per-source progress and invalidate stale context after source changes."""
    def __init__(self):self.source=None;self.contexts={};self.last_frame=None;self.resets=[]

    def set_context(self,fixture,context):
        current,_=expected_positions(fixture,context);old=self.contexts.get(fixture)
        if old and old['order_id']==context.get('order_id'):
            previous,_=expected_positions(fixture,old)
            if current is not None and previous is not None and len(current)<len(previous):
                raise ValueError('Placement count rewound without an explicit new order')
        self.contexts[fixture]=deepcopy(context)

    def new_order(self,fixture,order_id):
        if fixture not in ('G','S') or not order_id:raise ValueError('Explicit fixture/order identity required')
        self.contexts.pop(fixture,None);self.resets.append(dict(reason='new_order',fixture=fixture,order_id=order_id))

    def ingest(self,source_id,frame_index,predictions):
        if not source_id or type(frame_index) is not int or frame_index<0:raise ValueError('Explicit source/frame identity required')
        reason=None
        if self.source is not None and source_id!=self.source:reason='source_changed'
        elif self.last_frame is not None and frame_index<self.last_frame:reason='source_rewound'
        if reason:self.contexts.clear();self.resets.append(dict(reason=reason,source_id=source_id,frame_index=frame_index))
        self.source=source_id;self.last_frame=frame_index
        result=assess(predictions,self.contexts.get(predictions[0]['slot'][0]))
        result.update(source_id=source_id,frame_index=frame_index,context_reset=reason)
        return result
