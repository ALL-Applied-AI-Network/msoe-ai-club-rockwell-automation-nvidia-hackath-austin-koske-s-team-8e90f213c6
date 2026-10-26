"""Opt-in evidence/reference guards for integration rehearsal, not app defaults.

Human attestations are attributed inputs, not authenticated physical proof.
Model output cannot qualify its own reference or approve a product.
"""
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import math


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def native_seconds(source):
    if source.get('clock_kind')!='native' or type(source.get('pts')) is not int or not source.get('time_base'):
        return None
    scale=Fraction(source['time_base'])
    if scale<=0:raise ValueError('Native time base must be positive')
    value=source['pts']*scale
    if value<0:raise ValueError('Native source time cannot be negative')
    return value


def validate_observation(observation):
    if observation.get('schema')!='station-evidence/v1' or observation.get('station') not in ('vial_fill','final_fixture'):
        raise ValueError('Unknown observation contract/station')
    if observation.get('data_domain') not in ('real','synthetic'):
        raise ValueError('Explicit data domain required')
    source=observation['source'];target=observation['target']
    for key in ('camera_id','stream_id','source_id'):
        if not isinstance(source.get(key),str) or not source[key]:raise ValueError('Missing source identity: '+key)
    if source.get('dimensions')!=[640,480]:raise ValueError('Unsupported camera dimensions; do not resize silently')
    if not(source.get('decoded_rgb_sha256') or source.get('encoded_file_sha256')):
        raise ValueError('Pixel/file hash kind must be explicit')
    if not target.get('id') or not target.get('context_id') or not observation.get('profile_id'):
        raise ValueError('Target/context/profile identity required')
    if source.get('clock_kind')=='native':
        if not source.get('clock_domain_id') or native_seconds(source) is None:raise ValueError('Incomplete native clock')
    elif source.get('clock_kind')=='capture_only':
        if source.get('pts') is not None or source.get('time_base') is not None or source.get('frame_index') is not None:
            raise ValueError('Capture-only still cannot invent a native frame/PTS')
    else:raise ValueError('Unknown source clock kind')
    box=target.get('box_xyxy')
    if box is not None and (len(box)!=4 or not all(type(v) in (int,float) and math.isfinite(v) for v in box)
            or not(0<=box[0]<box[2]<=640 and 0<=box[1]<box[3]<=480)):
        raise ValueError('Target crop outside original source')
    if observation.get('quality_decision') is not None or observation.get('hardware_command') is not None:
        raise ValueError('Evidence ingestion does not authorize product quality or hardware action')
    if not observation.get('models') or any(not r.get('sha256') for r in observation['models']):
        raise ValueError('Model identities are required')


class ReferenceGate:
    """Invalidate temporal references on context/profile/stream changes or rewind.

    max_age_seconds is an explicit integration policy, not a fitted model/QA
    threshold. Default None imposes no invented lifetime on historical evidence.
    V1 requires comparable native source clocks for causal qualification.
    """
    CONDITIONS={'temporal_background':'empty','capped_golden':'capped','recipe_endpoint':'accepted_recipe_endpoint'}

    def __init__(self,max_age_seconds=None):
        if max_age_seconds is not None and (not math.isfinite(max_age_seconds) or max_age_seconds<=0):
            raise ValueError('Reference age limit must be positive')
        self.max_age=max_age_seconds;self.references={};self.last={};self.resets=[]

    def install(self,reference):
        if reference.get('purpose') not in self.CONDITIONS or not reference.get('reference_id'):
            raise ValueError('Reference purpose/identity required')
        validate_observation(reference['observation'])
        self.references[reference['purpose']]=deepcopy(reference)

    def invalidate(self,purpose,reason):
        old=self.references.pop(purpose,None)
        if old:self.resets.append(dict(purpose=purpose,reason=reason,reference_id=old['reference_id']))

    def check(self,observation,purpose='temporal_background',recipe_id=None,completion_confirmed=False):
        validate_observation(observation)
        if purpose not in self.CONDITIONS:raise ValueError('Unknown reference role')
        source=observation['source'];target=observation['target'];now=native_seconds(source)
        key=(purpose,source['camera_id']);previous=self.last.get(key)
        if previous and source['stream_id']==previous['stream_id'] and now is not None and previous['time'] is not None and now<previous['time']:
            self.invalidate(purpose,'source_clock_rewound')
        self.last[key]=dict(stream_id=source['stream_id'],time=now)
        reference=self.references.get(purpose)
        if reference is None:return dict(usable=False,reasons=['reference_missing_or_reset'],reference_id=None,purpose=purpose)
        ref=reference['observation'];old=ref['source'];reasons=[]
        resets=[]
        if source['camera_id']!=old['camera_id']:resets.append('camera_changed')
        if observation['profile_id']!=ref['profile_id']:resets.append('measurement_profile_changed')
        if source.get('discontinuity') is True:resets.append('declared_source_discontinuity')
        if observation['qualification'].get('alignment')=='invalid':resets.append('alignment_invalid')
        if purpose=='temporal_background':
            if source['stream_id']!=old['stream_id']:resets.append('source_stream_changed')
            if target['context_id']!=ref['target']['context_id']:resets.append('target_context_changed')
        if purpose=='capped_golden' and target['id']!=ref['target']['id']:resets.append('target_slot_changed')
        if resets:
            self.invalidate(purpose,','.join(resets))
            return dict(usable=False,reasons=resets,reference_id=reference['reference_id'],purpose=purpose)
        if ref['data_domain']!=observation['data_domain']:reasons.append('reference_domain_mismatch')
        qualification=reference.get('qualification',{})
        if (qualification.get('status')!='human_reviewed' or not qualification.get('reviewer')
            or not qualification.get('evidence_id') or qualification.get('origin')!='human_review'
            or reference.get('demonstration') is True or qualification.get('demonstration') is True):
            reasons.append('reference_not_human_qualified')
        if qualification.get('confirmed_condition')!=self.CONDITIONS[purpose]:reasons.append('reference_condition_unconfirmed')
        if observation['qualification'].get('alignment')!='confirmed':reasons.append('current_alignment_unreviewed')
        if ref['qualification'].get('alignment')!='confirmed':reasons.append('reference_alignment_unreviewed')
        earlier=native_seconds(old)
        if now is None or earlier is None or not source.get('clock_domain_id') or source['clock_domain_id']!=old.get('clock_domain_id'):
            reasons.append('reference_causality_unverified')
        elif earlier>=now:reasons.append('reference_not_strictly_earlier')
        elif self.max_age is not None and float(now-earlier)>self.max_age:reasons.append('reference_expired')
        if purpose=='recipe_endpoint':
            if not recipe_id or reference.get('recipe_id')!=recipe_id:reasons.append('intended_recipe_or_reference_missing')
            if not completion_confirmed:reasons.append('fill_completion_unconfirmed')
            if not target.get('physical_vial_id'):reasons.append('physical_vial_identity_unconfirmed')
        return dict(usable=not reasons,reasons=reasons,reference_id=reference['reference_id'],purpose=purpose,
                    qualification='attributed input; not independently authenticated by this module')


def evidence_readiness(observation,temporal=None,recipe_reference=None,operator_context=None):
    validate_observation(observation)
    reasons=[];context=operator_context or {};recipe_ready=False;placement_ready=False
    if observation['station']=='vial_fill':
        if not observation['measurement'].get('qa_pass'):reasons.append('measurement_qa_failed')
        if not temporal or not temporal['usable']:reasons.append('temporal_reference_unqualified')
        if not context.get('recipe_id'):reasons.append('intended_recipe_unconfirmed')
        if not recipe_reference or not recipe_reference.get('usable'):reasons.append('accepted_recipe_reference_missing')
        if context.get('completion_confirmed') is not True:reasons.append('fill_completion_unconfirmed')
        recipe_ready=not reasons and bool(observation['target'].get('physical_vial_id'))
    else:
        if observation['qualification'].get('alignment')!='confirmed':reasons.append('slot_mapping_unreviewed')
        if observation['qualification'].get('visibility')!='confirmed':reasons.append('visibility_or_occlusion_unreviewed')
        if type(context.get('expected_occupied')) is not bool or not context.get('reviewer') or not context.get('evidence_id'):
            reasons.append('placement_context_unknown')
        placement_ready=not reasons
        supported=observation['proposal'].get('supported_classes',[])
        if not {'crooked_cap','damaged_vial'}.issubset(supported):
            reasons.append('crooked_cap_and_damaged_body_unsupported')
    if not observation['target'].get('physical_vial_id'):reasons.append('physical_vial_association_unlinked')
    return dict(proposal_display_available=True,frozen_measurement_qa_pass=observation['measurement'].get('qa_pass'),
                recipe_comparison_ready=recipe_ready,placement_assessment_ready=placement_ready,
                qualified_product_decision_available=False,reasons=reasons,quality_decision=None,hardware_command=None)


def associate_physical_vial(fill,fixture,attestation=None):
    for observation in (fill,fixture):validate_observation(observation)
    if fill['station']!='vial_fill' or fixture['station']!='final_fixture':raise ValueError('Expected fill and fixture observations')
    if fill['data_domain']!=fixture['data_domain']:return dict(status='unlinked',physical_vial_id=None,reason='Synthetic and real observations cannot describe the same physical vial')
    if not attestation:return dict(status='unlinked',physical_vial_id=None,reason='No attributed physical association evidence')
    if attestation.get('demonstration') or any(not attestation.get(k) for k in ('reviewer','physical_vial_id','evidence_id','basis')):
        return dict(status='unlinked',physical_vial_id=None,reason='Demonstration/incomplete association is not physical proof')
    if attestation.get('observation_ids')!=[fill['observation_id'],fixture['observation_id']]:raise ValueError('Association evidence names different observations')
    return dict(status='operator_attributed',physical_vial_id=attestation['physical_vial_id'],attestation=deepcopy(attestation),
                limitation='Attributed operator statement, not automated or independently authenticated identity proof')
