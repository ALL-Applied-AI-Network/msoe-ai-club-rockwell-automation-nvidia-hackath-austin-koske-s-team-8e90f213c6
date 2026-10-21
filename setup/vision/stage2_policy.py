"""Offline Stage 2 experiment: qualified evidence and advisory action states.

No hardware API or authority to release product. Expected trajectories must come
from a separately commissioned recipe/phase reference, not an observed color ID.
"""
from dataclasses import dataclass, asdict
import math
import numpy as np
from replay.measurement import delta_e


@dataclass(frozen=True)
class Policy:
    threshold_de2000: float = 5.
    consecutive_samples: int = 3
    max_sample_age_s: float = .4
    max_sample_gap_s: float = .4
    repeated_bad_vials: int = 2
    recovery_good_vials: int = 2

    def __post_init__(self):
        if (not math.isfinite(self.threshold_de2000) or self.threshold_de2000<=0 or
            self.consecutive_samples<1 or self.max_sample_age_s<0 or self.max_sample_gap_s<=0 or
            self.repeated_bad_vials<1 or self.recovery_good_vials<1):
            raise ValueError('Invalid experimental policy')

    def to_dict(self):return asdict(self)


class EvidencePolicy:
    def __init__(self,config=Policy()):
        self.config=config;self.identity=None;self.context=None;self.last_s=None
        self.phase=None;self.count=0;self.warning_time=None

    def update(self,s):
        identity=(s['session_id'],s['vial_id'])
        if identity!=self.identity:
            self.identity=identity;self.context=None;self.last_s=None
            self.phase=None;self.count=0;self.warning_time=None
        now=float(s['source_time_s']);decision=float(s['decision_time_s'])
        reason=None
        if not all(math.isfinite(x) for x in (now,decision)):reason='nonfinite_time'
        elif self.last_s is not None and now<=self.last_s:reason='nonincreasing_sample_time'
        elif decision<now or decision-now>self.config.max_sample_age_s+1e-9:reason='stale_or_future_sample'
        elif not s.get('recipe_key'):reason='unknown_intended_recipe'
        elif not s.get('phase_key'):reason='unknown_ingredient_phase'
        elif not s.get('reference_id'):reason='missing_reference_identity'
        elif s.get('expected_phase_key')!=s['phase_key']:reason='phase_reference_mismatch'
        elif s.get('reference_recipe_key')!=s['recipe_key']:reason='recipe_reference_mismatch'
        elif s.get('measurement_profile')!=s.get('reference_measurement_profile'):reason='measurement_profile_mismatch'
        elif not s.get('measurement_profile'):reason='unknown_measurement_profile'
        elif s.get('reference_available_s',math.inf)>s['fill_started_s']:reason='reference_not_available_before_fill'
        elif s.get('context_available_s',math.inf)>now:reason='recipe_context_from_future'
        elif not s.get('vial_identity_unambiguous'):reason='ambiguous_vial_identity'
        elif s.get('geometry_status') not in ('simulated','reviewed'):reason='unreviewed_measurement_geometry'
        elif not s.get('measurement_valid'):reason='measurement_qa_failed'
        else:
            try:
                observed=np.asarray(s['observed_lab'],float);expected=np.asarray(s['expected_lab'],float)
                if observed.shape!=(3,) or expected.shape!=(3,) or not np.isfinite([observed,expected]).all():
                    reason='invalid_color_vector'
            except (TypeError,ValueError,KeyError):reason='invalid_color_vector'
        context=(s.get('recipe_key'),s.get('reference_id'),s.get('measurement_profile'))
        if reason is None and self.context is not None and context!=self.context:
            reason='recipe_or_reference_changed_during_fill'
        if (self.last_s is not None and now-self.last_s>self.config.max_sample_gap_s+1e-9) or s.get('phase_key')!=self.phase:
            self.count=0
        self.phase=s.get('phase_key')
        if math.isfinite(now) and (self.last_s is None or now>self.last_s):self.last_s=now
        result=dict(source_time_s=now,decision_time_s=decision,session_id=s['session_id'],vial_id=s['vial_id'],
                    delta_e2000=None,warning_time_s=self.warning_time,status='unavailable',reason=reason,
                    hardware_action_authorized=False,advisory_only=True)
        if reason:
            self.count=0
            if self.warning_time is not None:result['status']='review_required'
            return result
        self.context=context
        score=delta_e(observed,expected)
        self.count=self.count+1 if score>self.config.threshold_de2000 else 0
        if self.count>=self.config.consecutive_samples and self.warning_time is None:self.warning_time=decision
        result.update(delta_e2000=score,consecutive_deviations=self.count,warning_time_s=self.warning_time,
            status='review_required' if self.warning_time is not None else
                   'collecting_evidence' if self.count else 'within_expected_band')
        return result


class ActionCoordinator:
    """Simulated requests only. A flush acknowledgement cannot clear a hold."""
    def __init__(self,session_id,config=Policy()):
        self.session_id=session_id;self.config=config;self.recipe=None;self.seen=set()
        self.bad=0;self.good=0;self.hold=False;self.flush_acknowledged=False;self.events=[];self.last_s=-math.inf

    def complete(self,vial_id,recipe_key,status,time_s,*,session_id):
        if session_id!=self.session_id:raise ValueError('New source requires a new action coordinator')
        if vial_id in self.seen:raise ValueError('A completed vial cannot count twice')
        if time_s<=self.last_s:raise ValueError('Action evidence must advance in time')
        self.last_s=time_s;self.seen.add(vial_id)
        if recipe_key!=self.recipe:
            self.recipe=recipe_key;self.bad=self.good=0;self.flush_acknowledged=False
        if not recipe_key or status=='unavailable':
            self.bad=self.good=0;action='review_missing_evidence'
        elif status=='review_required':
            self.bad+=1;self.good=0;self.hold=True;self.flush_acknowledged=False
            action='flush_recommended' if self.bad>=self.config.repeated_bad_vials else 'hold_review_requested'
        elif status=='within_expected_band':
            self.bad=0
            self.good=self.good+1 if self.flush_acknowledged else 0
            if self.hold and self.flush_acknowledged and self.good>=self.config.recovery_good_vials:
                action='resume_recommended';self.hold=False;self.flush_acknowledged=False
            else:action='verify_next_fill' if self.hold else 'continue_monitoring'
        else:
            self.bad=self.good=0;action='review_missing_evidence'
        event=dict(session_id=session_id,vial_id=vial_id,time_s=time_s,recipe_key=recipe_key,
                   action=action,hold_latched=self.hold,hardware_action_authorized=False,advisory_only=True)
        self.events.append(event);return event

    def acknowledge_flush(self,time_s,operator,receipt_id,*,session_id):
        if session_id!=self.session_id or time_s<=self.last_s or not operator or not receipt_id or not self.hold:
            raise ValueError('Flush acknowledgement needs active hold, matching source, time and operator receipt')
        self.last_s=time_s;self.flush_acknowledged=True;self.good=self.bad=0
        event=dict(time_s=time_s,action='flush_acknowledged_verify_next_fills',operator=operator,
                   receipt_id=receipt_id,hold_latched=True,hardware_action_authorized=False,advisory_only=True)
        self.events.append(event);return event
