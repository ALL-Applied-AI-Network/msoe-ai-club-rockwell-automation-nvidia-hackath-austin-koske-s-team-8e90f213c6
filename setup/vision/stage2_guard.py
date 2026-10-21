"""Opt-in Stage 2 input-health gate, separate from deployed replay behavior."""
import math
from stage2_policy import EvidencePolicy,Policy


def number(value):
    try:
        v=float(value)
        return v if math.isfinite(v) else None
    except (TypeError,ValueError,OverflowError):return None


class GuardedEvidence:
    def __init__(self,config=Policy()):
        self.config=config;self.engine=EvidencePolicy(config)

    def blocked(self,s,reason):
        identity=(s.get('session_id'),s.get('vial_id'))
        if identity!=self.engine.identity:
            self.engine=EvidencePolicy(self.config);self.engine.identity=identity
        self.engine.count=0
        t=number(s.get('source_time_s'))
        if t is not None and (self.engine.last_s is None or t>self.engine.last_s):self.engine.last_s=t
        return dict(session_id=s.get('session_id'),vial_id=s.get('vial_id'),source_time_s=t,
            decision_time_s=number(s.get('decision_time_s')),delta_e2000=None,
            status='review_required' if self.engine.warning_time is not None else 'unavailable',
            warning_time_s=self.engine.warning_time,reason=reason,advisory_only=True,hardware_action_authorized=False)

    def update(self,s):
        if not s.get('session_id') or not s.get('vial_id'):return self.blocked(s,'unknown_source_or_vial')
        if not s.get('recipe_key'):return self.blocked(s,'unknown_intended_recipe')
        for key in ('source_time_s','decision_time_s','fill_started_s','reference_available_s','context_available_s','process_state_source_s'):
            if number(s.get(key)) is None:return self.blocked(s,'missing_or_invalid_'+key)
        now=number(s['source_time_s']);state=number(s['process_state_source_s'])
        if state>now+1e-9 or now-state>self.config.max_sample_age_s+1e-9:
            return self.blocked(s,'stale_or_future_process_state')
        if not s.get('phase_key') or s.get('process_phase_key')!=s['phase_key']:
            return self.blocked(s,'process_phase_unconfirmed')
        if not s.get('camera_reference_key') or s.get('camera_reference_key')!=s.get('reference_camera_key'):
            return self.blocked(s,'camera_reference_mismatch')
        if s.get('camera_health_valid') is not True:return self.blocked(s,'camera_reference_unhealthy')
        # Original geometry/recipe/reference/measurement gates remain mandatory.
        return self.engine.update(s)
