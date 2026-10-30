"""Frozen-reference ablations for uncommissioned synthetic camera experiments."""
import bisect
import numpy as np
from replay.geometry import decode_mask
from replay.measurement import delta_e
from stage2_camera_reference import anchor_health,correction,feature


PATCHES=((192,224),(112,224),(112,160))
METHODS=('exact_raw','metered_raw','metered_normalized','metered_normalized_guarded')
SETTINGS=dict(maximum_reference_gap_ml=2.5,threshold_de2000=5.,consecutive_samples=3,
    maximum_anchor_disagreement=.05,background_patches_xy=list(PATCHES),patch_size_px=48,
    interpolation='linear in Lab, within phase/range only',real_commissioning=False)


def lookup(reference,phase,progress,exact=False):
    key=(phase,round(progress,6))
    if key in reference:return reference[key],None
    if exact:return None,'missing_exact_meter_reference'
    knots=sorted((p,lab) for (ph,p),lab in reference.items() if ph==phase)
    x=[p for p,_ in knots];at=bisect.bisect_left(x,progress)
    if at==0 or at==len(knots):return None,'outside_reference_range'
    low,a=knots[at-1];high,b=knots[at]
    if high-low>SETTINGS['maximum_reference_gap_ml']+1e-8:return None,'reference_gap_too_large'
    weight=(progress-low)/(high-low)
    return (np.asarray(a)*(1-weight)+np.asarray(b)*weight).tolist(),None


class DiagnosticSuite:
    def __init__(self):
        self.reference={'raw':{},'normalized':{}}
        self.anchors=None;self.reset()

    def reset(self):
        self.states={k:dict(count=0,warning=False,phase=None) for k in METHODS}

    def update(self,header,row,rgb,reference):
        anchors=np.asarray([np.median(rgb[y:y+48,x:x+48].reshape(-1,3),axis=0) for x,y in PATCHES])
        if reference and self.anchors is None:self.anchors=anchors.copy()
        healthy,gain,spread=anchor_health(anchors,self.anchors,SETTINGS['maximum_anchor_disagreement']) if self.anchors is not None else (False,None,None)
        m=row['measurement'];mask=decode_mask(m['geometry']['mask'],rgb.shape[:2]).astype(bool)
        normalized=None
        if mask.any() and self.anchors is not None:
            corrected,_,_=correction(rgb[mask],anchors,self.anchors,'srgb_gain')
            if corrected is not None:
                value,_=feature(corrected)
                if value is not None:normalized=value.tolist()
        phase=header['phase_key'];progress=round(sum(header['metered_ml']),6);key=(phase,progress)
        if reference and m['measurement_valid']:
            self.reference['raw'][key]=m['lab']
            if healthy and normalized is not None:self.reference['normalized'][key]=normalized
        results={}
        for method,state in self.states.items():
            if phase!=state['phase']:state['count']=0
            state['phase']=phase
            reason=None;score=None;expected=None
            is_normalized='normalized' in method
            observed=normalized if is_normalized else m['lab']
            if reference:reason='building_reference'
            elif not m['measurement_valid'] or observed is None:reason='measurement_qa_failed'
            elif method.endswith('_guarded') and not healthy:reason='background_anchor_unhealthy'
            else:
                expected,reason=lookup(self.reference['normalized' if is_normalized else 'raw'],
                    phase,progress,exact=method=='exact_raw')
            if reason:state['count']=0
            else:
                score=float(delta_e(observed,expected))
                state['count']=state['count']+1 if score>SETTINGS['threshold_de2000'] else 0
                state['warning']=state['warning'] or state['count']>=SETTINGS['consecutive_samples']
            results[method]=dict(status='reference' if reference else 'unavailable' if reason else
                'provisional_review' if state['warning'] else 'within_simulated_reference',
                reason=reason,delta_e2000=score,warning=state['warning'],
                consecutive_deviations=state['count'],expected_lab=expected)
        return dict(methods=results,background_healthy=bool(healthy),
            background_gain=gain.tolist() if gain is not None else None,background_spread=spread,
            background_anchors=anchors.tolist(),normalized_lab=normalized)

    def export(self):
        return dict(settings=SETTINGS,reference_background_anchors=self.anchors.tolist() if self.anchors is not None else None,
            references={kind:[dict(phase=k[0],metered_total_ml=k[1],lab=v) for k,v in table.items()]
                for kind,table in self.reference.items()})
