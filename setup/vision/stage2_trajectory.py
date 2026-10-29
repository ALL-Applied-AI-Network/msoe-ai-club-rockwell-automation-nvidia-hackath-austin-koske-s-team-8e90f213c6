"""Explicit recipe/phase/progress reference lookup; no image-derived recipe ID."""
import numpy as np


class TrajectoryBank:
    def __init__(self,profiles):
        self.profiles={}
        for p in profiles:
            key=(p['recipe_key'],p['phase_key'],p['camera_profile'])
            x=np.asarray(p['cumulative_ml'],float);lab=np.asarray(p['lab'],float)
            if key in self.profiles or len(x)<1 or lab.shape!=(len(x),3) or not np.isfinite(x).all() or not np.isfinite(lab).all():
                raise ValueError('Invalid or duplicate trajectory')
            if len(x)>1 and not np.all(np.diff(x)>0):raise ValueError('Progress knots must increase')
            if not np.isfinite(p['available_s']) or p['reference_status'] not in ('simulated','commissioned'):
                raise ValueError('Reference availability and provenance are required')
            self.profiles[key]=(dict(p),x,lab)

    def lookup(self,recipe_key,phase_key,camera_profile,cumulative_ml,fill_started_s):
        item=self.profiles.get((recipe_key,phase_key,camera_profile))
        if item is None:return dict(status='unavailable',reason='missing_recipe_phase_camera_reference')
        p,x,lab=item
        try:progress=float(cumulative_ml)
        except (TypeError,ValueError):return dict(status='unavailable',reason='invalid_meter_progress')
        if not np.isfinite(progress):return dict(status='unavailable',reason='invalid_meter_progress')
        if p['available_s']>fill_started_s:return dict(status='unavailable',reason='future_reference')
        if progress<x[0]-1e-7 or progress>x[-1]+1e-7:
            return dict(status='unavailable',reason='meter_progress_outside_reference')
        expected=[float(np.interp(progress,x,lab[:,i])) for i in range(3)]
        return dict(status='available',reason=None,expected_lab=expected,reference_id=p['reference_id'],
                    reference_status=p['reference_status'],available_s=p['available_s'],progress_ml=progress)
