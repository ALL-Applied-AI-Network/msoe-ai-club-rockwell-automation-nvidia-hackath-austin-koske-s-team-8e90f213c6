"""Opt-in fixture reference gain correction; never changes liquid measurement RGB.

Estimate one gain for a complete60-target scene, excluding each central32px disk.
No labels or predicted classes enter the estimate. This does not align images or
qualify a real reference, and cannot recover clipped or missing image information.
"""
def estimate_scene_gain(current,golden):
    import numpy as np
    current=np.asarray(current);golden=np.asarray(golden)
    if current.shape!=golden.shape or current.shape!=(60,3,96,96):raise ValueError('Need one matched60-target source scene')
    if not np.isfinite(current).all() or not np.isfinite(golden).all():raise ValueError('Nonfinite RGB')
    if current.min()<0 or current.max()>1 or golden.min()<0 or golden.max()>1:raise ValueError('RGB must be normalized0–1')
    yy,xx=np.mgrid[:96,:96];outside=(xx-48)**2+(yy-48)**2>32**2
    cur=current.mean(1);ref=golden.mean(1)
    valid=(cur>.06)&(cur<.92)&(ref>.06)&(ref<.92)&outside[None]
    fraction=float(valid.mean())
    if fraction<.15:return dict(available=False,gain=None,usable_fraction=fraction,reason='insufficient_unclipped_context')
    gain=float(np.median(cur[valid]/ref[valid]))
    if not .7<=gain<=1.4:return dict(available=False,gain=gain,usable_fraction=fraction,reason='gain_outside_declared_range')
    return dict(available=True,gain=gain,usable_fraction=fraction,reason=None,
        reference_alignment_qualified=False,physical_reference_qualified=False)
