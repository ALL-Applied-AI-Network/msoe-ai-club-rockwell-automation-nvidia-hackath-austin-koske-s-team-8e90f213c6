"""Synthetic process only; the retained path is advanced, never a quality flag."""
import math


def fill_states(plan, path, cadence=.2):
    if not math.isfinite(cadence) or cadence<=0:raise ValueError('Positive finite cadence required')
    stages, at = [], plan['prefill_s']
    for position, tank in enumerate(plan['order']):
        stop = at + plan['volumes_ml'][tank]/plan['rates_ml_s'][tank]
        stages.append((at, stop, tank, str(position+1)))
        at = stop + (plan['pause_s'] if position < 2 else 0)
    recipe = 'synthetic-'+'-'.join(f'{v:g}' for v in plan['volumes_ml'])+'-order-'+'-'.join(map(str,plan['order']))
    actual, meter = [0.,0.,0.], [0.,0.,0.]
    for index in range(math.ceil((stages[-1][1]+plan['settle_s'])/cadence)+1):
        now = round(index*cadence,9); previous = max(0.,now-cadence); active = None; phase = '1'
        for start, stop, tank, key in stages:
            amount = max(0.,min(now,stop)-max(previous,start))*plan['rates_ml_s'][tank]
            delivered = path.dispense(tank,amount)
            actual = [a+b for a,b in zip(actual,delivered)]; meter[tank] += amount
            if now >= start: phase = key
            if start <= now < stop: active = tank
        yield dict(frame_index=index,sim_time_s=now,phase_key=phase,recipe_key=recipe,
                   metered_ml=list(meter),dispense_complete=now>=stages[-1][1],
                   active_tank=active,actual_components_ml=list(actual),path_components_ml=list(path.inventory))
