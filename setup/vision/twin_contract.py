"""Synthetic-only process state and loopback transport; no plant adapters."""
import hashlib
import json
import math
import socket
import struct


MAX_HEADER = 128 * 1024
FRAME_BYTES = 640 * 480 * 3
PUBLIC_FIELDS = frozenset(('schema_version', 'type', 'session_id', 'episode_id',
    'frame_index', 'sim_time_s', 'phase_key', 'recipe_key', 'metered_ml',
    'dispense_complete', 'rgb_sha256', 'shape'))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def recv_exact(connection, size):
    data = bytearray()
    while len(data) < size:
        part = connection.recv(size-len(data))
        if not part:
            raise EOFError('Simulation/inference connection closed before a complete message')
        data.extend(part)
    return bytes(data)


def send(connection, header, pixels=b''):
    encoded = json.dumps(header, allow_nan=False, separators=(',', ':')).encode()
    if len(encoded) > MAX_HEADER or len(pixels) > FRAME_BYTES:
        raise ValueError('Message exceeds fixed local simulation contract')
    connection.sendall(struct.pack('!II', len(encoded), len(pixels))+encoded+pixels)


def receive(connection):
    header_bytes, pixel_bytes = struct.unpack('!II', recv_exact(connection, 8))
    if not 0 < header_bytes <= MAX_HEADER or pixel_bytes > FRAME_BYTES:
        raise ValueError('Invalid message length')
    header = json.loads(recv_exact(connection, header_bytes))
    return header, recv_exact(connection, pixel_bytes)


def validate_frame(header, pixels, previous=None):
    if set(header) != PUBLIC_FIELDS or header['schema_version'] != 1 or header['type'] != 'frame':
        raise ValueError('Unexpected public fields; private simulation truth must not enter inference')
    if len(pixels) != FRAME_BYTES or header['shape'] != [480, 640, 3] or digest(pixels) != header['rgb_sha256']:
        raise ValueError('Invalid frame shape, byte length or hash')
    if not header['session_id'] or not header['episode_id'] or header['frame_index'] < 0:
        raise ValueError('Missing source identity')
    if not math.isfinite(header['sim_time_s']) or header['sim_time_s'] < 0:
        raise ValueError('Invalid simulation time')
    if not math.isclose(header['sim_time_s'],header['frame_index']*.2,abs_tol=1e-8):
        raise ValueError('Simulation clock must match the fixed 5 Hz frame contract')
    if len(header['metered_ml']) != 3 or any(not math.isfinite(v) or v < 0 for v in header['metered_ml']):
        raise ValueError('Invalid public meter quantities')
    if previous is not None:
        if (header['session_id'], header['episode_id']) != (previous['session_id'], previous['episode_id']):
            raise ValueError('Episode changes require an explicit reset')
        if header['frame_index'] != previous['frame_index']+1 or header['sim_time_s'] <= previous['sim_time_s']:
            raise ValueError('Missing, duplicate or out-of-order simulation frame')


class SharedPath:
    """Perfectly mixed finite hold-up compartment, three constituent volumes.

    Exponential displacement is integrated exactly for a constant inlet over
    each step. Parameters and flush effects are hypotheses, not CSI measurements.
    """
    def __init__(self, capacity_ml=2., initial_fraction=(0., 0., 1.)):
        if not math.isfinite(capacity_ml) or capacity_ml < 0:
            raise ValueError('Nonnegative finite hold-up required')
        if (len(initial_fraction) != 3 or any(not math.isfinite(f) or f<0 for f in initial_fraction)
                or abs(sum(initial_fraction)-1) > 1e-9):
            raise ValueError('Initial fractions must sum to one')
        self.capacity_ml = capacity_ml
        self.inventory = [capacity_ml*f for f in initial_fraction]

    def dispense(self, tank, amount_ml):
        if tank not in (0, 1, 2) or not math.isfinite(amount_ml) or amount_ml < 0:
            raise ValueError('Invalid tank or metered amount')
        inlet = [amount_ml if i == tank else 0. for i in range(3)]
        if self.capacity_ml == 0:
            return inlet
        decay = math.exp(-amount_ml/self.capacity_ml)
        after = [self.inventory[i]*decay + (self.capacity_ml*(1-decay) if i == tank else 0.) for i in range(3)]
        delivered = [self.inventory[i]+inlet[i]-after[i] for i in range(3)]
        self.inventory = after
        if min(delivered) < -1e-9 or abs(sum(delivered)-amount_ml) > 1e-8:
            raise AssertionError('Compartment mass balance failed')
        return [max(0., v) for v in delivered]

    def flush(self, tank, volume_ml):
        discharged = self.dispense(tank, volume_ml)
        return dict(flushed_ml=volume_ml, discharged_components_ml=discharged,
                    residual_components_ml=list(self.inventory), physical_effect='hypothesized_mixed_compartment')


def default_plan():
    return dict(schema_version=1, kind='live_isaac_camera_pipeline_pilot', seed=41,
        cadence_s=.2, volumes_ml=[10., 10., 10.], rates_ml_s=[6., 6., 6.],
        order=[0, 1, 2], prefill_s=.6, pause_s=.4, settle_s=.8,
        nominal_path_capacity_ml=2., initial_fraction=[1., 0., 0.],
        prior_fraction=[0., 1., 0.], vial_envelope_ml=40.,
        episodes=[dict(id='reference', reference=True, prior_residue=False, flush_ml=0.),
                  dict(id='normal-repeat', reference=False, prior_residue=False, flush_ml=0.),
                  dict(id='carryover', reference=False, prior_residue=True, flush_ml=0.),
                  dict(id='carryover-after-scripted-flush', reference=False, prior_residue=True, flush_ml=8.)],
        reference_status='rendered_development_reference_not_commissioned',
        action_origin='predeclared_simulated_operator_only',
        calibration_status='uncalibrated_process_and_optics', hardware_actions=0, run5_used=False)


def validate_plan(plan):
    if plan.get('schema_version')!=1 or plan.get('cadence_s')!=.2:
        raise ValueError('Live contract requires schema 1 and 5 Hz cadence')
    for key in ['volumes_ml','rates_ml_s']:
        values=plan[key]
        if len(values)!=3 or any(not math.isfinite(v) or v<=0 for v in values):
            raise ValueError('Three finite positive '+key+' values required')
    if sorted(plan['order'])!=[0,1,2]:raise ValueError('Ingredient order must be a permutation')
    for key in ['prefill_s','pause_s','settle_s','nominal_path_capacity_ml']:
        if not math.isfinite(plan[key]) or plan[key]<0:raise ValueError('Invalid '+key)
    if not math.isfinite(plan['vial_envelope_ml']) or sum(plan['volumes_ml'])>plan['vial_envelope_ml']:
        raise ValueError('Volume must fit the declared envelope')
    SharedPath(plan['nominal_path_capacity_ml'],plan['initial_fraction'])
    SharedPath(plan['nominal_path_capacity_ml'],plan['prior_fraction'])
    if not plan['episodes'] or not plan['episodes'][0]['reference'] or sum(e['reference'] for e in plan['episodes'])!=1:
        raise ValueError('Exactly one prior reference episode must come first')
    if len({e['id'] for e in plan['episodes']})!=len(plan['episodes']):raise ValueError('Duplicate case identities')
    for e in plan['episodes']:
        if not math.isfinite(e['flush_ml']) or e['flush_ml']<0:raise ValueError('Invalid flush volume')
        if e.get('flow_scale',1.)<=0 or not math.isfinite(e.get('flow_scale',1.)):raise ValueError('Invalid flow scale')
        if 'prior_fraction' in e:SharedPath(plan['nominal_path_capacity_ml'],e['prior_fraction'])
        for key,lo,hi in [('stream_radius_m',.0003,.003),('absorption_depth_m',.002,.3),('light_scale',.5,1.5)]:
            if key in e and not lo<=e[key]<=hi:raise ValueError('Out-of-range synthetic '+key)
        gains=e.get('sensor_gain',[1.,1.,1.])
        if len(gains)!=3 or any(not math.isfinite(v) or not .7<=v<=1.3 for v in gains):raise ValueError('Invalid sensor gain')


def episode_states(plan, episode, limit=None):
    """Advance a process sequentially; private truth never forms public metadata."""
    initial = episode.get('prior_fraction',plan['prior_fraction']) if episode['prior_residue'] else plan['initial_fraction']
    path = SharedPath(plan['nominal_path_capacity_ml'], initial)
    intervention = path.flush(plan['order'][0], episode['flush_ml']) if episode['flush_ml'] else None
    stages = []
    at = plan['prefill_s']
    rates=[v*episode.get('flow_scale',1.) for v in plan['rates_ml_s']]
    recipe_key='synthetic-'+'-'.join(f'{v:g}' for v in plan['volumes_ml'])+'-order-'+'-'.join(map(str,plan['order']))
    for position, tank in enumerate(plan['order']):
        end = at + plan['volumes_ml'][tank]/rates[tank]
        stages.append((at, end, tank, str(position+1)))
        at = end + (plan['pause_s'] if position < len(plan['order'])-1 else 0)
    end_s = stages[-1][1]
    count = math.ceil((end_s+plan['settle_s'])/plan['cadence_s'])+1
    actual = [0., 0., 0.]
    meter = [0., 0., 0.]
    for index in range(min(count, limit) if limit else count):
        now = round(index*plan['cadence_s'], 9)
        previous = max(0., now-plan['cadence_s'])
        active = None
        phase = '1'
        for start, stop, tank, key in stages:
            amount = max(0., min(now, stop)-max(previous, start))*rates[tank]
            out = path.dispense(tank, amount)
            actual = [a+b for a,b in zip(actual,out)]
            meter[tank] += amount
            if now >= start:
                phase = key
            if start <= now < stop:
                active = tank
        yield dict(frame_index=index, sim_time_s=now, phase_key=phase,
            recipe_key=recipe_key, metered_ml=list(meter),
            dispense_complete=now>=end_s, active_tank=active,
            actual_components_ml=list(actual), path_components_ml=list(path.inventory),
            intervention=intervention if index==0 else None)
