"""Append-only manual operator journal; sensor/machine actions are never inferred."""
from copy import deepcopy
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import threading
import uuid

from replay.measurement import delta_e
from .evidence import digest


class Conflict(ValueError):
    pass


def text(value, label, required=True, maximum=500):
    if value is None and not required:
        return ''
    if not isinstance(value, str) or len(value.strip()) > maximum or (required and not value.strip()):
        raise ValueError(f'{label} must be {"nonempty " if required else ""}text (up to {maximum} characters)')
    return value.strip()


class OperatorStore:
    def __init__(self, output, evidence):
        self.output, self.evidence = Path(output).resolve(), evidence
        self.output.mkdir(parents=True, exist_ok=True)
        self.path = self.output/'operator-journal.sqlite'
        self.lock = threading.RLock()
        with self.connect() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, request_id TEXT NOT NULL UNIQUE,
                    request_digest TEXT NOT NULL, case_id TEXT NOT NULL, kind TEXT NOT NULL,
                    source_frame INTEGER NOT NULL, actor TEXT NOT NULL, recorded_at_utc TEXT NOT NULL,
                    payload TEXT NOT NULL);
                CREATE TRIGGER IF NOT EXISTS no_event_update BEFORE UPDATE ON events BEGIN SELECT RAISE(ABORT,'History is append-only'); END;
                CREATE TRIGGER IF NOT EXISTS no_event_delete BEFORE DELETE ON events BEGIN SELECT RAISE(ABORT,'History is append-only'); END;
            ''')
            identity = json.dumps(evidence.identity, sort_keys=True)
            old = db.execute('SELECT value FROM metadata WHERE key=?', ('source_identity',)).fetchone()
            if old and old[0] != identity:
                raise ValueError('This workspace belongs to different video/model evidence; choose another --out')
            if not old:
                db.execute('INSERT INTO metadata VALUES (?,?)', ('source_identity', identity))
                db.execute('INSERT INTO metadata VALUES (?,?)', ('session_id', 'operator-'+uuid.uuid4().hex[:12]))
            self.session_id = db.execute('SELECT value FROM metadata WHERE key=?', ('session_id',)).fetchone()[0]

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def events(self, db=None):
        if db is None:
            with self.connect() as connection:
                return self.events(connection)
        return [dict(dict(r), payload=json.loads(r['payload'])) for r in db.execute('SELECT * FROM events ORDER BY seq')]

    @staticmethod
    def reduce(events):
        cases, references = {}, []
        for event in events:
            cid, kind, data = event['case_id'], event['kind'], event['payload']
            if kind == 'vial_started':
                cases[cid] = dict(data, case_id=cid, start_frame=event['source_frame'],
                    operator=event['actor'], revision=0, last_frame=event['source_frame'], lifecycle='open',
                    review_state='none', review_reason=None, actions=[], resolution=None, endpoint=None,
                    completed_frame=None, context_recorded_at_utc=event['recorded_at_utc'])
            case = cases[cid]
            case['revision'] += 1; case['last_frame'] = event['source_frame']
            if kind == 'fill_completed':
                case.update(lifecycle='complete', completed_frame=event['source_frame'], endpoint=data['endpoint'])
            elif kind == 'vial_abandoned':
                case.update(lifecycle='closed_uncertain', closure_note=data['note'])
            elif kind == 'review_requested':
                case.update(review_state='requested', review_reason=data['note'],
                            review_cycle_start=event['seq'], resolution=None)
            elif kind == 'review_acknowledged':
                case.update(review_state='acknowledged', acknowledged_by=event['actor'])
            elif kind == 'operator_action_recorded':
                case['actions'].append(dict(data, actor=event['actor'], recorded_at_utc=event['recorded_at_utc'], source_frame=event['source_frame'], event_seq=event['seq']))
            elif kind == 'follow_up_closed':
                case.update(review_state='closed', resolution=dict(data, actor=event['actor']))
            elif kind == 'development_reference_saved':
                references.append(dict(data, available_frame=event['source_frame'], actor=event['actor']))
        return cases, references

    def command(self, body):
        if not isinstance(body, dict):
            raise ValueError('Expected an operator command object')
        actor = text(body.get('actor'), 'Operator name', maximum=80)
        request_id = text(body.get('request_id'), 'Request identity', maximum=80)
        kind = text(body.get('command'), 'Command', maximum=50)
        frame = body.get('source_frame'); self.evidence.frame(frame)
        fingerprint = digest(body)
        with self.lock, self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            previous = db.execute('SELECT * FROM events WHERE request_id=?', (request_id,)).fetchone()
            if previous:
                if previous['request_digest'] != fingerprint:
                    raise Conflict('Request identity was already used for different input')
                return dict(case_id=previous['case_id'], event_seq=previous['seq'], duplicate=True)
            events = self.events(db); cases, refs = self.reduce(events)
            if kind == 'start':
                if any(c['lifecycle'] == 'open' for c in cases.values()):
                    raise Conflict('Complete or close the current vial record before starting another')
                if cases and frame <= max(c['last_frame'] for c in cases.values()):
                    raise Conflict('Start the next vial after the latest saved source point; use a fresh workspace to restart')
                physical = text(body.get('physical_vial_id'), 'Vial identifier', required=False, maximum=80)
                batch = text(body.get('batch_id'), 'Batch identifier', required=False, maximum=80)
                recipe = text(body.get('recipe_id'), 'Intended recipe', required=False, maximum=80)
                confirmed = body.get('identity_confirmed', False)
                if type(confirmed) is not bool or (confirmed and not physical):
                    raise ValueError('A confirmed physical identity needs its vial identifier')
                if physical and any(c['physical_vial_id'] == physical and c['batch_id'] == batch for c in cases.values()):
                    raise Conflict('This batch/vial identity already has a record')
                case_id = 'vial-'+uuid.uuid4().hex[:12]
                payload = dict(physical_vial_id=physical, batch_id=batch, recipe_id=recipe,
                               identity_confirmed=confirmed, context_kind='manual_recorded_demo',
                               context_note=text(body.get('note'), 'Context source/reason'))
                event_kind = 'vial_started'
            else:
                case_id = text(body.get('case_id'), 'Vial record')
                if case_id not in cases:
                    raise ValueError('Unknown vial record')
                case = cases[case_id]
                if type(body.get('expected_revision')) is not int or body['expected_revision'] != case['revision']:
                    raise Conflict('This record changed. Refresh before saving')
                if frame < case['last_frame']:
                    raise Conflict('Historical playback is read-only; return to the latest saved source point')
                if kind == 'complete':
                    if case['lifecycle'] != 'open':
                        raise Conflict('This fill was already completed or closed')
                    if body.get('confirmed') is not True:
                        raise ValueError('Explicit operator confirmation of dispensing completion is required')
                    payload = dict(endpoint=self.evidence.endpoint(case['start_frame'], frame), confirmation_source='manual_operator')
                    event_kind = 'fill_completed'
                elif kind == 'abandon':
                    if case['lifecycle'] != 'open':
                        raise Conflict('Only an open vial record can be closed as uncertain')
                    payload = dict(note=text(body.get('note'), 'Closure reason')); event_kind = 'vial_abandoned'
                elif kind == 'request_review':
                    if case['review_state'] in ('requested', 'acknowledged'):
                        raise Conflict('Review is already open for this record')
                    payload = dict(note=text(body.get('note'), 'Review reason')); event_kind = 'review_requested'
                elif kind == 'acknowledge':
                    if case['review_state'] != 'requested':
                        raise Conflict('Acknowledge an existing review request')
                    payload = {}; event_kind = 'review_acknowledged'
                elif kind == 'record_action':
                    if case['review_state'] != 'acknowledged':
                        raise Conflict('Acknowledge the review before recording a response')
                    action = body.get('action')
                    if action not in ('reviewed_evidence', 'set_aside_for_review', 'notified_csi', 'approved_procedure', 'other'):
                        raise ValueError('Choose a supported human action')
                    route = body.get('reported_station_route', 'not_observed')
                    if route not in ('not_observed', 'labeled_good', 'labeled_scrap', 'other'):
                        raise ValueError('Unknown reported station destination')
                    if body.get('performed') is not True:
                        raise ValueError('Only record an action that was actually performed in this demonstration')
                    procedure = text(body.get('procedure_id'), 'Approved procedure/receipt', required=action == 'approved_procedure', maximum=200)
                    payload = dict(action=action, performed=True, note=text(body.get('note'), 'Action details'),
                                   procedure_id=procedure, reported_station_route=route,
                                   route_semantics='unconfirmed', hardware_command_sent=False,
                                   outcome_status='not_independently_verified')
                    event_kind = 'operator_action_recorded'
                elif kind == 'close_review':
                    if case['review_state'] != 'acknowledged' or not any(a['event_seq'] > case['review_cycle_start'] for a in case['actions']):
                        raise Conflict('A human acknowledgement and recorded action are required before closing follow-up')
                    if case['lifecycle'] == 'open':
                        raise Conflict('Confirm completion or close the vial as uncertain first')
                    payload = dict(note=text(body.get('note'), 'Follow-up summary'), product_release_authorized=False,
                                   independent_quality_outcome='not_verified')
                    event_kind = 'follow_up_closed'
                elif kind == 'save_reference':
                    if not case['endpoint'] or not case['endpoint']['measurement_valid']:
                        raise ValueError('A completed QA-valid endpoint is needed for a development reference')
                    if not case['recipe_id'] or not case['identity_confirmed']:
                        raise ValueError('Enter intended recipe and confirm vial association before saving a reference')
                    if any(r['source_case_id'] == case_id for r in refs):
                        raise Conflict('This endpoint already has a development reference')
                    payload = dict(reference_id='dev-'+uuid.uuid4().hex[:12], recipe_id=case['recipe_id'],
                                   lab=case['endpoint']['lab'], rgb=case['endpoint']['rgb'],
                                   source_case_id=case_id, source_frame=case['completed_frame'],
                                   measurement_profile=self.evidence.profile, reference_kind='same_recording_development',
                                   description=text(body.get('note'), 'Reference description'), approved_reference=False)
                    event_kind = 'development_reference_saved'
                else:
                    raise ValueError('Unknown operator command')
            stamp = datetime.now(timezone.utc).isoformat()
            cursor = db.execute('INSERT INTO events (request_id,request_digest,case_id,kind,source_frame,actor,recorded_at_utc,payload) VALUES (?,?,?,?,?,?,?,?)',
                                (request_id, fingerprint, case_id, event_kind, frame, actor, stamp, json.dumps(payload, allow_nan=False)))
            return dict(case_id=case_id, event_seq=cursor.lastrowid, duplicate=False)

    def assessment(self, case, references):
        result = dict(status='unavailable', reason='Confirm fill completion to capture an endpoint',
                      diagnostic_status=None, delta_e2000=None, reference=None,
                      threshold_de2000=self.evidence.threshold, product_release_authorized=False)
        endpoint = case['endpoint']
        if not endpoint:
            return result
        if not endpoint['measurement_valid']:
            result['reason'] = 'Current sample is unavailable: '+', '.join(endpoint['quality_flags']); return result
        if not case['recipe_id']:
            result['reason'] = 'Intended recipe is unknown; appearance cannot supply recipe identity'; return result
        if not case['identity_confirmed']:
            result['reason'] = 'Physical vial association is unconfirmed'; return result
        eligible = [r for r in references if r['recipe_id'] == case['recipe_id']
                    and r['measurement_profile'] == endpoint['measurement_profile']
                    and r['available_frame'] < case['start_frame'] and r['source_case_id'] != case['case_id']]
        if not eligible:
            result['reason'] = 'No matching reference was available before this vial record started'; return result
        reference = max(eligible, key=lambda r: r['available_frame'])
        score = delta_e(endpoint['lab'], reference['lab'])
        result.update(reason='Development comparison only: geometry and reference are unreviewed',
                      diagnostic_status='deviation' if score > self.evidence.threshold else 'within_band',
                      delta_e2000=score, reference=reference)
        return result

    def snapshot(self, seconds, selected=None):
        sample = self.evidence.sample(seconds); frame = sample['source_frame']
        with self.lock:
            events = self.events(); canonical, _ = self.reduce(events)
            visible = [e for e in events if e['source_frame'] <= frame]
            cases, references = self.reduce(visible)
        values = []
        for case in cases.values():
            case['editable'] = canonical[case['case_id']]['last_frame'] <= frame
            case['assessment'] = self.assessment(case, references)
            values.append(case)
        current = next((c for c in values if c['case_id'] == selected), None)
        if current is None:
            current = next((c for c in reversed(values) if c['lifecycle'] == 'open'), values[-1] if values else None)
        recent = [dict(seq=e['seq'], case_id=e['case_id'], kind=e['kind'], source_frame=e['source_frame'],
                       seconds=self.evidence.frame(e['source_frame'])['seconds'], actor=e['actor'],
                       recorded_at_utc=e['recorded_at_utc'], payload=e['payload']) for e in visible[-20:]]
        return dict(session_id=self.session_id, source=self.evidence.public, sample=sample,
                    cases=values, current=current, references=references, events=recent,
                    counts=dict(vials=len(values), completed=sum(c['lifecycle'] == 'complete' for c in values),
                                open_reviews=sum(c['review_state'] in ('requested', 'acknowledged') for c in values),
                                actions=sum(len(c['actions']) for c in values)), hardware_commands_sent=0,
                    source_mode='recorded_demonstration', record_kind='practice_operator_workflow',
                    independent_outcomes_available=False)

    def export(self, seconds):
        snapshot = self.snapshot(seconds)
        return dict(schema='operator-workspace-export/v1', session_id=self.session_id,
                    source_identity=self.evidence.identity, source_mode='recorded_demonstration',
                    through_frame=snapshot['sample']['source_frame'], independent_outcomes_available=False,
                    hardware_commands_sent=0, cases=snapshot['cases'], references=snapshot['references'],
                    events=[{k: v for k, v in e.items() if k != 'request_digest'} for e in self.events()
                            if e['source_frame'] <= snapshot['sample']['source_frame']])
