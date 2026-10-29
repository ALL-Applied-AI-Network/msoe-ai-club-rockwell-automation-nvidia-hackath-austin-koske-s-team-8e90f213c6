"""Read-only catalog, append-only human revisions and portable correction exports."""
import hashlib
import json
from pathlib import Path
import sqlite3
from datetime import datetime, timezone
import uuid
from contextlib import contextmanager

SCHEMA = 'first-vial-spatial-review/v1'
STATUSES = {'accepted', 'changed', 'rejected', 'uncertain', 'unreviewed'}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def box(value):
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 4 or any(isinstance(v, bool) or not isinstance(v, (int,float)) for v in value):
        raise ValueError('Box must be [left, top, right, bottom], normalized to the original full image')
    if not 0 <= value[0] < value[2] <= 1 or not 0 <= value[1] < value[3] <= 1:
        raise ValueError('Box must have positive area and stay inside [0,1]')
    return value


class Catalog:
    def __init__(self, dataset, prelabels, cases=None):
        self.dataset = Path(dataset).resolve()
        self.prelabels = Path(prelabels).resolve()
        manifest = json.loads((self.prelabels/'manifest.json').read_text(encoding='utf-8'))
        pack = self.prelabels/'review-pack' if (self.prelabels/'review-pack').is_dir() else self.prelabels
        machine = {}
        self.classes = {}
        for camera in ('vial_fill','tank_level','cap_seal'):
            self.classes[camera] = (pack/camera/'classes.txt').read_text().splitlines()
            for record in json.loads((pack/camera/'annotations.json').read_text()):
                machine.setdefault(record['sample_id'], []).append(record)
        self.samples = {}
        for row in manifest:
            camera, filename = row['camera'], row['source']
            if camera not in self.classes or Path(filename).name != filename:
                raise ValueError('Unsafe or unsupported source identity')
            source = dict(kind='still', camera=camera, run=str(row['video_id']), filename=filename,
                relative_path=f'{camera}/images/{filename}', sha256=row['source_sha256'],
                capture_timestamp=row['timestamp'], frame_index=None, video_pts_seconds=None,
                width=row['source_size'][0], height=row['source_size'][1])
            self.samples[row['id']] = self.make_sample(row['id'], source, machine.get(row['id'], []))
        self.case_root = Path(cases).resolve().parent if cases else None
        self.case_manifest = json.loads(Path(cases).read_text()) if cases else {'cases':[]}
        for case in self.case_manifest['cases']:
            if case.get('sample_id') in self.samples:
                self.samples[case['sample_id']]['case'] = case
            elif case.get('source'):
                source = case['source']
                sample = self.make_sample(case['sample_id'], source, [])
                sample['case'] = case
                self.samples[sample['id']] = sample

    @staticmethod
    def make_sample(sample_id, source, records):
        candidates = []
        for r in records:
            b = r['box']
            original = box([b[k] for k in ('left','top','right','bottom')])
            candidates.append(dict(id='machine:'+str(r['class_id']), class_id=r['class_id'],
                original_box=original, machine_record=r, machine_record_sha256=hashlib.sha256(canonical(r)).hexdigest(),
                review_status='unreviewed'))
        identity = dict(source=source, machine=candidates)
        return dict(id=sample_id, source=source, candidates=candidates,
            fingerprint=hashlib.sha256(canonical(identity)).hexdigest(),
            association={'status':'unmatched','note':'No reviewed still/video mapping loaded. Capture time is not MP4 presentation time.'})

    def path(self, sample):
        source = sample['source']
        if source['kind'] == 'still':
            root, relative = self.dataset, source['relative_path']
        else:
            root, relative = self.case_root, source.get('image_relative_path')
            if not relative:
                raise FileNotFoundError('Video-reference case has no extracted image; use its source video/frame locator')
        if root is None:
            raise ValueError('Case image root unavailable')
        path = (root/relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Source path escapes its configured root')
        expected = source.get('sha256')
        if expected and sha(path) != expected:
            raise ValueError('Source image hash mismatch; review blocked')
        return path


class Reviews:
    def __init__(self, database, catalog, demo=False):
        self.database = Path(database)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self.catalog = catalog
        self.demo = demo
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS revisions (event_id TEXT PRIMARY KEY, sample_id TEXT NOT NULL, candidate_id TEXT NOT NULL, payload TEXT NOT NULL)')

    @contextmanager
    def connect(self):
        db=sqlite3.connect(self.database, timeout=20)
        try:
            with db:
                yield db
        finally:
            db.close()

    def history(self, sample_id=None):
        with self.connect() as db:
            rows = db.execute('SELECT payload FROM revisions '+('WHERE sample_id=? ' if sample_id else '')+'ORDER BY rowid', (sample_id,) if sample_id else ()).fetchall()
        return [json.loads(r[0]) for r in rows]

    def current(self, sample_id):
        current = {}
        for r in self.history(sample_id):
            current[r['candidate_id']] = r
        return current

    def validate(self, data, sample):
        if data['status'] not in STATUSES or not str(data.get('reviewer','')).strip():
            raise ValueError('Explicit review status and reviewer are required')
        if len(data.get('notes','')) > 10000:
            raise ValueError('Notes exceed 10000 characters')
        if not isinstance(data['class_id'], int) or not 0 <= data['class_id'] < len(self.catalog.classes[sample['source']['camera']]):
            raise ValueError('Unknown camera-local class ID')
        candidate = next((c for c in sample['candidates'] if c['id']==data['candidate_id']),None)
        if candidate:
            if candidate['class_id'] != data['class_id']:
                raise ValueError('Change class by rejecting and adding a new candidate')
        elif not data['candidate_id'].startswith('human:'):
            raise ValueError('New boxes need a human: identifier')
        proposed = box(data.get('proposed_box'))
        if data['status']=='accepted' and (not candidate or proposed != candidate['original_box']):
            raise ValueError('Accept must retain the original machine box; use changed for revisions')
        if data['status'] in ('accepted','changed') and proposed is None:
            raise ValueError('A reviewed positive box is required')
        if data['status']=='rejected' and proposed is not None:
            raise ValueError('Rejected candidates must have a null proposed box')
        if data['status']=='uncertain' and not data.get('notes','').strip():
            raise ValueError('Explain uncertainty in notes')
        return candidate

    def save(self, data):
        sample = self.catalog.samples[data['sample_id']]
        if sample['fingerprint'] != data.get('sample_fingerprint'):
            raise ValueError('Source/candidate fingerprint mismatch')
        self.catalog.path(sample)  # Check source bytes at decision time.
        candidate = self.validate(data,sample)
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            rows = db.execute('SELECT payload FROM revisions WHERE sample_id=? AND candidate_id=? ORDER BY rowid', (sample['id'],data['candidate_id'])).fetchall()
            if int(data.get('expected_revision',0)) != len(rows):
                raise ValueError('Revision conflict: reload before saving')
            record = dict(event_id=str(uuid.uuid4()), sample_id=sample['id'], candidate_id=data['candidate_id'],
                class_id=data['class_id'], status=data['status'], reviewer=data['reviewer'].strip(),
                reviewer_kind='demonstration' if self.demo else 'human_self_reported',
                notes=data.get('notes',''), proposed_box=data.get('proposed_box'),
                original_box=candidate['original_box'] if candidate else None,
                association_at_review=sample['association'],
                context_references_at_review=sample.get('case',{}).get('context_references'),
                sample_fingerprint=sample['fingerprint'], revision=len(rows)+1,
                saved_utc=datetime.now(timezone.utc).isoformat(),
                eligible_as_reviewed_label=not self.demo and data['status'] in ('accepted','changed'))
            db.execute('INSERT INTO revisions VALUES (?,?,?,?)', (record['event_id'],sample['id'],data['candidate_id'],json.dumps(record)))
        return record

    def export(self):
        history = self.history()
        samples = {r['sample_id']:self.catalog.samples[r['sample_id']] for r in history}
        return dict(schema=SCHEMA, coordinate_convention='normalized_full_image_xyxy; x in [0,1] relative to original width; y relative to original height',
            exported_utc=datetime.now(timezone.utc).isoformat(), purpose='demonstration' if self.demo else 'human_spatial_review',
            classes=self.catalog.classes, samples=samples, revisions=history,
            temporal_contract='Recipe identity, vial assignment, timing and measurement decisions are separate reviewed inputs. No clock conversion or still-to-vial join is performed.')

    def import_bundle(self, bundle):
        if bundle.get('schema')!=SCHEMA or bundle.get('classes')!=self.catalog.classes:
            raise ValueError('Incompatible correction schema or camera-local classes')
        if (bundle.get('purpose')=='demonstration') != self.demo:
            raise ValueError('Keep demonstration and real human workspaces separate')
        for sample_id, sample in bundle['samples'].items():
            if self.catalog.samples[sample_id]['fingerprint']!=sample['fingerprint'] or self.catalog.samples[sample_id]['source']!=sample['source'] or self.catalog.samples[sample_id]['candidates']!=sample['candidates']:
                raise ValueError('Correction source or original suggestions differ')
            self.catalog.path(self.catalog.samples[sample_id])
        with self.connect() as db:
            for record in bundle['revisions']:
                if record['sample_id'] not in bundle['samples']:
                    raise ValueError('Revision has no source provenance')
                sample=self.catalog.samples[record['sample_id']]
                candidate=self.validate(record,sample)
                if record.get('sample_fingerprint')!=sample['fingerprint'] or record.get('original_box')!=(candidate['original_box'] if candidate else None):
                    raise ValueError('Revision provenance differs from original source/candidate')
                if record.get('reviewer_kind') not in ('demonstration','human_self_reported'):
                    raise ValueError('Unknown reviewer kind')
                if (record['reviewer_kind']=='demonstration') != self.demo:
                    raise ValueError('Revision kind must match the demonstration/human workspace')
                eligible=record['reviewer_kind']=='human_self_reported' and record['status'] in ('accepted','changed')
                if record.get('eligible_as_reviewed_label')!=eligible:
                    raise ValueError('Inconsistent label eligibility')
                existing=db.execute('SELECT payload FROM revisions WHERE event_id=?',(record['event_id'],)).fetchone()
                if existing:
                    if json.loads(existing[0])!=record:
                        raise ValueError('Event identity collision')
                    continue
                previous=db.execute('SELECT payload FROM revisions WHERE sample_id=? AND candidate_id=? ORDER BY rowid',(record['sample_id'],record['candidate_id'])).fetchall()
                if record['revision']!=len(previous)+1:
                    raise ValueError('Divergent review history: use a separate workspace for this reviewer')
                if record.get('reviewer_kind')=='demonstration' and record.get('eligible_as_reviewed_label'):
                    raise ValueError('Demonstration cannot approve a real label')
                db.execute('INSERT INTO revisions VALUES (?,?,?,?)',(record['event_id'],record['sample_id'],record['candidate_id'],json.dumps(record)))
        return len(bundle['revisions'])
