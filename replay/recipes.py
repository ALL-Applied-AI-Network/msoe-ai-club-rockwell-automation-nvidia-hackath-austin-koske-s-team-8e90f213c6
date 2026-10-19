"""Explicit intended-recipe selections, separate from images and evaluation labels."""
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

TANKS = ('T100', 'T200', 'T400')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def recipe_definition(recipe_id, volumes_ml, dispense_order):
    if not isinstance(recipe_id, str) or not recipe_id.strip():
        raise ValueError('Recipe ID must be a nonempty string')
    if not isinstance(volumes_ml, dict) or set(volumes_ml) - set(TANKS):
        raise ValueError('Use only T100, T200 and T400 volumes')
    volumes = {tank: volumes_ml.get(tank, 0) for tank in TANKS}
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in volumes.values()):
        raise ValueError('Tank volumes must be finite nonnegative numbers')
    volumes = {tank: float(value) for tank, value in volumes.items()}
    order = dispense_order.split('>') if isinstance(dispense_order, str) else dispense_order
    if not isinstance(order, list) or not order or not all(isinstance(t, str) for t in order) or len(order) != len(set(order)) or set(order) != {t for t, v in volumes.items() if v > 0}:
        raise ValueError('Dispense order must name each positive-volume tank exactly once')
    return dict(recipe_id=recipe_id.strip(), volumes_ml=volumes, dispense_order='>'.join(order))


class RecipePlan:
    def __init__(self, document=None):
        document = deepcopy(document or dict(schema_version=1, plan_id='manual', source_description='Explicit manual selections', selections=[]))
        if document.get('schema_version') != 1 or not isinstance(document.get('plan_id'), str) or not document['plan_id'].strip() or not isinstance(document.get('source_description'), str) or not document['source_description'].strip():
            raise ValueError('Recipe plan needs schema_version=1, plan_id and source_description')
        if not isinstance(document.get('selections'), list):
            raise ValueError('Recipe plan selections must be a list')
        seen, activations = set(), set()
        for s in document['selections']:
            if not isinstance(s, dict):
                raise ValueError('Each recipe selection must be an object')
            allowed = {'selection_id', 'run_id', 'source_video_sha256', 'available_frame', 'active_from_frame',
                       'recipe_id', 'volumes_ml', 'dispense_order', 'recorded_by', 'recorded_at_utc', 'provenance'}
            if set(s) - allowed:
                raise ValueError('Unexpected recipe-selection fields; evaluation/vial labels are not recipe inputs')
            if not isinstance(s.get('selection_id'), str) or not s['selection_id'].strip() or s['selection_id'] in seen:
                raise ValueError('Selection IDs must be unique nonempty strings')
            if any(type(s.get(k)) is not int for k in ('run_id', 'available_frame', 'active_from_frame')) or s['run_id'] < 1 or not 0 <= s['available_frame'] <= s['active_from_frame']:
                raise ValueError('Selection availability must be at or before activation; frames are nonnegative integers')
            if not isinstance(s.get('source_video_sha256'), str) or not s['source_video_sha256']:
                raise ValueError('Bind each selection to its recording hash')
            key = (s['run_id'], s['source_video_sha256'], s['active_from_frame'])
            if key in activations:
                raise ValueError('Two selections cannot activate at the same source frame')
            if not isinstance(s.get('recorded_by'), str) or not s['recorded_by'].strip():
                raise ValueError('Record who supplied the intended recipe')
            try:
                stamp = datetime.fromisoformat(s['recorded_at_utc'].replace('Z', '+00:00'))
                if stamp.utcoffset() is None:
                    raise ValueError()
            except (KeyError, AttributeError, ValueError):
                raise ValueError('recorded_at_utc needs an ISO timestamp with timezone') from None
            provenance = s.get('provenance', {})
            if not isinstance(provenance, dict) or provenance.get('kind') not in ('manual_operator', 'operator_plan', 'illustrative_demo') or not isinstance(provenance.get('description'), str) or not provenance['description'].strip():
                raise ValueError('Recipe provenance needs an explicit kind and description')
            if s.get('recipe_id') is None:
                s.update(recipe_id=None, volumes_ml={}, dispense_order=None)
            else:
                s.update(recipe_definition(s['recipe_id'], s.get('volumes_ml'), s.get('dispense_order')))
            seen.add(s['selection_id'])
            activations.add(key)
        self.document = document

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding='utf-8-sig')))

    @property
    def fingerprint(self):
        return digest(self.document)

    def to_dict(self):
        return deepcopy(self.document)

    def selections(self, run_id, video_sha256, through=None):
        return sorted((deepcopy(s) for s in self.document['selections'] if s['run_id'] == run_id and
                       s['source_video_sha256'] == video_sha256 and
                       (through is None or max(s['active_from_frame'], s['available_frame']) <= through)),
                      key=lambda s: s['active_from_frame'])

    def at(self, run_id, video_sha256, frame):
        rows = self.selections(run_id, video_sha256, frame)
        return rows[-1] if rows else None

    def bind(self, observation, until=None):
        o = deepcopy(observation)
        run_id, video = o['run_id'], o.get('source_video_sha256')
        selection = self.at(run_id, video, o['start_frame'])
        o.update(recipe_id=None, volumes_ml={}, dispense_order=None, recipe_definition_id=None,
                 recipe_context=None, context_quality_flags=[], position_in_batch=None,
                 previous_recipe_id=None, baseline_hold_first=None, baseline_hold_first_two=None)
        if selection:
            o.update({key: selection[key] for key in ('recipe_id', 'volumes_ml', 'dispense_order')})
            o['recipe_context'] = dict(selection, plan_id=self.document['plan_id'])
            o['recipe_definition_id'] = digest({k: o[k] for k in ('recipe_id', 'volumes_ml', 'dispense_order')}) if o['recipe_id'] else None
        boundary = until if until is not None else o.get('departure_frame', o['available_frame'])
        for s in self.selections(run_id, video, boundary):
            if o['start_frame'] < s['active_from_frame'] and (s['active_from_frame'] <= boundary if until is not None else s['active_from_frame'] < boundary):
                definition = digest({k: s[k] for k in ('recipe_id', 'volumes_ml', 'dispense_order')}) if s['recipe_id'] else None
                if definition != o['recipe_definition_id']:
                    o['context_quality_flags'] = ['recipe_changed_during_fill']
        return o
