"""Finished-vial assessment and explicit reference eligibility."""
import numpy as np
from .measurement import delta_e


def rejection_reason(observation, reference):
    try:
        validate_references([reference])
    except (ValueError, TypeError, KeyError):
        return 'invalid_reference_provenance_or_color'
    if reference.get('scope') not in ('same_run_development', 'external_commissioning'):
        return 'unsupported_scope'
    if observation['recipe_id'] is None or str(reference.get('recipe_id')) != observation['recipe_id']:
        return 'recipe_mismatch'
    if observation.get('recipe_definition_id') and reference.get('recipe_definition_id') != observation['recipe_definition_id']:
        return 'recipe_definition_mismatch'
    if reference.get('measurement_config_id') != observation['measurement_config_id']:
        return 'measurement_profile_mismatch'
    if observation.get('measurement_profile_id') and reference.get('measurement_profile_id') != observation['measurement_profile_id']:
        return 'measurement_profile_mismatch'
    if reference.get('source_run_id') != observation['run_id'] or reference.get('source_video_sha256') != observation.get('source_video_sha256'):
        return 'recording_mismatch'
    if observation.get('source_annotation_sha256') != reference.get('source_annotation_sha256'):
        return 'annotation_source_mismatch'
    if type(reference.get('available_frame')) is not int or reference['available_frame'] >= observation['start_frame']:
        return 'not_available_before_fill'
    if reference.get('scope') == 'same_run_development' and reference['available_frame'] < reference.get('source_measurement_available_frame', 0):
        return 'reference_precedes_source_measurement'
    if reference.get('source_event_id') == observation['event_id']:
        return 'same_fill_reference'
    return None


def reference_for(observation, references):
    eligible = [r for r in references if rejection_reason(observation, r) is None]
    return max(enumerate(eligible), key=lambda pair: (pair[1]['available_frame'], pair[0]))[1] if eligible else None


def assess(observation, references, config):
    result = dict(status='unavailable', delta_e2000=None, reference_id=None,
                  expected_lab=None, threshold_de2000=config.review_threshold_de2000,
                  threshold_status='engineering_default_unvalidated',
                  explanation='', review_is_advisory=True, reference_rejections=[],
                  reference_available_frame=None, reference_source_description=None)
    if not observation['measurement_valid']:
        result['explanation'] = 'Unable to assess: ' + ', '.join(observation['quality_flags'])
        return result
    if observation.get('context_quality_flags'):
        result['explanation'] = 'Unable to assess: ' + ', '.join(observation['context_quality_flags'])
        return result
    if observation['recipe_id'] is None:
        result['explanation'] = 'Fill detected from images. Recipe context is unavailable; no recipe is inferred from color.'
        return result
    reference = reference_for(observation, references)
    result['reference_rejections'] = [dict(reference_id=r.get('reference_id'), reason=reason)
                                     for r in references if (reason := rejection_reason(observation, r))]
    if reference is None:
        result['explanation'] = 'No matching reference was available before this fill started.'
        return result
    score = delta_e(observation['lab'], reference['lab'])
    mismatch = score > config.review_threshold_de2000
    result.update(status='review' if mismatch else 'within_reference_band',
                  delta_e2000=score, reference_id=reference['reference_id'], expected_lab=reference['lab'],
                  reference_available_frame=reference['available_frame'], reference_source_description=reference.get('source_description'),
                  explanation=('Appearance differs from the selected recipe reference; operator review requested.'
                               if mismatch else 'Appearance is within the experimental reference band; this is not a release decision.'))
    return result


def validate_references(references):
    seen = set()
    for item in references:
        required = ('reference_id', 'recipe_id', 'lab', 'scope', 'measurement_config_id')
        if any(key not in item for key in required):
            raise ValueError('Reference is missing required provenance fields')
        if any(not isinstance(item[key], str) or not item[key].strip() for key in ('reference_id', 'recipe_id', 'scope', 'measurement_config_id')):
            raise ValueError('Reference identity, recipe, scope and measurement profile must be nonempty strings')
        lab = np.asarray(item['lab'], dtype=float)
        if lab.shape != (3,) or not np.isfinite(lab).all() or item['reference_id'] in seen:
            raise ValueError('Invalid or duplicate reference')
        if item['scope'] == 'same_run_development':
            if type(item.get('available_frame')) is not int or item['available_frame'] < 0 or not item.get('source_run_id'):
                raise ValueError('Same-run reference needs its source run and availability frame')
            if not item.get('source_video_sha256'):
                raise ValueError('Same-run reference needs a source video hash')
            if not item.get('source_event_id'):
                raise ValueError('Same-run reference needs its source detection/event ID')
            if (type(item.get('source_measurement_available_frame')) is not int or
                    not 0 <= item['source_measurement_available_frame'] <= item['available_frame'] or not item.get('source_description')):
                raise ValueError('Same-run reference needs source measurement availability and description')
        elif item['scope'] == 'external_commissioning':
            if not item.get('source_description'):
                raise ValueError('External reference needs a commissioning source description')
            if (type(item.get('available_frame')) is not int or item['available_frame'] < -1 or
                    not item.get('source_run_id') or not item.get('source_video_sha256')):
                raise ValueError('External reference needs explicit availability in the target recording; -1 means before recording')
        else:
            raise ValueError('Unknown reference scope')
        seen.add(item['reference_id'])
    return references
