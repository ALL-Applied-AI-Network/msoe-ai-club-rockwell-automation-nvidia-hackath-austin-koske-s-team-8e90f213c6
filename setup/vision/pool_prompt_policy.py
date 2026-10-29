"""Experimental current-frame prompt association and colored-pool guidance.

These are camera-profile search priors, not approved labels or measurement QA.
The seed arm intentionally has limited applicability to pale/clear liquid.
"""
import cv2
import numpy as np


POLICY = dict(detector_confidence=.35, association_margin_width=.20,
              association_margin_height=.03, expanded_containment=.90,
              seed_search_inset_width=.10, seed_search_start_height=.50,
              seed_component_must_reach_height=.80,
              seed_hsv_s_min=45, seed_hsv_v_min=12, seed_hsv_v_max_exclusive=240,
              opening_kernel=3, ambiguous_area_relative_tolerance=.01)

BRIDGE_POLICY = dict(max_prompt_age_s=.6, max_box_area_growth=2.,
                     minimum_station_iou=.65, carrier_motion_reset=3.)


def containment(box, region):
    x, y, r, b = box; a, c, d, e = region
    intersection = max(0, min(r, d)-max(x, a))*max(0, min(b, e)-max(y, c))
    return intersection/max(1e-9, (r-x)*(b-y))


def area(box):
    return (box[2]-box[0])*(box[3]-box[1])


def associated_predictions(predictions):
    """Keep original predictions; loosen only prompt search, never QA gates."""
    vials = [p['box'] for p in predictions if p['class_id'] == 0 and p['confidence'] >= POLICY['detector_confidence']
             and p['box'][0] < 306.5 < p['box'][2] and p['box'][3]-p['box'][1] > 180]
    if len(vials) != 1:
        return None, [], [dict(reason='station_vial_missing_or_ambiguous')]
    vial = vials[0]; x, y, r, b = vial; w, h = r-x, b-y
    mw, mh = POLICY['association_margin_width']*w, POLICY['association_margin_height']*h
    expanded = [x-mw, y-mh, r+mw, b+mh]
    accepted = []; decisions = []
    for i, prediction in enumerate(predictions):
        if prediction['class_id'] != 1 or prediction['confidence'] < POLICY['detector_confidence']:
            continue
        box = prediction['box']; cx = (box[0]+box[2])/2; cy = (box[1]+box[3])/2
        overlap = containment(box, expanded)
        allowed = x < cx < r and y < cy < b and overlap >= POLICY['expanded_containment']
        # Prompt is clipped to a one-pixel inset of the detected vial. Raw SAM
        # outputs are preserved without clipping or substituting this rectangle.
        clipped = [max(box[0], x+1), max(box[1], y+1), min(box[2], r-1), min(box[3], b-1)]
        allowed = allowed and clipped[2] > clipped[0] and clipped[3] > clipped[1]
        record = dict(prediction_index=i, source_box=list(box),
                      original_containment=containment(box, vial), expanded_containment=overlap,
                      clipped_box=clipped, accepted=allowed)
        decisions.append(record)
        if allowed:
            accepted.append(dict(**record, confidence=prediction['confidence']))
    return vial, accepted, decisions


def pool_prompts(rgb, predictions):
    """Return both ablation arms from one frame; never use past/future labels."""
    vial, candidates, decisions = associated_predictions(predictions)
    result = dict(vial=vial, association=decisions,
                  association_boxes=[p['clipped_box'] for p in candidates],
                  guided_boxes=[], points=[], point_labels=[], selected_prediction_index=None,
                  seed=None, seed_component_pixels=0, status='no_associated_liquid')
    if vial is None or not candidates:
        return result
    x, y, r, b = vial; w, h = r-x, b-y
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # Identical HSV values to the existing sampling gates, used only as weak
    # prompt evidence. No gate is changed and this mask is never returned as SAM.
    colored = ((hsv[:, :, 1] >= POLICY['seed_hsv_s_min']) &
               (hsv[:, :, 2] >= POLICY['seed_hsv_v_min']) &
               (hsv[:, :, 2] < POLICY['seed_hsv_v_max_exclusive'])).astype(np.uint8)
    search = np.zeros(rgb.shape[:2], np.uint8)
    inset = POLICY['seed_search_inset_width']*w
    left, right = max(0, int(x+inset)), min(rgb.shape[1], int(r-inset))
    top, bottom = max(0, int(y+POLICY['seed_search_start_height']*h)), min(rgb.shape[0], int(b-1))
    search[top:bottom, left:right] = colored[top:bottom, left:right]
    kernel = POLICY['opening_kernel']
    search = cv2.morphologyEx(search, cv2.MORPH_OPEN, np.ones((kernel, kernel), np.uint8))
    n, components, stats, _ = cv2.connectedComponentsWithStats(search)
    plausible = []
    for k in range(1, n):
        sx, sy, sw, sh, pixels = stats[k]
        if sy+sh < y+POLICY['seed_component_must_reach_height']*h:
            continue
        component = components == k
        # Component must be supported by an associated detection, never merely
        # a fixed lower-vial prompt producing foreground on empty glass.
        supported = np.zeros_like(search)
        for p in candidates:
            a, c, d, e = [int(v) for v in p['clipped_box']]
            supported[c:e, a:d] |= component[c:e, a:d].astype(np.uint8)
        if supported.any():
            plausible.append((int(supported.sum()), k, supported))
    if not plausible:
        result['status'] = 'no_colored_basal_seed'
        return result
    # Match the existing QA's largest-component prior, without turning seed
    # geometry into an approved pool outline or a fluid-presence score.
    _, _, supported = max(plausible, key=lambda row: row[0])
    distance = cv2.distanceTransform(supported, cv2.DIST_L2, 5)
    seed_y, seed_x = np.unravel_index(np.argmax(distance), distance.shape)
    result.update(seed=[int(seed_x), int(seed_y)], seed_component_pixels=int(supported.sum()))
    containing = [p for p in candidates if p['clipped_box'][0] <= seed_x < p['clipped_box'][2]
                  and p['clipped_box'][1] <= seed_y < p['clipped_box'][3]]
    containing.sort(key=lambda p: area(p['clipped_box']))
    if not containing:
        result['status'] = 'seed_not_localized'
        return result
    if (len(containing) > 1 and area(containing[1]['clipped_box']) <=
            (1+POLICY['ambiguous_area_relative_tolerance'])*area(containing[0]['clipped_box'])):
        result['status'] = 'ambiguous_pool_localization'
        return result
    chosen = containing[0]
    box = chosen['clipped_box']
    points = [[int(seed_x), int(seed_y)]]; labels = [1]
    # Tentative negatives on neutral glass/label around the pool. Skip any
    # colored patch; the current detector and these camera priors can still err.
    anchors = [(x+.5*w, y+.4*h), (x+.05*w, seed_y), (r-.05*w, seed_y),
               (x+.2*w, y+.65*h), (r-.2*w, y+.65*h),
               ((box[0]+box[2])/2, max(y+2, box[1]-8))]
    for px, py in anchors:
        px, py = int(px), int(py)
        if not (1 <= px < rgb.shape[1]-1 and 1 <= py < rgb.shape[0]-1):
            continue
        if colored[py-1:py+2, px-1:px+2].any():
            continue
        if (seed_x-px)**2+(seed_y-py)**2 < 9 or [px, py] in points:
            continue
        points.append([px, py]); labels.append(0)
    result.update(guided_boxes=[box], points=[points], point_labels=[labels],
                  selected_prediction_index=chosen['prediction_index'], status='tentative_colored_pool_prompt')
    return result


class CausalPromptBridge:
    """Short-lived box prior, always requiring current detection and seed.

No SAM masks are propagated. Bridged prompts never refresh the original age.
"""
    def __init__(self):
        self.previous_vial = None
        self.last = None
        self.previous_seconds = None

    def update(self, policy, frame, seconds, motion):
        if self.previous_seconds is not None and seconds <= self.previous_seconds:
            raise ValueError('Prompt bridge requires strictly increasing source time')
        self.previous_seconds = seconds
        vial = policy['vial']
        reset = vial is None or motion is None or motion > BRIDGE_POLICY['carrier_motion_reset']
        if vial is not None and self.previous_vial is not None:
            intersection = containment(vial, self.previous_vial)*area(vial)
            overlap = intersection/max(1., area(vial)+area(self.previous_vial)-intersection)
            reset |= overlap < BRIDGE_POLICY['minimum_station_iou']
        if reset:
            self.last = None
        self.previous_vial = vial
        result = dict(boxes=[], points=[], labels=[], mode='no_current_pool_prompt',
                      prompt_source_frame=None, prompt_age_s=None)
        if vial is None or motion is None or motion > BRIDGE_POLICY['carrier_motion_reset'] or not policy['guided_boxes']:
            return result
        box = policy['guided_boxes'][0]
        if self.last is not None:
            age = seconds-self.last['seconds']
            x, y, r, b = vial; old = self.last['vial']; prior = self.last['box']
            mapped = [x+(prior[0]-old[0])/(old[2]-old[0])*(r-x),
                      y+(prior[1]-old[1])/(old[3]-old[1])*(b-y),
                      x+(prior[2]-old[0])/(old[2]-old[0])*(r-x),
                      y+(prior[3]-old[1])/(old[3]-old[1])*(b-y)]
            if area(box) > BRIDGE_POLICY['max_box_area_growth']*area(mapped):
                sx, sy = policy['seed']
                supported = mapped[0] <= sx < mapped[2] and mapped[1] <= sy < mapped[3]
                if age <= BRIDGE_POLICY['max_prompt_age_s']+1e-9 and supported:
                    return dict(boxes=[mapped], points=policy['points'], labels=policy['point_labels'],
                                mode='past_box_with_current_seed', prompt_source_frame=self.last['frame'],
                                prompt_age_s=age)
                result['mode'] = 'unresolved_box_jump'
                return result
        self.last = dict(box=list(box), vial=list(vial), frame=frame, seconds=seconds)
        return dict(boxes=[box], points=policy['points'], labels=policy['point_labels'],
                    mode='fresh_box_with_current_seed', prompt_source_frame=frame, prompt_age_s=0.)
