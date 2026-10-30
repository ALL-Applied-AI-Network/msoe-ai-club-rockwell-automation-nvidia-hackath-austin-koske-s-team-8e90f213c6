"""One-command evidence-bound evaluation status; score only supplied human exports.

No exports means unavailable accuracy, never an implicit all-correct score.
Existing scorers retain source/packet checks and demonstration rejection.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from .slot_review import score as score_slots,validate_packet


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def pending(tasks,priority=None):
    return dict(status='awaiting_human_review',total_tasks=tasks,priority_tasks=priority,reviewed_targets=0,
                accuracy=None,metrics_available=False,reason='No eligible human export supplied; predictions are not evaluation truth')


def build(args):
    root=args.results;iterations=root/'synthetic-iterations-20261022'
    slot_packet=root/'dataset1-real-transfer-20261022/review-v1'
    pool_packet=iterations/'visible-pool-review-v2'
    slot=read(slot_packet/'packet.json');validate_packet(slot)
    pool=read(pool_packet/'manifest.json')
    out=args.out;out.mkdir(parents=True,exist_ok=False)
    slot_result=pending(len(slot['tasks']),sum(t['priority'] for t in slot['tasks']))
    pool_result=pending(pool['frames'])
    used={}
    if args.slot_review:
        review=read(args.slot_review);result=score_slots(slot,review)
        used['slot_review_sha256']=sha(args.slot_review)
        write(out/'fixture-reviewed-metrics.json',result)
        slot_result=dict(status='reviewed_subset' if result['scored_targets'] else 'no_scorable_labels',total_tasks=len(slot['tasks']),
            reviewed_targets=result['human_decisions'],scored_targets=result['scored_targets'],accuracy=result['accuracy'],
            metrics_available=result['scored_targets']>0,report='fixture-reviewed-metrics.json',
            reason='Only confirmed mapping/visibility/definite labels are scored; preserve exclusions and unsupported-class misses')
    if args.pool_review:
        review=read(args.pool_review)
        if review.get('packet_id')!=pool['packet_id']:raise ValueError('Pool review packet mismatch')
        # Invoke the established multi-model scorer; no duplicate labels or scorer logic.
        from .score_sam_temporal_review import score as score_pool
        score_pool(SimpleNamespace(comparison=iterations/'gb10-results/sam-comparison-verified-v2',
            source_probe=iterations/'gb10-results/mask-probe-v1',refinement=iterations/'gb10-results/temporal-sam-refinement-v1',
            packet=pool_packet,review=args.pool_review,out=out/'liquid-reviewed'))
        result=read(out/'liquid-reviewed/comparison.json');used['pool_review_sha256']=sha(args.pool_review)
        pool_result=dict(status='reviewed_subset',total_tasks=pool['frames'],reviewed_targets=result['reviewed'],
            scored_targets=result['scored'],uncertain=result['uncertain'],metrics_available=True,report='liquid-reviewed/comparison.json',
            reason='Presence precision/recall, visible-pool IoU/Dice and reviewed onset intervals; six inspected development clips')
    report=dict(schema='two-station-evaluation-status/v1',fixture=slot_result,liquid=pool_result,
        slot_packet_id=slot['packet_id'],pool_packet_id=pool['packet_id'],exports_used=used,
        training_eligible=False,approved_labels_created=False,model_or_threshold_changes=False,
        limits=['Human exports are attributed input, not authenticated reviewer identities.',
            'Coverage, uncertainty and reviewed positive counts must accompany accuracy.',
            'Missing-cap recall cannot be established by only reviewed normal/empty examples.',
            'Reviews are evaluation-only; no weights/checkpoints/thresholds are selected here.'])
    write(out/'evaluation-status.json',report)
    (out/'README.md').write_text(
        '# Evaluation status\n\n'
        f'Liquid: {pool_result["status"]}; {pool_result["reviewed_targets"]}/{pool_result["total_tasks"]} reviewed targets.\n'
        f'Fixture: {slot_result["status"]}; {slot_result["reviewed_targets"]}/{slot_result["total_tasks"]} reviewed targets.\n\n'
        'No human export was fabricated. Pending data is unavailable, never counted as correct.\n'
        'Run this command again with --slot-review and/or --pool-review and a fresh --out after receiving actual exports.\n'
        'Existing source-matched scorers are used without changing their masks, labels, confidence or QA thresholds.\n')
    print(json.dumps(report),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--slot-review',type=Path);p.add_argument('--pool-review',type=Path)
    build(p.parse_args())
