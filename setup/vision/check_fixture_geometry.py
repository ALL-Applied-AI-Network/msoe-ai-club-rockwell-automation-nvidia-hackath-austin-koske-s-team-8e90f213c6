"""CPU-only diagnostic grid stability; never approves mapping or changes crops."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())


def main():
    import cv2
    import numpy as np
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    registration=read(a.batch/'registration.json');samples=read(a.batch/'samples.json')
    anchor=next(r for r in samples if r['source_sha256']==registration['anchor_sha256'])
    gray_anchor=cv2.cvtColor(cv2.imread(str(a.batch/anchor['image'])),cv2.COLOR_BGR2GRAY)
    targets=np.array([registration['aperture_points'][f'G{i:02d}'] for i in range(1,31)],dtype=np.float32)

    def estimate(gray):
        window=(slice(140,330),slice(210,395));template=gray_anchor[window].astype(np.float32);current=gray[window].astype(np.float32)
        shift,response=cv2.phaseCorrelate(template,current)
        circles=cv2.HoughCircles(gray[window],cv2.HOUGH_GRADIENT,1,22,param1=80,param2=15,minRadius=8,maxRadius=12)
        if circles is None:return dict(status='unavailable',reason='No usable circle candidates',requires_alignment_review=True)
        candidates=circles[0,:,:2]+[210,140];solutions=[]
        for dx,dy in [(0.,0.),shift]:
            pairs=[];taken=set()
            distances=np.linalg.norm((targets+[dx,dy])[:,None,:]-candidates[None,:,:],axis=2)
            for target_index in np.argsort(distances.min(axis=1)):
                order=np.argsort(distances[target_index]);match=next((int(i) for i in order if int(i) not in taken and distances[target_index,i]<=6),None)
                if match is not None:taken.add(match);pairs.append((int(target_index),match))
            if len(pairs)<8:continue
            source=np.array([targets[i] for i,j in pairs],dtype=np.float32);dest=np.array([candidates[j] for i,j in pairs],dtype=np.float32)
            transform,inliers=cv2.estimateAffinePartial2D(source,dest,method=cv2.RANSAC,ransacReprojThreshold=2.5,maxIters=2000,confidence=.99)
            if transform is None:continue
            selected=inliers.ravel().astype(bool);projected=np.column_stack([source,np.ones(len(source))])@transform.T
            residual=np.linalg.norm(projected-dest,axis=1)[selected]
            displacement=np.column_stack([targets,np.ones(30)])@transform.T-targets
            median_shift=np.median(displacement,axis=0)
            solutions.append(dict(status='diagnostic_only',candidate_circles=len(candidates),matched_grid_points=int(selected.sum()),
                median_grid_displacement_px=median_shift.tolist(),median_inlier_residual_px=float(np.median(residual)),
                p95_inlier_residual_px=float(np.percentile(residual,95)),phase_shift_px=list(shift),phase_response=float(response),
                transform=transform.tolist(),mapping_approved=False,
                requires_alignment_review=bool(selected.sum()<12 or np.max(np.abs(median_shift))>3 or np.median(residual)>2)))
        return max(solutions,key=lambda r:(r['matched_grid_points'],-r['median_inlier_residual_px'])) if solutions else dict(status='unavailable',reason='Insufficient consistent circle correspondences',requires_alignment_review=True)

    rows=[]
    for sample in samples:
        source=a.batch/sample['image']
        if sha(source)!=sample['source_sha256']:raise ValueError('Source image changed')
        gray=cv2.cvtColor(cv2.imread(str(source)),cv2.COLOR_BGR2GRAY)
        rows.append(dict(sample_id=sample['sample_id'],run_id=sample['run_id'],source_sha256=sample['source_sha256'],result=estimate(gray)))
    controls=[]
    for dx,dy in [(8,0),(0,8)]:
        changed=cv2.warpAffine(gray_anchor,np.array([[1,0,dx],[0,1,dy]],dtype=np.float32),(640,480),borderMode=cv2.BORDER_REFLECT)
        result=estimate(changed)
        controls.append(dict(artificial_shift_px=[dx,dy],result=result,passed_review_trigger=result['requires_alignment_review'],
            synthetic_perturbation=True,physical_validation=False))
    if not all(r['passed_review_trigger'] for r in controls):raise AssertionError('Known grid displacement was not flagged for review')
    report=dict(source_stills=len(rows),results=rows,controlled_image_shifts=controls,
        engineering_review_rule='Flag fewer than12 consistent matches, median displacement over3px or median residual over2px; commissioning diagnostic, not calibrated acceptance.',
        mapping_approved=False,crops_changed=False,model_predictions_changed=False,training=False,run5_used=False,
        limits=['Circular caps/rims can also match grid points; correspondences are not occupancy labels.',
            'Repeated aperture patterns and occlusion can make matching ambiguous. Numerical stability cannot approve a target map.',
            'Digitally shifted anchor controls test the diagnostic response, not real camera robustness.'])
    (a.out/'geometry-checks.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(stills=16,review_flags=sum(r['result']['requires_alignment_review'] for r in rows),shift_controls_passed=2)),flush=True)


if __name__=='__main__':main()
