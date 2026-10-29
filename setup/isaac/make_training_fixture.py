"""Seeded camera-fixed training pilot. No real footage or annotations are read."""
import argparse
import colorsys
from copy import deepcopy
import json
from pathlib import Path
import random

from box_annotations import validate_fixture, write_json


def make(count=768, seed=19):
    if count < 24 or count % 8:
        raise ValueError('Use a multiple of eight, at least 24')
    base = json.loads(Path(__file__).with_name('box_fixture.json').read_text(encoding='utf-8'))
    rows = deepcopy(base['cases'])
    rng = random.Random(seed)
    kinds = ['empty']*(count//8-1) + ['stream-only']*(count//8-1) + ['fill']*(count*3//4-10)
    rng.shuffle(kinds)
    for i, kind in enumerate(kinds):
        palette = rng.choices(['colored', 'pale', 'clear-proxy'], weights=[.4,.4,.2])[0]
        if palette == 'colored':
            rgb = colorsys.hsv_to_rgb(rng.random(), rng.uniform(.65,.95), rng.uniform(.18,.65))
            opacity = rng.uniform(.45,.8)
        elif palette == 'pale':
            rgb = colorsys.hsv_to_rgb(rng.random(), rng.uniform(.06,.3), rng.uniform(.65,.95))
            opacity = rng.uniform(.25,.5)
        else:
            rgb = [rng.uniform(.86,.96)]*3
            opacity = rng.uniform(.10,.20)
        level = 0 if kind != 'fill' else (rng.uniform(.02,.12) if rng.random()<.5 else rng.uniform(.12,.98))
        mode = 'short' if kind == 'stream-only' else ('off' if kind == 'empty' else rng.choices(['continuous','off'],[.7,.3])[0])
        rows.append(dict(id=f'syn-{i:06d}', level=level, stream=mode, rgb=list(rgb), opacity=opacity,
            light_scale=rng.uniform(.65,1.35), roughness=rng.uniform(.06,.30),
            stream_radius_m=rng.uniform(.0008,.0023), palette=palette, kind=kind))
    assert len(rows) == count
    result = dict(base, purpose='Provisional mixed-data detector experiment authorized 2026-10-19',
        seed=seed, real_data_inputs=[], recipe_or_camera_fitting=False,
        nuisance_policy='Fixed supplied FillCam; vary proxy color, opacity, roughness, level, light intensity and stream radius', cases=rows)
    validate_fixture(result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--count', type=int, default=768)
    parser.add_argument('--seed', type=int, default=19)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Use a fresh fixture path')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, make(args.count, args.seed))
    print(f'Prepared {args.count} cases; seed {args.seed}; no real input')
