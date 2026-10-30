"""Read-only CSI recording inventory with exact decoded-frame/native-clock traces.

Does not infer labels, align station clocks, run models, or connect to cameras.
Use the existing data/vision Python environment containing PyAV.
"""
import argparse
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def inspect(source, out):
    import av
    source, out = source.resolve(), out.resolve()
    source_root = source if source.is_dir() else source.parent
    if not source.exists() or out.exists() or out.is_relative_to(source_root):
        raise ValueError('Use an existing source file/directory and fresh output outside its directory')
    files = [source] if source.is_file() else sorted(p for p in source.rglob('*') if p.is_file())
    if not files:
        raise ValueError('No source files')
    out.mkdir(parents=True)
    shutil.copy2(__file__, out/'executed-inspector.py')
    records = []
    for number, path in enumerate(files):
        digest = sha(path)
        record = dict(relative_path=path.relative_to(source_root).as_posix(), bytes=path.stat().st_size,
                      sha256=digest, kind='file')
        if path.suffix.lower() in ('.mp4', '.mov', '.mkv', '.avi'):
            trace = out/f'video-{number:04d}-native-frames.csv'
            count = missing = nonincreasing = 0
            last = None
            with av.open(str(path)) as c, trace.open('w', newline='', encoding='utf-8') as f:
                stream = c.streams.video[0]
                writer = csv.writer(f)
                writer.writerow(['decoded_frame', 'pts', 'time_base', 'pts_seconds', 'width', 'height'])
                for index, frame in enumerate(c.decode(stream)):
                    count += 1
                    native = Fraction(frame.pts)*frame.time_base if frame.pts is not None else None
                    missing += int(native is None)
                    if native is not None:
                        nonincreasing += int(last is not None and native <= last)
                        last = native
                    writer.writerow([index, frame.pts, str(frame.time_base),
                                     float(native) if native is not None else None, frame.width, frame.height])
                record.update(kind='video', decoded_frames=count, missing_pts=missing,
                              nonincreasing_pts=nonincreasing,
                              native_clock_usable=bool(count and not missing and not nonincreasing),
                              codec=stream.codec_context.name,
                              nominal_rate_metadata=str(stream.average_rate),
                              native_frame_trace=trace.name, trace_sha256=sha(trace))
        if sha(path) != digest:
            raise ValueError('Source changed during inspection: '+str(path))
        records.append(record)
    result = dict(schema='csi-capture-inventory/v1', source_files=records,
                  hardware_connections=0, source_modified=False,
                  station_clock_mapping='unreviewed', physical_vial_identity='unreviewed',
                  approved_outcomes=0, accuracy=None,
                  limits='Decoded frame/PTS inventory is not station clock alignment or label approval.')
    (out/'inventory.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    (out/'completed.json').write_text(json.dumps(dict(
        files=len(records), videos=sum(r['kind'] == 'video' for r in records),
        inventory_sha256=sha(out/'inventory.json'), helper_sha256=sha(out/'executed-inspector.py')),
        indent=2)+'\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(inspect(args.source, args.out), indent=2))
