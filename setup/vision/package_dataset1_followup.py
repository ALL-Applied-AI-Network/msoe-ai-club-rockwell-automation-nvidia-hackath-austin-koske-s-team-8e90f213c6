"""Package completed experiment evidence, excluding linked source assets and caches."""
import argparse
from pathlib import Path
import tarfile

from dataset1_fiveclass import read,sha,write


def package(root):
    read(root/'completed.json')
    if (root/'completed-artifacts-v1.tar.gz').exists():raise ValueError('Existing bundle; verify it rather than replacing')
    files=[p for p in root.rglob('*') if p.is_file() and not p.is_symlink() and 'input' not in p.relative_to(root).parts
        and '__pycache__' not in p.relative_to(root).parts and p.suffix in ('.py','.json','.jsonl','.png','.pt','.log')]
    manifest={p.relative_to(root).as_posix():sha(p) for p in files};write(root/'artifact-manifest.json',manifest);files.append(root/'artifact-manifest.json')
    archive=root/'completed-artifacts-v1.tar.gz'
    with tarfile.open(archive,'x:gz') as tar:
        for path in files:tar.add(path,arcname=path.relative_to(root).as_posix(),recursive=False)
    (root/'completed-artifacts-v1.tar.gz.sha256').write_text(sha(archive)+'  completed-artifacts-v1.tar.gz\n')
    print(dict(files=len(files),archive_sha256=sha(archive),bytes=archive.stat().st_size))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args();package(a.root)
