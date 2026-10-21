"""Add source/launcher/docs to a fresh case batch, without model calls."""
import argparse
from pathlib import Path
import shutil
import zipfile
import json
from .core import sha


def package(root,readme=None):
    root=Path(root).resolve();source=Path(__file__).parent
    target=root/'annotation_review';target.mkdir(exist_ok=True)
    for path in source.iterdir():
        if path.is_file() and path.suffix in ('.py','.md','.toml'):
            shutil.copyfile(path,target/path.name)
    for directory in ('static','tests','verification'):
        if not (source/directory).exists():continue
        (target/directory).mkdir(exist_ok=True)
        for path in (source/directory).iterdir():
            if path.is_file() and path.suffix in ('.py','.js','.css','.html','.json','.md'):
                shutil.copyfile(path,target/directory/path.name)
    shutil.copyfile(source/'launch_review.py',root/'launch_review.py')
    shutil.copyfile(Path(readme) if readme else source/'HANDOFF.md',root/'README.md')
    hashes={str(p.relative_to(root)).replace('\\','/'):sha(p) for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='package-hashes.json'}
    (root/'package-hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
    archive=root.parent/(root.name+'.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for relative in (*hashes,'package-hashes.json'):
            z.write(root/relative,relative)
    return archive


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',type=Path,required=True)
    p.add_argument('--readme',type=Path,help='Use this task-specific handoff at the package root')
    args=p.parse_args();print(package(args.batch,args.readme))


if __name__=='__main__':main()
