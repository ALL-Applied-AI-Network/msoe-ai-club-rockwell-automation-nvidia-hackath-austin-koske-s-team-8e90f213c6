"""Extract required public Ubuntu FFmpeg libraries into an experiment sysroot.

Uses the host's read-only apt metadata. No sudo, package install, dpkg database
change, system-library replacement or change to a teammate environment.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import apt
import apt_pkg

ROOTS=['libavformat60','libavfilter9','libavcodec60','libavutil58','libswscale7','libswresample4']


def main(a):
    cache=apt.Cache();selected={}
    def satisfies(version,relation,wanted):
        return not wanted or apt_pkg.check_dep(version,relation,wanted)
    def available(name,relation='',wanted=''):
        if name not in cache:return None
        package=cache[name]
        return next((v for v in package.versions if satisfies(v.version,relation,wanted)
            and any(o.site in ['ports.ubuntu.com','archive.ubuntu.com','security.ubuntu.com'] for o in v.origins)),None)
    def choose(name,relation='',wanted=''):
        if name in selected:
            if not satisfies(selected[name].version,relation,wanted):raise ValueError('Inconsistent private dependency')
            return
        version=available(name,relation,wanted)
        if version is None:raise ValueError('No public compatible package: '+name)
        selected[name]=version
        for group in version.dependencies:
            if group.rawtype not in ['Depends','PreDepends']:continue
            satisfied=False
            for dep in group.or_dependencies:
                if dep.name in cache and cache[dep.name].installed:
                    if satisfies(cache[dep.name].installed.version,dep.relation,dep.version):satisfied=True;break
                if dep.name in selected and satisfies(selected[dep.name].version,dep.relation,dep.version):satisfied=True;break
            if satisfied:continue
            dep=next((d for d in group.or_dependencies if available(d.name,d.relation,d.version)),None)
            if dep is None:raise ValueError('Unresolved private runtime dependency')
            choose(dep.name,dep.relation,dep.version)
    for name in ROOTS:choose(name)
    directory=a.root/'runtime-packages';directory.mkdir(exist_ok=False)
    specs=[name+'='+v.version for name,v in selected.items()]
    subprocess.run(['apt-get','download',*specs],cwd=directory,check=True)
    evidence=[]
    for path in sorted(directory.glob('*.deb')):
        subprocess.run(['dpkg-deb','-x',str(path),str(a.root/'sysroot')],check=True)
        evidence.append(dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (a.root/'runtime-packages.json').write_text(json.dumps(dict(packages=specs,archives=evidence,
        system_packages_modified=False),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);main(p.parse_args())
