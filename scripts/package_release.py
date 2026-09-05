"""Build public release archives from an explicit allowlist of measured artifacts."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return {'bytes': path.stat().st_size, 'sha256': h.hexdigest()}


def archive(path, files):
    with path.open('wb') as destination, gzip.GzipFile(filename='', fileobj=destination, mode='wb', mtime=0, compresslevel=1) as compressed:
        with tarfile.open(fileobj=compressed, mode='w') as tar:
            for source, name in sorted(files, key=lambda item: item[1]):
                if not source.is_file():
                    raise FileNotFoundError(source)
                info = tar.gettarinfo(str(source), arcname=name)
                info.uid = info.gid = 0
                info.uname = info.gname = ''
                info.mtime = 0
                info.mode = 0o644
                with source.open('rb') as stream:
                    tar.addfile(info, stream)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, default=Path('artifacts'))
    parser.add_argument('--output', type=Path, default=Path('dist'))
    args = parser.parse_args()
    a, out = args.artifacts, args.output
    out.mkdir(parents=True, exist_ok=True)
    summary = json.loads((a/'pilot/summary.json').read_text())
    if summary['status'] != 'completed':
        raise ValueError('Only completed pilots can be packaged')
    paths = []
    inference_names = ['model.pt','tokenizer.json','run.json','history.json','benchmark.json','sample.txt']
    files = [(a/'showcase'/name, f'branchlab-19m/{name}') for name in inference_names]
    files += [(Path('docs/MODEL_CARD.md'),'branchlab-19m/MODEL_CARD.md'),(Path('LICENSE'),'branchlab-19m/LICENSE'),
              (Path('THIRD_PARTY.md'),'branchlab-19m/THIRD_PARTY.md')]
    p = out/'branchlab-v0.1.0-model.tar.gz'
    archive(p, files)
    paths.append(p)
    files = [(a/'showcase/checkpoint.pt','checkpoints/showcase.pt')]
    files += [(p, f'checkpoints/pilot/{p.name}') for p in sorted((a/'pilot/checkpoints').glob('*.pt'))]
    if len(files) != 1 + sum(len(seeds) for seeds in summary['config']['seeds'].values()):
        raise ValueError('Missing baseline checkpoints')
    files += [(Path('docs/REPRODUCIBILITY.md'),'checkpoints/REPRODUCIBILITY.md'),
              (a/'tokenizer.json','checkpoints/tokenizer.json'),(a/'preparation.json','checkpoints/preparation.json')]
    p = out/'branchlab-v0.1.0-checkpoints.tar.gz'
    archive(p, files)
    paths.append(p)
    files = [(p, f"artifacts/pilot/{p.relative_to(a / 'pilot')}")
             for p in sorted((a/'pilot').rglob('*')) if p.is_file() and p.suffix != '.pt']
    files += [(p, str(p)) for root in [Path('reports'), Path('configs'), Path('protocols'), Path('sources')]
              for p in sorted(root.rglob('*')) if p.is_file()]
    files += [(a/'preparation.json','artifacts/preparation.json'),(a/'tokenizer.json','artifacts/tokenizer.json'),
              (a/'showcase-training.log','artifacts/showcase-training.log'),
              (a/'pilot-training.log','artifacts/pilot-training.log'),
              (a/'test-results.txt','artifacts/test-results.txt')]
    p = out/'branchlab-v0.1.0-evidence.tar.gz'
    archive(p, files)
    paths.append(p)
    paths += sorted(out.glob('branchlab-*.whl')) + sorted(out.glob('branchlab-0.1.0.tar.gz'))
    manifest = {'created_utc':datetime.now(timezone.utc).isoformat(),
        'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'gate':summary['gate']['status'], 'assets':{p.name:digest(p) for p in paths},
        'scope':'Allowlisted code/model/evidence artifacts; no source corpus or local account files'}
    (out/'release_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    paths.append(out/'release_manifest.json')
    (out/'SHA256SUMS').write_text(''.join(f'{digest(p)["sha256"]}  {p.name}\n' for p in sorted(paths)))
    print(json.dumps(manifest,indent=2))


if __name__ == '__main__':
    main()
