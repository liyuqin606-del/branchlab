"""Descriptive constant-action controls for the development lifetime audit.

Added after the v2 NOGO, before inspecting the lifetime replay. No independent
confirmation or automated promotion. Retain all fixed controls; learned constant
choices use discovery seeds only. No text or model execution is performed.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np


def analyze(rows,curves,horizons=(4,8,16,32,64)):
    if {r['split'] for r in rows}!={'discovery','development'}:raise ValueError('Development only')
    if {r['id'] for r in rows}!=set(curves):raise ValueError('Incomplete curves')
    result=[]
    for h in horizons:
        losses={s:np.array([[curves[r['id']][a][str(t)] for t in (h,h-1) for a in ('keep','lr_half','momentum_zero')] for r in rows if r['split']==s]) for s in ('discovery','development')}
        choices={f'constant_{a}_h{t}':i for i,(t,a) in enumerate((t,a) for t in (h,h-1) for a in ('keep','lr_half','momentum_zero'))}
        # Macro-average training seeds, not correlated state replicates.
        discovery=[r for r in rows if r['split']=='discovery'];dev=[r for r in rows if r['split']=='development']
        mean=np.mean([losses['discovery'][[r['seed']==s for r in discovery]].mean(0) for s in sorted({r['seed'] for r in discovery})],0)
        choices['discovery_selected_constant_full']=int(mean[:3].argmin())
        choices['discovery_selected_constant_joint']=int(mean.argmin())
        records=[]
        for name,i in choices.items():
            per_seed=[{'seed':s,'loss':float(losses['development'][[r['seed']==s for r in dev],i].mean())} for s in sorted({r['seed'] for r in dev})]
            records.append({'name':name,'action':('keep','lr_half','momentum_zero')[i%3],'updates':h if i<3 else h-1,'probe_cost':0,'mean_development_loss':float(np.mean([p['loss'] for p in per_seed])),'per_seed':per_seed})
        result.append({'horizon':h,'controls':records})
    return {'scope':__doc__,'confirmatory':False,'configurations':result}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--curves',type=Path,default=Path('artifacts/repair_lifetime_v2/curves.json'));p.add_argument('--rows',type=Path,default=Path('artifacts/development_v2/episodes.json'));p.add_argument('--output',type=Path,default=Path('artifacts/repair_lifetime_v2_controls.json'));a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if json.loads((a.curves.parent/'collection.json').read_text())['status']!='completed' or not json.loads((a.curves.parent/'parity.json').read_text())['passed']:raise ValueError('Incomplete or invalid lifetime replay')
    result=analyze(json.loads(a.rows.read_text()),json.loads(a.curves.read_text()))
    result['source_sha256']={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in [('curves',a.curves),('episodes',a.rows),('analysis',Path(__file__))]}
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for c in result['configurations']:print(c['horizon'],[(r['name'],round(r['mean_development_loss'],6)) for r in c['controls']])


if __name__=='__main__':main()
