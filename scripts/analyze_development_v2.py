"""Replay development policies; never report this as held-out confirmation."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time

import numpy as np
from branchlab.synthesis import _Ridge
from branchlab.value_aware import ValueAwareDiagnosticLearner
from branchlab.training import json_write


def score_predictions(name, predictions, rows, cfg, search_seed=None):
    records = {r['id']:r for r in rows if r['split']=='development'}
    results=[]
    for eid, action, probe, cost, route in zip(predictions['episode_ids'],predictions['chosen_repairs'],
            predictions['probe_ids'],predictions['probe_costs'],predictions['budget_costs']):
        row=records[eid]
        h=(cfg['audit_budget_forward_batches']-cfg['reserved_final_eval_cost']-route)//3
        results.append({'episode_id':eid,'seed':row['seed'],'condition':row['condition'],
            'action':action,'probe':probe,'probe_cost':cost,'budget_route':route,'total_updates':h,
            'development_loss':row['budget_losses'][str(route)][action],
            'unspent_units':cfg['audit_budget_forward_batches']-cfg['reserved_final_eval_cost']-3*h-cost})
    seeds=sorted({r['seed'] for r in results})
    per_seed=[{'seed':s,'development_loss':float(np.mean([r['development_loss'] for r in results if r['seed']==s]))}
              for s in seeds]
    return {'name':name,'search_seed':search_seed,'mean_development_loss':float(np.mean([r['development_loss'] for r in results])),
        'mean_probe_cost':float(np.mean([r['probe_cost'] for r in results])),
        'acquisition_count':sum(r['probe'] is not None for r in results),'episodes':len(results),
        'search_revealed_cells':predictions.get('search_revealed_cells',0),
        'search_replay_probe_cost':predictions.get('search_replay_probe_cost',0),
        'per_seed':per_seed,'decisions':results}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',default='artifacts/development_v2')
    parser.add_argument('--output',default='artifacts/development_v2_analysis')
    args=parser.parse_args()
    source,out=Path(args.data),Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Use a new analysis directory to preserve development attempts')
    rows=json.loads((source/'episodes.json').read_text())
    cfg=json.loads((source/'config.json').read_text())
    collection=json.loads((source/'collection.json').read_text())
    expected=sum(len(seeds) for seeds in cfg['seeds'].values())*len(cfg['conditions'])
    if len(rows)!=expected or len({r['id'] for r in rows})!=expected:
        raise ValueError('Incomplete or duplicate development episodes')
    started=time.perf_counter()
    methods=[]
    for name,probe,always,budget in [('logs_only',None,False,0),('logs_only_short',None,False,2),
        ('fixed_expert','lr_half:1:next_batch_loss',True,0),
        ('fixed_conditional_expert','lr_half:1:next_batch_loss',False,0)]:
        policy=ValueAwareDiagnosticLearner(search_budget_cells=cfg['search_budget_cells']).fit_fixed(
                    rows,probe_id=probe,always_probe=always,budget_cost=budget)
        json_write(out/'programs'/f'{name}.json',policy.state_dict())
        methods.append(score_predictions(name,policy.predict(rows,split='development'),rows,cfg))
    discovery=sorted([r for r in rows if r['split']=='discovery'],key=lambda r:r['id'])
    dev=sorted([r for r in rows if r['split']=='development'],key=lambda r:r['id'])
    x=lambda group:np.asarray([r['log_features'] for r in group])
    y=np.asarray([r['budget_losses']['0']+r['budget_losses']['2'] for r in discovery])
    model=_Ridge.fit(x(discovery),y,1.0)
    decisions=np.argmin(model.predict(x(dev)),axis=1)
    pred={'episode_ids':[r['id'] for r in dev],'chosen_repairs':(decisions%3).tolist(),
          'probe_ids':[None]*len(dev),'probe_costs':[0]*len(dev),'budget_costs':[0 if i<3 else 2 for i in decisions]}
    methods.append(score_predictions('logs_joint_action_horizon',pred,rows,cfg))
    for strategy in ('counterexample','random','enumeration','full_enumeration'):
        for seed in (cfg['search_seeds'] if strategy=='random' else [0]):
            policy=ValueAwareDiagnosticLearner(strategy='enumeration' if strategy=='full_enumeration' else strategy,
                search_budget_cells=100000 if strategy=='full_enumeration' else cfg['search_budget_cells'],seed=seed).fit(rows)
            json_write(out/'programs'/f'{strategy}-{seed}.json',policy.state_dict())
            methods.append(score_predictions(strategy,policy.predict(rows,split='development'),rows,cfg,seed))
    ce=next(m for m in methods if m['name']=='counterexample')
    comparisons=[{'baseline':m['name'],'search_seed':m['search_seed'],
                  'candidate_minus_baseline':ce['mean_development_loss']-m['mean_development_loss']}
                 for m in methods if m['name'] not in ('counterexample','full_enumeration')]
    gate='DEVELOPMENT_SIGNAL' if not collection['failures'] and all(r['candidate_minus_baseline'] < -0.001 for r in comparisons) else 'DEVELOPMENT_NOGO'
    summary={'status':'completed','confirmatory':False,'method':'shared-gradient cost-aware conditional diagnostics',
             'gate':gate,'config':cfg,'methods':methods,'comparisons':comparisons,
             'analysis_seconds':time.perf_counter()-started,'collection':collection,
             'claim':'Development only. No test text or confirmatory cohort has been evaluated.'}
    json_write(out/'summary.json',summary)
    print(json.dumps({'gate':gate,'methods':[{k:m[k] for k in ['name','search_seed','mean_development_loss','acquisition_count','search_revealed_cells']} for m in methods]},indent=2))


if __name__=='__main__':
    main()
