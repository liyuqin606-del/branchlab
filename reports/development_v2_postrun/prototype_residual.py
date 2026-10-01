"""Post-run development-only residual-head feasibility experiment.

All settings and every probe are retained. This script is not the frozen method,
search implementation or confirmation entrypoint. No test text or test labels.
"""
import json
from pathlib import Path
import numpy as np
from branchlab.synthesis import _Ridge


def labels(rows,cost):
    y=np.array([r['budget_losses'][str(cost)] for r in rows]);return y-y[:,:1]


def free(rows,cost,alpha):
    return _Ridge.fit(np.array([r['log_features'] for r in rows]),labels(rows,cost),alpha)


def x(rows):return np.array([r['log_features'] for r in rows])
def z(rows,p):return np.array([r['probe_features'][p] for r in rows])
def folds(rows):
    for seed in sorted({r['seed'] for r in rows}):
        yield [r for r in rows if r['seed']!=seed],[i for i,r in enumerate(rows) if r['seed']==seed]


def residual_head(rows,p,a,ra):
    pred=np.empty((len(rows),3))
    for tr,ids in folds(rows):pred[ids]=free(tr,2,a).predict(x([rows[i] for i in ids]))
    residual=labels(rows,2)-pred
    return free(rows,2,a),_Ridge.fit(z(rows,p),residual,ra)


def head_predict(h,rows,p):return h[0].predict(x(rows))+h[1].predict(z(rows,p))
def gate_x(p0,p2):
    sort=np.sort(p0,axis=1)
    # Pairwise output centering removes a common per-row scale. The second
    # feature is a margin at the shorter horizon, not an absolute horizon gap.
    sort2=np.sort(p2,axis=1)
    return np.column_stack((sort[:,1]-sort[:,0],sort2[:,1]-sort2[:,0]))


def run(rows):
    d=[r for r in rows if r['split']=='discovery'];v=[r for r in rows if r['split']=='development']
    y0=np.array([r['budget_losses']['0'] for r in v]);y2=np.array([r['budget_losses']['2'] for r in v]);allrows=[]
    for a in [1,10]:
      b0=free(d,0,a).predict(x(v));b2=free(d,2,a).predict(x(v));i=np.arange(len(v));l0=y0[i,b0.argmin(1)];l2=y2[i,b2.argmin(1)]
      for ra in [1,10,100]:
       for p in sorted(d[0]['probe_features']):
        p0=np.empty((len(d),3));p2=np.empty_like(p0);pp=np.empty_like(p0)
        for tr,ids in folds(d):
            val=[d[j] for j in ids]
            p0[ids]=free(tr,0,a).predict(x(val));p2[ids]=free(tr,2,a).predict(x(val));pp[ids]=head_predict(residual_head(tr,p,a,ra),val,p)
        d0=np.array([r['budget_losses']['0'] for r in d]);d2=np.array([r['budget_losses']['2'] for r in d]);j=np.arange(len(d))
        gain=d0[j,p0.argmin(1)]-d2[j,pp.argmin(1)]
        gate=_Ridge.fit(gate_x(p0,p2),gain[:,None],10)
        use=gate.predict(gate_x(b0,b2))[:,0]>0
        paid=head_predict(residual_head(d,p,a,ra),v,p);lp=y2[i,paid.argmin(1)];lc=np.where(use,lp,l0)
        allrows.append({'alpha':a,'residual_alpha':ra,'probe':p,'free':float(l0.mean()),'free_short':float(l2.mean()),'paid':float(lp.mean()),'conditional':float(lc.mean()),'queries':int(use.sum()),'oof_net_gain':float(gain.mean()),'per_seed':[{'seed':s,'free':float(l0[[r['seed']==s for r in v]].mean()),'conditional':float(lc[[r['seed']==s for r in v]].mean())} for s in sorted({r['seed'] for r in v})]})
    return {'scope':'Post-run development only. No candidate search budget or confirmation claim. Nested whole-seed out-of-fold residual/gain fits. All48settings retained.','settings':allrows}

if __name__=='__main__':
    out=Path('notes/goal_v2/prototype_residual_results.json')
    if out.exists():raise FileExistsError(out)
    result=run(json.loads(Path('artifacts/development_v2/episodes.json').read_text()));out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(sorted(result['settings'],key=lambda r:r['conditional'])[:3],indent=2))
