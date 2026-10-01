"""Evaluate the fixed, DEVELOPMENT-only four-step calibration observation screen."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from branchlab.synthesis import _Ridge

ACTIONS=('keep','lr_half','momentum_zero')
REPO_ROOT = Path(__file__).resolve().parents[1]


def _validate_coverage(rows, config):
    """Check the declared sampling units, including every condition per seed."""
    seeds = config['seeds']
    conditions = config['conditions']
    if set(seeds) != {'discovery', 'development'} or not conditions or len(conditions) != len(set(conditions)):
        raise ValueError('Invalid development seed/condition configuration')
    seen = set()
    for split, values in seeds.items():
        if not values or any(type(seed) is not int for seed in values) or len(values) != len(set(values)) or seen.intersection(values):
            raise ValueError('Discovery/development seeds must be nonempty, unique and disjoint')
        seen.update(values)
    expected = {(split, seed, condition) for split, values in seeds.items()
                for seed in values for condition in conditions}
    actual = [(r['split'], r['seed'], r['condition']) for r in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('Missing or duplicated configured seed/condition state')
    if any(r['id'] != f"{r['split']}-{r['seed']:03d}-{r['condition']}" for r in rows):
        raise ValueError('Episode identity disagrees with configured sampling unit')


def _validate_receipts(rows, curves, observations, config, collection, parity, verification):
    """Admit only a complete physical replay, separately from logical acquisition."""
    _validate_coverage(rows, config)
    constants = {'audit_budget_forward_batches': 224, 'reserved_final_eval_cost': 32,
                 'horizons': [58, 61, 64], 'actions': list(ACTIONS),
                 'prefix_steps': 4, 'prefix_len': 4, 'calibration_positions': [65, 81, 97, 113],
                 'pair_extra_cost': 17, 'pair_retained_updates': 58,
                 'prefix_only_extra_cost': 9, 'prefix_only_retained_updates': 61,
                 'eval_split': 'dev', 'confirmatory': False}
    if any(config.get(key) != value for key, value in constants.items()):
        raise ValueError('Collection config differs from frozen prefix/budget design')
    n = len(rows)
    if (collection.get('status') != 'completed' or collection.get('confirmatory') is not False
            or collection.get('failures') or collection.get('source_episode_count') != n
            or collection.get('completed_episodes') != n):
        raise ValueError('Incomplete or invalid collection receipt')
    expected_counts = {'gradient_batches': n * 190, 'candidate_optimizer_steps': n * 3,
                       'ordinary_optimizer_steps': n * 189, 'calibration_forward_batches': n * 12,
                       'development_eval_forward_batches': n * 9 * config['eval_batches']}
    attempts = {'gradient_batches': 'gradient_batch_attempts',
                'candidate_optimizer_steps': 'candidate_optimizer_attempts',
                'ordinary_optimizer_steps': 'ordinary_optimizer_attempts',
                'calibration_forward_batches': 'calibration_forward_attempts',
                'development_eval_forward_batches': 'development_eval_forward_attempts'}
    if (collection.get('expected_counts') != expected_counts
            or any(collection.get(key) != value or collection.get(attempts[key]) != value
                   for key, value in expected_counts.items())
            or collection.get('shared_gradient_batches') != n
            or collection.get('baseline_retraining_updates') != 0 or collection.get('test_forward_batches') != 0):
        raise ValueError('Physical replay counts are incomplete or inconsistent')
    units = 3 * expected_counts['gradient_batches'] + expected_counts['calibration_forward_batches'] + expected_counts['development_eval_forward_batches']
    tokens = expected_counts['gradient_batches'] * config['batch_size'] * config['seq_len']
    if (collection.get('approx_forward_batch_units') != units or collection.get('trained_tokens') != tokens
            or collection.get('attempted_training_tokens') != tokens):
        raise ValueError('Physical replay token/unit totals disagree with counts')
    expected_pairs = {(r['id'], a) for r in rows for a in ACTIONS}
    row_lookup = {r['id']: r for r in rows}
    for name, receipt, records_key, count_key, expected_key in (
            ('parity', parity, 'comparisons', 'comparison_count', 'expected_comparisons'),
            ('prefix verification', verification, 'records', 'count', 'expected_count')):
        records = receipt.get(records_key, [])
        keys = [(r['episode_id'], r['action']) for r in records]
        if (receipt.get('status') != 'completed' or receipt.get('passed') is not True
                or receipt.get(count_key) != len(expected_pairs) or receipt.get(expected_key) != len(expected_pairs)
                or len(keys) != len(expected_pairs) or set(keys) != expected_pairs
                or any(r.get('passed') is not True for r in records)):
            raise ValueError(f'Incomplete or invalid {name} receipt coverage')
    tolerance = parity.get('tolerance', float('nan'))
    if not np.isfinite(tolerance) or not 0 <= tolerance <= 1e-6 or tolerance != config.get('parity_tolerance'):
        raise ValueError('Invalid endpoint parity tolerance')
    for item in parity['comparisons']:
        eid, action = item['episode_id'], item['action']
        actual = curves[eid][action]['64']
        expected = row_lookup[eid]['budget_losses']['0'][ACTIONS.index(action)]
        difference = abs(actual - expected)
        if (item['horizon'] != 64 or item['actual'] != actual or item['expected'] != expected
                or not np.isfinite(difference) or difference > tolerance
                or item['abs_difference'] != difference or item['exact'] != (actual == expected)):
            raise ValueError('Endpoint parity receipt disagrees with replay/source values')
    if (parity.get('max_abs_difference') != max(r['abs_difference'] for r in parity['comparisons'])
            or parity.get('all_exact') != all(r['exact'] for r in parity['comparisons'])):
        raise ValueError('Endpoint parity aggregate disagrees with records')
    unchanged = ('model', 'optimizer', 'stream', 'step', 'scheduler', 'model_training')
    for item in verification['records']:
        before, after, restored = (item[k] for k in ('before_calibration', 'after_calibration', 'after_restore'))
        if (item.get('calibration_preserved_training_state') is not True or item.get('restored_exactly') is not True
                or item['step'] != item['expected_step'] or item['optimizer_steps'] != [item['step']]
                or before != restored
                or any(before['components'][key] != after['components'][key] for key in unchanged)
                or observations[item['episode_id']][item['action']]['prefix_state_sha256'] != before['sha256']):
            raise ValueError('Prefix restoration receipt disagrees with committed state')


def evaluate_screen(rows,curves,observations,proposer,config=None):
    """Score the frozen screen; only discovery outcomes fit learned controllers.

    Calibration values for the unproposed third action are never acquired.
    Physical completion/provenance admission is additionally mandatory in CLI.
    """
    if config is not None:
        _validate_coverage(rows, config)
    ids=[r['id'] for r in rows]
    if len(ids)!=len(set(ids)) or set(ids)!=set(curves) or set(ids)!=set(observations):
        raise ValueError('Missing or duplicated state evidence')
    if {r['split'] for r in rows}!={'discovery','development'}:
        raise ValueError('This screen accepts development data only')
    if set(proposer['proposals'])!=set(ids):raise ValueError('Proposer coverage differs')
    disc=[i for i,r in enumerate(rows) if r['split']=='discovery']
    dev=[i for i,r in enumerate(rows) if r['split']=='development']
    if {rows[i]['seed'] for i in disc}&{rows[i]['seed'] for i in dev}:raise ValueError('Seed overlap')
    if set(proposer['discovery_seeds']) != {rows[i]['seed'] for i in disc}:raise ValueError('Frozen proposer discovery seeds changed')
    x=np.array([r['log_features'] for r in rows], dtype=float)
    if x.ndim != 2 or not x.shape[1] or not np.isfinite(x).all():raise ValueError('Invalid free log features')
    order=np.argsort(_Ridge.from_state_dict(proposer['model']).predict(x),axis=1,kind='stable')
    pair=np.array([proposer['proposals'][eid]['pair'] for eid in ids])
    if pair.dtype.kind not in 'iu' or not np.array_equal(pair,order[:,:2]):raise ValueError('Frozen pair differs from proposer output')
    if any(proposer['proposals'][eid]['full_order'] != list(order[i]) for i,eid in enumerate(ids)):
        raise ValueError('Frozen full order differs from proposer output')
    y={h:np.array([[curves[eid][a][str(h)] for a in ACTIONS] for eid in ids]) for h in (58,61,64)}
    if not all(np.isfinite(value).all() for value in y.values()):raise ValueError('Nonfinite endpoints')
    compact,full,calibration,prefix_losses=[],[],[],[]
    parameter_names=None
    for r,ps in zip(rows,pair):
        vector=list(r['log_features'])+np.eye(3)[ps].reshape(-1).tolist()
        detail=[];cs=[];ls=[]
        for a in ps:
            obs=observations[r['id']][ACTIONS[a]]
            metrics=obs['prefix_metrics']
            if [m['update_index'] for m in metrics]!=[1,2,3,4]:raise ValueError('Incomplete prefix metrics')
            vector.extend(float(m[k]) for m in metrics for k in ('loss','grad_norm'))
            stat=obs['parameter_statistics'];names=[p['name'] for p in stat]
            if parameter_names is None:parameter_names=names
            if not names or len(names)!=len(set(names)) or names!=parameter_names:raise ValueError('Parameter feature order changed or missing')
            detail.extend(float(p[k]) for p in stat for k in ('gradient_norm','moment_norm','gradient_moment_cosine'))
            # Logical acquisition touches exactly the frozen pair, never the
            # offline counterfactual third candidate's calibration observation.
            raw=obs['calibration_losses']
            if len(raw)!=4:raise ValueError('Four calibration batches required')
            cs.append(float(np.mean(raw)));ls.append(float(np.mean([m['loss'] for m in metrics[1:]])))
        compact.append(vector);full.append(vector+detail);calibration.append(cs);prefix_losses.append(ls)
    compact,full,calibration,prefix_losses=map(np.asarray,(compact,full,calibration,prefix_losses))
    if not all(np.isfinite(v).all() for v in (x,compact,full,calibration,prefix_losses)):
        raise ValueError('Nonfinite observation or features')
    devseeds=sorted({rows[i]['seed'] for i in dev});methods=[]
    def score(name,actions,horizons,extra=0,alpha=None,model=None):
        results=[]
        for i in dev:
            h=int(horizons[i]);a=int(actions[i]);spent=3*h+extra+32
            if spent>224:raise ValueError('Logical policy exceeded budget')
            results.append({'id':ids[i],'seed':rows[i]['seed'],'action':ACTIONS[a],'updates':h,'loss':float(y[h][i,a]),'extra_units':extra,'unspent_units':224-spent})
        ps=[{'seed':s,'loss':float(np.mean([r['loss'] for r in results if r['seed']==s]))} for s in devseeds]
        result={'name':name,'alpha':alpha,'mean_development_loss':float(np.mean([r['loss'] for r in ps])),'per_seed':ps,'decisions':results}
        if model is not None:result['model']=model.state_dict()
        methods.append(result);return result
    for h in (58,61,64):
        for a in range(3):score(f'constant_{ACTIONS[a]}_h{h}',np.full(len(rows),a),np.full(len(rows),h))
    origin_joint_names=[];prefix61_names=[];prefix58_names=[]
    for alpha in (10,100):
        m=_Ridge.fit(x[disc],y[64][disc],alpha)
        score(f'origin64_a{alpha}',m.predict(x).argmin(1),np.full(len(rows),64),alpha=alpha,model=m)
        target=np.concatenate([y[h] for h in (58,61,64)],axis=1)
        m=_Ridge.fit(x[disc],target[disc],alpha);pred=m.predict(x).argmin(1)
        name=f'origin_joint_a{alpha}';origin_joint_names.append(name)
        score(name,pred%3,np.array((58,61,64))[pred//3],alpha=alpha,model=m)
        for form,features in (('compact',compact),('full',full)):
            for h in (58,61):
                target=y[h][np.arange(len(rows))[:,None],pair]
                m=_Ridge.fit(features[disc],target[disc],alpha)
                chosen=pair[np.arange(len(rows)),m.predict(features).argmin(1)]
                name=f'prefix_{form}_h{h}_a{alpha}'
                (prefix61_names if h==61 else prefix58_names).append(name)
                score(name,chosen,np.full(len(rows),h),extra=9,alpha=alpha,model=m)
    for h in (58,61):
        name=f'prefix_loss_expert_h{h}'
        (prefix61_names if h==61 else prefix58_names).append(name)
        score(name,pair[np.arange(len(rows)),prefix_losses.argmin(1)],np.full(len(rows),h),extra=9)
    chosen=pair[np.arange(len(rows)),calibration.argmin(1)]
    paid=score('calibration_expert',chosen,np.full(len(rows),58),extra=17)
    lookup={m['name']:m for m in methods}
    baseline_losses=np.array([y[64][i,pair[i,0]] for i in dev])
    pair_oracle=np.array([min(y[58][i,pair[i]]) for i in dev])
    conditional=np.minimum(baseline_losses,pair_oracle)
    oracle_per_seed=[{'seed':s,'improvement':float(np.mean((baseline_losses-conditional)[[rows[i]['seed']==s for i in dev]]))} for s in devseeds]
    oracle_gain=float(np.mean([r['improvement'] for r in oracle_per_seed]))
    comparisons=[]
    for name in origin_joint_names+prefix61_names+prefix58_names:
        ref=lookup[name];diff=[{'seed':p['seed'],'gain':r['loss']-p['loss']} for p,r in zip(paid['per_seed'],ref['per_seed'])]
        comparisons.append({'baseline':name,'gain':ref['mean_development_loss']-paid['mean_development_loss'],'per_seed':diff,'required_mean_gain':0 if name in prefix58_names else .002,'requires_all_seeds_positive':name not in prefix58_names})
    adequate=all(c['gain']>=c['required_mean_gain'] and (not c['requires_all_seeds_positive'] or all(r['gain']>0 for r in c['per_seed'])) for c in comparisons)
    # Equal performance is not incremental information value for h58 controls.
    adequate=adequate and all(c['gain']>0 for c in comparisons if c['baseline'] in prefix58_names)
    gate='DEVELOPMENT_ORACLE_NOGO' if oracle_gain<.002 else ('DEVELOPMENT_OBSERVATION_SIGNAL' if adequate else 'DEVELOPMENT_OBSERVATION_NOGO')
    rankings=[]
    for h in (58,64):
        target=y[h][np.arange(len(rows))[:,None],pair]
        for s in devseeds:
            ii=np.array([i for i in dev if rows[i]['seed']==s]);best=target[ii].argmin(1);pred=calibration[ii].argmin(1)
            rankings.append({'horizon':h,'seed':s,'calibration_pair_accuracy':float(np.mean(pred==best)),'incumbent_pair_accuracy':float(np.mean(best==0)),'calibration_pair_regret':float(np.mean(target[ii,pred]-target[ii].min(1)))})
    ranking_means=[{'horizon':h,**{key:float(np.mean([r[key] for r in rankings if r['horizon']==h]))
                    for key in ('calibration_pair_accuracy','incumbent_pair_accuracy','calibration_pair_regret')}} for h in (58,64)]
    return {'status':'completed','confirmatory':False,'gate':gate,'scope':'Fixed development observation screen, not learned search advantage or RSI','methods':methods,'comparisons':comparisons,'oracle_mean_gain':oracle_gain,'oracle_per_seed':oracle_per_seed,'rankings':rankings,'ranking_mean_over_seeds':ranking_means,'logical_costs':{'origin':0,'prefix':9,'prefix_and_calibration':17},'feature_dimensions':{'origin':x.shape[1],'compact_prefix':compact.shape[1],'full_prefix':full.shape[1]}}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,default=Path('artifacts/committed_prefix_v3'));p.add_argument('--rows',type=Path,default=Path('artifacts/development_v2/episodes.json'));p.add_argument('--proposer',type=Path,default=Path('protocols/prefix_proposer_v3.json'));p.add_argument('--output',type=Path,default=Path('artifacts/committed_prefix_v3_analysis'));a=p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):raise FileExistsError(a.output)
    names=('curves.json','observations.json','config.json','collection.json','parity.json','freeze.json','prefix_verification.json')
    for name in names:
        if not (a.data/name).is_file():raise FileNotFoundError(f'Required prefix completion evidence is missing: {name}')
    if (a.data/'run_failure.json').exists():raise ValueError('Collection has a failure receipt')
    bundle={name:json.loads((a.data/name).read_text()) for name in names}
    rows=json.loads(a.rows.read_text());proposer=json.loads(a.proposer.read_text())
    config=bundle['config.json'];freeze=bundle['freeze.json']
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    def frozen_path_key(path):
        try:return str(path.resolve().relative_to(REPO_ROOT))
        except ValueError:return str(path.resolve())
    source_hash=digest(a.rows)
    if source_hash != proposer['source_episode_sha256'] or source_hash != config.get('source_episode_sha256'):
        raise ValueError('Proposer or collection source changed')
    source_dir=Path(config['source_data'])
    if not source_dir.is_absolute():source_dir=REPO_ROOT/source_dir
    source_config_path=source_dir/'config.json'
    source_config=json.loads(source_config_path.read_text())
    if any(config.get(key)!=value for key,value in source_config.items()):
        raise ValueError('Collection config differs from source development configuration')
    required_source_paths=(a.rows,source_config_path,source_dir/'curves.json',source_dir/'freeze.json',source_dir/'collection.json')
    required_code_paths=(Path(__file__),REPO_ROOT/'scripts/audit_committed_prefix.py',
                         REPO_ROOT/'protocols/committed_prefix_v3.md',a.proposer)
    if freeze.get('confirmatory') is not False:raise ValueError('Invalid development freeze')
    for group, paths in (('source_files',required_source_paths),('code_and_protocol',required_code_paths)):
        recorded=freeze.get(group,{})
        if not {frozen_path_key(path) for path in paths}.issubset(recorded):
            raise ValueError(f'Frozen {group} coverage is incomplete')
        for key,expected in recorded.items():
            path=Path(key)
            if not path.is_absolute():path=REPO_ROOT/path
            if digest(path)!=expected:raise ValueError(f'Frozen {group} changed: {key}')
    if config.get('source_curves_sha256')!=digest(source_dir/'curves.json'):
        raise ValueError('Source curve hash changed')
    _validate_receipts(rows,bundle['curves.json'],bundle['observations.json'],config,
                       bundle['collection.json'],bundle['parity.json'],bundle['prefix_verification.json'])
    result=evaluate_screen(rows,bundle['curves.json'],bundle['observations.json'],proposer,config)
    result['collection']=bundle['collection.json']
    paths={'source_episodes':a.rows,'frozen_proposer':a.proposer,'analysis_script':Path(__file__),
           'prefix_protocol':REPO_ROOT/'protocols/committed_prefix_v3.md','source_config':source_config_path,
           **{name:a.data/name for name in names}}
    result['source_sha256']={name:digest(path) for name,path in paths.items()}
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'gate':result['gate'],'oracle_mean_gain':result['oracle_mean_gain'],'methods':[{k:m[k] for k in ('name','mean_development_loss')} for m in result['methods']]},indent=2))


if __name__=='__main__':main()
