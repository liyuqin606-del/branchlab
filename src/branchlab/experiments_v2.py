"""Development-only shared-gradient diagnostics and explicit history conditions.

No final text-test stream is loaded by this collector. This is a benchmark
construction process, not a claim of net wall-clock benefit.
"""
from __future__ import annotations
from dataclasses import asdict
from datetime import datetime, timezone
import copy
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from .lookahead import prepare_decision, materialize_action
from .model import ModelConfig, TransformerLM
from .optim import AdamW
from .tokenizer import ByteBPETokenizer
from .training import (TokenStream, seed_all, step, capture_state, restore_state,
                       evaluate_loss, evaluation_batches, json_write)

ACTIONS = ('keep', 'lr_half', 'momentum_zero')
CONDITIONS = ('native_80', 'native_160', 'stale_40', 'stale_80', 'matched_stale_80',
              'blend_80', 'lr_half', 'lr_double')


def validate_development_config(cfg):
    """Check development isolation and all preconditions of the fixed history design."""
    if cfg['eval_split'] != 'dev' or set(cfg['seeds']) != {'discovery', 'development'}:
        raise ValueError('This collector supports development only; confirmation needs a separate frozen entrypoint')
    seeds = []
    for split, group in cfg['seeds'].items():
        if not isinstance(group, list) or not group:
            raise ValueError(f'{split} seeds must be a nonempty list')
        if any(not isinstance(seed, int) or isinstance(seed, bool) or not 8 <= seed < 2**32 for seed in group):
            raise ValueError('Development seeds must be integers in [8, 2**32)')
        seeds.extend(group)
    if len(seeds) != len(set(seeds)):
        raise ValueError('Development seeds must be unique within and disjoint across splits')
    for name in ('baseline_steps', 'batch_size', 'seq_len', 'eval_batches',
                 'audit_budget_forward_batches', 'reserved_final_eval_cost'):
        if not isinstance(cfg[name], int) or isinstance(cfg[name], bool) or cfg[name] <= 0:
            raise ValueError(f'{name} must be a positive integer')
    if cfg['baseline_steps'] < 160:
        raise ValueError('baseline_steps must reach the required step-160 history')
    checkpoints = cfg['checkpoint_steps']
    if (not isinstance(checkpoints, list) or any(not isinstance(value, int) or isinstance(value, bool)
            or not 1 <= value <= cfg['baseline_steps'] for value in checkpoints)
            or len(checkpoints) != len(set(checkpoints)) or not {40, 80, 120, 160} <= set(checkpoints)):
        raise ValueError('checkpoint_steps must uniquely include 40,80,120,160 within baseline_steps')
    conditions = cfg['conditions']
    if (not isinstance(conditions, list) or not conditions or any(value not in CONDITIONS for value in conditions)
            or len(conditions) != len(set(conditions))):
        raise ValueError('conditions must be a nonempty unique subset of the declared condition catalog')
    offsets = cfg['probe_offsets']
    if (not isinstance(offsets, list) or not offsets or any(not isinstance(value, int) or isinstance(value, bool)
            or value <= 0 for value in offsets) or len(offsets) != len(set(offsets))):
        raise ValueError('probe_offsets must be unique positive integers')
    if (not isinstance(cfg['eval_offset_tokens'], int) or isinstance(cfg['eval_offset_tokens'], bool)
            or cfg['eval_offset_tokens'] < 0):
        raise ValueError('eval_offset_tokens must be a nonnegative integer')
    if cfg['reserved_final_eval_cost'] < 2 * cfg['eval_batches']:
        raise ValueError('Reserve must cover both declared final dev and future test evaluation batches')
    if (cfg['audit_budget_forward_batches'] - cfg['reserved_final_eval_cost'] - 2) // 3 < 2:
        raise ValueError('Every permitted probe budget must leave at least two total training updates')
    if not math.isfinite(cfg['lr']) or cfg['lr'] <= 0:
        raise ValueError('lr must be finite and positive')
    model_config = ModelConfig(**cfg['model'])
    if cfg['seq_len'] > model_config.max_seq_len:
        raise ValueError('seq_len exceeds model max_seq_len')


def _flat(values):
    return torch.cat([x.detach().reshape(-1).cpu() for x in values])


def _cos(a, b):
    return float(torch.dot(a, b) / (a.norm()*b.norm()).clamp_min(1e-12))


def _moment(state):
    return _flat([v['exp_avg'] for v in state['optimizer']['state'].values()])


def variant(snapshots, condition):
    source_step = 80 if condition == 'native_80' else 160
    state = copy.deepcopy(snapshots[source_step])
    age, fraction, scale = 0, 0.0, 1.0
    raw_norm = float(_moment(state).norm())
    if condition in ('stale_40', 'stale_80', 'matched_stale_80', 'blend_80'):
        past = 120 if condition == 'stale_40' else 80
        age = source_step - past
        fraction = .5 if condition == 'blend_80' else 1.0
        old = snapshots[past]['optimizer']['state']
        raw_norm = float(_moment(snapshots[past]).norm())
        if condition == 'matched_stale_80':
            scale = float(_moment(state).norm()) / max(raw_norm, 1e-12)
        for index, value in state['optimizer']['state'].items():
            value['exp_avg'].mul_(1-fraction).add_(old[index]['exp_avg'], alpha=fraction*scale)
    elif condition in ('lr_half', 'lr_double'):
        for group in state['optimizer']['param_groups']:
            group['lr'] *= .5 if condition == 'lr_half' else 2
    elif condition not in ('native_80', 'native_160'):
        raise ValueError(condition)
    return state, {'origin_step':source_step, 'moment_age':age, 'old_moment_fraction':fraction,
                   'moment_scale':scale, 'old_moment_norm':raw_norm, 'condition':condition}


def features(model, optimizer, prepared, history, metadata):
    gradients = prepared.gradients
    moment = [optimizer.state[p]['exp_avg'] for p in model.parameters()]
    g, m = _flat(gradients), _flat(moment)
    past = history[metadata['origin_step']-8:metadata['origin_step']]
    losses = [r['loss'] for r in past]
    result = [metadata['origin_step']/160, metadata['moment_age']/160,
              metadata['old_moment_fraction'], metadata['moment_scale'], metadata['old_moment_norm'],
              math.log(optimizer.param_groups[0]['lr']), prepared.metrics['loss'], prepared.metrics['grad_norm'],
              float(np.mean(losses)), float(np.std(losses)), losses[-1]-losses[0],
              float(g.norm()), float(m.norm()), _cos(g,m)]
    # Parameter-wise cheap summaries are exposed equally to all learned heads.
    for grad, mom in zip(gradients, moment):
        ga, mo = grad.detach().reshape(-1).cpu(), mom.detach().reshape(-1).cpu()
        result.extend([float(ga.norm()), float(mo.norm()), _cos(ga,mo)])
    return result


def candidate_first_order_features(prepared, candidate_states):
    """Expose exact cheap candidate directions, without probe/future outcomes.

    Values are (g dot delta, ||delta||, cosine(g, delta)) for each fixed action.
    Tied parameters appear once through prepared.parameter_names.  The only
    inputs are the shared current gradient and the saved one-update states.
    """
    gradient = _flat(prepared.gradients)
    origin_weights = prepared.post_gradient_state['model']
    result = []
    for action in ACTIONS:
        candidate_weights = candidate_states[action]['model']
        delta = _flat([candidate_weights[name] - origin_weights[name]
                       for name in prepared.parameter_names])
        result.extend([float(torch.dot(gradient, delta)), float(delta.norm()), _cos(gradient, delta)])
    return result


def train_history(cfg, tokens, seed):
    seed_all(seed)
    model = TransformerLM(ModelConfig(**cfg['model'], vocab_size=cfg['vocab_size']))
    optimizer = AdamW(model.parameters(), lr=cfg['lr'],betas=(.9,.95),weight_decay=.01)
    stream = TokenStream(tokens,cfg['batch_size'],cfg['seq_len'])
    history, snapshots = [], {}
    for index in range(cfg['baseline_steps']):
        optimizer.param_groups[0]['lr'] = cfg['lr'] * min(1, (index+1)/10)
        history.append(step(model,optimizer,stream,'cpu'))
        if index+1 in cfg['checkpoint_steps']:
            snapshots[index+1] = capture_state(model,optimizer,stream,index+1,{'name':'fixed_after_warmup','lr':cfg['lr']})
    return model,optimizer,stream,history,snapshots


def _collect_development(config_path, artifacts, output):
    cfg = json.loads(Path(config_path).read_text())
    validate_development_config(cfg)
    out, a = Path(output),Path(artifacts)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Preserve completed or partial development evidence; choose a new output')
    out.mkdir(parents=True,exist_ok=True)
    inputs = ['tokenizer.json','train_tokens.npy','dev_tokens.npy']
    freeze = {'created_utc':datetime.now(timezone.utc).isoformat(), 'confirmatory':False,
        'sources':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                   [Path(config_path),Path('protocols/development_v2.md'),*Path('src/branchlab').glob('*.py')]},
        'inputs':{name:hashlib.sha256((a/name).read_bytes()).hexdigest() for name in inputs}}
    json_write(out/'freeze.json',freeze)
    tokenizer = ByteBPETokenizer.load(a/'tokenizer.json')
    cfg['vocab_size'] = tokenizer.vocab_size
    tokens = np.load(a/'train_tokens.npy')
    ends = np.flatnonzero(tokens == tokenizer.eos_id)+1
    docs = [d for d in np.split(tokens,ends) if len(d)]
    evaluation_tokens = np.load(a/'dev_tokens.npy')[cfg['eval_offset_tokens']:]
    if len(evaluation_tokens) < cfg['eval_batches'] * cfg['batch_size'] * cfg['seq_len'] + 1:
        raise ValueError('Declared dev evaluation window would wrap or repeat tokens')
    evals = evaluation_batches(evaluation_tokens,cfg['batch_size'],cfg['seq_len'],count=cfg['eval_batches'])
    horizons = {str(cost):(cfg['audit_budget_forward_batches']-cfg['reserved_final_eval_cost']-cost)//3 for cost in (0,2)}
    ledger = {'gradient_batches':0,'candidate_optimizer_steps':0,'ordinary_optimizer_steps':0,
              'probe_forward_batches':0,'development_eval_forward_batches':0,'failures':[],
              'scope':'Actual construction counts; one shared gradient creates three candidate optimizer states. No test-text evaluations.'}
    episodes, curves = [],{}
    start = time.perf_counter()
    for split, group in cfg['seeds'].items():
        for seed in group:
            order = np.random.default_rng(seed).permutation(len(docs))
            train_tokens = np.concatenate([docs[i] for i in order])
            model,opt,stream,history,snapshots = train_history(cfg,train_tokens,seed)
            ledger['gradient_batches'] += cfg['baseline_steps']
            ledger['ordinary_optimizer_steps'] += cfg['baseline_steps']
            json_write(out/'histories'/f'seed-{seed}.json',history)
            for condition in cfg['conditions']:
                episode_id = f'{split}-{seed:03d}-{condition}'
                state,meta = variant(snapshots,condition)
                restore_state(state,model,opt,stream)
                ledger['gradient_batches'] += 1
                prepared = prepare_decision(model,opt,stream,'cpu',step_number=meta['origin_step'],scheduler=state['scheduler'])
                free = features(model,opt,prepared,history,meta)
                candidate_states, probe_losses, action_curves = {}, {}, {}
                # Inputs after the common first gradient batch; sampling does
                # not change the training cursor of any committed candidate.
                peek = TokenStream(train_tokens,cfg['batch_size'],cfg['seq_len'])
                peek.load_state_dict(prepared.post_batch_stream)
                peek_batches = {offset:peek.batch('cpu') for offset in range(1,max(cfg['probe_offsets'])+1)}
                for action in ACTIONS:
                    candidate_states[action] = materialize_action(prepared,model,opt,stream,action)
                    ledger['candidate_optimizer_steps'] += 1
                    probe_losses[action] = {}
                    for offset in cfg['probe_offsets']:
                        ledger['probe_forward_batches'] += 1
                        probe_losses[action][str(offset)] = evaluate_loss(model,[peek_batches[offset]])
                        if not math.isfinite(probe_losses[action][str(offset)]):
                            raise FloatingPointError(f'Nonfinite probe loss: {episode_id}/{action}/offset-{offset}')
                    restore_state(candidate_states[action],model,opt,stream)
                    action_curves[action] = {}
                    try:
                        for update in range(2,max(horizons.values())+1):
                            ledger['gradient_batches'] += 1
                            step(model,opt,stream,'cpu')
                            ledger['ordinary_optimizer_steps'] += 1
                            if update in horizons.values():
                                ledger['development_eval_forward_batches'] += len(evals)
                                loss = evaluate_loss(model,evals)
                                if not math.isfinite(loss):
                                    raise FloatingPointError('Nonfinite development loss')
                                action_curves[action][str(update)] = loss
                    except FloatingPointError as error:
                        ledger['failures'].append({'episode':episode_id,'action':action,'error':str(error)})
                        for h in horizons.values():
                            action_curves[action].setdefault(str(h),100.0)
                probe_features = {f'{action}:{offset}:next_batch_loss':
                                  [probe_losses[action][str(offset)]-probe_losses['keep'][str(offset)]]
                                  for action in ACTIONS[1:] for offset in cfg['probe_offsets']}
                # This may run after the table continuations, but reads only
                # frozen origin/current-gradient/first-update snapshots above.
                # Every cheap-log policy receives these nine features too.
                free.extend(candidate_first_order_features(prepared, candidate_states))
                episode = {'id':episode_id,'seed':seed,'split':split,'condition':condition,'metadata':meta,
                    'log_features':free,'probe_features':probe_features,
                    'probe_costs':{name:2 for name in probe_features},
                    'budget_losses':{cost:[action_curves[action][str(h)] for action in ACTIONS] for cost,h in horizons.items()}}
                episodes.append(episode)
                curves[episode_id] = action_curves
                json_write(out/'episodes.json',episodes)
                json_write(out/'curves.json',curves)
                json_write(out/'collection.json',ledger)
                print(json.dumps({'event':'development_episode','id':episode_id,'losses':episode['budget_losses']}),flush=True)
            checkpoint_dir = out/'checkpoints'
            checkpoint_dir.mkdir(exist_ok=True)
            torch.save({'snapshots':snapshots,'document_order':order.tolist()},checkpoint_dir/f'seed-{seed}.pt')
    ledger['elapsed_seconds'] = time.perf_counter()-start
    ledger['trained_tokens'] = ledger['gradient_batches']*cfg['batch_size']*cfg['seq_len']
    ledger['approx_forward_batch_units'] = 3*ledger['gradient_batches']+ledger['probe_forward_batches']+ledger['development_eval_forward_batches']
    json_write(out/'collection.json',ledger)
    json_write(out/'config.json',cfg)
    print(json.dumps({'event':'development_collection_complete','episodes':len(episodes),'failures':len(ledger['failures'])}),flush=True)
    return episodes,curves,ledger


def collect_development(config_path='configs/development_v2.json', artifacts='artifacts', output='artifacts/development_v2'):
    """Preserve fatal run evidence without retrying or dropping failed states.

    Preflight validation and existing-output checks occur outside the failure
    handler, so an invalid request never overwrites another run's evidence.
    Failed collection does not produce the final config.json completion marker.
    """
    cfg = json.loads(Path(config_path).read_text())
    validate_development_config(cfg)
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Preserve completed or partial development evidence; choose a new output')
    try:
        return _collect_development(config_path, artifacts, output)
    except Exception as error:
        out.mkdir(parents=True, exist_ok=True)
        receipt = {'created_utc': datetime.now(timezone.utc).isoformat(),
                   'status': 'FAILED_INCOMPLETE', 'confirmatory': False,
                   'exception_type': type(error).__name__, 'error': str(error),
                   'completion_admitted': False, 'automatic_retry': False,
                   'scope': 'Fatal collection failure; preserve all partial files. In-flight work may not yet appear in the last flushed episode ledger.'}
        # Never overwrite an existing receipt, even in an unexpected race.
        try:
            with (out/'run_failure.json').open('x') as destination:
                json.dump(receipt, destination, indent=2, allow_nan=False)
                destination.write('\n')
        except FileExistsError:
            pass
        raise


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/development_v2.json')
    parser.add_argument('--artifacts',default='artifacts')
    parser.add_argument('--output',default='artifacts/development_v2')
    args=parser.parse_args()
    torch.set_num_threads(4)
    collect_development(args.config,args.artifacts,args.output)
