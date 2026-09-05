"""Synthetic admission, information-boundary and fixed-budget screen tests."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from branchlab.synthesis import _Ridge


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/analyze_committed_prefix.py'
SPEC = importlib.util.spec_from_file_location('committed_prefix_analysis', SCRIPT)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def screen_fixture():
    """Discovery prefers keep, development prefers the challenger; logs are equal."""
    config = {'seeds': {'discovery': [8, 9], 'development': [12, 13]},
              'conditions': ['c0', 'c1'], 'eval_batches': 2, 'batch_size': 1, 'seq_len': 4,
              'audit_budget_forward_batches': 224, 'reserved_final_eval_cost': 32,
              'horizons': [58, 61, 64], 'actions': list(analysis.ACTIONS),
              'prefix_steps': 4, 'prefix_len': 4, 'calibration_positions': [65, 81, 97, 113],
              'pair_extra_cost': 17, 'pair_retained_updates': 58,
              'prefix_only_extra_cost': 9, 'prefix_only_retained_updates': 61,
              'eval_split': 'dev', 'confirmatory': False, 'parity_tolerance': 1e-6}
    rows, curves, observations = [], {}, {}
    for split, seeds in config['seeds'].items():
        for seed in seeds:
            for condition in config['conditions']:
                eid = f'{split}-{seed:03d}-{condition}'
                losses = [1., 2., 3.] if split == 'discovery' else [2., 1., 3.]
                rows.append({'id': eid, 'split': split, 'seed': seed, 'condition': condition,
                             'log_features': [1., 0., 0.], 'budget_losses': {'0': losses}})
                curves[eid] = {a: {str(h): losses[j] for h in (58, 61, 64)}
                               for j, a in enumerate(analysis.ACTIONS)}
                observations[eid] = {a: {
                    'prefix_metrics': [{'update_index': k, 'loss': 1., 'grad_norm': 1.} for k in (1, 2, 3, 4)],
                    'parameter_statistics': [{'name': 'weight', 'gradient_norm': 1., 'moment_norm': 1., 'gradient_moment_cosine': .5}],
                    'calibration_losses': [losses[j]] * 4,
                    'prefix_state_sha256': f'{eid}:{a}',
                } for j, a in enumerate(analysis.ACTIONS)}
    disc = [r for r in rows if r['split'] == 'discovery']
    model = _Ridge.fit(np.array([r['log_features'] for r in disc]),
                       np.array([r['budget_losses']['0'] for r in disc]), 10.)
    order = np.argsort(model.predict(np.array([r['log_features'] for r in rows])), axis=1, kind='stable')
    proposer = {'discovery_seeds': config['seeds']['discovery'], 'model': model.state_dict(),
                'proposals': {r['id']: {'pair': p[:2].tolist(), 'full_order': p.tolist()} for r, p in zip(rows, order)}}
    return rows, curves, observations, proposer, config


def receipts(rows, curves, observations, config):
    n = len(rows)
    counts = {'gradient_batches': n * 190, 'candidate_optimizer_steps': n * 3,
              'ordinary_optimizer_steps': n * 189, 'calibration_forward_batches': n * 12,
              'development_eval_forward_batches': n * 9 * config['eval_batches']}
    ledger = {'status': 'completed', 'confirmatory': False, 'source_episode_count': n, 'completed_episodes': n,
              'expected_counts': counts, **counts,
              'gradient_batch_attempts': counts['gradient_batches'], 'shared_gradient_batches': n,
              'candidate_optimizer_attempts': counts['candidate_optimizer_steps'],
              'ordinary_optimizer_attempts': counts['ordinary_optimizer_steps'],
              'calibration_forward_attempts': counts['calibration_forward_batches'],
              'development_eval_forward_attempts': counts['development_eval_forward_batches'],
              'baseline_retraining_updates': 0, 'test_forward_batches': 0,
              'approx_forward_batch_units': 3 * counts['gradient_batches'] + counts['calibration_forward_batches'] + counts['development_eval_forward_batches'],
              'trained_tokens': counts['gradient_batches'] * config['batch_size'] * config['seq_len'],
              'attempted_training_tokens': counts['gradient_batches'] * config['batch_size'] * config['seq_len']}
    comparisons, records = [], []
    for row in rows:
        for action in analysis.ACTIONS:
            eid = row['id']
            value = curves[eid][action]['64']
            comparisons.append({'episode_id': eid, 'action': action, 'horizon': 64, 'expected': value,
                                'actual': value, 'abs_difference': 0., 'exact': True, 'passed': True})
            signature = {'sha256': observations[eid][action]['prefix_state_sha256'],
                         'components': {k: f'{eid}:{action}:{k}' for k in ('model', 'optimizer', 'stream', 'step', 'scheduler', 'model_training', 'rng')}}
            records.append({'episode_id': eid, 'action': action, 'step': 84, 'expected_step': 84,
                            'optimizer_steps': [84], 'before_calibration': signature,
                            'after_calibration': copy.deepcopy(signature), 'after_restore': copy.deepcopy(signature),
                            'calibration_preserved_training_state': True, 'restored_exactly': True, 'passed': True})
    parity = {'status': 'completed', 'passed': True, 'tolerance': 1e-6, 'comparison_count': n * 3,
              'expected_comparisons': n * 3, 'comparisons': comparisons, 'all_exact': True, 'max_abs_difference': 0.}
    verification = {'status': 'completed', 'passed': True, 'count': n * 3, 'expected_count': n * 3, 'records': records}
    return ledger, parity, verification


def test_fixed_screen_costs_methods_seed_units_and_observation_gate():
    fixture = screen_fixture()
    before = copy.deepcopy(fixture)
    result = analysis.evaluate_screen(*fixture)
    assert fixture == before
    assert result['confirmatory'] is False and result['gate'] == 'DEVELOPMENT_OBSERVATION_SIGNAL'
    assert len(result['methods']) == 24 and len(result['comparisons']) == 12
    assert result['oracle_mean_gain'] == 1.
    assert all(r['calibration_pair_accuracy'] == 1. for r in result['ranking_mean_over_seeds'])
    for method in result['methods']:
        assert {p['seed'] for p in method['per_seed']} == {12, 13}
        assert method['mean_development_loss'] == np.mean([p['loss'] for p in method['per_seed']])
        for decision in method['decisions']:
            assert 3 * decision['updates'] + decision['extra_units'] + decision['unspent_units'] + 32 == 224
            if method['name'].startswith('prefix_') or method['name'] == 'calibration_expert':
                assert decision['action'] in ('keep', 'lr_half')
    lookup = {m['name']: m for m in result['methods']}
    assert lookup['calibration_expert']['decisions'][0]['updates'] == 58
    assert lookup['calibration_expert']['decisions'][0]['unspent_units'] == 1
    assert lookup['prefix_compact_h61_a10']['decisions'][0]['unspent_units'] == 0
    assert lookup['prefix_full_h58_a100']['decisions'][0]['unspent_units'] == 9
    assert lookup['origin64_a10']['decisions'][0]['unspent_units'] == 0
    json.dumps(result, allow_nan=False)


class ForbiddenMapping(dict):
    def __getitem__(self, key):
        raise AssertionError(f'Third-action observation was acquired: {key}')


def test_only_frozen_pair_observations_are_acquired():
    rows, curves, observations, proposer, config = screen_fixture()
    for values in observations.values():
        values['momentum_zero'] = ForbiddenMapping(values['momentum_zero'])
    assert analysis.evaluate_screen(rows, curves, observations, proposer, config)['gate'] == 'DEVELOPMENT_OBSERVATION_SIGNAL'


def test_calibration_is_not_a_feature_of_any_free_or_prefix_control():
    rows, curves, observations, proposer, config = screen_fixture()
    original = analysis.evaluate_screen(rows, curves, observations, proposer, config)
    changed = copy.deepcopy(observations)
    for values in changed.values():
        values['keep']['calibration_losses'] = [-100.] * 4
        values['lr_half']['calibration_losses'] = [100.] * 4
    other = analysis.evaluate_screen(rows, curves, changed, proposer, config)
    assert original['methods'][:-1] == other['methods'][:-1]
    assert other['gate'] == 'DEVELOPMENT_OBSERVATION_NOGO'
    assert other['oracle_mean_gain'] == original['oracle_mean_gain'] == 1.


def test_development_outcomes_do_not_fit_models_or_choose_actions():
    rows, curves, observations, proposer, config = screen_fixture()
    first = analysis.evaluate_screen(rows, curves, observations, proposer, config)
    for row in rows:
        if row['split'] == 'development':
            for j, action in enumerate(analysis.ACTIONS):
                curves[row['id']][action] = {str(h): float(j * 20) for h in (58, 61, 64)}
    second = analysis.evaluate_screen(rows, curves, observations, proposer, config)
    for a, b in zip(first['methods'], second['methods']):
        assert a.get('model') == b.get('model')
        assert [(d['action'], d['updates']) for d in a['decisions']] == [(d['action'], d['updates']) for d in b['decisions']]
    assert second['gate'] == 'DEVELOPMENT_ORACLE_NOGO'


def test_calibration_tie_keeps_the_frozen_incumbent():
    rows, curves, observations, proposer, config = screen_fixture()
    for values in observations.values():
        values['keep']['calibration_losses'] = values['lr_half']['calibration_losses'] = [1.] * 4
    result = analysis.evaluate_screen(rows, curves, observations, proposer, config)
    assert all(d['action'] == 'keep' for d in result['methods'][-1]['decisions'])
    assert result['gate'] == 'DEVELOPMENT_OBSERVATION_NOGO'


def test_mean_gain_cannot_override_a_seed_without_positive_gain():
    rows, curves, observations, proposer, config = screen_fixture()
    for row in rows:
        if row['seed'] == 12:
            observations[row['id']]['keep']['calibration_losses'] = [0.] * 4
    result = analysis.evaluate_screen(rows, curves, observations, proposer, config)
    assert result['oracle_mean_gain'] == 1.
    assert all(c['gain'] >= c['required_mean_gain'] for c in result['comparisons'])
    assert result['gate'] == 'DEVELOPMENT_OBSERVATION_NOGO'


def test_seed_condition_coverage_and_confirmation_states_fail_closed():
    rows, curves, observations, proposer, config = screen_fixture()
    dropped = rows.pop()
    for table in (curves, observations, proposer['proposals']):
        del table[dropped['id']]
    with pytest.raises(ValueError, match='seed/condition'):
        analysis.evaluate_screen(rows, curves, observations, proposer, config)
    rows, curves, observations, proposer, config = screen_fixture()
    rows[-1]['split'] = 'test'
    with pytest.raises(ValueError, match='development data only'):
        analysis.evaluate_screen(rows, curves, observations, proposer)
    rows, curves, observations, proposer, config = screen_fixture()
    rows[-1]['condition'] = rows[-2]['condition']
    with pytest.raises(ValueError, match='seed/condition'):
        analysis.evaluate_screen(rows, curves, observations, proposer, config)


@pytest.mark.parametrize('failure', ['incomplete', 'prefix_failed', 'parity_duplicate', 'prefix_duplicate', 'physical_count', 'test_forward', 'endpoint', 'restoration', 'budget'])
def test_incomplete_or_inconsistent_receipts_are_rejected(failure):
    rows, curves, observations, _, config = screen_fixture()
    ledger, parity, verification = receipts(rows, curves, observations, config)
    analysis._validate_receipts(rows, curves, observations, config, ledger, parity, verification)
    if failure == 'incomplete': ledger['completed_episodes'] -= 1
    if failure == 'prefix_failed': verification['passed'] = False
    if failure == 'parity_duplicate': parity['comparisons'][-1] = copy.deepcopy(parity['comparisons'][0])
    if failure == 'prefix_duplicate': verification['records'][-1] = copy.deepcopy(verification['records'][0])
    if failure == 'physical_count': ledger['gradient_batch_attempts'] -= 1
    if failure == 'test_forward': ledger['test_forward_batches'] = 1
    if failure == 'endpoint': curves[rows[0]['id']]['keep']['64'] += .1
    if failure == 'restoration': verification['records'][0]['after_restore']['components']['rng'] = 'changed'
    if failure == 'budget': config['pair_retained_updates'] = 59
    with pytest.raises(ValueError):
        analysis._validate_receipts(rows, curves, observations, config, ledger, parity, verification)


def test_cli_requires_all_receipts_and_frozen_sources(tmp_path, monkeypatch):
    rows, curves, observations, proposer, config = screen_fixture()
    data, source, output = (tmp_path / name for name in ('data', 'source', 'output'))
    data.mkdir(); source.mkdir()
    rows_path, proposer_path = source / 'episodes.json', tmp_path / 'proposer.json'
    rows_path.write_text(json.dumps(rows))
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    config['source_episode_sha256'] = proposer['source_episode_sha256'] = digest(rows_path)
    proposer_path.write_text(json.dumps(proposer))
    argv = ['analyze', '--data', str(data), '--rows', str(rows_path), '--proposer', str(proposer_path), '--output', str(output)]
    monkeypatch.setattr(sys, 'argv', argv)
    with pytest.raises(FileNotFoundError, match='completion evidence'):
        analysis.main()
    for name, value in (('config.json', config), ('curves.json', curves), ('freeze.json', {}), ('collection.json', {})):
        (source / name).write_text(json.dumps(value))
    config['source_data'] = str(source)
    config['source_curves_sha256'] = digest(source / 'curves.json')
    ledger, parity, verification = receipts(rows, curves, observations, config)
    def portable(path):
        try: return str(path.resolve().relative_to(analysis.REPO_ROOT))
        except ValueError: return str(path.resolve())
    code_paths = (SCRIPT, analysis.REPO_ROOT / 'scripts/audit_committed_prefix.py',
                  analysis.REPO_ROOT / 'protocols/committed_prefix_v3.md', proposer_path)
    freeze = {'confirmatory': False,
              'source_files': {portable(p): digest(p) for p in source.iterdir()},
              'code_and_protocol': {portable(p): digest(p) for p in code_paths}}
    for name, value in (('config.json', config), ('curves.json', curves), ('observations.json', observations),
                        ('collection.json', ledger), ('parity.json', parity), ('freeze.json', freeze), ('prefix_verification.json', verification)):
        (data / name).write_text(json.dumps(value))
    bad = copy.deepcopy(verification); bad['passed'] = False
    (data / 'prefix_verification.json').write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='prefix verification'):
        analysis.main()
    (data / 'prefix_verification.json').write_text(json.dumps(verification))
    frozen = copy.deepcopy(freeze)
    frozen['code_and_protocol'][portable(SCRIPT)] = 'wrong-hash'
    (data / 'freeze.json').write_text(json.dumps(frozen))
    with pytest.raises(ValueError, match='Frozen code_and_protocol changed'):
        analysis.main()
    (data / 'freeze.json').write_text(json.dumps(freeze))
    analysis.main()
    result = json.loads((output / 'summary.json').read_text())
    assert result['source_sha256']['analysis_script'] == digest(SCRIPT)
    assert result['source_sha256']['prefix_protocol'] == digest(analysis.REPO_ROOT / 'protocols/committed_prefix_v3.md')
    assert result['gate'] == 'DEVELOPMENT_OBSERVATION_SIGNAL'
    assert result['collection']['gradient_batches'] == len(rows) * 190
    with pytest.raises(FileExistsError):
        analysis.main()
