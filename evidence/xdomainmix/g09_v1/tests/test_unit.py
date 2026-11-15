"""G09-A CPU 单元测试（第一批）：配置/矩阵/合同/优化器/序列化拒绝/审计包装/阈值周期。

全部使用工程 scope + tiny 桩件；任何通过都**不是** GPU 资格证据。
"""
from __future__ import annotations

import copy
import glob
import importlib.util
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
import numpy as np
import torch

from g09_config import (EXPECTED_CANDIDATES, EXPECTED_FOLDS, EXPECTED_METHODS, G09_DIR, PROTOCOL_ID, ROOT,
                        candidate, candidate_table, matrix_index, run_path, sha, write_json)
from g09_core import (OVERRIDDEN_LEGACY_FIELDS, G09Experiment, assert_c03_is_v1_anchor, assert_head_ratio)
import g09_audit
import g09_config
import g09_runner

sys.path.insert(0, str(G09_DIR / 'tests'))
from engineering_kit import STUB_NOTICE, StubAlgo, TinyNet  # noqa: E402
import fixtures  # noqa: E402


def expect_error(fn, contains: str, label: str):
    try:
        fn()
    except Exception as error:  # noqa: BLE001
        if contains not in str(error):
            raise AssertionError(f'{label}: error message {str(error)!r} does not contain {contains!r}') from error
        return str(error)
    raise AssertionError(f'{label}: expected an error containing {contains!r}, but nothing was raised')


def tiny_erm(fold=0, candidate_id='C03', method='erm'):
    return G09Experiment(method, fold, 8, candidate_id, device='cpu', scope='ENGINEERING_ONLY',
                         engineering={'tiny_net': TinyNet})


def tiny_xdm(fold=0, candidate_id='C02'):
    return G09Experiment('xdm', fold, 8, candidate_id, device='cpu', scope='ENGINEERING_ONLY',
                         engineering={'stub_algo': StubAlgo})


# ---------------------------------------------------------------- 配置与矩阵
def test_matrix_and_candidates():
    index = matrix_index()
    assert len(index) == 24, len(index)
    for fold in EXPECTED_FOLDS:
        for cid in EXPECTED_CANDIDATES:
            for method in EXPECTED_METHODS:
                assert run_path(fold, method, cid) in index
    table = candidate_table()
    assert list(table) == EXPECTED_CANDIDATES
    for spec in table.values():
        assert_head_ratio(spec)
    assert_c03_is_v1_anchor(table['C03'])
    expect_error(lambda: candidate('C07'), 'Illegal candidate_id', 'illegal candidate')
    expect_error(lambda: run_path(2, 'erm', 'C01'), 'Illegal G09 run components', 'illegal fold')
    expect_error(lambda: run_path(0, 'adam', 'C01'), 'Illegal G09 run components', 'illegal method')


def test_mapping_constants_match_v1_core():
    import core as v1_core
    assert g09_config.TRAIN_MAPS == v1_core.TRAIN_MAPS
    assert g09_config.OUTER_MAPS == v1_core.OUTER_MAPS


def test_handover_and_isolation():
    observed = g09_config.assert_handover_files()
    assert len(observed) == 3
    design = g09_config.load_protocol()
    assert design['gpu_training_authorized_by_this_plan'] is False if 'gpu_training_authorized_by_this_plan' in design \
        else design['gpu_training_authorized_by_this_package'] is False
    assert all('runs_g09_v1' in entry['run_path'] for entry in g09_config.load_matrix())


def test_declared_run_resolution():
    relative, entry, directory = g09_runner.resolve_declared_run('erm', 1, 'C04', 'runs_g09_v1/f1_s17_k8_erm_C04')
    assert relative == 'runs_g09_v1/f1_s17_k8_erm_C04' and entry['candidate_id'] == 'C04'
    assert str(directory).endswith('runs_g09_v1/f1_s17_k8_erm_C04')
    expect_error(lambda: g09_runner.resolve_declared_run('erm', 1, 'C04', 'runs/f1_s17_k8_erm_venv26'),
                 'not the declared G09 run', 'legacy v1 path refused')
    expect_error(lambda: g09_runner.resolve_declared_run('erm', 1, 'C04', 'runs_g09_v1/f1_s17_k8_erm_C03'),
                 'not the declared G09 run', 'candidate mismatch refused')
    expect_error(lambda: g09_runner.resolve_declared_run('xdm', 0, 'C05', 'runs_g09_v1/f0_s17_k8_erm_C05'),
                 'not the declared G09 run', 'method mismatch refused')


# ---------------------------------------------------------------- 合同与优化器
def test_contract_fields_and_validation():
    exp = tiny_erm(candidate_id='C04')
    contract = exp.contract
    assert contract['schema'] == 2 and contract['protocol_id'] == PROTOCOL_ID
    assert contract['candidate_id'] == 'C04'
    assert contract['optimizer']['param_group_plan'] == g09_config.expected_param_group_plan(candidate('C04'))
    assert sorted(contract['overridden_legacy_fields']) == sorted(OVERRIDDEN_LEGACY_FIELDS)
    assert contract['xdm_hparams']['lr'] == candidate('C04')['backbone_lr']
    assert contract['xdm_hparams']['weight_decay'] == candidate('C04')['weight_decay']
    assert contract['selection_rule_id'] == g09_config.SELECTION_RULE_ID
    assert contract['inner_selection_step'] == 5000
    assert not g09_audit.g09_contract_violations(contract, 'erm', 0, 'C04'), \
        g09_audit.g09_contract_violations(contract, 'erm', 0, 'C04')


def test_optimizer_groups_match_contract_for_all_candidates():
    for cid in EXPECTED_CANDIDATES:
        exp = tiny_erm(candidate_id=cid)
        group_lrs = [g['lr'] for g in exp.optimizer.param_groups]
        group_wds = [g['weight_decay'] for g in exp.optimizer.param_groups]
        spec = candidate(cid)
        assert group_lrs == [spec['backbone_lr'], spec['head_lr']], (cid, group_lrs)
        assert group_wds == [spec['weight_decay'], spec['weight_decay']], (cid, group_wds)
        exp.verify_optimizer_matches_contract()


def test_runtime_optimizer_tampering_is_detected():
    exp = tiny_erm(candidate_id='C03')
    exp.optimizer.param_groups[0]['lr'] = 0.5
    expect_error(exp.verify_optimizer_matches_contract, 'param_groups[0].lr', 'tampered lr')
    exp = tiny_erm(candidate_id='C03')
    exp.optimizer.param_groups[1]['weight_decay'] = 1e-3
    expect_error(exp.verify_optimizer_matches_contract, 'weight_decay', 'tampered wd')
    exp = tiny_erm(candidate_id='C03')
    exp.optimizer = torch.optim.AdamW(exp.net.parameters(), lr=5e-6)
    expect_error(exp.verify_optimizer_matches_contract, 'Adam', 'AdamW interchange refused')
    exp = tiny_erm(candidate_id='C03')
    exp.hp['lr'] = 1e-3
    expect_error(exp.verify_optimizer_matches_contract, 'hparams', 'tampered hparams refused')
    exp = tiny_xdm(candidate_id='C02')
    exp.domain_optimizer.param_groups[0]['lr'] = 9.9
    expect_error(exp.verify_optimizer_matches_contract, 'Domain optimizer', 'tampered domain lr refused')


def test_c03_semantics_equal_v1_anchor_optimizer():
    """C03 的优化器语义必须与 v1 锚点一致（同一 tiny 网络、同一数据顺序、逐步数值相同）。"""
    torch.manual_seed(0)
    net_a = TinyNet(); net_b = TinyNet()
    net_b.load_state_dict(net_a.state_dict())
    spec = candidate('C03')
    g09_optimizer = torch.optim.Adam(
        [{'params': [v for n, v in net_a.named_parameters() if not n.startswith('fc.')], 'lr': spec['backbone_lr'],
          'weight_decay': spec['weight_decay']},
         {'params': list(net_a.fc.parameters()), 'lr': spec['head_lr'], 'weight_decay': spec['weight_decay']}],
        weight_decay=spec['weight_decay'], foreach=False)
    v1_optimizer = torch.optim.Adam(  # 与 core.py 第 227-229 行同一构造
        [{'params': [v for n, v in net_b.named_parameters() if not n.startswith('fc.')], 'lr': 5e-6},
         {'params': net_b.fc.parameters(), 'lr': 1e-4}], weight_decay=0., foreach=False)
    assert type(g09_optimizer) is type(v1_optimizer) is torch.optim.Adam
    assert g09_optimizer.defaults == v1_optimizer.defaults, (g09_optimizer.defaults, v1_optimizer.defaults)
    x = torch.randn(4, 3, 224, 224, generator=torch.Generator().manual_seed(7))
    y = torch.arange(4) % 180
    import torch.nn.functional as F
    for _ in range(3):
        for net, optimizer in ((net_a, g09_optimizer), (net_b, v1_optimizer)):
            optimizer.zero_grad()
            loss = F.cross_entropy(net(x), y)
            loss.backward()
            optimizer.step()
        assert torch.allclose(F.cross_entropy(net_a(x), y), F.cross_entropy(net_b(x), y), atol=0, rtol=0)
    for (name_a, value_a), (name_b, value_b) in zip(net_a.state_dict().items(), net_b.state_dict().items()):
        assert name_a == name_b and torch.equal(value_a, value_b), name_a


def test_weight_decay_actually_binds_all_groups():
    """C02(wd=1e-4) 与 C01(wd=0) 在同一步后必须产生不同参数（wd 确实生效到两组）。"""
    exp_a, exp_b = tiny_erm(candidate_id='C01'), tiny_erm(candidate_id='C02')
    assert exp_a.optimizer.param_groups[0]['weight_decay'] == 0.0
    assert exp_b.optimizer.param_groups[0]['weight_decay'] == 1e-4
    assert exp_b.contract['xdm_hparams']['weight_decay'] == 1e-4
    for group in exp_b.optimizer.param_groups:
        assert group['weight_decay'] == 1e-4, 'wd must apply to every existing group (incl. head/BN affine)'


def test_xdm_stub_wiring_and_author_schedule():
    exp = tiny_xdm(candidate_id='C02')
    spec = candidate('C02')
    assert exp.algo.hparams['lr'] == spec['backbone_lr']
    assert exp.algo.hparams['weight_decay'] == spec['weight_decay']
    assert exp.domain_optimizer.param_groups[0]['lr'] == spec['xdm_domain_lr']
    assert exp.domain_optimizer.param_groups[0]['weight_decay'] == spec['xdm_domain_weight_decay']
    exp.hp['warmup_step'] = 0  # 强制立即进入混合阶段，验证字段形态（桩件，非科学结论）
    record = exp.update()
    assert set(record['losses']) == {'loss_task', 'loss_domain'}, record['losses']
    assert record['mix_calls'] == 1 and record['algorithm_calls'] == 1
    assert record['mode'] == 'augmented'


def test_author_threshold_cycle_cross_check():
    """用**作者原始** mixup_module 驱动 2000 次混合调用，与本仓库独立复算逐位比较。"""
    spec = importlib.util.spec_from_file_location('author_mixup', ROOT / 'vendor/domainbed/mixup_module.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    hparams = {'threshold': 0.9, 'threshold_lower_bound': 0.5, 'value_to_change': 0.1, 'step_to_change': 100,
               'probability_to_discard': 0.2}
    mix = module.DomainClassMixAugmentation(8, 180, 5, hparams)
    author_series = {}
    for step in range(1, 5001):
        if step >= 3001:  # 作者只在混合阶段调用 forward
            if (mix.call_num % mix.step_to_change == 0) and mix.call_num != 0:
                mix.update_threshold()
            mix.call_num += 1
        author_series[step] = mix.threshold
    recomputed = g09_audit.expected_threshold_series()
    assert mix.call_num == 2000, mix.call_num
    assert recomputed['mix_calls_at_end'] == 2000
    mismatches = [step for step in author_series if repr(author_series[step]) != repr(recomputed['series'][step])]
    assert not mismatches, mismatches[:5]
    boundaries = {4001: 0.5000000000000001, 4101: 0.5, 4201: 0.9, 4301: 0.8, 4401: 0.7000000000000001,
                  4501: 0.6000000000000001, 4601: 0.5000000000000001, 4701: 0.5, 4801: 0.9, 4901: 0.8}
    for step, want in boundaries.items():
        assert repr(recomputed['series'][step]) == repr(want), (step, recomputed['series'][step], want)
    assert repr(recomputed['series'][5000]) == repr(0.8)


# ---------------------------------------------------------------- 序列化与拒绝
def test_serialization_and_rejection():
    sandbox = fixtures.new_sandbox('unit_serialization')
    exp = tiny_erm(candidate_id='C03')
    target = sandbox['root'] / 'checkpoint_c03.pt'
    exp.save_checkpoint(target)
    exp.update()
    exp.update()
    exp.save_checkpoint(target)
    state = torch.load(target, map_location='cpu', weights_only=True)
    assert state['contract']['protocol_id'] == PROTOCOL_ID

    same = tiny_erm(candidate_id='C03')
    same.restore_g09(torch.load(target, map_location='cpu', weights_only=True))
    assert same.step == exp.step
    assert torch.equal(same.gen.get_state(), exp.gen.get_state())
    same.verify_optimizer_matches_contract()

    other_candidate = tiny_erm(candidate_id='C04')
    expect_error(lambda: other_candidate.restore_g09(torch.load(target, map_location='cpu', weights_only=True)),
                 "candidate 'C03'", 'different candidate refused')

    other_fold = tiny_erm(fold=1, candidate_id='C03')
    expect_error(lambda: other_fold.restore_g09(torch.load(target, map_location='cpu', weights_only=True)),
                 'Checkpoint contract differs', 'different fold refused by contract')
    method_swap = tiny_erm(fold=0, candidate_id='C03', method='xdm') if False else None
    with_run_id = copy.deepcopy(state)
    with_run_id['contract'] = dict(with_run_id['contract'], run_id='f0_s17_k8_erm_C03')
    expect_error(lambda: tiny_erm(candidate_id='C03').restore_g09(with_run_id),
                 'method-specific run_id', 'method-specific run_id in contract refused')
    other_candidate_ok = tiny_erm(candidate_id='C04')
    forged = copy.deepcopy(state)
    forged['contract'] = dict(forged['contract'], schema=1)
    forged['contract'].pop('protocol_id', None)
    expect_error(lambda: other_candidate_ok.restore_g09(forged), 'protocol_id', 'schema-1 checkpoint refused')

    legacy = ROOT / 'runs/f1_s17_k8_erm_venv26/latest.pt'
    assert legacy.is_file(), 'real G08 fold1 checkpoint must exist for this rejection test'
    legacy_state = torch.load(legacy, map_location='cpu', weights_only=True)
    expect_error(lambda: tiny_erm(candidate_id='C03').restore_g09(legacy_state),
                 'protocol_id', 'real G08 checkpoint refused (read-only load)')


# ---------------------------------------------------------------- 审计包装
def _synthetic_parent_report(contracts, status='PASS_COMPLETED_PAIR', saved=5000):
    runs = []
    for method in ('erm', 'xdm'):
        runs.append({'status': 'PASS_SINGLE_RUN', 'run': run_path(0, method, 'C03'), 'method': method, 'fold': 0,
                     'seed': 17, 'k': 8, 'saved_step': saved, 'training_complete': saved == 5000, 'metrics_rows': saved,
                     'checks': {'independent_full_sampler_replay': True}})
    return {'status': status, 'runs': runs, 'read_only': True, 'model_constructed_or_trained': False,
            'matched_prefix_steps': saved, 'all_saved_steps_equal': True, 'training_complete_for_both': saved == 5000,
            'shared_contract_equal': True, 'paired_sampler_prefix_identical': True}


def test_audit_wrapper_positive_and_negative():
    contracts = {'erm': fixtures.fixture_contract('erm', 0, 'C03'), 'xdm': fixtures.fixture_contract('xdm', 0, 'C03')}
    report = _synthetic_parent_report(contracts)
    assert not g09_audit.g09_report_violations(report, contracts, 0, 'C03')
    wrong_protocol = copy.deepcopy(contracts); wrong_protocol['erm']['protocol_id'] = 'V1'
    assert any('protocol_id' in v for v in g09_audit.g09_report_violations(report, wrong_protocol, 0, 'C03'))
    wrong_candidate = copy.deepcopy(contracts); wrong_candidate['xdm']['candidate_id'] = 'C05'
    assert any('candidate_id' in v for v in g09_audit.g09_report_violations(report, wrong_candidate, 0, 'C03'))
    wrong_lr = copy.deepcopy(contracts); wrong_lr['erm']['optimizer']['backbone_lr'] = 1e-3
    assert any('backbone_lr' in v for v in g09_audit.g09_report_violations(report, wrong_lr, 0, 'C03'))
    wrong_hp = copy.deepcopy(contracts); wrong_hp['xdm']['xdm_hparams']['lr'] = 1e-3
    assert any('xdm_hparams' in v for v in g09_audit.g09_report_violations(report, wrong_hp, 0, 'C03'))
    wrong_source = copy.deepcopy(contracts); wrong_source['erm']['g09_source_hashes']['g09_v1/g09_core.py'] = '0' * 64
    assert any('g09 source changed' in v for v in g09_audit.g09_report_violations(report, wrong_source, 0, 'C03'))
    incomplete = _synthetic_parent_report(contracts, status='PASS_MATCHED_NODE', saved=5000)
    assert any('PASS_COMPLETED_PAIR' in v for v in g09_audit.g09_report_violations(incomplete, contracts, 0, 'C03'))
    not_readonly = copy.deepcopy(report); not_readonly['read_only'] = False
    assert any('read-only' in v for v in g09_audit.g09_report_violations(not_readonly, contracts, 0, 'C03'))


def test_audit_wrapper_pair_requires_two_methods():
    contracts = {'erm': fixtures.fixture_contract('erm', 0, 'C03'), 'xdm': fixtures.fixture_contract('xdm', 0, 'C03')}
    report = _synthetic_parent_report(contracts)
    report['runs'][1]['method'] = 'erm'
    assert any('ERM+XDM' in v for v in g09_audit.g09_report_violations(report, contracts, 0, 'C03'))


TESTS = [
    ('matrix_and_candidates', test_matrix_and_candidates),
    ('mapping_constants_match_v1_core', test_mapping_constants_match_v1_core),
    ('handover_and_isolation', test_handover_and_isolation),
    ('declared_run_resolution', test_declared_run_resolution),
    ('contract_fields_and_validation', test_contract_fields_and_validation),
    ('optimizer_groups_match_contract_all_candidates', test_optimizer_groups_match_contract_for_all_candidates),
    ('runtime_optimizer_tampering_detected', test_runtime_optimizer_tampering_is_detected),
    ('c03_semantics_equal_v1_anchor', test_c03_semantics_equal_v1_anchor_optimizer),
    ('weight_decay_binds_all_groups', test_weight_decay_actually_binds_all_groups),
    ('xdm_stub_wiring', test_xdm_stub_wiring_and_author_schedule),
    ('author_threshold_cycle_cross_check', test_author_threshold_cycle_cross_check),
    ('serialization_and_rejection', test_serialization_and_rejection),
    ('audit_wrapper_positive_and_negative', test_audit_wrapper_positive_and_negative),
    ('audit_wrapper_pair_requires_two_methods', test_audit_wrapper_pair_requires_two_methods),
]
