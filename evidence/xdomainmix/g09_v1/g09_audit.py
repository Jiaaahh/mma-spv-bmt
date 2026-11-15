"""G09-A 审计包装：复用 v1 `audit_runs.py`（**未修改**）的全部逐行审计，
再叠加 G09 专有校验（协议/候选/有效优化器/新源码散列/配对声明）。

设计理由：v1 审计器已经覆盖采样重放、作者浮点阈值周期、终点 PASS_COMPLETED_PAIR 等；
G09 不复制它的逻辑，而是把它当作**独立核验器**调用，然后要求它看到的合同满足 G09 约束。
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
import sys

from g09_config import (EXPECTED_METHODS, INNER_DIAGNOSTIC_STEPS, INNER_SELECTION_STEP, PROTOCOL_ID, ROOT,
                        SELECTION_RULE_ID, TRAINING_TARGET, expected_param_group_plan, matrix_index, run_path, sha,
                        write_json)
from g09_core import OVERRIDDEN_LEGACY_FIELDS, assert_c03_is_v1_anchor, assert_head_ratio
from g09_config import candidate as candidate_spec

import audit_runs as v1_audit  # noqa: E402  只读复用 v1 审计器


def expected_threshold_series(steps: int = TRAINING_TARGET, warmup: int = 3000, start: float = 0.9,
                              lower: float = 0.5, change: float = 0.1, every: int = 100) -> dict:
    """冻结作者浮点阈值周期的独立复算：返回 {起始步: 该步起生效的阈值}。"""
    threshold, series = start, {}
    for step in range(1, steps + 1):
        before_mix = max(0, step - warmup - 1)
        if before_mix != 0 and before_mix % every == 0:
            nxt = threshold - change
            if threshold == lower:
                threshold = start
            elif nxt < lower:
                threshold = lower
            else:
                threshold = nxt
        series[step] = threshold
    change_points = [step for step in range(2, steps + 1) if series[step] != series[step - 1]]
    return {'series': series, 'change_points': change_points,
            'first_mix_step': warmup + 1, 'mix_calls_at_end': max(0, steps - warmup),
            'before_mix_at_final_step': max(0, steps - warmup - 1)}


def g09_contract_violations(contract: dict, method: str, fold: int, candidate_id: str) -> list:
    """G09 合同必须满足的全部约束；返回违规列表（空即通过）。"""
    bad = []
    from g09_validation import validate_contract
    try:
        validate_contract(contract, method, fold, candidate_id)
    except (RuntimeError, ValueError, KeyError, OSError) as error:
        bad.append(str(error))
    spec = candidate_spec(candidate_id)
    if contract.get('protocol_id') != PROTOCOL_ID:
        bad.append(f'protocol_id != {PROTOCOL_ID}')
    if contract.get('schema') != 2:
        bad.append('schema != 2')
    if (contract.get('method'), contract.get('fold')) != (method, fold):
        bad.append('method/fold mismatch')
    if contract.get('candidate_id') != candidate_id:
        bad.append('candidate_id mismatch')
    if 'run_id' in contract:
        bad.append('contract must not store a method-specific run_id (would break ERM/XDM contract symmetry)')
    if contract.get('training_target') != TRAINING_TARGET:
        bad.append('training_target != 5000')
    if contract.get('inner_selection_step') != INNER_SELECTION_STEP or contract.get('inner_diagnostic_steps') != INNER_DIAGNOSTIC_STEPS:
        bad.append('inner step contract mismatch')
    if contract.get('selection_rule_id') != SELECTION_RULE_ID:
        bad.append('selection_rule_id mismatch')
    if contract.get('historical_run_reuse', 'absent') not in ('absent', False):
        bad.append('historical_run_reuse must not be enabled')
    opt = contract.get('optimizer') or {}
    if Decimal(str(opt.get('backbone_lr'))) != Decimal(str(spec['backbone_lr'])):
        bad.append('optimizer.backbone_lr differs from candidate')
    if Decimal(str(opt.get('head_lr'))) != Decimal(str(spec['head_lr'])):
        bad.append('optimizer.head_lr differs from candidate')
    if Decimal(str(opt.get('weight_decay'))) != Decimal(str(spec['weight_decay'])):
        bad.append('optimizer.weight_decay differs from candidate')
    if Decimal(str(opt.get('domain_lr_candidate'))) != Decimal(str(spec['xdm_domain_lr'])):
        bad.append('optimizer.domain_lr_candidate differs from candidate')
    if Decimal(str(opt.get('domain_weight_decay_candidate'))) != Decimal(str(spec['xdm_domain_weight_decay'])):
        bad.append('optimizer.domain_weight_decay_candidate differs from candidate')
    if opt.get('name') != 'Adam' or opt.get('foreach') is not False:
        bad.append('optimizer must be Adam with foreach=False')
    if opt.get('param_group_plan') != expected_param_group_plan(spec):
        bad.append('optimizer.param_group_plan differs from the declared plan')
    hp = contract.get('xdm_hparams') or {}
    if Decimal(str(hp.get('lr'))) != Decimal(str(spec['backbone_lr'])) or \
            Decimal(str(hp.get('weight_decay'))) != Decimal(str(spec['weight_decay'])):
        bad.append('xdm_hparams lr/weight_decay differ from candidate (author hparams must match the effective optimizer)')
    if (hp.get('warmup_step'), hp.get('threshold'), hp.get('step_to_change')) != (3000, 0.9, 100):
        bad.append('author warmup/threshold schedule fields changed')
    if sorted(contract.get('overridden_legacy_fields') or []) != sorted(OVERRIDDEN_LEGACY_FIELDS):
        bad.append('overridden_legacy_fields does not declare exactly the intended v1 overrides')
    sources = contract.get('g09_source_hashes') or {}
    if not sources:
        bad.append('g09_source_hashes missing')
    for name, digest in sources.items():
        path = ROOT / name
        if not path.is_file() or sha(path) != digest:
            bad.append(f'g09 source changed since contract: {name}')
    for key in ('g09_protocol_sha256', 'g09_candidate_table_sha256', 'g09_run_matrix_sha256', 'g09_handover_sha256'):
        if not contract.get(key):
            bad.append(f'{key} missing')
    for name, digest in (contract.get('source_hashes') or {}).items():
        path = ROOT / name
        if not path.is_file() or sha(path) != digest:
            bad.append(f'legacy source changed since contract: {name}')
    return bad


def g09_report_violations(report: dict, contracts: dict, fold: int, candidate_id: str) -> list:
    """审计报告层面的 G09 约束（用于单元测试与真实审计后的复核）。"""
    bad = []
    if report.get('read_only') is not True or report.get('model_constructed_or_trained') is not False:
        bad.append('audit must be read-only and must not construct/train a model')
    if report.get('status') not in ('PASS_SINGLE_RUN', 'PASS_MATCHED_NODE', 'PASS_COMPLETED_PAIR', 'INCOMPLETE_PAIR'):
        bad.append(f'unexpected audit status {report.get("status")!r}')
    runs = report.get('runs') or []
    if len(runs) != 2:
        bad.append('a G09 pair audit must contain exactly two runs')
    methods = [r.get('method') for r in runs]
    if sorted(methods) != sorted(EXPECTED_METHODS):
        bad.append(f'pair must be exactly ERM+XDM, got {methods}')
    for entry in runs:
        name = entry.get('method')
        contract = contracts.get(name)
        if contract is None:
            bad.append(f'missing contract for {name}')
            continue
        bad.extend(f'{name}: {item}' for item in g09_contract_violations(contract, name, fold, candidate_id))
    if len(runs) == 2:
        if report.get('matched_prefix_steps') != min(r.get('saved_step', 0) for r in runs):
            bad.append('matched_prefix_steps inconsistent with saved steps')
        if report.get('shared_contract_equal') is not True:
            bad.append('paired contracts must be equal beyond method')
        finished = all(r.get('saved_step') == TRAINING_TARGET for r in runs)
        if finished and report.get('status') != 'PASS_COMPLETED_PAIR':
            bad.append(f'completed 5000-step pair must be PASS_COMPLETED_PAIR, got {report.get("status")!r}')
        if finished and report.get('training_complete_for_both') is not True:
            bad.append('training_complete_for_both must be true for a completed pair')
    return bad


def audit_pair(fold: int, candidate_id: str, out: Path) -> dict:
    """对已声明的同折同候选 ERM/XDM 配对执行 G09 审计（复用 v1 审计器 + G09 复核）。"""
    spec = candidate_spec(candidate_id)
    assert_head_ratio(spec)
    if candidate_id == 'C03':
        assert_c03_is_v1_anchor(spec)
    pair = {'erm': run_path(fold, 'erm', candidate_id), 'xdm': run_path(fold, 'xdm', candidate_id)}
    index = matrix_index()
    for name, relative in pair.items():
        if relative not in index:
            raise RuntimeError(f'{relative} is not a declared G09 run')
    report = v1_audit.audit([pair['erm'], pair['xdm']])
    contracts = {name: json.loads((ROOT / relative / 'contract.json').read_text(encoding='utf-8'))
                 for name, relative in pair.items()}
    violations = g09_report_violations(report, contracts, fold, candidate_id)
    threshold = expected_threshold_series()
    wrapper = {'protocol_id': PROTOCOL_ID, 'candidate_id': candidate_id, 'fold': fold,
               'utc': datetime.now(timezone.utc).isoformat(), 'argv': sys.argv,
               'wrapper_source_sha256': sha(__file__),
               'parent_audit_source_sha256': sha(ROOT / 'audit_runs.py'),
               'parent_audit': report,
               'g09_violations': violations,
               'g09_checks': {'protocol_and_candidate_contract': not violations,
                              'effective_optimizer_matches_candidate': not [v for v in violations if 'optimizer' in v or 'xdm_hparams' in v],
                              'author_threshold_cycle_recomputed': True,
                              'threshold_change_points': threshold['change_points'],
                              'threshold_mix_calls_at_end': threshold['mix_calls_at_end'],
                              'final_status_expected': 'PASS_COMPLETED_PAIR at step 5000'},
               'status': ('G09_AUDIT_PASS' if not violations else 'G09_AUDIT_FAILED')}
    write_json(out, wrapper)
    return wrapper


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fold', type=int, choices=[0, 1], required=True)
    parser.add_argument('--candidate', choices=['C01', 'C02', 'C03', 'C04', 'C05', 'C06'], required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    out = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    if out.exists():
        raise RuntimeError(f'Refusing to overwrite an existing audit report: {out}')
    wrapper = audit_pair(args.fold, args.candidate, out)
    print(json.dumps({'status': wrapper['status'], 'violations': wrapper['g09_violations'],
                      'parent_status': wrapper['parent_audit']['status'], 'report': str(out)}, ensure_ascii=False, indent=2))
    return 0 if wrapper['status'] == 'G09_AUDIT_PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
