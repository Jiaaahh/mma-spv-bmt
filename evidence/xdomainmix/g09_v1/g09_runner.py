"""G09-A 训练入口（候选化）。只续训**已声明的 G09 run**，不创建临时或自造目录。

用法要点：
  * `--run` 必须逐字等于运行矩阵声明的 `runs_g09_v1/f{fold}_s17_k8_{method}_{candidate}`；
  * 载入检查点前先拒绝：非 G09 合同（v1/G07/G08）、不同 candidate、不同 run_id、不同源码散列；
  * 每次节点保存 `invocation_/node_step/status.json`，并记录 G09 字段（protocol_id/candidate_id/run_id）；
  * 训练完成后**不自动**评价、不自动推进下一节点。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import sys
import time
import traceback

import torch

from g09_config import (INNER_DIAGNOSTIC_STEPS, PROTOCOL_ID, ROOT, TRAINING_TARGET, matrix_index, run_path, sha,
                        candidate as candidate_spec, write_json)
from g09_core import G09Experiment, assert_c03_is_v1_anchor, assert_head_ratio

import core as v1_core  # noqa: E402
from runner import reconcile_metrics, utc_now  # noqa: E402  只读复用 v1 的日志协调工具


def resolve_declared_run(method: str, fold: int, candidate_id: str, run_value: str) -> tuple:
    """把 (method, fold, candidate) 与 --run 交叉核对；任何不一致立即拒绝。"""
    want = run_path(fold, method, candidate_id)
    given = Path(run_value).as_posix().strip('/')
    if given != want:
        raise RuntimeError(f'--run {given!r} is not the declared G09 run for '
                           f'method={method} fold={fold} candidate={candidate_id}; expected {want!r}. '
                           'G09 never writes to legacy runs/ paths or ad-hoc directories.')
    index = matrix_index()
    if want not in index:
        raise RuntimeError(f'{want} is missing from the frozen G09 run matrix')
    return want, index[want], ROOT / want


def train(args) -> dict:
    from g09_validation import require_fixed_arguments, verify_source_manifest
    require_fixed_arguments(args.method, args.fold, args.k, args.seed, args.threads)
    verify_source_manifest()
    if not 1 <= args.until <= TRAINING_TARGET:
        raise ValueError(f'--until is an absolute completed step between 1 and {TRAINING_TARGET}')
    if args.max_wall_seconds <= 0 or args.save_every < 1 or args.threads < 1:
        raise ValueError('Time limit, save interval and thread count must be positive')
    spec = candidate_spec(args.candidate)
    assert_head_ratio(spec)
    if args.candidate == 'C03':
        assert_c03_is_v1_anchor(spec)
    relative, entry, directory = resolve_declared_run(args.method, args.fold, args.candidate, args.run)
    directory.parent.mkdir(parents=True, exist_ok=True)
    with v1_core.RunLock(directory):
        if (directory / 'PAUSE_REQUESTED').exists():
            raise RuntimeError('PAUSE_REQUESTED remains present. Review the checkpoint, then remove this file explicitly before resuming.')
        exp = G09Experiment(args.method, args.fold, args.k, args.candidate, seed=args.seed,
                            device=args.device, threads=args.threads)
        print(json.dumps({'entry': 'G09-A train', **exp.describe(), 'run_path': relative,
                          'scope': exp.scope, 'until': args.until}, ensure_ascii=False), flush=True)
        contract_path = directory / 'contract.json'
        checkpoint_path = directory / 'latest.pt'
        if contract_path.exists():
            existing = json.loads(contract_path.read_text(encoding='utf-8'))
            if existing != exp.contract:
                raise RuntimeError('Run directory belongs to a different immutable G09 contract '
                                   '(protocol/candidate/source/asset/runtime); use a new run directory')
            if not checkpoint_path.exists():
                raise RuntimeError('Existing run has no latest.pt. Inspect previous checkpoint / initialization failure manually; never silently restart.')
            exp.restore_g09(torch.load(checkpoint_path, map_location='cpu', weights_only=True))
            reconcile_metrics(directory, exp.step)
        else:
            if checkpoint_path.exists() or (directory / 'metrics.jsonl').exists():
                raise RuntimeError('Run artifacts exist without contract; refusing to overwrite them')
            write_json(contract_path, exp.contract)
            exp.save_checkpoint(checkpoint_path)
        if args.until <= exp.step:
            result = {'status': 'ALREADY_AT_OR_BEYOND_REQUESTED_NODE', 'saved_step': exp.step,
                      'requested_until': args.until, 'run_path': relative}
            print(json.dumps(result, ensure_ascii=False), flush=True)
            return result
        stop = {'requested': False, 'reason': None}

        def request_stop(number, frame):
            stop.update(requested=True, reason=f'signal_{number}')
            print('STOP requested; finishing the active optimizer update before saving.', flush=True)

        old_handlers = {}
        for name in ('SIGINT', 'SIGTERM'):
            sig = getattr(signal, name, None)
            if sig is not None:
                old_handlers[sig] = signal.signal(sig, request_stop)
        invocation = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        start_step, started, last_saved = exp.step, time.monotonic(), exp.step
        write_json(directory / f'invocation_{invocation}.json', {
            'utc': utc_now(), 'argv': sys.argv, 'cwd': str(Path.cwd()), 'protocol_id': PROTOCOL_ID,
            'candidate_id': args.candidate, 'run_id': entry['run_id'],
            'start_step': start_step, 'method': args.method, 'fold': args.fold,
            'seed': args.seed, 'k': args.k, 'until': args.until,
            'max_wall_seconds': args.max_wall_seconds, 'threads': args.threads,
            'save_every': args.save_every, 'device': args.device,
            'effective_optimizer': exp.contract['candidate'],
            'contract_sha256': sha(contract_path)})
        records = []
        status = 'NODE_REACHED'
        try:
            with (directory / 'metrics.jsonl').open('a', encoding='utf-8', newline='\n') as metrics:
                while exp.step < args.until:
                    if stop['requested'] or (directory / 'PAUSE_REQUESTED').exists() or \
                            time.monotonic() - started >= args.max_wall_seconds:
                        status = 'PAUSED'; break
                    record = exp.update()
                    record.update(utc=utc_now(), invocation=invocation, elapsed_invocation_seconds=time.monotonic() - started)
                    metrics.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n'); metrics.flush()
                    records.append(record)
                    print(json.dumps(record, ensure_ascii=False, allow_nan=False), flush=True)
                    if exp.step % args.save_every == 0:
                        exp.save_checkpoint(checkpoint_path); last_saved = exp.step
                exp.save_checkpoint(checkpoint_path); last_saved = exp.step
            if exp.step < args.until:
                status = 'PAUSED'
            exp.verify_optimizer_matches_contract()
            checkpoint_hash = sha(checkpoint_path)
            write_json(directory / f'node_step{exp.step:04d}_{invocation}.json', {
                'status': status, 'protocol_id': PROTOCOL_ID, 'candidate_id': args.candidate,
                'run_id': entry['run_id'], 'method': args.method, 'fold': args.fold,
                'seed': args.seed, 'k': args.k, 'start_step': start_step, 'saved_step': exp.step,
                'requested_until': args.until, 'fixed_final_step': TRAINING_TARGET,
                'training_finished': exp.step == TRAINING_TARGET,
                'inner_diagnostic_steps': list(INNER_DIAGNOSTIC_STEPS), 'inner_selection_step': TRAINING_TARGET,
                'evaluation_performed': False, 'scope': exp.scope,
                'effective_optimizer': exp.contract['candidate'],
                'optimizer_contract_verified': True,
                'checkpoint_sha256': checkpoint_hash, 'contract_sha256': sha(contract_path),
                'seconds_this_invocation': time.monotonic() - started,
                'updates_this_invocation': len(records), 'stop_reason': stop['reason'],
                'last_losses': records[-1]['losses'] if records else None,
                'mean_loss_task_this_invocation': sum(r['losses']['loss_task'] for r in records) / len(records) if records else None,
                'all_gradients_finite': all(r['finite_gradients'] for r in records),
                'all_bn_buffers_unchanged': all(r['bn_buffers_unchanged'] for r in records),
                'next_action': 'Export node evidence and stop for review. No next stage was launched.'})
            write_json(directory / 'status.json', {
                'status': status, 'saved_step': exp.step, 'requested_until': args.until,
                'checkpoint_sha256': checkpoint_hash, 'utc': utc_now(),
                'training_finished': exp.step == TRAINING_TARGET, 'evaluation_performed': False,
                'protocol_id': PROTOCOL_ID, 'candidate_id': args.candidate, 'run_id': entry['run_id']})
            print(f'{status}: safely saved step {exp.step}; no subsequent node was started.', flush=True)
        except BaseException as error:
            failure = {'status': 'FAILED_USE_LAST_SAVED_CHECKPOINT', 'safe_saved_step': last_saved,
                       'in_memory_step_do_not_resume': exp.step, 'error': str(error), 'utc': utc_now(),
                       'traceback': traceback.format_exc(), 'invocation': invocation,
                       'protocol_id': PROTOCOL_ID, 'candidate_id': args.candidate, 'run_id': entry['run_id'],
                       'checkpoint_sha256': sha(checkpoint_path) if checkpoint_path.exists() else None}
            write_json(directory / f'failure_{invocation}.json', failure)
            write_json(directory / 'status.json', failure)
            raise
        finally:
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
    return {'status': status, 'saved_step': records[-1]['step'] if records else start_step, 'run_path': relative}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--traceback', action='store_true',
                        help='Print full Python tracebacks instead of the short G09_REJECTED message')
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('train', help='Train or resume one declared G09 candidate run until one explicit node')
    run.add_argument('--method', choices=['erm', 'xdm'], required=True)
    run.add_argument('--fold', type=int, choices=[0, 1], required=True)
    run.add_argument('--candidate', choices=['C01', 'C02', 'C03', 'C04', 'C05', 'C06'], required=True)
    run.add_argument('--seed', type=int, choices=[17], default=17)
    run.add_argument('--k', type=int, choices=[8], default=8)
    run.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    run.add_argument('--until', type=int, required=True,
                     help=f'Absolute completed step; final G09 target is {TRAINING_TARGET}')
    run.add_argument('--run', required=True, help='Must equal the declared runs_g09_v1/... path for this candidate')
    run.add_argument('--max-wall-seconds', type=float, default=1800)
    run.add_argument('--threads', type=int, choices=[4], default=4)
    run.add_argument('--save-every', type=int, default=25)
    evaluation = sub.add_parser('evaluate', help='Inner diagnostics (1000/2000/3000/5000) or gated outer evaluation')
    evaluation.add_argument('--run', required=True)
    evaluation.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    evaluation.add_argument('--split', choices=['inner', 'outer'], required=True)
    evaluation.add_argument('--batch-size', type=int, default=10)
    select = sub.add_parser('select', help='Freeze the per-(method, fold) selection manifest from the six 5000-step inner results')
    select.add_argument('--method', choices=['erm', 'xdm'], required=True)
    select.add_argument('--fold', type=int, choices=[0, 1], required=True)
    compare = sub.add_parser('compare', help='Gated three-way comparison of the selected ERM/XDM outer results')
    compare.add_argument('--fold', type=int, choices=[0, 1], required=True)
    compare.add_argument('--out', required=True)
    gate = sub.add_parser('gate-status', help='Read-only report of the four selection manifests and outer gate state')
    gate.add_argument('--json', action='store_true')
    args = parser.parse_args(argv)
    try:
        return dispatch(args)
    except (RuntimeError, ValueError) as error:
        if args.traceback:
            raise
        print(f'G09_REJECTED: {error}', file=sys.stderr, flush=True)
        print('（完整回溯请加 --traceback）', file=sys.stderr, flush=True)
        return 3


def dispatch(args) -> int:
    if args.command == 'train':
        train(args)
    elif args.command == 'evaluate':
        from g09_evaluate import evaluate
        evaluate(args.run, args.split, args.device, args.batch_size)
    elif args.command == 'select':
        from g09_select import select
        manifest = select(args.method, args.fold)
        print(json.dumps({'winner': manifest['winner'], 'manifest': manifest.get('manifest_path')}, ensure_ascii=False, indent=2))
    elif args.command == 'compare':
        from g09_compare import compare_selected
        report = compare_selected(args.fold, args.out)
        print(json.dumps(report['methods'], ensure_ascii=False, indent=2))
    elif args.command == 'gate-status':
        from g09_gate import frozen_status
        status = {f'{m}_f{f}': {k: v for k, v in entry.items()} for (m, f), entry in frozen_status().items()}
        print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
