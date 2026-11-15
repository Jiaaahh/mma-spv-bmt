"""G09-A 内验选择器：每 (method, fold) 独立选优并冻结。

排序规则（固定，写入 selection_manifest）：
  1. 5000 步 inner 的 900 视图正确总数**降序**；
  2. 并列时用同一冻结 CPU 分析器，从保存的 float32 logits 转 float64、稳定 log-sum-exp
     复算平均交叉熵，按十进制 ROUND_HALF_EVEN 保留 8 位小数后**升序**（原精确值同时保存）；
  3. 仍并列时 candidate_id 字典序**升序**。
只接受该 (method, fold) 的 6 份 5000 步 inner 产物；拒绝 outer、其他折、早期 inner、
缺失/重复候选、错误标签与映射。**不读取任何 outer 分数**，直到四个记录全部冻结。
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_EVEN
from pathlib import Path

import numpy as np

from g09_config import (EXPECTED_CANDIDATES, EXPECTED_METHODS, EXPECTED_FOLDS, FOLD_INNER_FONT,
                        INNER_DIAGNOSTIC_STEPS, INNER_SELECTION_STEP, PROTOCOL_ID, ROOT, SELECTION_DIR_NAME,
                        SELECTION_RULE_ID, TRAIN_MAPS, matrix_index, resolve_sandbox, run_path, sha, write_json)

INNER_FILES = ('results.json', 'predictions.npz', 'predictions.csv', 'confusion_matrix.npy', 'artifact_hashes.json')
ROUNDING_QUANT = Decimal('1e-8')
ROUNDING_CONVENTION = ('exact binary float64 value -> Decimal(ROUND_HALF_EVEN, 8dp); '
                       'the shortest-repr value is stored alongside for transparency')


class SelectionError(RuntimeError):
    pass


def _require(condition, message):
    if not condition:
        raise SelectionError(message)


def stable_cross_entropy_float64(logits, labels) -> float:
    """稳定 log-sum-exp 的平均交叉熵（float64；输入为保存的 float32 logits）。"""
    x = np.asarray(logits, dtype=np.float64)
    _require(x.ndim == 3 and x.shape[1] == 180 and x.shape[2] == 180, f'unexpected logits shape {x.shape}')
    _require(np.isfinite(x).all(), 'nonfinite logits')
    flat = x.reshape(-1, x.shape[-1])
    target = np.tile(np.asarray(labels, dtype=np.int64), x.shape[0])
    _require(target.shape[0] == flat.shape[0], 'label/logit length mismatch')
    maximum = flat.max(axis=1, keepdims=True)
    logsumexp = maximum[:, 0] + np.log(np.exp(flat - maximum).sum(axis=1))
    return float(np.mean(logsumexp - flat[np.arange(flat.shape[0]), target]))


def round_rhe8(value: float) -> str:
    return str(Decimal(value).quantize(ROUNDING_QUANT, rounding=ROUND_HALF_EVEN))


def round_repr8(value: float) -> str:
    return str(Decimal(repr(float(value))).quantize(ROUNDING_QUANT, rounding=ROUND_HALF_EVEN))


def read_inner(run_relative: str, method: str, fold: int, candidate_id: str, step: int = INNER_SELECTION_STEP,
               run_root=None, *, revalidate_frozen=False) -> dict:
    """读取并独立复算一个候选的 inner 产物；任何不合规输入都以明确信息拒绝。

    `run_root` 只允许工程沙箱（见 `resolve_sandbox`）；正式调用不传，使用 runs_g09_v1/。
    """
    run = resolve_sandbox(run_root, ROOT) / run_relative
    contract_path = run / 'contract.json'
    _require(contract_path.is_file(), f'{run_relative}: missing contract.json')
    contract = json.loads(contract_path.read_text(encoding='utf-8'))
    from g09_validation import validate_contract
    validate_contract(contract, method, fold, candidate_id)
    _require(run_relative == run_path(fold, method, candidate_id), 'Noncanonical run path')
    _require(step == INNER_SELECTION_STEP, 'Selection only accepts step 5000')
    _require(contract.get('protocol_id') == PROTOCOL_ID, f'{run_relative}: not a G09 contract')
    _require(contract.get('candidate_id') == candidate_id, f'{run_relative}: contract candidate mismatch')
    _require(contract.get('method') == method and contract.get('fold') == fold, f'{run_relative}: contract method/fold mismatch')
    _require(run_relative.split('/')[-1] == run_path(fold, method, candidate_id).split('/')[-1],
             f'{run_relative}: path does not match the declared G09 run id for this candidate')
    _require(contract.get('training_target') == 5000, f'{run_relative}: training_target must be 5000')
    evaluation_root = run / 'evaluations'
    _require(evaluation_root.is_dir(), f'{run_relative}: no evaluations directory')
    names = sorted(p.name for p in evaluation_root.iterdir() if p.is_dir())
    inners = [n for n in names if n.startswith(f'inner_step{step:04d}_')]
    early = sorted({n.split('_')[1] for n in names if n.startswith('inner_step')} - {f'step{step:04d}'})
    outers = [n for n in names if n.startswith('outer_step')]
    _require(revalidate_frozen or not outers, f'{run_relative}: outer evaluation directories are not valid selection inputs {outers}')
    if not inners:
        if early:
            raise SelectionError(f'{run_relative}: only early inner diagnostics {early} present; '
                                 f'the {step}-step inner is the only selection input')
        raise SelectionError(f'{run_relative}: missing inner_step{step:04d} evaluation')
    _require(len(inners) == 1, f'{run_relative}: duplicate inner_step{step:04d} directories {inners}')
    inner = evaluation_root / inners[0]
    files = {name: inner / name for name in INNER_FILES}
    for name, path in files.items():
        _require(path.is_file(), f'{run_relative}: inner artifact missing: {name}')
    hashes = json.loads(files['artifact_hashes.json'].read_text(encoding='utf-8'))
    for name in INNER_FILES:
        if name == 'artifact_hashes.json':
            continue
        _require(name in hashes, f'{run_relative}: artifact_hashes.json lacks {name}')
        _require(sha(files[name]) == hashes[name], f'{run_relative}: inner artifact hash mismatch: {name}')
    report = json.loads(files['results.json'].read_text(encoding='utf-8'))
    _require(report.get('split') == 'inner', f'{run_relative}: selection input must be an inner evaluation')
    _require(report.get('status') == 'INNER_DIAGNOSTIC_ONLY', f'{run_relative}: unexpected inner status {report.get("status")!r}')
    _require(report.get('step') == step, f'{run_relative}: inner step {report.get("step")!r} != {step}')
    _require(report.get('method') == method and report.get('fold') == fold, f'{run_relative}: inner method/fold mismatch')
    _require(report.get('protocol_id') == PROTOCOL_ID, f'{run_relative}: inner report protocol mismatch')
    _require(report.get('candidate_id') == candidate_id, f'{run_relative}: inner report candidate mismatch')
    _require(report.get('font') == FOLD_INNER_FONT[fold],
             f'{run_relative}: inner font {report.get("font")!r} is not fold{fold} inner font {FOLD_INNER_FONT[fold]!r}')
    _require(report.get('mapping_names') == TRAIN_MAPS, f'{run_relative}: inner mapping list mismatch')
    _require(report.get('predictions_per_method') == 900 and report.get('original_characters') == 180,
             f'{run_relative}: inner must report 900 views over 180 characters')
    _require(report.get('contract_sha256') == sha(contract_path), f'{run_relative}: report/contract hash mismatch')
    _require(report.get('checkpoint_sha256') == sha(run / 'latest.pt'),
             f'{run_relative}: evaluated checkpoint is no longer the current latest.pt')
    with np.load(files['predictions.npz'], allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    for key in ('logits', 'cnn', 'labels', 'rows', 'mapping_names', 'font', 'sample_ids'):
        _require(key in arrays, f'{run_relative}: predictions.npz missing {key}')
    logits, cnn, labels = arrays['logits'], arrays['cnn'], arrays['labels']
    _require(logits.dtype == np.float32, 'Saved selection logits must be float32')
    _require(logits.shape == (5, 180, 180), f'{run_relative}: logits shape {logits.shape}')
    _require(cnn.shape == (5, 180) and logits.argmax(-1).shape == cnn.shape, f'{run_relative}: prediction shape mismatch')
    _require(np.array_equal(logits.argmax(-1), cnn), f'{run_relative}: stored cnn != logits argmax')
    _require(np.isfinite(logits).all(), f'{run_relative}: nonfinite logits')
    _require(labels.shape == (180,), f'{run_relative}: labels shape {labels.shape}')
    label_set = np.unique(labels).tolist()
    _require(label_set == list(range(180)), f'{run_relative}: label support must be exactly 0..179 (got {label_set[:5]}...)')
    _require(arrays['rows'].shape == (180,) and len(np.unique(arrays['rows'])) == 180, f'{run_relative}: duplicate/missing rows')
    _require(arrays['sample_ids'].shape == (180,) and len(np.unique(arrays['sample_ids'])) == 180,
             f'{run_relative}: duplicate/missing sample ids')
    # 与冻结划分/metadata 交叉核对：行、标签、样本 ID、内验字体必须逐项对应本折 inner 划分
    protocol = json.loads((ROOT / 'protocols' / f'fold{fold}.json').read_text(encoding='utf-8'))
    expected_rows = np.asarray(protocol['splits'][fold]['inner_validation'], dtype=np.int64)
    _require(np.array_equal(arrays['rows'], expected_rows),
             f'{run_relative}: rows must equal the fold{fold} inner-validation split of the frozen protocol')
    with np.load(ROOT / 'data' / 'metadata.npz', allow_pickle=False) as meta:
        meta_labels = np.asarray(meta['labels'])[expected_rows]
        meta_ids = [str(x) for x in meta['sample_ids'][expected_rows]]
        meta_fonts = sorted(set(map(str, meta['fonts'][expected_rows])))
    _require(np.array_equal(labels, meta_labels),
             f'{run_relative}: labels must equal metadata labels for the fold{fold} inner split')
    _require([str(x) for x in arrays['sample_ids']] == meta_ids,
             f'{run_relative}: sample ids must match the metadata rows of the fold{fold} inner split')
    _require(meta_fonts == [FOLD_INNER_FONT[fold]],
             f'{run_relative}: metadata inner font {meta_fonts} != expected {FOLD_INNER_FONT[fold]!r}')
    _require(list(map(str, arrays['mapping_names'])) == TRAIN_MAPS, f'{run_relative}: npz mapping order mismatch')
    _require(arrays['font'].shape == () and str(arrays['font']) == report['font'], f'{run_relative}: npz font mismatch')
    correct = int((cnn == labels[None, :]).sum())
    reported = float(report.get('cnn_accuracy_percent', float('nan')))
    _require(abs(correct / 900 * 100 - reported) <= 1e-9,
             f'{run_relative}: reported accuracy {reported} inconsistent with sample predictions')
    ce = stable_cross_entropy_float64(logits, labels)
    return {'candidate_id': candidate_id, 'run_path': run_relative, 'inner_dir': inners[0],
            'correct': correct, 'total': 900, 'accuracy_percent': correct / 900 * 100,
            'cross_entropy_float64': ce, 'cross_entropy_rhe8': round_rhe8(ce),
            'cross_entropy_repr8': round_repr8(ce),
            'files': {name: sha(files[name]) for name in INNER_FILES if name != 'artifact_hashes.json'},
            'artifact_hashes_sha256': sha(files['artifact_hashes.json']),
            'contract_sha256': sha(contract_path),
            'checkpoint_sha256': report['checkpoint_sha256']}


def rank_candidates(records: list) -> list:
    """固定排序：正确数降序 → CE(8dp ROUND_HALF_EVEN) 升序 → candidate_id 升序。"""
    by_id = {}
    for record in records:
        _require(record['candidate_id'] not in by_id, f'duplicate candidate {record["candidate_id"]}')
        by_id[record['candidate_id']] = record
    _require(set(by_id) == set(EXPECTED_CANDIDATES), 'Candidate set must be exactly C01..C06')
    _require(len(by_id) == len(EXPECTED_CANDIDATES), f'expected {len(EXPECTED_CANDIDATES)} distinct candidates, got {len(by_id)}')

    def key(candidate_id):
        record = by_id[candidate_id]
        return (-record['correct'], Decimal(record['cross_entropy_rhe8']), candidate_id)

    ordered = sorted(by_id, key=key)
    ranking = []
    for position, candidate_id in enumerate(ordered, 1):
        record = by_id[candidate_id]
        decider = []
        if position > 1:
            previous = by_id[ordered[position - 2]]
            if record['correct'] != previous['correct']:
                decider.append('correct_count')
            elif record['cross_entropy_rhe8'] != previous['cross_entropy_rhe8']:
                decider.append('cross_entropy_rhe8')
            else:
                decider.append('candidate_id')
        ranking.append({'rank': position, **{k: record[k] for k in
                        ('candidate_id', 'run_path', 'inner_dir', 'correct', 'total', 'accuracy_percent',
                         'cross_entropy_float64', 'cross_entropy_rhe8', 'cross_entropy_repr8')},
                        'decided_over_previous_by': decider})
    return ranking


def select(method: str, fold: int, write: bool = True, candidates_order=None, run_root=None, out_dir=None) -> dict:
    _require(method in EXPECTED_METHODS, f'unknown method {method!r}')
    _require(fold in EXPECTED_FOLDS, f'unknown fold {fold!r}')
    order = list(candidates_order) if candidates_order is not None else list(EXPECTED_CANDIDATES)
    _require(sorted(order) == sorted(EXPECTED_CANDIDATES),
             f'selection requires exactly the six frozen candidates, got {sorted(order)}')
    index = matrix_index()
    records, inputs = [], {}
    for candidate_id in sorted(order):
        relative = run_path(fold, method, candidate_id)
        _require(relative in index, f'{relative} is not a declared G09 run')
        record = read_inner(relative, method, fold, candidate_id, run_root=run_root)
        records.append(record)
        inputs[candidate_id] = {'run_path': relative, 'inner_dir': record['inner_dir'],
                                'files': record['files'], 'artifact_hashes_sha256': record['artifact_hashes_sha256'],
                                'contract_sha256': record['contract_sha256'],
                                'checkpoint_sha256': record['checkpoint_sha256']}
    ranking = rank_candidates(records)
    winner = ranking[0]
    manifest = {
        'schema': 2,
        'protocol_id': PROTOCOL_ID,
        'selection_rule_id': SELECTION_RULE_ID,
        'method': method, 'fold': fold,
        'inner_font': FOLD_INNER_FONT[fold],
        'selection_step': INNER_SELECTION_STEP,
        'diagnostic_steps_not_used_for_selection': INNER_DIAGNOSTIC_STEPS,
        'required_candidates': len(EXPECTED_CANDIDATES),
        'ordering_rule': ['correct_count_descending',
                          'cross_entropy_float64_rhe8_ascending',
                          'candidate_id_ascending'],
        'cross_entropy_definition': ('stable float64 log-sum-exp mean cross entropy over the 900 stored float32 logits; '
                                     'same 0..179 label support and mapping order as the evaluation artifact'),
        'cross_entropy_rounding': ROUNDING_CONVENTION,
        'cross_fold_selection': False,
        'outer_scores_read': False,
        'input_hashes': inputs,
        'ranking': ranking,
        'winner': {k: winner[k] for k in ('candidate_id', 'run_path', 'inner_dir', 'correct', 'total',
                                          'accuracy_percent', 'cross_entropy_float64', 'cross_entropy_rhe8')},
        'frozen': True,
        'frozen_utc': datetime.now(timezone.utc).isoformat(),
        'note': ('Winner is the unique top of the fixed rule. Outer scores were not read at selection time; '
                 'outer access requires all four selection manifests to be frozen and is limited to this run.'),
    }
    from g09_gate import selection_digest
    manifest['integrity_sha256'] = selection_digest(manifest)
    if write:
        out = resolve_sandbox(out_dir, ROOT / SELECTION_DIR_NAME) / f'{method}_f{fold}.json'
        if out.exists():
            existing = json.loads(out.read_text(encoding='utf-8'))
            same_core = {k: v for k, v in existing.items() if k not in ('frozen_utc',)} == \
                        {k: v for k, v in manifest.items() if k not in ('frozen_utc',)}
            raise SelectionError(f'selection manifest already exists: {out.relative_to(ROOT).as_posix()} '
                                 f'(identical={same_core}); frozen records are never overwritten')
        write_json(out, manifest)
        manifest['manifest_path'] = out.relative_to(ROOT).as_posix()
        manifest['manifest_sha256'] = sha(out)
    return manifest
