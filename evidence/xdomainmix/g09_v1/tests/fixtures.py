"""G09-A 测试夹具：全部写在 `engineering_g09_v1/` 之下，绝不创建正式训练状态。

两类夹具：
  * `make_run(...)`：合成 inner 产物（小数组），用于选择器/门禁测试；
  * `make_outer_run(...)`：以**真实 G08 fold1 外评产物**为底，改写 G09 元数据，
    用于把 G09 比较器与 v1 `compare.py` 放在同一批真实预测上对比。
夹具合同由**真实 G09 合同模板**派生（构建一次 tiny 工程 experiment），保证字段结构可信。
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil

import numpy as np

from g09_config import (FOLD_INNER_FONT, FOLD_OUTER_FONT, G09_DIR, INNER_SELECTION_STEP, PROTOCOL_ID, ROOT,
                        SELECTION_RULE_ID, TRAIN_MAPS, candidate as candidate_spec, expected_param_group_plan,
                        resolve_sandbox, run_path, sha, write_json)
from g09_core import OVERRIDDEN_LEGACY_FIELDS, G09Experiment

import core as v1_core  # noqa: E402
import sys  # noqa: E402
sys.path.insert(0, str(G09_DIR / 'tests'))
from engineering_kit import TinyNet  # noqa: E402

OUTER_MAPS = [f'moderate_seed{s}' for s in range(5301, 5306)]


def new_sandbox(name: str) -> dict:
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    root = ROOT / 'engineering_g09_v1' / f'{name}_{stamp}'
    (root / 'runs_g09_v1').mkdir(parents=True, exist_ok=False)
    (root / 'selections_g09_v1').mkdir(parents=True, exist_ok=False)
    # 沙箱根**镜像项目根**：run_path() 返回的 'runs_g09_v1/...' 在沙箱内解析得到同样的相对结构。
    return {'root': root, 'runs': root, 'selections': root / 'selections_g09_v1', 'name': name}


_CONTRACT_TEMPLATE = None


def contract_template() -> dict:
    """用一次 tiny 工程 experiment 生成真实结构的 G09 合同模板（不训练、不落盘）。"""
    global _CONTRACT_TEMPLATE
    if _CONTRACT_TEMPLATE is None:
        exp = G09Experiment('erm', 0, 8, 'C03', device='cpu', scope='ENGINEERING_ONLY',
                            engineering={'tiny_net': TinyNet})
        _CONTRACT_TEMPLATE = json.loads(json.dumps(exp.contract))
    return json.loads(json.dumps(_CONTRACT_TEMPLATE))


def fixture_contract(method: str, fold: int, candidate_id: str, **overrides) -> dict:
    spec = candidate_spec(candidate_id)
    contract = contract_template()
    contract.update({'scope': 'TRAINING_CANDIDATE', 'method': method, 'fold': fold, 'candidate_id': candidate_id,
                     'candidate': {'backbone_lr': spec['backbone_lr'], 'head_lr': spec['head_lr'],
                                   'weight_decay': spec['weight_decay'],
                                   'xdm_domain_lr': spec['xdm_domain_lr'],
                                   'xdm_domain_weight_decay': spec['xdm_domain_weight_decay']}})
    contract['optimizer'].update({'backbone_lr': spec['backbone_lr'], 'head_lr': spec['head_lr'],
                                  'weight_decay': spec['weight_decay'],
                                  'domain_lr_candidate': spec['xdm_domain_lr'],
                                  'domain_weight_decay_candidate': spec['xdm_domain_weight_decay'],
                                  'param_group_plan': expected_param_group_plan(spec)})
    contract['xdm_hparams'].update({'lr': spec['backbone_lr'], 'weight_decay': spec['weight_decay']})
    contract['overridden_legacy_fields'] = list(OVERRIDDEN_LEGACY_FIELDS)
    contract.update(overrides)
    return contract


def _logits_for(correct_target: int, margin: float, labels: np.ndarray) -> np.ndarray:
    """构造 900 个视图的 logits：前 correct_target 个视图正确，其余错误；正确类给 margin 优势。"""
    logits = np.zeros((5, 180, 180), dtype=np.float32)
    flat = logits.reshape(-1, 180)
    targets = np.tile(labels, 5)
    order = np.arange(flat.shape[0])
    correct_rows = order[:correct_target]
    wrong_rows = order[correct_target:]
    flat[correct_rows, targets[correct_rows]] = margin
    flat[wrong_rows, targets[wrong_rows]] = -margin
    flat[wrong_rows, (targets[wrong_rows] + 1) % 180] = margin
    return logits


def fold_rows(fold: int):
    protocol = json.loads((ROOT / 'protocols' / f'fold{fold}.json').read_text(encoding='utf-8'))
    rows = np.asarray(protocol['splits'][fold]['inner_validation'], dtype=np.int64)
    with np.load(ROOT / 'data' / 'metadata.npz', allow_pickle=False) as meta:
        return rows, np.asarray(meta['labels'])[rows], np.asarray(meta['sample_ids'])[rows]


def make_run(sandbox: dict, fold: int, method: str, candidate_id: str, correct_target: int = 400,
             margin: float = 6.8, labels=None, mapping_names=None, steps=(INNER_SELECTION_STEP,),
             contract_overrides=None, write_contract: bool = True) -> Path:
    """在沙箱中合成一个 run 的 inner 产物（可指定步点、正确数、CE 档位与错误元素）。"""
    relative = run_path(fold, method, candidate_id)
    run = sandbox['runs'] / relative
    run.mkdir(parents=True, exist_ok=True)
    contract = fixture_contract(method, fold, candidate_id, **(contract_overrides or {}))
    if write_contract:
        write_json(run / 'contract.json', contract)
    (run / 'latest.pt').write_bytes(f'fixture-checkpoint-{relative}'.encode())
    (run / 'metrics.jsonl').write_text('', encoding='utf-8')
    write_json(run / 'status.json', {'status': 'NODE_REACHED', 'saved_step': 5000, 'requested_until': 5000,
                                     'checkpoint_sha256': sha(run / 'latest.pt'), 'training_finished': True,
                                     'evaluation_performed': False, 'protocol_id': PROTOCOL_ID,
                                     'candidate_id': candidate_id,
                                     'run_id': run_path(fold, method, candidate_id).split('/')[-1]})
    rows, meta_labels, meta_ids = fold_rows(fold)
    truth = np.asarray(labels if labels is not None else meta_labels, dtype=np.int64)
    names = list(mapping_names if mapping_names is not None else TRAIN_MAPS)
    for step in steps:
        logits = _logits_for(correct_target, margin, truth)
        cnn = logits.argmax(-1)
        correctness = cnn == truth[None, :]
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        inner = run / 'evaluations' / f'inner_step{step:04d}_{stamp}'
        inner.mkdir(parents=True, exist_ok=True)
        arrays = {'logits': logits, 'cnn': cnn, 'labels': truth, 'rows': rows,
                  'mapping_names': np.asarray(names), 'font': np.asarray(FOLD_INNER_FONT[fold]),
                  'sample_ids': meta_ids}
        np.savez_compressed(inner / 'predictions.npz', **arrays)
        with (inner / 'predictions.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['row', 'sample_id', 'class_id', 'mapping', 'cnn_prediction', 'cnn_correct'])
            writer.writeheader()
            for index, name in enumerate(names):
                for row in range(180):
                    writer.writerow({'row': row, 'sample_id': f'fixture-{row}', 'class_id': int(truth[row]),
                                     'mapping': name, 'cnn_prediction': int(cnn[index, row]),
                                     'cnn_correct': int(correctness[index, row])})
        np.save(inner / 'confusion_matrix.npy', np.bincount((np.tile(truth, 5) * 180 + cnn.flatten()).astype(np.int64),
                                                            minlength=180 * 180).reshape(180, 180))
        report = {'status': 'INNER_DIAGNOSTIC_ONLY', 'split': 'inner', 'step': step, 'method': method, 'fold': fold,
                  'seed': 17, 'k': 8, 'font': FOLD_INNER_FONT[fold], 'mapping_names': names,
                  'predictions_per_method': 900, 'original_characters': 180,
                  'cnn_accuracy_percent': float(correctness.mean() * 100),
                  'protocol_id': PROTOCOL_ID, 'candidate_id': candidate_id,
                  'run_id': run_path(fold, method, candidate_id).split('/')[-1],
                  'checkpoint_sha256': sha(run / 'latest.pt'), 'contract_sha256': sha(run / 'contract.json'),
                  'fixture': 'SYNTHETIC_TEST_FIXTURE'}
        write_json(inner / 'results.json', report)
        write_json(inner / 'artifact_hashes.json',
                   {p.name: sha(p) for p in inner.iterdir() if p.is_file()})
    return run


def add_outer_dir(run: Path, method: str, fold: int, candidate_id: str, source_outer: Path) -> Path:
    """以真实 G08 外评目录为底，改写为 G09 元数据（预测数组保持真实）。"""
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    outer = run / 'evaluations' / f'outer_step5000_{stamp}'
    shutil.copytree(source_outer, outer)
    results = json.loads((outer / 'results.json').read_text(encoding='utf-8'))
    with np.load(outer / 'predictions.npz', allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    labels = arrays['labels']
    results.update({'protocol_id': PROTOCOL_ID, 'candidate_id': candidate_id, 'run_id': run.name,
                    'method': method, 'fold': fold, 'seed': 17, 'k': 8, 'step': 5000, 'split': 'outer',
                    'font': str(arrays['font']), 'mapping_names': list(map(str, arrays['mapping_names'])),
                    'predictions_per_method': 900, 'original_characters': 180,
                    'contract_sha256': sha(run / 'contract.json'),
                    'checkpoint_sha256': sha(run / 'latest.pt'),
                    'cnn_accuracy_percent': float((arrays['cnn'] == labels[None, :]).mean() * 100),
                    'matched_mma_accuracy_percent': float((arrays['mma'] == labels[None, :]).mean() * 100),
                    'fixture': 'REAL_G08_PREDICTIONS_WITH_G09_METADATA'})
    write_json(outer / 'results.json', results)
    write_json(outer / 'artifact_hashes.json', {p.name: sha(p) for p in outer.iterdir() if p.is_file()})
    return outer
