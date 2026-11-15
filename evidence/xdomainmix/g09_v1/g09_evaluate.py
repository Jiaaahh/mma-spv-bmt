"""G09-A 评价入口：inner 诊断（含 5000 步选择输入）与受门禁约束的 outer 外评。

与 v1 `evaluate.py` 的差别：
  * 用 `G09Experiment`（候选有效优化器）+ G09 合同校验；
  * inner 只允许 1000/2000/3000/5000 步，其中 5000 步是**唯一选择输入**；
  * outer 必须先通过 `g09_gate.require_outer_allowed`：四个 selection_manifest 全部冻结、
    run 必须是该 (method, fold) 的胜出模型（逐候选跑 outer 一律拒绝）；
  * 报告保留 v1 字段（便于 `compare.py` 交叉核对），并附 `protocol_id/candidate_id/run_id`。
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

from g09_config import INNER_DIAGNOSTIC_STEPS, INNER_SELECTION_STEP, PROTOCOL_ID, ROOT, matrix_index, sha, write_json
from g09_core import G09Experiment

import core as v1_core  # noqa: E402
from g09_gate import require_outer_allowed  # noqa: E402

INNER_ALLOWED_STEPS = sorted(set(INNER_DIAGNOSTIC_STEPS) | {INNER_SELECTION_STEP})


def normalized(values):
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(norms <= 0) or not np.isfinite(norms).all():
        raise ValueError('Zero or nonfinite HOG/prototype norm')
    return values / norms


def resolve_g09_run(run_value: str) -> tuple:
    """run 必须是运行矩阵里声明的 G09 run；拒绝 v1 runs/ 与任意自造目录。"""
    relative = Path(run_value).as_posix().strip('/')
    index = matrix_index()
    if relative not in index:
        raise RuntimeError(f'Not a declared G09 run: {relative!r}. Allowed pattern: runs_g09_v1/f{{0,1}}_s17_k8_{{erm,xdm}}_C0{{1..6}}')
    entry = index[relative]
    run = ROOT / relative
    if not (run / 'contract.json').is_file():
        raise RuntimeError(f'{relative}: missing contract.json; training has not started for this run')
    return relative, entry, run


def evaluate(run_value: str, split: str, device: str, batch_size: int = 10, allow_outer: bool = True) -> dict:
    if split not in ('inner', 'outer'):
        raise ValueError("split must be 'inner' or 'outer'")
    if batch_size < 1:
        raise ValueError('Evaluation batch size must be positive')
    relative, entry, run = resolve_g09_run(run_value)
    contract = __import__('json').loads((run / 'contract.json').read_text(encoding='utf-8'))
    from g09_validation import validate_contract
    validate_contract(contract, entry['method'], entry['fold'], entry['candidate_id'])
    if contract.get('protocol_id') != PROTOCOL_ID:
        raise RuntimeError(f'{relative}: contract is not a G09 contract')
    if contract.get('scope') != 'TRAINING_CANDIDATE':
        raise RuntimeError(f'{relative}: engineering scope cannot be evaluated as a scientific run')
    if split == 'outer':
        if not allow_outer:
            raise RuntimeError('outer evaluation disabled by caller')
        require_outer_allowed(contract['method'], contract['fold'], relative)  # 硬门禁：未冻结/非胜出一律阻断
    with v1_core.RunLock(run):
        exp = G09Experiment(contract['method'], contract['fold'], contract['k'], contract['candidate_id'],
                            seed=contract['seed'], device=device, threads=contract['runtime']['threads'])
        if exp.contract != contract:
            raise RuntimeError('Evaluation contract differs from on-disk contract')
        exp.restore_g09(torch.load(run / 'latest.pt', map_location='cpu', weights_only=True))
        if exp.step == 0:
            raise RuntimeError('No fine-tuning completed; this evaluation stage is not available')
        if split == 'outer':
            if exp.step != INNER_SELECTION_STEP:
                raise RuntimeError('Outer-development evaluation is blocked until the fixed final step 5000')
        else:
            if exp.step not in INNER_ALLOWED_STEPS:
                raise RuntimeError(f'Inner evaluation is only allowed at steps {INNER_ALLOWED_STEPS}; current step {exp.step}')
        names = v1_core.TRAIN_MAPS if split == 'inner' else v1_core.OUTER_MAPS
        rows = np.asarray(exp.split['inner_validation' if split == 'inner' else 'outer_development'], dtype=np.int64)
        truth = exp.meta['labels'][rows]
        if len(rows) != 180 or sorted(truth.tolist()) != list(range(180)):
            raise RuntimeError('Expected one held-out sample per character class')
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        destination = run / 'evaluations' / f'{split}_step{exp.step:04d}_{stamp}'
        destination.mkdir(parents=True, exist_ok=False)
        all_logits, all_pred, sample_records = [], [], []
        exp.net.eval()
        with torch.inference_mode():
            for name in names:
                images = np.load(ROOT / 'data' / f'{name}_images.npy', mmap_mode='r')
                chunks = []
                for offset in range(0, len(rows), batch_size):
                    current = rows[offset:offset + batch_size]
                    logits = exp.predict(v1_core.preprocess(images[current])).cpu()
                    if logits.shape != (len(current), 180) or not bool(torch.isfinite(logits).all()):
                        raise FloatingPointError('Invalid evaluation logits')
                    chunks.append(logits.numpy())
                matrix = np.concatenate(chunks, axis=0)
                all_logits.append(matrix); all_pred.append(matrix.argmax(axis=1))
        logits = np.stack(all_logits); predictions = np.stack(all_pred)
        correctness = predictions == truth[None, :]
        confidence = torch.from_numpy(logits).softmax(-1).max(-1).values.numpy()
        report = {'status': 'INNER_DIAGNOSTIC_ONLY' if split == 'inner' else 'COMPLETE_SINGLE_FOLD_DEVELOPMENT_COMPARISON',
                  'invocation_utc': datetime.now(timezone.utc).isoformat(), 'argv': sys.argv, 'cwd': str(Path.cwd()),
                  'protocol_id': PROTOCOL_ID, 'candidate_id': contract['candidate_id'], 'run_id': relative.split('/')[-1],
                  'method': exp.method, 'fold': exp.fold, 'seed': exp.seed, 'k': exp.k,
                  'step': exp.step, 'split': split, 'mapping_names': names,
                  'font': exp.split['inner_validation_font' if split == 'inner' else 'heldout_font'],
                  'predictions_per_method': int(predictions.size), 'original_characters': 180,
                  'cnn_accuracy_percent': float(correctness.mean() * 100),
                  'cross_entropy': float(F.cross_entropy(torch.from_numpy(logits.reshape(-1, 180)), torch.from_numpy(np.tile(truth, 5)).long())),
                  'per_mapping_accuracy_percent': {n: float(v.mean() * 100) for n, v in zip(names, correctness)},
                  'predicted_class_count': int(np.unique(predictions).size),
                  'dominant_predicted_class_fraction': float(np.bincount(predictions.flatten(), minlength=180).max() / predictions.size),
                  'mean_max_softmax': float(confidence.mean()),
                  'checkpoint_sha256': sha(run / 'latest.pt'), 'contract_sha256': sha(run / 'contract.json'),
                  'effective_optimizer': contract['candidate'],
                  'training_data_original_glyphs': 720, 'training_views': 3600,
                  'checkpoint_selection': ('Inner diagnostics are not used to pick a checkpoint; the 5000-step inner is the '
                                           'only candidate-selection input; outer uses the fixed final 5000 checkpoint.'),
                  'limitation': 'One initialization and one development font; five mappings are repeated observations of 180 characters, not 900 independent character samples. No superiority or convergence guarantee.'}
        arrays = {'logits': logits, 'cnn': predictions, 'labels': truth, 'rows': rows, 'mapping_names': np.asarray(names),
                  'font': np.asarray(report['font']), 'sample_ids': exp.meta['sample_ids'][rows]}
        mma = None
        if split == 'outer':
            train = exp.trainrows; labels = exp.meta['labels']
            means = []
            for name in v1_core.TRAIN_MAPS:
                hog = np.load(ROOT / 'data' / f'{name}_hog.npy', mmap_mode='r')
                means.append(np.stack([hog[train[labels[train] == c]].astype(np.float64).mean(0) for c in range(180)]))
            prototypes = normalized(np.mean(means, axis=0))
            reference = ROOT / 'reference' / f'fold{exp.fold}_original_mma_prototypes.npy'
            if not reference.is_file():
                raise RuntimeError('Missing original MMA prototype reference; do not substitute a new classifier')
            manifest = __import__('json').loads((ROOT / 'PACKAGE_MANIFEST.json').read_text(encoding='utf-8'))
            hashes = {item['path']: item['sha256'] for item in manifest['files']}
            if hashes.get(reference.relative_to(ROOT).as_posix()) != sha(reference):
                raise RuntimeError('MMA reference prototype package hash mismatch')
            np.testing.assert_allclose(prototypes, np.load(reference), rtol=1e-13, atol=1e-14)
            np.save(destination / 'matched_mma_prototypes.npy', prototypes)
            scores = np.stack([normalized(np.load(ROOT / 'data' / f'{name}_hog.npy', mmap_mode='r')[rows].astype(np.float64)) @ prototypes.T
                               for name in names])
            mma = scores.argmax(-1)
            differences = ((mma == truth[None, :]).astype(float) - correctness).mean(0) * 100
            rng = np.random.default_rng(90917)
            weights = rng.multinomial(180, np.full(180, 1 / 180), size=20000) / 180
            report.update(matched_mma_accuracy_percent=float((mma == truth[None, :]).mean() * 100),
                          mma_minus_cnn_pp=float(differences.mean()),
                          conditional_character_95ci_pp=np.quantile(weights @ differences, [.025, .975]).tolist(),
                          bootstrap={'unit': 'character with all 5 mappings retained', 'replicates': 20000, 'seed': 90917,
                                     'conditional_on': 'this one font, this trained checkpoint, these five fixed mapping seeds'},
                          mma_original_prototype_equivalence=True,
                          mma_definition='Average raw HOG over same 4 training fonts per class per mapping, then mean over 5 mappings, then L2 normalize; normalized test HOG cosine similarity')
            arrays.update(mma=mma, mma_cosine_scores=scores, per_character_mma_minus_cnn_pp=differences)
        for mapping_index, name in enumerate(names):
            for sample_index, row in enumerate(rows):
                record = {'row': int(row), 'sample_id': str(exp.meta['sample_ids'][row]),
                          'class_id': int(truth[sample_index]), 'character': str(exp.meta['characters'][row]),
                          'font': str(exp.meta['fonts'][row]), 'mapping': name,
                          'cnn_prediction': int(predictions[mapping_index, sample_index]),
                          'cnn_correct': int(correctness[mapping_index, sample_index]),
                          'cnn_max_softmax': float(confidence[mapping_index, sample_index])}
                if mma is not None:
                    record.update(mma_prediction=int(mma[mapping_index, sample_index]),
                                  mma_correct=int(mma[mapping_index, sample_index] == truth[sample_index]))
                sample_records.append(record)
        np.savez_compressed(destination / 'predictions.npz', **arrays)
        with (destination / 'predictions.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(sample_records[0])); writer.writeheader(); writer.writerows(sample_records)
        np.save(destination / 'confusion_matrix.npy',
                np.bincount((np.tile(truth, 5) * 180 + predictions.flatten()).astype(np.int64), minlength=180 * 180).reshape(180, 180))
        write_json(destination / 'results.json', report)
        write_json(destination / 'artifact_hashes.json', {p.name: sha(p) for p in destination.iterdir() if p.is_file()})
        print(f'{split} evaluation written: {destination}', flush=True)
        return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, help='Declared G09 run path, e.g. runs_g09_v1/f1_s17_k8_erm_C03')
    parser.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    parser.add_argument('--split', choices=['inner', 'outer'], required=True)
    parser.add_argument('--batch-size', type=int, default=10)
    args = parser.parse_args(argv)
    evaluate(args.run, args.split, args.device, args.batch_size)
    return 0


if __name__ == '__main__':
    sys.exit(main())
