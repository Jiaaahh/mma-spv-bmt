"""Recount frozen predictions; distinguish recalculation from archived intervals."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def score(row):
    y = next(row[k] for k in ('class_id', 'true_label', 'y_true') if k in row)
    p = next(row[k] for k in ('predicted_class_id', 'predicted_label', 'y_pred') if k in row)
    value = int(y == p)
    if 'correct' in row and float(row['correct']) != value:
        raise ValueError('Prediction and saved correct flag disagree')
    return value


def summarize_pair(rows, reference, treatment):
    data = {reference: {}, treatment: {}}
    for row in rows:
        if row['method'] not in data:
            continue
        truth = next(row[k] for k in ('class_id', 'true_label', 'y_true') if k in row)
        key = (truth, row['font_id'], row.get('mapping_seed', row.get('seed')))
        if key in data[row['method']]:
            raise ValueError('Duplicate paired observation')
        data[row['method']][key] = score(row)
    if not data[reference] or data[reference].keys() != data[treatment].keys():
        raise ValueError('Unpaired or empty observations')
    counts = [sum(data[m].values()) for m in (reference, treatment)]
    n = len(data[reference])
    return {'methods': [reference, treatment], 'correct': counts, 'n_per_method': n,
            'accuracy_percent': [100 * c / n for c in counts],
            'difference_pp': 100 * (counts[1] - counts[0]) / n}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'checks/main_results')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    for frozen in [ROOT / 'evidence', ROOT / 'assets', ROOT / 'figure_data', ROOT / 'figures', ROOT / 'code', ROOT / 'historical']:
        if out == frozen.resolve() or frozen.resolve() in out.parents:
            raise ValueError('Output must not be written into frozen inputs')
    evidence = ROOT / 'evidence'
    frozen = json.loads((evidence / 'whole_glyph/primary_recheck.json').read_text(encoding='utf-8-sig'))['results']
    report = {'scope': 'Saved-prediction recount and archived-summary export. No training, model inference, bootstrap replay, or new hardware timing.', 'whole_glyph': {}, 'point': {}, 'cpu_latency_archived': {}}
    rows = read_csv(evidence / 'whole_glyph/C1/predictions.csv')
    c1 = summarize_pair(rows, 'ideal_only', 'ideal_plus_multi4')
    rows_e1 = read_csv(evidence / 'whole_glyph/E1/predictions.csv')
    e1 = summarize_pair([r for r in rows_e1 if r['condition'] == 'moderate'], 'proto_skeleton_ideal_only', 'proto_skeleton_ideal_plus_multi4')
    for name, result in [('C1', c1), ('E1', e1)]:
        expected = frozen[name]
        if result['correct'] != expected['correct'] or result['n_per_method'] != expected['total_per_method']:
            raise ValueError(f'{name} counts disagree with archived primary recheck')
        result['archived_ci95_pp_not_recomputed'] = expected['ci95_pp']
        report['whole_glyph'][name] = result
    budget_rows = read_csv(evidence / 'whole_glyph/StudyB/merged_predictions_with_truth.csv')
    budgets = [32, 48, 64, 96, 128]
    budget_results = [summarize_pair([r for r in budget_rows if int(r['budget']) == b], 'ideal_only', 'ideal_plus_multi4') | {'budget': b} for b in budgets]
    values = np.asarray([r['accuracy_percent'] for r in budget_results], dtype=float).T / 100
    weights = np.asarray([8, 16, 24, 32, 16]) / 96
    areas = values @ weights
    delta = float(areas[1] - areas[0])
    if not np.allclose(areas, frozen['StudyB']['naubc'], rtol=0, atol=1e-12):
        raise ValueError('nAUBC mismatch')
    report['whole_glyph']['StudyB'] = {'budgets': budget_results, 'naubc': areas.tolist(), 'delta_naubc': delta, 'archived_ci95_delta_naubc_not_recomputed': frozen['StudyB']['ci95_delta_naubc']}
    mismatch = []
    for name in ['mild', 'moderate', 'severe', 'position_only', 'scale_only']:
        mismatch.append(summarize_pair([r for r in rows_e1 if r['condition'] == name], 'proto_skeleton_ideal_only', 'proto_skeleton_ideal_plus_multi4') | {'condition': name})
    report['whole_glyph']['mismatch'] = mismatch
    # Compare the data used by the current plots, retaining negative gains.
    for expected, actual in zip(read_csv(ROOT / 'figure_data/mismatch_source_data.csv'), mismatch):
        if expected['condition_id'] != actual['condition'] or not np.allclose([float(expected[k]) for k in ['baseline', 'mma', 'delta']], actual['accuracy_percent'] + [actual['difference_pp']], atol=1e-10, rtol=0):
            raise ValueError('Current mismatch figure data disagree with predictions')
    for expected, actual in zip(read_csv(ROOT / 'figure_data/budget_source_data.csv'), budget_results):
        if int(expected['budget']) != actual['budget'] or not np.allclose([float(expected[k]) for k in ['baseline', 'mma', 'delta']], actual['accuracy_percent'] + [actual['difference_pp']], atol=1e-10, rtol=0):
            raise ValueError('Current budget figure data disagree with predictions')
    point_summary = json.loads((evidence / 'point/results/summary.json').read_text(encoding='utf-8-sig'))
    reference = None
    for p in sorted((evidence / 'point/results').glob('*.npz')):
        with np.load(p, allow_pickle=False) as a:
            current = {k: a[k].copy() for k in ['labels', 'fonts', 'rows', 'points', 'mapping_seeds']}
            if reference is None:
                reference = current
            if not all(np.array_equal(current[k], reference[k]) for k in reference):
                raise ValueError('Point method metadata disagree')
            predictions, labels = a['predictions'], a['labels']
            if predictions.shape != (5, 720) or labels.shape != (720,):
                raise ValueError('Unexpected Point panel shape')
            scores = a['scores'] if 'scores' in a.files else a['logits']
            if not np.isfinite(scores).all() or not np.array_equal(scores.argmax(-1), predictions):
                raise ValueError('Point scores disagree with saved predictions')
            correct = int((predictions == labels[None]).sum())
            if correct != point_summary['methods'][p.stem]['correct']:
                raise ValueError('Point count mismatch')
            report['point'][p.stem] = {'correct': correct, 'n': int(predictions.size), 'accuracy_percent': 100 * correct / predictions.size}
    for name in ('ideal', 'multi'):
        vals = [report['point'][f'{name}_seed{s}']['accuracy_percent'] for s in [17, 29, 43]]
        report['point']['point_' + name] = {'mean_accuracy_percent': float(np.mean(vals)), 'seed_accuracy_percent': vals, 'ensemble': False}
    report['point']['archived_contrasts_not_recomputed'] = point_summary['contrasts']
    costs = json.loads((evidence / 'cost_server/cost_summary.json').read_text(encoding='utf-8-sig'))
    report['cpu_latency_archived'] = {k: v for k, v in costs['latency'].items() if k.endswith('_cpu')}
    out.mkdir(parents=True, exist_ok=True)
    (out / 'main_results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'PASS', 'C1_difference_pp': c1['difference_pp'], 'E1_difference_pp': e1['difference_pp'], 'delta_naubc': delta, 'Point_multi_mean_percent': report['point']['point_multi']['mean_accuracy_percent'], 'point_npz_streams': 10, 'output': str(out / 'main_results.json')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
