"""Verify a frozen evidence bundle; never replay models or overwrite inputs.

Derived from evidence_bundle/verify_bundle.py (2026-10-04). Python 3.9+
and NumPy are required for the optional recorded-cost/nAUBC calculations.
"""
from pathlib import Path
from collections import defaultdict
import argparse
import csv
import hashlib
import json
import math
import sys


LABEL_PAIRS = [
    ('y_true', 'y_pred', 'correct'),
    ('true_label', 'predicted_label', 'correct'),
    ('class_id', 'predicted_class_id', 'correct'),
    ('class_id', 'predicted_class', 'correct'),
    ('class_id', 'predicted', 'correct'),
    ('truth', 'predicted', 'correct'),
    ('class_id', 'cnn_prediction', 'cnn_correct'),
    ('class_id', 'mma_prediction', 'mma_correct'),
]
GROUP_COLUMNS = [
    'panel', 'split', 'variant', 'method', 'condition', 'budget',
    'projection_seed',
]
PREDICTION_COLUMNS = {b for _, b, _ in LABEL_PAIRS}


def binary_flag(value):
    number = float(value)
    if not math.isfinite(number) or number not in (0, 1):
        raise ValueError('correctness flag must be 0 or 1')
    return int(number)


def grouped_score(rows, group_columns, pair=None, flag=None):
    groups = defaultdict(lambda: [0, 0])
    disagreements = []
    for row_number, row in enumerate(rows, start=2):
        if None in row or any(v is None for v in row.values()):
            raise ValueError('malformed CSV row at line ' + str(row_number))
        if pair:
            if not row[pair[0]].strip() or not row[pair[1]].strip():
                raise ValueError('empty label at line ' + str(row_number))
            correct = int(row[pair[0]] == row[pair[1]])
            if flag and correct != binary_flag(row[flag]):
                disagreements.append(row_number)
        else:
            correct = binary_flag(row[flag])
        key = tuple(row[column] for column in group_columns)
        groups[key][0] += correct
        groups[key][1] += 1
    return {
        'label_columns': list(pair) if pair else None,
        'flag_column': flag,
        'rows': len(rows),
        'correct': sum(value[0] for value in groups.values()),
        'flag_disagreements': len(disagreements),
        'disagreement_lines_first_20': disagreements[:20],
        'groups': [
            {'group': dict(zip(group_columns, key)), 'correct': values[0],
             'n': values[1], 'accuracy_percent': 100 * values[0] / values[1]}
            for key, values in sorted(groups.items())
        ],
    }


def verify(bundle_root):
    manifest_path = bundle_root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if not isinstance(manifest, list):
        raise ValueError('manifest must contain a list of file records')
    report = {
        'scope': 'Frozen-file SHA-256 checks and recorded-prediction scoring; '
                 'no model replay, training, bootstrap CI replication, or hardware benchmark.',
        'path_base': '--bundle-root',
        'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'hash_checks': [], 'independent_label_scoring': [],
        'correct_flag_only': [], 'hash_only': [], 'skipped': [], 'failed': [],
        'numerical_checks': {},
    }
    seen = set()
    for record in manifest:
        relative = str(record.get('path', ''))
        try:
            if not relative or relative in seen:
                raise ValueError('missing or duplicate manifest path')
            seen.add(relative)
            path = (bundle_root / relative).resolve()
            if not path.is_relative_to(bundle_root):
                raise ValueError('manifest path escapes bundle root')
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            matched = actual == record.get('sha256')
            report['hash_checks'].append({'path': relative, 'matched': matched})
            if not matched:
                raise ValueError('SHA-256 mismatch; prediction scoring not attempted')
            if path.suffix.lower() != '.csv':
                report['hash_only'].append({'path': relative, 'reason': 'non-CSV frozen evidence'})
                continue
            with path.open(encoding='utf-8-sig', newline='') as stream:
                reader = csv.DictReader(stream)
                fields = reader.fieldnames or []
                rows = list(reader)
            if len(set(fields)) != len(fields):
                raise ValueError('duplicate CSV column names')
            pairs = [(a, b, flag if flag in fields else None)
                     for a, b, flag in LABEL_PAIRS if a in fields and b in fields]
            is_prediction_file = 'prediction' in path.stem.lower()
            if pairs:
                if not rows:
                    raise ValueError('recognizable prediction CSV is empty')
                group_columns = [column for column in GROUP_COLUMNS if column in fields]
                scores = [grouped_score(rows, group_columns, (a, b), flag)
                          for a, b, flag in pairs]
                item = {'path': relative, 'rows': len(rows),
                        'group_columns': group_columns, 'scores': scores}
                report['independent_label_scoring'].append(item)
                if any(score['flag_disagreements'] for score in scores):
                    raise ValueError('stored correctness flags disagree with independently compared labels')
            elif is_prediction_file:
                flag = next((name for name in ('correct', 'cnn_correct') if name in fields), None)
                if not rows:
                    raise ValueError('prediction CSV is empty')
                if flag and not PREDICTION_COLUMNS.intersection(fields):
                    group_columns = [column for column in GROUP_COLUMNS if column in fields]
                    report['correct_flag_only'].append({
                        'path': relative, 'reason': 'no recognized truth/prediction pair; stored flags only',
                        'group_columns': group_columns,
                        'score': grouped_score(rows, group_columns, flag=flag),
                    })
                else:
                    report['skipped'].append({'path': relative, 'reason': 'unrecognized prediction schema'})
                    raise ValueError('prediction-like CSV cannot be independently scored')
            elif PREDICTION_COLUMNS.intersection(fields):
                report['skipped'].append({'path': relative, 'reason': 'unpaired prediction column'})
                raise ValueError('unpaired prediction column; schema needs explicit review')
            else:
                report['hash_only'].append({
                    'path': relative,
                    'reason': 'CSV summary/metadata; no recognized per-sample label pair',
                })
        except Exception as error:
            # Do not leak machine-specific absolute paths through OS exceptions.
            message = str(error) if isinstance(error, ValueError) else type(error).__name__
            report['failed'].append({'path': relative, 'reason': message})

    numerical_checks(bundle_root, seen, report)
    report['summary'] = {
        'manifest_entries': len(manifest),
        'hashes_passed': sum(item['matched'] for item in report['hash_checks']),
        'independent_label_csv_files': len(report['independent_label_scoring']),
        'independent_prediction_streams': sum(len(item['scores']) for item in report['independent_label_scoring']),
        'correct_flag_only_csv_files': len(report['correct_flag_only']),
        'hash_only_files': len(report['hash_only']),
        'skipped_files': len(report['skipped']),
        'failure_count': len(report['failed']),
        'passed': not report['failed'],
    }
    return report


def numerical_checks(root, manifested_paths, report):
    checks = report['numerical_checks']
    paths = ['randumb/costs.json', 'whole_glyph/StudyB/merged_predictions_with_truth.csv']
    verified = {entry['path'] for entry in report['hash_checks'] if entry['matched']}
    try:
        import numpy as np
    except ImportError:
        checks['status'] = 'skipped: NumPy not installed; recorded-cost and nAUBC not recomputed'
        return
    for relative in paths:
        if relative not in manifested_paths:
            checks[relative] = {'status': 'skipped: optional input absent from manifest'}
            continue
        if relative not in verified:
            checks[relative] = {'status': 'skipped: input did not pass SHA-256 check'}
            continue
        try:
            if relative == paths[0]:
                cost = json.loads((root / relative).read_text(encoding='utf-8'))
                result = {}
                for name, obj in cost['methods'].items():
                    result[name] = {}
                    for batch, values in obj.get('inference', {}).items():
                        raw = values.get('raw_seconds')
                        if raw:
                            result[name][batch] = {'n': len(raw), 'median_ms': float(np.median(raw)*1000),
                                                   'p95_ms': float(np.quantile(raw, .95)*1000)}
                checks['randumb_raw_cost'] = result
                checks['randumb_mma_time_ratio'] = {
                    batch: result['R1'][batch]['median_ms'] / result['H1'][batch]['median_ms']
                    for batch in ('1', '32')}
            else:
                with (root / relative).open(encoding='utf-8-sig', newline='') as stream:
                    rows = list(csv.DictReader(stream))
                result = {}
                for method in ('ideal_only', 'ideal_plus_multi4'):
                    accuracy = []
                    for budget in (32, 48, 64, 96, 128):
                        values = [row['y_true'] == row['y_pred'] for row in rows
                                  if row['method'] == method and int(row['budget']) == budget]
                        if not values:
                            raise ValueError('missing method/budget group')
                        accuracy.append(float(np.mean(values)))
                    result[method] = {'accuracy': accuracy,
                                      'naubc': float(np.dot(accuracy, np.array([8, 16, 24, 32, 16])/96))}
                result['delta_naubc'] = result['ideal_plus_multi4']['naubc'] - result['ideal_only']['naubc']
                checks['budget'] = result
        except Exception as error:
            message = str(error) if isinstance(error, ValueError) else type(error).__name__
            report['failed'].append({'path': relative, 'reason': 'numerical check failed: ' + message})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    bundle_root = args.bundle_root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.is_relative_to(bundle_root):
        parser.error('--output-dir must be outside --bundle-root to preserve frozen inputs')
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        report = verify(bundle_root)
    except Exception as error:
        report = {'summary': {'passed': False}, 'failed': [
            {'path': 'manifest.json', 'reason': str(error) if isinstance(error, ValueError) else type(error).__name__}]}
    (output_dir / 'verification_result.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(report['summary'], ensure_ascii=False))
    return 0 if report['summary']['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
