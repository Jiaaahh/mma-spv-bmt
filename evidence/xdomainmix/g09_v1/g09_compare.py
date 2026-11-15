"""G09-A2 selected-model comparison: validate each complete frozen contract,
then normalize only prescribed candidate lr/weight-decay fields.
All four inner selections are revalidated before reading outer artifacts.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np

from g09_config import PROTOCOL_ID, ROOT, SELECTION_RULE_ID, sha, write_json
from g09_gate import selected_run

import compare as v1_compare  # noqa: E402  只读复用 v1 校验器

def find_outer_dir(run: Path) -> Path:
    root = run / 'evaluations'
    if not root.is_dir():
        raise RuntimeError(f'{run}: no evaluations directory')
    outers = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith('outer_step5000_'))
    if len(outers) != 1:
        raise RuntimeError(f'{run}: expected exactly one outer_step5000_* directory, found {[p.name for p in outers]}')
    return outers[0]


def declared_differences(erm_contract: dict, xdm_contract: dict) -> dict:
    import copy
    from g09_validation import validate_contract
    for method, contract in (('erm', erm_contract), ('xdm', xdm_contract)):
        validate_contract(contract, method, contract.get('fold'), contract.get('candidate_id'))
    def normalized(contract):
        item = copy.deepcopy(contract)
        for key in ('method', 'candidate_id', 'candidate'):
            item.pop(key)
        opt = item['optimizer']
        for key in ('backbone_lr', 'head_lr', 'weight_decay', 'domain_lr_candidate', 'domain_weight_decay_candidate'):
            opt.pop(key)
        for group in opt['param_group_plan']:
            group.pop('lr'); group.pop('weight_decay')
        item['xdm_hparams'].pop('lr'); item['xdm_hparams'].pop('weight_decay')
        return item
    if normalized(erm_contract) != normalized(xdm_contract):
        raise RuntimeError('ERM/XDM differ outside prescribed candidate fields')
    return {'same_candidate': erm_contract['candidate_id'] == xdm_contract['candidate_id'],
            'validated_against_frozen_contracts': True,
            'observed_differing_keys': sorted(k for k in erm_contract if erm_contract[k] != xdm_contract[k])}


def compare_selected(fold: int, out_path, selection_root=None, run_root=None) -> dict:
    from g09_config import resolve_sandbox
    runs_root = resolve_sandbox(run_root, ROOT)
    erm_selection = selected_run('erm', fold, selection_root, run_root)
    xdm_selection = selected_run('xdm', fold, selection_root, run_root)
    erm_run = runs_root / erm_selection['winner']['run_path']
    xdm_run = runs_root / xdm_selection['winner']['run_path']
    erm_dir, xdm_dir = find_outer_dir(erm_run), find_outer_dir(xdm_run)
    erm = v1_compare.load_result(erm_dir, 'erm')
    xdm = v1_compare.load_result(xdm_dir, 'xdm')
    declarations = declared_differences(erm['contract'], xdm['contract'])
    for name, loaded in (('ERM', erm), ('XDM', xdm)):
        if loaded['contract'].get('protocol_id') != PROTOCOL_ID:
            raise RuntimeError(f'{name}: not a G09 contract')
        if loaded['report'].get('protocol_id') != PROTOCOL_ID:
            raise RuntimeError(f'{name}: evaluation report is not G09')
    for key in ('rows', 'labels', 'mapping_names', 'font', 'sample_ids', 'mma', 'mma_cosine_scores'):
        v1_compare.require(np.array_equal(erm['arrays'][key], xdm['arrays'][key]), f'ERM/XDM do not share identical {key}')
    v1_compare.require(np.array_equal(np.load(erm['directory'] / 'matched_mma_prototypes.npy'),
                                      np.load(xdm['directory'] / 'matched_mma_prototypes.npy')), 'MMA prototypes differ')
    arrays, truth = erm['arrays'], erm['arrays']['labels']
    predictions = {'ERM': arrays['cnn'], 'XDM': xdm['arrays']['cnn'], 'MMA': arrays['mma']}
    correct = {name: value == truth[None, :] for name, value in predictions.items()}
    metrics = {name: {'correct': int(value.sum()), 'total': 900, 'accuracy_percent': float(value.mean() * 100)}
               for name, value in correct.items()}
    rng = np.random.default_rng(90917)
    bootstrap = rng.multinomial(180, np.full(180, 1 / 180), size=20000) / 180
    pairs = [('MMA', 'ERM'), ('MMA', 'XDM'), ('XDM', 'ERM')]
    comparisons, per_character = {}, {}
    for left, right in pairs:
        key = f'{left}_minus_{right}'
        delta = (correct[left].astype(float) - correct[right]).mean(axis=0) * 100
        comparisons[key] = {'difference_pp': float(delta.mean()),
                            'conditional_character_95ci_pp': np.quantile(bootstrap @ delta, [.025, .975]).tolist()}
        per_character[key] = delta
    destination = Path(out_path)
    if not destination.is_absolute():
        destination = ROOT / destination
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    report = {'status': 'COMPLETE_MATCHED_SINGLE_FOLD_DEVELOPMENT_COMPARISON',
              'utc': datetime.now(timezone.utc).isoformat(), 'argv': sys.argv,
              'protocol_id': PROTOCOL_ID, 'selection_rule_id': SELECTION_RULE_ID,
              'fold': erm['contract']['fold'], 'seed': erm['contract']['seed'], 'k': erm['contract']['k'],
              'step': 5000, 'font': str(arrays['font']), 'methods': metrics, 'paired_comparisons': comparisons,
              'candidate_ids': {'ERM': erm['contract']['candidate_id'], 'XDM': xdm['contract']['candidate_id']},
              'declared_candidate_differences': declarations,
              'selection_manifests': {'ERM': str(erm_selection['selection_path']),
                                      'XDM': str(xdm_selection['selection_path'])},
              'bootstrap': {'unit': 'character, retaining all five maps together', 'replicates': 20000, 'seed': 90917},
              'verified': {'equal_shared_contract': True, 'equal_rows_labels_maps_font_ids': True,
                           'equal_mma_scores_and_prototypes': True, 'finite_scores_and_prediction_argmax': True,
                           'evaluation_hashes': True,
                           'erm_checkpoint_file_sha256': erm['checkpoint_file_verified'],
                           'xdm_checkpoint_file_sha256': xdm['checkpoint_file_verified'],
                           'g09_all_selections_frozen': True, 'g09_selected_models_only': True,
                           'g09_declared_differences_only': True},
              'input_artifact_hashes': {'ERM': erm['hashes'], 'XDM': xdm['hashes']},
              'evaluated_checkpoint_sha256': {'ERM': erm['report']['checkpoint_sha256'],
                                              'XDM': xdm['report']['checkpoint_sha256']},
              'limitation': ('G09 development comparison after a finite per-fold hyper-parameter search. The winner was chosen on the '
                             '5000-step inner font only; outer scores were read once, after all four selection manifests were frozen. '
                             'CIs condition on this font, these mappings and the two selected models; they do not account for search '
                             'selection, seed or unseen-font uncertainty.')}
    write_json(destination / 'comparison.json', report)
    with (destination / 'method_summary.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['method', 'correct', 'total', 'accuracy_percent']); writer.writeheader()
        for name, metric in metrics.items():
            writer.writerow({'method': name, **metric})
    with (destination / 'per_mapping.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['mapping', 'method', 'correct', 'total', 'accuracy_percent']); writer.writeheader()
        for index, name in enumerate(arrays['mapping_names']):
            for method in correct:
                writer.writerow({'mapping': str(name), 'method': method, 'correct': int(correct[method][index].sum()),
                                 'total': 180, 'accuracy_percent': float(correct[method][index].mean() * 100)})
    with (destination / 'per_character.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        names = ['row', 'sample_id', 'class_id'] + [f'{m}_correct_of_5' for m in correct] + [f'{pair}_pp' for pair in per_character]
        writer = csv.DictWriter(stream, fieldnames=names); writer.writeheader()
        for i in range(180):
            item = {'row': int(arrays['rows'][i]), 'sample_id': str(arrays['sample_ids'][i]), 'class_id': int(truth[i])}
            item.update({f'{m}_correct_of_5': int(correct[m][:, i].sum()) for m in correct})
            item.update({f'{pair}_pp': float(values[i]) for pair, values in per_character.items()})
            writer.writerow(item)
    np.savez_compressed(destination / 'comparison_arrays.npz',
                        **{m: value for m, value in predictions.items()}, labels=truth, rows=arrays['rows'],
                        mapping_names=arrays['mapping_names'], sample_ids=arrays['sample_ids'], font=arrays['font'])
    write_json(destination / 'artifact_hashes.json',
               {p.name: sha(p) for p in destination.iterdir() if p.is_file()})
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fold', type=int, choices=[0, 1], required=True)
    parser.add_argument('--out', required=True, help='New comparison directory under reports_g09_v1/ or reports/')
    args = parser.parse_args(argv)
    report = compare_selected(args.fold, args.out)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
