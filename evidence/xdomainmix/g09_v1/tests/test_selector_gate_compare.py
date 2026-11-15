"""G09-A CPU 测试（第二批）：内验选择器、outer 门禁、G09 比较器。

夹具全部在 `engineering_g09_v1/` 沙箱内；外评数组取自真实 G08 产物但元数据为合成，
因此这些测试只证明**代码路径与拒绝规则**，不产生任何 G09 科学结果。
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

import numpy as np

from g09_config import EXPECTED_CANDIDATES, ROOT, TRAIN_MAPS, run_path, sha, write_json
import g09_compare
import g09_gate
import g09_select

import fixtures  # noqa: E402

REAL_OUTER = {
    'erm': ROOT / 'runs/f1_s17_k8_erm_venv26/evaluations/outer_step5000_20260919T093739_451506Z',
    'xdm': ROOT / 'runs/f1_s17_k8_xdm_venv26/evaluations/outer_step5000_20260919T093747_103516Z',
}

DEFAULT_CORRECT = {'C01': 500, 'C02': 400, 'C03': 400, 'C04': 400, 'C05': 300, 'C06': 200}
DEFAULT_MARGIN = {'C01': 6.9, 'C02': 6.8, 'C03': 6.8, 'C04': 7.2, 'C05': 6.0, 'C06': 6.0}


def expect_error(fn, contains: str, label: str):
    try:
        fn()
    except Exception as error:  # noqa: BLE001
        if contains not in str(error):
            raise AssertionError(f'{label}: message {str(error)!r} lacks {contains!r}') from error
        return str(error)
    raise AssertionError(f'{label}: expected failure containing {contains!r}')


def build_all(sandbox, fold, method, correct=None, margins=None):
    correct = correct or DEFAULT_CORRECT
    margins = margins or DEFAULT_MARGIN
    return {cid: fixtures.make_run(sandbox, fold, method, cid, correct_target=correct[cid], margin=margins[cid])
            for cid in EXPECTED_CANDIDATES}


# ------------------------------------------------------------------ 选择器
def test_selector_ranking_and_tiebreaks():
    sandbox = fixtures.new_sandbox('selector_ranking')
    build_all(sandbox, 0, 'erm')
    manifest = g09_select.select('erm', 0, write=True, run_root=sandbox['runs'], out_dir=sandbox['selections'])
    order = [entry['candidate_id'] for entry in manifest['ranking']]
    assert order == ['C01', 'C02', 'C03', 'C04', 'C05', 'C06'], order
    assert manifest['winner']['candidate_id'] == 'C01'
    reasons = {entry['candidate_id']: entry['decided_over_previous_by'] for entry in manifest['ranking']}
    assert reasons['C03'] == ['candidate_id'], reasons
    assert reasons['C04'] == ['cross_entropy_rhe8'], reasons
    assert reasons['C02'] == ['correct_count'], reasons
    corrects = {entry['candidate_id']: entry['correct'] for entry in manifest['ranking']}
    assert corrects == DEFAULT_CORRECT, corrects
    ce = {entry['candidate_id']: entry['cross_entropy_rhe8'] for entry in manifest['ranking']}
    assert ce['C02'] == ce['C03'], 'identical logits must give identical rounded CE'
    assert ce['C02'] != ce['C04']
    on_disk = json.loads((sandbox['selections'] / 'erm_f0.json').read_text(encoding='utf-8'))
    assert on_disk['frozen'] is True and on_disk['outer_scores_read'] is False
    assert len(on_disk['input_hashes']) == 6
    for entry in on_disk['input_hashes'].values():
        assert set(entry['files']) == {'results.json', 'predictions.npz', 'predictions.csv', 'confusion_matrix.npy'}


def test_selector_rounding_helper_half_even():
    # 约定：对**精确二进制值**做十进制 ROUND_HALF_EVEN（同时保存 repr 版本供透明核对）。
    # 注意 0.123456795 的精确值是 ...7949999...，因此 half-even 进位到 0.12345679；
    # 而 repr 版本会得到 0.12345680 —— 这正是两种约定必须显式记录的原因。
    assert g09_select.round_rhe8(0.123456785) == '0.12345678'
    assert g09_select.round_rhe8(0.123456795) == '0.12345679'
    assert g09_select.round_rhe8(0.123456775) == '0.12345678'
    assert g09_select.round_rhe8(-0.000000004) == '-0E-8'
    assert g09_select.round_repr8(0.123456795) == '0.12345680'
    assert g09_select.round_repr8(0.123456785) == '0.12345678'
    value = g09_select.stable_cross_entropy_float64(np.zeros((5, 180, 180), dtype=np.float32), np.arange(180))
    assert abs(value - float(np.log(180))) < 1e-12, value


def test_selector_rejects_invalid_inputs():
    sandbox = fixtures.new_sandbox('selector_rejects')
    build_all(sandbox, 0, 'erm')
    outer_only = fixtures.new_sandbox('selector_outer_only')
    run = fixtures.make_run(outer_only, 0, 'erm', 'C01')
    shutil.copytree(REAL_OUTER['erm'], run / 'evaluations' / 'outer_step5000_fixture')
    shutil.rmtree(next((run / 'evaluations').glob('inner_step5000_*')))
    expect_error(lambda: g09_select.read_inner(run_path(0, 'erm', 'C01'), 'erm', 0, 'C01', run_root=outer_only['runs']),
                 'outer evaluation directories are not valid selection inputs', 'outer input refused')
    early = fixtures.new_sandbox('selector_early')
    fixtures.make_run(early, 0, 'erm', 'C01', steps=(1000,))
    expect_error(lambda: g09_select.read_inner(run_path(0, 'erm', 'C01'), 'erm', 0, 'C01', run_root=early['runs']),
                 'only early inner diagnostics', 'early inner refused')
    duplicate = fixtures.new_sandbox('selector_duplicate')
    run = fixtures.make_run(duplicate, 0, 'erm', 'C01')
    shutil.copytree(next((run / 'evaluations').glob('inner_step5000_*')), run / 'evaluations' / 'inner_step5000_dup')
    expect_error(lambda: g09_select.read_inner(run_path(0, 'erm', 'C01'), 'erm', 0, 'C01', run_root=duplicate['runs']),
                 'duplicate inner_step5000', 'duplicate inner refused')
    duplicate_labels = fixtures.new_sandbox('selector_labels')
    bad = fixtures.fold_rows(0)[1].copy()
    bad[1] = bad[0]
    fixtures.make_run(duplicate_labels, 0, 'erm', 'C01', labels=bad)
    expect_error(lambda: g09_select.read_inner(run_path(0, 'erm', 'C01'), 'erm', 0, 'C01',
                                               run_root=duplicate_labels['runs']),
                 'label support', 'duplicate/missing labels refused')
    permuted = fixtures.new_sandbox('selector_labels_permuted')
    shuffled_labels = fixtures.fold_rows(0)[1][np.roll(np.arange(180), 1)]
    fixtures.make_run(permuted, 0, 'erm', 'C01', labels=shuffled_labels)
    expect_error(lambda: g09_select.read_inner(run_path(0, 'erm', 'C01'), 'erm', 0, 'C01',
                                               run_root=permuted['runs']),
                 'labels must equal metadata labels', 'labels not matching metadata refused')
    scrambled = fixtures.new_sandbox('selector_mapping')
    fixtures.make_run(scrambled, 0, 'erm', 'C01', mapping_names=list(reversed(TRAIN_MAPS)))
    expect_error(lambda: g09_select.read_inner(run_path(0, 'erm', 'C01'), 'erm', 0, 'C01', run_root=scrambled['runs']),
                 'inner mapping list mismatch', 'wrong mapping refused')
    nan_box = fixtures.new_sandbox('selector_nan')
    run = fixtures.make_run(nan_box, 0, 'erm', 'C01')
    inner = next((run / 'evaluations').glob('inner_step5000_*'))
    with np.load(inner / 'predictions.npz', allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays['logits'] = arrays['logits'].copy()
    arrays['logits'][0, 0, 0] = np.nan
    np.savez_compressed(inner / 'predictions.npz', **arrays)
    write_json(inner / 'artifact_hashes.json', {p.name: sha(p) for p in inner.iterdir() if p.is_file()})
    expect_error(lambda: g09_select.read_inner(run_path(0, 'erm', 'C01'), 'erm', 0, 'C01', run_root=nan_box['runs']),
                 'nonfinite logits', 'NaN logits refused')
    font_box = fixtures.new_sandbox('selector_font')
    run = fixtures.make_run(font_box, 0, 'erm', 'C01', correct_target=400)
    inner = next((run / 'evaluations').glob('inner_step5000_*'))
    results = json.loads((inner / 'results.json').read_text(encoding='utf-8'))
    results['font'] = 'noto_sans_sc_thin'          # fold0 内验字体是 zcool_kuaile
    write_json(inner / 'results.json', results)
    with np.load(inner / 'predictions.npz', allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays['font'] = np.asarray('noto_sans_sc_thin')
    np.savez_compressed(inner / 'predictions.npz', **arrays)
    write_json(inner / 'artifact_hashes.json', {p.name: sha(p) for p in inner.iterdir() if p.is_file()})
    expect_error(lambda: g09_select.read_inner(run_path(0, 'erm', 'C01'), 'erm', 0, 'C01', run_root=font_box['runs']),
                 'inner font', 'wrong inner font refused')
    expect_error(lambda: g09_select.select('erm', 0, write=False, candidates_order=EXPECTED_CANDIDATES[:5],
                                           run_root=sandbox['runs'], out_dir=sandbox['selections']),
                 'exactly the six frozen candidates', 'missing candidate refused')
    expect_error(lambda: g09_select.select('erm', 0, write=False,
                                           candidates_order=['C01', 'C01', 'C02', 'C03', 'C04', 'C05'],
                                           run_root=sandbox['runs'], out_dir=sandbox['selections']),
                 'exactly the six frozen candidates', 'duplicate candidate refused')
    sparse = fixtures.new_sandbox('selector_sparse')
    fixtures.make_run(sparse, 0, 'erm', 'C01')
    expect_error(lambda: g09_select.select('erm', 0, write=False, run_root=sparse['runs'],
                                           out_dir=sparse['selections']),
                 'missing contract.json', 'missing candidate run refused')


def test_selector_order_insensitivity():
    """两类顺序无关性：

    1. 候选/文件传入顺序：打乱后排序与胜出者不变（`select` 内部按 candidate_id 归一）。
    2. 视图枚举顺序：把 (rows, labels, cnn, logits, sample_ids) **一致地**重排后，正确数与本仓库
       独立复算的 CE 完全相同 —— 即统计量只依赖 (预测, 标签) 配对，不依赖列举顺序。
       注意：正式产物必须按冻结划分顺序存储，`read_inner` 会拒绝顺序被改写的产物（这是更强的冻结要求）。
    """
    sandbox = fixtures.new_sandbox('selector_order')
    build_all(sandbox, 0, 'erm')
    first = g09_select.select('erm', 0, write=False, run_root=sandbox['runs'], out_dir=sandbox['selections'])
    second = g09_select.select('erm', 0, write=False, candidates_order=['C04', 'C01', 'C06', 'C03', 'C05', 'C02'],
                               run_root=sandbox['runs'], out_dir=sandbox['selections'])
    assert [e['candidate_id'] for e in first['ranking']] == [e['candidate_id'] for e in second['ranking']]
    run = sandbox['runs'] / run_path(0, 'erm', 'C02')
    inner = next((run / 'evaluations').glob('inner_step5000_*'))
    with np.load(inner / 'predictions.npz', allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    baseline_correct = int((arrays['cnn'] == arrays['labels'][None, :]).sum())
    baseline_ce = g09_select.stable_cross_entropy_float64(arrays['logits'], arrays['labels'])
    permutation = np.random.default_rng(3).permutation(180)
    permuted = {'logits': arrays['logits'][:, permutation, :], 'cnn': arrays['cnn'][:, permutation],
                'labels': arrays['labels'][permutation], 'rows': arrays['rows'][permutation],
                'sample_ids': arrays['sample_ids'][permutation]}
    assert int((permuted['cnn'] == permuted['labels'][None, :]).sum()) == baseline_correct
    assert g09_select.stable_cross_entropy_float64(permuted['logits'], permuted['labels']) == baseline_ce
    # 顺序被改写的产物必须被拒绝（冻结划分顺序是硬要求）
    np.savez_compressed(inner / 'predictions.npz', **{**arrays, **permuted})
    write_json(inner / 'artifact_hashes.json', {p.name: sha(p) for p in inner.iterdir() if p.is_file()})
    try:
        g09_select.read_inner(run_path(0, 'erm', 'C02'), 'erm', 0, 'C02', run_root=sandbox['runs'])
        raise AssertionError('a re-ordered inner artifact must be rejected')
    except g09_select.SelectionError as error:
        assert 'inner-validation split' in str(error), str(error)


def test_selector_frozen_manifest_is_not_overwritten():
    sandbox = fixtures.new_sandbox('selector_freeze')
    build_all(sandbox, 0, 'erm')
    g09_select.select('erm', 0, write=True, run_root=sandbox['runs'], out_dir=sandbox['selections'])
    expect_error(lambda: g09_select.select('erm', 0, write=True, run_root=sandbox['runs'], out_dir=sandbox['selections']),
                 'already exists', 'frozen manifest not overwritten')


# ------------------------------------------------------------------ 门禁
def _freeze_all(sandbox):
    for method in ('erm', 'xdm'):
        for fold in (0, 1):
            build_all(sandbox, fold, method)
    for method in ('erm', 'xdm'):
        for fold in (0, 1):
            g09_select.select(method, fold, write=True, run_root=sandbox['runs'], out_dir=sandbox['selections'])


def test_gate_blocks_until_all_four_frozen():
    sandbox = fixtures.new_sandbox('gate_blocks')
    build_all(sandbox, 0, 'erm')
    expect_error(lambda: g09_gate.require_all_selections_frozen(sandbox['selections']),
                 'blocked until all four', 'no manifest')
    g09_select.select('erm', 0, write=True, run_root=sandbox['runs'], out_dir=sandbox['selections'])
    expect_error(lambda: g09_gate.require_all_selections_frozen(sandbox['selections']),
                 'blocked until all four', 'one of four')
    status = g09_gate.frozen_status(sandbox['selections'])
    assert status[('erm', 0)]['frozen'] and not status[('xdm', 0)]['frozen']
    expect_error(lambda: g09_gate.require_outer_allowed('erm', 0, run_path(0, 'erm', 'C01'),
                                                        selection_root=sandbox['selections'], run_root=sandbox['runs']),
                 'blocked until all four', 'outer blocked before freeze')


def test_gate_allows_only_selected_winner():
    sandbox = fixtures.new_sandbox('gate_winner')
    _freeze_all(sandbox)
    winner = g09_gate.selected_run('erm', 0, sandbox['selections'], sandbox['runs'])['winner']['candidate_id']
    assert winner == 'C01'
    info = g09_gate.require_outer_allowed('erm', 0, run_path(0, 'erm', winner),
                                          selection_root=sandbox['selections'], run_root=sandbox['runs'])
    assert info['winner']['candidate_id'] == 'C01'
    expect_error(lambda: g09_gate.require_outer_allowed('erm', 0, run_path(0, 'erm', 'C05'),
                                                        selection_root=sandbox['selections'], run_root=sandbox['runs']),
                 'restricted to the selected winner', 'non-winner refused')
    expect_error(lambda: g09_gate.require_outer_allowed('xdm', 1, run_path(0, 'erm', 'C01'),
                                                        selection_root=sandbox['selections'], run_root=sandbox['runs']),
                 'restricted to the selected winner', 'cross-method run refused')


def test_gate_detects_post_selection_tampering():
    sandbox = fixtures.new_sandbox('gate_tamper')
    _freeze_all(sandbox)
    inner = next((sandbox['runs'] / run_path(0, 'erm', 'C02') / 'evaluations').glob('inner_step5000_*'))
    (inner / 'predictions.csv').write_text('tampered\n', encoding='utf-8')
    expect_error(lambda: g09_gate.selected_run('erm', 0, sandbox['selections'], sandbox['runs']),
                 'changed after freezing', 'tampered inner artifact detected')
    sandbox2 = fixtures.new_sandbox('gate_tamper_ckpt')
    _freeze_all(sandbox2)
    (sandbox2['runs'] / run_path(0, 'erm', 'C03') / 'latest.pt').write_bytes(b'other')
    expect_error(lambda: g09_gate.selected_run('erm', 0, sandbox2['selections'], sandbox2['runs']),
                 'evaluated checkpoint changed after selection', 'tampered checkpoint detected')


# ------------------------------------------------------------------ 比较器
def _compare_fixture(name, xdm_best='C01'):
    sandbox = fixtures.new_sandbox(name)
    for fold in (0, 1):
        for method in ('erm', 'xdm'):
            correct = dict(DEFAULT_CORRECT)
            if fold == 0 and method == 'xdm':
                correct = {cid: 400 for cid in EXPECTED_CANDIDATES}
                correct[xdm_best] = 500
            build_all(sandbox, fold, method, correct=correct)
    for method in ('erm', 'xdm'):
        for fold in (0, 1):
            g09_select.select(method, fold, write=True, run_root=sandbox['runs'], out_dir=sandbox['selections'])
    for fold in (0, 1):
        for method in ('erm', 'xdm'):
            winner = g09_gate.selected_run(method, fold, sandbox['selections'], sandbox['runs'])['winner']['candidate_id']
            fixtures.add_outer_dir(sandbox['runs'] / run_path(fold, method, winner), method, fold, winner,
                                   REAL_OUTER[method])
    return sandbox


def test_compare_matches_v1_statistics_on_equal_candidates():
    sandbox = _compare_fixture('compare_equal')
    erm_outer = next((sandbox['runs'] / run_path(1, 'erm', 'C01') / 'evaluations').glob('outer_step5000_*'))
    xdm_outer = next((sandbox['runs'] / run_path(1, 'xdm', 'C01') / 'evaluations').glob('outer_step5000_*'))
    import compare as v1_compare
    v1_out = sandbox['root'] / 'v1_comparison'
    v1_report = v1_compare.compare(erm_outer, xdm_outer, v1_out)
    g09_out = sandbox['root'] / 'g09_comparison'
    g09_report = g09_compare.compare_selected(1, g09_out, selection_root=sandbox['selections'], run_root=sandbox['runs'])
    assert v1_report['methods'] == g09_report['methods'], (v1_report['methods'], g09_report['methods'])
    assert v1_report['paired_comparisons'] == g09_report['paired_comparisons']
    assert g09_report['declared_candidate_differences'] == {'same_candidate': True, 'allowed_keys': ['method'],
                                                           'observed_differing_keys': ['method']}
    for name in ('method_summary.csv', 'per_mapping.csv', 'per_character.csv', 'comparison_arrays.npz'):
        assert (v1_out / name).read_bytes() == (g09_out / name).read_bytes(), f'{name} differs from the v1 comparator'
    assert g09_report['verified']['g09_selected_models_only'] is True
    assert g09_report['verified']['g09_declared_differences_only'] is True


def test_compare_allows_only_declared_candidate_differences():
    sandbox = _compare_fixture('compare_different', xdm_best='C02')
    erm_winner = g09_gate.selected_run('erm', 0, sandbox['selections'], sandbox['runs'])['winner']['candidate_id']
    xdm_winner = g09_gate.selected_run('xdm', 0, sandbox['selections'], sandbox['runs'])['winner']['candidate_id']
    assert (erm_winner, xdm_winner) == ('C01', 'C02'), (erm_winner, xdm_winner)
    report = g09_compare.compare_selected(0, sandbox['root'] / 'g09_comparison_different',
                                          selection_root=sandbox['selections'], run_root=sandbox['runs'])
    declarations = report['declared_candidate_differences']
    assert declarations['same_candidate'] is False
    assert set(declarations['observed_differing_keys']) <= set(declarations['allowed_keys'])
    assert report['candidate_ids'] == {'ERM': 'C01', 'XDM': 'C02'}
    contract_path = sandbox['runs'] / run_path(0, 'xdm', 'C02') / 'contract.json'
    outer = next((sandbox['runs'] / run_path(0, 'xdm', 'C02') / 'evaluations').glob('outer_step5000_*'))

    def rewrite_contract_and_report(mutate):
        contract = json.loads(contract_path.read_text(encoding='utf-8'))
        mutate(contract)
        write_json(contract_path, contract)
        results = json.loads((outer / 'results.json').read_text(encoding='utf-8'))
        results['contract_sha256'] = sha(contract_path)
        write_json(outer / 'results.json', results)
        write_json(outer / 'artifact_hashes.json', {p.name: sha(p) for p in outer.iterdir() if p.is_file()})

    # 受保护键（data_hashes）差异：单目录校验仍通过，但 G09 白名单必须拒绝
    def tamper_protected(contract):
        contract['data_hashes'] = dict(list(contract['data_hashes'].items())[:-1])

    rewrite_contract_and_report(tamper_protected)
    expect_error(lambda: g09_compare.compare_selected(0, sandbox['root'] / 'g09_comparison_protected',
                                                      selection_root=sandbox['selections'], run_root=sandbox['runs']),
                 'Protected G09 contract key differs: data_hashes', 'protected key difference refused')
    # 纵深防御：k 被改写时，v1 单目录校验器先一步发现 report/contract 不一致
    def tamper_k(contract):
        contract['k'] = 4

    rewrite_contract_and_report(tamper_k)
    expect_error(lambda: g09_compare.compare_selected(0, sandbox['root'] / 'g09_comparison_k',
                                                      selection_root=sandbox['selections'], run_root=sandbox['runs']),
                 'report/contract k mismatch', 'report/contract inconsistency refused')


def test_compare_refuses_non_selected_directories():
    sandbox = _compare_fixture('compare_nonselected')
    fixtures.add_outer_dir(sandbox['runs'] / run_path(1, 'erm', 'C05'), 'erm', 1, 'C05', REAL_OUTER['erm'])
    report = g09_compare.compare_selected(1, sandbox['root'] / 'g09_comparison_nonselected',
                                          selection_root=sandbox['selections'], run_root=sandbox['runs'])
    assert report['candidate_ids'] == {'ERM': 'C01', 'XDM': 'C01'}
    assert 'C05' not in json.dumps(report['input_artifact_hashes'])
    expect_error(lambda: g09_gate.require_outer_allowed('erm', 1, run_path(1, 'erm', 'C05'),
                                                        selection_root=sandbox['selections'], run_root=sandbox['runs']),
                 'restricted to the selected winner', 'non-selected dir refused by gate')


TESTS = [
    ('selector_ranking_and_tiebreaks', test_selector_ranking_and_tiebreaks),
    ('selector_rounding_helper_half_even', test_selector_rounding_helper_half_even),
    ('selector_rejects_invalid_inputs', test_selector_rejects_invalid_inputs),
    ('selector_order_insensitivity', test_selector_order_insensitivity),
    ('selector_frozen_manifest_not_overwritten', test_selector_frozen_manifest_is_not_overwritten),
    ('gate_blocks_until_all_four_frozen', test_gate_blocks_until_all_four_frozen),
    ('gate_allows_only_selected_winner', test_gate_allows_only_selected_winner),
    ('gate_detects_post_selection_tampering', test_gate_detects_post_selection_tampering),
    ('compare_matches_v1_statistics_equal_candidates', test_compare_matches_v1_statistics_on_equal_candidates),
    ('compare_allows_only_declared_differences', test_compare_allows_only_declared_candidate_differences),
    ('compare_refuses_non_selected_directories', test_compare_refuses_non_selected_directories),
]
