"""G09-A 静态预检（只读，不训练）：矩阵/候选/合同/隔离/旧资产一致性的机器可查结论。

检查项：
  A 交接三文件散列
  B 24 运行组合唯一、顺序、命名、目标步数；共同候选对称
  C 候选表：3 档 lr × 2 档 wd、head=20×backbone、域头绑定、C03 = v1 锚点
  D 方案字段与选择规则一致；无 GPU 训练授权；历史 run 不可复用
  E 旧 v1 源码/协议/manifest 与**上一交付包**内嵌清单逐字节一致
  F data/weights 与 PACKAGE_MANIFEST.json 一致
  G 隔离：G09 代码不会写到 v1 路径（静态源码扫描）
  H 旧资产守卫（baseline/compare）结论
输出 JSON + PASS/FAIL，退出码非零即失败。
"""
from __future__ import annotations

import argparse
import io
from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
import re
import sys
import zipfile

from g09_config import (FROZEN_PACKAGE_MANIFEST_SHA256, EXPECTED_CANDIDATES, EXPECTED_FOLDS, EXPECTED_FORMAL_UPDATES, EXPECTED_METHODS,
                        EXPECTED_RUN_COUNT, G09_DIR, HANDOVER_SHA256, HEAD_TO_BACKBONE_RATIO, PROTOCOL_ID,
                        ROOT, RUNS_DIR_NAME, SELECTION_RULE_ID, assert_handover_files, candidate_table,
                        matrix_index, run_path, sha, write_json)
from g09_core import assert_c03_is_v1_anchor, assert_head_ratio

PREVIOUS_PACKAGE = ROOT / 'final_packages' / 'G08_final_20260919T094208Z.zip'
LEGACY_SOURCES = ['core.py', 'runner.py', 'evaluate.py', 'compare.py', 'audit_runs.py', 'collect_results.py',
                  'make_final_package.sh', 'PACKAGE_MANIFEST.json', 'protocols/fold0.json', 'protocols/fold1.json']
VENDOR_FILES = ['vendor/domainbed/algorithms.py', 'vendor/domainbed/networks.py', 'vendor/domainbed/mixup_module.py']


def _check(results, name, condition, detail=''):
    results.append({'check': name, 'ok': bool(condition), 'detail': detail})
    return bool(condition)


def previous_package_hashes() -> tuple:
    """从上一交付包读取两份清单：嵌套 return 包（runs/reports 证据）与外层包（根文件/协议）。"""
    hashes, outer_hashes = {}, {}
    if not PREVIOUS_PACKAGE.is_file():
        return hashes
    with zipfile.ZipFile(PREVIOUS_PACKAGE) as outer:
        for name in outer.namelist():
            if name.startswith('returns/') and name.endswith('.zip'):
                with zipfile.ZipFile(io.BytesIO(outer.read(name))) as inner:
                    manifest = json.loads(inner.read('RETURN_MANIFEST.json').decode('utf-8'))
                    for record in manifest['files']:
                        hashes[record['path']] = record['sha256']
        if 'FINAL_MANIFEST.json' in outer.namelist():
            for record in json.loads(outer.read('FINAL_MANIFEST.json').decode('utf-8'))['files']:
                outer_hashes[record['path']] = record['sha256']
    return hashes, outer_hashes


def run() -> dict:
    results = []
    # A 交接文件
    try:
        observed = assert_handover_files()
        _check(results, 'A.handover_files', True, json.dumps(observed, ensure_ascii=False))
    except Exception as error:  # noqa: BLE001
        _check(results, 'A.handover_files', False, str(error))
    # B 矩阵
    try:
        index = matrix_index()
        _check(results, 'B.matrix_unique_and_ordered', len(index) == EXPECTED_RUN_COUNT,
               f'{len(index)} unique declared runs')
        legacy_paths = [p for p in index if p.startswith('runs/')]
        _check(results, 'B.no_legacy_run_paths', not legacy_paths, f'{legacy_paths}')
        table = candidate_table()
        symmetric = True
        for fold in EXPECTED_FOLDS:
            for cid in EXPECTED_CANDIDATES:
                for method in EXPECTED_METHODS:
                    entry = index[run_path(fold, method, cid)]
                    if (entry['fold'], entry['method'], entry['candidate_id']) != (fold, method, cid):
                        symmetric = False
        _check(results, 'B.candidate_symmetry', symmetric, 'ERM/XDM share the same six candidates per fold')
        _check(results, 'B.formal_updates_total',
               EXPECTED_RUN_COUNT * 5000 == EXPECTED_FORMAL_UPDATES == 120000,
               f'{EXPECTED_RUN_COUNT} runs x 5000 = {EXPECTED_RUN_COUNT * 5000}')
    except Exception as error:  # noqa: BLE001
        _check(results, 'B.matrix_unique_and_ordered', False, str(error))
        table = {}
    # C 候选
    try:
        grid = {}
        for cid, spec in table.items():
            assert_head_ratio(spec)
            grid.setdefault((Decimal(str(spec['backbone_lr'])), Decimal(str(spec['weight_decay']))), []).append(cid)
        _check(results, 'C.grid_3x2', len(grid) == 6 and all(len(v) == 1 for v in grid.values()),
               f'{sorted((str(k[0]), str(k[1]), v) for k, v in grid.items())}')
        levels = sorted({Decimal(str(s['backbone_lr'])) for s in table.values()})
        _check(results, 'C.three_learning_rate_levels', len(levels) == 3, f'{levels}')
        _check(results, 'C.two_weight_decay_levels',
               sorted({Decimal(str(s['weight_decay'])) for s in table.values()}) == [Decimal('0'), Decimal('0.0001')], '')
        _check(results, 'C.head_is_20x_backbone', all(
            Decimal(str(s['head_lr'])) == HEAD_TO_BACKBONE_RATIO * Decimal(str(s['backbone_lr'])) for s in table.values()), '')
        _check(results, 'C.domain_head_bound', all(
            Decimal(str(s['xdm_domain_lr'])) == Decimal(str(s['backbone_lr'])) and
            Decimal(str(s['xdm_domain_weight_decay'])) == Decimal(str(s['weight_decay'])) for s in table.values()), '')
        assert_c03_is_v1_anchor(table['C03'])
        _check(results, 'C.c03_is_v1_anchor', True, 'backbone 5e-6 / head 1e-4 / wd 0 / domain 5e-6 / domain wd 0')
    except Exception as error:  # noqa: BLE001
        _check(results, 'C.c03_is_v1_anchor', False, str(error))
    # D 方案
    try:
        design = json.loads((G09_DIR / 'protocol' / 'G09_protocol_design.json').read_text(encoding='utf-8'))
        _check(results, 'D.protocol_id_and_selection_rule',
               design['protocol_id'] == PROTOCOL_ID and design['selection']['required_candidates'] == 6 and
               design['selection']['outer_access'].startswith('after all four selection manifests frozen'),
               SELECTION_RULE_ID)
        _check(results, 'D.no_gpu_authorization', design['gpu_training_authorized_by_this_package'] is False,
               design['current_execution_scope'])
        _check(results, 'D.no_historical_reuse', design['historical_run_reuse'] is False and
               design['candidates'][2]['candidate_id'] == 'C03', '')
        _check(results, 'D.selection_step_is_5000', design['inner_selection_step'] == 5000 and
               design['inner_diagnostic_steps'] == [1000, 2000, 3000], '')
    except Exception as error:  # noqa: BLE001
        _check(results, 'D.protocol_id_and_selection_rule', False, str(error))
    # E 旧源码/资产与上一交付包和 PACKAGE_MANIFEST 交叉核对
    try:
        manifest = json.loads((ROOT / 'PACKAGE_MANIFEST.json').read_text(encoding='utf-8'))
        listed = {item['path']: item for item in manifest['files']}
        if sha(ROOT / 'PACKAGE_MANIFEST.json') != FROZEN_PACKAGE_MANIFEST_SHA256:
            raise RuntimeError('PACKAGE_MANIFEST.json itself differs from the frozen handover hash')
        return_hashes, outer_hashes = previous_package_hashes()
        mismatches = []
        for relative in LEGACY_SOURCES + VENDOR_FILES:
            want = (return_hashes.get(relative) or outer_hashes.get(relative)
                    or (listed.get(relative) or {}).get('sha256'))
            if want is None:
                mismatches.append(f'{relative}: no reference hash found')
            elif sha(ROOT / relative) != want:
                mismatches.append(f'{relative}: hash differs from the delivered reference')
        _check(results, 'E.legacy_files_match_previous_package', not mismatches,
               '; '.join(mismatches) or f'{len(LEGACY_SOURCES + VENDOR_FILES)} legacy files identical to delivered references')
        bad = []
        for relative, item in listed.items():
            path = ROOT / relative
            if not path.is_file():
                bad.append(f'{relative}: missing')
            elif sha(path) != item['sha256'] or path.stat().st_size != item['size']:
                bad.append(f'{relative}: hash/size mismatch')
        _check(results, 'E2.full_package_manifest', not bad,
               f"{len(listed)} manifest entries verified; mismatches={bad[:5]}")
    except Exception as error:  # noqa: BLE001
        _check(results, 'E.legacy_files_match_previous_package', False, str(error))
        _check(results, 'E2.full_package_manifest', False, str(error))
    # F data/weights vs PACKAGE_MANIFEST
    try:
        manifest = json.loads((ROOT / 'PACKAGE_MANIFEST.json').read_text(encoding='utf-8'))
        listed = {item['path']: item['sha256'] for item in manifest['files']}
        needed = [f'data/{name}' for name in
                  ['metadata.npz'] + [f'{m}_{kind}.npy' for m in ['ideal'] + [f'moderate_seed{s}' for s in range(2026, 2030)]
                                      for kind in ('images', 'hog')]]
        bad = [relative for relative in needed if listed.get(relative) != sha(ROOT / relative)]
        _check(results, 'F.data_matches_manifest', not bad, f'{len(needed)} data files checked; mismatches={bad}')
    except Exception as error:  # noqa: BLE001
        _check(results, 'F.data_matches_manifest', False, str(error))
    # G 隔离静态扫描：G09 入口不得写 v1 路径；测试中的 v1 路径只允许出现在只读语境
    try:
        legacy_prefix = 'runs' + '/'            # 运行期拼装，避免扫描器匹配到自身字面量
        legacy_names = ['f1_', 'f0_']
        patterns = [re.compile(r"ROOT\s*/\s*'runs'\s*/"), re.compile(r'ROOT\s*/\s*"runs"\s*/')] + \
                   [re.compile(re.escape(legacy_prefix + name)) for name in legacy_names]
        forbidden = []
        for path in sorted(G09_DIR.glob('*.py')):          # 顶层入口模块：任何 v1 run 路径字面量都不允许
            text = path.read_text(encoding='utf-8')
            for pattern in patterns:
                for match in pattern.finditer(text):
                    forbidden.append(f'{path.relative_to(ROOT).as_posix()}:{text[:match.start()].count(chr(10)) + 1}')
        _check(results, 'G.no_v1_run_writes_in_g09_code', not forbidden, f'{forbidden}')
        # 测试里的 v1 路径：只允许作为拒绝用例/只读夹具；出现写动作即失败（启发式）
        write_verbs = (r'write_text\(', r'write_bytes\(', r'write_json\(', r'shutil\.copy', r'shutil\.copytree',
                       r'torch\.save\(', r'np\.save', r'open\([^)]*["\']w')
        legacy_lines = []
        for path in sorted((G09_DIR / 'tests').glob('*.py')):
            for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
                if any(pattern.search(line) for pattern in patterns):
                    legacy_lines.append((f'{path.relative_to(ROOT).as_posix()}:{number}', line.strip()))
        offenders = [f'{where}: {line}' for where, line in legacy_lines
                     if any(re.search(verb, line) for verb in write_verbs)]
        _check(results, 'G2.test_legacy_paths_readonly_only', not offenders,
               f'{len(legacy_lines)} legacy-path lines in tests; write-looking offenders={offenders}')
        _check(results, 'G3.runs_dir_name_isolated', RUNS_DIR_NAME == 'runs_g09_v1', RUNS_DIR_NAME)
    except Exception as error:  # noqa: BLE001
        _check(results, 'G.no_v1_run_writes_in_g09_code', False, str(error))
        _check(results, 'G2.test_legacy_paths_readonly_only', False, str(error))
    # H 旧资产守卫
    try:
        before = G09_DIR / 'evidence' / 'old_assets_before.json'
        after = G09_DIR / 'evidence' / 'old_assets_after.json'
        if before.is_file() and after.is_file():
            payload = json.loads(after.read_text(encoding='utf-8'))
            _check(results, 'H.old_assets_unchanged', bool(payload.get('unchanged')),
                   f"added={len(payload.get('added', []))} removed={len(payload.get('removed', []))} changed={len(payload.get('changed', []))}")
        else:
            _check(results, 'H.old_assets_unchanged', False,
                   f'guard snapshots missing: {before.name}/{after.name}')
    except Exception as error:  # noqa: BLE001
        _check(results, 'H.old_assets_unchanged', False, str(error))
    report = {'stage': 'G09-A static precheck', 'utc': datetime.now(timezone.utc).isoformat(),
              'checks': results, 'passed': sum(1 for r in results if r['ok']), 'total': len(results),
              'verdict': 'G09_A_STATIC_PRECHECK_PASS' if all(r['ok'] for r in results) else 'G09_A_STATIC_PRECHECK_FAIL',
              'not_covered': ['GPU training numerics', 'real 5000-step runs', 'outer evaluation', 'selection over real candidates']}
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='g09_v1/evidence/static_precheck.json')
    args = parser.parse_args(argv)
    report = run()
    out = Path(args.out)
    write_json(out if out.is_absolute() else ROOT / out, report)
    for item in report['checks']:
        print(f"[{'OK' if item['ok'] else 'FAIL'}] {item['check']} {item['detail']}")
    print(f"{report['verdict']}（{report['passed']}/{report['total']}）")
    return 0 if report['verdict'].endswith('PASS') else 1


if __name__ == '__main__':
    sys.exit(main())
