"""G09-A 旧资产守卫：A 阶段前后对 v1 关键文件做散列快照并比对。

保护范围（A 阶段期间**必须逐字节不变**）：
  data/、weights/、protocols/、protocols_frozen/、vendor/、reference/、根目录 v1 源码与 manifest，
  runs/ 下的合同/metrics/status/检查点/invocation/node/failure 记录与评价产物的元数据，
  reports/ 下的 G07_*、G08_* 文件。

明确排除（本阶段会变动或新增）：reports/decision_log.json（活文档）、G09 新目录
（g09_v1/、runs_g09_v1/、engineering_g09_v1/、selections_g09_v1/）、打包产物（final_packages/、returns/、
deliverables_manifest.json）、backups/。排除项在输出里显式列出，避免“静默漏检”。
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

from g09_config import ROOT, sha, write_json

PROTECTED_ROOT_FILES = ['PACKAGE_MANIFEST.json', 'MANIFEST.json', 'core.py', 'runner.py', 'evaluate.py',
                        'compare.py', 'audit_runs.py', 'collect_results.py', 'make_final_package.sh',
                        'qualification.py', 'README.md', 'AGENTS.md']
PROTECTED_DIRS = ['data', 'weights', 'protocols', 'protocols_frozen', 'vendor', 'reference']
RUN_RECORD_NAMES = ('contract.json', 'metrics.jsonl', 'status.json', 'latest.pt', 'latest.previous.pt')
RUN_RECORD_PREFIXES = ('invocation_', 'node_step', 'failure_', 'uncheckpointed_metrics_')
EVAL_META_NAMES = ('results.json', 'artifact_hashes.json')
EXCLUDED_REPORTS = ('decision_log.json',)


def collect() -> dict:
    files = {}
    for name in PROTECTED_ROOT_FILES:
        path = ROOT / name
        if path.is_file():
            files[path.relative_to(ROOT).as_posix()] = {'bytes': path.stat().st_size, 'sha256': sha(path)}
    for directory in PROTECTED_DIRS:
        root = ROOT / directory
        if not root.is_dir():
            continue
        for path in sorted(root.rglob('*')):
            if path.is_file() and not path.is_symlink():
                files[path.relative_to(ROOT).as_posix()] = {'bytes': path.stat().st_size, 'sha256': sha(path)}
    runs = ROOT / 'runs'
    if runs.is_dir():
        for path in sorted(runs.rglob('*')):
            if not path.is_file() or path.is_symlink():
                continue
            name = path.name
            if name in RUN_RECORD_NAMES or name.startswith(RUN_RECORD_PREFIXES) or \
                    (path.parent.name.startswith(('inner_step', 'outer_step')) and name in EVAL_META_NAMES):
                files[path.relative_to(ROOT).as_posix()] = {'bytes': path.stat().st_size, 'sha256': sha(path)}
    reports = ROOT / 'reports'
    if reports.is_dir():
        for path in sorted(reports.rglob('*')):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(ROOT).as_posix()
            name = path.name
            if name in EXCLUDED_REPORTS or relative.startswith('reports/G09'):
                continue
            if name.startswith(('G07_', 'G08_')):
                files[relative] = {'bytes': path.stat().st_size, 'sha256': sha(path)}
    return files


def baseline(out_path) -> dict:
    files = collect()
    payload = {'stage': 'G09-A protected-asset snapshot (taken after g09_v1/ code was written; g09_v1/ is not part of the protected set)', 'utc': datetime.now(timezone.utc).isoformat(),
               'file_count': len(files), 'excluded': {'reports': list(EXCLUDED_REPORTS),
                                                      'new_g09_dirs': ['g09_v1/', 'runs_g09_v1/', 'engineering_g09_v1/',
                                                                       'selections_g09_v1/'],
                                                      'packaging': ['final_packages/', 'returns/', 'deliverables_manifest.json',
                                                                    'backups/']},
               'files': files}
    write_json(out_path, payload)
    return payload


def compare(before_path, out_path) -> dict:
    before = json.loads(Path(before_path).read_text(encoding='utf-8'))['files']
    after = collect()
    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    changed = sorted(k for k in set(before) & set(after) if before[k]['sha256'] != after[k]['sha256'])
    payload = {'stage': 'G09-A protected-asset re-check', 'utc': datetime.now(timezone.utc).isoformat(),
               'before_file': str(before_path), 'before_file_count': len(before), 'after_file_count': len(after),
               'added': added, 'removed': removed, 'changed': changed,
               'unchanged': not (added or removed or changed),
               'verdict': 'OLD_ASSETS_UNCHANGED' if not (added or removed or changed) else 'OLD_ASSETS_DIFFER',
               'note': ('Protected v1 assets must be byte-identical across G09-A. Any added/removed/changed entry here '
                        'is a violation and must be reported, not silently accepted.'),
               'files': after}
    write_json(out_path, payload)
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    base = sub.add_parser('baseline')
    base.add_argument('--out', required=True)
    cmp_ = sub.add_parser('compare')
    cmp_.add_argument('--before', required=True)
    cmp_.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    if args.command == 'baseline':
        payload = baseline(ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out))
        print(json.dumps({'verdict': 'BASELINE_WRITTEN', 'file_count': payload['file_count']}, ensure_ascii=False))
        return 0
    payload = compare(args.before, ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out))
    print(json.dumps({k: payload[k] for k in ('verdict', 'added', 'removed', 'changed')}, ensure_ascii=False, indent=2))
    return 0 if payload['unchanged'] else 1


if __name__ == '__main__':
    sys.exit(main())
