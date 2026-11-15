"""G09-A CLI 证据采集：真实 help 输出 + 拒绝示例（含退出码与完整 stderr）。

注意：所有示例都使用**非法或未声明的路径/参数**，在任何 mkdir/训练/审计之前就被拒绝，
因此不会创建 `runs_g09_v1/` 正式训练状态，也不会产生外评或比较产物。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
G09_DIR = HERE.parent
PY = str(ROOT / '.venv' / 'bin' / 'python')
ENTRY = 'g09_v1/g09_runner.py'

HELP_CASES = [
    ('help_root', [ENTRY, '--help']),
    ('help_train', [ENTRY, 'train', '--help']),
    ('help_evaluate', [ENTRY, 'evaluate', '--help']),
    ('help_select', [ENTRY, 'select', '--help']),
    ('help_compare', [ENTRY, 'compare', '--help']),
    ('help_gate_status', [ENTRY, 'gate-status', '--help']),
]

REJECTION_CASES = [
    ('reject_legacy_run_path',
     [ENTRY, 'train', '--method', 'erm', '--fold', '0', '--candidate', 'C03', '--device', 'cpu', '--until', '1',
      '--run', 'runs/f1_s17_k8_erm_venv26']),
    ('reject_candidate_mismatch',
     [ENTRY, 'train', '--method', 'erm', '--fold', '0', '--candidate', 'C04', '--device', 'cpu', '--until', '1',
      '--run', 'runs_g09_v1/f0_s17_k8_erm_C03']),
    ('reject_illegal_candidate',
     [ENTRY, 'train', '--method', 'erm', '--fold', '0', '--candidate', 'C07', '--device', 'cpu', '--until', '1',
      '--run', 'runs_g09_v1/f0_s17_k8_erm_C07']),
    ('reject_until_beyond_target',
     [ENTRY, 'train', '--method', 'erm', '--fold', '0', '--candidate', 'C03', '--device', 'cpu', '--until', '6000',
      '--run', 'runs_g09_v1/f0_s17_k8_erm_C03']),
    ('reject_outer_on_legacy_run',
     [ENTRY, 'evaluate', '--run', 'runs/f1_s17_k8_erm_venv26', '--device', 'cpu', '--split', 'outer']),
    ('reject_evaluate_unstarted_run',
     [ENTRY, 'evaluate', '--run', 'runs_g09_v1/f0_s17_k8_erm_C03', '--device', 'cpu', '--split', 'inner']),
    ('reject_select_without_runs', [ENTRY, 'select', '--method', 'erm', '--fold', '0']),
    ('reject_compare_gate_closed', [ENTRY, 'compare', '--fold', '0', '--out', 'reports_g09_v1/g09_comparison_gate_closed']),
]

OBSERVATION_CASES = [('gate_status_closed', [ENTRY, 'gate-status'])]


def run_case(name: str, argv: list, out_dir: Path) -> dict:
    completed = subprocess.run([PY] + argv, cwd=str(ROOT), capture_output=True, text=True)
    body = [f'# command: {" ".join(argv)}', f'# cwd: {ROOT}', f'# exit_code: {completed.returncode}', '']
    body.append('## stdout')
    body.append(completed.stdout.rstrip('\n'))
    body.append('')
    body.append('## stderr')
    body.append(completed.stderr.rstrip('\n'))
    text = '\n'.join(body) + '\n'
    path = out_dir / f'cli_{name}.txt'
    path.write_text(text, encoding='utf-8')
    return {'name': name, 'argv': argv, 'exit_code': completed.returncode, 'evidence': path.name,
            'stderr_head': completed.stderr.strip().splitlines()[:2]}


def main(argv=None) -> int:
    out_dir = G09_DIR / 'evidence'
    out_dir.mkdir(parents=True, exist_ok=True)
    records = {'help': [], 'rejections': [], 'observations': []}
    for name, case in HELP_CASES:
        record = run_case(name, case, out_dir)
        assert record['exit_code'] == 0, (name, record)
        records['help'].append(record)
    for name, case in REJECTION_CASES:
        record = run_case(name, case, out_dir)
        assert record['exit_code'] != 0, (name, record)
        records['rejections'].append(record)
    for name, case in OBSERVATION_CASES:
        records['observations'].append(run_case(name, case, out_dir))
    summary = {'stage': 'G09-A CLI evidence', 'entry': ENTRY, 'interpreter': PY,
               'notice': ('Help output is real. Every rejection case uses an illegal or undeclared path/argument and is '
                          'refused before any mkdir/training/audit, so no runs_g09_v1 formal state is created.'),
               'runs_dir_created': (ROOT / 'runs_g09_v1').exists(),
               'selections_dir_created': (ROOT / 'selections_g09_v1').exists(),
               **records}
    (out_dir / 'cli_evidence_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n',
                                                       encoding='utf-8')
    print(json.dumps({'help_cases': len(records['help']), 'rejection_cases': len(records['rejections']),
                      'observations': len(records['observations']),
                      'runs_g09_v1_exists': summary['runs_dir_created'],
                      'selections_g09_v1_exists': summary['selections_dir_created']}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
