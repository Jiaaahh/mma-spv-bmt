"""G09-A CPU 测试总入口：逐项执行、逐项记录、写出可回传的证据文件。

用法（必须用项目 .venv 解释器）：
  /historical/root/autodl-tmp/mma_pkg/project/.venv/bin/python g09_v1/tests/run_all_tests.py [--only name]
注意：这里所有通过项都是 **CPU 桩件/小数组** 证据，不能当作 GPU 资格或科学结论。
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
import traceback

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

HERE = Path(__file__).resolve().parent
G09_DIR = HERE.parent
ROOT = G09_DIR.parent
for path in (str(G09_DIR), str(HERE), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

MODULES = ['test_unit', 'test_selector_gate_compare']


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', default=None, help='Run only tests whose name contains this substring')
    parser.add_argument('--out-dir', default=str(G09_DIR / 'evidence'))
    args = parser.parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    results, started_all = [], time.time()
    for module_name in MODULES:
        module = importlib.import_module(module_name)
        for name, function in getattr(module, 'TESTS', []):
            if args.only and args.only not in name:
                continue
            started = time.time()
            entry = {'module': module_name, 'name': name}
            try:
                function()
                entry.update(status='PASS', seconds=round(time.time() - started, 3), error=None)
            except Exception as error:  # noqa: BLE001
                entry.update(status='FAIL', seconds=round(time.time() - started, 3), error=str(error),
                             traceback=traceback.format_exc())
            results.append(entry)
            print(f"[{entry['status']}] {module_name}.{name} ({entry['seconds']}s)"
                  + ('' if entry['status'] == 'PASS' else f" -> {entry['error']}"), flush=True)
    payload = {'stage': 'G09-A CPU tests', 'utc': datetime.now(timezone.utc).isoformat(),
               'python': sys.version, 'argv': sys.argv,
               'cwd': os.getcwd(), 'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS'),
               'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
               'notice': 'CPU stub/small-array evidence only; NOT GPU qualification and NOT scientific results',
               'totals': {'passed': sum(1 for r in results if r['status'] == 'PASS'),
                          'failed': sum(1 for r in results if r['status'] == 'FAIL'),
                          'seconds': round(time.time() - started_all, 3)},
               'tests': results}
    payload['verdict'] = 'G09_A_CPU_TESTS_PASS' if payload['totals']['failed'] == 0 else 'G09_A_CPU_TESTS_FAIL'
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f'cpu_tests_{stamp}.json'
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    txt_path = out_dir / f'cpu_tests_{stamp}.txt'
    lines = [f"# G09-A CPU 测试（{payload['utc']}）", f"# {payload['notice']}", '']
    for entry in results:
        lines.append(f"[{entry['status']}] {entry['module']}.{entry['name']} ({entry['seconds']}s)"
                     + ('' if entry['status'] == 'PASS' else f" -> {entry['error']}"))
        if entry['status'] == 'FAIL':
            lines.append(entry['traceback'].rstrip())
    lines += ['', f"{payload['verdict']}：{payload['totals']['passed']} passed / "
                  f"{payload['totals']['failed']} failed in {payload['totals']['seconds']}s"]
    txt_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f"\n{payload['verdict']}：{payload['totals']['passed']} passed / {payload['totals']['failed']} failed")
    print(f'evidence: {json_path}')
    print(f'evidence: {txt_path}')
    return 0 if payload['totals']['failed'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
