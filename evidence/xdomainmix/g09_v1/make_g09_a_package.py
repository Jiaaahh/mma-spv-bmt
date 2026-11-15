"""打包 G09-A 交付：新源码 + 协议副本 + 证据 + 节点报告 → ZIP + SHA256。

只读取并打包；不修改任何 G09/v1 产物。排除：`.pt` 实体、`data/`、`weights/`、`vendor/`、
`reference/`、凭据、缓存、工程沙箱夹具（`engineering_g09_v1/`）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
import zipfile

from g09_config import G09_DIR, ROOT, sha, write_json

DEFAULT_OUT_DIR = ROOT / 'final_packages'
EXCLUDE_PARTS = {'__pycache__'}
EXCLUDE_SUFFIX = {'.pt', '.pth', '.pyc'}
MAX_SIZE = 32 * 1024 * 1024


def collect() -> list:
    members = []
    for path in sorted(G09_DIR.rglob('*')):
        if path.is_dir() or path.is_symlink():
            continue
        if EXCLUDE_PARTS & set(path.parts) or path.suffix in EXCLUDE_SUFFIX:
            continue
        if path.stat().st_size > MAX_SIZE:
            continue
        if path.name.startswith('cpu_tests_') and path.suffix == '.txt':
            members.append(path)
        elif path.name.startswith('cpu_tests_') and path.suffix == '.json':
            members.append(path)
        else:
            members.append(path)
    for extra in (ROOT / 'reports' / 'G09_A_NODE_REPORT.md', ROOT / 'reports' / 'decision_log.json'):
        if extra.is_file():
            members.append(extra)
    return members


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def main(argv=None) -> int:
    from g09_validation import verify_source_manifest
    verify_source_manifest()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', default=str(DEFAULT_OUT_DIR))
    parser.add_argument('--stamp', default=None)
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = args.stamp or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    target = out_dir / f'G09_A_final_{stamp}.zip'
    if target.exists():
        raise RuntimeError(f'refusing to overwrite {target}')
    members = collect()
    records = [{'path': relative(path), 'bytes': path.stat().st_size, 'sha256': sha(path)} for path in members]
    payload = {'node': 'G09_A', 'utc': datetime.now(timezone.utc).isoformat(), 'project_root': str(ROOT),
               'scope': 'G09_A_ISOLATED_IMPLEMENTATION_NO_TRAINING',
               'notice': ('G09-A contains new source, protocol copies, CPU tests and CLI evidence only. '
                          'No GPU training, no formal runs, no outer evaluation and no selection were executed.'),
               'excludes': ['*.pt', 'data/', 'weights/', 'vendor/', 'reference/', 'engineering_g09_v1/', '__pycache__', 'credentials'],
               'member_count': len(records), 'files': records}
    contents = ['# G09-A 交付包', '', f"- 生成 UTC: {payload['utc']}", f'- 成员数: {len(records)}',
                f"- 说明: {payload['notice']}", '', '## 包内文件', '']
    contents += [f"- `{item['path']}` ({item['bytes']} B, sha256={item['sha256']})" for item in records]
    with zipfile.ZipFile(target, 'x', zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for path in members:
            archive.write(path, relative(path))
        archive.writestr('G09_A_PACKAGE_CONTENTS.md', '\n'.join(contents) + '\n')
        archive.writestr('G09_A_PACKAGE_MANIFEST.json', json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    # 独立复核：重新打开 zip 并逐成员核对大小与散列
    with zipfile.ZipFile(target) as archive:
        bad = []
        for item in records:
            if item['path'] not in archive.namelist():
                bad.append(f"{item['path']}: missing")
                continue
            data = archive.read(item['path'])
            if hashlib.sha256(data).hexdigest() != item['sha256'] or len(data) != item['bytes']:
                bad.append(f"{item['path']}: mismatch")
    digest = sha(target)
    sha_file = target.with_suffix('.sha256.txt')
    sha_file.write_text(f'{digest}  {target.name}\n', encoding='utf-8')
    summary = {'zip': str(target), 'zip_sha256': digest, 'zip_bytes': target.stat().st_size,
               'sha256_file': str(sha_file), 'member_count': len(records),
               'verified_after_write': not bad, 'problems': bad}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not bad else 1


if __name__ == '__main__':
    sys.exit(main())
