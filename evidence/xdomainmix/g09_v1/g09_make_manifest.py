"""生成 G09-A 的 `G09_PACKAGE_MANIFEST.json`：新源码 + 方案/候选表/矩阵/交接文件 + 旧资产锚点。

与 v1 `PACKAGE_MANIFEST.json` 的关系：不改写旧清单；本清单**另存**于 g09_v1/ 下，
记录新入口全部源码散列、G09 协议三件散列、以及旧资产清单散列（`original_manifest_sha256`）。
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

from g09_config import (EXPECTED_CANDIDATES, EXPECTED_FOLDS, EXPECTED_METHODS, G09_DIR, HANDOVER_SHA256,
                        PROTOCOL_ID, ROOT, SELECTION_RULE_ID, g09_source_hashes, matrix_index, sha, write_json)

MANIFEST_PATH = G09_DIR / 'G09_PACKAGE_MANIFEST.json'


def build() -> dict:
    sources = g09_source_hashes()
    index = matrix_index()
    protocol_files = {name: sha(G09_DIR / 'protocol' / name) for name in HANDOVER_SHA256}
    candidates = json.loads((G09_DIR / 'protocol' / 'G09_protocol_design.json').read_text(encoding='utf-8'))['candidates']
    return {
        'version': 'g09_a2_isolated_entry_v1',
        'protocol_id': PROTOCOL_ID,
        'status': 'G09-A implemented, no training executed',
        'current_execution_scope': 'isolated code + no-training verification only',
        'gpu_training_authorized_by_this_package': False,
        'utc': datetime.now(timezone.utc).isoformat(),
        'project_root': str(ROOT),
        'new_source_files': sources,
        'new_source_count': len(sources),
        'protocol_files': protocol_files,
        'candidate_table': candidates,
        'candidate_ids': EXPECTED_CANDIDATES,
        'folds': EXPECTED_FOLDS,
        'methods': EXPECTED_METHODS,
        'run_matrix_entries': len(index),
        'run_paths': sorted(index),
        'selection_rule_id': SELECTION_RULE_ID,
        'original_manifest': {'path': 'PACKAGE_MANIFEST.json',
                              'sha256': sha(ROOT / 'PACKAGE_MANIFEST.json'),
                              'note': 'v1 manifest untouched; its hash is pinned in the G09 manifest and every G09 contract'},
        'legacy_source_anchors': {'core.py': sha(ROOT / 'core.py'), 'runner.py': sha(ROOT / 'runner.py'),
                                  'evaluate.py': sha(ROOT / 'evaluate.py'), 'compare.py': sha(ROOT / 'compare.py'),
                                  'audit_runs.py': sha(ROOT / 'audit_runs.py')},
        'author_source_anchors': {name: sha(ROOT / 'vendor/domainbed' / name)
                                  for name in ('algorithms.py', 'networks.py', 'mixup_module.py')},
        'written_artifacts': {'runs_g09_v1': 'not created in G09-A',
                              'selections_g09_v1': 'not created in G09-A',
                              'engineering_g09_v1': 'test sandboxes only',
                              'final_packages': 'G09-A delivery zip + sha256'},
        'excludes': ['data', 'weights', 'vendor bodies', 'reference', 'runs/*.pt', 'credentials', 'caches'],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=str(MANIFEST_PATH))
    args = parser.parse_args(argv)
    payload = build()
    write_json(args.out, payload)
    print(json.dumps({'manifest': args.out, 'new_source_files': payload['new_source_count'],
                      'matrix_entries': payload['run_matrix_entries'],
                      'manifest_sha256': sha(args.out)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
