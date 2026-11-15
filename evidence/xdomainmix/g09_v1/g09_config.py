"""G09-A 配置层：载入冻结的 G09 方案/候选表/运行矩阵，提供散列与常量。

只读复用 v1 资产（data/weights/protocols/vendor/PACKAGE_MANIFEST.json），
**不写、不改**任何 v1 文件；所有 G09 新产物只落在 g09_v1/、runs_g09_v1/、
engineering_g09_v1/、selections_g09_v1/ 之下。
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

G09_DIR = Path(__file__).resolve().parent
ROOT = G09_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROTOCOL_ID = 'G09_SYMMETRIC_DEV_V1'
SELECTION_RULE_ID = 'G09_SELECT_V1__CORRECT_DESC__CE_FLOAT64_RHE8_ASC__CANDIDATE_ID_ASC'
INNER_DIAGNOSTIC_STEPS = [1000, 2000, 3000]
INNER_SELECTION_STEP = 5000
TRAINING_TARGET = 5000
SAMPLER_ID = 'torch_cpu_generator_170909_distinct_class_randperm_then_one_of_four_font_randint_v1'
TOTAL_BATCH_PER_CLASS = 5
HEAD_TO_BACKBONE_RATIO = 20
EXPECTED_METHODS = ['erm', 'xdm']
EXPECTED_FOLDS = [0, 1]
EXPECTED_CANDIDATES = ['C01', 'C02', 'C03', 'C04', 'C05', 'C06']
EXPECTED_RUN_COUNT = 24
EXPECTED_FORMAL_UPDATES = 120000
TRAIN_MAPS = ['ideal'] + [f'moderate_seed{s}' for s in range(2026, 2030)]
OUTER_MAPS = [f'moderate_seed{s}' for s in range(5301, 5306)]
FOLD_INNER_FONT = {0: 'zcool_kuaile', 1: 'noto_sans_sc_thin'}
FOLD_OUTER_FONT = {0: 'liu_jian_mao_cao', 1: 'ma_shan_zheng'}
V1_ANCHOR_CANDIDATE = 'C03'
FROZEN_PACKAGE_MANIFEST_SHA256 = '11d0ef4651ce6d3d438d00d1e7a7236499cc623e1f8092ffadc9153cf9eb5e78'

RUNS_DIR_NAME = 'runs_g09_v1'
ENGINEERING_DIR_NAME = 'engineering_g09_v1'
SELECTION_DIR_NAME = 'selections_g09_v1'
PROTOCOL_DIR = G09_DIR / 'protocol'
PROTOCOL_FILE = PROTOCOL_DIR / 'G09_protocol_design.json'
MATRIX_FILE = PROTOCOL_DIR / 'G09_run_matrix.json'
HANDOVER_FILE = PROTOCOL_DIR / 'G09_对称开发搜索方案与A阶段交接.md'
RUNS_DIR = ROOT / RUNS_DIR_NAME
ENGINEERING_DIR = ROOT / ENGINEERING_DIR_NAME
SELECTION_DIR = ROOT / SELECTION_DIR_NAME

# 交接包内三个文件的散列（= 收到时的 .zip 与本目录副本，A 阶段开始时已核对）
HANDOVER_SHA256 = {
    'G09_protocol_design.json': 'b494f62db7431d8a0b24f2e48637833e26df96854f7f4d28e9df18e2ffe4f770',
    'G09_run_matrix.json': '5161ba54bcb13d927da8ef1153f25244bf700018c7b330c93f4d27578afb8b57',
    'G09_对称开发搜索方案与A阶段交接.md': '93eb7516da3e12833bb32bab67fec0462f96a8eca39b9b0b20c05d32240f89c1',
}

EXPECTED_SELECTION_RULE = {
    'scope': 'per_method_per_fold_only',
    'required_candidates': 6,
    'primary': 'correct_count_descending',
    'secondary': 'float64_logsumexp_cross_entropy_decimal_round_half_even_8dp_ascending',
    'tertiary': 'candidate_id_ascending',
    'cross_fold_selection': False,
    'outer_access': 'after all four selection manifests frozen; selected final checkpoints only',
}


def sha(path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return sha(path)


def assert_handover_files() -> dict:
    """A 阶段入口的硬性前置：三份交接文件必须与收到时逐字节一致。"""
    observed = {}
    for name, expected in HANDOVER_SHA256.items():
        path = PROTOCOL_DIR / name
        if not path.is_file():
            raise RuntimeError(f'G09 handover file missing: {path}')
        got = sha(path)
        if got != expected:
            raise RuntimeError(f'G09 handover file differs from received copy: {name} {got} != {expected}')
        observed[name] = got
    return observed


def load_protocol() -> dict:
    """载入 G09 方案；核对 A 阶段边界（不得把本包当训练授权）。"""
    assert_handover_files()
    design = read_json(PROTOCOL_FILE)
    if design.get('protocol_id') != PROTOCOL_ID:
        raise RuntimeError(f'Unexpected G09 protocol_id: {design.get("protocol_id")!r}')
    if design.get('gpu_training_authorized_by_this_package') is not False:
        raise RuntimeError('G09-A package must not authorize GPU training')
    if design.get('status') != 'DESIGN_FOR_IMPLEMENTATION_REVIEW':
        raise RuntimeError(f'Unexpected G09 design status: {design.get("status")!r}')
    if [c['candidate_id'] for c in design.get('candidates', [])] != EXPECTED_CANDIDATES:
        raise RuntimeError('Candidate table is not the frozen C01..C06 sequence')
    if design.get('folds') != EXPECTED_FOLDS or design.get('methods') != EXPECTED_METHODS:
        raise RuntimeError('Protocol fold/method list mismatch')
    if (design.get('seed'), design.get('k'), design.get('total_batch')) != (17, 8, 40):
        raise RuntimeError('Protocol seed/k/total_batch mismatch')
    if (design.get('training_target'), design.get('inner_selection_step')) != (TRAINING_TARGET, INNER_SELECTION_STEP):
        raise RuntimeError('Protocol target/selection step mismatch')
    if design.get('inner_diagnostic_steps') != INNER_DIAGNOSTIC_STEPS:
        raise RuntimeError('Protocol diagnostic step list mismatch')
    if design.get('sampler_seed') != 170909:
        raise RuntimeError('Protocol sampler seed mismatch')
    if design.get('selection') != EXPECTED_SELECTION_RULE:
        raise RuntimeError('Protocol selection rule mismatch')
    if design.get('historical_run_reuse') is not False:
        raise RuntimeError('G09 must not reuse historical runs')
    if design.get('candidate_training_runs') != EXPECTED_RUN_COUNT or design.get('formal_updates_total') != EXPECTED_FORMAL_UPDATES:
        raise RuntimeError('Protocol run/update count mismatch')
    optimizer = design.get('optimizer') or {}
    if (optimizer.get('name'), tuple(optimizer.get('betas', [])), optimizer.get('eps'),
            optimizer.get('foreach'), optimizer.get('weight_decay_exemptions')) != (
            'Adam', (0.9, 0.999), 1e-08, False, []):
        raise RuntimeError('Protocol optimizer base semantics mismatch')
    return design


def candidate_table() -> dict:
    return {c['candidate_id']: dict(c) for c in load_protocol()['candidates']}


def candidate(candidate_id: str) -> dict:
    table = candidate_table()
    if candidate_id not in table:
        raise ValueError(f'Illegal candidate_id {candidate_id!r}; allowed: {EXPECTED_CANDIDATES}')
    return table[candidate_id]


def run_id(fold: int, method: str, candidate_id: str) -> str:
    if fold not in EXPECTED_FOLDS or method not in EXPECTED_METHODS or candidate_id not in EXPECTED_CANDIDATES:
        raise ValueError(f'Illegal G09 run components: fold={fold!r} method={method!r} candidate={candidate_id!r}')
    return f'f{fold}_s17_k8_{method}_{candidate_id}'


def run_path(fold: int, method: str, candidate_id: str) -> str:
    return f'{RUNS_DIR_NAME}/{run_id(fold, method, candidate_id)}'


def load_matrix() -> list:
    matrix = read_json(MATRIX_FILE)
    if not isinstance(matrix, list) or len(matrix) != EXPECTED_RUN_COUNT:
        raise RuntimeError('G09 run matrix must contain exactly 24 entries')
    return matrix


def matrix_index() -> dict:
    """run_path -> matrix entry，并核对矩阵自身的唯一性/对称性/命名/顺序。"""
    matrix = load_matrix()
    index = {}
    expected_order = []
    for ordinal, entry in enumerate(matrix, 1):
        if entry.get('ordinal') != ordinal:
            raise RuntimeError(f'Run matrix ordinal mismatch at position {ordinal}')
        cid = entry['candidate_id']
        fold, method = entry['fold'], entry['method']
        want_id = run_id(fold, method, cid)
        want_path = f'{RUNS_DIR_NAME}/{want_id}'
        if entry.get('run_id') != want_id or entry.get('run_path') != want_path:
            raise RuntimeError(f'Run matrix naming mismatch: {entry!r}')
        if entry.get('start_step') != 0 or entry.get('target_step') != TRAINING_TARGET:
            raise RuntimeError(f'Run matrix start/target mismatch: {entry!r}')
        if entry.get('source_initialization') != f'ImageNetV1+fold{fold}_warm_head':
            raise RuntimeError(f'Run matrix initialization mismatch: {entry!r}')
        if entry.get('execution_status') != 'NOT_STARTED':
            raise RuntimeError(f'Run matrix must start as NOT_STARTED: {entry!r}')
        if want_path in index:
            raise RuntimeError(f'Duplicate G09 run path in matrix: {want_path}')
        index[want_path] = entry
        expected_order.append((cid, fold, method))
    want_order = [(cid, fold, method) for cid in EXPECTED_CANDIDATES for fold in EXPECTED_FOLDS for method in EXPECTED_METHODS]
    if expected_order != want_order:
        raise RuntimeError('Run matrix order must be C01..C06, each fold0 then fold1, each ERM then XDM')
    for cid in EXPECTED_CANDIDATES:
        for fold in EXPECTED_FOLDS:
            if not any(e['candidate_id'] == cid and e['fold'] == fold and e['method'] == m
                       for m in EXPECTED_METHODS for e in matrix):
                raise RuntimeError(f'Matrix misses {cid} fold{fold}')
    return index


def pair_contract_plan(fold: int, candidate_id: str) -> dict:
    """同折同候选的 ERM/XDM 配对计划（用于审计与比较的声明式差异）。"""
    return {'fold': fold, 'candidate_id': candidate_id,
            'runs': {'erm': run_path(fold, 'erm', candidate_id), 'xdm': run_path(fold, 'xdm', candidate_id)}}


def protocol_hashes() -> dict:
    return {'g09_protocol_sha256': sha(PROTOCOL_FILE),
            'g09_candidate_table_sha256': sha(PROTOCOL_FILE),
            'g09_run_matrix_sha256': sha(MATRIX_FILE),
            'g09_handover_sha256': sha(HANDOVER_FILE),
            'original_manifest_sha256': sha(ROOT / 'PACKAGE_MANIFEST.json') if (ROOT / 'PACKAGE_MANIFEST.json').is_file() else None}


def g09_source_hashes() -> dict:
    """新入口源码散列（含测试），供合同与 manifest 记录。"""
    files = sorted(p for p in G09_DIR.rglob('*.py') if p.is_file())
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in files}


def resolve_sandbox(path, real_root):
    """工程测试用沙箱根解析：只允许落在 engineering_g09_v1/ 之下。

    CLI **不暴露**该参数（正式入口永远使用 runs_g09_v1/ 与 selections_g09_v1/），
    因此这不是绕过门禁的开关；测试必须把夹具放进工程目录，避免创建正式训练状态。
    """
    if path is None:
        return Path(real_root)
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    candidate = candidate.resolve()
    engineering = (ROOT / ENGINEERING_DIR_NAME).resolve()
    if candidate != engineering and engineering not in candidate.parents:
        raise RuntimeError(f'Sandbox root must live under {ENGINEERING_DIR_NAME}/: {candidate}')
    return candidate


def expected_param_group_plan(cand: dict) -> list:
    """合同里声明的 optimizer param_groups 计划（角色 + 有效 lr/wd）。"""
    return [
        {'index': 0, 'role': 'backbone_and_other_trainable', 'lr': cand['backbone_lr'], 'weight_decay': cand['weight_decay']},
        {'index': 1, 'role': 'classification_head_fc', 'lr': cand['head_lr'], 'weight_decay': cand['weight_decay']},
    ]
