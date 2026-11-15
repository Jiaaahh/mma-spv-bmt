"""A2 outer gate: validate all four selections; recompute rankings from inner inputs."""
import hashlib
import json
from g09_config import (EXPECTED_FOLDS, EXPECTED_METHODS, EXPECTED_CANDIDATES,
    SELECTION_DIR_NAME, SELECTION_RULE_ID, PROTOCOL_ID, ROOT, resolve_sandbox,
    FOLD_INNER_FONT, run_path)

def selection_digest(data):
    payload = {k:v for k,v in data.items() if k != 'integrity_sha256'}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()

def selection_path(method, fold, selection_root=None):
    if method not in EXPECTED_METHODS or fold not in EXPECTED_FOLDS:
        raise RuntimeError('Invalid selection identity')
    return resolve_sandbox(selection_root, ROOT / SELECTION_DIR_NAME) / f'{method}_f{fold}.json'

def load_selection(method, fold, selection_root=None):
    path = selection_path(method, fold, selection_root)
    if not path.is_file():
        raise RuntimeError(f'Selection manifest missing: {path}')
    return json.loads(path.read_text(encoding='utf-8'))

def verify_selection_inputs(method, fold, selection_root=None, run_root=None):
    from g09_select import read_inner, rank_candidates
    data = load_selection(method, fold, selection_root)
    expected = dict(schema=2, protocol_id=PROTOCOL_ID, selection_rule_id=SELECTION_RULE_ID,
        method=method, fold=fold, frozen=True, required_candidates=6,
        selection_step=5000, inner_font=FOLD_INNER_FONT[fold],
        diagnostic_steps_not_used_for_selection=[1000,2000,3000],
        cross_fold_selection=False, outer_scores_read=False,
        ordering_rule=['correct_count_descending','cross_entropy_float64_rhe8_ascending','candidate_id_ascending'])
    for key, value in expected.items():
        if data.get(key) != value or type(data.get(key)) is not type(value):
            raise RuntimeError(f'{method} f{fold}: invalid selection {key}')
    if not isinstance(data.get('frozen_utc'), str) or not data['frozen_utc']:
        raise RuntimeError('Missing freeze timestamp')
    if data.get('integrity_sha256') != selection_digest(data):
        raise RuntimeError('Frozen selection integrity digest mismatch')
    inputs = data.get('input_hashes')
    if not isinstance(inputs, dict) or set(inputs) != set(EXPECTED_CANDIDATES):
        raise RuntimeError('Selection must cover exactly C01..C06')
    records = []
    for cid in EXPECTED_CANDIDATES:
        record = read_inner(run_path(fold, method, cid), method, fold, cid,
                            run_root=run_root, revalidate_frozen=True)
        expected_input = {k:record[k] for k in ('run_path','inner_dir','files',
            'artifact_hashes_sha256','checkpoint_sha256','contract_sha256')}
        if inputs[cid] != expected_input:
            raise RuntimeError(f'{method} f{fold} {cid}: frozen input changed')
        records.append(record)
    ranking = rank_candidates(records)
    if data.get('ranking') != ranking:
        raise RuntimeError('Frozen ranking differs from recomputed inner ranking')
    winner = {k:ranking[0][k] for k in ('candidate_id','run_path','inner_dir','correct',
        'total','accuracy_percent','cross_entropy_float64','cross_entropy_rhe8')}
    if data.get('winner') != winner:
        raise RuntimeError('Frozen winner differs from recomputed rank 1')
    return data

def frozen_status(selection_root=None, run_root=None):
    status = {}
    for method in EXPECTED_METHODS:
        for fold in EXPECTED_FOLDS:
            path = selection_path(method, fold, selection_root)
            entry = dict(path=path.relative_to(ROOT).as_posix(), exists=path.is_file(),
                         frozen=False, winner=None, reason=None)
            try:
                data = verify_selection_inputs(method, fold, selection_root, run_root)
                entry.update(frozen=True, winner=data['winner']['run_path'])
            except (RuntimeError, ValueError, KeyError, TypeError, OSError) as error:
                entry['reason'] = str(error)
            status[(method,fold)] = entry
    return status

def require_all_selections_frozen(selection_root=None, run_root=None):
    result = {}
    for method in EXPECTED_METHODS:
        for fold in EXPECTED_FOLDS:
            result[(method,fold)] = verify_selection_inputs(method,fold,selection_root,run_root)
    return result

def selected_run(method, fold, selection_root=None, run_root=None):
    all_data = require_all_selections_frozen(selection_root, run_root)
    data = all_data[(method,fold)]
    return dict(method=method,fold=fold,selection_path=selection_path(method,fold,selection_root),
                selection=data,winner=data['winner'])

def require_outer_allowed(method, fold, run_path, selection_root=None, run_root=None):
    info = selected_run(method,fold,selection_root,run_root)
    if run_path != info['winner']['run_path']:
        raise RuntimeError('Outer evaluation is restricted to the selected winner')
    return info
