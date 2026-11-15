"""A2: shared fail-closed contract validation; never constructs a model."""
import copy
import json
from types import SimpleNamespace
from g09_config import (ROOT, G09_DIR, FROZEN_PACKAGE_MANIFEST_SHA256, candidate,
                        g09_source_hashes, load_protocol, sha, run_id)

ANCHOR_HASHES = {'fold0_v1_contract.json': 'e678455de5c82952a0bf40a5fa405bf7a10732d9782ef2706b672292121b1541', 'fold1_v1_contract.json': '71364edbc84e504312a1a1e3e424eda229b373bdc65572248a9e220d2d953227'}  # Filled and pinned by the delivery builder.

def require_fixed_arguments(method, fold, k, seed, threads):
    if method not in ('erm', 'xdm') or fold not in (0, 1):
        raise ValueError('G09 requires method erm/xdm and fold 0/1')
    if (k, seed, threads) != (8, 17, 4):
        raise ValueError('G09 formal runs require K=8, seed=17, threads=4')

def verify_source_manifest():
    manifest = json.loads((G09_DIR / 'G09_PACKAGE_MANIFEST.json').read_text(encoding='utf-8'))
    actual = g09_source_hashes()
    if manifest.get('new_source_files') != actual or manifest.get('new_source_count') != len(actual):
        raise RuntimeError('G09 source manifest is stale or source file set changed')
    for name, digest in ANCHOR_HASHES.items():
        if sha(G09_DIR / 'anchors' / name) != digest:
            raise RuntimeError('Frozen v1 contract anchor changed: ' + name)
    if sha(ROOT / 'PACKAGE_MANIFEST.json') != FROZEN_PACKAGE_MANIFEST_SHA256:
        raise RuntimeError('Original package manifest changed')
    protocol = load_protocol()
    for name, digest in protocol['source_anchors'].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError('Frozen legacy source/protocol changed: ' + name)

def expected_contract(method, fold, candidate_id):
    """Build exact formal contract from reviewed per-fold v1 contracts, without training."""
    require_fixed_arguments(method, fold, 8, 17, 4)
    name = f'fold{fold}_v1_contract.json'
    if sha(G09_DIR / 'anchors' / name) != ANCHOR_HASHES[name]:
        raise RuntimeError('Frozen contract anchor changed')
    legacy = json.loads((G09_DIR / 'anchors' / name).read_text(encoding='utf-8'))
    legacy['method'] = method
    from g09_core import build_g09_contract
    obj = SimpleNamespace(contract=legacy, hp=copy.deepcopy(legacy['xdm_hparams']))
    return build_g09_contract(obj, candidate(candidate_id), run_id(fold, method, candidate_id))

def validate_contract(contract, method, fold, candidate_id):
    verify_source_manifest()
    expected = expected_contract(method, fold, candidate_id)
    if contract != expected:
        keys = sorted(k for k in set(contract) | set(expected) if contract.get(k) != expected.get(k))
        raise RuntimeError('G09 immutable contract mismatch: ' + ', '.join(keys))
    return contract

def verify_adam_groups(optimizer):
    if optimizer is None:
        return
    for group in optimizer.param_groups:
        if (tuple(group.get('betas', ())), group.get('eps'), group.get('foreach')) != ((.9, .999), 1e-8, False):
            raise RuntimeError('Effective Adam param_group betas/eps/foreach changed')
        for flag in ('amsgrad', 'maximize', 'capturable', 'differentiable', 'decoupled_weight_decay'):
            if group.get(flag, False) is not False:
                raise RuntimeError('Unsupported Adam param_group flag: ' + flag)
        if group.get('fused') not in (None, False):
            raise RuntimeError('Fused Adam is not the frozen optimizer')
