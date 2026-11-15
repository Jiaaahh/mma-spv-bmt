"""Shared, development-only supplement contracts."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
CACHE = ROOT.parents[1] / 'work' / 'mma_supplement_cache'
PLAN_PATH = ROOT / 'configs' / 'supplement.json'
PLAN = json.loads(PLAN_PATH.read_text(encoding='utf-8'))
OLD = Path(r'/historical/user/Documents/Codex/2026-08-25/referenced-chatgpt-conversation-this-is-an-2/outputs/hanzi-spv-calibrated-v2')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)

def normalize(x):
    x = np.asarray(x, dtype=np.float64)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    if np.any(norm == 0) or not np.isfinite(x).all():
        raise ValueError('nonfinite or zero feature')
    return x / norm

def class_means(features, labels):
    features = np.asarray(features, dtype=np.float64)
    classes = np.unique(labels)
    assert np.array_equal(classes, np.arange(180))
    return np.stack([features[labels == c].mean(0) for c in classes])

def validate_split(fonts, heldout, train_seeds, eval_seeds):
    if set(train_seeds) & set(eval_seeds):
        raise ValueError('mapping leakage')
    train = np.flatnonzero(fonts != heldout)
    val = np.flatnonzero(fonts == heldout)
    if len(train) != 900 or len(val) != 180:
        raise ValueError('unbalanced font split')
    if set(fonts[train]) & set(fonts[val]):
        raise ValueError('font leakage')
    return train, val

def check_cache():
    m = json.loads((CACHE / 'manifest.json').read_text(encoding='utf-8'))
    if m['plan_sha256'] != sha(PLAN_PATH):
        raise ValueError('cache protocol differs')
    return m

def mapping_name(seed):
    return 'ideal' if seed == 0 else f'moderate_seed{seed}'
