"""Read-only input audit: never compute development or test recognition accuracy."""
from pathlib import Path
import sys, json, hashlib, platform, shutil, importlib.util
from datetime import datetime, timezone
import numpy as np
import psutil

sys.stdout.reconfigure(encoding='utf-8')
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
CACHE = ROOT/'work/mma_supplement_cache'
EXT = ROOT/'outputs/randumb_cpu_20260928/external'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def save(name, value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

names=['ideal']+[f'moderate_seed{s}' for s in range(2026,2030)]+[f'moderate_seed{s}' for s in range(5301,5306)]
paths=[CACHE/f'{name}_images.npy' for name in names]+[EXT/f'{p}_images.npy' for p in ['E1','C1']]
records=[]
old=json.loads((EXT/'manifest.json').read_text(encoding='utf-8'))
for p in paths:
    a=np.load(p,mmap_mode='r',allow_pickle=False)
    expected=3600 if p.name=='E1_images.npy' else 1800 if p.name=='C1_images.npy' else 1080
    assert a.shape==(expected,96,96), (p,a.shape)
    assert np.isfinite(a).all() and a.min()>=0
    sums=a.sum((1,2),dtype=np.float64)
    assert np.allclose(sums,64,atol=2e-5,rtol=0)
    digest=sha(p)
    if p.parent==EXT:
        assert digest==old['panels'][p.name[:2]]['image_file_sha256']
    # No test-derived brightness or normalization statistic is computed here.
    records.append(dict(path=str(p),sha256=digest,shape=list(a.shape),dtype=str(a.dtype),bytes=p.stat().st_size,status='PASS'))

meta=np.load(CACHE/'metadata.npz',allow_pickle=False)
y=meta['labels']; fonts=meta['fonts']; chars=meta['characters']
assert len(np.unique(fonts))==6 and np.array_equal(np.unique(y),np.arange(180))
classes=[]
for c in range(180):
    ix=y==c; v=np.unique(chars[ix]); assert len(v)==1 and ix.sum()==6
    assert len(np.unique(fonts[ix]))==6
    classes.append(dict(class_id=c,character=str(v[0]),codepoint=f'U+{ord(str(v[0])):04X}'))
assert len({v['character'] for v in classes})==180
save('classes.json',classes)
save('input_audit.json',dict(status='PASS',timestamp=datetime.now(timezone.utc).isoformat(),files=records,
    metadata_sha256=sha(CACHE/'metadata.npz'),classes=180,fonts=6,unique_images=sum(r['shape'][0] for r in records),
    accuracy_computed=False,external_hash_matches_previous_manifest=True))
save('environment.json',dict(timestamp=datetime.now(timezone.utc).isoformat(),python=sys.version,executable=sys.executable,
    platform=platform.platform(),cpu_logical=psutil.cpu_count(),memory=psutil.virtual_memory()._asdict(),
    disk=shutil.disk_usage(ROOT)._asdict(),packages={k:bool(importlib.util.find_spec(k)) for k in ['torch','clip','open_clip','transformers','ftfy','regex']},
    checked_other_environment='work/resnet50_cpu_env: Python 3.12.14, torch 2.6.0+cpu, torchvision present; clip/open_clip/transformers/ftfy/regex absent',
    torch_probe='/historical/Anaconda and work/resnet50_cpu_env both report torch 2.6.0+cpu and cuda.is_available() false',
    hardware_gpu='Unconfirmed: Win32_VideoController inventory denied; nvidia-smi not on PATH. CPU torch does not prove absence of a physical GPU.',
    weight_search_scope=['~/.cache/clip','~/.cache/huggingface/hub'],clip_weights_found_in_checked_default_locations=False,
    runtime_benchmark='NOT_RUN: CLIP package and RN50 weights absent'))
print(json.dumps(dict(status='PASS',unique_images=16200,classes=180,formal_scores_computed=False),ensure_ascii=False))
