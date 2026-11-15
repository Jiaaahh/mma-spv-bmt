"""Frozen CLIP feature extraction, independent of labels and accuracy."""
from pathlib import Path
import os,sys,json,hashlib,time,threading
from datetime import datetime,timezone
os.environ['OMP_NUM_THREADS']='2'; os.environ['MKL_NUM_THREADS']='2'
import numpy as np
import torch,clip,psutil
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent
PRE=ROOT/'outputs/proto_adapter_full_preflight_20260929'
sys.stdout.reconfigure(encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def dump(n,v):
    p=OUT/n; p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.tmp'); tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8'); os.replace(tmp,p)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
protocol=read(PRE/'frozen_protocol.json')
assert sha(PRE/'frozen_protocol.json')==read(PRE/'readiness.json')['frozen_protocol_sha256']
for n,h in protocol['evidence_hashes'].items():assert sha(PRE/n)==h,n
weight=ROOT/'work/proto_adapter_clip_weights/RN50.pt'
assert sha(weight)==protocol['weight_sha256']
source=read(PRE/'clip_source_manifest.json'); import clip.clip as clip_module
installed=Path(clip_module.__file__).parents[1]
for n,h in source['source_files'].items():
    if Path(n).parts[0]=='clip' and Path(n).suffix in ['.py','.gz']:assert sha(installed/n)==h,n
manifest=dict(protocol_sha256=sha(PRE/'frozen_protocol.json'),script_sha256=sha(__file__),source_commit=source['commit'],weight_sha256=sha(weight))
if (OUT/'extraction_manifest.json').exists():assert read(OUT/'extraction_manifest.json')==manifest
else:dump('extraction_manifest.json',manifest)
torch.set_num_threads(2); torch.set_num_interop_threads(1)
proc=psutil.Process(); peak=[0]; stop=threading.Event(); started=time.perf_counter()
def monitor():
    while not stop.wait(.5):peak[0]=max(peak[0],proc.memory_info().rss)
threading.Thread(target=monitor,daemon=True).start()
model,preprocess=clip.load(str(weight),device='cpu',jit=False)
model.eval(); model.requires_grad_(False)
assert not any(p.requires_grad for p in model.parameters())
def convert(a):
    assert np.isfinite(a).all() and a.min()>=0 and a.max()>0
    return Image.fromarray(np.rint(np.clip(a/a.max(),0,1)*255).astype(np.uint8)).convert('RGB')
files=read(PRE/'input_audit.json')['files']; completed=[]
with torch.inference_mode():
    for item in files:
        path=Path(item['path']); assert sha(path)==item['sha256']
        tag=path.stem.removesuffix('_images'); dest=OUT/'features'/f'{tag}.npy'; info=dest.with_suffix('.json')
        dest.parent.mkdir(exist_ok=True)
        if dest.exists() and info.exists():
            old=read(info); assert sha(dest)==old['feature_sha256'] and old['input_sha256']==item['sha256']
            completed.append(old); print('REUSED',tag,flush=True); continue
        x=np.load(path,mmap_mode='r'); chunks=[]; t=time.perf_counter()
        for i in range(0,len(x),8):
            tensor=torch.stack([preprocess(convert(a)) for a in x[i:i+8]])
            z=model.encode_image(tensor); z=z/z.norm(dim=1,keepdim=True)
            assert torch.isfinite(z).all(); chunks.append(z.numpy())
            if i%256==0:
                progress=dict(stage='extracting',file=tag,done_in_file=min(i+8,len(x)),n_in_file=len(x),completed_files=len(completed),total_files=len(files),elapsed_seconds=time.perf_counter()-started)
                dump('progress.json',progress); print(json.dumps(progress),flush=True)
        z=np.concatenate(chunks); assert z.shape==(len(x),1024) and np.allclose(np.linalg.norm(z,axis=1),1,atol=1e-5)
        if tag=='ideal': assert np.allclose(z[:256],np.load(PRE/'benchmark_train_features.npy'),rtol=0,atol=1e-6)
        temp=dest.with_suffix('.tmp.npy'); np.save(temp,z); os.replace(temp,dest)
        details=dict(tag=tag,n=len(x),input_sha256=item['sha256'],feature_sha256=sha(dest),seconds=time.perf_counter()-t,dtype=str(z.dtype),shape=list(z.shape))
        dump(str(info.relative_to(OUT)),details); completed.append(details)
        print('COMPLETE',tag,len(x),flush=True)
stop.set()
dump('feature_audit.json',dict(status='PASS',files=completed,total_images=sum(d['n'] for d in completed),unit_norm=True,preflight_256_features_replayed=True,accuracy_computed=False))
dump('extraction_resources.json',dict(seconds=time.perf_counter()-started,peak_rss_bytes=peak[0],threads=2,batch=8,device='cpu'))
dump('progress.json',dict(stage='FEATURE_EXTRACTION_COMPLETE',images=sum(d['n'] for d in completed),utc=datetime.now(timezone.utc).isoformat()))
print('ALL FEATURES COMPLETE',flush=True)
