"""Post-result engineering diagnostics only; do not change the scored protocol."""
from pathlib import Path
import os,sys,json
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
import numpy as np
import torch,clip
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent;PRE=OUT.parent/'proto_adapter_full_preflight_20260929'
sys.stdout.reconfigure(encoding='utf-8');torch.set_num_threads(2);torch.set_num_interop_threads(1)
model,preprocess=clip.load(str(ROOT/'work/proto_adapter_clip_weights/RN50.pt'),device='cpu',jit=False);model.eval();model.requires_grad_(False)
checks=[]
with torch.inference_mode():
    for name in ['ideal','moderate_seed2026','E1','C1']:
        cache=ROOT/'work/mma_supplement_cache' if name not in ['E1','C1'] else ROOT/'outputs/randumb_cpu_20260928/external'
        images=np.load(cache/f'{name}_images.npy',mmap_mode='r'); stored=np.load(OUT/'features'/f'{name}.npy')
        ix=np.linspace(0,len(images)-1,8,dtype=int)
        batch=[]
        for a in images[ix]:
            # Independently spell out the fixed conversion.
            pixels=np.round(255*np.minimum(1,np.maximum(0,a/np.max(a)))).astype('uint8')
            batch.append(preprocess(Image.fromarray(pixels).convert('RGB')))
        v=model.encode_image(torch.stack(batch));v=v/v.norm(dim=1,keepdim=True)
        err=float(np.max(np.abs(v.numpy()-stored[ix])));assert err<1e-6
        checks.append(dict(panel=name,indices=ix.tolist(),max_abs_error=err))
def norm(x):
    x=x.astype(np.float64);return x/np.linalg.norm(x,axis=1,keepdims=True)
labels=np.load(ROOT/'work/mma_supplement_cache/metadata.npz')['labels']
text=norm(np.load(PRE/'text_features.npy'));selection=json.loads((OUT/'selected_alpha.json').read_text())['methods'];support=[]
for multi in [0,1]:
    names=['ideal']+([f'moderate_seed{s}' for s in range(2026,2030)] if multi else [])
    q=norm(np.concatenate([np.load(OUT/'features'/f'{n}.npy') for n in names]));y=np.tile(labels,len(names))
    p=np.load(OUT/'prototypes'/f'full_PA{multi}.npy');u=np.load(OUT/'prototypes'/f'full_PLAIN{multi}.npy')
    for label,s in [('PA',selection[f'PA{multi}']['alpha']*q@p.T+q@text.T),('VIS',q@p.T),('PLAIN',q@u.T)]:
        pred=s.argmax(1);cnt=np.bincount(pred,minlength=180)
        support.append(dict(method=f'{label}{multi}',n=len(y),training_resubstitution_accuracy_percent=100*float(np.mean(pred==y)),predicted_class_count=int((cnt>0).sum()),largest_predicted_class_fraction=float(cnt.max()/len(y))))
    # A self-retrieval diagnostic for feature identity, not a generalization score.
    correct=0
    for i in range(0,len(q),128):correct+=int(np.sum(y[(q[i:i+128]@q.T).argmax(1)]==y[i:i+128]))
    support.append(dict(method=f'SELF_RETRIEVAL{multi}',n=len(y),training_resubstitution_accuracy_percent=100*correct/len(y),note='Includes query itself; checks identity only, not a benchmark.'))
result=dict(status='PASS',performed_after_observing_low_accuracy=True,protocol_changed=False,
    direct_reencoding=checks,training_diagnostics=support,
    interpretation='Technical consistency audit only. Does not establish reproduction of author benchmark performance or ecological validity of this task adaptation.')
(OUT/'low_accuracy_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False))
