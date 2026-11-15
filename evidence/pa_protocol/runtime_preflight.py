"""Engineering checks only: no accuracy, development selection, or test-set access."""
from pathlib import Path
import os, sys, json, time, hashlib, threading, platform
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
import numpy as np
import torch, clip, psutil
from PIL import Image, ImageDraw
sys.stdout.reconfigure(encoding='utf-8')
OUT=Path(__file__).resolve().parent; ROOT=OUT.parents[1]
torch.set_num_threads(2); torch.set_num_interop_threads(1)
def save(n,v): (OUT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def norm(x):
    n=np.linalg.norm(x,axis=-1,keepdims=True)
    assert np.isfinite(x).all() and (n>0).all()
    return x/n
def prototype(x,y,c):
    p=np.stack([norm(x)[y==i].mean(0) for i in range(c)])
    d=np.linalg.norm(p,axis=0,keepdims=True)
    return norm(np.divide(p,d,out=np.zeros_like(p),where=d>0))

# Independent scalar normalization and score checks.
rng=np.random.default_rng(260929); x=rng.normal(size=(30,11)); x[:,-1]=0
y=np.repeat(np.arange(3),10); p=prototype(x,y,3)
ref=np.array([[sum(x[i,j]/np.sqrt(sum(x[i,k]**2 for k in range(11))) for i in range(30) if y[i]==c)/10 for j in range(11)] for c in range(3)])
for j in range(11):
    d=np.sqrt(sum(ref[c,j]**2 for c in range(3)))
    if d>0: ref[:,j]/=d
for c in range(3): ref[c]/=np.sqrt(sum(ref[c,j]**2 for j in range(11)))
assert np.allclose(p,ref,atol=1e-12)
q=norm(rng.normal(size=(7,11))); t=norm(rng.normal(size=(3,11)))
s=.7*q@p.T+q@t.T
assert np.allclose(s*100,70*q@p.T+100*q@t.T)
save('math_audit.json',dict(status='PASS',max_abs_prototype_error=float(abs(p-ref).max()),zero_column_preserved=True,fusion_scale_equivalence=True))

initial=psutil.virtual_memory().available; proc=psutil.Process(); peak=[0]; stop=threading.Event()
def monitor():
    while not stop.wait(.1): peak[0]=max(peak[0],proc.memory_info().rss)
threading.Thread(target=monitor,daemon=True).start()
print('Loading verified RN50 on CPU',flush=True)
model,preprocess=clip.load(str(ROOT/'work/proto_adapter_clip_weights/RN50.pt'),device='cpu',jit=False)
model.eval(); model.requires_grad_(False)
assert not any(p.requires_grad for p in model.parameters())
classes=json.loads((OUT/'classes.json').read_text(encoding='utf-8'))
templates=['a Chinese character: {}.','an image of the Chinese character {}.','the Chinese character {}.',
           'a printed Chinese character {}.','a glyph of the Chinese character {}.',
           'a phosphene image of the Chinese character {}.','the Chinese character {} in simulated prosthetic vision.']
prompts=[template.format(c['character']) for template in templates for c in classes]
tokens=clip.tokenize(prompts,truncate=False)
for a in tokens.reshape(7,180,77): assert len(torch.unique(a,dim=0))==180
np.save(OUT/'prompt_tokens.npy',tokens.numpy())
save('prompts.json',dict(templates=templates,prompts=prompts))
text=[]; start=time.perf_counter()
with torch.inference_mode():
    for i in range(0,len(tokens),16):
        v=model.encode_text(tokens[i:i+16]); v=v/v.norm(dim=-1,keepdim=True); text.append(v.numpy())
        if i%256==0: print(f'TEXT {i}/{len(tokens)}',flush=True)
text=np.concatenate(text).reshape(7,180,-1); ensemble=norm(text.mean(0))
assert np.isfinite(ensemble).all() and len(np.unique(ensemble,axis=0))==180
np.save(OUT/'text_features.npy',ensemble)
save('text_audit.json',dict(status='PASS',classes=180,templates=7,token_collisions=0,truncated=False,shape=list(ensemble.shape),seconds=time.perf_counter()-start,
    note='Token and embedding uniqueness does not establish semantic understanding of Chinese characters.'))

images=np.load(ROOT/'work/mma_supplement_cache/ideal_images.npy',mmap_mode='r')
def pil(a):
    assert np.isfinite(a).all() and a.max()>0 and a.min()>=0
    return Image.fromarray(np.rint(np.clip(a/a.max(),0,1)*255).astype(np.uint8)).convert('RGB')
sheet=Image.new('RGB',(6*224,3*248),(35,35,35)); draw=ImageDraw.Draw(sheet)
for j,index in enumerate([0,1,2,8,20,60]):
    for row,name in enumerate(['ideal','moderate_seed2026','moderate_seed2029']):
        a=np.load(ROOT/f'work/mma_supplement_cache/{name}_images.npy',mmap_mode='r')[index]
        im=pil(a).resize((224,224),Image.Resampling.BICUBIC)
        sheet.paste(im,(j*224,row*248)); draw.text((j*224+2,row*248+226),f'{name} id={index}',fill='white')
sheet.save(OUT/'training_input_preview.png')
with torch.inference_mode():
    model.encode_image(torch.stack([preprocess(pil(a)) for a in images[:16]]))
    start=time.perf_counter(); features=[]
    for i in range(0,256,8):
        batch=torch.stack([preprocess(pil(a)) for a in images[i:i+8]])
        v=model.encode_image(batch); v=v/v.norm(dim=-1,keepdim=True)
        assert torch.isfinite(v).all(); features.append(v.numpy())
        if i%64==0: print(f'IMAGE {i}/256',flush=True)
    elapsed=time.perf_counter()-start
features=np.concatenate(features)
with torch.inference_mode():
    one=model.encode_image(preprocess(pil(images[0])).unsqueeze(0))
    one=(one/one.norm(dim=-1,keepdim=True)).numpy()[0]
batch_error=float(np.max(np.abs(one-features[0])))
assert batch_error<1e-4 and np.allclose(np.linalg.norm(features,axis=1),1,atol=1e-5)
np.save(OUT/'benchmark_train_features.npy',features)
stop.set(); peak[0]=max(peak[0],proc.memory_info().rss)
hours=elapsed/256*16200/3600
result=dict(status='PASS' if hours<=24 and peak[0]<=.6*initial else 'RESOURCE_REVIEW_REQUIRED',device='cpu',torch=torch.__version__,python=sys.version,
    platform=platform.platform(),threads=2,batch=8,warmup_images=16,timed_images=256,seconds=elapsed,images_per_second=256/elapsed,
    projected_16200_image_hours=hours,peak_process_rss_bytes=peak[0],available_ram_at_start_bytes=initial,
    text_seconds=json.loads((OUT/'text_audit.json').read_text(encoding='utf-8'))['seconds'],
    inference_only=True,recognition_accuracy_computed=False,panels_accessed=['training_only'],preprocess=str(preprocess),
    batch_vs_single_max_abs_feature_error=batch_error,
    conversion='float32 SPV / per-image positive maximum -> round(255*x) uint8 RGB -> official CLIP RN50 preprocessing',
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
save('runtime_benchmark.json',result); print(json.dumps(result,ensure_ascii=False),flush=True)
