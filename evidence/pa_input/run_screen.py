from pathlib import Path
import os,sys,json,hashlib,ast,time,csv
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
import numpy as np
import torch,clip
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
PRE=OUT.parent/'proto_adapter_full_preflight_20260929';OLD=OUT.parent/'proto_adapter_full_run_20260929';CACHE=ROOT/'work/mma_supplement_cache'
sys.stdout.reconfigure(encoding='utf-8');torch.set_num_threads(2);torch.set_num_interop_threads(1)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def save(n,x):
    p=OUT/n;p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(t,p)
cfg=read(OUT/'protocol.json');oldcfg=read(PRE/'frozen_protocol.json');source=OLD/'evaluate.py'
assert sha(source)==read(OLD/'evaluation_manifest.json')['script_sha256']
nodes=[n for n in ast.parse(source.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name in ['norm','build','choose']]
ns={'np':np};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns);norm,build,choose=[ns[k] for k in ['norm','build','choose']]
meta=np.load(CACHE/'metadata.npz');original_y=meta['labels'];classids=np.sort(np.random.default_rng(cfg['seed']).choice(180,60,replace=False));ix=np.flatnonzero(np.isin(original_y,classids))
y=np.searchsorted(classids,original_y[ix]);fonts=meta['fonts'][ix];samples=meta['sample_ids'][ix]
save('subset.json',dict(original_class_ids=classids.tolist(),source_rows=ix.tolist(),samples=samples.tolist(),new_labels=y.tolist(),fonts=fonts.tolist()))
assert len(ix)==360
for f in np.unique(fonts):assert np.array_equal(np.sort(y[fonts==f]),np.arange(60))
assert sha(CACHE/'metadata.npz')==read(PRE/'input_audit.json')['metadata_sha256']
weight=ROOT/'work/proto_adapter_clip_weights/RN50.pt';assert sha(weight)==oldcfg['weight_sha256']
manifest=dict(protocol_sha256=sha(OUT/'protocol.json'),script_sha256=sha(__file__),subset_sha256=sha(OUT/'subset.json'),shared_scoring_sha256=sha(source),weight_sha256=sha(weight))
if (OUT/'implementation_manifest.json').exists():assert read(OUT/'implementation_manifest.json')==manifest
else:save('implementation_manifest.json',manifest)
names=cfg['train_maps']+cfg['selection_maps']+cfg['confirmation_maps'];assert all(n=='ideal' or 2026<=int(n[-4:])<=2029 or 5301<=int(n[-4:])<=5305 for n in names)
inputhash={Path(r['path']).stem.removesuffix('_images'):r['sha256'] for r in read(PRE/'input_audit.json')['files']}
def convert(a,variant):
    u=np.clip(a/a.max(),0,1)
    if variant=='sqrt_brightened':v=np.rint(255*np.sqrt(u)).astype('uint8')
    else:
        v=np.rint(255*u).astype('uint8')
        if variant=='inverted':v=255-v
    return Image.fromarray(v).convert('RGB')
sheet=Image.new('RGB',(6*192,3*214),'#444444');draw=ImageDraw.Draw(sheet)
a=np.load(CACHE/'moderate_seed2026_images.npy',mmap_mode='r')
for row,variant in enumerate(cfg['variants']):
    for col,index in enumerate(ix[:6]):
        im=convert(a[index],variant).resize((192,192),Image.Resampling.BICUBIC);sheet.paste(im,(col*192,row*214));draw.text((col*192+2,row*214+193),f'{variant} id={original_y[index]}',fill='white')
sheet.save(OUT/'input_preview.png')
start=time.perf_counter();model,preprocess=clip.load(str(weight),device='cpu',jit=False);model.eval();model.requires_grad_(False)
bank={};cachelog=[]
with torch.inference_mode():
    for variant in cfg['variants']:
        bank[variant]={}
        for name in names:
            path=CACHE/f'{name}_images.npy';assert sha(path)==inputhash[name]
            if variant=='original':
                p=OLD/'features'/f'{name}.npy';assert sha(p)==read(p.with_suffix('.json'))['feature_sha256'];bank[variant][name]=norm(np.load(p)[ix]);continue
            dest=OUT/'features'/f'{variant}_{name}.npy';info=dest.with_suffix('.json');dest.parent.mkdir(exist_ok=True)
            if dest.exists() and info.exists():
                entry=read(info);assert sha(dest)==entry['sha256'];bank[variant][name]=norm(np.load(dest));cachelog.append(entry);continue
            a=np.load(path,mmap_mode='r');chunks=[];t=time.perf_counter()
            for i in range(0,len(ix),8):
                z=model.encode_image(torch.stack([preprocess(convert(a[j],variant)) for j in ix[i:i+8]]));z=z/z.norm(dim=1,keepdim=True);assert torch.isfinite(z).all();chunks.append(z.numpy())
            z=np.concatenate(chunks);assert z.shape==(360,1024)
            tmp=dest.with_suffix('.tmp.npy');np.save(tmp,z);os.replace(tmp,dest)
            entry=dict(variant=variant,mapping=name,sha256=sha(dest),input_sha256=inputhash[name],seconds=time.perf_counter()-t,n=360)
            save(str(info.relative_to(OUT)),entry);cachelog.append(entry);bank[variant][name]=norm(z)
            progress=dict(stage='encoding',completed_new_groups=len(cachelog),total_new_groups=20,variant=variant,mapping=name,elapsed_seconds=time.perf_counter()-start)
            save('progress.json',progress);print(json.dumps(progress),flush=True)
del model
text=norm(np.load(PRE/'text_features.npy')[classids]);scores={};records={};foldaudit=[]
for panel,mapnames in [('selection',cfg['selection_maps']),('confirmation',cfg['confirmation_maps'])]:
    rec=[]
    for fi,f in enumerate(np.unique(fonts)):
        va=fonts==f;tr=~va;assert not set(samples[va])&set(samples[tr])
        if panel=='selection':foldaudit.append(dict(fold=fi,held_out_font=str(f),ideal_support=int(tr.sum()),multi_support=5*int(tr.sum())))
        for name in mapnames:
            for j in np.flatnonzero(va):rec.append(dict(font=str(f),mapping=name,sample=str(samples[j]),truth=int(y[j]),original_class=int(original_y[ix[j]])))
    records[panel]=rec
for variant in cfg['variants']:
    for panel,mapnames in [('selection',cfg['selection_maps']),('confirmation',cfg['confirmation_maps'])]:
        lists={k:[] for k in ['text','vis0','vis1','plain0','plain1']}
        for fi,f in enumerate(np.unique(fonts)):
            va=fonts==f;tr=~va;q=np.concatenate([bank[variant][n][va] for n in mapnames]);lists['text'].append(q@text.T)
            for multi in [0,1]:
                maps=cfg['train_maps'] if multi else cfg['train_maps'][:1];sx=np.concatenate([bank[variant][n][tr] for n in maps]);sy=np.tile(y[tr],len(maps));p=build(sx,sy,c=60);u=build(sx,sy,c=60,column=False)
                # Independent mean/axis audit against reused original function.
                ref=np.zeros((60,1024));np.add.at(ref,sy,norm(sx));ref/=np.bincount(sy)[:,None];d=np.sqrt((ref*ref).sum(0));ref[:,d>0]/=d[d>0];ref[:,d==0]=0;ref=norm(ref);assert np.allclose(ref,p,atol=1e-12)
                lists[f'vis{multi}'].append(q@p.T);lists[f'plain{multi}'].append(q@u.T)
        scores[variant,panel]={k:np.concatenate(v) for k,v in lists.items()}
selection=[];trials=[]
for variant in cfg['variants']:
    s=scores[variant,'selection'];truth=np.array([r['truth'] for r in records['selection']])
    for multi in [0,1]:
        best,rows,expanded=choose(s[f'vis{multi}'],s['text'],truth,oldcfg['alpha_grid']);selection.append(dict(variant=variant,method=f'PA{multi}',**best,expanded=expanded))
        trials.extend(dict(variant=variant,method=f'PA{multi}',**r) for r in rows)
winner=max([r for r in selection if r['method']=='PA1'],key=lambda r:(r['correct'],-cfg['variants'].index(r['variant'])))
save('selection_decision.json',dict(selected_variant=winner['variant'],alphas=selection,rule=cfg['selection'],confirmation_accuracy_not_computed_yet=True))
save('selection_search.json',trials);metrics=[];preds=[]
for variant in cfg['variants']:
    for panel in ['selection','confirmation']:
        s=scores[variant,panel];truth=np.array([r['truth'] for r in records[panel]]);sc={'TEXT':s['text'],'VIS0':s['vis0'],'VIS1':s['vis1'],'PLAIN0':s['plain0'],'PLAIN1':s['plain1']}
        for multi in [0,1]:
            alpha=next(r['alpha'] for r in selection if r['variant']==variant and r['method']==f'PA{multi}');sc[f'PA{multi}']=alpha*s[f'vis{multi}']+s['text']
        np.savez_compressed(OUT/f'{variant}_{panel}_scores.npz',truth=truth,**sc)
        for method,z in sc.items():
            p=z.argmax(1);correct=int((p==truth).sum());metrics.append(dict(variant=variant,panel=panel,method=method,n=len(p),correct=correct,accuracy_percent=100*correct/len(p)))
            for rec,pred in zip(records[panel],p):preds.append(dict(variant=variant,panel=panel,method=method,**rec,predicted=int(pred)))
with (OUT/'predictions.csv').open('w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(preds[0]));w.writeheader();w.writerows(preds)
with (OUT/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
def acc(v,p):return next(r['accuracy_percent'] for r in metrics if r['variant']==v and r['panel']==p and r['method']=='PA1')
wv=winner['variant'];ds=acc(wv,'selection')-acc('original','selection');dc=acc(wv,'confirmation')-acc('original','confirmation')
gate=wv!='original' and ds>=5 and dc>=5 and acc(wv,'confirmation')>=20
save('result.json',dict(status='COMPLETE',selected_variant=wv,selection_gain_pp=ds,confirmation_gain_pp=dc,recommend_full_180_DEV_verification=bool(gate),metrics=metrics,fold_audit=foldaudit,
    prototype_independent_math='PASS',external_panels_accessed=False,cache_groups=cachelog,seconds_this_process=time.perf_counter()-start,interpretation='post-hoc 60-way screen; historical development confirmation, not fresh test or 180-way result'))
save('progress.json',dict(stage='COMPLETE'));print(json.dumps(read(OUT/'result.json')),flush=True)
