"""Fixed 20-way natural-image functionality check with shared SPV scoring code."""
from pathlib import Path
import os,sys,json,hashlib,zipfile,tarfile,io,ast,re,csv,time
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
import numpy as np
import torch,clip
from PIL import Image,ImageDraw
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];PRE=OUT.parent/'proto_adapter_full_preflight_20260929';OLD=OUT.parent/'proto_adapter_full_run_20260929'
sys.stdout.reconfigure(encoding='utf-8');torch.set_num_threads(2);torch.set_num_interop_threads(1)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
cfg=read(OUT/'protocol.json');source=OLD/'evaluate.py';assert sha(source)==read(OLD/'evaluation_manifest.json')['script_sha256']
# Execute exactly the existing mathematical functions without importing its runner/dependencies.
tree=ast.parse(source.read_text(encoding='utf-8'));nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['norm','build','choose']]
assert len(nodes)==3
ns={'np':np};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns)
norm,build,choose=[ns[k] for k in ['norm','build','choose']]
assert sha(ROOT/'work/proto_adapter_clip_weights/RN50.pt')==read(PRE/'frozen_protocol.json')['weight_sha256']
archive=ROOT/'work/caltech101_official/caltech-101.zip';assert hashlib.md5(archive.read_bytes()).hexdigest()==cfg['archive_md5']
with zipfile.ZipFile(archive) as z:
    inner=next(n for n in z.namelist() if Path(n).name=='101_ObjectCategories.tar.gz')
    tarbytes=z.read(inner)
tar=tarfile.open(fileobj=io.BytesIO(tarbytes),mode='r:gz')
members=[m for m in tar.getmembers() if m.isfile() and m.name.lower().endswith('.jpg') and not Path(m.name).name.startswith('._')]
categories=sorted(set(Path(m.name).parent.name for m in members)-{'BACKGROUND_Google','Faces'})
groups={c:sorted([m for m in members if Path(m.name).parent.name==c],key=lambda m:m.name) for c in categories}
eligible=[c for c in categories if len(groups[c])>=28];rng=np.random.default_rng(cfg['seed']);chosen=sorted(rng.choice(eligible,20,replace=False).tolist())
selected={c:[groups[c][int(index)] for index in rng.permutation(len(groups[c]))[:28]] for c in chosen}
# Read gzip members in archive order to avoid repeated decompression on random seeks.
rawcache={m.name:tar.extractfile(m).read() for m in sorted([m for c in chosen for m in selected[c]],key=lambda m:m.offset_data)}
records=[];images=[];rawhashes=set();pixelhashes=set()
for label,c in enumerate(chosen):
    for j,member in enumerate(selected[c]):
        raw=rawcache[member.name];im=Image.open(io.BytesIO(raw)).convert('RGB')
        rawhash=hashlib.sha256(raw).hexdigest();ph=hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()
        assert rawhash not in rawhashes and ph not in pixelhashes,'duplicate selected image: stop before inference'
        rawhashes.add(rawhash);pixelhashes.add(ph)
        records.append(dict(index=len(images),class_id=label,category=c,split='support' if j<8 else 'validation' if j<12 else 'test',member=member.name,sha256=rawhash,pixel_sha256=ph))
        images.append(im)
tar.close();del tarbytes,rawcache
save('split_manifest.json',dict(classes=chosen,eligible_classes=eligible,records=records,seed=cfg['seed'],disjoint_paths_and_content=True))
freeze=dict(script_sha256=sha(__file__),protocol_sha256=sha(OUT/'protocol.json'),source_scoring_sha256=sha(source),split_manifest_sha256=sha(OUT/'split_manifest.json'),archive_sha256=sha(archive),shared_functions=['norm','build','choose'])
if (OUT/'implementation_manifest.json').exists():assert read(OUT/'implementation_manifest.json')==freeze
else:save('implementation_manifest.json',freeze)
sheet=Image.new('RGB',(5*224,4*248),'#eeeeee');draw=ImageDraw.Draw(sheet)
for i,c in enumerate(chosen):
    preview=images[i*28].copy();preview.thumbnail((224,224));xx=(i%5)*224;yy=(i//5)*248;sheet.paste(preview,(xx,yy));draw.text((xx+3,yy+226),c,fill='black')
sheet.save(OUT/'support_preview.png')
model,preprocess=clip.load(str(ROOT/'work/proto_adapter_clip_weights/RN50.pt'),device='cpu',jit=False);model.eval();model.requires_grad_(False)
start=time.perf_counter();chunks=[]
with torch.inference_mode():
    for i in range(0,len(images),8):
        v=model.encode_image(torch.stack([preprocess(im) for im in images[i:i+8]]));v=v/v.norm(dim=1,keepdim=True)
        chunks.append(v.numpy())
        if i%128==0:print(f'IMAGES {i}/{len(images)}',flush=True)
    x=norm(np.concatenate(chunks));assert np.isfinite(x).all()
    rename={'Faces_easy':'face','Leopards':'leopard','Motorbikes':'motorbike'}
    classnames=[rename.get(c,re.sub(r'_\d+$','',c).replace('_',' ')) for c in chosen]
    prompts=[t.format(c) for t in cfg['templates'] for c in classnames]
    tokens=clip.tokenize(prompts,truncate=False);ts=[]
    for i in range(0,len(tokens),8):
        z=model.encode_text(tokens[i:i+8]);z=z/z.norm(dim=1,keepdim=True);ts.append(z.numpy())
text=norm(np.concatenate(ts).reshape(7,20,-1).mean(0));np.save(OUT/'image_features.npy',x);np.save(OUT/'text_features.npy',text)
save('prompts.json',dict(classnames=classnames,prompts=prompts));y=np.array([r['class_id'] for r in records]);split=np.array([r['split'] for r in records]);support=split=='support';val=split=='validation';test=split=='test'
p=build(x[support],y[support],c=20);plain=build(x[support],y[support],c=20,column=False)
# Independently construct class sums and normalize axes.
ref=np.zeros((20,1024));np.add.at(ref,y[support],norm(x[support]));ref/=8
cn=np.sqrt(np.einsum('ij,ij->j',ref,ref));ref[:,cn>0]/=cn[cn>0];ref[:,cn==0]=0;ref=ref/np.sqrt(np.einsum('ij,ij->i',ref,ref))[:,None]
assert np.allclose(ref,p,atol=1e-12)
np.save(OUT/'pa_prototypes.npy',p);np.save(OUT/'plain_prototypes.npy',plain)
grid=read(PRE/'frozen_protocol.json')['alpha_grid'];best,trials,expanded=choose(x[val]@p.T,x[val]@text.T,y[val],grid)
save('validation_search.json',trials);save('selected_alpha.json',dict(**best,boundary_expanded=expanded,selection='validation only',test_scoring_not_started=True))
metrics=[];predrows=[]
for splitname,mask in [('support',support),('validation',val),('test',test)]:
    q=x[mask];truth=y[mask];scores={'TEXT':q@text.T,'PLAIN':q@plain.T,'PA_VISUAL':q@p.T,'PA_FULL':best['alpha']*q@p.T+q@text.T}
    for method,s in scores.items():
        pred=s.argmax(1);correct=int((pred==truth).sum());metrics.append(dict(split=splitname,method=method,n=len(pred),correct=correct,accuracy_percent=100*correct/len(pred)))
        for ix,pr in zip(np.flatnonzero(mask),pred):predrows.append(dict(index=int(ix),split=splitname,method=method,truth=int(y[ix]),predicted=int(pr)))
with (OUT/'predictions.csv').open('w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(predrows[0]));w.writeheader();w.writerows(predrows)
# Recheck every prediction with independent scalar class scoring; includes held-out test.
for r in predrows:
    q=x[r['index']];mat=text if r['method']=='TEXT' else plain if r['method']=='PLAIN' else ref
    s=np.array([np.sum(q*row) for row in mat])
    if r['method']=='PA_FULL':s=best['alpha']*s+np.array([np.sum(q*row) for row in text])
    assert int(s.argmax())==r['predicted']
testmetrics={r['method']:r['accuracy_percent'] for r in metrics if r['split']=='test'}
save('result.json',dict(status='COMPLETE',engineering_screen='PASS' if all(testmetrics[k]>=50 for k in ['PA_FULL','PA_VISUAL','PLAIN']) else 'INVESTIGATE',metrics=metrics,selected_alpha=best['alpha'],encoding_seconds=time.perf_counter()-start,
    correctness_audit=dict(status='PASS',independent_predictions=len(predrows),prototype_max_abs_error=float(abs(ref-p).max()),sample_splits_disjoint=True),
    limitation='One small fixed split, no stochastic augmentation; not author benchmark replication; passing does not validate SPV adaptation'))
print(json.dumps(read(OUT/'result.json')),flush=True)
