"""DEV-only parameter selection then locked historical E1/C1 scoring."""
from pathlib import Path
import os,sys,json,csv,hashlib,time
from datetime import datetime,timezone
import numpy as np
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent
PRE=ROOT/'outputs/proto_adapter_full_preflight_20260929'; OLD=ROOT/'outputs/proto_adapter_B_run_20260928'
CACHE=ROOT/'work/mma_supplement_cache'; EXT=ROOT/'outputs/randumb_cpu_20260928/external'
METHODS=['H0','H1','PA0','PA1','TEXT','VIS0','VIS1','PLAIN0','PLAIN1']
PAIRS=[('H1','PA1'),('H1','H0'),('PA1','PA0'),('H1','PA0'),('PA1','VIS1')]
sys.stdout.reconfigure(encoding='utf-8')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def dump(n,v):
    p=OUT/n; p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix('.tmp'); t.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8'); os.replace(t,p)
def table(n,rows):
    with (OUT/n).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def norm(x):
    x=np.asarray(x,dtype=np.float64); d=np.linalg.norm(x,axis=1,keepdims=True)
    assert np.isfinite(x).all() and (d>0).all();return x/d
def build(x,y,c=180,column=True):
    x=norm(x); p=np.stack([x[y==i].mean(0) for i in range(c)])
    if column:
        d=np.linalg.norm(p,axis=0,keepdims=True);p=np.divide(p,d,out=np.zeros_like(p),where=d>0)
    return norm(p)
def choose(v,t,y,grid):
    rows=[dict(alpha=float(a),correct=int(np.sum(np.argmax(a*v+t,axis=1)==y)),n=len(y)) for a in sorted(grid)]
    best=max(rows,key=lambda r:(r['correct'],-r['alpha']))
    positives=sorted(a for a in grid if a>0);extra=[]
    if best['alpha']==positives[0]:extra=[10**(np.log10(positives[0])-.25*i) for i in range(1,17)]
    elif best['alpha']==positives[-1]:extra=[10**(np.log10(positives[-1])+.25*i) for i in range(1,17)]
    for a in extra:rows.append(dict(alpha=float(a),correct=int(np.sum(np.argmax(a*v+t,axis=1)==y)),n=len(y)))
    best=max(rows,key=lambda r:(r['correct'],-r['alpha']))
    return best,sorted(rows,key=lambda r:r['alpha']),bool(extra)
def contracts():
    rng=np.random.default_rng(129);x=rng.normal(size=(18,7)); x[:,-1]=0;y=np.repeat(np.arange(3),6)
    p=build(x,y,3); rows=x/np.sqrt((x*x).sum(1))[:,None]
    ref=np.array([rows[y==c].sum(0)/6 for c in range(3)])
    for j in range(7):
        d=sum(ref[c,j]**2 for c in range(3))**.5
        if d:ref[:,j]/=d
    for c in range(3):ref[c]/=sum(ref[c,j]**2 for j in range(7))**.5
    assert np.allclose(p,ref,atol=1e-12) and np.all(p[:,-1]==0)
    assert np.allclose(build(x,2-y,3),p[::-1])
    # Flat score plateau: deterministic smallest-alpha tie handling, no external data.
    v=np.array([[2.,0.],[0.,2.]]); t=v.copy(); labels=np.array([0,1])
    selected,_,extended=choose(v,t,labels,[0,.1,1,10]); assert selected['alpha']==0 and not extended
    # Boundary expansion is defined before scoring, and occurs once only.
    selected,tr,extended=choose(v,-v,labels,[0,.1,1,10]); assert extended and len(tr)==20
    return dict(status='PASS',manual_prototype_error=float(abs(p-ref).max()),column_axis=True,class_permutation=True,zero_column=True,selection_ties=True,edge_expansion=True)

def setup(require_features):
    protocol=read(PRE/'frozen_protocol.json'); assert sha(PRE/'frozen_protocol.json')==read(PRE/'readiness.json')['frozen_protocol_sha256']
    for n,h in protocol['evidence_hashes'].items():assert sha(PRE/n)==h,n
    old_manifest={d['path']:d['sha256'] for d in read(OLD/'delivery_manifest.json')['files']}
    for n in ['predictions.csv','independent_audit.json','baseline_replay.json']: assert sha(OLD/n)==old_manifest[n]
    assert read(OLD/'independent_audit.json')['status']=='PASS'
    oldrows=list(csv.DictReader((OLD/'predictions.csv').open(encoding='utf-8-sig')))
    lookup={}
    for r in oldrows:
        if r['method'] in ['H0','H1']:
            key=(r['panel'],r['method'],r['font'],int(r['mapping_seed']),int(r['class_id']))
            assert key not in lookup;lookup[key]=(int(r['predicted']),r['sample_id'])
    assert len(lookup)==21600
    for entry in read(OLD/'input_audit.json')['files']:assert sha(entry['path'])==entry['sha256'],entry['path']
    meta=dict(np.load(CACHE/'metadata.npz',allow_pickle=False)); y=meta['labels'];fs=meta['fonts']
    classes=read(PRE/'classes.json')
    for c in range(180): assert set(meta['characters'][y==c])=={classes[c]['character']}
    for f in np.unique(fs): assert np.array_equal(np.sort(y[fs==f]),np.arange(180))
    manifest=dict(protocol_sha256=sha(PRE/'frozen_protocol.json'),script_sha256=sha(__file__),old_prediction_sha256=sha(OLD/'predictions.csv'),frozen_before_external_scoring=True)
    if (OUT/'evaluation_manifest.json').exists(): assert read(OUT/'evaluation_manifest.json')==manifest
    else:dump('evaluation_manifest.json',manifest)
    dump('implementation_audit.json',contracts())
    if require_features:
        audit=read(OUT/'feature_audit.json'); assert audit['status']=='PASS' and audit['total_images']==16200
        for entry in audit['files']:assert sha(OUT/'features'/f"{entry['tag']}.npy")==entry['feature_sha256']
    dump('evaluation_input_audit.json',dict(status='PASS',baseline_predictions_verified=21600,input_metadata_and_old_HOG_hashes_verified=True,label_character_order=True,features_required=require_features))
    return protocol,meta,lookup

def baseline(panel,d,lookup):
    result={}
    for m in ['H0','H1']:
        vals=[]
        for f,s,c,sample in zip(d['fonts'],d['mapping_seeds'],d['truth'],d['samples']):
            p,oldsample=lookup[(panel,m,str(f),int(s),int(c))]; assert str(sample)==oldsample; vals.append(p)
        result[m]=np.array(vals)
    return result
def feature(n):return norm(np.load(OUT/'features'/f'{n}.npy',mmap_mode='r'))
def support(bank,y,mask,names):return np.concatenate([bank[n][mask] for n in names]),np.tile(y[mask],len(names))
def saveproto(tag,p):
    path=OUT/'prototypes'/f'{tag}.npy';path.parent.mkdir(exist_ok=True);np.save(path,p); assert np.array_equal(p,np.load(path))

def execute(protocol,meta,lookup):
    y,fonts=meta['labels'],meta['fonts']; names=protocol['train_maps'];devnames=protocol['dev_maps']
    bank={n:feature(n) for n in names+devnames};text=norm(np.load(PRE/'text_features.npy'))
    records={k:[] for k in ['truth','fonts','mapping_seeds','samples']};vs={0:[],1:[]};plain={0:[],1:[]};ts=[]; folds=[]
    for fi,f in enumerate(np.unique(fonts)):
        tr=fonts!=f;va=fonts==f
        assert not set(meta['sample_ids'][tr]) & set(meta['sample_ids'][va])
        qs=np.concatenate([bank[n][va] for n in devnames]);ts.append(qs@text.T)
        for n in devnames:
            records['truth'].extend(y[va]);records['fonts'].extend(fonts[va]);records['samples'].extend(meta['sample_ids'][va]);records['mapping_seeds'].extend([int(n[-4:])]*int(va.sum()))
        for multi in [0,1]:
            sx,sy=support(bank,y,tr,names if multi else names[:1]);p=build(sx,sy);u=build(sx,sy,column=False)
            saveproto(f'fold{fi}_PA{multi}',p);saveproto(f'fold{fi}_PLAIN{multi}',u)
            vs[multi].append(qs@p.T);plain[multi].append(np.argmax(qs@u.T,axis=1))
        folds.append(dict(fold=fi,held_out_font=str(f),support_fonts=list(map(str,np.unique(fonts[tr]))),ideal_n=int(tr.sum()),multi_n=int(tr.sum())*5,query_n=len(qs)))
    d={k:np.array(v) for k,v in records.items()};t=np.concatenate(ts);pred=baseline('DEV',d,lookup);pred['TEXT']=t.argmax(1)
    selections={};search=[]
    for multi in [0,1]:
        v=np.concatenate(vs[multi]);best,trials,expanded=choose(v,t,d['truth'],protocol['alpha_grid']); selections[f'PA{multi}']={**best,'boundary_expanded':expanded}
        for trial in trials:search.append(dict(method=f'PA{multi}',**trial))
        pred[f'PA{multi}']=(best['alpha']*v+t).argmax(1);pred[f'VIS{multi}']=v.argmax(1);pred[f'PLAIN{multi}']=np.concatenate(plain[multi])
        np.savez_compressed(OUT/f'dev_scores_PA{multi}.npz',visual=v,text=t,truth=d['truth'])
    table('development_search.csv',search); dump('fold_audit.json',dict(status='PASS',folds=folds))
    # Write selected values before reading external feature/metadata panels below.
    selection=dict(status='FROZEN_DEV_ONLY',utc=datetime.now(timezone.utc).isoformat(),methods=selections,protocol_sha256=sha(PRE/'frozen_protocol.json'),search_sha256=sha(OUT/'development_search.csv'))
    if (OUT/'selected_alpha.json').exists():
        prior=read(OUT/'selected_alpha.json'); assert prior['methods']==selection['methods'] and prior['search_sha256']==selection['search_sha256']
    else:dump('selected_alpha.json',selection)
    panels={'DEV':(d,pred)};pfull={};ufull={}
    for multi in [0,1]:
        sx,sy=support(bank,y,np.ones(len(y),bool),names if multi else names[:1]);pfull[multi]=build(sx,sy);ufull[multi]=build(sx,sy,column=False)
        saveproto(f'full_PA{multi}',pfull[multi]);saveproto(f'full_PLAIN{multi}',ufull[multi])
    for panel,n in [('E1',3600),('C1',1800)]:
        em=dict(np.load(EXT/f'{panel}_metadata.npz',allow_pickle=False)); q=feature(panel)
        assert len(q)==n and len(set(zip(em['fonts'],em['mapping_seeds'],em['truth'])))==n
        preds=baseline(panel,em,lookup); t=q@text.T;preds['TEXT']=t.argmax(1)
        for multi in [0,1]:
            v=q@pfull[multi].T;preds[f'PA{multi}']=(selections[f'PA{multi}']['alpha']*v+t).argmax(1)
            preds[f'VIS{multi}']=v.argmax(1);preds[f'PLAIN{multi}']=(q@ufull[multi].T).argmax(1)
        panels[panel]=(em,preds)
    return panels

def summarize(panels,protocol):
    rows=[];metrics=[];strata=[];comparisons={}
    for panel,(d,pred) in panels.items():
        y=d['truth'];ca=[]
        for m in METHODS:
            p=pred[m];assert p.shape==y.shape and ((p>=0)&(p<180)).all()
            metrics.append(dict(panel=panel,method=m,n=len(y),correct=int((p==y).sum()),accuracy_percent=float(100*np.mean(p==y)),macro_f1_pooled_percent=float(100*f1_score(y,p,labels=np.arange(180),average='macro',zero_division=0))))
            ca.append([np.mean(p[y==c]==c) for c in range(180)])
            for i in range(len(y)):rows.append(dict(panel=panel,method=m,font=str(d['fonts'][i]),mapping_seed=int(d['mapping_seeds'][i]),sample_id=str(d['samples'][i]),class_id=int(y[i]),predicted=int(p[i]),correct=int(y[i]==p[i])))
            for axis in ['fonts','mapping_seeds']:
                for value in np.unique(d[axis]):
                    mask=d[axis]==value;strata.append(dict(panel=panel,method=m,axis=axis,value=str(value),n=int(mask.sum()),accuracy_percent=float(100*np.mean(p[mask]==y[mask]))))
            for f in np.unique(d['fonts']):
                for s in np.unique(d['mapping_seeds']):
                    mask=(d['fonts']==f)&(d['mapping_seeds']==s)
                    strata.append(dict(panel=panel,method=m,axis='font_x_mapping',value=f'{f}|{s}',n=int(mask.sum()),accuracy_percent=float(100*np.mean(p[mask]==y[mask]))))
        ca=np.asarray(ca).T; diff=np.stack([ca[:,METHODS.index(a)]-ca[:,METHODS.index(b)] for a,b in PAIRS],axis=1)*100
        if panel!='DEV':
            rng=np.random.default_rng(protocol['bootstrap'][panel+'_seed']);draws=np.empty((100000,len(PAIRS)))
            for i in range(0,100000,2000):draws[i:i+2000]=rng.multinomial(180,np.full(180,1/180),size=2000)@diff/180
            ci=np.quantile(draws,[.025,.975],axis=0).T;np.savez_compressed(OUT/f'{panel}_bootstrap.npz',paired_deltas=draws,character_accuracy=ca)
        comparisons[panel]=[dict(comparison=a+'-'+b,delta_pp=float(diff[:,j].mean()),ci95_pp=ci[j].tolist() if panel!='DEV' else None,
            role='primary' if panel=='E1' and j==0 else 'secondary_or_diagnostic',note='DEV is selection performance, not unbiased validation of tuned PA' if panel=='DEV' else 'Character-cluster interval conditional on fixed fonts/mappings') for j,(a,b) in enumerate(PAIRS)]
    table('predictions.csv',rows);table('metrics.csv',metrics);table('stratified_metrics.csv',strata);dump('paired_comparisons.json',comparisons)
    assert len(rows)==97200 and len({(r['panel'],r['method'],r['font'],r['mapping_seed'],r['class_id']) for r in rows})==97200
    dump('completion_audit.json',dict(status='PASS',prediction_rows=len(rows),methods=METHODS,panels=list(panels),prototype_matrices=len(list((OUT/'prototypes').glob('*.npy'))),baseline_predictions_aligned=21600,dev_selection_saved_before_external_scoring=True,negative_results_retained=True))
    print(json.dumps(dict(metrics=metrics,comparisons=comparisons),ensure_ascii=False),flush=True)
def main():
    start=time.perf_counter()
    with threadpool_limits(limits=2):
        protocol,meta,lookup=setup('--preflight-only' not in sys.argv)
        if '--preflight-only' in sys.argv:print('SCORING PREFLIGHT PASS; NO NEW ACCURACY');return
        panels=execute(protocol,meta,lookup);summarize(panels,protocol)
    dump('evaluation_resources.json',dict(seconds=time.perf_counter()-start,blas_threads=2,python=sys.version,numpy=np.__version__))
if __name__=='__main__':main()
