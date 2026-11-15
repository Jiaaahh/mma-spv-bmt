"""Frozen six-configuration CPU experiment; no selection or tuning."""
from pathlib import Path
import os, sys, json, hashlib, time, csv, platform, threading
from datetime import datetime, timezone
import numpy as np
import psutil, sklearn
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits, threadpool_info

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CACHE = ROOT/'work/mma_supplement_cache'
EXT = ROOT/'outputs/randumb_cpu_20260928/external'
DESIGN = ROOT/'outputs/proto_adapter_B_design_20260928'
METHODS = ['H0','H1','U0','U1','P0','P1']
MAPS = ['ideal']+[f'moderate_seed{s}' for s in range(2026,2030)]
PAIRS = [['H1','P1'],['H1','H0'],['P1','P0'],['U1','U0'],['P1','U1'],['U1','H1']]
START=time.perf_counter()
RESOURCE={'peak_rss_bytes':0,'samples':0,'cpu_only':True,'blas_threads':2}
STOP=threading.Event()

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def dump(name,data):
    p=OUT/name; p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    os.replace(tmp,p)

def monitor():
    while not STOP.wait(.1):
        rss=psutil.Process().memory_info().rss
        RESOURCE['peak_rss_bytes']=max(RESOURCE['peak_rss_bytes'],rss)
        RESOURCE['samples']+=1
        if rss>2*1024**3:
            dump('resource_limit_abort.json',RESOURCE)
            os._exit(75)

def norm(x):
    x=np.asarray(x,dtype=np.float64)
    assert np.isfinite(x).all(), 'nonfinite input'
    n=np.linalg.norm(x,axis=1,keepdims=True)
    assert (n>0).all(), 'zero row'
    return x/n

def colrow(m):
    a=np.linalg.norm(m,axis=0)
    q=np.divide(m,a,out=np.zeros_like(m),where=a>0)
    return norm(q), int(np.sum(a==0))

def fit(bank,labels,mask,method):
    names=MAPS if method.endswith('1') else MAPS[:1]
    # Equal map means equals pooled class means on the verified balanced grid.
    means=[]
    for name in names:
        x=np.asarray(bank[name][mask],dtype=np.float64)
        assert np.isfinite(x).all() and (np.linalg.norm(x,axis=1)>0).all()
        if method[0]!='H': x=norm(x)
        means.append(np.stack([x[labels[mask]==c].mean(0) for c in range(180)]))
    m=np.mean(means,axis=0)
    p,z=colrow(m) if method[0]=='P' else (norm(m),int(np.sum(np.linalg.norm(m,axis=0)==0)))
    assert p.shape==(180,4356)
    return p,z

def predict(x,p):
    preds=[]; ties=0
    for start in range(0,len(x),256):
        scores=norm(x[start:start+256])@p.T
        preds.append(np.argmax(scores,axis=1))
        ties+=int(np.sum(np.sum(scores==scores.max(1,keepdims=True),axis=1)>1))
    return np.concatenate(preds),ties

def contracts():
    m=np.array([[3.,4.,0.],[4.,3.,0.]])
    p,z=colrow(m)
    np.testing.assert_allclose(p,[[.6,.8,0],[.8,.6,0]],rtol=0,atol=1e-15)
    m=np.array([[1.,4.,0.],[2.,1.,0.]])
    q=np.array([[1/np.sqrt(5),4/np.sqrt(17),0],[2/np.sqrt(5),1/np.sqrt(17),0]])
    p,z=colrow(m)
    np.testing.assert_allclose(p,q/np.sqrt((q*q).sum(1))[:,None])
    np.testing.assert_allclose(colrow(m[::-1])[0],p[::-1])
    assert z==1 and np.all(p[:,2]==0) and not np.allclose(p,norm(m))
    x=np.array([[10.,0.],[0.,1.]])
    assert not np.allclose(norm(x.mean(0)[None]),norm(norm(x).mean(0)[None]))
    try: norm(np.zeros((1,3)))
    except AssertionError: pass
    else: raise AssertionError('zero row not rejected')
    query=np.array([[1.,2.,0.]])
    assert predict(query,p)[0][0]==np.argmax((query/np.sqrt(5))@p.T)
    return ['manual column then row axes','zero columns','class permutation','P differs from U','sample norm control','zero row rejected','query normalized once']

def load_inputs():
    audit=[]
    for item in json.loads((DESIGN/'resource_inventory.json').read_text(encoding='utf-8'))['files']:
        actual=sha(item['path']); assert actual==item['sha256'],item['path']
        audit.append({'path':item['path'],'sha256':actual,'match':True})
    meta=dict(np.load(CACHE/'metadata.npz',allow_pickle=False))
    labels,fonts=meta['labels'],meta['fonts']
    assert len(labels)==1080 and len(np.unique(fonts))==6
    assert len(set(zip(fonts.tolist(),labels.tolist())))==1080
    for f in np.unique(fonts): assert np.array_equal(np.sort(labels[fonts==f]),np.arange(180))
    assert len(np.unique(meta['sample_ids']))==1080
    for c in range(180): assert len(np.unique(meta['characters'][labels==c]))==1
    bank={n:np.load(CACHE/(n+'_hog.npy'),mmap_mode='r') for n in MAPS+[f'moderate_seed{s}' for s in range(5301,5306)]}
    for n,x in bank.items(): assert x.shape==(1080,4356) and x.dtype==np.float32 and np.isfinite(x).all()
    ex={}
    for panel,n,nf,seeds in [('E1',3600,4,range(3101,3106)),('C1',1800,2,range(2031,2036))]:
        d=dict(np.load(EXT/(panel+'_metadata.npz'),allow_pickle=False)); x=np.load(EXT/(panel+'_hog.npy'),mmap_mode='r')
        assert x.shape==(n,4356) and x.dtype==np.float64 and np.isfinite(x).all()
        assert len(np.unique(d['fonts']))==nf and set(d['mapping_seeds'])==set(seeds)
        assert len(set(zip(d['fonts'].tolist(),d['mapping_seeds'].tolist(),d['truth'].tolist())))==n
        for f in np.unique(d['fonts']):
            for s in seeds: assert np.array_equal(np.sort(d['truth'][(d['fonts']==f)&(d['mapping_seeds']==s)]),np.arange(180))
        ex[panel]=(d,x)
    dump('input_audit.json',{'status':'PASS','files':audit,'balanced_complete_grids':True,'contracts':contracts()})
    return meta,bank,ex

def baseline(meta,bank,ex):
    oldpath=ROOT/'outputs/mma_supplement_20260908/results/hog/predictions.npz'
    old=np.load(oldpath); labels=meta['labels']; fonts=meta['fonts']; rows=[]
    for fi,f in enumerate(np.unique(fonts)):
        for method,key in [('H0','ideal_only'),('H1','mma_g0')]:
            p,_=fit(bank,labels,fonts!=f,method)
            for mi,s in enumerate(range(5301,5306)):
                va=fonts==f; pred,_=predict(bank[f'moderate_seed{s}'][va],p)
                mismatch=int(np.sum(pred!=old[key][fi,mi,labels[va]]))
                rows.append({'panel':'DEV','font':str(f),'method':method,'mapping':s,'n':len(pred),'mismatches':mismatch})
    for method in ['H0','H1']:
        p,_=fit(bank,labels,np.ones(len(labels),bool),method)
        for panel,(d,x) in ex.items():
            pred,_=predict(x,p)
            rows.append({'panel':panel,'method':method,'n':len(pred),'mismatches':int(np.sum(pred!=d[method]))})
    result={'status':'PASS' if sum(r['mismatches'] for r in rows)==0 else 'FAIL','checked_predictions':sum(r['n'] for r in rows),'old_dev_prediction_sha256':sha(oldpath),'details':rows}
    dump('baseline_replay.json',result)
    assert result['status']=='PASS','baseline mismatch; stop before new results'
    print('BASELINE_REPLAY PASS',result['checked_predictions'],flush=True)

def freeze():
    protocol={'identity':'Proto-Adapter inspired HOG adaptation, not full reproduction','methods':METHODS,'train_maps':MAPS,'primary':'E1 Accuracy(H1)-Accuracy(P1)','pairs':PAIRS,'query':'row L2 only; no column scaling','P':'sample L2 -> class mean -> column L2 -> row L2','U':'sample L2 -> class mean -> row L2','H':'raw class mean -> row L2','bootstrap_draws':100000,'bootstrap_seeds':{'DEV':26092831,'E1':26092832,'C1':26092833},'bootstrap_unit':'character cluster; paired multinomial weights','panels':'historical fixed panels, no fresh blinded test','precision':'float64','blas_threads':2,'batch':256,'design_sha256':sha(DESIGN/'实验设计.md')}
    codehash=sha(__file__)
    if (OUT/'implementation_manifest.json').exists():
        previous=json.loads((OUT/'implementation_manifest.json').read_text(encoding='utf-8'))
        assert previous['script_sha256']==codehash,'frozen implementation changed'
        assert json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))==protocol
    else:
        dump('protocol.json',protocol)
        dump('implementation_manifest.json',{'frozen_utc':datetime.now(timezone.utc).isoformat(),'script_sha256':codehash,'python':sys.version,'numpy':np.__version__,'sklearn':sklearn.__version__,'platform':platform.platform(),'threadpools':threadpool_info(),'input_audit_sha256':sha(OUT/'input_audit.json')})

def execute(meta,bank,ex):
    labels,fonts=meta['labels'],meta['fonts']
    (OUT/'prototypes').mkdir(exist_ok=True); (OUT/'checkpoints').mkdir(exist_ok=True)
    fits=[]
    for fold,f in list(enumerate(np.unique(fonts)))+[(6,None)]:
        tag=f'fold{fold}' if f is not None else 'full'
        mask=fonts!=f if f is not None else np.ones(len(fonts),bool)
        for method in METHODS:
            checkpoint=OUT/'checkpoints'/f'{tag}_{method}.npz'
            if checkpoint.exists():
                fits.append(json.loads((checkpoint.with_suffix('.json')).read_text(encoding='utf-8'))); continue
            t=time.perf_counter(); p,z=fit(bank,labels,mask,method); fit_s=time.perf_counter()-t
            pp=OUT/'prototypes'/f'{tag}_{method}.npy'; np.save(pp,p)
            reloaded=np.load(pp); assert np.array_equal(p,reloaded)
            result={}; ties=0; t=time.perf_counter()
            if f is not None:
                va=fonts==f; predlist=[]
                for s in range(5301,5306):
                    pred,nt=predict(bank[f'moderate_seed{s}'][va],reloaded); predlist.append(pred); ties+=nt
                result['DEV']=np.concatenate(predlist)
            else:
                for panel,(d,x) in ex.items():
                    result[panel],nt=predict(x,reloaded); ties+=nt
            info={'tag':tag,'method':method,'zero_columns':z,'exact_ties':ties,'fit_seconds':fit_s,'score_seconds':time.perf_counter()-t,'prototype_sha256':sha(pp)}
            temp=checkpoint.with_suffix('.tmp.npz'); np.savez_compressed(temp,**result); os.replace(temp,checkpoint)
            dump(str(checkpoint.with_suffix('.json').relative_to(OUT)),info); fits.append(info)
        print('COMPLETED',tag,flush=True)
    dump('fit_log.json',fits)

def table(name,rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def analyze(meta,ex):
    panels={}; fonts=meta['fonts']; labels=meta['labels']
    y=[]; fs=[]; seeds=[]; samples=[]
    for f in np.unique(fonts):
        va=fonts==f
        for s in range(5301,5306):
            y.extend(labels[va]); fs.extend(fonts[va]); seeds.extend([s]*180); samples.extend(meta['sample_ids'][va])
    panels['DEV']={'truth':np.array(y),'fonts':np.array(fs),'mapping_seeds':np.array(seeds),'samples':np.array(samples)}
    for method in METHODS: panels['DEV'][method]=np.concatenate([np.load(OUT/'checkpoints'/f'fold{i}_{method}.npz')['DEV'] for i in range(6)])
    for panel,(d,x) in ex.items():
        panels[panel]={k:d[k] for k in ['truth','fonts','mapping_seeds','samples']}
        for method in METHODS: panels[panel][method]=np.load(OUT/'checkpoints'/f'full_{method}.npz')[panel]
    rows=[]; metrics=[]; strata=[]; comparisons={}
    for pi,(panel,d) in enumerate(panels.items()):
        y=d['truth']; n=len(y); characc=[]
        for method in METHODS:
            pred=d[method]; assert pred.shape==y.shape and ((pred>=0)&(pred<180)).all()
            characc.append(np.array([np.mean(pred[y==c]==c) for c in range(180)]))
            for i in range(n): rows.append({'panel':panel,'method':method,'font':str(d['fonts'][i]),'mapping_seed':int(d['mapping_seeds'][i]),'sample_id':str(d['samples'][i]),'class_id':int(y[i]),'predicted':int(pred[i]),'correct':int(y[i]==pred[i])})
            cells=[]
            for s in np.unique(d['mapping_seeds']):
                for f in np.unique(d['fonts']) if panel=='DEV' else [None]:
                    mask=d['mapping_seeds']==s
                    if f is not None: mask &= d['fonts']==f
                    cells.append(f1_score(y[mask],pred[mask],labels=np.arange(180),average='macro',zero_division=0))
            metrics.append({'panel':panel,'method':method,'n':n,'correct':int(np.sum(pred==y)),'accuracy_percent':100*np.mean(pred==y),'macro_f1_percent':100*np.mean(cells)})
            for axis in ['fonts','mapping_seeds']:
                for value in np.unique(d[axis]):
                    mask=d[axis]==value
                    strata.append({'panel':panel,'method':method,'axis':axis,'value':str(value),'n':int(mask.sum()),'correct':int(np.sum(pred[mask]==y[mask])),'accuracy_percent':100*np.mean(pred[mask]==y[mask])})
        ca=np.array(characc).T
        differences=np.stack([ca[:,METHODS.index(a)]-ca[:,METHODS.index(b)] for a,b in PAIRS],axis=1)*100
        rng=np.random.default_rng(26092831+pi); draws=np.empty((100000,len(PAIRS)))
        for start in range(0,100000,2000):
            weights=rng.multinomial(180,np.full(180,1/180),size=2000)
            draws[start:start+2000]=weights@differences/180
        comparisons[panel]=[{'comparison':a+'-'+b,'delta_pp':float(differences[:,j].mean()),'ci95_pp':np.quantile(draws[:,j],[.025,.975]).tolist(),'role':'primary' if panel=='E1' and j==0 else 'secondary_or_exploratory'} for j,(a,b) in enumerate(PAIRS)]
        np.savez_compressed(OUT/(panel+'_bootstrap.npz'),paired_deltas=draws,character_accuracy=ca)
    table('predictions.csv',rows); table('metrics.csv',metrics); table('stratified_metrics.csv',strata); dump('paired_comparisons.json',comparisons)
    keys=[(r['panel'],r['method'],r['font'],r['mapping_seed'],r['class_id']) for r in rows]
    assert len(rows)==64800 and len(set(keys))==64800
    assert len(list((OUT/'prototypes').glob('*.npy')))==42
    # Re-read delivery CSV and independently check all aggregate correct counts.
    with (OUT/'predictions.csv').open(encoding='utf-8-sig',newline='') as f: reread=list(csv.DictReader(f))
    for m in metrics:
        subset=[r for r in reread if r['panel']==m['panel'] and r['method']==m['method']]
        assert len(subset)==m['n'] and sum(int(r['predicted'])==int(r['class_id']) for r in subset)==m['correct']
    for panel,d in panels.items():
        for j,(a,b) in enumerate(PAIRS):
            direct=100*(np.mean(d[a]==d['truth'])-np.mean(d[b]==d['truth']))
            assert abs(direct-comparisons[panel][j]['delta_pp'])<1e-10
    dump('completion_audit.json',{'status':'PASS','prediction_rows':len(rows),'unique_keys':len(set(keys)),'prototype_matrices':42,'all_six_methods_all_panels':True,'prototype_save_reload_equal':True,'csv_correct_counts_independently_rechecked':True,'delta_sign_and_unrounded_arithmetic_checked':True,'baseline_replay':'PASS','no_selection':True})
    table('evidence_crosswalk.csv',[{'experiment':'PA-HOG B','raw_evidence':'prototypes/; checkpoints/; predictions.csv','summary':'metrics.csv; paired_comparisons.json; stratified_metrics.csv','manuscript_role':'原型构建机制对照；正文待整合','supplement_role':'全部六组与分层结果；待整合','status':'新证据完整保存，未修改Word'}, {'experiment':'既有MMA/Point-SPV/XDomainMix/RanDumb及旧控制','raw_evidence':'既有输出目录及全实验核对版论文','summary':'本轮不重算或筛除','manuscript_role':'保留正负结果及协议差异','supplement_role':'后续修改Word时逐项交叉核对','status':'本轮未修改旧文件，未宣称完成全稿审查'}])
    print(json.dumps({'metrics':metrics,'comparisons':comparisons},ensure_ascii=False,indent=2),flush=True)

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    threading.Thread(target=monitor,daemon=True).start()
    with threadpool_limits(limits=2):
        meta,bank,ex=load_inputs(); print('INPUTS AND CONTRACTS PASS',flush=True)
        baseline(meta,bank,ex); freeze()
        if '--preflight-only' not in sys.argv:
            execute(meta,bank,ex); analyze(meta,ex)
    STOP.set(); RESOURCE['elapsed_seconds']=time.perf_counter()-START
    RESOURCE['peak_rss_gib']=RESOURCE['peak_rss_bytes']/1024**3
    dump('resource_log.json',RESOURCE)

if __name__=='__main__': main()
