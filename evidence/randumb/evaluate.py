import json,time,csv,gc,threading
from pathlib import Path
import numpy as np
import psutil
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
from core import ResidualOAS,embedder,transform,predict
from preflight import ROOT,CACHE,sha,write

def projection_hash(obj):
    import hashlib
    h=hashlib.sha256()
    for a in [obj.random_weights_,obj.random_offset_]:
        h.update(memoryview(np.ascontiguousarray(a)).cast('B'))
    return h.hexdigest()

def main():
    cfg=json.loads((ROOT/'selected_config.json').read_text(encoding='utf-8'))
    assert cfg['core_hash']==sha(ROOT/'core.py')
    ext=ROOT/'external';assert (ext/'manifest.json').exists()
    result=ROOT/'evaluation';result.mkdir(exist_ok=True)
    models=ROOT/'models';models.mkdir(exist_ok=True)
    protocol=dict(selection_hash=sha(ROOT/'selected_config.json'),external_manifest_hash=sha(ext/'manifest.json'),evaluator_hash=sha(Path(__file__)),core_hash=sha(ROOT/'core.py'),bootstrap_seed=20260928,bootstrap_repetitions=100000,primary='H1 minus R1 mean seed accuracy on E1',dtype='float64')
    p=ROOT/'evaluation_protocol.json'
    if p.exists():assert json.loads(p.read_text(encoding='utf-8'))==protocol
    else:write(p.name,protocol)
    meta=np.load(CACHE/'metadata.npz');y=meta['labels'];names=['ideal']+[f'moderate_seed{s}' for s in range(2026,2030)]
    bank=[np.load(CACHE/f'{n}_images.npy',mmap_mode='r').reshape(1080,-1) for n in names]
    allrows=[];timings=[]
    for seed in cfg['final_seeds']:
        for condition,method in [('ideal','R0'),('multi','R1')]:
            par=cfg[condition];dim=cfg['dim'];path=result/f'{method}_{seed}.json'
            if path.exists():continue
            start=time.perf_counter();mem=[psutil.Process().memory_info().rss];done=threading.Event()
            def monitor():
                while not done.wait(.1):mem[0]=max(mem[0],psutil.Process().memory_info().rss)
            threading.Thread(target=monitor,daemon=True).start()
            x=np.concatenate(bank if condition=='multi' else bank[:1]);labels=np.tile(y,5 if condition=='multi' else 1)
            obj=embedder(bank[0],dim,par['mult'],seed);z=transform(obj,x);features_seconds=time.perf_counter()-start
            t=time.perf_counter();fit=ResidualOAS(z,labels);w,b=fit.solve(par['rho']);fit_seconds=time.perf_counter()-t
            np.savez_compressed(models/f'{method}_{seed}.npz',w=w,b=b)
            info=dict(method=method,seed=seed,dim=dim,mult=par['mult'],rho=par['rho'],gamma=obj.gamma,feature_seconds=features_seconds,fit_seconds=fit_seconds,shrinkage=fit.s,projection_bytes=obj.random_weights_.nbytes+obj.random_offset_.nbytes,classifier_bytes=w.nbytes+b.nbytes,projection_reconstruction='RBFSampler fitted with recorded gamma, seed, dimension and pinned versions; no query data',projection_sha256=projection_hash(obj),panels={})
            for panel in ['E1','C1']:
                t=time.perf_counter();q=np.load(ext/f'{panel}_images.npy',mmap_mode='r').reshape(-1,9216);zq=transform(obj,q);pred=predict(zq,w,b)
                np.save(result/f'{method}_{seed}_{panel}_pred.npy',pred)
                info['panels'][panel]={'predict_total_seconds':time.perf_counter()-t,'n':len(q)}
                del zq
            # Fixed preprocessing/configuration diagnostic for ideal duplication.
            if method=='R0':
                dfit=ResidualOAS(z,labels,repeat=5);dw,db=dfit.solve(par['rho'])
                np.savez_compressed(models/f'R0repeat5_{seed}.npz',w=dw,b=db)
                for panel in ['E1','C1']:
                    q=np.load(ext/f'{panel}_images.npy',mmap_mode='r').reshape(-1,9216);zq=transform(obj,q)
                    np.save(result/f'R0repeat5_{seed}_{panel}_pred.npy',predict(zq,dw,db));del zq
                info['repeat5_shrinkage']=dfit.s;del dfit,dw,db
            # Shared multi-selected configuration: R1 is already available,
            # only ideal training has to be fitted again, no selection here.
            if method=='R1':
                zi=z[:len(y)];sf=ResidualOAS(zi,y);sw,sb=sf.solve(par['rho'])
                np.savez_compressed(models/f'R0shared_{seed}.npz',w=sw,b=sb)
                for panel in ['E1','C1']:
                    q=np.load(ext/f'{panel}_images.npy',mmap_mode='r').reshape(-1,9216);zq=transform(obj,q)
                    np.save(result/f'R0shared_{seed}_{panel}_pred.npy',predict(zq,sw,sb));del zq
                del sf,sw,sb
            done.set();info['peak_rss_including_diagnostics']=mem[0];info['group_total_seconds']=time.perf_counter()-start
            path.write_text(json.dumps(info,indent=2),encoding='utf-8')
            write('status.json',dict(stage='EXTERNAL_EVALUATION',last_model=f'{method}_{seed}'))
            print('EVALUATED',method,seed,round(info['group_total_seconds'],1),flush=True)
            del x,obj,z,fit,w,b;gc.collect()
    print('ALL EXTERNAL PREDICTIONS COMPLETE',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
