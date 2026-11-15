import sys,os,json,time,hashlib,threading,platform
from pathlib import Path
import numpy as np
from sklearn.covariance import OAS
from scipy.linalg import solve
from threadpoolctl import threadpool_limits
import psutil
from core import ResidualOAS,embedder,transform,predict

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parents[1]
CACHE=WORK/'work/mma_supplement_cache'
def write(name,x):
    (ROOT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    start=time.perf_counter(); process=psutil.Process(); initial=psutil.virtual_memory().available
    peak=[process.memory_info().rss];done=threading.Event()
    def monitor():
        while not done.wait(.1):peak[0]=max(peak[0],process.memory_info().rss)
    threading.Thread(target=monitor,daemon=True).start()
    tests=[]
    rng=np.random.RandomState(729)
    for n,p in [(60,120),(120,30)]:
        y=np.arange(n)%6;z=rng.normal(size=(n,p));q=rng.normal(size=(20,p))
        for repeat in [1,5]:
            low=ResidualOAS(z,y,repeat)
            e=np.repeat(z-low.means[y],repeat,axis=0)
            ref=OAS(assume_centered=True).fit(e)
            assert abs(ref.shrinkage_-low.s)<1e-10
            for rho in [1e-5,1e-4,1e-3]:
                w,b=low.solve(rho)
                a=(1-rho)*ref.covariance_+rho*np.eye(p)
                wr=solve(a,low.means.T,assume_a='pos')
                br=-.5*np.sum(low.means.T*wr,axis=0)
                error=float(np.max(np.abs(q@w+b-(q@wr+br))))
                assert error<1e-8
                # legacy uses float32; audit precision change separately.
                wf=solve(a.astype('float32'),low.means.T.astype('float32'),assume_a='pos')
                bf=-.5*np.sum(low.means.T.astype('float32')*wf,axis=0)
                assert np.array_equal(predict(q,w,b),predict(q,wf,bf))
                tests.append(dict(n=n,p=p,repeat=repeat,rho=rho,max_abs_score_error=error))
    write('correctness.json',dict(status='PASS',tests=tests,precision='float64; compared float32 reference predictions'))
    print('CORRECTNESS PASS',flush=True)
    meta=np.load(CACHE/'metadata.npz');y=meta['labels'];fonts=meta['fonts'];hold=np.unique(fonts)[0];tr=fonts!=hold
    names=['ideal']+[f'moderate_seed{s}' for s in range(2026,2030)]
    source_hashes={str(CACHE/'metadata.npz'):sha(CACHE/'metadata.npz')}
    xs=[]
    for name in names:
        path=CACHE/f'{name}_images.npy';source_hashes[str(path)]=sha(path)
        a=np.load(path,mmap_mode='r');assert a.shape==(1080,96,96)
        assert np.isfinite(a).all() and np.allclose(a.sum((1,2),dtype=np.float64),64,atol=2e-5,rtol=0)
        xs.append(a[tr].reshape(-1,9216))
    x=np.concatenate(xs);labels=np.tile(y[tr],5)
    write('environment.json',dict(python=sys.version,platform=platform.platform(),cpu_threads=psutil.cpu_count(),ram_total=psutil.virtual_memory().total,ram_available_start=initial,blas_threads=2,source_hashes=source_hashes,core_sha256=sha(ROOT/'core.py')))
    trials=[]
    for dim in [10000,4096,2048]:
        peak[0]=process.memory_info().rss;t=time.perf_counter()
        print('RESOURCE START',dim,flush=True)
        obj=embedder(xs[0],dim,1,17);z=transform(obj,x)
        t_feature=time.perf_counter()-t
        print('FEATURE DONE',dim,round(t_feature,2),flush=True)
        fit_start=time.perf_counter();fit=ResidualOAS(z,labels);w,b=fit.solve(1e-4)
        fit_seconds=time.perf_counter()-fit_start
        trial=dict(dim=dim,feature_seconds=t_feature,fit_seconds=fit_seconds,total_seconds=time.perf_counter()-t,peak_rss=peak[0],oas_shrinkage=fit.s,n=len(x),heldout_font=str(hold),accuracy_not_used=True)
        # Conservative estimate includes 156 full fits without assuming reuse.
        trial['projected_development_hours']=trial['total_seconds']*156/3600
        trial['pass']=bool(trial['peak_rss']<=initial*.6 and trial['total_seconds']<=1800 and trial['projected_development_hours']<=24)
        trials.append(trial);write('resource_preflight.json',dict(status='PASS' if trial['pass'] else 'RETRY_LOWER_DIM',trials=trials))
        print(json.dumps(trial),flush=True)
        if trial['pass']:
            write('dimension_decision.json',dict(dim=dim,reason='resource_gate_only',trial=trial,source_hashes=source_hashes));break
        del obj,z,fit,w,b
    else: raise RuntimeError('No dimension passed resource gate')
    done.set();print('PREFLIGHT COMPLETE',round(time.perf_counter()-start,2),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
