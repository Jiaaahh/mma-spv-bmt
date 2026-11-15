import sys,json,time,gc,platform
import numpy as np
from threadpoolctl import threadpool_limits,threadpool_info
from core import embedder,transform
from preflight import ROOT,CACHE,WORK,sha,write
sys.path.insert(0,str(WORK/'outputs/mma_supplement_20260908'))
from src.features import HogFeatureExtractor

def bench(fn):
    for _ in range(20):fn()
    t=[]
    for _ in range(100):
        s=time.perf_counter();fn();t.append(time.perf_counter()-s)
    return dict(median_ms=float(np.median(t)*1000),p95_ms=float(np.quantile(t,.95)*1000),raw_seconds=t,warmups=20,repeats=100)

def main():
    cfg=json.loads((ROOT/'selected_config.json').read_text(encoding='utf-8'))
    q=np.load(ROOT/'external/E1_images.npy',mmap_mode='r')[:32]
    ideal=np.load(CACHE/'ideal_images.npy',mmap_mode='r').reshape(1080,-1)
    costs={'hardware':platform.platform(),'blas_threads':2,'batch_sizes':[1,32],'scope':'SPV float image to class scores; excludes shared rendering; first frozen seed only for inference timing','methods':{}}
    hog=HogFeatureExtractor(image_shape=(96,96),orientations=9,pixels_per_cell=(8,8),cells_per_block=(2,2),block_norm='L2-Hys')
    for method in ['H0','H1']:
        proto=np.load(ROOT/f'external/{method}_prototypes.npy')
        def run(n):return hog.extract_batch(q[:n]).astype('float64')@proto.T
        costs['methods'][method]=dict(classifier_bytes=proto.nbytes,inference={str(n):bench(lambda:run(n)) for n in [1,32]})
        # Refitting from existing HOG features, separate from extraction.
        names=['ideal'] if method=='H0' else ['ideal']+[f'moderate_seed{s}' for s in range(2026,2030)]
        y=np.load(CACHE/'metadata.npz')['labels'];h=np.concatenate([np.load(CACHE/f'{s}_hog.npy').astype('float64') for s in names]);labels=np.tile(y,len(names))
        t=time.perf_counter();p=np.stack([h[labels==c].mean(0) for c in range(180)]);p/=np.linalg.norm(p,axis=1,keepdims=True)
        costs['methods'][method]['statistics_fit_seconds']=time.perf_counter()-t
        # Same full training image sets; actual extraction wall time.
        t=time.perf_counter()
        for name in names:
            ims=np.load(CACHE/f'{name}_images.npy',mmap_mode='r')
            hh=hog.extract_batch(ims)
        costs['methods'][method]['full_training_hog_seconds']=time.perf_counter()-t
        del h,hh
        print('COST MEASURED',method,flush=True)
    for method,condition in [('R0','ideal'),('R1','multi')]:
        seed=cfg['final_seeds'][0];par=cfg[condition];obj=embedder(ideal,cfg['dim'],par['mult'],seed)
        model=np.load(ROOT/f'models/{method}_{seed}.npz');w=model['w'];b=model['b']
        def run(n):return transform(obj,q[:n].reshape(n,-1))@w+b
        infos=[json.loads((ROOT/f'evaluation/{method}_{s}.json').read_text(encoding='utf-8')) for s in cfg['final_seeds']]
        costs['methods'][method]=dict(classifier_bytes=w.nbytes+b.nbytes,materialized_projection_bytes=obj.random_weights_.nbytes+obj.random_offset_.nbytes,inference={str(n):bench(lambda:run(n)) for n in [1,32]},training_feature_seconds=[i['feature_seconds'] for i in infos],statistics_fit_seconds=[i['fit_seconds'] for i in infos],peak_rss_including_diagnostics=[i['peak_rss_including_diagnostics'] for i in infos],stored_classifier_file_bytes=(ROOT/f'models/{method}_{seed}.npz').stat().st_size,projection_stored_as='seed + gamma + version metadata, materialized RAM reported separately')
        del obj,w,b,model;gc.collect()
        print('COST MEASURED',method,flush=True)
    costs['libraries']=threadpool_info();write('costs.json',costs)
    write('status.json',dict(stage='COMPLETE',results='results_summary.json',costs='costs.json'))

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
