"""Window HOG adapters; training uses no test inputs or scores."""
import argparse,time
from collections import Counter
from core import *
from skimage.feature import canny
from threadpoolctl import threadpool_limits
from src.grid import make_regular_grid
from src.mapping import make_ideal_mapping
from src.spv_forward import IdealGaussianUtilityProjector
from src.encoders.skeleton_topk import SkeletonTopKEncoder
from src.encoders.base import deterministic_topk
from src.features import HogFeatureExtractor

def norm(a):
    a=np.asarray(a,dtype=np.float64);n=np.linalg.norm(a,axis=-1,keepdims=True)
    require(np.isfinite(a).all() and (n>0).all(),'nonzero finite prototype/features')
    return a/n

class Adapter:
    def __init__(self,kind):
        self.kind=kind;self.grid=make_regular_grid();self.projector=IdealGaussianUtilityProjector(make_ideal_mapping(self.grid));self.hog=HogFeatureExtractor(image_shape=(96,96))
    def select(self,patch):
        im=F.interpolate(patch[None,:1],size=(96,96),mode='bilinear',align_corners=False)[0,0].numpy().copy()
        if self.kind=='skeleton':
            return SkeletonTopKEncoder().encode(im,self.grid.electrodes,64,projector=self.projector).selected_indices
        edges=canny(im,sigma=1.,low_threshold=.1,high_threshold=.2,use_quantiles=False,mode='constant',cval=0).astype(np.float32)
        return deterministic_topk(self.projector.project(edges),64)
    def feature(self,selected,kernel):
        # Same linear unit-amplitude renderer; reduction order may differ in float32.
        image=kernel[selected].sum(axis=0,dtype=np.float32)
        return self.hog.extract(image)

def event_counts(schedule):
    result={}
    for step,rr in enumerate(schedule['rows']):
        mi=int(schedule['mapping_index'][step])
        for r,pt in zip(rr,schedule['points'][step]):
            key=(int(r),int(pt[0]),int(pt[1]))
            if key not in result:result[key]=np.zeros(5,dtype=np.int64)
            result[key][mi]+=1
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--benchmark',type=int,default=0);a=ap.parse_args()
    verify_manifest();torch.set_num_threads(2);d=GlyphData();s=np.load(ROOT/'data/train/schedule.npz');kernels=np.load(ROOT/'data/train/kernels.npy');events=event_counts(s)
    out=ROOT/('engineering' if a.benchmark else 'hog_models');out.mkdir(exist_ok=True)
    if not a.benchmark:require(not any(out.glob('*.npz')),'HOG output exists; do not overwrite')
    reports={}
    for kind in ['skeleton','canny']:
        adapter=Adapter(kind);sums=np.zeros((6,180,5,4356),dtype=np.float64);counts=np.zeros((6,180,5),dtype=np.int64)
        ideal_sums=np.zeros((6,180,4356),dtype=np.float64);ideal_counts=np.zeros((6,180),dtype=np.int64);fonts=np.unique(d.fonts);start=time.perf_counter();feature_calls=0
        for j,(key,weights) in enumerate(events.items()):
            r,y,x=key;patch,_=d.patch(r,(y,x));selected=adapter.select(patch);fi=int(np.searchsorted(fonts,d.fonts[r]));c=int(d.y[r]);feature0=adapter.feature(selected,kernels[0]);feature_calls+=1
            ideal_sums[fi,c]+=feature0.astype(np.float64)*int(weights.sum());ideal_counts[fi,c]+=int(weights.sum())
            for mi in np.flatnonzero(weights):
                feat=feature0 if mi==0 else adapter.feature(selected,kernels[mi]);feature_calls+=int(mi!=0)
                sums[fi,c,mi]+=feat.astype(np.float64)*int(weights[mi]);counts[fi,c,mi]+=weights[mi]
            if (j+1)%1000==0:print(kind,j+1,len(events),time.perf_counter()-start,flush=True)
            if a.benchmark and j+1>=a.benchmark:break
        elapsed=time.perf_counter()-start
        reports[kind]={'unique_crops_processed':j+1,'total_unique_crops':len(events),'feature_calls':feature_calls,'seconds':elapsed,'rough_full_seconds':elapsed*len(events)/(j+1),'benchmark_extrapolation_only':bool(a.benchmark)}
        if not a.benchmark:
            require(counts.min()>0 and ideal_counts.min()>0,'empty prototype group')
            require(int(ideal_counts.sum())==80000 and int(counts.sum())==80000,'schedule multiplicity')
            ideal=norm((ideal_sums/ideal_counts[...,None]).mean(0));mma=norm((sums/counts[...,None]).mean(0).mean(1))
            np.savez_compressed(out/f'{kind}.npz',ideal=ideal,mma=mma,ideal_counts=ideal_counts,multi_counts=counts,fonts=fonts)
        print(kind,reports[kind],flush=True)
    save(out/('hog_benchmark.json' if a.benchmark else 'completion.json'),{'completed':not bool(a.benchmark),'manifest_sha256':sha(ROOT/'manifest.json'),'reports':reports})

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
