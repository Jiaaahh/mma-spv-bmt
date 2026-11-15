"""Frozen budget-adapted Canny x MMA development comparison. CPU only."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,csv,platform
from importlib.metadata import version
import numpy as np
from skimage.feature import canny
from threadpoolctl import threadpool_limits
from src.grid import make_regular_grid
from src.mapping import make_ideal_mapping,make_moderate_mapping
from src.spv_forward import GaussianPhospheneForwardModel,IdealGaussianUtilityProjector
from src.encoders.skeleton_topk import SkeletonTopKEncoder
from src.encoders.base import deterministic_topk
from src.features import HogFeatureExtractor

ROOT=Path(__file__).resolve().parent
P=json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,obj): p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def require(ok,msg):
    if not ok: raise ValueError(msg)
def normalize(x):
    x=np.asarray(x,dtype=np.float64); n=np.linalg.norm(x,axis=-1,keepdims=True)
    require(np.isfinite(x).all() and (n>0).all(),'invalid features')
    return x/n
def edge_selection(image,projector):
    require(image.shape==(96,96) and image.dtype==np.float32 and np.isfinite(image).all() and image.min()>=0 and image.max()<=1,'invalid glyph')
    drive=canny(image,**P['canny']).astype(np.float32)
    scores=projector.project(drive)
    selected=deterministic_topk(scores,P['budget'])
    return selected,{'edge_pixels':int(drive.sum()),'positive_candidates':int((scores>0).sum()),'zero_selected':int((scores[selected]==0).sum())}
def classify(bank,labels,fonts):
    order=np.unique(fonts); result={k:np.empty((6,5,180),dtype=np.int64) for k in ['ideal','mma']}
    require(not set(P['train_seeds'])&set(P['eval_seeds']),'mapping overlap')
    for fi,font in enumerate(order):
        tr=np.flatnonzero(fonts!=font);te=np.flatnonzero(fonts==font)
        require(len(tr)==900 and len(te)==180 and np.array_equal(np.sort(labels[te]),np.arange(180)),'invalid split')
        means={s:np.stack([np.asarray(bank[s][tr][labels[tr]==c],dtype=np.float64).mean(0) for c in range(180)]) for s in [0,*P['train_seeds']]}
        prototypes={'ideal':normalize(means[0]),'mma':normalize(np.stack(list(means.values())).mean(0))}
        for mi,s in enumerate(P['eval_seeds']):
            x=normalize(bank[s][te])
            for name,proto in prototypes.items():result[name][fi,mi,labels[te]]=(x@proto.T).argmax(1)
    return result
def main():
    for name,digest in json.loads((ROOT/'input_manifest.json').read_text()).items():
        require(sha(ROOT/name)==digest,'input changed: '+name)
    for name,v in {'numpy':'1.26.4','scipy':'1.13.1','scikit-image':'0.25.0'}.items():
        require(version(name)==v,'use pinned requirements: '+name)
    out=ROOT/'results';out.mkdir(exist_ok=False)
    save(out/'execution_manifest.json',{'protocol_sha256':sha(ROOT/'protocol.json'),'run_sha256':sha(__file__),'python':platform.python_version(),'packages':{n:version(n) for n in ['numpy','scipy','scikit-image','threadpoolctl']}})
    data=ROOT/'data'; meta=np.load(data/'metadata.npz');glyphs=np.load(data/'glyphs.npy');labels=meta['labels'];fonts=meta['fonts']
    require(glyphs.shape==(1080,96,96) and len(np.unique(fonts))==6,'dataset shape')
    for f in np.unique(fonts): require(np.array_equal(np.sort(labels[fonts==f]),np.arange(180)),'unbalanced classes')
    grid=make_regular_grid();ideal=make_ideal_mapping(grid);projector=IdealGaussianUtilityProjector(ideal)
    skeleton=np.stack([SkeletonTopKEncoder().encode(im,grid.electrodes,64,projector=projector).selected_indices for im in glyphs])
    require(np.array_equal(skeleton,meta['selections']),'historical skeleton selection mismatch')
    mappings={s:ideal if s==0 else make_moderate_mapping(grid,seed=s) for s in [0,*P['train_seeds'],*P['eval_seeds']]}
    for s,m in mappings.items():
        n='ideal' if s==0 else f'moderate_seed{s}';old=np.load(data/f'{n}_geometry.npz')
        require(np.array_equal(m.centers_xy,old['centers_xy']) and np.array_equal(m.sigma_px,old['sigma_px']),'geometry mismatch')
    extractor=HogFeatureExtractor(image_shape=(96,96));predictions={};audit={}
    # The edge outcomes are computed only after the skeleton reproduction gate.
    for encoder in ['skeleton','edge']:
        if encoder=='skeleton':selections=skeleton
        else:
            pairs=[edge_selection(im,projector) for im in glyphs];selections=np.stack([x[0] for x in pairs]);save(out/'edge_diagnostics.json',[{'sample_id':str(meta['sample_ids'][i]),**x[1]} for i,x in enumerate(pairs)])
        np.save(out/f'{encoder}_selections.npy',selections)
        bank={}
        for s,m in mappings.items():
            images=GaussianPhospheneForwardModel(m).generate_variable_batch(selections,np.ones(selections.shape,dtype=np.float32))
            require(np.allclose(images.sum((1,2),dtype=np.float64),64,atol=2e-5,rtol=0),'energy mismatch')
            bank[s]=extractor.extract_batch(images)
            print(encoder,s,'features ready',flush=True)
        cells=classify(bank,labels,fonts)
        if encoder=='skeleton':
            reference=np.load(data/'reference_predictions.npz')
            for new,old in [('ideal','ideal_only'),('mma','mma_g0')]:require(np.array_equal(cells[new],reference[old]),'historical prediction mismatch: '+old)
            audit['skeleton_reference_predictions_exact_match']=True
            print('Historical 10800 predictions exactly reproduced',flush=True)
        predictions.update({encoder+'_'+k:v for k,v in cells.items()})
    np.savez_compressed(out/'predictions.npz',**predictions)
    with (out/'predictions.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['method','font','mapping_seed','class_id','predicted'])
        for k,p in predictions.items():
            for fi,font in enumerate(np.unique(fonts)):
                for mi,s in enumerate(P['eval_seeds']):
                    for c in range(180):w.writerow([k,font,s,c,int(p[fi,mi,c])])
    correct={k:(p==np.arange(180)).astype(float) for k,p in predictions.items()}
    rng=np.random.default_rng(P['statistics']['seed']);weights=rng.multinomial(180,np.full(180,1/180),size=P['statistics']['replicates'])/180
    vectors={k:c.mean((0,1)) for k,c in correct.items()}
    diffs={'skeleton_mma_gain':vectors['skeleton_mma']-vectors['skeleton_ideal'],'edge_mma_gain':vectors['edge_mma']-vectors['edge_ideal'],'skeleton_minus_edge_ideal':vectors['skeleton_ideal']-vectors['edge_ideal'],'skeleton_minus_edge_mma':vectors['skeleton_mma']-vectors['edge_mma']}
    diffs['interaction_skeleton_gain_minus_edge_gain']=diffs['skeleton_mma_gain']-diffs['edge_mma_gain']
    summary={'status':'development_only','audit':audit,'n_predictions_per_cell':5400,'cells':{k:{'accuracy_percent':float(v.mean()*100),'correct':int(correct[k].sum()),'ci95_percent':(np.quantile(weights@v,[.025,.975])*100).tolist()} for k,v in vectors.items()},'contrasts':{k:{'difference_pp':float(v.mean()*100),'ci95_pp':(np.quantile(weights@v,[.025,.975])*100).tolist()} for k,v in diffs.items()},'statistics':P['statistics']}
    save(out/'summary.json',summary)
    with (out/'per_font_mapping.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['method','font','mapping_seed','accuracy_percent'])
        for k,c in correct.items():
            for fi,font in enumerate(np.unique(fonts)):
                for mi,s in enumerate(P['eval_seeds']):w.writerow([k,font,s,float(c[fi,mi].mean()*100)])
    save(out/'output_hashes.json',{p.name:sha(p) for p in out.iterdir() if p.is_file()})
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=2):main()
