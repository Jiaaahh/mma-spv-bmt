import sys,json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from preflight import ROOT,WORK,CACHE,sha,write
SUPP=WORK/'outputs/mma_supplement_20260908'
sys.path.insert(0,str(SUPP))
from src.v2.mma_development import load_mma_development_config,load_development_dataset,_ideal_skeleton_selections,_load_protocol_mappings
from src.mapping import make_moderate_mapping
from src.grid import make_regular_grid
from src.spv_forward import GaussianPhospheneForwardModel

def main():
    config=load_mma_development_config(SUPP/'configs/legacy_d2.yaml',project_root=SUPP)
    assert sha(config.dataset_path)==config.expected_dataset_sha256
    ds=load_development_dataset(config);meta=np.load(CACHE/'metadata.npz')
    assert np.array_equal(ds.class_ids,meta['labels']) and np.array_equal(ds.font_ids,meta['fonts'])
    assert np.array_equal(ds.images,np.load(CACHE/'glyphs.npy'))
    mappings=_load_protocol_mappings(config);ideal=mappings['ideal'];selected=_ideal_skeleton_selections(ds,ideal)
    assert np.array_equal(selected,meta['selections'])
    manifest=json.loads((CACHE/'manifest.json').read_text(encoding='utf-8'));audits=[]
    for seed in [0,2026,2027,2028,2029,5301,5302,5303,5304,5305]:
        name='ideal' if seed==0 else f'moderate_seed{seed}'
        mapping=ideal if seed==0 else make_moderate_mapping(make_regular_grid(),seed=seed)
        if seed in [2026,2027,2028,2029]:
            assert np.array_equal(mapping.centers_xy,mappings[name].centers_xy)
            assert np.array_equal(mapping.sigma_px,mappings[name].sigma_px)
        forward=GaussianPhospheneForwardModel(mapping);images=np.load(CACHE/f'{name}_images.npy',mmap_mode='r')
        maxerr=0.
        for i in range(0,len(selected),12):
            sel=selected[i:i+12];expected=forward.generate_variable_batch(sel,np.ones(sel.shape,dtype='float32'))
            error=float(np.max(np.abs(images[i:i+12]-expected)));maxerr=max(maxerr,error)
        assert maxerr==0,(name,maxerr)
        assert sha(CACHE/f'{name}_hog.npy')==manifest['mapping_audit'][name]['feature_sha256']
        audits.append(dict(name=name,n=len(images),max_abs_render_error=maxerr,hog_hash_match=True))
    # Reproduce all existing HOG predictions on the development panel.
    old=np.load(SUPP/'results/hog/predictions.npz');y=meta['labels'];fonts=meta['fonts'];checks={}
    for name,maps in [('ideal_only',['ideal']),('mma_g0',['ideal']+[f'moderate_seed{s}' for s in range(2026,2030)])]:
        ps=[]
        for font in np.unique(fonts):
            tr=fonts!=font;va=~tr;h=np.concatenate([np.load(CACHE/f'{m}_hog.npy')[tr].astype('float64') for m in maps]);labels=np.tile(y[tr],len(maps))
            means=np.stack([h[labels==c].mean(0) for c in range(180)]);means/=np.linalg.norm(means,axis=1,keepdims=True)
            pp=[]
            for s in range(5301,5306):
                hq=np.load(CACHE/f'moderate_seed{s}_hog.npy')[va].astype('float64');p=np.argmax(hq@means.T,axis=1)
                ordered=np.empty(180,dtype='int64');ordered[y[va]]=p;pp.append(ordered)
            ps.append(pp)
        assert np.array_equal(np.array(ps),old[name]);checks[name]=True
    write('bank_audit.json',dict(status='PASS',original_glyph_sha256=sha(config.dataset_path),renders=audits,old_development_prediction_matches=checks))
    print('BANK AUDIT PASS',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
