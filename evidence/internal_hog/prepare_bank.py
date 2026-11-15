"""Reuse frozen rendering code; generate ONLY development-font feature banks."""
import sys
sys.dont_write_bytecode = True
import time
import platform
from importlib.metadata import version
import numpy as np
from scipy.ndimage import affine_transform
from threadpoolctl import threadpool_limits
from common import ROOT, CACHE, PLAN, PLAN_PATH, sha, save_json, mapping_name
from src.v2.mma_development import load_mma_development_config, load_development_dataset, _ideal_skeleton_selections, _build_hog, _load_protocol_mappings
from src.mapping import make_moderate_mapping
from src.grid import make_regular_grid
from src.spv_forward import GaussianPhospheneForwardModel

def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    if (CACHE/'manifest.json').exists():
        from common import check_cache
        check_cache();print('valid existing bank', flush=True);return
    config = load_mma_development_config(ROOT/'configs/legacy_d2.yaml', project_root=ROOT)
    assert sha(config.dataset_path) == config.expected_dataset_sha256
    ds = load_development_dataset(config)
    assert len(ds.images) == 1080 and len(np.unique(ds.class_ids)) == 180
    mappings = _load_protocol_mappings(config)
    ideal = mappings['ideal']
    grid = make_regular_grid()
    # Cross-check the newly generated mappings against frozen original assets.
    for s in (2026,2027,2028,2029):
        new = make_moderate_mapping(grid, seed=s)
        assert np.array_equal(new.centers_xy, mappings[mapping_name(s)].centers_xy)
        assert np.array_equal(new.sigma_px, mappings[mapping_name(s)].sigma_px)
    selections = _ideal_skeleton_selections(ds, ideal)
    np.savez(CACHE/'metadata.npz', labels=ds.class_ids, fonts=ds.font_ids, characters=ds.characters, sample_ids=ds.sample_ids, selections=selections)
    np.save(CACHE/'glyphs.npy', ds.images)
    groups = PLAN['training_mapping_groups']
    seeds = [0] + sorted({s for g in groups for s in g} | set(PLAN['development_evaluation_mapping_seeds']))
    extractor = _build_hog(config)
    amp = np.ones(selections.shape, dtype=np.float32)
    audit = {}
    for s in seeds:
        name = mapping_name(s)
        start = time.perf_counter()
        mapping = ideal if s == 0 else make_moderate_mapping(grid, seed=s)
        np.savez(CACHE/f'{name}_geometry.npz', centers_xy=mapping.centers_xy, sigma_px=mapping.sigma_px)
        images = GaussianPhospheneForwardModel(mapping).generate_variable_batch(selections, amp)
        assert np.allclose(images.sum((1,2),dtype=np.float64),64,atol=2e-5,rtol=0)
        features = extractor.extract_batch(images)
        np.save(CACHE/f'{name}_hog.npy', features)
        # Neural pilot uses original mapping group only. Other groups need HOG only.
        if s in [0,*groups[0],*PLAN['development_evaluation_mapping_seeds']]:
            np.save(CACHE/f'{name}_images.npy', images)
        audit[name] = {'seconds':time.perf_counter()-start,'feature_sha256':sha(CACHE/f'{name}_hog.npy')}
        print(f'bank {name} {features.shape} {audit[name]["seconds"]:.1f}s',flush=True)
    # Four pre-specified ordinary affine replicas, same replica count as MMA.
    images = np.load(CACHE/'ideal_images.npy')
    cfg = PLAN['ordinary_augmentation'];rng=np.random.default_rng(cfg['seed']);center=np.array([47.5,47.5])
    for k in range(4):
        transformed=[];params=[]
        for im in images:
            shift=np.clip(rng.normal(0,cfg['translation_std_px'],2),-cfg['translation_clip_px'],cfg['translation_clip_px'])
            scale=rng.uniform(*cfg['scale_uniform']);matrix=np.eye(2)/scale
            offset=center-matrix@(center+shift)
            out=affine_transform(im,matrix,offset=offset,order=1,mode='constant',cval=0,prefilter=False)
            out=np.asarray(out,dtype=np.float32)
            assert out.sum()>0
            out *= np.float32(64/out.sum(dtype=np.float64))
            transformed.append(out);params.append([*shift,scale])
        np.save(CACHE/f'affine{k}_params.npy',params)
        np.save(CACHE/f'affine{k}_hog.npy',extractor.extract_batch(np.stack(transformed)))
        print('affine',k,flush=True)
    save_json(CACHE/'manifest.json',{'plan_sha256':sha(PLAN_PATH),'dataset_sha256':sha(config.dataset_path),'fonts':list(config.development_fonts),'n':1080,'classes':180,'budget':64,'mapping_audit':audit,'python':platform.python_version(),'packages':{p:version(p) for p in ['numpy','scipy','scikit-image','scikit-learn','torch']}})
    save_json(ROOT/'results/bank_manifest.json',__import__('json').loads((CACHE/'manifest.json').read_text(encoding='utf-8')))

if __name__ == '__main__':
    with threadpool_limits(limits=2):main()
