"""Historical panel preparation, gated on frozen development selection."""
import sys,json,csv,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from preflight import ROOT,WORK,CACHE,sha,write
OLD_BASE=Path('/historical/user/Documents/Codex/2026-08-25/referenced-chatgpt-conversation-this-is-an-2/outputs')
E1=OLD_BASE/'hanzi-spv-mma-extension-e1'
C1=OLD_BASE/'hanzi-spv-calibrated-v2'
sys.path.insert(0,str(E1))
from src.e1.experiment import load_glyph_dataset,load_mapping,compute_selections,build_hog
from src.spv_forward import GaussianPhospheneForwardModel

def main():
    assert json.loads((ROOT/'bank_audit.json').read_text(encoding='utf-8'))['status']=='PASS'
    cfg=json.loads((ROOT/'selected_config.json').read_text(encoding='utf-8'))
    assert cfg['status']=='FROZEN_BEFORE_NEW_EXTERNAL_PREDICTIONS'
    out=ROOT/'external';out.mkdir(exist_ok=True)
    if (out/'manifest.json').exists():return
    import yaml
    # Verify against the old asset manifests, not only record current hashes.
    historical=json.loads((E1/'data/e1_assets/manifest.json').read_text(encoding='utf-8'))
    for r in historical['asset_files']:
        if r['file']=='glyphs_e1_fonts.npz' or r['file'].startswith('mappings/e1_moderate_seed'):
            assert sha(E1/'data/e1_assets'/r['file'])==r['sha256']
    c1cfg=yaml.safe_load((C1/'configs/c1_mma_confirmation_windows_r3.yaml').read_text(encoding='utf-8'))
    assert sha(C1/'data/mma_confirmation/glyphs_confirmation.npz')==c1cfg['confirmation']['glyphs_sha256']
    for r in c1cfg['confirmation']['moderate_mappings'].values():
        assert sha(C1/'data/mma_confirmation'/r['path'])==r['sha256']
    meta=np.load(CACHE/'metadata.npz');y=meta['labels']
    def norm(x):return x/np.linalg.norm(x,axis=1,keepdims=True)
    h=[np.load(CACHE/f'{n}_hog.npy').astype('float64') for n in ['ideal']+[f'moderate_seed{s}' for s in range(2026,2030)]]
    prototypes={}
    for method,features in [('H0',h[:1]),('H1',h)]:
        train=np.concatenate(features);labels=np.tile(y,len(features))
        prototypes[method]=norm(np.stack([train[labels==c].mean(0) for c in range(180)]))
        np.save(out/f'{method}_prototypes.npy',prototypes[method])
    extractor=build_hog({});manifest={'config_sha256':sha(ROOT/'selected_config.json'),'panels':{},'code_hash':sha(Path(__file__))}
    for panel,base,glyph,maps,oldpred in [
        ('E1',E1,E1/'data/e1_assets/glyphs_e1_fonts.npz',[(s,E1/f'data/e1_assets/mappings/e1_moderate_seed{s}.npz') for s in range(3101,3106)],E1/'outputs/e1_confirmation/predictions.csv'),
        ('C1',C1,C1/'data/mma_confirmation/glyphs_confirmation.npz',[(s,C1/f'data/mma_confirmation/mappings/moderate_seed_{s}.npz') for s in range(2031,2036)],C1/'outputs/c1_mma_confirmation_windows_r3/predictions.csv')]:
        start=time.perf_counter();ds=load_glyph_dataset(glyph);sel,audit=compute_selections(ds.images,selector='skeleton',workers=1)
        old={}
        with oldpred.open(encoding='utf-8-sig',newline='') as f:
            for r in csv.DictReader(f):
                if panel=='E1':
                    if r['condition']!='moderate':continue
                    lookup={'proto_skeleton_ideal_only':'H0','proto_skeleton_ideal_plus_multi4':'H1'}
                    if r['method'] not in lookup:continue
                    old[(lookup[r['method']],int(r['mapping_seed']),r['sample_id'])]=int(r['predicted_label'])
                elif r['method'] in ['ideal_only','ideal_plus_multi4']:
                    old[({'ideal_only':'H0','ideal_plus_multi4':'H1'}[r['method']],int(r['mapping_seed']),r['sample_id'])]=int(r['predicted_class_id'])
        images=[];features=[]
        for seed,path in maps:
            mapping=load_mapping(path);forward=GaussianPhospheneForwardModel(mapping)
            if panel=='E1':
                blocks=[];kernels=forward.kernel_bank.reshape(256,-1)
                for i in range(0,len(sel),12):blocks.append(kernels[sel[i:i+12]].sum(axis=1,dtype=np.float32).reshape(-1,96,96))
                im=np.concatenate(blocks)
            else:
                im=forward.generate_variable_batch(sel,np.ones(sel.shape,dtype=np.float32))
            assert np.allclose(im.sum((1,2),dtype='float64'),64,atol=2e-4,rtol=0)
            images.append(im);features.append(extractor.extract_batch(im))
        ims=np.concatenate(images);hog=np.concatenate(features).astype('float64');truth=np.tile(ds.class_ids,5)
        fonts=np.tile(ds.font_ids,5);samples=np.tile(ds.sample_ids,5);seeds=np.repeat([s for s,_ in maps],len(ds.images))
        preds={}
        for method,proto in prototypes.items():
            p=np.argmax(hog@proto.T,axis=1)
            expected=np.array([old[(method,int(s),str(i))] for s,i in zip(seeds,samples)])
            nbad=int(np.sum(p!=expected))
            if nbad:raise RuntimeError(f'{panel} {method}: {nbad} old prediction mismatches; stop before RanDumb evaluation')
            preds[method]=p
        np.save(out/f'{panel}_images.npy',ims)
        np.save(out/f'{panel}_hog.npy',hog)
        np.savez_compressed(out/f'{panel}_metadata.npz',truth=truth,fonts=fonts,samples=samples,mapping_seeds=seeds,**preds)
        asset_manifest_path=E1/'data/e1_assets/manifest.json' if panel=='E1' else C1/'data/mma_confirmation/assets_manifest.json'
        original=json.loads(asset_manifest_path.read_text(encoding='utf-8'))
        manifest['panels'][panel]=dict(glyph_path=str(glyph),glyph_sha256=sha(glyph),font_hashes={r['font_id']:r['file_sha256'] for r in original['fonts']},asset_manifest_sha256=sha(asset_manifest_path),mapping_hashes={str(s):sha(p) for s,p in maps},old_predictions_sha256=sha(oldpred),n=len(ims),legacy_prediction_matches=True,image_file_sha256=sha(out/f'{panel}_images.npy'),selection_audit=audit,preparation_seconds=time.perf_counter()-start)
        print('EXTERNAL PREPARED AND HOG MATCH',panel,len(ims),flush=True)
    # Original source/asset manifests retained by hash for provenance.
    manifest['source_code_hashes']={str(p):sha(p) for p in (E1/'src').rglob('*.py')}
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
