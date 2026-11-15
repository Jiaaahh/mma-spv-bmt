"""Post-run independent arithmetic audit; no parameter or protocol changes."""
from pathlib import Path
import json, csv, hashlib
import numpy as np
from threadpoolctl import threadpool_limits

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
CACHE=ROOT/'work/mma_supplement_cache'
EXT=ROOT/'outputs/randumb_cpu_20260928/external'
methods=['H0','H1','U0','U1','P0','P1']
meta=np.load(CACHE/'metadata.npz')
fonts,labels=meta['fonts'],meta['labels']
comparisons=json.loads((OUT/'paired_comparisons.json').read_text(encoding='utf-8'))
with (OUT/'predictions.csv').open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
checks=[]
with threadpool_limits(limits=2):
    for fold,font in list(enumerate(np.unique(fonts)))+[(6,None)]:
        tag=f'fold{fold}' if font is not None else 'full'
        mask=fonts!=font if font is not None else np.ones(len(fonts),bool)
        for multi in [0,1]:
            names=['ideal']+([f'moderate_seed{s}' for s in range(2026,2030)] if multi else [])
            x=np.concatenate([np.load(CACHE/(n+'_hog.npy'),mmap_mode='r')[mask] for n in names]).astype(np.float64)
            y=np.tile(labels[mask],len(names))
            unit=x/np.sqrt(np.sum(x*x,axis=1))[:,None]
            for prefix in ['H','U','P']:
                method=prefix+str(multi)
                source=x if prefix=='H' else unit
                means=np.array([np.sum(source[y==c],axis=0)/np.sum(y==c) for c in range(180)])
                if prefix=='P':
                    column=np.sqrt(np.sum(means*means,axis=0))
                    positive=column>0
                    means[:,positive]/=column[positive]
                    means[:,~positive]=0
                manual=means/np.sqrt(np.sum(means*means,axis=1))[:,None]
                saved=np.load(OUT/'prototypes'/f'{tag}_{method}.npy')
                error=float(np.max(np.abs(manual-saved)))
                assert error<1e-13,(tag,method,error)
                checkpoint=np.load(OUT/'checkpoints'/f'{tag}_{method}.npz')
                query_sets=[]
                if font is not None:
                    for i,s in enumerate(range(5301,5306)):
                        q=np.load(CACHE/f'moderate_seed{s}_hog.npy',mmap_mode='r')[fonts==font]
                        query_sets.append((q,checkpoint['DEV'][i*180:(i+1)*180]))
                else:
                    for panel in ['E1','C1']: query_sets.append((np.load(EXT/(panel+'_hog.npy'),mmap_mode='r'),checkpoint[panel]))
                count=0
                for q,expected in query_sets:
                    for start in range(0,len(q),128):
                        raw=np.asarray(q[start:start+128],dtype=np.float64)
                        # Positive query normalization cannot change argmax.
                        predicted=np.argmax(raw@manual.T,axis=1)
                        assert np.array_equal(predicted,expected[start:start+128])
                        count+=len(predicted)
                checks.append({'tag':tag,'method':method,'pooled_vs_saved_max_abs_error':error,'independent_predictions_matched':count})
        print('AUDITED',tag,flush=True)
    statistics=[]
    for i,panel in enumerate(['DEV','E1','C1']):
        nr=30 if panel=='DEV' else 20 if panel=='E1' else 10
        counts=np.zeros((180,6),dtype=np.int64)
        for r in rows:
            if r['panel']==panel:
                counts[int(r['class_id']),methods.index(r['method'])]+=int(r['class_id']==r['predicted'])
        diff=np.stack([counts[:,methods.index(c['comparison'].split('-')[0])]-counts[:,methods.index(c['comparison'].split('-')[1])] for c in comparisons[panel]],axis=1)
        rng=np.random.default_rng(26092831+i)
        boot=[]
        for start in range(0,100000,2000):
            w=rng.multinomial(180,np.full(180,1/180),size=2000)
            boot.append((w@diff)*(100/(180*nr)))
        boot=np.concatenate(boot)
        for j,c in enumerate(comparisons[panel]):
            ci=np.quantile(boot[:,j],[.025,.975]); point=float(diff[:,j].sum()*100/(180*nr))
            np.testing.assert_allclose(ci,c['ci95_pp'],atol=1e-10,rtol=0)
            assert abs(point-c['delta_pp'])<1e-10
            statistics.append({'panel':panel,'comparison':c['comparison'],'integer_count_bootstrap_matches':True})
    inventory=json.loads((ROOT/'outputs/proto_adapter_B_design_20260928/resource_inventory.json').read_text(encoding='utf-8'))
    unchanged=[]
    for item in inventory['files']:
        h=hashlib.sha256()
        with open(item['path'],'rb') as f:
            for b in iter(lambda:f.read(1048576),b''): h.update(b)
        assert h.hexdigest()==item['sha256']
        unchanged.append(item['path'])
result={'status':'PASS','prototype_checks':checks,'independent_predictions':sum(c['independent_predictions_matched'] for c in checks),'statistics':statistics,'18_source_files_unchanged':len(unchanged)==18,'scope':'post-run independent implementation audit; no new configurations'}
(OUT/'independent_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print('INDEPENDENT AUDIT PASS',result['independent_predictions'],flush=True)
