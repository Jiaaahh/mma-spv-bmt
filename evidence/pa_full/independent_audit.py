"""Recompute every prototype and prediction without importing evaluate.py."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
PRE=ROOT/'outputs/proto_adapter_full_preflight_20260929'; CACHE=ROOT/'work/mma_supplement_cache';EXT=ROOT/'outputs/randumb_cpu_20260928/external'
def unit(a):
    a=np.asarray(a,np.float64);return a/np.sqrt(np.einsum('ij,ij->i',a,a))[:,None]
def prototype(a,y,col):
    a=unit(a);p=np.zeros((180,a.shape[1])); counts=np.bincount(y,minlength=180)
    np.add.at(p,y,a);p/=counts[:,None]
    if col:
        d=np.sqrt(np.einsum('ij,ij->j',p,p));p[:,d>0]/=d[d>0];p[:,d==0]=0
    return unit(p)
meta=np.load(CACHE/'metadata.npz');y=meta['labels'];fonts=meta['fonts']
protocol=json.loads((PRE/'frozen_protocol.json').read_text(encoding='utf-8'));names=protocol['train_maps']
selection=json.loads((OUT/'selected_alpha.json').read_text())['methods']
rows=list(csv.DictReader((OUT/'predictions.csv').open(encoding='utf-8-sig')))
lookup={(r['panel'],r['method'],r['font'],int(r['mapping_seed']),int(r['class_id'])):int(r['predicted']) for r in rows}
assert len(lookup)==len(rows)==97200
text=unit(np.load(PRE/'text_features.npy'));checks=[];total=0;ties=0
def check(panel,method,pred,fs,seeds,truth):
    global total
    reference=np.array([lookup[(panel,method,str(f),int(s),int(c))] for f,s,c in zip(fs,seeds,truth)])
    assert np.array_equal(pred,reference),(panel,method,int((pred!=reference).sum()))
    total+=len(pred)
with threadpool_limits(limits=2):
    for fi,f in list(enumerate(np.unique(fonts)))+[(6,None)]:
        mask=fonts!=f if f is not None else np.ones(len(y),bool);tag=f'fold{fi}' if f is not None else 'full'
        queries=[]
        if f is not None:
            va=fonts==f
            for name in protocol['dev_maps']:
                queries.append(('DEV',unit(np.load(OUT/'features'/f'{name}.npy')[va]),fonts[va],np.full(180,int(name[-4:])),y[va],np.load(CACHE/f'{name}_hog.npy',mmap_mode='r')[va]))
        else:
            for panel in ['E1','C1']:
                m=np.load(EXT/f'{panel}_metadata.npz'); queries.append((panel,unit(np.load(OUT/'features'/f'{panel}.npy')),m['fonts'],m['mapping_seeds'],m['truth'],np.load(EXT/f'{panel}_hog.npy',mmap_mode='r')))
        for multi in [0,1]:
            maps=names if multi else names[:1];sy=np.tile(y[mask],len(maps))
            a=np.concatenate([np.load(OUT/'features'/f'{n}.npy')[mask] for n in maps]);p=prototype(a,sy,True);u=prototype(a,sy,False)
            for label,pr in [(f'PA{multi}',p),(f'PLAIN{multi}',u)]:
                saved=np.load(OUT/'prototypes'/f'{tag}_{label}.npy');err=float(abs(saved-pr).max());assert err<1e-12
                checks.append(dict(prototype=tag+'_'+label,max_abs_error=err))
            # Independently reconstruct the old MMA/ideal raw-HOG class means.
            a=np.concatenate([np.load(CACHE/f'{n}_hog.npy',mmap_mode='r')[mask] for n in maps]).astype(np.float64)
            h=np.zeros((180,a.shape[1])); np.add.at(h,sy,a); h/=np.bincount(sy)[:,None];h=unit(h)
            for panel,q,fs,seeds,truth,hog in queries:
                v=q@p.T;t=q@text.T;s=selection[f'PA{multi}']['alpha']*v+t
                for label,score in [(f'PA{multi}',s),(f'VIS{multi}',v),(f'PLAIN{multi}',q@u.T)]:
                    check(panel,label,score.argmax(1),fs,seeds,truth)
                    ties+=int(np.sum(np.sum(score==score.max(1,keepdims=True),axis=1)>1))
                hp=[]
                for i in range(0,len(hog),256):hp.extend((unit(hog[i:i+256])@h.T).argmax(1))
                check(panel,f'H{multi}',np.array(hp),fs,seeds,truth)
                if multi==0:check(panel,'TEXT',t.argmax(1),fs,seeds,truth)
    # Verify development selection globally, including tie handling.
    search=list(csv.DictReader((OUT/'development_search.csv').open(encoding='utf-8-sig')))
    for method in ['PA0','PA1']:
        scores=np.load(OUT/f'dev_scores_{method}.npz');subset=[r for r in search if r['method']==method]
        for r in subset:
            correct=int(np.sum(np.argmax(float(r['alpha'])*scores['visual']+scores['text'],axis=1)==scores['truth']))
            assert correct==int(r['correct'])
        best=sorted(subset,key=lambda r:(-int(r['correct']),float(r['alpha'])))[0]
        assert float(best['alpha'])==selection[method]['alpha']
    metrics=list(csv.DictReader((OUT/'metrics.csv').open(encoding='utf-8-sig')))
    for m in metrics:
        ss=[r for r in rows if r['panel']==m['panel'] and r['method']==m['method']]
        correct=sum(r['class_id']==r['predicted'] for r in ss)
        assert correct==int(m['correct']) and len(ss)==int(m['n']) and abs(correct/len(ss)*100-float(m['accuracy_percent']))<1e-10
    comparisons=json.loads((OUT/'paired_comparisons.json').read_text());methods=json.loads((OUT/'completion_audit.json').read_text())['methods']
    for panel in ['E1','C1']:
        arr=np.load(OUT/f'{panel}_bootstrap.npz');ca=arr['character_accuracy'];draws=arr['paired_deltas']
        assert draws.shape==(100000,5)
        for j,r in enumerate(comparisons[panel]):
            a,b=r['comparison'].split('-'); diff=100*(ca[:,methods.index(a)]-ca[:,methods.index(b)])
            assert abs(diff.mean()-r['delta_pp'])<1e-10
            assert np.allclose(np.quantile(draws[:,j],[.025,.975]),r['ci95_pp'],atol=1e-10)
        weights=np.random.default_rng(protocol['bootstrap'][panel+'_seed']).multinomial(180,np.full(180,1/180),size=3)
        for i in range(3):
            for j,r in enumerate(comparisons[panel]):
                a,b=r['comparison'].split('-');v=sum(weights[i,c]*100*(ca[c,methods.index(a)]-ca[c,methods.index(b)]) for c in range(180))/180
                assert abs(v-draws[i,j])<1e-10
assert total==97200
result=dict(status='PASS',independently_recomputed_predictions=total,prototypes=checks,baseline_raw_HOG_replayed=True,dev_selection_rechecked=True,aggregate_counts_and_CIs_rechecked=True,exact_score_ties_in_PA_visual_plain=ties)
(OUT/'independent_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='prototypes'}))
