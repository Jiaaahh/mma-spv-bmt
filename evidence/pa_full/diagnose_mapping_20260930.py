"""Post-hoc TRAINING-only diagnostics; mapping-matched condition is an oracle, not a baseline."""
from pathlib import Path
import json,time
import numpy as np
from threadpoolctl import threadpool_limits
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
meta=np.load(ROOT/'work/mma_supplement_cache/metadata.npz');y=meta['labels'];fonts=meta['fonts']
names=['ideal']+[f'moderate_seed{s}' for s in range(2026,2030)]
def norm(x):return x/np.linalg.norm(x,axis=1,keepdims=True)
def fit(x,labels,col):
    p=np.stack([x[labels==c].mean(0) for c in range(180)])
    if col:
        d=np.linalg.norm(p,axis=0,keepdims=True);p=np.divide(p,d,out=np.zeros_like(p),where=d>0)
    return norm(p)
start=time.perf_counter();results=[]
with threadpool_limits(limits=2):
    bank=np.stack([norm(np.load(OUT/'features'/f'{n}.npy').astype('float64')) for n in names])
    g=bank.mean((0,1));total=float(((bank-g)**2).sum())
    mapping=float((1080*(bank.mean(1)-g)**2).sum())
    cls=np.stack([bank[:,y==c,:].mean((0,1)) for c in range(180)])
    class_ss=float((30*(cls-g)**2).sum())
    ff=np.stack([bank[:,fonts==f,:].mean((0,1)) for f in np.unique(fonts)])
    font_ss=float((900*(ff-g)**2).sum())
    for mi,name in enumerate(names):
        counts={k:0 for k in ['same_map_PA_visual','pooled_PA_visual','same_map_plain','pooled_plain']}
        for f in np.unique(fonts):
            tr=fonts!=f;va=fonts==f;q=bank[mi,va]
            mixed=bank[:,tr].reshape(-1,1024);labels=np.tile(y[tr],5)
            for col,suffix in [(True,'PA_visual'),(False,'plain')]:
                p=fit(bank[mi,tr],y[tr],col);pm=fit(mixed,labels,col)
                counts['same_map_'+suffix]+=int(np.sum((q@p.T).argmax(1)==y[va]))
                counts['pooled_'+suffix]+=int(np.sum((q@pm.T).argmax(1)==y[va]))
        results.append(dict(mapping=name,n=1080,accuracy_percent={k:100*v/1080 for k,v in counts.items()}))
result=dict(status='COMPLETE',scope='Post-hoc diagnosis on five training mappings only; leave-one-font-out; no E1/C1 or DEV530x queried',
    oracle_warning='same_map uses known query mapping to select support; unavailable in main target protocol and must not replace baseline',
    results=results,mean_accuracy_percent={k:float(np.mean([r['accuracy_percent'][k] for r in results])) for k in counts},
    balanced_feature_variance_percent=dict(mapping_main_effect=100*mapping/total,class_main_effect=100*class_ss/total,font_main_effect=100*font_ss/total,residual_and_interactions=100*(total-mapping-class_ss-font_ss)/total),
    interpretation='Descriptive feature geometry and controlled pooling diagnostic, not causal proof or a new tuned method',seconds=time.perf_counter()-start)
(OUT/'mapping_diagnosis_20260930.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
