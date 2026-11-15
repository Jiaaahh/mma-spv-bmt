import sys,json,time,gc,hashlib,itertools
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from core import ResidualOAS,embedder,transform,predict
from preflight import ROOT,CACHE,sha,write

MULT=[.25,1.,4.];RHOS=[1e-5,1e-4,1e-3]
def load_bank():
    meta=np.load(CACHE/'metadata.npz')
    names=['ideal']+[f'moderate_seed{s}' for s in range(2026,2030)]
    train=[np.load(CACHE/f'{n}_images.npy',mmap_mode='r').reshape(1080,-1) for n in names]
    val=[np.load(CACHE/f'moderate_seed{s}_images.npy',mmap_mode='r').reshape(1080,-1) for s in range(5301,5306)]
    return meta,train,val

def run_group(fold,font,tr,va,train,val,y,condition,mult,seed,rhos,dim,records):
    key=f'{condition}_f{fold}_a{mult}_s{seed}'
    path=ROOT/'development'/f'{key}.json'
    if path.exists():
        loaded=json.loads(path.read_text(encoding='utf-8'))
        if sorted(x['rho'] for x in loaded)!=sorted(rhos):raise ValueError('resume config mismatch')
        records.extend(loaded);return
    t=time.perf_counter();x0=np.array(train[0][tr]);xs=[a[tr] for a in (train if condition=='multi' else train[:1])]
    x=np.concatenate(xs);labels=np.tile(y[tr],len(xs));q=np.concatenate([a[va] for a in val]);truth=np.tile(y[va],5)
    obj=embedder(x0,dim,mult,seed);z=transform(obj,x);zq=transform(obj,q)
    model=ResidualOAS(z,labels);rows=[];preds={}
    for rho in rhos:
        w,b=model.solve(rho);pred=predict(zq,w,b);preds[str(rho)]=pred
        rows.append(dict(condition=condition,fold=fold,font=str(font),mult=mult,seed=seed,rho=rho,dim=dim,accuracy=float(np.mean(pred==truth)),gamma=obj.gamma,shrinkage=model.s,elapsed_group_seconds=time.perf_counter()-t))
    np.savez_compressed(path.with_suffix('.npz'),truth=truth,**preds)
    path.write_text(json.dumps(rows,indent=2),encoding='utf-8');records.extend(rows)
    print(key,'DONE',round(time.perf_counter()-t,1),flush=True)
    write('status.json',dict(stage='DEVELOPMENT',last_group=key,completed_fits=len(records)))

def rank(rows,condition):
    items=[]
    for a,rho in itertools.product(MULT,RHOS):
        selected=[r for r in rows if r['condition']==condition and r['mult']==a and r['rho']==rho]
        if selected:items.append(dict(mult=a,rho=rho,mean_accuracy=float(np.mean([r['accuracy'] for r in selected])),n=len(selected)))
    return sorted(items,key=lambda r:(-r['mean_accuracy'],abs(np.log(r['mult'])),abs(np.log(r['rho']/1e-4)),r['mult'],r['rho']))

def main():
    assert json.loads((ROOT/'correctness.json').read_text(encoding='utf-8'))['status']=='PASS'
    decision=json.loads((ROOT/'dimension_decision.json').read_text(encoding='utf-8'));dim=decision['dim']
    meta,train,val=load_bank();y=meta['labels'];fonts=meta['fonts'];order=np.unique(fonts)
    protocol=dict(dim=dim,multipliers=MULT,rhos=RHOS,stage1_seed=17,stage2_seeds=[29,43],final_seeds=[101,211,307,401,503],fold_order=order.tolist(),core_hash=sha(ROOT/'core.py'),runner_hash=sha(Path(__file__)),bank_hashes={str(CACHE/f'{n}_images.npy'):sha(CACHE/f'{n}_images.npy') for n in ['ideal']+[f'moderate_seed{s}' for s in list(range(2026,2030))+list(range(5301,5306))]},precision='float64',classifier='pooled residual OAS + legacy regularized linear scores; exact dual solve',upstream_commit='UNRESOLVED; web source inspected, local implementation hashes frozen; not claimed verbatim reproduction')
    p=ROOT/'development_protocol.json'
    if p.exists():assert json.loads(p.read_text(encoding='utf-8'))==protocol
    else:write(p.name,protocol)
    (ROOT/'development').mkdir(exist_ok=True)
    rows=[]
    for fi,font in enumerate(order):
        for condition in ['ideal','multi']:
            for a in MULT:run_group(fi,font,fonts!=font,fonts==font,train,val,y,condition,a,17,RHOS,dim,rows)
    top={c:rank(rows,c)[:2] for c in ['ideal','multi']};write('stage1_ranking.json',{c:rank(rows,c) for c in top})
    for fi,font in enumerate(order):
        for condition in ['ideal','multi']:
            for seed in [29,43]:
                for a in sorted(set(r['mult'] for r in top[condition])):
                    rhos=[r['rho'] for r in top[condition] if r['mult']==a]
                    run_group(fi,font,fonts!=font,fonts==font,train,val,y,condition,a,seed,rhos,dim,rows)
    finalists={}
    for c in top:
        selected=[r for r in rows if r['condition']==c and any(r['mult']==z['mult'] and r['rho']==z['rho'] for z in top[c])]
        finalists[c]=rank(selected,c)
        assert all(r['n']==18 for r in finalists[c])
    write('development_results.json',dict(rows=rows,finalists=finalists))
    write('selected_config.json',dict(status='FROZEN_BEFORE_NEW_EXTERNAL_PREDICTIONS',dim=dim,ideal=finalists['ideal'][0],multi=finalists['multi'][0],final_seeds=protocol['final_seeds'],protocol_hash=sha(p),core_hash=sha(ROOT/'core.py'),primary_panel='E1',secondary_panel='C1',shared_configuration='multi_selected_configuration',precision='float64'))
    write('status.json',dict(stage='DEVELOPMENT_COMPLETE',completed_fits=len(rows)))
    print('DEVELOPMENT COMPLETE',len(rows),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
