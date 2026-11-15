"""Check completed artifacts independently of the summary writer."""
import csv,json,hashlib
from collections import Counter
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from preflight import ROOT,CACHE,sha,write
from core import embedder,transform,predict

def main():
    cfg=json.loads((ROOT/'selected_config.json').read_text(encoding='utf-8'))
    protocol=json.loads((ROOT/'development_protocol.json').read_text(encoding='utf-8'))
    assert sha(ROOT/'core.py')==protocol['core_hash']
    assert sha(ROOT/'develop.py')==protocol['runner_hash']
    dev=json.loads((ROOT/'development_results.json').read_text(encoding='utf-8'))
    assert len(dev['rows'])==156
    assert len({(r['condition'],r['fold'],r['mult'],r['rho'],r['seed']) for r in dev['rows']})==156
    for p,h in protocol['bank_hashes'].items():assert sha(p)==h
    summary=json.loads((ROOT/'results_summary.json').read_text(encoding='utf-8'))
    counts=Counter();keys=set();correct=Counter()
    with (ROOT/'predictions.csv').open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            key=(r['panel'],r['method'],r['projection_seed'],r['sample_id'],r['mapping_seed'])
            assert key not in keys;keys.add(key)
            k=(r['panel'],r['method']);counts[k]+=1;correct[k]+=int(r['correct'])
            assert int(r['correct'])==int(r['class_id']==r['predicted_class'])
    assert len(keys)==118800
    for panel,n in [('E1',3600),('C1',1800)]:
        for method in ['H0','H1','R0','R1','R0repeat5','R0shared']:
            assert counts[panel,method]==n*(1 if method.startswith('H') else 5)
            assert abs(correct[panel,method]/counts[panel,method]*100-summary['panels'][panel]['methods'][method]['accuracy_percent'])<1e-10
    # Reloaded model check, independent query batching and serialisation.
    x0=np.load(CACHE/'ideal_images.npy',mmap_mode='r').reshape(1080,-1)
    checks=[]
    for seed in cfg['final_seeds']:
        for method,condition in [('R0','ideal'),('R1','multi'),('R0repeat5','ideal'),('R0shared','multi')]:
            obj=embedder(x0,cfg['dim'],cfg[condition]['mult'],seed)
            model=np.load(ROOT/f'models/{method}_{seed}.npz')
            for panel in ['E1','C1']:
                q=np.load(ROOT/f'external/{panel}_images.npy',mmap_mode='r').reshape(-1,9216)
                ix=np.array([0,7,179,len(q)-1]);z=transform(obj,q[ix],batch=1)
                p=predict(z,model['w'],model['b']);stored=np.load(ROOT/f'evaluation/{method}_{seed}_{panel}_pred.npy')
                assert np.array_equal(p,stored[ix])
                checks.append(dict(method=method,seed=seed,panel=panel,matched=4))
            del obj,model
    for seed in cfg['final_seeds']:
        if cfg['ideal']['mult']==cfg['multi']['mult'] and cfg['ideal']['rho']==cfg['multi']['rho']:
            for panel in ['E1','C1']:
                assert np.array_equal(np.load(ROOT/f'evaluation/R0_{seed}_{panel}_pred.npy'),np.load(ROOT/f'evaluation/R0shared_{seed}_{panel}_pred.npy'))
    costs=json.loads((ROOT/'costs.json').read_text(encoding='utf-8'))
    for r in costs['methods'].values():
        for b in ['1','32']:
            assert len(r['inference'][b]['raw_seconds'])==100
            assert min(r['inference'][b]['raw_seconds'])>0
    write('completion_audit.json',dict(status='PASS',development_fits=156,prediction_rows=118800,serialization_and_batch_checks=checks,config_hash=sha(ROOT/'selected_config.json'),accuracy_result_hash=sha(ROOT/'results_summary.json'),cost_result_hash=sha(ROOT/'costs.json'),source_files_sha256={p.name:sha(p) for p in ROOT.glob('*.py')}))
    print('COMPLETION AUDIT PASS: 156 fits; 118800 unique predictions; reloaded models match',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
