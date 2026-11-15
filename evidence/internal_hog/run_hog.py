"""Paired six-font LOFO, five new DEVELOPMENT mappings, five training sets."""
import sys
sys.dont_write_bytecode=True
import csv
import time
import numpy as np
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
from common import ROOT,CACHE,PLAN,PLAN_PATH,sha,save_json,check_cache,normalize,class_means,validate_split,mapping_name

def main():
    check_cache()
    out=ROOT/'results/hog';out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():raise RuntimeError('completed run exists; do not overwrite')
    meta=np.load(CACHE/'metadata.npz');labels=meta['labels'];fonts=meta['fonts'];font_order=np.unique(fonts)
    groups=PLAN['training_mapping_groups'];eval_seeds=PLAN['development_evaluation_mapping_seeds']
    names=['ideal',*[mapping_name(s) for g in groups for s in g],*[mapping_name(s) for s in eval_seeds],*[f'affine{k}' for k in range(4)]]
    bank={n:np.load(CACHE/f'{n}_hog.npy',mmap_mode='r') for n in set(names)}
    results={};fold_rows=[];prediction_rows=[]
    start=time.perf_counter()
    for fi,font in enumerate(font_order):
        tr,va=validate_split(fonts,font,[s for g in groups for s in g],eval_seeds)
        means={n:class_means(x[tr],labels[tr]) for n,x in bank.items() if n not in [mapping_name(s) for s in eval_seeds]}
        models={'ideal_only':('mean',[means['ideal']]),'ordinary_affine4':('mean',[means['ideal'],*[means[f'affine{k}'] for k in range(4)]])}
        for gi,g in enumerate(groups):
            reps=[means['ideal'],*[means[mapping_name(s)] for s in g]]
            models[f'mma_g{gi}']=('mean',reps)
            models[f'ensemble_mean_g{gi}']=('score_mean',reps)
            models[f'ensemble_max_g{gi}']=('score_max',reps)
            for s in g:models[f'ideal_single_{s}']=('mean',[means['ideal'],means[mapping_name(s)]])
        for mi,s in enumerate(eval_seeds):
            x=normalize(bank[mapping_name(s)][va]);truth=labels[va]
            for name,(mode,reps) in models.items():
                if mode=='mean':scores=x@normalize(np.stack(reps).mean(0)).T
                else:
                    all_scores=np.stack([x@normalize(r).T for r in reps])
                    scores=all_scores.mean(0) if mode=='score_mean' else all_scores.max(0)
                pred=np.argmax(scores,axis=1)
                if name not in results:results[name]=np.zeros((6,5,180),dtype=np.int64)
                results[name][fi,mi,truth]=pred
                fold_rows.append({'method':name,'font':font,'mapping_seed':s,'accuracy':float((pred==truth).mean()),'macro_f1':float(f1_score(truth,pred,labels=np.arange(180),average='macro',zero_division=0))})
                prediction_rows.extend({'method':name,'font':font,'mapping_seed':s,'class_id':int(c),'predicted':int(p)} for c,p in zip(truth,pred))
        print(f'HOG font {fi+1}/6 complete',flush=True)
    for filename,rows in [('fold_metrics.csv',fold_rows),('predictions.csv',prediction_rows)]:
        with (out/filename).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    np.savez_compressed(out/'predictions.npz',**results)
    rng=np.random.default_rng(PLAN['bootstrap_seed'])
    # Same character bootstrap weights used for every paired comparison.
    weights=rng.multinomial(180,np.full(180,1/180),size=PLAN['bootstrap_replicates'])/180
    correct={k:(v==np.arange(180)).astype(float) for k,v in results.items()}
    def compare(a,b):
        d=(correct[a]-correct[b]).mean((0,1));ci=np.quantile(weights@d,[.025,.975])*100
        return {'a':a,'b':b,'difference_pp':float(d.mean()*100),'character_conditional_95ci_pp':ci.tolist()}
    comparisons=[compare('mma_g0','ideal_only'),compare('mma_g0','ordinary_affine4'),compare('ensemble_mean_g0','mma_g0'),compare('ensemble_max_g0','mma_g0')]
    comparisons += [compare('mma_g0',f'ideal_single_{s}') for s in groups[0]]
    comparisons += [compare(f'mma_g{g}','ideal_only') for g in range(1,5)]
    summary={'status':'COMPLETED_DEVELOPMENT_ONLY','plan_sha256':sha(PLAN_PATH),'methods':len(results),'folds_per_method':30,'predictions_per_method':5400,'elapsed_seconds':time.perf_counter()-start,'accuracies':{k:float(v.mean()*100) for k,v in correct.items()},'comparisons':comparisons,'training_group_mma_accuracy':[float(correct[f'mma_g{i}'].mean()*100) for i in range(5)],'caution':'Development LOFO; fixed finite panel; CIs unadjusted and exploratory. No best-group selection. Generic affine is image-level, not physical mapping randomization.'}
    save_json(out/'summary.json',summary)
    print(__import__('json').dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
