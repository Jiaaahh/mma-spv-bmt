"""Independent, read-only audit. Never imports or executes archive source code."""
from pathlib import Path
import io, json, zipfile, hashlib
import numpy as np

ROOT = Path('/historical/BaiduNetdiskDownload') / '\u4e0b\u8f7d'
OUT = Path(__file__).parent
ARCHIVE = ROOT/'SEND_point_spv_formal_20260925T093324Z.zip'

def sha(b):
    return hashlib.sha256(b).hexdigest()

def f1(y, p, n=180):
    cm = np.bincount(y*n+p, minlength=n*n).reshape(n,n)
    den = cm.sum(0)+cm.sum(1)
    return float(np.divide(2*np.diag(cm), den, out=np.zeros(n), where=den!=0).mean())

def boot(v, seed=26092515, n=100000):
    rng = np.random.default_rng(seed)
    draws = []
    for start in range(0,n,1000):
        counts=rng.multinomial(len(v),np.full(len(v),1/len(v)),size=min(1000,n-start))
        draws.append(counts @ v / len(v))
    return np.quantile(np.concatenate(draws),[.025,.975]).tolist()

def main():
    archive_bytes=ARCHIVE.read_bytes()
    outer=zipfile.ZipFile(io.BytesIO(archive_bytes))
    inner_bytes=outer.read('point_spv_formal_return.zip')
    z=zipfile.ZipFile(io.BytesIO(inner_bytes))
    old=json.loads(z.read('final_results/summary.json'))
    manifest=json.loads(z.read('return_manifest.json'))['files']
    hash_checks={n:sha(z.read(n))==v for n,v in manifest.items()}
    out={'archive':str(ARCHIVE),'archive_sha256':sha(archive_bytes),'nested_sha256':sha(inner_bytes),
         'all_return_manifest_hashes_match':all(hash_checks.values()),'return_manifest_checks':hash_checks,
         'numpy_version':np.__version__,'methods':{},'contrasts':{}}
    arrays={}
    reference=None
    for name in old['methods']:
        if name in ('point_ideal','point_multi'):continue
        raw=z.read('final_results/'+name+'.npz')
        with np.load(io.BytesIO(raw),allow_pickle=False) as a:
            y=a['labels'];p=a['predictions'];fonts=a['fonts']
            keys=['labels','fonts','rows','points','mapping_seeds']
            if reference is None:reference={k:a[k].copy()for k in keys}
            assert all(np.array_equal(a[k],reference[k]) for k in keys),name
            assert p.shape==(5,720) and y.shape==(720,)
            assert np.array_equal(np.unique(y),np.arange(180))
            assert len(np.unique(fonts))==4
            assert len(set(zip(fonts.tolist(),y.tolist())))==720
            assert (np.bincount(y,minlength=180)==4).all()
            score_key='scores' if 'scores' in a.files else 'logits'
            score=a[score_key]
            assert np.isfinite(score).all()
            assert np.array_equal(score.argmax(-1),p)
            assert ((p>=0)&(p<180)).all()
            correct=(p==y[None,:])
            # Group by explicit character label, keeping all fonts/mappings paired.
            chars=np.array([correct[:,y==c].mean() for c in range(180)])
            arrays[name]=chars
            perfont={str(f):float(correct[:,fonts==f].mean()*100)for f in np.unique(fonts)}
            audit={'npz_sha256':sha(raw),'correct':int(correct.sum()),'n':int(correct.size),
                   'accuracy_percent':float(correct.mean()*100),
                   'macro_f1_by_mapping_mean':float(np.mean([f1(y,p[m])for m in range(5)])),
                   'per_font_percent':perfont,'per_mapping_percent':(correct.mean(1)*100).tolist(),
                   'saved_predictions_match_argmax':True,'all_scores_finite':True,
                   'matches_archived_count':int(correct.sum())==old['methods'][name]['correct'],
                   'metadata_matches_all_methods':True}
            if 'stimulation' in a.files:
                stim=a['stimulation']
                audit.update(stimulation_binary=bool(np.isin(stim,[0,1]).all()),
                             exactly_64_active_every_view=bool((stim.sum(-1)==64).all()),
                             stimulation_independent_of_mapping=bool((stim==stim[0]).all()))
            else:
                selected=a['selected']
                audit['selected_64_unique_in_range']=bool(selected.shape==(720,64) and
                    ((selected>=0)&(selected<256)).all() and all(len(set(r.tolist()))==64 for r in selected))
            out['methods'][name]=audit
    for group in ('ideal','multi'):
        names=[f'{group}_seed{s}'for s in [17,29,43]]
        vals=[out['methods'][n]['accuracy_percent']for n in names]
        arrays['point_'+group]=np.mean([arrays[n]for n in names],axis=0)
        out['methods']['point_'+group]={'accuracy_percent':float(np.mean(vals)),
            'seed_sample_sd_pp':float(np.std(vals,ddof=1)),'seed_accuracy_percent':vals}
    contrasts={
        'primary_skeleton_mma_minus_point_multi':('skeleton_mma','point_multi'),
        'point_multi_minus_ideal':('point_multi','point_ideal'),
        'skeleton_mma_gain':('skeleton_mma','skeleton_ideal'),
        'canny_mma_gain':('canny_mma','canny_ideal')}
    for name,(a,b) in contrasts.items():
        v=(arrays[a]-arrays[b])*100
        ci=boot(v)
        out['contrasts'][name]={'difference_pp':float(v.mean()),'ci95_pp':ci,
            'archived_ci95_pp':old['contrasts'][name]['ci95_pp'],
            'max_ci_difference_pp':float(np.max(np.abs(np.array(ci)-old['contrasts'][name]['ci95_pp']))) }
    (OUT/'stat-point-recomputed.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    np.savez_compressed(OUT/'stat-point-character-scores.npz',**arrays)
    print(json.dumps({'hash_checks':out['all_return_manifest_hashes_match'],
        'methods':{k:{a:v[a]for a in ['correct','n','accuracy_percent','seed_sample_sd_pp']if a in v}for k,v in out['methods'].items()},
        'contrasts':out['contrasts']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
