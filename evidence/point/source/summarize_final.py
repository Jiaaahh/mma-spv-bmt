from core import *
def main():
    verify_manifest();p=json.loads((ROOT/'protocol.json').read_text());out=ROOT/'final_results';completion=json.loads((out/'completion.json').read_text());require(completion['completed'],'evaluation incomplete')
    vectors={};details={};ref=np.load(ROOT/'data/test/metadata.npz');views=np.load(ROOT/'data/test/views.npz')
    for name,h in completion['files'].items():
        path=out/name;require(sha(path)==h,'changed predictions');a=np.load(path);require(np.array_equal(a['labels'],ref['labels']) and np.array_equal(a['fonts'],ref['fonts']),'label/font alignment');require(np.array_equal(a['points'],views['points']) and np.array_equal(a['rows'],views['rows']),'view alignment')
        correct=(a['predictions']==a['labels'][None]);require(correct.shape==(5,720),'panel shape');key=path.stem
        vectors[key]=np.array([correct[:,a['labels']==c].mean() for c in range(180)])
        details[key]={'correct':int(correct.sum()),'n':3600,'accuracy_percent':100*float(correct.mean()),'per_font_percent':{str(f):100*float(correct[:,a['fonts']==f].mean()) for f in np.unique(a['fonts'])},'per_mapping_percent':(100*correct.mean(1)).tolist()}
    for condition in ['ideal','multi']:
        vv=np.stack([vectors[f'{condition}_seed{s}'] for s in p['initialization_seeds']]);vectors['point_'+condition]=vv.mean(0);details['point_'+condition]={'mean_accuracy_percent':100*float(vv.mean()),'seed_accuracy_percent':(100*vv.mean(1)).tolist(),'seed_sample_sd_pp':float((100*vv.mean(1)).std(ddof=1)),'ensemble':False}
    diffs={'primary_skeleton_mma_minus_point_multi':vectors['skeleton_mma']-vectors['point_multi'],'point_multi_minus_ideal':vectors['point_multi']-vectors['point_ideal'],'skeleton_mma_gain':vectors['skeleton_mma']-vectors['skeleton_ideal'],'canny_mma_gain':vectors['canny_mma']-vectors['canny_ideal']}
    rng=np.random.default_rng(p['statistics']['seed']);boot={k:[] for k in diffs}
    for _ in range(100):
        weights=rng.multinomial(180,np.full(180,1/180),size=1000)/180
        for k,v in diffs.items():boot[k].extend((weights@v*100).tolist())
    report={'status':'complete_fixed_panel','methods':details,'contrasts':{k:{'difference_pp':float(v.mean()*100),'ci95_pp':np.quantile(boot[k],[.025,.975]).tolist()} for k,v in diffs.items()},'statistics':p['statistics'],'scope':p['scope']};save(out/'summary.json',report);print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
