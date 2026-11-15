import json,csv,time
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
from preflight import ROOT,sha,write

def main():
    cfg=json.loads((ROOT/'selected_config.json').read_text(encoding='utf-8'));ext=ROOT/'external';ev=ROOT/'evaluation'
    rng=np.random.default_rng(20260928)
    weights=rng.multinomial(180,np.full(180,1/180),size=100000)/180
    summaries={};rows=[];metrics=[];subgroups=[]
    manifest=json.loads((ext/'manifest.json').read_text(encoding='utf-8'))
    for panel in ['E1','C1']:
        meta=np.load(ext/f'{panel}_metadata.npz');truth=meta['truth'];fonts=meta['fonts'];maps=meta['mapping_seeds'];samples=meta['samples']
        preds={m:meta[m][None,:] for m in ['H0','H1']}
        for m in ['R0','R1','R0repeat5','R0shared']:
            preds[m]=np.stack([np.load(ev/f'{m}_{s}_{panel}_pred.npy') for s in cfg['final_seeds']])
        correct={m:(p==truth[None,:]).astype(float) for m,p in preds.items()}
        summary={}
        for method,pred in preds.items():
            seedacc=correct[method].mean(axis=1)
            f1=[]
            for j,p in enumerate(pred):
                seed='fixed' if method.startswith('H') else cfg['final_seeds'][j]
                f1.append(float(np.mean([f1_score(truth[maps==s],p[maps==s],labels=np.arange(180),average='macro',zero_division=0) for s in np.unique(maps)])))
                for k,v in enumerate(p):rows.append(dict(panel=panel,method=method,projection_seed=seed,sample_id=str(samples[k]),font_id=str(fonts[k]),font_sha256=manifest['panels'][panel]['font_hashes'][str(fonts[k])],input_bank_sha256=manifest['panels'][panel]['image_file_sha256'],mapping_seed=int(maps[k]),class_id=int(truth[k]),predicted_class=int(v),correct=int(v==truth[k]),config_hash=sha(ROOT/'selected_config.json')))
            summary[method]=dict(accuracy_percent=float(seedacc.mean()*100),seed_accuracy_percent=(seedacc*100).tolist(),seed_sd_percent=float(seedacc.std(ddof=1)*100) if len(seedacc)>1 else None,macro_f1=float(np.mean(f1)))
            metrics.append(dict(panel=panel,method=method,accuracy_percent=summary[method]['accuracy_percent'],seed_sd_percent=summary[method]['seed_sd_percent'],macro_f1=summary[method]['macro_f1']))
            for font in np.unique(fonts):
                for s in np.unique(maps):subgroups.append(dict(panel=panel,method=method,font_id=str(font),mapping_seed=int(s),accuracy_percent=float(correct[method][:,(fonts==font)&(maps==s)].mean()*100)))
        comparisons=[]
        for a,b in [('H1','R1'),('H1','H0'),('R1','R0'),('R1','R0shared'),('R0repeat5','R0'),('H0','R0')]:
            d=(correct[a].mean(0)-correct[b].mean(0))
            perclass=np.array([d[truth==c].mean() for c in range(180)])
            ci=np.quantile(weights@perclass,[.025,.975])*100
            comparisons.append(dict(a=a,b=b,difference_pp=float(d.mean()*100),conditional95ci_pp=ci.tolist(),primary=(panel=='E1' and a=='H1' and b=='R1')))
        summaries[panel]=dict(methods=summary,comparisons=comparisons)
    for name,data in [('predictions.csv',rows),('metrics.csv',metrics),('font_mapping_metrics.csv',subgroups)]:
        with (ROOT/name).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    write('results_summary.json',dict(status='COMPLETED_ACCURACY',panels=summaries,bootstrap=dict(n=100000,seed=20260928,unit='character',conditional_on='fixed fonts mappings and projection seeds'),historical_panel_caution=True))
    lines=['# RanDumb-SPV 实际结果','', '本次为历史固定面板的补充比较；不是新增盲测。随机种子结果按准确率平均，不是模型集成。','', '|面板|方法|准确率%|种子标准差/百分点|宏F1|','|---|---|---:|---:|---:|']
    for r in metrics:lines.append(f"|{r['panel']}|{r['method']}|{r['accuracy_percent']:.2f}|{r['seed_sd_percent'] if r['seed_sd_percent'] is not None else '确定性'}|{r['macro_f1']:.4f}|")
    lines+=['','H0=理想HOG，H1=MMA-HOG，R0=理想RanDumb，R1=多映射RanDumb，R0repeat5=重复理想样本诊断，R0shared=使用R1所选配置但只拟合理想图像。','','|面板|对比|差值/百分点|条件95%区间|','|---|---|---:|---|']
    for panel,s in summaries.items():
        for c in s['comparisons']:lines.append(f"|{panel}|{c['a']} − {c['b']}|{c['difference_pp']:.2f}|[{c['conditional95ci_pp'][0]:.2f}, {c['conditional95ci_pp'][1]:.2f}]|")
    lines+=['','只有E1的H1−R1为本轮主要比较；其余是未校正的次要/探索性对比。区间条件于有限字体、映射和5次随机投影，不代表任意患者或字体总体。','', '代码采用明确记录的任务适配和float64低秩求解；不能把结果称为对原持续学习榜单的完整复现。Point-SPV与XDomainMix原有结果不被替换。','', '配置、源数据与源码散列见development_protocol.json、selected_config.json、evaluation_protocol.json、external/manifest.json；成本见costs.json（生成后可用）。']
    (ROOT/'实验结果.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({p:s['comparisons'][0] for p,s in summaries.items()}),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
