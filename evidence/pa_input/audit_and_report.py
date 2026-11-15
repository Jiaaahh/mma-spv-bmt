from pathlib import Path
import csv,json,hashlib
import numpy as np
from threadpoolctl import threadpool_limits
OUT=Path(__file__).resolve().parent;OLD=OUT.parent/'proto_adapter_full_run_20260929';PRE=OUT.parent/'proto_adapter_full_preflight_20260929'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def unit(x):
    x=np.asarray(x,np.float64);return x/np.sqrt(np.einsum('ij,ij->i',x,x))[:,None]
cfg=read(OUT/'protocol.json');sub=read(OUT/'subset.json');manifest=read(OUT/'implementation_manifest.json');res=read(OUT/'result.json');decision=read(OUT/'selection_decision.json')
assert res['status']=='COMPLETE'
for name,k in [('protocol.json','protocol_sha256'),('run_screen.py','script_sha256'),('subset.json','subset_sha256')]:assert sha(OUT/name)==manifest[k]
rows=list(csv.DictReader((OUT/'predictions.csv').open(encoding='utf-8')))
lookup={(r['variant'],r['panel'],r['method'],r['font'],r['mapping'],r['sample']):int(r['predicted']) for r in rows};assert len(lookup)==len(rows)==37800
y=np.array(sub['new_labels']);fonts=np.array(sub['fonts']);samples=np.array(sub['samples']);ix=np.array(sub['source_rows']);classes=np.array(sub['original_class_ids'])
text=unit(np.load(PRE/'text_features.npy')[classes]);total=0;names=cfg['train_maps']+cfg['selection_maps']+cfg['confirmation_maps']
with threadpool_limits(limits=2):
    for variant in cfg['variants']:
        bank={}
        for name in names:
            path=OLD/'features'/f'{name}.npy' if variant=='original' else OUT/'features'/f'{variant}_{name}.npy'
            info=read(path.with_suffix('.json'));assert sha(path)==info['feature_sha256' if variant=='original' else 'sha256']
            z=np.load(path);bank[name]=unit(z[ix] if variant=='original' else z)
        for f in np.unique(fonts):
            tr=fonts!=f;va=fonts==f;ps={}
            for multi in [0,1]:
                maps=cfg['train_maps'] if multi else cfg['train_maps'][:1];z=np.concatenate([bank[n][tr] for n in maps]);labels=np.tile(y[tr],len(maps));p=np.zeros((60,1024));np.add.at(p,labels,unit(z));p/=np.bincount(labels)[:,None];ps[f'PLAIN{multi}']=unit(p)
                d=np.sqrt((p*p).sum(0));p[:,d>0]/=d[d>0];p[:,d==0]=0;ps[f'VIS{multi}']=unit(p)
            for panel,maps in [('selection',cfg['selection_maps']),('confirmation',cfg['confirmation_maps'])]:
                for name in maps:
                    q=bank[name][va];scores={'TEXT':q@text.T}
                    for k,p in ps.items():scores[k]=q@p.T
                    for multi in [0,1]:
                        alpha=next(r['alpha'] for r in decision['alphas'] if r['variant']==variant and r['method']==f'PA{multi}')
                        scores[f'PA{multi}']=alpha*scores[f'VIS{multi}']+scores['TEXT']
                    for method,s in scores.items():
                        expect=np.array([lookup[variant,panel,method,str(f),name,str(sample)] for sample in samples[va]])
                        assert np.array_equal(s.argmax(1),expect),(variant,panel,method,name,f);total+=len(expect)
    searches=read(OUT/'selection_search.json')
    for r in decision['alphas']:
        candidates=[v for v in searches if v['variant']==r['variant'] and v['method']==r['method']]
        score=np.load(OUT/f"{r['variant']}_selection_scores.npz")
        vis=score['VIS'+r['method'][-1]];txt=score['TEXT'];truth=score['truth']
        for candidate in candidates:
            correct=int(np.sum((candidate['alpha']*vis+txt).argmax(1)==truth));assert correct==candidate['correct']
        best=min(candidates,key=lambda v:(-v['correct'],v['alpha']));assert best['alpha']==r['alpha'] and best['correct']==r['correct']
    ranked=[r for r in decision['alphas'] if r['method']=='PA1']
    best_display=min(ranked,key=lambda r:(-r['correct'],cfg['variants'].index(r['variant'])))['variant']
    assert best_display==decision['selected_variant']==res['selected_variant']
    for m in res['metrics']:
        rr=[r for r in rows if r['variant']==m['variant'] and r['panel']==m['panel'] and r['method']==m['method']]
        assert len(rr)==m['n'] and sum(r['truth']==r['predicted'] for r in rr)==m['correct']
assert total==37800
audit=dict(status='PASS',independent_predictions=total,feature_hashes=True,protocol_hashes=True,selection_optimum_rechecked=True,external_panels_accessed=False,input_preview='Viewed: inversion preserves spatial structure; sqrt boosts dim phosphenes without cropping or mirroring.')
(OUT/'independent_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
labels={'original':'原黑底显示','inverted':'白底反色','sqrt_brightened':'暗光点增强（平方根）'}
lines=['# SPV输入适配小规模筛查','',
    '**这是60类开发数据诊断，不是180类主实验，也不是E1/C1测试结果。** 保留六字体、五个训练mapping；三个开发mapping用于选择显示方案和alpha，另两个用于检查改善是否保留。后一部分也是历史开发数据，不宣称新盲测。','',
    '候选只有原显示、反色、固定gamma=0.5的平方根亮度变换。RN50、七模板和Proto-Adapter核心不变。反色是输入显示适配，不能被描述为真实假体产生的光幻视反色。','',
    '| 输入 | PA1选参部分/% | PA1确认部分/% | PA0确认部分/% | 纯视觉PA1确认/% | 普通原型multi确认/% |','|---|---:|---:|---:|---:|---:|']
def metric(v,p,m):return next(r['accuracy_percent'] for r in res['metrics'] if r['variant']==v and r['panel']==p and r['method']==m)
for v in cfg['variants']:
    vals=[metric(v,'selection','PA1'),metric(v,'confirmation','PA1'),metric(v,'confirmation','PA0'),metric(v,'confirmation','VIS1'),metric(v,'confirmation','PLAIN1')]
    lines.append('| '+labels[v]+' | '+' | '.join(f'{x:.2f}' for x in vals)+' |')
winner=res['selected_variant'];lines+=['',f"仅根据选参部分固定的显示方案：{labels[winner]}。相对原显示，PA1选参部分变化{res['selection_gain_pp']:+.2f}个百分点，确认部分变化{res['confirmation_gain_pp']:+.2f}个百分点。没有根据确认结果更改获选方案。",'']
if res['recommend_full_180_DEV_verification']:
    lines+=['预设继续门槛已达到：两个开发部分均至少提升5个百分点、确认准确率至少20%。值得下一步验证完整180类开发集，但本轮没有重测E1/C1，不能将改善直接外推为主实验结果。']
else:
    lines+=['预设继续门槛未达到。当前没有足够证据支持仅凭这轮显示转换就启动完整180类外部重跑。可以保留这轮诊断并暂时结束此输入分支；不能通过继续增加候选直到获得满意结果来替代固定方案。']
lines+=['','## 边界与复核','',
    '60类筛查的类别难度和跨类别通道归一化均不同于180类，绝对准确率不得与此前180类表格直接比较。开发选择及确认属于事后研究，不证明因果或发表价值；预设门槛只是控制计算投入的规则，不是统计显著性标准。',
    '', '全部37800条预测已用独立原型累加与评分重算，输入/特征/协议哈希和选参最优值核对通过。全部候选、alpha搜索、预测和缓存保留，未访问E1/C1数据，未修改论文。',
    '',f"本次进程耗时约{res['seconds_this_process']/60:.2f}分钟；新增完成缓存记录的编码合计{sum(r['seconds'] for r in res['cache_groups'])/60:.2f}分钟。候选名单、子集与划分详见protocol.json和subset.json。"]
(OUT/'输入适配筛查结论.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
files={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='delivery_manifest.json'}
(OUT/'delivery_manifest.json').write_text(json.dumps(dict(status='COMPLETE',files=files),ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(status='PASS',independent_predictions=total,selected_variant=winner,recommend_full_180_DEV_verification=res['recommend_full_180_DEV_verification'])))
