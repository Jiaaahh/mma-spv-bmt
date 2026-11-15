from pathlib import Path
import csv,json,hashlib
OUT=Path(__file__).resolve().parent;PRE=OUT.parent/'proto_adapter_full_preflight_20260929'
def read(n):return json.loads((OUT/n).read_text(encoding='utf-8'))
assert read('independent_audit.json')['status']=='PASS'
rows=list(csv.DictReader((OUT/'metrics.csv').open(encoding='utf-8-sig')))
metric={(r['panel'],r['method']):r for r in rows};comps=read('paired_comparisons.json');sel=read('selected_alpha.json')['methods']
near_chance=all(float(metric[p,'PA1']['accuracy_percent'])<5 for p in ['E1','C1'])
labels={'H0':'HOG ideal','H1':'MMA-HOG','PA0':'Proto-Adapter ideal','PA1':'Proto-Adapter multi','TEXT':'CLIP 文本分支','VIS0':'PA ideal 纯视觉','VIS1':'PA multi 纯视觉','PLAIN0':'CLIP ideal 普通类原型','PLAIN1':'CLIP multi 普通类原型'}
lines=['# 完整核心流程 Proto-Adapter 免训练版：SPV 汉字比较结果','',
    '此处比较保留冻结CLIP图像/文本编码、提示集成、原型双轴归一化与分数融合的免训练Proto-Adapter任务适配。不是PA-HOG，也不是作者原代码逐项复现；使用论文公式重建。无下游梯度更新，但依赖大规模预训练CLIP。','',
    '## 外部面板结果','',
    '| 方法 | E1准确率/% | C1准确率/% |','|---|---:|---:|']
for m in ['H0','H1','PA0','PA1']:
    lines.append(f"| {labels[m]} | {float(metric['E1',m]['accuracy_percent']):.2f} | {float(metric['C1',m]['accuracy_percent']):.2f} |")
if near_chance:
    lines+=['','**关键限制：PA1在两个面板均低于5%，处于很低的识别水平（180类均匀随机猜测期望约0.56%）。尽管MMA在数值上明显领先，本对比不能单独承担论文“优于有竞争力近年方法”的核心论据。应如实定位为冻结RN50方案在该任务上的迁移局限，优先放补充材料或正文简要报告；不能只展示巨大提升而省略低基线表现。**']
lines+=['','## 配对差值','', '| 面板 | 对比 | 差值/百分点 | 字符簇bootstrap 95%区间 |','|---|---|---:|---:|']
for panel in ['E1','C1']:
    for r in comps[panel]:
        low,high=r['ci95_pp'];lines.append(f"| {panel} | {r['comparison']} | {r['delta_pp']:+.2f} | [{low:+.2f}, {high:+.2f}] |")
lines+=['','E1的H1−PA1为预先指定主比较；C1为第二终点，其余用于解释机制。区间以汉字为簇、配对重采样100000次，条件于现有字体和mapping，不代表未采样患者/全部字体的总体不确定性。','',
    '## 开发选择及诊断','',f"PA0选择alpha={sel['PA0']['alpha']:.8g}；PA1选择alpha={sel['PA1']['alpha']:.8g}。分别使用六折留一字体的5400条DEV查询选择，完整搜索见development_search.csv；未使用E1/C1进行参数选择。",
    '', '| 方法 | DEV选择集准确率/% | E1准确率/% | C1准确率/% |','|---|---:|---:|---:|']
for m in ['PA0','PA1','TEXT','VIS0','VIS1','PLAIN0','PLAIN1']:
    lines.append('| '+labels[m]+' | '+' | '.join(f"{float(metric[p,m]['accuracy_percent']):.2f}" for p in ['DEV','E1','C1'])+' |')
lines+=['','DEV是参与选参的开发表现，不是调参后的无偏测试估计。VIS为双轴归一化原型的纯视觉分支；PLAIN只做样本L2、类均值及类原型L2，不做通道归一化，也不融合文本。诊断不替换PA0/PA1主结果。宏F1采用整个面板合并计算，列名为macro_f1_pooled_percent；与此前按子面板平均的宏F1口径不同，不能直接混排比较。','',
    '## 可以支持怎样的结论','']
for panel in ['E1','C1']:
    r=comps[panel][0];lo,hi=r['ci95_pp']
    if lo>0: verdict='在当前固定面板和适配协议下，MMA准确率高于PA1，字符簇区间未跨零。'
    elif hi<0:verdict='在当前固定面板和适配协议下，MMA准确率低于PA1，字符簇区间未跨零。'
    else:verdict='当前差值区间跨零，不能据此声称明确优于、等效或非劣。'
    lines.append(f'- {panel}：{verdict}')
strata=list(csv.DictReader((OUT/'stratified_metrics.csv').open(encoding='utf-8-sig')))
for panel in ['E1','C1']:
    h={r['value']:float(r['accuracy_percent']) for r in strata if r['panel']==panel and r['method']=='H1' and r['axis']=='font_x_mapping'}
    p={r['value']:float(r['accuracy_percent']) for r in strata if r['panel']==panel and r['method']=='PA1' and r['axis']=='font_x_mapping'}
    ds=[h[k]-p[k] for k in h];lines.append(f'- {panel}字体×mapping单元中：MMA胜/平/负为{sum(v>0 for v in ds)}/{sum(v==0 for v in ds)}/{sum(v<0 for v in ds)}（共{len(ds)}个）。这是描述性计数，不是各单元均显著的证据。')
lines+=['','即使MMA在本次所有面板均领先，也不能写成“全面超过近年方法”或“原型聚合算法普遍优于Proto-Adapter”。CLIP的预训练任务、汉字任务差异和输入适配都可能影响结果。本实验只覆盖RN50这一个预先固定骨干，不评价其他CLIP骨干或微调版Proto-Adapter-F。','',
    '## 复核与资源','',
    f"全部{read('independent_audit.json')['independently_recomputed_predictions']}条预测已独立重算，原型、MMA原始HOG重放、DEV选参、准确率和区间检查通过。原始记录、缓存、协议和脚本哈希均保留。",
    '',f"最后一次提取进程实耗{read('extraction_resources.json')['seconds']/60:.2f}分钟，进程峰值RSS {read('extraction_resources.json')['peak_rss_bytes']/2**30:.2f} GiB；所有成功缓存记录的提取时间合计{sum(f['seconds'] for f in read('feature_audit.json')['files'])/60:.2f}分钟，未包含中断时丢失的部分计算。CPU两线程、批量8。该记录不能代替严格同硬件端到端方法效率比较。",
    '', '## 写作与证据保留','',
    ('本次对手准确率过低，不建议以它作为论文主要阳性结果或用巨大提升作标题/摘要卖点。建议完整保留，作为补充材料中的跨任务适配结果，正文说明局限。' if near_chance else '可以增加一张同任务下免训练方法结果表，并把完整诊断放入补充材料。')+'方法中写明：CLIP RN50冻结、七模板汉字提示、每图最大值显示归一化、整图缩放、不做翻转或随机裁剪、六折开发选参，以及E1/C1是既有固定面板。',
    '', '已有Point-SPV、XDomainMix、RanDumb和PA-HOG结果全部保留，不能因本次结果更有利而替换掉它们。不同测试协议分别成表，不跨表排总名次。本文尚未修改Word正文或补充材料，也不保证期刊接收。',
    '', '实现来源：[Proto-Adapter 原论文](http/historical/www.mdpi.com/1424-8220/24/11/3624)；[OpenAI CLIP](http/historical/github.com/openai/CLIP)。']
if (OUT/'low_accuracy_audit.json').exists():
    a=read('low_accuracy_audit.json')
    lines+=['','## 低准确率触发的追加核查','',
        '此核查发生在看到低结果之后，仅用于发现工程错误，未改变协议、参数或主结果。重新加载官方RN50，对ideal、2026、E1、C1各取8张重新编码，逐一与已保存特征核对；缓存一致性通过。以下回代指标包含支持样本，不是泛化结果，不能用于论文性能主张。','',
        '| 训练诊断 | 回代准确率/% |','|---|---:|']
    for r in a['training_diagnostics']:lines.append(f"| {r['method']} | {r['training_resubstitution_accuracy_percent']:.2f} |")
    lines+=['','多mapping融合版PA1在训练支持集上的回代准确率也仅约5.56%，普通CLIP原型PLAIN1约11.13%；因此问题在支持集上已经存在，不能只归因于未见测试mapping。自身检索100%仅排查特征身份/缓存问题，不证明类原型具有泛化能力。',
        '', '工程复核通过只说明当前实现、输入和输出内部一致，不等于已在作者原始自然图像基准上复现其成绩。现有证据不能把全部性能下降明确归因于某一个因素，也不足以证明Proto-Adapter一般无效。不得按外部测试成绩尝试多个版本再只报告获胜版本。']
(OUT/'结果解读与论文建议.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
# Static research figure; confidence intervals only where actually computed.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
fig,ax=plt.subplots(1,2,figsize=(10,4),gridspec_kw={'width_ratios':[1.5,1]})
colors=['#9ca3af','#2679b5','#d8ac63','#cc7052']; x=np.arange(2)
for j,m in enumerate(['H0','H1','PA0','PA1']):
    vals=[float(metric[p,m]['accuracy_percent']) for p in ['E1','C1']]
    bars=ax[0].bar(x+(j-1.5)*.19,vals,width=.18,color=colors[j],label={'H0':'HOG ideal','H1':'MMA-HOG','PA0':'PA ideal','PA1':'PA multi'}[m])
    ax[0].bar_label(bars,fmt='%.1f',fontsize=8,padding=2)
ax[0].set_xticks(x,['E1','C1']);ax[0].set_ylim(0,105);ax[0].set_ylabel('Accuracy (%)');ax[0].legend(loc='upper center',ncol=2,fontsize=8)
for i,panel in enumerate(['E1','C1']):
    r=comps[panel][0];lo,hi=r['ci95_pp'];d=r['delta_pp']
    ax[1].errorbar(d,i,xerr=[[d-lo],[hi-d]],fmt='o',color='#2679b5',capsize=4)
    ax[1].annotate(f'{d:+.2f} [{lo:+.2f}, {hi:+.2f}]',(d,i),xytext=(0,14),textcoords='offset points',ha='center',fontsize=8)
ax[1].axvline(0,color='gray',lw=1,ls='--');ax[1].set_yticks([0,1],['E1','C1']);ax[1].set_ylim(-.5,1.6)
ax[1].set_xlabel('MMA minus PA multi (percentage points)');ax[1].set_title('Paired character-cluster 95% CI',fontsize=10)
for a in ax:a.spines[['top','right']].set_visible(False)
fig.tight_layout();fig.savefig(OUT/'main_comparison.png',dpi=180);fig.savefig(OUT/'main_comparison.svg');plt.close(fig)
cross=[dict(experiment='Full-core training-free Proto-Adapter SPV adaptation',raw_evidence='features/; prototypes/; predictions.csv; selected_alpha.json',summary='metrics.csv; paired_comparisons.json; stratified_metrics.csv',manuscript='same-task frozen-RN50 comparison; pending integration',supplement='all diagnostics/search/protocol deviations; pending integration',status='all outcomes retained'),
dict(experiment='Existing MMA / Point-SPV / XDomainMix / RanDumb / PA-HOG',raw_evidence='existing experiment directories and evidence manifests',summary='not replaced or removed',manuscript='retain protocol differences and positive/negative outcomes',supplement='crosscheck all prior experiments when editing manuscript',status='no Word edits in this turn')]
with (OUT/'evidence_crosswalk.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(cross[0]));w.writeheader();w.writerows(cross)
manifest={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file() and p.name!='delivery_manifest.json'}
(OUT/'delivery_manifest.json').write_text(json.dumps(dict(status='COMPLETE',files=manifest),ensure_ascii=False,indent=2),encoding='utf-8')
print('REPORT AND DELIVERY MANIFEST COMPLETE')
