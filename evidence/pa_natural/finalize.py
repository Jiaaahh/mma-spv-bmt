from pathlib import Path
import json,csv,hashlib
from decimal import Decimal,ROUND_HALF_UP
OUT=Path(__file__).resolve().parent
def read(n):return json.loads((OUT/n).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
r=read('result.json');m=read('implementation_manifest.json')
assert r['status']=='COMPLETE' and r['correctness_audit']['status']=='PASS'
for n,key in [('run_sanity.py','script_sha256'),('protocol.json','protocol_sha256'),('split_manifest.json','split_manifest_sha256')]:assert sha(OUT/n)==m[key]
def fmt(v):return str(Decimal(str(v)).quantize(Decimal('.01'),rounding=ROUND_HALF_UP))
labels={'TEXT':'CLIP文本分支','PLAIN':'普通CLIP类原型','PA_VISUAL':'Proto-Adapter纯视觉分支','PA_FULL':'完整Proto-Adapter融合版'}
test=[x for x in r['metrics'] if x['split']=='test']
lines=['# Proto-Adapter自然图像健全性检查结果','',
    '结论：同一套原型构建与融合选参实现，在本次固定Caltech-101子集上正常工作。未发现整套实现普遍失效的证据；汉字SPV中的低分更指向任务特征与当前适配选择，但不能据此认定SPV端所有细节已无误，也不能确证单一原因。','',
    '## 固定方案','',
    '官方Caltech-101数据包校验通过。按种子20260930固定随机选择20类，每类8张支持、4张验证、16张测试，共560张。文件路径、原始字节和解码像素去重检查通过。测试集320张；没有用测试集调参数。',
    '', '采用与SPV实验相同的官方CLIP RN50权重，冻结图像/文本编码器。从原评分脚本按AST提取并直接复用norm、build、choose三个函数；另用独立累加与逐类分数重算核对。自然图像采用官方RGB/缩放/中心裁剪/归一化，不使用SPV的每图最大值显示转换。',
    '', '七个自然图像提示模板预先固定。支持图每张使用一个确定性视图，不使用随机增强；不是原论文完整数据划分与增强设置。',
    '', '## 测试结果','', '| 方法 | 正确数/320 | 准确率 |','|---|---:|---:|']
for row in test:lines.append(f"| {labels[row['method']]} | {row['correct']}/320 | {fmt(row['accuracy_percent'])}% |")
lines+=['',f"融合alpha仅根据80张验证图选择，为{r['selected_alpha']:.8g}，在测试评分前保存。完整融合版准确率为314/320=98.125%；表格按四舍五入显示。",
    '', '本次预先设定的粗略工程门槛为普通原型、PA纯视觉和完整PA均超过50%（随机期望5%），三者均通过。这只是发现严重功能失效的筛查阈值，不是论文显著性或复现成功标准。',
    '', '## 独立核对','',
    f"全部{r['correctness_audit']['independent_predictions']}条支持/验证/测试预测均通过独立逐类计算；原型最大绝对差{r['correctness_audit']['prototype_max_abs_error']:.3g}。支持样例预览已检查。编码与评分阶段约{r['encoding_seconds']:.1f}秒，不含下载和前置读取。",
    '', '## 对汉字论文意味着什么','',
    '1. 当前证据不支持把汉字低分归结为一个普遍性的原型公式、CLIP加载或分类代码错误；相同核心实现在自然图像上能工作。',
    '2. 这不能排除SPV专有预处理、提示、特征适用性和多mapping聚合方面的问题。此前去掉文本后仍低、支持集回代也低，以及mapping条件诊断，均需结合考虑，不能只调整alpha便声称修复。',
    '3. 原汉字结果不因本次自然图像高分而自动变成强竞争基线胜出。应保留为明确协议下的适配结果，且不能把98.13%与汉字MMA准确率放在同一任务排行榜比较。',
    '4. 不建议在没有具体可验证修改理由时直接重跑整个汉字比较。若继续，下一步限定为少量SPV输入适配的训练/开发集敏感性检查，保留原版，不据已见E1/C1选择配置。换骨干、中文CLIP或多原型是新变体。',
    '', '## 边界','',
    '只有20类、一个固定划分、8-shot及确定性视图；未复现作者完整Caltech101表格，也没有证明Proto-Adapter在所有自然图像上均优于普通原型。这里只是实现健全性检查，不是新的方法贡献或可直接用于论文的SOTA复现结论。汉字协议、预测和原论文文件均未修改。',
    '', '数据来源：[CaltechDATA](http/historical/data.caltech.edu/records/mzrjq-6wc02)。方法来源：[Proto-Adapter原论文](http/historical/www.mdpi.com/1424-8220/24/11/3624)。',
    '', '关键证据：protocol.json、split_manifest.json、implementation_manifest.json、selected_alpha.json、validation_search.json、predictions.csv、result.json。']
(OUT/'自然图像验证结论.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
with (OUT/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(r['metrics'][0]));w.writeheader();w.writerows(r['metrics'])
(OUT/'completion_audit.json').write_text(json.dumps(dict(status='PASS',source_and_protocol_hashes=True,independent_predictions=2240,visual_support_preview='reviewed',original_SPV_protocol_changed=False,original_SPV_test_rescored=False),indent=2),encoding='utf-8')
files={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='delivery_manifest.json'}
(OUT/'delivery_manifest.json').write_text(json.dumps(dict(status='COMPLETE',files=files),ensure_ascii=False,indent=2),encoding='utf-8')
print('REPORT AND AUDIT COMPLETE')
