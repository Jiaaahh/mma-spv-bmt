# 多映射原型聚合结果复核包 20261004

这是与2026年10月4日修订稿配套的本地结果复核材料，不是已公开的存储库，也不是完整的端到端训练环境。原始证据按模块原样保存；部分原JSON含历史机器路径，匿名投稿或公开发布前需要据目标期刊政策制作脱敏分发副本，不能直接宣称本包匿名。

## 入口与范围

| 模块目录 | 论文对应 | 入口 |
|---|---|---|
| whole_glyph/C1、E1、StudyB | 表2、4、7、8，图4、5及S12–S21 | predictions.csv或merged_predictions_with_truth.csv；primary_recheck.json |
| internal_hog、edge_four_cell | 表3、5，图3，S4–S6 | 原始predictions.csv、summary.json、run_hog.py或run.py |
| randumb | 表2、11，图7a，S22–S24、S29 | predictions.csv、costs.json、core.py、measure_costs.py |
| pa_hog、pa_full、pa_protocol、pa_natural、pa_input | 表6、S25–S28 | predictions.csv、paired_comparisons.json、frozen_protocol.json |
| xdomainmix | 表9、S7–S9 | 四份选择记录、固定终点预测及原报告 |
| point | 表10、图6、S10–S11、S32 | results/*.npz、source/summarize_final.py、selection/selection.json |
| cost_server | 表12、图7b、S30–S31 | cost_summary.json、raw下的计时/测量记录与协议 |

## 本地核验

在解压目录运行 `python verify_bundle.py`（Python 3.10或更高；NumPy用于可选数值计算）。该脚本校验每份原始文件的SHA-256，复算含标签或正确标志CSV的分组正确数、StudyB五预算曲线面积、RanDumb原始计时分位数及耗时比，并将结果写到 verification_result.json。缺少NumPy仍可完成哈希及CSV检查。原模型脚本用于阅读实现与统计口径，可能仍依赖作者原目录、环境或未分发资产，不保证直接执行所有训练脚本。

深度权重、字体文件、派生字形/SPV缓存及完整第三方环境不包含在本包。已有原始统计不因本次编辑重新选模型、改终点或重跑训练。原始输入清单与服务器返回清单职责不同，不能将前者未回传的源文件均判为结果丢失；Point源代码、vendor/provenance和audit文件另列在source中。

本机RanDumb计时原costs.json保留每条件100次raw_seconds；CPU型号未被原运行记录捕获，因此不能事后补写当前主机型号。Point配对统计以字符为簇，同时保留字体、映射和初始化；源码使用default_rng(26092515)与多项式频数抽样。nAUBC分母96为128−32，不能改成128。
