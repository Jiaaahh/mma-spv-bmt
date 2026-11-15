# 冻结结果核验入口（2026-10-13 修订）

在附件解压根目录运行：

```sh
python tools/verify_results.py --bundle-root evidence --output-dir checks
```

Python 3.9 或更高版本即可检查 SHA-256 和冻结预测标签。安装 NumPy 后还可重算已记录的 RanDumb 推理计时中位数、95% 分位数、时间比，以及多预算准确率和 nAUBC；未安装时报告会明确记录这些数值检查被跳过。无需 GPU，也不会训练或执行模型。

运行结果写入 `checks/verification_result.json`。`evidence/` 为冻结输入，程序禁止将输出目录设在其中，原版 `evidence/verify_bundle.py` 保留用于历史追溯；本次使用 `tools/verify_results.py`。初始核验快照置于 `verification/`，与读者运行时产生的 `checks/` 分开。

新入口逐项读取 `evidence/manifest.json`，先核对文件哈希，再按表头识别逐样本预测。新增 `truth/predicted` 和 `class_id/predicted_class` 组合；XDomainMix 外层文件中的 CNN 和 MMA 标签分别评分。分组保留文件中存在的 `panel`、`split`、`variant`、`method`、`condition`、`budget` 和 `projection_seed`，避免把自然图像支持集、验证集、测试集或输入变体混为同一结果。运行折、检查点等其他协议信息保留在逐文件路径中。

报告分别列出 `independent_label_scoring`（真实标签与预测标签独立比较）、`correct_flag_only`（仅累计已有逐样本正确标志）、`hash_only`（仅哈希）、`skipped`（无法识别的预测模式）及 `failed`（失败）。汇总表中的 `correct` 是计数，不当作逐样本标签重新评分。若有标签与正确标志矛盾、空标签、不可解析的预测表或哈希不一致，程序记录失败并返回非零退出码，不会静默通过。

本次冻结包实跑结果为：647 个清单文件哈希全部一致；110 个逐样本 CSV 的 114 条预测流完成标签复算（4 个外层 CSV 各有 CNN/MMA 两条预测流）；537 个文件仅核对哈希；仅累计正确标志、跳过和失败均为 0。**这不等于对全部 CSV 重新评分，也不等于重新运行实验。** 不同协议、不同开发阶段和不同模型流的结果不能汇总成一项正式测试准确率。

此工具不复现模型训练、输入生成、所有置信区间或真实硬件计时，不检验未列入原始 manifest 的文件。统计区间仍应按对应原始统计脚本和记录核对；历史多轴 bootstrap 的随机数调用路径尚未完全对齐，不宣称逐位复现。

工具完整性检查见 `verification/verifier_selfcheck.json`。四组检查覆盖新标签模式及分组、刻意制造的正确标志矛盾、空标签/未知预测表头、篡改文件哈希；矛盾预测的命令行返回码确认为 1。
