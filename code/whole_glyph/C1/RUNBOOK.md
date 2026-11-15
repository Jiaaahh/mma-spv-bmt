# MMA v0.3 运行、审计与投稿复现手册

本手册用于在当前电脑上查看、核验和后续复现实验。所有命令均在项目根目录执行。

## 1. 项目位置

```powershell
cd "/historical/user\Documents\Codex\2026-08-25\referenced-chatgpt-conversation-this-is-an-2\outputs\hanzi-spv-calibrated-v2"
```

旧项目 `/historical/deepseek\hanzi-spv-benchmark` 仅作只读输入。不要向旧项目写结果，也不要修改其数据和映射文件。

## 2. 建立 Windows 环境

正式 C1 冻结后确认实际使用 CPython 3.12.7。若要建立独立环境：

```powershell
py -3.12 -m venv .venv-win
.\.venv-win\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-windows-dev.txt
```

检查关键版本：

```powershell
python --version
python -c "import numpy, scipy, skimage, sklearn, yaml, threadpoolctl; print(numpy.__version__, scipy.__version__, skimage.__version__, sklearn.__version__, yaml.__version__, threadpoolctl.__version__)"
```

项目也保留了 `requirements.txt` 对应的 CPython 3.11 环境方案，但它不是本次 C1 Windows 确认的数值环境。两套依赖应放在不同虚拟环境中。

## 3. 代码自检

```powershell
python -m pytest -q -p no:cacheprovider
python -m ruff check src\v2 tests_v2 run_mma_development.py run_mma_confirmation.py freeze_mma_confirmation.py prepare_mma_confirmation_assets.py
```

最终版本应为 41 项测试通过，v2 模块与入口代码检查通过。继承的旧 Stage9/10 文件有冻结前遗留的未使用导入，不在此命令范围内；不要为了全仓 lint 修改已绑定的正式 source tree。若上述自检失败，不要运行新实验，先保留错误输出并排查环境或接口。

## 4. 查看已完成结果

一次性确认已经完成，不需要再次运行。主要文件为：

```text
outputs\c1_mma_confirmation_windows_r3\summary.json
outputs\c1_mma_confirmation_windows_r3\fold_metrics.csv
outputs\c1_mma_confirmation_windows_r3\predictions.csv
outputs\c1_mma_confirmation_windows_r3\manifest.json
outputs\c1_mma_confirmation_windows_r3\completion.json
outputs\c1_mma_confirmation_windows_r3.completion.json
```

PowerShell 查看汇总：

```powershell
Get-Content outputs\c1_mma_confirmation_windows_r3\summary.json
Import-Csv outputs\c1_mma_confirmation_windows_r3\fold_metrics.csv | Format-Table
```

开发交叉验证结果位于：

```text
outputs\d2_mma_cv_windows\summary.json
outputs\d2_mma_cv_windows\fold_metrics.csv
outputs\d2_mma_cv_windows\predictions.csv
outputs\d2_mma_cv_windows\manifest.json
```

## 5. 已完成确认不得原地重跑

`c1_mma_confirmation_windows_r3` 是一次性冻结后仿真确认。这里的“一次性”指最终冻结方案只有一次成功的性能读取并生成正式预测与指标；成功前的三次工程停止均未读取确认性能。运行器通过结果目录和外部 completion 双重标记拒绝覆盖：

```powershell
python run_mma_confirmation.py --config configs\c1_mma_confirmation_windows_r3.yaml
```

上述命令现在应拒绝运行，这是预期行为。不要删除输出、completion 或 lock 来绕过保护，也不要在看到结果后修改主方法、映射 seeds、字体、预算或 bootstrap 设置。

如果未来有真正的新数据，应创建新的实验编号、新配置、新输出路径和新冻结锁；旧 C1 继续作为本论文的确认结果，不被替换。

## 6. 便携重放与投稿补充材料

`publication_replay/` 是从正式 C1 派生的便携重放区，使用相对路径，并把输出明确标为 `NON_CONFIRMATORY_DERIVED_REPRODUCTION`。它不改变、删除或覆盖正式 C1，也不能被解释为第二次独立确认。

先核验资产与冻结结果合同：

```powershell
python publication_replay\tools\verify_replay.py
python publication_replay\tools\run_replay.py --dry-run
```

确需完整重放时，使用一个新的、不会覆盖既有结果的 run ID：

```powershell
python publication_replay\tools\run_replay.py --run-id replay_local_001
python publication_replay\tools\verify_replay.py --replay-output publication_replay\results\replay_local_001
```

`publication_materials/` 保存投稿用派生表格和事后描述性图。可重复生成并核对：

```powershell
python publication_materials\scripts\generate_supplements.py
```

这些材料只能解释既有主结果，不增加新的验证性终点或显著性结论。

## 7. 冻结与完整性记录

最终绑定：

```text
Experiment: c1_mma_confirmation_windows_r3
Config SHA-256: 4d09162f3b9f2a337610e838e1fcc781e7f2068b5cca9ba193a75dfa1827ecfe
Source-tree SHA-256: 2875a2917549a6add3adc39f373aaa2df249119afdf06cdd7837c5824d4ab292
Lock SHA-256: c2b4cb4676ed4f67d185b633518278b2a886adc7c431dfd378f6d38301fdcf4d
Manifest SHA-256: f96e090a8a34f7a7deda0e8e7cd172d988e0389c266c30462526d89e02ec79d3
```

输出哈希：

```text
summary.json: bb0610c15ce9abab75e542daf038417e68c9d0f912e3989eac19b1f121afbfb2
fold_metrics.csv: 3e2949557d14489ea2f313816d02d3f2b5bea00a0a9a00f18459627662ac56d0
predictions.csv: 01460997e7cc3b898241486d6881137e19225ea9f38f3fc9c027577be312f3d3
```

核验单个文件：

```powershell
(Get-FileHash outputs\c1_mma_confirmation_windows_r3\summary.json -Algorithm SHA256).Hash.ToLower()
(Get-FileHash outputs\c1_mma_confirmation_windows_r3\manifest.json -Algorithm SHA256).Hash.ToLower()
```

## 8. 结果解释顺序

写作时按预先冻结的层级报告：

1. 主比较：`ideal_plus_multi4 - ideal_only`；
2. 5 个独立 mapping seeds 的逐 seed 结果；
3. 180 字符 cluster 的配对 bootstrap 95% CI；
4. Macro-F1 作为一致性指标；
5. 两种确认字体的分层结果；
6. `single_moderate`、`multi4` 用于消融；
7. `oracle_matched` 仅作映射匹配诊断参照，不能包装为实际方法或数学上界。

不得只报告最有利的字体或映射。FZShuTi 明显比 DengXian 难，应作为字体域差异和局限性明确报告。

## 9. 三次工程停止如何处理

正式成功前的 attempt0、r1 和 r2 均因接口检查停止，审计文件位于 `audits/`。它们没有产生测试预测、Accuracy、Macro-F1 或 bootstrap 结果；对应输出、临时目录和 completion 均为 0。修复只针对类型/接口错误，每次均更换实验编号并重新冻结。

论文方法正文不需要展开工程调试细节，但复现材料中应保留审计文件。若被问及，可如实说明：三次停止发生在任何确认性能读取之前，因此没有依据确认结果修改方法。

## 10. 科学边界

- 本研究不包含真人、患者或植入者实验；
- 结果衡量的是模拟光幻视图像的机器识别性能；
- 所有 Accuracy 和 Macro-F1 都是机器分类指标，不是受试者识别率；
- 不能宣称提高真实阅读速度、视觉质量或临床获益；
- 仅验证了 180 个汉字、当前映射生成模型、两种独立字体，以及每幅刺激从 16×16 候选网格的 256 个位置中激活 64 个位置的条件；B=64 不是物理电极总数或最低视觉需求；
- 训练与确认映射虽使用不同 seeds，但来自同一参数化仿真生成器；
- `moderate` 是人为预先设定的受控仿真压力条件，不代表患者、疾病或器械人群中的临床分布；
- bootstrap 仅重采样 180 个汉字，不代表字体、映射机制或真实植入者总体的不确定性；
- 后续扩展到更多字体、预算或失真等级属于新实验，应与已完成 C1 区分。

## 11. 对外发布前的许可边界

- 便携区不包含 TTF、OTF、TTC 等字体二进制；
- 训练字体的 OFL 文本和上游元数据已保存；
- 等线、方正舒体生成的确认字形数组公开再分发前仍需权利核查；
- 项目源码尚未选择公开软件许可证；
- 未完成上述事项前，可供导师和审稿过程内部核验，但不要直接上传公开仓库或公开归档。
