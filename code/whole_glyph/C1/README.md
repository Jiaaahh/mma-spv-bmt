# MMA 汉字仿真假体视觉实验（v0.3 投稿准备版）

这是一个完全在个人电脑上运行的 CPU-only 仿真实验项目，不采集真人数据，也不需要 GPU。当前论文方向是：在理想 16×16 候选网格的 256 个位置中，每幅刺激固定激活 64 个位置时，使用多个可能的光幻视映射生成训练表征，能否提高解码器对**未见字体和未见映射**的 180 类汉字机器分类稳健性。这里的 B=64 是刺激级稀疏激活预算，不是植入阵列的物理电极总数，也不是最低视觉需求的估计。

项目已完成一次性冻结后仿真确认，状态为 `PASS_COMMITTED`。主方法 `ideal_plus_multi4` 的平均 Accuracy 为 **64.72%**，`ideal_only` 基线为 **49.56%**，差值 **+15.17 pp**；5/5 个新映射均为正，字符级配对 bootstrap 95% CI 为 **[11.83, 18.56] pp**。

> 该结论只表示仿真假体视觉图像的机器可解码性提高，不能直接推断真实植入者的视觉感知、汉字识别或阅读表现。

## 方法概览

所有比较方法共享同一套输入和编码：

1. 对二值汉字图像提取骨架；
2. 在理想 16×16 候选网格的 256 个位置中，按 Skeleton-TopK 选取并激活 64 个位置；
3. 用不同光幻视映射渲染 96×96 仿真假体视觉图像；
4. 提取固定 HOG 特征；
5. 对每一汉字类别，将训练字体与指定映射下的 HOG 向量等权平均并做 L2 归一化；
6. 测试时以最大余弦相似度分类。

| 方法 | 构建类别原型时使用的映射 | 角色 |
| --- | --- | --- |
| `ideal_only` | 理想映射 | 主要基线 |
| `single_moderate` | moderate seed 2026 | 单映射增强对照 |
| `multi4` | moderate seeds 2026–2029 | 多映射原型对照 |
| `ideal_plus_multi4` | 理想映射 + moderate seeds 2026–2029 | 预先冻结的主方法 |
| `oracle_matched` | 与测试条件相同的映射 | 仅作映射匹配诊断参照，不是可部署方法 |

主方法没有可搜索的权重、温度或分类器超参数。理想映射和四个 moderate 映射对每个训练字体、每个类别各贡献一条表征，避免某一字体或映射被重复加权。

## 实验设计

- 汉字类别：SUBTLEX-CH 字符频率文件按原始发布顺序排列的前 180 条数据行，不按算法结果筛选；
- 激活预算：B=64，即每幅刺激从 256 个候选位置中激活 64 个位置；
- 开发集：6 种旧字体，6 字体 × 5 held-out mappings，共 30 个交叉验证单元；
- 冻结后确认集：2 种此前未用于开发的字体（DengXian Regular、FZShuTi Regular）；
- 冻结后确认映射：moderate seeds 2031–2035，与开发映射不重叠；
- 每个确认映射：2 字体 × 180 字，共 360 个样本；
- 主要终点：`ideal_plus_multi4 - ideal_only` 的 Accuracy 差值；
- 区间估计：180 个汉字 cluster 的配对 bootstrap，100,000 次，每个 cluster 保留 2 字体 × 5 mappings 的 10 个配对观测。

开发交叉验证中，主方法相对基线为 **+13.94 pp**，30/30 个字体×映射单元为正；冻结后确认中为 **+15.17 pp**，10/10 个字体×映射单元为正。所有 Accuracy 和 Macro-F1 均为机器分类指标，不是受试者识别率。

训练和确认映射虽然使用不同 seed 与不同具体几何，但来自同一个参数化生成器。`moderate` 是预先固定的受控仿真压力条件，不是由患者数据拟合的临床扰动等级；确认结果只支持同一生成器分布内的未见随机实现泛化。

## 项目在哪里运行

当前项目目录：

```text
/historical/user\Documents\Codex\2026-08-25\referenced-chatgpt-conversation-this-is-an-2\outputs\hanzi-spv-calibrated-v2
```

冻结后确认实际运行环境：

```text
Windows 11
CPython 3.12.7
NumPy 1.26.4
SciPy 1.13.1
scikit-image 0.25.0
scikit-learn 1.5.1
PyYAML 6.0.1
threadpoolctl 3.5.0
CPU-only
```

精确的 Windows 开发环境见 `requirements-windows-dev.txt`。原 `requirements.txt` 保留为另一套 CPython 3.11 复现环境，不要把两套依赖混在同一个虚拟环境中。

## 当前结果与审计文件

- 开发结果：`outputs/d2_mma_cv_windows/`
- 一次性冻结后确认结果：`outputs/c1_mma_confirmation_windows_r3/`
- 外部完成记录：`outputs/c1_mma_confirmation_windows_r3.completion.json`
- 最终冻结锁：`locks/c1_mma_confirmation_windows_r3.lock.json`
- 三次确认前工程失败记录：`audits/`
- 确认字体与映射资产：`data/mma_confirmation/`

确认结果目录包含：

- `summary.json`：主要结果和 bootstrap CI；
- `fold_metrics.csv`：每个映射、每种字体、每种方法的 Accuracy 与 Macro-F1；
- `predictions.csv`：全部样本级预测；
- `manifest.json`：输入、源码、配置、环境与输出哈希；
- `completion.json`：提交完成记录。

当前一次性确认已经完成，运行器会拒绝在同一路径重跑或覆盖结果。不要删除完成记录后重复确认。

本文所称“一次性确认”，是指最终冻结方案只有一次成功的 C1 性能读取并产生正式预测与指标。成功前的三次工程执行均在预测或性能读取前停止，未依据确认表现修改科学配置。

## 投稿补充材料与便携复现

投稿准备阶段新增的内容全部位于正式 C1 之外，不会覆盖冻结结果：

- `publication_materials/`：180 字清单、字符级事后描述性分析、字体×seed 汇总表、混淆表和可重复生成脚本；
- `publication_replay/`：采用相对路径的 `NON_CONFIRMATORY_DERIVED_REPRODUCTION` 便携重放区，包含静态训练字形、上游字体元数据、预期结果合同和完整性工具。

常用核验命令：

```powershell
python publication_materials\scripts\generate_supplements.py
python publication_replay\tools\verify_replay.py
python publication_replay\tools\run_replay.py --dry-run
```

便携重放只能证明代码与已冻结 C1 输出可被重现，不能解释为第二次独立确认。完整用法和输出标签见 `publication_replay/README.md`。

## 自检

在项目目录运行：

```powershell
python -m pytest -q -p no:cacheprovider
python -m ruff check src\v2 tests_v2 run_mma_development.py run_mma_confirmation.py freeze_mma_confirmation.py prepare_mma_confirmation_assets.py publication_materials\scripts publication_replay\tools
```

最终状态为 41 项测试通过，v2 模块与入口代码检查通过。继承的旧 Stage9/10 源文件不属于本方法的检查范围，且为保持正式 source-tree hash 不再改动。完整的环境建立、结果查看、哈希核验和安全边界见 `RUNBOOK.md`；实验演进及阴性开发分支见 `DEVELOPMENT_LOG.md`。

## 数据与许可边界

旧仓库 `/historical/deepseek\hanzi-spv-benchmark` 始终作为只读输入。确认集只保存渲染后的字形数组和质控图，不分发 Windows 原始字体文件。训练字体的上游许可文本和元数据已整理到便携复现区，但等线、方正舒体所生成确认字形数组的公开再分发权仍需作者或单位核查；项目源码也尚需选择公开软件许可证。向外分享以前必须完成这些核查，详见 `LICENSE_NOTES.md` 和 `publication_replay/LICENSE_AND_RIGHTS.md`。
