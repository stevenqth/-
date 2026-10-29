# 文献驱动的论文与代码优化

操作者：铱豪；日期：2026-10-29；分支：`b/literature-revision`。

基于 `c/stage2-q1` 的外部数据修订版、用户提供的两篇论文和新增五篇原始研究文献。原始结果与论文保持不变，本次论文及全部新结果单独保存，便于队友对照和审阅。

## 先看哪些文件

- `output/literature_revision/啤酒风味论文_七篇文献优化版.pdf`：完整修订论文。
- `output/literature_revision/论文正文_七篇文献优化版.md`：可编辑论文正文。
- `output/literature_revision/逐篇文献内容与修改对照.md`：逐篇说明具体文献内容、页码或章节、修改的论文部分、代码函数和验证文件。
- `literature/sources.json`：七篇来源的结构化记录；`literature/evidence/download_manifest.json`：新增五篇下载来源与SHA256。
- `results/literature_revision/validation_checks.csv`：实际数据及算法检查；PDF逐页渲染另已核验，见交付验证记录。

## 文献对应的改动

1. 用户《啤酒风味预测与约束迁移》：分开NA与重复底值、秩正态PLS、尺度账本、218项单项化学候选、条件生成与经验重采样基准。只采用对应模块，未声称复现其全部算法。
2. 用户 Schreurs 等 2024：明确原始量纲与专业评分、按真实感官响应验证，增加后续加标盲评设计，不搬用原文结果。
3. Cawley 与 Talbot 2010：内层选型、外层评价的重复嵌套验证。
4. Strobl 等 2008：普通置换与近邻条件置换对照，解释相关变量的重要性；近邻为近似实现。
5. Meinshausen 与 Bühlmann 2010：100次无放回半样本选择频率与Monte Carlo误差；不套用未经满足假设的误选率保证。
6. Sohn 等 2015：后验重参数化采样、固定噪声和标准先验对照；明确原cVAE使用标准先验，未学习条件先验网络。
7. Deb 等 2002：在8维潜空间实际执行Python NSGA-II和四种偏好决策，不冒称MATLAB结果。

## 实际执行与复现

安装 `requirements_literature.txt` 后，在本目录依次运行：

```text
python code/literature_revision.py --stage all --subsamples 100
python code/build_literature_paper.py
python code/validate_literature_revision.py
```

可选阶段为 `audit`、`q1`、`q2`、`q3`。已保存的cVAE权重与预处理参数用于推理，本次不重新训练它，也无需PyTorch。`literature_inference.py`只接受明确列出的数值张量重建与OrderedDict，拒绝任意pickle全局对象；仍应只加载可信的本项目检查点。

重训原版cVAE请另按原 `requirements.txt` 安装PyTorch，并运行原 `question3.py`；重训后必须重新执行审计和本次实验，不能沿用原检查点对应的结果。

## 新输出与原输出

- `results/` 原实验：保留，未被本次运行覆盖。
- `results/literature_revision/`：新尺度证据、嵌套OOF、选择记录、条件重要性、半样本频率、生成诊断、外部对照、Pareto候选、推荐及验证。
- `figures/literature_revision/`：本次实际结果图。
- `code/literature_*.py`：新实验与算法模块；`build_literature_paper.py`生成论文；`validate_literature_revision.py`独立复核。

## 证据边界

已验证结论：新的平均完整OOF R²约0.2113，同划分固定随机森林约0.1966；仅是描述性小幅改善，没有显著性保证。半样本频率表保留不稳定条目。条件高斯优于无条件基准，但未超过风格内重采样。

假设：风格语义映射、近邻条件置换、潜空间候选的可实现性及TOPSIS权重均有建模假设。固定cVAE的全数据分类器一致率是内部诊断，不是新留出生成验证。

待验证问题：双乙酰和反式2壬烯醛存在标定局部差异；源样品4VG参考为零，不能把其附件底值解释成已核实正浓度。新酒厂或新批次外推、真实感官改善和酿造可行性均尚未验证。MATLAB未安装，本轮NSGA-II实际由Python执行。

原外部消费者评分实验未重新训练，论文明确沿用该基线；新后验生成谱的外部风格对照已独立重新计算，不与旧生成谱混淆。


## RYH优化1最终排版
2026-10-29：正文各级小标题及参考文献标题按用户要求统一左对齐，关键词左对齐，工作过程附录移除。最新版位于 output/literature_revision/RYH优化1.docx 和同名 PDF（13页）；源码包同步更新。
