# 基于铱豪多目标模型的外部数据增强版

本项目在铱豪 main 分支 `fc94403` 的 NSGA-II/Pareto 研究上扩展。`baseline/` 保留其原始文件。本次输出为模型、代码、数值结果和论文改进意见，不生成论文。

## 做了哪些改变

1. 用新增5558款啤酒数据及Excel词典，把原附件7.csv的消费者描述符聚合为10个粗粒度目标，训练“化学输入→专业感官及消费者描述”的60维代理。
2. 外部数据按酒厂分组，约65%用于优化目标，约35%留作检查。词频先转行份额，再以优化集计算经验百分位。零词频样本不建立画像；原始数据完整保留。
3. 以适应小风格样本数的嵌套交叉验证，比较Ridge、Extra Trees和随机森林，在训练折内按目标选择模型；插补、截尾、标准化全部在训练折拟合。
4. 在铱豪三个目标上增加“与外部消费者风味画像的差距”，共四个目标。外部目标是跨域语义对应假设，不是新增化学标签。
5. 原三参数扩展为六参数：迁移强度、四种缺陷各自削减强度、一个代理相关化合物调节强度。候选仍是统计画像，不是物料配方。
6. 化学浓度硬约束、三个代理模型的陈旧酒花一致性约束和样本支持距离约束共同筛选候选。模型一致不等同于真实品评保证。
7. 实际运行Python NSGA-II，含非支配排序、拥挤距离、可行性优先、SBX交叉、 polynomial mutation、精英保留；使用已知ZDT1前沿检查。原MATLAB程序保留，但本次未运行MATLAB。

## 运行

Python 3.12，先安装 requirements.txt。依次执行：

```text
python -m pip install -r requirements.txt
python code/prepare_external.py
python code/train_surrogates.py
python code/optimize.py --population 64 --generations 50
python code/validate.py
python code/report.py
```

`models/surrogates.joblib`为已拟合模型，需要`code/train_surrogates.py`和`code/common.py`在导入路径中。示例见`code/predict.py`。输入必须使用原附件231列的顺序及原始混合尺度，不能全部取对数。

`results/*/case_config.json`记录每个迁移案例的源样品、目标风格、参数、浓度上限、控制变量和外部目标权重；`*_pareto.csv`为近似前沿，`recommendations.csv`为四种偏好下的TOPSIS推荐，另附对应化学谱和60维代理预测。

主案例与铱豪一致：beer218，Pils/Lager→Tripel。另检查两个Pils/Lager样本及Wheat→Saison、Stout/Porter→Dubbel，共5个案例、3个迁移方向；三参数与六参数版本均已运行。主案例额外有移除外部目标和等评估预算随机搜索对照。

TOPSIS偏好权重仅用于展示。支持距离阈值取真实样本留一最近邻距离的95%分位数；它是经验统计保护，不是物理可行性约束。四种缺陷阈值沿用铱豪附件依据，4VG的风格适用性仍需核验。

逐元素复核校准表时，二乙酰有3条、反式2壬烯醛有2条差异超过1e-6，已记入results/defect_scale_audit.csv；未用校准表静默覆盖原输入。大多数条目支持当前尺度，但不能声称所有测量均一致。

## 如何读结果

- `results/surrogate_summary.json`：匹配验证协议下的基线和改进代理表现。
- `results/optimization_summary.csv`：各案例可行性和前沿描述。
- `results/all_recommendations.csv`：预先按优化集选择的推荐，以及随后计算的外部留出差距。
- `results/validation_summary.json`：独立重算检查。
- `模型说明与结果.md`：实际结果概览和限制。
- `基于模型的论文改进意见.md`：本次需要的论文改进建议，不是论文正文。

外部留出的是评论酒厂，不是独立化学品评实验；5个源案例也来自已有250样本，不能称为化学外部测试集。模型代理的低预测能力、评论高评分采样偏差、词典类别重叠及工艺可实施性都需保留在结论中。

数据来源：用户提供的MathorCup赛题附件，以及Kaggle Beer Information - Tasting Profiles（stephenpolozoff/sp1222，BeerAdvocate评论，页面标注CC BY4.0）：https://www.kaggle.com/datasets/stephenpolozoff/top-beer-information 。原文件SHA256见results/input_manifest.csv。
