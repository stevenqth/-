# 代码目录说明

本目录存放本项目的全部可复现脚本。**每一份结果文件（`成果/results/`、`成果/figures/`）都必须能由这里的脚本重新生成**；论文中出现的每个数字都应能追到某个脚本输出。

## 运行环境

- Python：`D:\miniconda\python.exe`（Python 3.11.13，conda-forge）
- 已安装并使用的库：`numpy`、`pandas`、`scipy`、`scikit-learn`、`matplotlib`、`networkx`、`openpyxl`（经离线方式提供）、`torch`（问题三备用）
- 工作目录：项目根目录 `D:\github mama`
- 随机种子：所有脚本统一使用 `42`；交叉验证折数、Bootstrap 次数等写在各脚本参数区

## 关于离线 openpyxl

本机当前无外网（沙箱内 TLS 不可用），但读取 `3.xlsx`/`8.xlsx` 需要 openpyxl。当前做法是把纯 Python 的 `openpyxl` 与 `et_xmlfile` 复制到工作区 `.work/pylibs`，并通过 `PYTHONPATH` 注入：

```powershell
cd "D:\github mama"
$env:PYTHONPATH = "D:\github mama\.work\pylibs"
& "D:\miniconda\python.exe" "成果\code\00_data_audit.py"
```

若在联网环境运行，直接 `pip install openpyxl` 后无需 `PYTHONPATH`。`.work/` 已被 `.gitignore` 排除，不进入版本库。

## 脚本清单

| 脚本 | 作用 | 输出 |
| --- | --- | --- |
| `00_data_audit.py` | 阶段 0 数据审计：材料 SHA256、逐列变换核验、缺失模式、相关矩阵复现、变量映射、感官维度映射、缺陷阈值基线 | `成果/results/00_*.json|csv` |
| `01_extract_problem_text.py` | 从官方题面 PDF 提取纯文本（自实现 PDF 对象/CMap 解析，无第三方 PDF 库） | `成果/00_材料清点/题面文本.txt` |
| `02_extract_reference_paper.py` | 从参考论文 PDF（Springer/iText）提取全文（Type1 字体宽度 + WinAnsi + 坐标重组，无第三方 PDF 库） | `资料/参考论文_文本提取.txt` |
| `02_data_prep.py` | 待编写：统一数据视图（缺失处理、分层划分、分级映射） | `成果/results/02_*` |
| `03_q1_regression.py` | 待编写：问题一多输出回归、分级校准、交叉验证与不确定度 | `成果/results/03_*`、`成果/figures/fig_q1_*` |
| `04_q2_top3.py` | 待编写：问题二共表达图、多方法重要性、Bootstrap 稳定性 | `成果/results/04_*`、`成果/figures/fig_q2_*` |
| `05_q3_generation.py` | 待编写：问题三 cVAE、跨风格迁移、缺陷约束与代理评估 | `成果/results/05_*`、`成果/figures/fig_q3_*` |
| `06_assemble_paper.py` | 待编写：结果→论文表格/正文数字引用与 docx 汇编 | `成果/02_论文/` |

## 约定

1. 脚本按编号执行顺序可独立运行，但共享数据加载逻辑集中在 `02_data_prep.py`，避免各脚本重复实现清洗。
2. 所有随机性固定种子；所有输出写入 `成果/results/`（数据）与 `成果/figures/`（图），不覆盖原始 `资料/`。
3. 不把 `2.csv`/`5.csv`/`6.csv` 或 `8.xlsx` 的 GBM/SHAP 当作模型输入特征（同源泄漏），仅作结构先验与外部对照。
4. 论文正文中的数值必须由脚本写入 `results/` 后引用，不手工抄写。
