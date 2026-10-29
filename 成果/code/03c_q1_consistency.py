# -*- coding: utf-8 -*-
"""03c_q1_consistency.py — 问题一：逐目标指标与分级混淆矩阵的“折内口径”一致性重算。

背景：2B 主脚本的逐目标指标/混淆矩阵原先用“3 次重复平均后的 OOF 预测 + 全数据映射”，
会比“逐折训练/验证口径”的宏观指标乐观（例如分级精确率 0.64 vs 逐折 0.46）。
本脚本用与主 CV 完全相同的 ET 超参与折划分，按“折内映射”重算：
    - 逐目标 R²/MAE/Spearman/分级精确率/±1/Kappa（15 折平均）；
    - 分级混淆矩阵（seed=42 的 5 折验证预测拼接，每样品恰好一次）；
并更新 03_q1_改进_逐目标指标.csv、fig_q1_target_r2.png、fig_q1_grade_confusion.png，
同时把折内口径写回 03_q1_改进_结果.json（保留原 OOF 口径作为补充）。

运行：
    $env:PYTHONPATH = "D:\\github mama\\.work\\pylibs"
    D:\\miniconda\\python.exe 成果\\code\\03c_q1_consistency.py
"""
import os
import sys
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error, cohen_kappa_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_prep_common import (load_data, grade_mapping_fit, grade_mapping_apply,
                              make_folds, save_json, RANDOM_SEED, OUT, FIG)

for name in ["Microsoft YaHei", "SimHei", "SimSun"]:
    try:
        font_manager.findfont(name, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [name]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

REPEATS = [RANDOM_SEED, 1024, 2024]
N_SPLITS = 5

X, Y, styles, ids = load_data()
Xv, Yv = X.values.astype(float), Y.values.astype(float)
targets = list(Y.columns)
n_targets = len(targets)

acc = {k: [] for k in ["r2", "mae", "spearman", "grade_exact", "grade_pm1", "kappa"]}
conf_true, conf_pred = None, None
for rep, seed in enumerate(REPEATS):
    folds = make_folds(styles, N_SPLITS, seed)
    if rep == 0:
        conf_true = np.zeros((len(Xv), n_targets), dtype=int)
        conf_pred = np.zeros((len(Xv), n_targets), dtype=int)
    for fi, (tr, va) in enumerate(folds):
        imp = SimpleImputer(strategy="median", add_indicator=True).fit(Xv[tr])
        model = ExtraTreesRegressor(n_estimators=400, max_features="sqrt",
                                    min_samples_leaf=1, n_jobs=8, random_state=seed)
        model.fit(imp.transform(Xv[tr]), Yv[tr])
        pred = model.predict(imp.transform(Xv[va]))
        params = grade_mapping_fit(Yv[tr], "minmax")
        gt = grade_mapping_apply(Yv[va], params)
        gp = grade_mapping_apply(pred, params)
        if rep == 0:
            conf_true[va] = gt
            conf_pred[va] = gp
        for j in range(n_targets):
            yt, yp = Yv[va][:, j], pred[:, j]
            acc["r2"].append(r2_score(yt, yp))
            acc["mae"].append(mean_absolute_error(yt, yp))
            acc["spearman"].append(spearmanr(yt, yp).correlation)
            acc["grade_exact"].append(float(np.mean(gt[:, j] == gp[:, j])))
            acc["grade_pm1"].append(float(np.mean(np.abs(gt[:, j] - gp[:, j]) <= 1)))
            acc["kappa"].append(cohen_kappa_score(gt[:, j], gp[:, j], weights="quadratic",
                                                  labels=list(range(7))))
print("fitted 15 folds", flush=True)

rows = []
for j, target in enumerate(targets):
    rows.append({"target": target,
                 **{k: float(np.nanmean(acc[k][j::n_targets])) for k in acc}})
per_target = pd.DataFrame(rows).sort_values("r2", ascending=False)
per_target.to_csv(os.path.join(OUT, "03_q1_改进_逐目标指标.csv"), index=False, encoding="utf-8-sig")

# 逐目标 R² 图
fig, ax = plt.subplots(figsize=(14, 5))
pt = per_target.sort_values("r2")
ax.bar(range(len(pt)), pt["r2"], color="#55A868")
ax.set_xticks(range(len(pt)))
ax.set_xticklabels(pt["target"], rotation=90, fontsize=7)
ax.axhline(0, color="black", lw=0.8)
ax.set_ylabel("R²")
ax.set_title("逐目标预测 R²（ExtraTrees，15 折平均，折内口径）")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_q1_target_r2.png"), dpi=200)
plt.close(fig)

# 分级混淆矩阵（折内映射，seed=42 的 5 折拼接）
cm = np.zeros((7, 7), dtype=int)
for a, b in zip(conf_true.ravel(), conf_pred.ravel()):
    cm[int(a), int(b)] += 1
pd.DataFrame(cm, index=[f"true_{i}" for i in range(7)],
             columns=[f"pred_{i}" for i in range(7)]).to_csv(
    os.path.join(OUT, "03_q1_改进_混淆矩阵_seed42.csv"), encoding="utf-8-sig")
fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.imshow(cm, cmap="Blues")
for i in range(7):
    for j in range(7):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
ax.set_xlabel("预测等级"); ax.set_ylabel("真实等级（映射后）")
ax.set_title("问题一 0–6 分级混淆矩阵（ExtraTrees，折内映射，seed=42）")
fig.colorbar(im, ax=ax, shrink=0.8)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_q1_grade_confusion.png"), dpi=200)
plt.close(fig)
exact = float(np.mean(conf_true == conf_pred))
pm1 = float(np.mean(np.abs(conf_true - conf_pred) <= 1))
kappas = []
for j in range(n_targets):
    kappas.append(cohen_kappa_score(conf_true[:, j], conf_pred[:, j], weights="quadratic",
                                    labels=list(range(7))))
print("foldwise(seed42) exact=%.4f pm1=%.4f kappa=%.4f" % (exact, pm1, float(np.mean(kappas))), flush=True)

# 写回 JSON（保留 OOF 口径作为补充）
jp = os.path.join(OUT, "03_q1_改进_结果.json")
out = json.load(open(jp, encoding="utf-8")) if os.path.exists(jp) else {}
out["per_target_summary_foldwise"] = {
    "best_targets": per_target.head(8)[["target", "r2", "spearman"]].to_dict(orient="records"),
    "worst_targets": per_target.tail(8)[["target", "r2", "spearman"]].to_dict(orient="records"),
    "macro": {k: float(per_target[k].mean()) for k in ["r2", "mae", "spearman", "grade_exact", "grade_pm1", "kappa"]},
    "note": "逐折口径：15 折（5 折×3 次重复）逐折训练/验证、折内 min–max 映射；与主表宏观指标口径一致。",
}
out["grade_confusion_foldwise"] = {"exact": exact, "pm1": pm1, "kappa_macro": float(np.mean(kappas)),
                                   "note": "seed=42 的 5 折验证预测拼接（每样品一次），折内映射"}
save_json(out, jp)
print("updated", jp)
