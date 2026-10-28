# -*- coding: utf-8 -*-
"""03_q1_baseline.py — 问题一：论文 10 模型基线复现（单次 70/30 风格分层划分 + 5 折 CV 调参）。

对应决策：1A（论文模型作基线骨架）、2A（复现 10 模型对照表）、3A（解释工具后续用置换重要性/PDP）。
论文口径（Schreurs et al., Nat Commun 2024）：
    - 化学特征：对数正态属性取 log（本数据已为逐列变换后的值，见数据字典 4.1）；
    - 缺失：按属性均值填补（0.1% 缺失）；
    - 特征：按训练集均值/标准差标准化；
    - 感官目标：逐品评员 z 分数再平均（4.csv 即该值）；
    - 划分：70% 训练 / 30% 测试，按风格分层；
    - 模型：LR+Lasso+PLSR / AdaBoost+ET+GBR+RF+XGBoost(此处 HistGBDT 替代) / SVR / ANN(MLP)；
    - 调参：训练集 5 折 CV 网格搜索，评价指标 R²（多输出取均匀平均）；
    - 报告：测试集多输出 R²、逐维 R² 的平均排名（论文 Table 1 口径）。

为在本机可计算，做两处**显式简化**（论文中会注明）：
    1. XGBoost 用 HistGradientBoosting 替代（本机无 xgboost、无外网）；
    2. LR/Lasso 的“一阶交互项”限制为“训练集上平均 |Spearman| 前 60 个特征的两两交互”，
       而非论文的全部 231 特征两两交互（论文特征数约 2.7 万，本机不可承受）。
    3. 平均排名用“多输出模型的逐维 R² 排名”近似论文的“逐属性单独建模排名”，论文中注明。

运行：
    $env:PYTHONPATH = "D:\\github mama\\.work\\pylibs"
    D:\\miniconda\\python.exe 成果\\code\\03_q1_baseline.py
输出：
    成果/results/03_q1_基线结果.csv / .json
    成果/results/03_q1_基线逐维R2.csv
    成果/figures/fig_q1_baseline_comparison.png
"""
import os
import sys
import json
import time
import warnings

import os as _os
for _k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    _os.environ.setdefault(_k, "1")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import (AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor,
                              RandomForestRegressor, HistGradientBoostingRegressor)
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_prep_common import load_data, save_json, RANDOM_SEED, OUT, FIG

warnings.filterwarnings("ignore")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

for name in ["Microsoft YaHei", "SimHei", "SimSun"]:
    try:
        font_manager.findfont(name, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [name]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

# 论文 Table 1 的品评组 R² 参照（XGBoost 行用于替代后的对照说明）
PAPER_R2 = {
    "AdaBoost": 0.21, "ANN": 0.14, "Extra Trees": 0.22, "Gradient Boosting": 0.21,
    "Lasso": 0.05, "Linear Regression": -4.13, "PLSR": -0.25, "Random Forest": 0.22,
    "SVR": 0.18, "XGBoost": 0.22,
}
PAPER_RANK = {
    "AdaBoost": 2.88, "ANN": 5.60, "Extra Trees": 3.02, "Gradient Boosting": 3.42,
    "Lasso": 4.94, "Linear Regression": 7.88, "PLSR": 7.56, "Random Forest": 2.86,
    "SVR": 6.50, "XGBoost": 4.12,
}


def impute_and_scale(X_train, X_test):
    imp = SimpleImputer(strategy="mean").fit(X_train)
    sc = StandardScaler().fit(imp.transform(X_train))
    return sc.transform(imp.transform(X_train)), sc.transform(imp.transform(X_test)), imp, sc


def select_interaction_features(X_train, Y_train, n_base=60):
    """训练集上逐特征对 50 个目标计算 |Spearman|，取平均最大的 n_base 个。"""
    scores = []
    for j in range(X_train.shape[1]):
        rs = []
        x = X_train[:, j]
        for k in range(Y_train.shape[1]):
            y = Y_train[:, k]
            if np.std(x) > 0 and np.std(y) > 0:
                rs.append(abs(spearmanr(x, y).correlation))
        scores.append(np.nanmean(rs) if rs else 0.0)
    order = np.argsort(-np.array(scores))
    return order[:n_base]


def add_pairwise_interactions(X, idx):
    base = X[:, idx]
    cols = [X]
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            cols.append((base[:, a] * base[:, b])[:, None])
    return np.hstack(cols)


def make_models():
    """返回 {显示名: (estimator, param_grid)}；均为多输出模型。"""
    models = {}
    models["Linear Regression"] = (LinearRegression(), {})
    models["Lasso"] = (
        MultiOutputRegressor(Lasso(max_iter=5000, random_state=RANDOM_SEED), n_jobs=1),
        {"estimator__alpha": [0.01, 0.1]},
    )
    models["PLSR"] = (PLSRegression(), {"n_components": [2, 5, 10]})
    models["AdaBoost"] = (
        MultiOutputRegressor(AdaBoostRegressor(random_state=RANDOM_SEED), n_jobs=1),
        {"estimator__n_estimators": [150], "estimator__learning_rate": [0.05]},
    )
    models["Extra Trees"] = (
        ExtraTreesRegressor(n_estimators=300, max_features="sqrt",
                            n_jobs=8, random_state=RANDOM_SEED),
        {"min_samples_leaf": [1, 3]},
    )
    models["Gradient Boosting"] = (
        MultiOutputRegressor(GradientBoostingRegressor(random_state=RANDOM_SEED), n_jobs=1),
        {"estimator__n_estimators": [150], "estimator__learning_rate": [0.05]},
    )
    models["Random Forest"] = (
        RandomForestRegressor(n_estimators=300, max_features="sqrt",
                              n_jobs=8, random_state=RANDOM_SEED),
        {"min_samples_leaf": [1, 3]},
    )
    models["HistGBDT (替代 XGBoost)"] = (
        MultiOutputRegressor(HistGradientBoostingRegressor(random_state=RANDOM_SEED), n_jobs=1),
        {"estimator__max_iter": [200], "estimator__learning_rate": [0.05, 0.1]},
    )
    models["SVR"] = (
        MultiOutputRegressor(SVR(), n_jobs=1),
        {"estimator__C": [1, 10]},
    )
    models["ANN (MLP)"] = (
        MLPRegressor(max_iter=1500, alpha=0.01, early_stopping=True, n_iter_no_change=25, random_state=RANDOM_SEED),
        {"hidden_layer_sizes": [(64,), (128, 64)]},
    )
    return models


def main():
    t0 = time.time()
    X, Y, styles, ids = load_data()
    Xv = X.values.astype(float)
    Yv = Y.values.astype(float)

    # 70/30 风格分层划分
    idx = np.arange(len(Xv))
    tr_idx, te_idx = train_test_split(idx, test_size=0.30, random_state=RANDOM_SEED,
                                      stratify=styles.values)
    X_train, X_test = Xv[tr_idx], Xv[te_idx]
    Y_train, Y_test = Yv[tr_idx], Yv[te_idx]

    # 论文预处理：均值填补 + 训练集标准化
    X_train_s, X_test_s, imp, sc = impute_and_scale(X_train, X_test)

    # 交互特征（仅 LR / Lasso 使用）
    inter_idx = select_interaction_features(X_train_s, Y_train, n_base=60)
    X_train_i = add_pairwise_interactions(X_train_s, inter_idx)
    X_test_i = add_pairwise_interactions(X_test_s, inter_idx)

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    rows, per_dim = [], {}
    detail = {}
    for name, (est, grid) in make_models().items():
        start = time.time()
        use_inter = name in ("Linear Regression", "Lasso")
        Xtr = X_train_i if use_inter else X_train_s
        Xte = X_test_i if use_inter else X_test_s
        if grid:
            gs = GridSearchCV(est, grid, cv=cv, scoring="r2", n_jobs=8)
            gs.fit(Xtr, Y_train)
            best = gs.best_estimator_
            best_params = gs.best_params_
            cv_r2 = float(gs.best_score_)
        else:
            best = est.fit(Xtr, Y_train)
            best_params = {}
            cv_r2 = np.nan
        pred = best.predict(Xte)
        if pred.ndim == 1:
            pred = pred.reshape(-1, Y_test.shape[1])
        r2s = np.array([r2_score(Y_test[:, j], pred[:, j]) for j in range(Y_test.shape[1])])
        maes = np.array([mean_absolute_error(Y_test[:, j], pred[:, j]) for j in range(Y_test.shape[1])])
        per_dim[name] = r2s
        detail[name] = {
            "best_params": best_params,
            "cv_r2": cv_r2,
            "test_r2_mean": float(np.mean(r2s)),
            "test_r2_median": float(np.median(r2s)),
            "test_mae_mean": float(np.mean(maes)),
            "test_r2_per_target": r2s.tolist(),
            "n_features": int(Xtr.shape[1]),
            "seconds": round(time.time() - start, 1),
        }
        rows.append({"model": name, "n_features": int(Xtr.shape[1]),
                     "cv_r2": cv_r2, "test_r2_mean": float(np.mean(r2s)),
                     "test_r2_median": float(np.median(r2s)),
                     "test_mae_mean": float(np.mean(maes)),
                     "seconds": round(time.time() - start, 1),
                     "best_params": json.dumps(best_params, ensure_ascii=False)})
        print(f"  {name}: test R2={np.mean(r2s):.3f}, cv R2={cv_r2 if np.isfinite(cv_r2) else float('nan'):.3f}, {time.time()-start:.1f}s", flush=True)

    res = pd.DataFrame(rows)
    # 平均排名（逐维 R²，1=最好）
    R = pd.DataFrame(per_dim)  # index=target, columns=model
    ranks = R.rank(axis=1, ascending=False, method="average")
    res["avg_rank"] = [float(ranks[m].mean()) for m in res["model"]]
    res["paper_panel_r2"] = res["model"].map(PAPER_R2)
    res["paper_panel_rank"] = res["model"].map(PAPER_RANK)
    res.to_csv(os.path.join(OUT, "03_q1_基线结果.csv"), index=False, encoding="utf-8-sig")
    R.to_csv(os.path.join(OUT, "03_q1_基线逐维R2.csv"), encoding="utf-8-sig")

    # 图：测试 R² / CV R² / 平均排名
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    names = res["model"].tolist()
    for ax, col, title in zip(axes, ["test_r2_mean", "cv_r2", "avg_rank"],
                              ["测试集多输出 R²（越大越好）", "训练集 5 折 CV R²",
                               "逐维平均排名（越小越好）"]):
        vals = res[col].values.astype(float)
        ax.barh(names[::-1], vals[::-1], color="#4C72B0")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_q1_baseline_comparison.png"), dpi=200)
    plt.close(fig)

    # 论文对照图：本文 test R² vs 论文 panel R²
    fig, ax = plt.subplots(figsize=(9, 5.6))
    sub = res.dropna(subset=["paper_panel_r2"]).sort_values("paper_panel_r2").reset_index(drop=True)
    ax.scatter(sub["paper_panel_r2"], sub["test_r2_mean"], s=45, color="#C44E52", zorder=3)
    for i, (_, r) in enumerate(sub.iterrows()):
        dx, dy = (8, 6) if i % 2 == 0 else (-52, -12)
        ax.annotate(r["model"], (r["paper_panel_r2"], r["test_r2_mean"]), fontsize=8,
                    xytext=(dx, dy), textcoords="offset points")
    lim = [-4.5, 0.5]
    ax.plot(lim, lim, "--", color="gray", lw=0.8, label="y = x（完全一致）")
    ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlim(lim); ax.set_ylim(-1.2, 0.5)
    ax.set_xlabel("论文 Table 1：品评组多输出 R²")
    ax.set_ylabel("本文：测试集多输出 R²")
    ax.set_title("基线复现与论文 Table 1 对照（XGBoost→HistGBDT、ANN 为简化配置）")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_q1_baseline_vs_paper.png"), dpi=200)
    plt.close(fig)

    out = {
        "config": {
            "split": "70/30, stratified by tasting_category_fine, random_state=42",
            "n_train": int(len(tr_idx)), "n_test": int(len(te_idx)),
            "imputation": "train mean", "scaling": "train StandardScaler",
            "cv": "5-fold KFold on training set; GridSearchCV scoring=r2",
            "interactions": "LR/Lasso: pairwise products of top-60 features by mean |Spearman| (train only)",
            "xgb_substitute": "HistGradientBoostingRegressor",
            "rank_definition": "per-target test R² rank (1=best) averaged over 50 targets; multi-output model per-descriptor R² (paper uses per-attribute models)",
        },
        "paper_table1": {"panel_r2": PAPER_R2, "panel_rank": PAPER_RANK},
        "results": res.drop(columns=["best_params"]).to_dict(orient="records"),
        "detail": detail,
        "split": {"train_index": tr_idx.tolist(), "test_index": te_idx.tolist()},
        "runtime_seconds": round(time.time() - t0, 1),
        "env": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__},
    }
    save_json(out, os.path.join(OUT, "03_q1_基线结果.json"))
    print("saved results; total", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
