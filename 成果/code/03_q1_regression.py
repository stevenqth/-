# -*- coding: utf-8 -*-

# 沙箱环境：禁止 OpenMP/线程池创建管道（避免 WinError 5）
import os as _os
for _k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    _os.environ.setdefault(_k, '1')
"""03_q1_regression.py — 问题一：化学指纹 -> 50 维感官评分（含 0–6 分级）的建模与验证。

设计对应方案文档“问题一”：
    基线：Ridge 多输出回归
    主模型：RandomForest / ExtraTrees 多输出树集成；HistGradientBoosting 逐输出集成
    分级校准：训练折上拟合 minmax（主口径）与 quantile（敏感性）两种 0–6 映射
    验证：按风格分层的 5 折 × 2 次重复交叉验证；指标为逐维 R²/MAE/Spearman 与分级
          准确率/±1 准确率/二次加权 Kappa 的宏平均
    缺失处理：折内中位数插补 + 缺失指示（主口径）；HistGB 额外比较原生 NaN 输入

运行：
    $env:PYTHONPATH = "D:\\github mama\\.work\\pylibs"
    D:\\miniconda\\python.exe 成果\\code\\03_q1_regression.py

输出：
    成果/results/03_q1_模型对比.json
    成果/results/03_q1_逐目标指标.csv
    成果/results/03_q1_预测结果_OOF.csv
    成果/results/03_q1_特征重要性.csv
    成果/results/03_q1_模型.joblib（供问题三做感官代理）
    成果/figures/fig_q1_*.png
"""
import os
import sys
import json
import time
import warnings

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_error, cohen_kappa_score
from sklearn.inspection import permutation_importance

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_prep_common import (load_data, fit_imputer, grade_mapping_fit, grade_mapping_apply,
                              classify_transforms, make_folds, save_json, RANDOM_SEED)

warnings.filterwarnings("ignore")
ROOT = r"D:\github mama"
OUT = os.path.join(ROOT, "成果", "results")
FIG = os.path.join(ROOT, "成果", "figures")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

# 中文字体
for name in ["Microsoft YaHei", "SimHei", "SimSun"]:
    try:
        font_manager.findfont(name, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [name]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

N_SPLITS = 5
REPEATS = [RANDOM_SEED, 1024]
GRADE_METHODS = ["minmax", "quantile"]


def build_models(seed):
    """返回 {模型名: 构造函数}。"""
    return {
        "Ridge": lambda: Ridge(alpha=10.0),
        "RandomForest": lambda: RandomForestRegressor(
            n_estimators=400, max_features="sqrt", min_samples_leaf=2,
            n_jobs=1, random_state=seed),
        "ExtraTrees": lambda: ExtraTreesRegressor(
            n_estimators=400, max_features="sqrt", min_samples_leaf=2,
            n_jobs=1, random_state=seed),
        "HistGBDT": lambda: MultiOutputRegressor(
            HistGradientBoostingRegressor(
                max_iter=200, learning_rate=0.06, max_leaf_nodes=15,
                l2_regularization=0.1, early_stopping=False, random_state=seed),
            n_jobs=1),
    }


def per_target_metrics(y_true, y_pred, grades_true, grades_pred):
    r2s, maes, rhos, accs, pm1s, kappas = [], [], [], [], [], []
    for j in range(y_true.shape[1]):
        yt, yp = y_true[:, j], y_pred[:, j]
        ok = np.isfinite(yt) & np.isfinite(yp)
        r2s.append(r2_score(yt[ok], yp[ok]) if ok.sum() > 2 else np.nan)
        maes.append(mean_absolute_error(yt[ok], yp[ok]) if ok.sum() > 0 else np.nan)
        if ok.sum() > 2 and np.std(yt[ok]) > 0 and np.std(yp[ok]) > 0:
            rhos.append(spearmanr(yt[ok], yp[ok]).correlation)
        else:
            rhos.append(np.nan)
        gt, gp = grades_true[:, j], grades_pred[:, j]
        accs.append(float(np.mean(gt == gp)))
        pm1s.append(float(np.mean(np.abs(gt - gp) <= 1)))
        kappas.append(cohen_kappa_score(gt, gp, weights="quadratic", labels=list(range(7))))
    return dict(r2=r2s, mae=maes, spearman=rhos, grade_exact=accs, grade_pm1=pm1s, kappa=kappas)


def summarize_folds(fold_metrics):
    """fold_metrics: list[dict[str, list]] -> 宏平均与标准差。"""
    keys = ["r2", "mae", "spearman", "grade_exact", "grade_pm1", "kappa"]
    out = {}
    for k in keys:
        vals = [np.nanmean(m[k]) for m in fold_metrics]
        out[f"{k}_mean"] = float(np.nanmean(vals))
        out[f"{k}_sd"] = float(np.nanstd(vals))
    return out


def run_cv(X, Y, styles, transform_flags):
    """返回：逐模型逐折指标、OOF 预测、全量训练所需工件。"""
    results = {}
    oof_store = {}
    t0 = time.time()

    for repeat, seed in enumerate(REPEATS):
        folds = make_folds(styles, N_SPLITS, seed)
        for model_name, ctor in build_models(seed).items():
            results.setdefault(model_name, [])
            oof_store.setdefault(model_name, [])
            for fi, (tr, va) in enumerate(folds):
                X_tr, X_va = X.iloc[tr], X.iloc[va]
                Y_tr, Y_va = Y.iloc[tr].values, Y.iloc[va].values

                imp = fit_imputer(X_tr)
                Xtr_i = imp.transform(X_tr)
                Xva_i = imp.transform(X_va)

                model = ctor()
                model.fit(Xtr_i, Y_tr)
                pred = model.predict(Xva_i)
                if pred.ndim == 1:
                    pred = pred.reshape(-1, Y_va.shape[1])

                fold = {"repeat": repeat, "fold": fi,
                        "n_train": int(len(tr)), "n_valid": int(len(va))}
                for gm in GRADE_METHODS:
                    params = grade_mapping_fit(Y_tr, gm)
                    g_true = grade_mapping_apply(Y_va, params)
                    g_pred = grade_mapping_apply(pred, params)
                    m = per_target_metrics(Y_va, pred, g_true, g_pred)
                    if gm == "minmax":
                        fold.update({k: v for k, v in m.items()})
                fold["macro"] = {k: float(np.nanmean(fold[k])) for k in
                                 ["r2", "mae", "spearman", "grade_exact", "grade_pm1", "kappa"]}
                results[model_name].append(fold)

                # 累积 OOF（repeat 级别）
                oof_store[model_name].append(pd.DataFrame(
                    pred, columns=Y.columns, index=Y.index[va]))

    summary = {}
    for model_name, folds in results.items():
        valid = [f for f in folds if "r2" in f]
        summary[model_name] = summarize_folds(valid)

    # 合并 OOF（各 repeat 平均）
    oof_avg = {}
    n_rep = len(REPEATS)
    for model_name, dfs in oof_store.items():
        acc = dfs[0].copy()
        for d in dfs[1:]:
            acc = acc.add(d, fill_value=np.nan)
        oof_avg[model_name] = acc / n_rep
    # 0–6 分级口径敏感性（minmax vs quantile，最优模型单独计算）
    return results, summary, oof_avg, time.time() - t0


def grade_sensitivity(X, Y, styles, model_name):
    """对最优模型比较两种分级映射口径的等级一致率。"""
    out = {}
    seed = REPEATS[0]
    folds = make_folds(styles, N_SPLITS, seed)
    ctor = build_models(seed)[model_name]
    rows = {m: [] for m in GRADE_METHODS}
    for tr, va in folds:
        imp = fit_imputer(X.iloc[tr])
        model = ctor()
        model.fit(imp.transform(X.iloc[tr]), Y.iloc[tr].values)
        pred = model.predict(imp.transform(X.iloc[va]))
        for gm in GRADE_METHODS:
            params = grade_mapping_fit(Y.iloc[tr].values, gm)
            gt = grade_mapping_apply(Y.iloc[va].values, params)
            gp = grade_mapping_apply(pred, params)
            rows[gm].append((float(np.mean(gt == gp)), float(np.mean(np.abs(gt - gp) <= 1))))
    for gm, vals in rows.items():
        arr = np.array(vals)
        out[gm] = {"grade_exact_mean": float(arr[:, 0].mean()), "grade_pm1_mean": float(arr[:, 1].mean())}
    # 两口径之间的等级一致性（同一折的预测）
    out["note"] = "0–6 原始整数不可恢复，两种映射均为假设；minmax 保留分布形状，quantile 为等频分级。"
    return out


def make_figures(summary, results, oof, Y, styles, X, best_model_name):
    # 图 1：模型对比
    names = list(summary.keys())
    metrics = [("spearman_mean", "平均 Spearman 相关"), ("mae_mean", "平均 MAE"),
               ("grade_pm1_mean", "分级 ±1 准确率"), ("kappa_mean", "二次加权 Kappa")]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    for ax, (key, label) in zip(axes, metrics):
        vals = [summary[n][key] for n in names]
        errs = [summary[n][key.replace("_mean", "_sd")] for n in names]
        ax.bar(names, vals, yerr=errs, capsize=3, color="#4C72B0")
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_q1_model_comparison.png"), dpi=200)
    plt.close(fig)

    # 图 2：预测-真实散点网格（OOF，最优模型）
    pred_df = oof[best_model_name]
    fig, axes = plt.subplots(5, 10, figsize=(22, 11))
    for j, col in enumerate(Y.columns):
        ax = axes[j // 10, j % 10]
        ax.scatter(Y[col].values, pred_df[col].values, s=5, alpha=0.5, color="#DD8452")
        lo = min(Y[col].min(), pred_df[col].min())
        hi = max(Y[col].max(), pred_df[col].max())
        ax.plot([lo, hi], [lo, hi], "--", color="gray", lw=0.8)
        ax.set_title(col, fontsize=7)
        ax.tick_params(labelsize=6)
    fig.suptitle(f"问题一 50 维感官评分：预测值与真实值（OOF，{best_model_name}）", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(FIG, "fig_q1_scatter_grid.png"), dpi=170)
    plt.close(fig)

    # 图 3：逐维 R²（最优模型）
    per_target = pd.read_csv(os.path.join(OUT, "03_q1_逐目标指标.csv"))
    sub = per_target[per_target["model"] == best_model_name].sort_values("r2")
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(len(sub)), sub["r2"].values, color="#55A868")
    ax.set_xticks(range(len(sub)))
    ax.set_xticklabels(sub["target"].values, rotation=90, fontsize=7)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("R²")
    ax.set_title(f"逐维预测 R²（{best_model_name}，交叉验证均值）")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_q1_target_r2.png"), dpi=200)
    plt.close(fig)

    # 图 4：分级混淆矩阵（全部折合并，最优模型、minmax 口径）
    # 使用最后一折的 OOF 近似：直接由预测与真值映射统计（重复一次训练成本低）
    folds = make_folds(styles, N_SPLITS, REPEATS[0])
    ctor = build_models(REPEATS[0])[best_model_name]
    gt_all, gp_all = [], []
    for tr, va in folds:
        imp = fit_imputer(X.iloc[tr])
        model = ctor()
        model.fit(imp.transform(X.iloc[tr]), Y.iloc[tr].values)
        pred = model.predict(imp.transform(X.iloc[va]))
        params = grade_mapping_fit(Y.iloc[tr].values, "minmax")
        gt_all.append(grade_mapping_apply(Y.iloc[va].values, params))
        gp_all.append(grade_mapping_apply(pred, params))
    gt = np.vstack(gt_all).ravel()
    gp = np.vstack(gp_all).ravel()
    cm = np.zeros((7, 7), dtype=int)
    for a, b in zip(gt, gp):
        cm[int(a), int(b)] += 1
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(7):
        for j in range(7):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    ax.set_xlabel("预测等级")
    ax.set_ylabel("真实等级（映射后）")
    ax.set_title("问题一 0–6 分级混淆矩阵（minmax 口径，全部验证折合并）")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_q1_confusion.png"), dpi=200)
    plt.close(fig)

    # 图 5：按风格的误差（最优模型，逐样品平均绝对误差）
    rows = []
    for (tr, va) in folds:
        imp = fit_imputer(X.iloc[tr])
        model = ctor()
        model.fit(imp.transform(X.iloc[tr]), Y.iloc[tr].values)
        pred = model.predict(imp.transform(X.iloc[va]))
        mae_sample = np.nanmean(np.abs(pred - Y.iloc[va].values), axis=1)
        for idx, v in zip(Y.index[va], mae_sample):
            rows.append(dict(style=styles.iloc[idx], mae=v))
    err_df = pd.DataFrame(rows)
    order = err_df.groupby("style")["mae"].median().sort_values().index
    fig, ax = plt.subplots(figsize=(14, 5))
    data = [err_df[err_df["style"] == s]["mae"].values for s in order]
    ax.boxplot(data, labels=list(order), showfliers=False)
    ax.set_xticklabels(order, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("样品平均绝对误差")
    ax.set_title(f"按啤酒风格的问题一误差（{best_model_name}）")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_q1_error_by_style.png"), dpi=200)
    plt.close(fig)

    # 图 6：特征重要性 Top20（置换重要性）
    fi_path = os.path.join(OUT, "03_q1_特征重要性.csv")
    fi = pd.read_csv(fi_path).head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.barh(fi["feature"], fi["importance_mean"], xerr=fi["importance_sd"],
            color="#C44E52", capsize=2)
    ax.set_xlabel("置换重要性（R² 下降）")
    ax.set_title("问题一 特征重要性 Top20（置换重要性）")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_q1_importance.png"), dpi=200)
    plt.close(fig)


def main():
    t_start = time.time()
    X, Y, styles, ids = load_data()
    transforms = classify_transforms()

    results, summary, oof, elapsed = run_cv(X, Y, styles, transforms)
    best_model_name = max(summary, key=lambda k: summary[k]["spearman_mean"] - 0.1 * summary[k]["mae_mean"])

    # 逐目标指标（全部折合并的宏平均）
    rows = []
    for model_name, folds in results.items():
        for j, target in enumerate(Y.columns):
            vals = {k: np.nanmean([f[k][j] for f in folds if "r2" in f]) for k in
                    ["r2", "mae", "spearman", "grade_exact", "grade_pm1", "kappa"]}
            rows.append(dict(model=model_name, target=target, **vals))
    per_target = pd.DataFrame(rows)
    per_target.to_csv(os.path.join(OUT, "03_q1_逐目标指标.csv"), index=False, encoding="utf-8-sig")

    # 模型对比表
    comp = pd.DataFrame([dict(model=k, **v) for k, v in summary.items()])
    comp.to_csv(os.path.join(OUT, "03_q1_模型对比.csv"), index=False, encoding="utf-8-sig")

    # OOF 预测长表（最优模型）
    best_pred = oof[best_model_name]
    params_all = grade_mapping_fit(Y.values, "minmax")
    long_rows = []
    for j, target in enumerate(Y.columns):
        gt = grade_mapping_apply(Y.values[:, [j]], grade_mapping_fit(Y.values, "minmax"))[:, 0]
        gp = grade_mapping_apply(best_pred.values[:, [j]], grade_mapping_fit(Y.values, "minmax"))[:, 0]
        for i in range(len(Y)):
            long_rows.append(dict(beer=int(ids["beer"].iloc[i]), style=styles.iloc[i],
                                  target=target, true=float(Y.iloc[i, j]),
                                  pred=float(best_pred.iloc[i, j]),
                                  grade_true=int(gt[i]), grade_pred=int(gp[i])))
    pd.DataFrame(long_rows).to_csv(os.path.join(OUT, "03_q1_预测结果_OOF.csv"), index=False, encoding="utf-8-sig")

    # 全量训练 + 特征重要性 + 保存模型
    imp_full = fit_imputer(X)
    X_full = imp_full.transform(X)
    final_model = build_models(RANDOM_SEED)[best_model_name]()
    final_model.fit(X_full, Y.values)
    t_fit = time.time()

    if best_model_name in ("RandomForest", "ExtraTrees"):
        base_fi = final_model.feature_importances_
    elif best_model_name == "Ridge":
        base_fi = np.abs(final_model.coef_).mean(axis=0)
    else:
        base_fi = np.mean([np.abs(est.coef_[0]) if hasattr(est, "coef_") else 0.0
                           for est in final_model.estimators_], axis=0) if hasattr(final_model, "estimators_") else np.zeros(X.shape[1])

    # 置换重要性（统一口径，便于跨模型比较）
    perm = permutation_importance(final_model, X_full, Y.values, scoring="r2",
                                  n_repeats=5, random_state=RANDOM_SEED, n_jobs=1)
    fi = pd.DataFrame({"feature": X.columns,
                       "importance_mean": perm.importances_mean,
                       "importance_sd": perm.importances_std,
                       "builtin_importance": base_fi})
    fi = fi.sort_values("importance_mean", ascending=False)
    fi.to_csv(os.path.join(OUT, "03_q1_特征重要性.csv"), index=False, encoding="utf-8-sig")

    grade_sens = grade_sensitivity(X, Y, styles, best_model_name)

    # 保存模型工件（供问题三代理使用）
    joblib.dump({
        "model_name": best_model_name,
        "imputer": imp_full,
        "model": final_model,
        "feature_columns": list(X.columns),
        "target_columns": list(Y.columns),
        "grade_mapping": grade_mapping_fit(Y.values, "minmax"),
        "grade_mapping_quantile": grade_mapping_fit(Y.values, "quantile"),
        "random_seed": RANDOM_SEED,
    }, os.path.join(OUT, "03_q1_模型.joblib"), compress=3)

    # 汇总落盘
    out = {
        "config": {"n_splits": N_SPLITS, "repeats": REPEATS, "grade_methods": GRADE_METHODS,
                   "missing_strategy": "折内中位数插补 + 缺失指示",
                   "models": list(summary.keys())},
        "models": summary,
        "best_model": best_model_name,
        "selection_rule": "spearman_mean - 0.1 * mae_mean 最大",
        "grade_sensitivity": grade_sens,
        "runtime_seconds": {"cv_total": round(elapsed, 1),
                            "full_fit_and_importance": round(time.time() - t_fit, 1),
                            "total": round(time.time() - t_start, 1)},
        "env": {"python": sys.version.split()[0],
                "numpy": np.__version__, "pandas": pd.__version__},
        "data": {"n_samples": int(len(X)), "n_features": int(X.shape[1]), "n_targets": int(Y.shape[1])},
    }
    save_json(out, os.path.join(OUT, "03_q1_模型对比.json"))

    make_figures(summary, results, oof, Y, styles, X, best_model_name)

    print(json.dumps({"best_model": best_model_name,
                      "summary": summary[best_model_name],
                      "runtime": out["runtime_seconds"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
