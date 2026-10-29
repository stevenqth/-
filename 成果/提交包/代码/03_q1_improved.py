# -*- coding: utf-8 -*-
"""03_q1_improved.py — 问题一：改进版模型评估（阶段 2B）。

相对论文基线的改进（对应决策 1A/2A/3A/6A）：
    1. 重复分层交叉验证（5 折 × 3 次，按 tasting_category_fine 分层），报告均值±标准差；
    2. 缺失处理敏感性：训练均值（论文做法）/ 中位数 / 中位数+缺失指示（主口径）/
       IterativeImputer / HistGBDT 原生 NaN；
    3. 0–6 分级层：min–max（主口径）与分位等频（备选）两种映射，报告精确率、±1 准确率、
       二次加权 Kappa；并做映射口径敏感性对比；
    4. 逐风格/逐目标误差分解、单样品预测不确定性（RF 树间方差 vs OOF 误差）；
    5. 解释性：置换重要性 Top20、PDP（overall / A_esters_all 两个目标），并与 8.xlsx
       的论文 GBM MDI 排名做外部对照（注明来自 RateBeer 喜好模型，非本题目标）。

运行：
    $env:PYTHONPATH = "D:\\github mama\\.work\\pylibs"
    D:\\miniconda\\python.exe 成果\\code\\03_q1_improved.py
输出（成果/results/、成果/figures/）：
    03_q1_改进_逐折指标.csv / 03_q1_改进_逐目标指标.csv / 03_q1_改进_缺失策略.csv
    03_q1_改进_特征重要性.csv / 03_q1_改进_OOF预测.csv / 03_q1_改进_结果.json
    03_q1_模型.joblib（FINAL 模型，供问题三感官代理）
    fig_q1_repeated_cv.png / missing_strategies / grade_confusion / per_style_error /
    target_r2 / importance_top20 / pdp_overall / uncertainty
"""
import os
import sys
import json
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_error, cohen_kappa_score
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_prep_common import (load_data, grade_mapping_fit, grade_mapping_apply,
                              make_folds, save_json, RANDOM_SEED, OUT, FIG)

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

REPEATS = [RANDOM_SEED, 1024, 2024]
N_SPLITS = 5
GRADE_METHODS = ["minmax", "quantile"]


def make_models():
    return {
        "Ridge": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=10.0))]),
        "RandomForest": RandomForestRegressor(n_estimators=400, max_features="sqrt",
                                               min_samples_leaf=3, n_jobs=8, random_state=RANDOM_SEED),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=400, max_features="sqrt",
                                          min_samples_leaf=1, n_jobs=8, random_state=RANDOM_SEED),
        "HistGBDT": MultiOutputRegressor(HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
            l2_regularization=0.1, early_stopping=False, random_state=RANDOM_SEED), n_jobs=1),
    }


def per_target_metrics(y_true, y_pred, g_true, g_pred):
    r2s, maes, rhos, accs, pm1s, kappas = [], [], [], [], [], []
    for j in range(y_true.shape[1]):
        yt, yp = y_true[:, j], y_pred[:, j]
        ok = np.isfinite(yt) & np.isfinite(yp)
        r2s.append(r2_score(yt[ok], yp[ok]) if ok.sum() > 2 else np.nan)
        maes.append(mean_absolute_error(yt[ok], yp[ok]) if ok.sum() else np.nan)
        if ok.sum() > 2 and np.std(yt[ok]) > 0 and np.std(yp[ok]) > 0:
            rhos.append(spearmanr(yt[ok], yp[ok]).correlation)
        else:
            rhos.append(np.nan)
        accs.append(float(np.mean(g_true[:, j] == g_pred[:, j])))
        pm1s.append(float(np.mean(np.abs(g_true[:, j] - g_pred[:, j]) <= 1)))
        kappas.append(cohen_kappa_score(g_true[:, j], g_pred[:, j], weights="quadratic",
                                        labels=list(range(7))))
    return dict(r2=r2s, mae=maes, spearman=rhos, grade_exact=accs, grade_pm1=pm1s, kappa=kappas)


def macro(m):
    return {k: float(np.nanmean(v)) for k, v in m.items()}


def main():
    t0 = time.time()
    X, Y, styles, ids = load_data()
    Xv, Yv = X.values.astype(float), Y.values.astype(float)
    targets = list(Y.columns)
    n_targets = len(targets)

    # ---------------- 主实验：重复分层 CV ----------------
    fold_rows = []
    oof = {}  # model -> {repeat: DataFrame(index=sample, columns=target)}
    for model_name, proto in make_models().items():
        oof[model_name] = {}
        for seed in REPEATS:
            folds = make_folds(styles, N_SPLITS, seed)
            preds = pd.DataFrame(index=Y.index, columns=Y.columns, dtype=float)
            for fi, (tr, va) in enumerate(folds):
                X_tr, X_va = Xv[tr], Xv[va]
                Y_tr, Y_va = Yv[tr], Yv[va]
                imp = SimpleImputer(strategy="median", add_indicator=True).fit(X_tr)
                Xtr_i, Xva_i = imp.transform(X_tr), imp.transform(X_va)
                if model_name == "Ridge":
                    # 标准化后裁剪：少量特征在训练折内近乎常量，验证样品会产生极端 z 值（>1000），
                    # 导致线性模型外推爆炸；裁剪到 ±5 是稳定化处理（论文中注明）。
                    sc = StandardScaler().fit(Xtr_i)
                    Xtr_i = np.clip(sc.transform(Xtr_i), -5.0, 5.0)
                    Xva_i = np.clip(sc.transform(Xva_i), -5.0, 5.0)
                    model = Ridge(alpha=10.0)
                else:
                    model = make_models()[model_name]
                    model.set_params(**({"random_state": seed} if model_name in ("RandomForest", "ExtraTrees") else {}))
                model.fit(Xtr_i, Y_tr)
                pred = model.predict(Xva_i)
                if pred.ndim == 1:
                    pred = pred.reshape(-1, n_targets)
                preds.iloc[va, :] = pred
                metrics = {}
                for gm in GRADE_METHODS:
                    params = grade_mapping_fit(Y_tr, gm)
                    g_true = grade_mapping_apply(Y_va, params)
                    g_pred = grade_mapping_apply(pred, params)
                    m = per_target_metrics(Y_va, pred, g_true, g_pred)
                    metrics[gm] = m
                row = {"model": model_name, "seed": seed, "fold": fi,
                       "r2": float(np.nanmean(metrics["minmax"]["r2"])),
                       "mae": float(np.nanmean(metrics["minmax"]["mae"])),
                       "spearman": float(np.nanmean(metrics["minmax"]["spearman"])),
                       "grade_exact_minmax": float(np.nanmean(metrics["minmax"]["grade_exact"])),
                       "grade_pm1_minmax": float(np.nanmean(metrics["minmax"]["grade_pm1"])),
                       "kappa_minmax": float(np.nanmean(metrics["minmax"]["kappa"])),
                       "grade_exact_quantile": float(np.nanmean(metrics["quantile"]["grade_exact"])),
                       "grade_pm1_quantile": float(np.nanmean(metrics["quantile"]["grade_pm1"])),
                       "kappa_quantile": float(np.nanmean(metrics["quantile"]["kappa"])),
                       "n_train": int(len(tr)), "n_valid": int(len(va))}
                fold_rows.append(row)
                print(f"  {model_name} seed={seed} fold={fi} R2={row['r2']:.3f}", flush=True)
            oof[model_name][seed] = preds

    folds_df = pd.DataFrame(fold_rows)
    folds_df.to_csv(os.path.join(OUT, "03_q1_改进_逐折指标.csv"), index=False, encoding="utf-8-sig")
    keys = ["r2", "mae", "spearman", "grade_exact_minmax", "grade_pm1_minmax", "kappa_minmax",
            "grade_exact_quantile", "grade_pm1_quantile", "kappa_quantile"]
    comp = folds_df.groupby("model")[keys].agg(["mean", "std"]).round(4)
    comp.columns = [f"{a}_{b}" for a, b in comp.columns]
    comp = comp.reset_index()
    comp.to_csv(os.path.join(OUT, "03_q1_改进_模型对比.csv"), index=False, encoding="utf-8-sig")
    best_model = comp.sort_values("r2_mean", ascending=False).iloc[0]["model"]
    print("best model:", best_model, flush=True)

    # ---------------- 缺失处理敏感性（RF / HistGBDT） ----------------
    miss_rows = []
    for model_name in ["RandomForest", "HistGBDT"]:
        for strat in ["mean", "median", "median_indicator", "iterative", "native_nan"]:
            if strat == "native_nan" and model_name != "HistGBDT":
                continue
            if strat == "iterative" and model_name == "HistGBDT":
                continue  # 迭代插补对 50 输出模型过慢，且 RF 已覆盖该策略
            for seed in [RANDOM_SEED]:
                folds = make_folds(styles, N_SPLITS, seed)
                r2s, maes = [], []
                for tr, va in folds:
                    X_tr, X_va = Xv[tr], Xv[va]
                    Y_tr, Y_va = Yv[tr], Yv[va]
                    if strat == "mean":
                        imp = SimpleImputer(strategy="mean").fit(X_tr)
                        Xtr, Xva = imp.transform(X_tr), imp.transform(X_va)
                    elif strat == "median":
                        imp = SimpleImputer(strategy="median").fit(X_tr)
                        Xtr, Xva = imp.transform(X_tr), imp.transform(X_va)
                    elif strat == "median_indicator":
                        imp = SimpleImputer(strategy="median", add_indicator=True).fit(X_tr)
                        Xtr, Xva = imp.transform(X_tr), imp.transform(X_va)
                    elif strat == "iterative":
                        imp = IterativeImputer(max_iter=10, random_state=RANDOM_SEED).fit(X_tr)
                        Xtr, Xva = imp.transform(X_tr), imp.transform(X_va)
                    else:  # native_nan
                        Xtr, Xva = X_tr, X_va
                    model = make_models()[model_name]
                    model.fit(Xtr, Y_tr)
                    pred = model.predict(Xva)
                    if pred.ndim == 1:
                        pred = pred.reshape(-1, n_targets)
                    r2s.append(float(np.nanmean([r2_score(Y_va[:, j], pred[:, j]) for j in range(n_targets)])))
                    maes.append(float(np.nanmean([mean_absolute_error(Y_va[:, j], pred[:, j]) for j in range(n_targets)])))
                miss_rows.append({"model": model_name, "strategy": strat,
                                  "r2_mean": float(np.mean(r2s)), "r2_sd": float(np.std(r2s)),
                                  "mae_mean": float(np.mean(maes))})
                print(f"  missing {model_name}/{strat}: R2={np.mean(r2s):.3f}", flush=True)
    miss_df = pd.DataFrame(miss_rows)
    miss_df.to_csv(os.path.join(OUT, "03_q1_改进_缺失策略.csv"), index=False, encoding="utf-8-sig")

    # ---------------- OOF 合并（各 repeat 平均） ----------------
    oof_avg = {}
    for model_name, seed_dict in oof.items():
        acc = None
        for seed, df in seed_dict.items():
            acc = df if acc is None else acc.add(df, fill_value=np.nan)
        oof_avg[model_name] = acc / len(seed_dict)
    best_oof = oof_avg[best_model]

    # 逐目标指标（最优模型）
    rows = []
    for j, target in enumerate(targets):
        yt, yp = Yv[:, j], best_oof.iloc[:, j].values
        ok = np.isfinite(yt) & np.isfinite(yp)
        params = grade_mapping_fit(Yv, "minmax")
        gt = grade_mapping_apply(Yv[:, [j]], params)[:, 0]
        gp = grade_mapping_apply(best_oof.iloc[:, [j]].values, params)[:, 0]
        rows.append({"target": target,
                     "r2": float(r2_score(yt[ok], yp[ok])),
                     "mae": float(mean_absolute_error(yt[ok], yp[ok])),
                     "spearman": float(spearmanr(yt[ok], yp[ok]).correlation),
                     "grade_exact": float(np.mean(gt == gp)),
                     "grade_pm1": float(np.mean(np.abs(gt - gp) <= 1)),
                     "kappa": float(cohen_kappa_score(gt, gp, weights="quadratic", labels=list(range(7))))})
    per_target = pd.DataFrame(rows).sort_values("r2", ascending=False)
    per_target.to_csv(os.path.join(OUT, "03_q1_改进_逐目标指标.csv"), index=False, encoding="utf-8-sig")

    # OOF 预测保存（最优模型，宽表）
    out_oof = pd.concat([ids.reset_index(drop=True), best_oof.reset_index(drop=True)], axis=1)
    out_oof.to_csv(os.path.join(OUT, "03_q1_改进_OOF预测.csv"), index=False, encoding="utf-8-sig")

    # ---------------- 不确定性（RF 树间方差） ----------------
    imp_full = SimpleImputer(strategy="median", add_indicator=True).fit(Xv)
    X_full = imp_full.transform(Xv)
    final_rf = RandomForestRegressor(n_estimators=400, max_features="sqrt",
                                     min_samples_leaf=3, n_jobs=8, random_state=RANDOM_SEED)
    final_rf.fit(X_full, Yv)
    missing_cols = [c for c in X.columns if X[c].isna().any()]
    feat_names_imp = list(X.columns) + [f"missing_{c}" for c in missing_cols]
    assert len(feat_names_imp) == X_full.shape[1], (len(feat_names_imp), X_full.shape[1])
    tree_preds = np.stack([t.predict(X_full) for t in final_rf.estimators_], axis=0)  # (trees, n, targets)
    unc = tree_preds.std(axis=0, ddof=1)  # (n, targets)
    sample_unc = unc.mean(axis=1)
    sample_mae = np.abs(best_oof.values - Yv).mean(axis=1)
    unc_corr = float(spearmanr(sample_unc, sample_mae).correlation)

    # 逐风格误差
    style_err = pd.DataFrame({"style": styles.values, "mae_sample": sample_mae, "unc": sample_unc})
    style_stats = style_err.groupby("style")["mae_sample"].agg(["mean", "median", "count"]).sort_values("mean", ascending=False)
    style_stats.to_csv(os.path.join(OUT, "03_q1_改进_逐风格误差.csv"), encoding="utf-8-sig")

    # ---------------- 特征重要性 / PDP / 外部 MDI 对照 ----------------
    perm = permutation_importance(final_rf, X_full, Yv, scoring="r2", n_repeats=2,
                                  random_state=RANDOM_SEED, n_jobs=8)
    impm = perm.importances_mean
    impsd = perm.importances_std
    if getattr(impm, "ndim", 1) > 1:  # 多输出评分器会返回 (n_features, n_targets)
        impm = impm.mean(axis=1)
        impsd = impsd.mean(axis=1)
    fi = pd.DataFrame({"feature": feat_names_imp, "importance_mean": impm,
                       "importance_sd": impsd}).sort_values("importance_mean", ascending=False)
    fi.to_csv(os.path.join(OUT, "03_q1_改进_特征重要性.csv"), index=False, encoding="utf-8-sig")

    # 8.xlsx 外部 MDI 排名（RateBeer 喜好模型；仅作外部对照）
    paper_mdi_rank_corr = np.nan
    try:
        mdi = pd.read_excel(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "资料", "附件", "8.xlsx"),
                            sheet_name="GBM importance MDI")
        # 表结构：第一列为重要性，第二列为化学名（可能是 field_name_python/短名）
        mdi.columns = ["importance", "name"]
        thr = pd.read_excel(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "资料", "附件", "8.xlsx"),
                            sheet_name="Sensory thresholds")
        import re
        def norm(s):
            return re.sub(r"[^a-z0-9]+", "", str(s).lower())
        name_map = {}
        for _, r in thr.iterrows():
            for k in [r.get("field_name_python"), r.get("field_name_R"), r.get("chemical_name_short"), r.get("chemical_name")]:
                if isinstance(k, str):
                    name_map.setdefault(norm(k), None)
        # 反向：把 8.xlsx 名称映射到 1.csv 列名
        rev = {}
        for _, r in thr.iterrows():
            for k in [r.get("field_name_python"), r.get("field_name_R")]:
                if isinstance(k, str) and isinstance(r.get("field_name_R"), str):
                    rev[norm(k)] = r.get("field_name_R")
        mdi["feature"] = mdi["name"].map(lambda x: rev.get(norm(x)))
        merged = fi.merge(mdi.dropna(subset=["feature"]), on="feature", how="inner")
        if len(merged) >= 10:
            paper_mdi_rank_corr = float(spearmanr(merged["importance_mean"], merged["importance"]).correlation)
    except Exception as e:
        print("MDI compare skipped:", e, flush=True)

    # PDP：overall 与 A_esters_all 两个目标 × 前 4 特征
    try:
        top4 = list(fi["feature"].head(4))
        for tname, fname in [("overall", "fig_q1_pdp_overall.png"), ("A_esters_all", "fig_q1_pdp_esters.png")]:
            ti = targets.index(tname)
            disp = PartialDependenceDisplay.from_estimator(final_rf, X_full, features=top4,
                                                           feature_names=feat_names_imp, target=ti,
                                                           n_cols=2, n_jobs=1)
            fig = disp.figure_
            fig.suptitle(f"问题一 Partial Dependence：目标={tname}，特征=前 4 重要特征", fontsize=12)
            fig.tight_layout()
            fig.savefig(os.path.join(FIG, fname), dpi=160)
            plt.close(fig)
    except Exception as e:
        print("PDP skipped:", e, flush=True)

    # ---------------- 图 ----------------
    # 1) 重复 CV 指标分布
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    models_order = comp.sort_values("r2_mean", ascending=False)["model"].tolist()
    for ax, col, title in zip(axes, ["r2", "mae", "grade_pm1_minmax"],
                              ["逐折 R²", "逐折 MAE", "分级 ±1 准确率"]):
        data = [folds_df[folds_df["model"] == m][col].values for m in models_order]
        ax.boxplot(data, labels=models_order)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_q1_repeated_cv.png"), dpi=200)
    plt.close(fig)

    # 2) 缺失策略
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, model_name in enumerate(miss_df["model"].unique()):
        sub = miss_df[miss_df["model"] == model_name]
        ax.bar(np.arange(len(sub)) + i * 0.35, sub["r2_mean"], width=0.35,
               yerr=sub["r2_sd"], capsize=2, label=model_name)
    ax.set_xticks(np.arange(len(miss_df["strategy"].unique())) + 0.17)
    ax.set_xticklabels(miss_df["strategy"].unique(), rotation=20)
    ax.set_ylabel("逐折平均 R²")
    ax.set_title("缺失处理策略敏感性")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_q1_missing_strategies.png"), dpi=200)
    plt.close(fig)

    # 3) 分级混淆矩阵（minmax，全部验证折合并，最优模型）
    cm_true, cm_pred = [], []
    for seed in REPEATS:
        # 用 OOF 预测 + 全数据映射（近似）
        params = grade_mapping_fit(Yv, "minmax")
        cm_true.append(grade_mapping_apply(Yv, params).ravel())
        cm_pred.append(grade_mapping_apply(oof[best_model][seed].values, params).ravel())
    gt = np.concatenate(cm_true); gp = np.concatenate(cm_pred)
    cm = np.zeros((7, 7), dtype=int)
    for a, b in zip(gt, gp):
        cm[int(a), int(b)] += 1
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(7):
        for j in range(7):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    ax.set_xlabel("预测等级"); ax.set_ylabel("真实等级（映射后）")
    ax.set_title(f"问题一 0–6 分级混淆矩阵（{best_model}，min–max 口径）")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_q1_grade_confusion.png"), dpi=200)
    plt.close(fig)

    # 4) 逐风格误差
    order = style_stats.index.tolist()
    fig, ax = plt.subplots(figsize=(14, 5))
    data = [style_err[style_err["style"] == s]["mae_sample"].values for s in order]
    ax.boxplot(data, labels=order, showfliers=False)
    ax.set_xticklabels(order, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("样品平均绝对误差")
    ax.set_title(f"按啤酒风格的问题一误差（{best_model}，OOF）")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_q1_per_style_error.png"), dpi=200)
    plt.close(fig)

    # 5) 逐目标 R²
    fig, ax = plt.subplots(figsize=(14, 5))
    pt = per_target.sort_values("r2")
    ax.bar(range(len(pt)), pt["r2"], color="#55A868")
    ax.set_xticks(range(len(pt)))
    ax.set_xticklabels(pt["target"], rotation=90, fontsize=7)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("R²")
    ax.set_title(f"逐目标预测 R²（{best_model}，OOF，按 R² 排序）")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_q1_target_r2.png"), dpi=200)
    plt.close(fig)

    # 6) 特征重要性 Top20
    top = fi.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_sd"], color="#C44E52", capsize=2)
    ax.set_xlabel("置换重要性（R² 下降）")
    ax.set_title("问题一 特征重要性 Top20（置换重要性，RF）")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_q1_importance_top20.png"), dpi=200)
    plt.close(fig)

    # 7) 不确定性 vs 误差
    fig, ax = plt.subplots(figsize=(6, 4.8))
    ax.scatter(sample_unc, sample_mae, s=12, alpha=0.5, color="#4C72B0")
    ax.set_xlabel("样品平均预测不确定性（RF 树间标准差）")
    ax.set_ylabel("样品平均绝对误差（OOF）")
    ax.set_title(f"预测不确定性与误差（Spearman={unc_corr:.3f}）")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_q1_uncertainty.png"), dpi=200)
    plt.close(fig)

    # 保存最终模型供问题三做感官代理
    joblib.dump({
        "model_name": "RandomForest(final, 2B)",
        "imputer": imp_full,
        "model": final_rf,
        "feature_columns": feat_names_imp,
        "target_columns": targets,
        "grade_mapping": grade_mapping_fit(Yv, "minmax"),
        "grade_mapping_quantile": grade_mapping_fit(Yv, "quantile"),
        "random_seed": RANDOM_SEED,
    }, os.path.join(OUT, "03_q1_模型.joblib"), compress=3)

    out = {
        "config": {"repeats": REPEATS, "n_splits": N_SPLITS,
                   "imputation_primary": "median + missing indicator (fold-internal)",
                   "grade_methods": GRADE_METHODS,
                   "selection_rule": "逐折平均 R² 最高", "best_model": best_model},
        "model_comparison": comp.to_dict(orient="records"),
        "missing_strategies": miss_df.to_dict(orient="records"),
        "per_target_summary": {
            "best_targets": per_target.head(8)[["target", "r2", "spearman"]].to_dict(orient="records"),
            "worst_targets": per_target.tail(8)[["target", "r2", "spearman"]].to_dict(orient="records"),
        },
        "per_style_summary": style_stats.head(6).reset_index().to_dict(orient="records"),
        "uncertainty": {"spearman_unc_vs_error": unc_corr,
                        "mean_sample_unc": float(sample_unc.mean()),
                        "mean_sample_mae": float(sample_mae.mean())},
        "importance": {"top20": fi.head(20)[["feature", "importance_mean", "importance_sd"]].to_dict(orient="records"),
                       "paper_mdi_rank_corr": paper_mdi_rank_corr,
                       "note": "8.xlsx 的 MDI 来自参考论文 RateBeer 喜好模型，与本题 50 维品评目标不同源，仅作外部对照"},
        "runtime_seconds": round(time.time() - t0, 1),
        "env": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__},
    }
    save_json(out, os.path.join(OUT, "03_q1_改进_结果.json"))
    print(json.dumps({"best_model": best_model,
                      "top": comp.sort_values("r2_mean", ascending=False)[["model", "r2_mean", "mae_mean"]].head(4).to_dict(orient="records"),
                      "missing": miss_df.to_dict(orient="records"),
                      "unc": out["uncertainty"],
                      "runtime": out["runtime_seconds"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
