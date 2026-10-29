# -*- coding: utf-8 -*-
"""03b_q1_postprocess.py — 2B 结果的后处理与补算（当主脚本的重复 CV 已缓存时使用）。

背景：`03_q1_improved.py` 的重复 CV 与缺失策略计算量较大（约 12 分钟）。本脚本复用其已落盘缓存：
    成果/results/03_q1_改进_逐折指标.csv、03_q1_改进_缺失策略.csv、03_q1_改进_OOF预测.csv、
    03_q1_改进_逐目标指标.csv
并补算：
    1) Ridge 的稳定化重算（标准化后裁剪 ±5；修复极端 z 值导致的外推爆炸）；
    2) 模型对比表、逐折图、缺失策略图、混淆矩阵、逐风格图、逐目标图；
    3) RF 最终模型的置换重要性、PDP、不确定性与外部 MDI 对照；
    4) 03_q1_改进_结果.json 与 03_q1_模型.joblib。

运行（需先跑过 03_q1_improved.py 的主 CV 部分，缓存文件存在）：
    $env:PYTHONPATH = "D:\\github mama\\.work\\pylibs"
    D:\\miniconda\\python.exe 成果\\code\\03b_q1_postprocess.py
"""
import os
import sys
import json
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import joblib

from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, cohen_kappa_score
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_prep_common import (load_data, grade_mapping_fit, grade_mapping_apply,
                              make_folds, save_json, RANDOM_SEED, OUT, FIG)

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
KEYS = ["r2", "mae", "spearman", "grade_exact_minmax", "grade_pm1_minmax", "kappa_minmax",
        "grade_exact_quantile", "grade_pm1_quantile", "kappa_quantile"]

t0 = time.time()
X, Y, styles, ids = load_data()
Xv, Yv = X.values.astype(float), Y.values.astype(float)
targets = list(Y.columns)
n_targets = len(targets)

folds_df = pd.read_csv(os.path.join(OUT, "03_q1_改进_逐折指标.csv"))
miss_df = pd.read_csv(os.path.join(OUT, "03_q1_改进_缺失策略.csv"))
per_target = pd.read_csv(os.path.join(OUT, "03_q1_改进_逐目标指标.csv"))
oof_all = pd.read_csv(os.path.join(OUT, "03_q1_改进_OOF预测.csv"))

# ---------------- 1) Ridge 稳定化重算 ----------------
ridge_rows = []
for seed in REPEATS:
    for fi, (tr, va) in enumerate(make_folds(styles, N_SPLITS, seed)):
        X_tr, X_va = Xv[tr], Xv[va]
        Y_tr, Y_va = Yv[tr], Yv[va]
        imp = SimpleImputer(strategy="median", add_indicator=True).fit(X_tr)
        sc = StandardScaler().fit(imp.transform(X_tr))
        Xtr = np.clip(sc.transform(imp.transform(X_tr)), -5, 5)
        Xva = np.clip(sc.transform(imp.transform(X_va)), -5, 5)
        model = Ridge(alpha=10.0).fit(Xtr, Y_tr)
        pred = model.predict(Xva)
        r2s = np.array([r2_score(Y_va[:, j], pred[:, j]) for j in range(n_targets)])
        maes = np.array([mean_absolute_error(Y_va[:, j], pred[:, j]) for j in range(n_targets)])
        rhos = np.array([spearmanr(Y_va[:, j], pred[:, j]).correlation for j in range(n_targets)])
        row = {"model": "Ridge", "seed": seed, "fold": fi,
               "r2": float(np.nanmean(r2s)), "mae": float(np.nanmean(maes)),
               "spearman": float(np.nanmean(rhos))}
        for gm in GRADE_METHODS:
            params = grade_mapping_fit(Y_tr, gm)
            gt = grade_mapping_apply(Y_va, params)
            gp = grade_mapping_apply(pred, params)
            acc = float(np.mean([np.mean(gt[:, j] == gp[:, j]) for j in range(n_targets)]))
            pm1 = float(np.mean([np.mean(np.abs(gt[:, j] - gp[:, j]) <= 1) for j in range(n_targets)]))
            kap = float(np.mean([cohen_kappa_score(gt[:, j], gp[:, j], weights="quadratic",
                                                   labels=list(range(7))) for j in range(n_targets)]))
            if gm == "minmax":
                row.update({"grade_exact_minmax": acc, "grade_pm1_minmax": pm1, "kappa_minmax": kap})
            else:
                row.update({"grade_exact_quantile": acc, "grade_pm1_quantile": pm1, "kappa_quantile": kap})
        row.update({"n_train": int(len(tr)), "n_valid": int(len(va))})
        ridge_rows.append(row)
folds_df = folds_df[folds_df["model"] != "Ridge"]
folds_df = pd.concat([folds_df, pd.DataFrame(ridge_rows)], ignore_index=True)
folds_df.to_csv(os.path.join(OUT, "03_q1_改进_逐折指标.csv"), index=False, encoding="utf-8-sig")
comp = folds_df.groupby("model")[KEYS].agg(["mean", "std"]).round(4)
comp.columns = [f"{a}_{b}" for a, b in comp.columns]
comp = comp.reset_index()
comp.to_csv(os.path.join(OUT, "03_q1_改进_模型对比.csv"), index=False, encoding="utf-8-sig")
best_model = comp.sort_values("r2_mean", ascending=False).iloc[0]["model"]
print("best model:", best_model, flush=True)

# ---------------- 2) OOF / 逐目标（缓存） ----------------
best_oof = oof_all[targets].copy()
best_oof.index = range(len(best_oof))
y_true = Yv
sample_mae = np.abs(best_oof.values - y_true).mean(axis=1)

# ---------------- 3) 最终 RF：不确定性 / 重要性 / PDP ----------------
imp_full = SimpleImputer(strategy="median", add_indicator=True).fit(Xv)
X_full = imp_full.transform(Xv)
final_rf = RandomForestRegressor(n_estimators=400, max_features="sqrt",
                                 min_samples_leaf=3, n_jobs=8, random_state=RANDOM_SEED)
final_rf.fit(X_full, Yv)
missing_cols = [c for c in X.columns if X[c].isna().any()]
feat_names_imp = list(X.columns) + [f"missing_{c}" for c in missing_cols]
assert len(feat_names_imp) == X_full.shape[1], (len(feat_names_imp), X_full.shape[1])
tree_preds = np.stack([t.predict(X_full) for t in final_rf.estimators_], axis=0)
unc = tree_preds.std(axis=0, ddof=1)
sample_unc = unc.mean(axis=1)
unc_corr = float(spearmanr(sample_unc, sample_mae).correlation)

perm = permutation_importance(final_rf, X_full, Yv, scoring="r2", n_repeats=2,
                              random_state=RANDOM_SEED, n_jobs=8)
impm, impsd = perm.importances_mean, perm.importances_std
if impm.ndim > 1:
    impm, impsd = impm.mean(axis=1), impsd.mean(axis=1)
fi = pd.DataFrame({"feature": feat_names_imp, "importance_mean": impm,
                   "importance_sd": impsd}).sort_values("importance_mean", ascending=False)
fi.to_csv(os.path.join(OUT, "03_q1_改进_特征重要性.csv"), index=False, encoding="utf-8-sig")

# 外部 MDI 对照（8.xlsx，参考论文 RateBeer 喜好模型）
paper_mdi_rank_corr = np.nan
try:
    import re
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    mdi = pd.read_excel(os.path.join(root, "资料", "附件", "8.xlsx"), sheet_name="GBM importance MDI")
    mdi.columns = ["importance", "name"]
    thr = pd.read_excel(os.path.join(root, "资料", "附件", "8.xlsx"), sheet_name="Sensory thresholds")
    def norm(s):
        return re.sub(r"[^a-z0-9]+", "", str(s).lower())
    rev = {}
    for _, r in thr.iterrows():
        rname = r.get("field_name_R")
        if isinstance(rname, str):
            for k in [r.get("field_name_python"), r.get("field_name_R"),
                      r.get("chemical_name_short"), r.get("chemical_name")]:
                if isinstance(k, str):
                    rev[norm(k)] = rname
    mdi["feature"] = mdi["name"].map(lambda x: rev.get(norm(x)))
    merged = fi.merge(mdi.dropna(subset=["feature"]), on="feature", how="inner")
    if len(merged) >= 10:
        paper_mdi_rank_corr = float(spearmanr(merged["importance_mean"], merged["importance"]).correlation)
except Exception as e:
    print("MDI compare skipped:", e, flush=True)

try:
    top4 = list(fi["feature"].head(4))
    for tname, fname in [("overall", "fig_q1_pdp_overall.png"), ("A_esters_all", "fig_q1_pdp_esters.png")]:
        ti = targets.index(tname)
        disp = PartialDependenceDisplay.from_estimator(final_rf, X_full, features=top4,
                                                       feature_names=feat_names_imp,
                                                       target=ti, n_cols=2, n_jobs=1)
        fig = disp.figure_
        fig.suptitle(f"问题一 Partial Dependence：目标={tname}，特征=前 4 重要特征", fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, fname), dpi=160)
        plt.close(fig)
except Exception as e:
    print("PDP skipped:", e, flush=True)

# ---------------- 4) 图 ----------------
# 重复 CV
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
order = comp.sort_values("r2_mean", ascending=False)["model"].tolist()
for ax, col, title in zip(axes, ["r2", "mae", "grade_pm1_minmax"],
                          ["逐折 R²", "逐折 MAE", "分级 ±1 准确率"]):
    data = [folds_df[folds_df["model"] == m][col].values for m in order]
    ax.boxplot(data, labels=order)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_q1_repeated_cv.png"), dpi=200)
plt.close(fig)

# 缺失策略
fig, ax = plt.subplots(figsize=(9, 4.8))
models = miss_df["model"].unique()
strats = miss_df["strategy"].unique()
for i, model_name in enumerate(models):
    sub = miss_df[miss_df["model"] == model_name]
    ax.bar(np.arange(len(sub)) + i * 0.35, sub["r2_mean"], width=0.35,
           yerr=sub["r2_sd"], capsize=2, label=model_name)
ax.set_xticks(np.arange(len(strats)) + 0.17)
ax.set_xticklabels(strats, rotation=20)
ax.set_ylabel("逐折平均 R²")
ax.set_title("缺失处理策略敏感性")
ax.legend()
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_q1_missing_strategies.png"), dpi=200)
plt.close(fig)

# 分级混淆矩阵
params = grade_mapping_fit(Yv, "minmax")
gt = grade_mapping_apply(Yv, params).ravel()
gp = grade_mapping_apply(best_oof.values, params).ravel()
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

# 逐风格误差
err_df = pd.DataFrame({"style": styles.values, "mae_sample": sample_mae, "unc": sample_unc})
style_stats = err_df.groupby("style")["mae_sample"].agg(["mean", "median", "count"]).sort_values("mean", ascending=False)
style_stats.to_csv(os.path.join(OUT, "03_q1_改进_逐风格误差.csv"), encoding="utf-8-sig")
order_s = style_stats.index.tolist()
fig, ax = plt.subplots(figsize=(14, 5))
data = [err_df[err_df["style"] == s]["mae_sample"].values for s in order_s]
ax.boxplot(data, labels=order_s, showfliers=False)
ax.set_xticklabels(order_s, rotation=60, ha="right", fontsize=8)
ax.set_ylabel("样品平均绝对误差")
ax.set_title(f"按啤酒风格的问题一误差（{best_model}，OOF）")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_q1_per_style_error.png"), dpi=200)
plt.close(fig)

# 逐目标 R²
pt = per_target.sort_values("r2")
fig, ax = plt.subplots(figsize=(14, 5))
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

# 特征重要性
top = fi.head(20).iloc[::-1]
fig, ax = plt.subplots(figsize=(8, 6.5))
ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_sd"], color="#C44E52", capsize=2)
ax.set_xlabel("置换重要性（R² 下降）")
ax.set_title("问题一 特征重要性 Top20（置换重要性，RF）")
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_q1_importance_top20.png"), dpi=200)
plt.close(fig)

# 不确定性
fig, ax = plt.subplots(figsize=(6, 4.8))
ax.scatter(sample_unc, sample_mae, s=12, alpha=0.5, color="#4C72B0")
ax.set_xlabel("样品平均预测不确定性（RF 树间标准差）")
ax.set_ylabel("样品平均绝对误差（OOF）")
ax.set_title(f"预测不确定性与误差（Spearman={unc_corr:.3f}）")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_q1_uncertainty.png"), dpi=200)
plt.close(fig)

# ---------------- 5) 保存模型与 JSON ----------------
joblib.dump({
    "model_name": "RandomForest(final, 2B)",
    "imputer": imp_full, "model": final_rf,
    "feature_columns": feat_names_imp, "target_columns": targets,
    "grade_mapping": grade_mapping_fit(Yv, "minmax"),
    "grade_mapping_quantile": grade_mapping_fit(Yv, "quantile"),
    "random_seed": RANDOM_SEED,
}, os.path.join(OUT, "03_q1_模型.joblib"), compress=3)

out = {
    "config": {"repeats": REPEATS, "n_splits": N_SPLITS,
               "imputation_primary": "fold-internal median + missing indicator",
               "grade_methods": GRADE_METHODS, "best_model": best_model,
               "ridge_note": "Ridge 在标准化后裁剪 ±5（少量近常量特征会造成极端 z 值与外推爆炸）",
               "postprocess_note": "本文件由 03b_q1_postprocess.py 在缓存的主 CV 结果上补算 Ridge/解释/图表"},
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
                  "unc": out["uncertainty"],
                  "mdi_rank_corr": paper_mdi_rank_corr,
                  "runtime": out["runtime_seconds"]}, ensure_ascii=False, indent=2))
