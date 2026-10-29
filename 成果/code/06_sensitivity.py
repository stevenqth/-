# -*- coding: utf-8 -*-
"""06_sensitivity.py — 模型分析检验：问题一/二/三的灵敏度与稳健性（T015，提示词第 7–8 部分）。

内容：
    A. 问题三（cVAE 生成/迁移）：
       - 潜维 {6, 8, 12}
       - KL 权重 β {0.001, 0.005, 0.02}
       - 后验采样噪声 {0.2, 0.3, 0.5}
       - 迁移候选数 K {4, 8, 12}
       - 阈值投影余量 {0.90, 0.95, 0.99}
       指标：重构 R²、生成风格分类准确率、Q1 代理画像 MAE、缺陷投影前后超阈值比例、
             迁移缺陷恶化比例/代理画像 MAE。
    B. 问题一：特征缺失比例 {0, 5%, 10%} × RF/ET 树数 {200, 400} 的 5 折稳健性。
    C. 问题二：证据消融（去官方证据 / 去 Ridge / RF 重要性加权）对 Top-3 的影响（平均 Jaccard）。

运行：
    $env:PYTHONPATH = "D:\\github mama\\.work\\pylibs"
    D:\\miniconda\\python.exe 成果\\code\\06_sensitivity.py
输出：
    成果/results/06_q3_灵敏度.csv、06_q1_灵敏度.csv、06_q2_灵敏度.csv、06_灵敏度汇总.json
    成果/figures/fig_06_*.png
"""
import os
import sys
import json
import time
import warnings
import importlib.util

import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score, StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_prep_common import load_data, classify_transforms, make_folds, grade_mapping_fit, grade_mapping_apply, save_json, RANDOM_SEED

warnings.filterwarnings("ignore")
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.set_num_threads(4)

ROOT = r"D:\github mama"
MAT = os.path.join(ROOT, "资料", "附件")
OUT = os.path.join(ROOT, "成果", "results")
FIG = os.path.join(ROOT, "成果", "figures")
CODEDIR = os.path.join(ROOT, "成果", "code")
for name in ["Microsoft YaHei", "SimHei", "SimSun"]:
    try:
        font_manager.findfont(name, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [name]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(CODEDIR, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


q3 = load_module("q3mod", "05_q3_generation.py")
q2 = load_module("q2mod", "04_q2_top3.py")


# ================================================================ 问题三
def fit_vae(Z, onehot, latent=8, beta=0.005, epochs=3000):
    model, info = q3.train_cvae(Z, onehot, beta=beta, epochs=epochs, latent=latent)
    with torch.no_grad():
        Zt = torch.tensor(Z, dtype=torch.float32)
        Ct = torch.tensor(onehot, dtype=torch.float32)
        mu_all, _ = model.encode(Zt, Ct)
    return model, info, mu_all.numpy()


def generate_eval(model, Z, onehot, style_mu, style_names, sidx, clf, q1_predict, space, cols,
                  noise=0.3, margin=0.95, n_per_style=30, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    latent = style_mu.shape[1]
    limits = {k: d["threshold"] for k, d in q3.DEFECTS.items() if d["threshold"] is not None}
    gen_Z, gen_style, pre, post = [], [], {k: [] for k in limits}, {k: [] for k in limits}
    for k, s in enumerate(style_names):
        zs = (style_mu[k][None, :] + rng.normal(0, noise, size=(n_per_style, latent))).astype(np.float32)
        with torch.no_grad():
            cond = torch.tensor(np.tile(onehot[sidx[s]], (n_per_style, 1)), dtype=torch.float32)
            rec = model.decode(torch.tensor(zs), cond).numpy()
        M_raw = rec * space["sd"] + space["mu"]
        conc_raw = q3.defect_concentrations(M_raw, space)
        cap_limits = {k: limits[k] * margin for k in limits}
        M_proj = q3.project_defects(M_raw, space, {k: limits[k] * margin for k in limits})
        conc_proj = q3.defect_concentrations(M_proj, space)
        for key in limits:
            pre[key].append(float(np.mean(conc_raw[key] > limits[key])))
            post[key].append(float(np.mean(conc_proj[key] > limits[key])))
        gen_Z.append(rec)
        gen_style.extend([sidx[s]] * n_per_style)
        Xrec = q3.m_to_original(M_proj, space)
        if k == 0:
            sens_all = q1_predict(Xrec)
        else:
            sens_all = pd.concat([sens_all, q1_predict(Xrec)], ignore_index=True)
    gen_Z = np.vstack(gen_Z)
    gen_style = np.array(gen_style)
    acc = float(np.mean(clf.predict(gen_Z) == gen_style))
    q1_targets = q1_predict.__globals__.get("q1_targets") if False else None
    return {"style_acc": acc, "pre_above": {k: float(np.mean(v)) for k, v in pre.items()},
            "post_above": {k: float(np.mean(v)) for k, v in post.items()}, "gen_Z": gen_Z,
            "gen_style": gen_style, "gen_sens": sens_all}


def transfer_eval(model, Z, onehot, style_mu, X, M, styles, style_names, sidx, clf, q1_predict, space, cols,
                  pairs, K=8, margin=0.95, noise=0.3, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    out = {}
    for src, tgt in pairs:
        if src not in style_names or tgt not in style_names:
            continue
        src_mask = styles.values == src
        Xsrc, Msrc = X.values[src_mask], M[src_mask]
        src_sens = q1_predict(Xsrc)
        src_conc = q3.defect_concentrations(Msrc, space)
        z_base = torch.tensor(np.repeat(style_mu[sidx[src]][None, :], len(Xsrc) * K, axis=0), dtype=torch.float32)
        cond = torch.tensor(np.tile(onehot[sidx[tgt]], (len(Xsrc) * K, 1)), dtype=torch.float32)
        with torch.no_grad():
            rec = model.decode(z_base + torch.randn_like(z_base) * noise, cond).numpy()
        Mrec = rec * space["sd"] + space["mu"]
        for key, d in q3.DEFECTS.items():
            i = cols.index(d["col"])
            base = src_conc[key].repeat(K)
            limit = base if d["threshold"] is None else np.minimum(base, d["threshold"] * margin)
            Mrec[:, i] = np.minimum(Mrec[:, i], np.log10(np.clip(limit, 1e-12, None)))
        Xrec = q3.m_to_original(Mrec, space)
        pred = q1_predict(Xrec)
        proba = clf.predict_proba(rec)
        stale = pred["stale_hops"].values
        src_stale = np.repeat(src_sens["stale_hops"].values, K)
        score = proba[:, sidx[tgt]] - 2.0 * np.maximum(0, stale - src_stale - 0.3)
        best = np.array([i * K + int(np.argmax(score[i * K:(i + 1) * K])) for i in range(len(Xsrc))])
        Xbest = Xrec[best]
        Mbest = Mrec[best]
        pred_best = q1_predict(Xbest)
        conc_best = q3.defect_concentrations(Mbest, space)
        # 目标风格真实均值（用全部真实目标样品）
        tgt_real = q1_predict(X.values[styles.values == tgt])
        out[f"{src}->{tgt}"] = {
            "style_confidence": float(proba[best, sidx[tgt]].mean()),
            "sensory_mae": float(np.mean(np.abs(pred_best.values.mean(axis=0) - tgt_real.values.mean(axis=0)))),
            "defect_worsened_share": {k: float(np.mean(conc_best[k] > src_conc[k] * 1.0001)) for k in q3.DEFECTS},
            "above_threshold_share": {k: (None if q3.DEFECTS[k]["threshold"] is None else
                                          float(np.mean(conc_best[k] > q3.DEFECTS[k]["threshold"]))) for k in q3.DEFECTS},
        }
    return out


def main():
    t0 = time.time()
    X, Y, styles, ids = load_data()
    cols = list(X.columns)
    transforms = classify_transforms()
    M, Z, space = q3.build_space(X, transforms)
    style_names = sorted(styles.unique())
    sidx = {s: i for i, s in enumerate(style_names)}
    y = np.array([sidx[s] for s in styles])
    onehot = np.eye(len(style_names))[y]

    q1 = joblib.load(os.path.join(OUT, "03_q1_模型.joblib"))
    q1_imp, q1_model, q1_targets = q1["imputer"], q1["model"], q1["target_columns"]

    def q1_predict(X_orig):
        Xdf = pd.DataFrame(X_orig, columns=cols)
        return pd.DataFrame(q1_model.predict(q1_imp.transform(Xdf)), columns=q1_targets)
    # 供 generate_eval 内取到列名：挂到函数属性
    q1_predict.targets = q1_targets

    clf = RandomForestClassifier(n_estimators=300, max_features="sqrt", n_jobs=8, random_state=RANDOM_SEED)
    clf.fit(Z, y)
    cv_acc = float(np.mean(cross_val_score(clf, Z, y, cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_SEED))))

    # 基准 cVAE
    t = time.time()
    base_model, base_info, base_mu = fit_vae(Z, onehot, latent=8, beta=0.005)
    style_mu_base = np.array([base_mu[y == k].mean(axis=0) for k in range(len(style_names))])
    pairs = [("Pils/Lager", "Tripel"), ("Stout/Porter", "Fruitbeer")]
    rows = []
    base_gen = generate_eval(base_model, Z, onehot, style_mu_base, style_names, sidx, clf, q1_predict, space, cols)
    base_tr = transfer_eval(base_model, Z, onehot, style_mu_base, X, M, styles, style_names, sidx,
                            clf, q1_predict, space, cols, pairs)
    rows.append({"variant": "baseline", "latent": 8, "beta": 0.005, "noise": 0.3, "K": 8, "margin": 0.95,
                 "recon_r2": base_info["mean_feature_r2"], "style_acc": base_gen["style_acc"],
                 "pre_above_DMS": base_gen["pre_above"]["DMS"], "post_above_DMS": base_gen["post_above"]["DMS"],
                 "pre_above_diacetyl": base_gen["pre_above"]["diacetyl"], "post_above_diacetyl": base_gen["post_above"]["diacetyl"],
                 **{f"transfer_{k}_{m}": v for k, tr in base_tr.items() for m, v in
                    [("sensory_mae", tr["sensory_mae"]), ("worsened_DMS", tr["defect_worsened_share"]["DMS"])]},
                 "train_s": round(time.time() - t, 1)})

    # 潜维 / β
    for latent in [6, 12]:
        model, info, mu = fit_vae(Z, onehot, latent=latent, beta=0.005)
        sm = np.array([mu[y == k].mean(axis=0) for k in range(len(style_names))])
        gen = generate_eval(model, Z, onehot, sm, style_names, sidx, clf, q1_predict, space, cols)
        tr = transfer_eval(model, Z, onehot, sm, X, M, styles, style_names, sidx, clf, q1_predict, space, cols, pairs)
        rows.append({"variant": f"latent={latent}", "latent": latent, "beta": 0.005, "noise": 0.3, "K": 8, "margin": 0.95,
                     "recon_r2": info["mean_feature_r2"], "style_acc": gen["style_acc"],
                     "pre_above_DMS": gen["pre_above"]["DMS"], "post_above_DMS": gen["post_above"]["DMS"],
                     "pre_above_diacetyl": gen["pre_above"]["diacetyl"], "post_above_diacetyl": gen["post_above"]["diacetyl"],
                     **{f"transfer_{k}_{m}": v for k, trr in tr.items() for m, v in
                        [("sensory_mae", trr["sensory_mae"]), ("worsened_DMS", trr["defect_worsened_share"]["DMS"])]}})
    for beta in [0.001, 0.02]:
        model, info, mu = fit_vae(Z, onehot, latent=8, beta=beta)
        sm = np.array([mu[y == k].mean(axis=0) for k in range(len(style_names))])
        gen = generate_eval(model, Z, onehot, sm, style_names, sidx, clf, q1_predict, space, cols)
        tr = transfer_eval(model, Z, onehot, sm, X, M, styles, style_names, sidx, clf, q1_predict, space, cols, pairs)
        rows.append({"variant": f"beta={beta}", "latent": 8, "beta": beta, "noise": 0.3, "K": 8, "margin": 0.95,
                     "recon_r2": info["mean_feature_r2"], "style_acc": gen["style_acc"],
                     "pre_above_DMS": gen["pre_above"]["DMS"], "post_above_DMS": gen["post_above"]["DMS"],
                     "pre_above_diacetyl": gen["pre_above"]["diacetyl"], "post_above_diacetyl": gen["post_above"]["diacetyl"],
                     **{f"transfer_{k}_{m}": v for k, trr in tr.items() for m, v in
                        [("sensory_mae", trr["sensory_mae"]), ("worsened_DMS", trr["defect_worsened_share"]["DMS"])]}})
    # 采样噪声（固定基准模型）
    for noise in [0.2, 0.5]:
        gen = generate_eval(base_model, Z, onehot, style_mu_base, style_names, sidx, clf, q1_predict,
                            space, cols, noise=noise)
        rows.append({"variant": f"noise={noise}", "latent": 8, "beta": 0.005, "noise": noise, "K": 8, "margin": 0.95,
                     "recon_r2": base_info["mean_feature_r2"], "style_acc": gen["style_acc"],
                     "pre_above_DMS": gen["pre_above"]["DMS"], "post_above_DMS": gen["post_above"]["DMS"],
                     "pre_above_diacetyl": gen["pre_above"]["diacetyl"], "post_above_diacetyl": gen["post_above"]["diacetyl"]})
    # K 与余量（固定基准模型）
    for K in [4, 12]:
        tr = transfer_eval(base_model, Z, onehot, style_mu_base, X, M, styles, style_names, sidx,
                           clf, q1_predict, space, cols, pairs, K=K)
        rows.append({"variant": f"K={K}", "latent": 8, "beta": 0.005, "noise": 0.3, "K": K, "margin": 0.95,
                     **{f"transfer_{k}_{m}": v for k, trr in tr.items() for m, v in
                        [("sensory_mae", trr["sensory_mae"]), ("worsened_DMS", trr["defect_worsened_share"]["DMS"]),
                         ("style_conf", trr["style_confidence"])]}})
    for margin in [0.90, 0.99]:
        tr = transfer_eval(base_model, Z, onehot, style_mu_base, X, M, styles, style_names, sidx,
                           clf, q1_predict, space, cols, pairs, margin=margin)
        rows.append({"variant": f"margin={margin}", "latent": 8, "beta": 0.005, "noise": 0.3, "K": 8, "margin": margin,
                     **{f"transfer_{k}_{m}": v for k, trr in tr.items() for m, v in
                        [("sensory_mae", trr["sensory_mae"]), ("worsened_DMS", trr["defect_worsened_share"]["DMS"]),
                         ("above_DMS", trr["above_threshold_share"]["DMS"])]}})
    q3_df = pd.DataFrame(rows)
    q3_df.to_csv(os.path.join(OUT, "06_q3_灵敏度.csv"), index=False, encoding="utf-8-sig")

    # ================================================================ 问题一
    q1_rows = []
    style_arr = styles.values
    for miss in [0.0, 0.05, 0.10]:
        for trees in [200, 400]:
            r2s, maes = [], []
            for tr, va in make_folds(styles, 5, RANDOM_SEED):
                Xtr, Xva = X.values[tr].copy(), X.values[va].copy()
                rng = np.random.default_rng(42)
                if miss > 0:
                    mask_tr = rng.random(Xtr.shape) < miss
                    mask_va = rng.random(Xva.shape) < miss
                    Xtr[mask_tr] = np.nan
                    Xva[mask_va] = np.nan
                imp = SimpleImputer(strategy="median", add_indicator=True).fit(Xtr)
                if trees <= 250:
                    m = ExtraTreesRegressor(n_estimators=trees, max_features="sqrt",
                                            min_samples_leaf=1, n_jobs=8, random_state=RANDOM_SEED)
                else:
                    m = ExtraTreesRegressor(n_estimators=trees, max_features="sqrt",
                                            min_samples_leaf=1, n_jobs=8, random_state=RANDOM_SEED)
                m.fit(imp.transform(Xtr), Y.values[tr])
                p = m.predict(imp.transform(Xva))
                r2s.append(float(np.nanmean([r2_score(Y.values[va][:, j], p[:, j]) for j in range(Y.shape[1])])))
                maes.append(float(np.nanmean([mean_absolute_error(Y.values[va][:, j], p[:, j]) for j in range(Y.shape[1])])))
            q1_rows.append({"missing_ratio": miss, "trees": trees,
                            "r2_mean": float(np.mean(r2s)), "r2_sd": float(np.std(r2s)),
                            "mae_mean": float(np.mean(maes))})
    q1_df = pd.DataFrame(q1_rows)
    q1_df.to_csv(os.path.join(OUT, "06_q1_灵敏度.csv"), index=False, encoding="utf-8-sig")

    # ================================================================ 问题二（证据消融）
    mp = q2.load_mapping()
    disp, source, threshold = q2.build_chem_display()
    df6 = pd.read_csv(os.path.join(MAT, "6.csv"), index_col=0)
    m6 = mp[mp["source"] == "6.csv"]
    chem_set, sens_set = set(cols), set(Y.columns)
    row_map, col_map = {}, {}
    for _, r in m6.iterrows():
        if r["target"] in chem_set:
            row_map[r["label"]] = r["target"]
        elif r["target"] in sens_set:
            col_map[r["label"]] = r["target"]
    official = df6.rename(index=row_map, columns=col_map)
    official = official.loc[[i for i in official.index if i in chem_set],
                            [c for c in official.columns if c in sens_set]]
    official = official.loc[~official.index.duplicated(keep="first"),
                            ~official.columns.duplicated(keep="first")]
    G, module, degree, communities = q2.build_graph(cols)
    imp = SimpleImputer(strategy="median").fit(X.values)
    Ximp = imp.transform(X.values)
    ev_base, _ = q2.compute_evidence(Ximp, Y, styles, cols, official, rf_trees=200)
    base_score = q2.composite_score(ev_base)
    base_top = q2.module_topk(base_score, cols, module)
    ablations = {
        "drop_official": {k: v for k, v in ev_base.items() if k != "official"},
        "drop_ridge": {k: v for k, v in ev_base.items() if k != "ridge"},
        "rf_double_weight": {**ev_base, "rf": ev_base["rf"]},
    }
    # rf double weight：手动加权（按证据数归一）
    def weighted_score(ev, weights):
        arr = np.stack([ev[k] for k in ev], axis=0)
        w = np.array([weights.get(k, 1.0) for k in ev], dtype=float)
        w = w / w.sum()
        with np.errstate(all="ignore"):
            return np.nansum(arr * w[:, None, None], axis=0) / np.sum(np.isfinite(arr) * w[:, None, None], axis=0)
    jac_rows = []
    for name, ev in ablations.items():
        if name == "rf_double_weight":
            sc = weighted_score(ev_base, {"partial": 1.0, "ridge": 1.0, "rf": 2.0, "official": 1.0})
        else:
            sc = q2.composite_score(ev)
        top = q2.module_topk(sc, cols, module)
        jac = [len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in zip(base_top, top)]
        jac_rows.append({"ablation": name, "mean_jaccard_vs_baseline": float(np.mean(jac))})
    q2_df = pd.DataFrame(jac_rows)
    q2_df.to_csv(os.path.join(OUT, "06_q2_灵敏度.csv"), index=False, encoding="utf-8-sig")

    summary = {
        "q3": {"baseline_cvae": {k: v for k, v in base_info.items() if k != "loss_curve"},
               "variants": q3_df.drop(columns=[c for c in ["gen_Z", "gen_style", "gen_sens"] if c in q3_df.columns])
               .to_dict(orient="records"),
               "transfer_pairs": [f"{a}->{b}" for a, b in pairs],
               "runtime_seconds": round(time.time() - t0, 1)},
        "q1": q1_df.to_dict(orient="records"),
        "q2": q2_df.to_dict(orient="records"),
        "env": {"python": sys.version.split()[0], "torch": torch.__version__,
                "numpy": np.__version__, "pandas": pd.__version__},
    }
    save_json(summary, os.path.join(OUT, "06_灵敏度汇总.json"))

    # ---------------- 图 ----------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].bar(q3_df["variant"], q3_df["style_acc"], color="#4C72B0")
    axes[0].set_title("问题三：不同生成配置的风格分类准确率")
    axes[0].tick_params(axis="x", rotation=60)
    axes[0].grid(axis="y", alpha=0.3)
    axes[1].bar(q3_df["variant"], q3_df["post_above_DMS"].fillna(0), color="#55A868")
    axes[1].set_title("问题三：投影后 DMS 超阈值比例（应≈0）")
    axes[1].tick_params(axis="x", rotation=60)
    axes[1].grid(axis="y", alpha=0.3)
    axes[2].bar(q1_df["missing_ratio"].astype(str) + "_" + q1_df["trees"].astype(str),
                q1_df["r2_mean"], yerr=q1_df["r2_sd"], capsize=3, color="#C44E52")
    axes[2].set_title("问题一：缺失比例 × 树数的 5 折 R²")
    axes[2].tick_params(axis="x", rotation=45)
    axes[2].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_06_sensitivity.png"), dpi=200)
    plt.close(fig)

    print(json.dumps({"q3_variants": q3_df[["variant", "recon_r2", "style_acc", "pre_above_DMS", "post_above_DMS"]]
                     .to_dict(orient="records"),
                      "q1": q1_df.to_dict(orient="records"),
                      "q2": q2_df.to_dict(orient="records"),
                      "runtime": summary["q3"]["runtime_seconds"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
