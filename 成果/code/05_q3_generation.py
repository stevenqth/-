# -*- coding: utf-8 -*-
"""05_q3_generation.py — 问题三：风格条件生成、跨风格迁移与缺陷约束（决策 1A/4B/5A）。

模型与流程：
    1) 统一“浓度/对数浓度空间”：按数据字典 4.1 的逐列变换分类，log10 列保留 v，
       原始浓度列取 log10(clip(v, floor))；再逐列 z-score → 生成空间 Z。
    2) 条件 VAE（cVAE）：风格 one-hot 作为条件，编码器/解码器均为小 MLP（250 样本防过拟合）；
       目标函数 = 重构 MSE + β·KL，全批量训练。
    3) 风格条件生成：按风格采样潜变量 → 解码 → 反变换回原始量纲；
    4) 跨风格迁移：源样品编码后以目标风格条件解码（best-of-K 候选 + 缺陷/代理评分筛选）；
       对照“潜空间风格方向平移”方法；
    5) 缺陷硬约束：DMS / Diacetyl / 4VG 按官方阈值投影；trans-2-nonenal 无官方阈值，
       按“不劣于源样品”约束；stale hops 无化学阈值，用问题一代理模型的感官预测做软约束；
    6) 评价：风格分类器准确率、MMD、相关结构保持度、Q1 感官代理画像误差、
       缺陷超阈值/恶化比例、7.csv 消费者描述符辅助目标差异（决策 4B）、
       in-silico 验证（决策 5A：迁移前后代理评分变化）。

运行：
    $env:PYTHONPATH = "D:\\github mama\\.work\\pylibs"
    D:\\miniconda\\python.exe 成果\\code\\05_q3_generation.py
输出：
    成果/results/05_q3_生成谱.csv、05_q3_迁移结果.csv、05_q3_评价.json、05_q3_代理画像.csv
    成果/figures/fig_q3_*.png
"""
import os
import sys
import json
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_prep_common import load_data, classify_transforms, save_json, RANDOM_SEED

warnings.filterwarnings("ignore")
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.set_num_threads(4)

ROOT = r"D:\github mama"
MAT = os.path.join(ROOT, "资料", "附件")
OUT = os.path.join(ROOT, "成果", "results")
FIG = os.path.join(ROOT, "成果", "figures")
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

# 缺陷化合物：列名、单位、官方阈值（8.xlsx / 论文阈值表）
DEFECTS = {
    "DMS": {"col": "DMS", "unit": "ug/L", "threshold": 17.25},
    "diacetyl": {"col": "X2.3.butanedione", "unit": "mg/L", "threshold": 0.29925},
    "4VG": {"col": "X4VG", "unit": "mg/L", "threshold": 0.206333333},
    "t2n": {"col": "X.E..2.nonenal", "unit": "mg/L", "threshold": None},  # 无官方阈值
}


# ---------------------------------------------------------------- 数据空间
def build_space(X, transforms):
    """返回 M（浓度/对数浓度空间）、Z（标准化生成空间）与逆变换参数。"""
    cols = list(X.columns)
    M = np.zeros_like(X.values, dtype=float)
    info = {}
    for i, c in enumerate(cols):
        v = X[c].values.astype(float)
        is_log = str(transforms.get(c, "")).startswith("log10")
        finite = np.isfinite(v)
        if is_log:
            m = v.copy()
            floor = None
        else:
            pos = v[finite & (v > 0)]
            floor = float(np.min(pos) / 2.0) if len(pos) else 1e-9
            m = np.log10(np.clip(v, floor, None))
        M[:, i] = m
        info[c] = {"is_log": is_log, "floor": floor}
    # 1.csv 的 73 个缺失格：在建模空间中用列中位数插补（生成空间必须有限）
    med = np.nanmedian(M, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    nan_idx = ~np.isfinite(M)
    M[nan_idx] = np.take(med, np.where(nan_idx)[1])
    mu = M.mean(axis=0)
    sd = M.std(axis=0, ddof=1)
    sd = np.where(sd > 1e-9, sd, 1.0)
    Z = (M - mu) / sd
    return M, Z, {"cols": cols, "info": info, "mu": mu, "sd": sd}


def m_to_original(M, space):
    """由 M（对数/浓度空间）恢复 1.csv 原始量纲（log10 列=v；原始列=10^m）。"""
    X = np.zeros_like(M)
    for i, c in enumerate(space["cols"]):
        if space["info"][c]["is_log"]:
            X[:, i] = M[:, i]
        else:
            X[:, i] = 10.0 ** M[:, i]
    return X


def defect_concentrations(M, space):
    """返回 {缺陷名: 浓度数组}；四列在建模空间中均已为 log10(浓度)。"""
    out = {}
    for key, d in DEFECTS.items():
        i = space["cols"].index(d["col"])
        out[key] = 10.0 ** M[:, i]
    return out


def project_defects(M, space, limits):
    """把缺陷列投影到 limits（浓度）以下（留 5% 余量）。"""
    M2 = M.copy()
    for key, limit in limits.items():
        if limit is None or not np.isfinite(limit) or limit <= 0:
            continue
        i = space["cols"].index(DEFECTS[key]["col"])
        cap = np.log10(limit * 0.95)
        M2[:, i] = np.minimum(M2[:, i], cap)
    return M2


# ---------------------------------------------------------------- cVAE
class CVAE(nn.Module):
    def __init__(self, n_feat, n_style, latent=8, hidden=128):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(n_feat + n_style, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden // 2, latent)
        self.logvar = nn.Linear(hidden // 2, latent)
        self.dec = nn.Sequential(
            nn.Linear(latent + n_style, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, hidden), nn.ReLU(),
            nn.Linear(hidden, n_feat),
        )

    def encode(self, x, c):
        h = self.enc(torch.cat([x, c], dim=1))
        return self.mu(h), self.logvar(h)

    def decode(self, z, c):
        return self.dec(torch.cat([z, c], dim=1))

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return self.decode(z, c), mu, logvar


def train_cvae(Z, onehot, beta=0.005, epochs=3000, lr=1e-3, latent=8, seed=RANDOM_SEED):
    torch.manual_seed(seed)
    n, p = Z.shape
    model = CVAE(p, onehot.shape[1], latent=latent)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt = torch.tensor(Z, dtype=torch.float32)
    Ct = torch.tensor(onehot, dtype=torch.float32)
    losses = []
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        recon, mu, logvar = model(Xt, Ct)
        rec = ((recon - Xt) ** 2).mean()
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = rec + beta * kl
        loss.backward()
        opt.step()
        losses.append(float(loss))
    model.eval()
    with torch.no_grad():
        recon, mu, logvar = model(Xt, Ct)
    per_feat_r2 = []
    for j in range(p):
        y = Z[:, j]
        yh = recon[:, j].numpy()
        ss_res = ((y - yh) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        per_feat_r2.append(1 - ss_res / ss_tot if ss_tot > 0 else np.nan)
    return model, {"final_loss": losses[-1], "recon_mse": float(((recon - Xt) ** 2).mean()),
                   "mean_feature_r2": float(np.nanmean(per_feat_r2)), "loss_curve": losses[::100]}


# ---------------------------------------------------------------- 评价工具
def rbf_mmd(x, y, gamma=None):
    x = np.asarray(x); y = np.asarray(y)
    if len(x) < 2 or len(y) < 2:
        return np.nan
    if gamma is None:
        z = np.vstack([x, y])
        d2 = ((z[:, None, :] - z[None, :, :]) ** 2).sum(-1)
        med = np.median(d2[d2 > 0])
        gamma = 1.0 / med if med > 0 else 1.0
    def k(a, b):
        d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
        return np.exp(-gamma * d2)
    return float(k(x, x).mean() + k(y, y).mean() - 2 * k(x, y).mean())


def corr_distance(A, B):
    ca = np.corrcoef(A, rowvar=False)
    cb = np.corrcoef(B, rowvar=False)
    mask = np.isfinite(ca) & np.isfinite(cb)
    if mask.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((np.nan_to_num(ca) - np.nan_to_num(cb)) ** 2)))


def main():
    t0 = time.time()
    X, Y, styles, ids = load_data()
    cols = list(X.columns)
    transforms = classify_transforms()
    M, Z, space = build_space(X, transforms)
    style_names = sorted(styles.unique())
    sidx = {s: i for i, s in enumerate(style_names)}
    y = np.array([sidx[s] for s in styles])
    onehot = np.eye(len(style_names))[y]

    # 训练 cVAE
    model, train_info = train_cvae(Z, onehot)
    with torch.no_grad():
        Zt = torch.tensor(Z, dtype=torch.float32)
        Ct = torch.tensor(onehot, dtype=torch.float32)
        mu_all, _ = model.encode(Zt, Ct)
        recon_all = model.decode(mu_all, Ct).numpy()
    style_mu = np.array([mu_all[y == k].mean(axis=0).numpy() for k in range(len(style_names))])

    # 风格分类器（用于评价生成谱是否像目标风格）
    clf = RandomForestClassifier(n_estimators=300, max_features="sqrt", n_jobs=8, random_state=RANDOM_SEED)
    cv_acc = float(np.mean(cross_val_score(clf, Z, y, cv=StratifiedKFold(5, shuffle=True,
                                                                        random_state=RANDOM_SEED))))
    clf.fit(Z, y)

    # Q1 感官代理
    q1 = joblib.load(os.path.join(OUT, "03_q1_模型.joblib"))
    q1_imp = q1["imputer"]
    q1_model = q1["model"]
    q1_targets = q1["target_columns"]

    def q1_predict(X_orig):
        Xdf = pd.DataFrame(X_orig, columns=cols)
        return pd.DataFrame(q1_model.predict(q1_imp.transform(Xdf)), columns=q1_targets)

    # 7.csv 消费者描述符辅助模型（决策 4B）
    df7 = pd.read_csv(os.path.join(MAT, "7.csv"))
    desc_cols = [c for c in df7.columns if c not in ["beer", "beer_id", "tasting_category_fine"]]
    desc_cols = [c for c in desc_cols if df7[c].notna().sum() > 100]
    cons_y = df7[desc_cols].values.astype(float)
    cons_imp = SimpleImputer(strategy="median").fit(X.values)
    cons_model = Ridge(alpha=10.0).fit(cons_imp.transform(X.values), cons_y)
    cons_r2 = {}
    for i in range(0, len(desc_cols), 25):
        cons_r2[desc_cols[i]] = None  # 占位，不逐列算 CV（避免过长）
    KEY_DESC = ["aroma_fruity", "aroma_estery", "aroma_hoppy", "aroma_malty", "aroma_sweet",
                "taste_bitter", "taste_sour", "palate_body", "palate_alcohol"]
    KEY_DESC = [c for c in KEY_DESC if c in desc_cols]

    def consumer_predict(X_orig):
        return pd.DataFrame(cons_model.predict(cons_imp.transform(X_orig)),
                            columns=desc_cols)

    # ---------------- 生成：每个风格采样 ----------------
    rng = np.random.default_rng(RANDOM_SEED)
    gen_rows = []
    n_per_style = 60
    gen_Z_by_style = {}
    gen_X_by_style = {}
    gen_sensory = []
    pre_defect = {k: [] for k, d in DEFECTS.items() if d["threshold"] is not None}
    capped_defect = {k: [] for k, d in DEFECTS.items() if d["threshold"] is not None}
    for k, s in enumerate(style_names):
        zs = (style_mu[k][None, :] + rng.normal(0, 0.3, size=(n_per_style, mu_all.shape[1]))).astype(np.float32)
        with torch.no_grad():
            cond = torch.tensor(np.tile(onehot[y == k][0], (n_per_style, 1)), dtype=torch.float32)
            rec = model.decode(torch.tensor(zs), cond).numpy()
        Mrec_raw = rec * space["sd"] + space["mu"]
        conc_raw = defect_concentrations(Mrec_raw, space)
        for key in pre_defect:
            pre_defect[key].append(float(np.mean(conc_raw[key] > DEFECTS[key]["threshold"])))
        # 生成谱的缺陷硬约束（阈值）
        limits = {key: d["threshold"] for key, d in DEFECTS.items() if d["threshold"] is not None}
        Mrec = project_defects(Mrec_raw, space, limits)
        conc_proj = defect_concentrations(Mrec, space)
        for key in capped_defect:
            capped_defect[key].append(float(np.mean(conc_raw[key] > limits[key] * 0.95)))
        Xrec = m_to_original(Mrec, space)
        gen_Z_by_style[s] = rec
        gen_X_by_style[s] = Xrec
        pred = q1_predict(Xrec)
        gen_sensory.append(pred)
        for i in range(n_per_style):
            row = {"style": s, "sample": i}
            row.update({c: Xrec[i, j] for j, c in enumerate(cols)})
            gen_rows.append(row)
    gen_df = pd.DataFrame(gen_rows)
    gen_df.to_csv(os.path.join(OUT, "05_q3_生成谱.csv"), index=False, encoding="utf-8-sig")
    gen_sens_df = pd.concat(gen_sensory, ignore_index=True)
    gen_sens_df.insert(0, "style", np.repeat(style_names, n_per_style))
    gen_sens_df.to_csv(os.path.join(OUT, "05_q3_生成_代理画像.csv"), index=False, encoding="utf-8-sig")

    # 生成评价
    gen_Z_all = np.vstack([gen_Z_by_style[s] for s in style_names])
    gen_style_true = np.repeat(np.arange(len(style_names)), n_per_style)
    gen_clf_acc = float(accuracy_score(gen_style_true, clf.predict(gen_Z_all)))
    gen_proba = clf.predict_proba(gen_Z_all)
    gen_conf = float(gen_proba.max(axis=1).mean())
    mmds, corr_ds = {}, {}
    real_sens = {}
    for k, s in enumerate(style_names):
        real_z = Z[y == k]
        gen_z = gen_Z_by_style[s]
        mmds[s] = rbf_mmd(gen_z, real_z)
        corr_ds[s] = corr_distance(gen_z, real_z)
        real_sens[s] = q1_predict(X.values[y == k])
    mmd_ref = {}
    real_defect_share = {k: [] for k, d in DEFECTS.items() if d["threshold"] is not None}
    real_conc_all = defect_concentrations(M, space)
    for k, s in enumerate(style_names):
        rz = Z[y == k]
        h = max(2, len(rz) // 2)
        mmd_ref[s] = rbf_mmd(rz[:h], rz[h:]) if len(rz) >= 4 else np.nan
        for key in real_defect_share:
            real_defect_share[key].append(float(np.mean(real_conc_all[key][y == k] > DEFECTS[key]["threshold"])))
    real_sens_mae_ref = float(np.mean([np.mean(np.abs(real_sens[s].values - real_sens[s].values.mean(axis=0)))
                                       for s in style_names]))
    gen_style_mean_sens = {s: gen_sens_df[gen_sens_df["style"] == s][q1_targets].values for s in style_names}
    sens_mae = {s: float(np.mean(np.abs(gen_style_mean_sens[s].mean(axis=0)
                                       - real_sens[s].values.mean(axis=0)))) for s in style_names}

    # 生成谱缺陷：投影后是否全部低于阈值
    gen_conc = defect_concentrations(gen_df[cols].apply(lambda r: 0, axis=1).values * 0 + 0, space) if False else None
    # 直接由生成的 M 反推：重建 gen_X -> M
    def orig_to_M(X_orig):
        Xdf = pd.DataFrame(X_orig, columns=cols)
        Mtmp = np.zeros_like(X_orig, dtype=float)
        for i, c in enumerate(cols):
            if space["info"][c]["is_log"]:
                Mtmp[:, i] = Xdf[c].values
            else:
                Mtmp[:, i] = np.log10(np.clip(Xdf[c].values, space["info"][c]["floor"], None))
        return Mtmp
    gen_M_all = orig_to_M(gen_df[cols].values)
    gen_conc_all = defect_concentrations(gen_M_all, space)
    gen_defect_stats = {}
    for key, d in DEFECTS.items():
        if d["threshold"] is None:
            gen_defect_stats[key] = {"threshold": None, "above_share": None,
                                     "median_conc": float(np.median(gen_conc_all[key]))}
        else:
            gen_defect_stats[key] = {"threshold": d["threshold"],
                                     "above_share": float(np.mean(gen_conc_all[key] > d["threshold"])),
                                     "pre_projection_above_share": float(np.mean(pre_defect[key])) if key in pre_defect else None,
                                     "capped_share": float(np.mean(capped_defect[key])) if key in capped_defect else None,
                                     "median_conc": float(np.median(gen_conc_all[key]))}

    # ---------------- 跨风格迁移 ----------------
    pairs = [("Pils/Lager", "Tripel"), ("Blond", "Tripel"), ("Lambic", "Kriek"),
             ("Wheat", "Saison"), ("Stout/Porter", "Fruitbeer")]
    pairs = [(a, b) for a, b in pairs if a in style_names and b in style_names]
    transfer_rows, transfer_metrics = [], {}
    K = 8
    for src, tgt in pairs:
        src_mask = (styles.values == src)
        Xsrc, Msrc = X.values[src_mask], M[src_mask]
        Zsrc = mu_all.numpy()[src_mask]  # 源样品在潜空间的均值（latent 向量）
        src_sens = q1_predict(Xsrc)
        src_conc = defect_concentrations(Msrc, space)
        tgt_cond = torch.tensor(np.tile(onehot[sidx[tgt]], (len(Xsrc) * K, 1)), dtype=torch.float32)
        with torch.no_grad():
            z_base = torch.tensor(np.repeat(Zsrc, K, axis=0), dtype=torch.float32)
            noise = torch.randn_like(z_base) * 0.3
            rec = model.decode(z_base + noise, tgt_cond).numpy()
        Mrec = rec * space["sd"] + space["mu"]
        # 缺陷上限：不劣于源样品，且不超过官方阈值（有阈值时）
        limits = {}
        for key, d in DEFECTS.items():
            base = src_conc[key].repeat(K)
            if d["threshold"] is None:
                limits[key] = base
            else:
                limits[key] = np.minimum(base, d["threshold"])
        # 向量化投影
        for key, limit in limits.items():
            i = cols.index(DEFECTS[key]["col"])
            cap = np.log10(np.clip(limit * 0.95, 1e-12, None))
            Mrec[:, i] = np.minimum(Mrec[:, i], cap)
        Xrec = m_to_original(Mrec, space)
        pred = q1_predict(Xrec)
        proba = clf.predict_proba(rec * space["sd"] + space["mu"] - 0) if False else clf.predict_proba(rec)
        stale = pred["stale_hops"].values
        src_stale = np.repeat(src_sens["stale_hops"].values, K)
        # best-of-K：风格概率高 + stale_hops 不恶化
        score = proba[:, sidx[tgt]] - 2.0 * np.maximum(0, stale - src_stale - 0.3)
        best = np.array([i * K + int(np.argmax(score[i * K:(i + 1) * K])) for i in range(len(Xsrc))])
        Xbest = Xrec[best]
        Mbest = Mrec[best]
        pred_best = q1_predict(Xbest)
        conc_best = defect_concentrations(Mbest, space)
        # 对照：潜空间风格方向平移
        z_shift = Zsrc + (style_mu[sidx[tgt]] - style_mu[sidx[src]])
        with torch.no_grad():
            cond_src = torch.tensor(onehot[sidx[tgt]][None, :].repeat(len(Xsrc), 0), dtype=torch.float32)
            rec_shift = model.decode(torch.tensor(z_shift, dtype=torch.float32), cond_src).numpy()
        M_shift = rec_shift * space["sd"] + space["mu"]
        for key, d in DEFECTS.items():
            i = cols.index(d["col"])
            base = src_conc[key]
            limit = base if d["threshold"] is None else np.minimum(base, d["threshold"])
            M_shift[:, i] = np.minimum(M_shift[:, i], np.log10(np.clip(limit * 0.95, 1e-12, None)))
        X_shift = m_to_original(M_shift, space)
        pred_shift = q1_predict(X_shift)
        # 方法选择（只看真实数据训练的风格分类器置信度，不使用目标画像信息）：
        # 条件解码 vs 潜空间风格方向平移
        conf_primary = float(proba[best, sidx[tgt]].mean())
        conf_shift = float(clf.predict_proba(rec_shift)[:, sidx[tgt]].mean())
        use_shift = conf_shift > conf_primary
        if use_shift:
            X_final, M_final, pred_final = X_shift, M_shift, pred_shift
        else:
            X_final, M_final, pred_final = Xbest, Mbest, pred_best
        conc_final = defect_concentrations(M_final, space)
        key_names = ["overall", "A_esters_all", "A_hops_all", "F_esters_all", "stale_hops", "diacetyl", "dms"]
        transfer_metrics[f"{src}->{tgt}"] = {
            "n_source": int(len(Xsrc)),
            "method": "latent_shift" if use_shift else "conditional_decode",
            "chosen_style_confidence": max(conf_primary, conf_shift),
            "primary_method_style_confidence": conf_primary,
            "shift_method_style_confidence": conf_shift,
            "primary_sensory_mae_vs_target": float(np.mean(np.abs(
                pred_best[q1_targets].values.mean(axis=0) - real_sens[tgt].values.mean(axis=0)))),
            "shift_sensory_mae_vs_target": float(np.mean(np.abs(
                pred_shift[q1_targets].values.mean(axis=0) - real_sens[tgt].values.mean(axis=0)))),
            "chosen_sensory_mae_vs_target": float(np.mean(np.abs(
                pred_final[q1_targets].values.mean(axis=0) - real_sens[tgt].values.mean(axis=0)))),
            "defect_worsened_share": {key: float(np.mean(conc_final[key] > src_conc[key] * 1.0001))
                                      for key in DEFECTS},
            "above_threshold_share": {key: (None if DEFECTS[key]["threshold"] is None else
                                            float(np.mean(conc_final[key] > DEFECTS[key]["threshold"])))
                                      for key in DEFECTS},
            "sensory_change": {k: float(pred_final[k].mean() - src_sens[k].mean()) for k in key_names},
        }
        # 保存迁移结果
        for i in range(len(Xsrc)):
            row = {"pair": f"{src}->{tgt}", "source_index": int(np.where(src_mask)[0][i])}
            row.update({c: X_final[i, j] for j, c in enumerate(cols)})
            transfer_rows.append(row)
    transfer_df = pd.DataFrame(transfer_rows)
    transfer_df.to_csv(os.path.join(OUT, "05_q3_迁移结果.csv"), index=False, encoding="utf-8-sig")

    # ---------------- 总体评价 ----------------
    eval_out = {
        "config": {"cvae": {"latent": 8, "beta": 0.005, "epochs": 3000, "hidden": 128},
                   "generation": {"n_per_style": n_per_style,
                                  "sampling": "风格后验均值 + N(0, 0.3^2)（小样本比先验采样更贴近真实分布）"},
                   "transfer": {"pairs": pairs, "K_candidates": K, "noise_sigma": 0.3,
                                "method_selection": "按真实数据训练的风格分类器置信度选择条件解码/潜空间平移"},
                   "defects": {k: {"unit": d["unit"], "threshold": d["threshold"]} for k, d in DEFECTS.items()},
                   "random_seed": RANDOM_SEED},
        "cvae_train": {k: v for k, v in train_info.items() if k != "loss_curve"},
        "style_classifier_cv_accuracy": cv_acc,
        "generated": {
            "style_classifier_accuracy": gen_clf_acc,
            "style_confidence_mean": gen_conf,
            "mmd_mean": float(np.nanmean(list(mmds.values()))),
            "mmd_reference_mean": float(np.nanmean(list(mmd_ref.values()))),
            "corr_structure_distance_mean": float(np.nanmean(list(corr_ds.values()))),
            "surrogate_sensory_mae_mean": float(np.mean(list(sens_mae.values()))),
            "surrogate_sensory_mae_reference": real_sens_mae_ref,
            "real_defect_above_share": {key: float(np.mean(v)) for key, v in real_defect_share.items()},
            "defect_stats": gen_defect_stats,
            "per_style": {s: {"mmd": mmds[s], "corr_distance": corr_ds[s],
                              "surrogate_mae": sens_mae[s]} for s in style_names},
        },
        "transfer": transfer_metrics,
        "consumer_target": {
            "n_descriptors": len(desc_cols),
            "note": "7.csv 消费者描述符作为辅助目标（决策 4B）；Ridge 代理仅用于相对比较，非真实消费者实验",
        },
        "runtime_seconds": round(time.time() - t0, 1),
        "env": {"python": sys.version.split()[0], "torch": torch.__version__,
                "numpy": np.__version__, "pandas": pd.__version__},
    }

    # 消费者描述符：生成 vs 真实目标风格
    cons_real = {s: consumer_predict(X.values[y == (sidx[s])]).values.mean(axis=0) for s in style_names}
    cons_gen = {s: consumer_predict(gen_X_by_style[s]).values.mean(axis=0) for s in style_names}
    eval_out["consumer_target"]["generated_vs_real_mae"] = {
        s: float(np.mean(np.abs(cons_gen[s] - cons_real[s]))) for s in style_names}
    eval_out["consumer_target"]["key_descriptors"] = {
        s: {c: {"real": float(cons_real[s][desc_cols.index(c)]),
                "generated": float(cons_gen[s][desc_cols.index(c)])} for c in KEY_DESC}
        for s in ["Tripel", "Kriek", "Lambic", "Pils/Lager"] if s in style_names}
    save_json(eval_out, os.path.join(OUT, "05_q3_评价.json"))

    # ---------------- 图 ----------------
    # 1) 潜空间 PCA
    pca = PCA(n_components=2).fit_transform(mu_all.numpy())
    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = plt.cm.get_cmap("tab20", len(style_names))
    for k, s in enumerate(style_names):
        mask = y == k
        ax.scatter(pca[mask, 0], pca[mask, 1], s=22, color=cmap(k), label=s, alpha=0.85)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("cVAE 潜空间（按风格着色）")
    ax.legend(fontsize=6, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_q3_latent_space.png"), dpi=200)
    plt.close(fig)

    # 2) 生成 vs 真实：若干风格的感官画像（Q1 代理预测均值）
    show_styles = [s for s in ["Tripel", "Kriek", "Lambic", "Pils/Lager"] if s in style_names]
    fig, axes = plt.subplots(1, len(show_styles), figsize=(5 * len(show_styles), 4.2))
    if len(show_styles) == 1:
        axes = [axes]
    for ax, s in zip(axes, show_styles):
        real = real_sens[s].values.mean(axis=0)
        gen = gen_style_mean_sens[s].mean(axis=0)
        ax.scatter(real, gen, s=8, alpha=0.6, color="#4C72B0")
        lim = [min(real.min(), gen.min()), max(real.max(), gen.max())]
        ax.plot(lim, lim, "--", color="gray", lw=0.8)
        ax.set_title(f"{s}（代理画像，MAE={sens_mae[s]:.3f}）")
        ax.set_xlabel("真实风格均值"); ax.set_ylabel("生成风格均值")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_q3_sensory_profile.png"), dpi=200)
    plt.close(fig)

    # 3) 迁移：缺陷浓度与阈值
    fig, axes = plt.subplots(1, len(pairs), figsize=(4.5 * len(pairs), 4.2))
    if len(pairs) == 1:
        axes = [axes]
    for ax, (src, tgt) in zip(axes, pairs):
        src_mask = styles.values == src
        src_conc = defect_concentrations(M[src_mask], space)
        # 读取该对的迁移结果
        sub = transfer_df[transfer_df.pair == f"{src}->{tgt}"]
        Xsub = sub[cols].values
        Msub = orig_to_M(Xsub)
        conc = defect_concentrations(Msub, space)
        labels, vals = [], []
        for key, d in DEFECTS.items():
            labels.extend([f"{key}\n源", f"{key}\n迁移"])
            vals.extend([np.median(src_conc[key]), np.median(conc[key])])
        ax.bar(range(len(vals)), vals, color=["#4C72B0", "#C44E52"] * len(DEFECTS))
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=70, fontsize=6)
        ax.set_yscale("log")
        ax.set_title(f"{src} → {tgt}")
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_q3_defects_transfer.png"), dpi=200)
    plt.close(fig)

    # 4) 迁移前后代理感官变化（Pils/Lager→Tripel 示例）
    pair0 = pairs[0] if pairs else None
    if pair0:
        src, tgt = pair0
        src_mask = styles.values == src
        Xsrc = X.values[src_mask]
        src_sens = q1_predict(Xsrc)
        sub = transfer_df[transfer_df.pair == f"{src}->{tgt}"]
        trans_sens = q1_predict(sub[cols].values)
        real_tgt = real_sens[tgt]
        dims = ["overall", "A_esters_all", "A_hops_all", "F_esters_all", "body", "bitternes",
                "stale_hops", "diacetyl", "dms"]
        dims = [d for d in dims if d in q1_targets]
        fig, ax = plt.subplots(figsize=(10, 4.5))
        xpos = np.arange(len(dims))
        ax.bar(xpos - 0.2, [src_sens[d].mean() for d in dims], width=0.4, label=f"源风格 {src}")
        ax.bar(xpos, [trans_sens[d].mean() for d in dims], width=0.4, label=f"迁移后（{tgt} 条件）")
        ax.bar(xpos + 0.2, [real_tgt[d].mean() for d in dims], width=0.4, label=f"真实 {tgt}")
        ax.set_xticks(xpos); ax.set_xticklabels(dims, rotation=30)
        ax.set_ylabel("Q1 代理感官 z 分数均值")
        ax.set_title(f"跨风格迁移示例：{src} → {tgt}")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "fig_q3_transfer_sensory.png"), dpi=200)
        plt.close(fig)

    # 5) 评价指标汇总
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = ["风格分类准确率(生成)", "代理画像 MAE", "MMD(均值)", "相关结构距离"]
    vals = [gen_clf_acc, float(np.mean(list(sens_mae.values()))),
            float(np.nanmean(list(mmds.values()))), float(np.nanmean(list(corr_ds.values())))]
    ax.bar(names, vals, color="#55A868")
    ax.set_title("问题三 生成质量评价（越小越好的指标已标注）")
    ax.set_ylabel("指标值")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_q3_metrics.png"), dpi=200)
    plt.close(fig)

    print(json.dumps({"cvae": eval_out["cvae_train"],
                      "generated": {k: v for k, v in eval_out["generated"].items() if k != "per_style"},
                      "transfer": list(eval_out["transfer"].keys()),
                      "consumer_mae": eval_out["consumer_target"]["generated_vs_real_mae"],
                      "runtime": eval_out["runtime_seconds"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
