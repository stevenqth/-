# -*- coding: utf-8 -*-
"""04_q2_top3.py — 问题二：每个感官维度的 Top-3 关键化合物（结构先验 + 多方法集成 + 稳定性）。

方法（对应方案文档“问题二”与决策 1A/3A/4B）：
    证据 1：风格控制偏相关（对 22 种风格哑变量做残差化后计算 Pearson 相关）
    证据 2：Ridge 系数（标准化特征上的线性监督重要性）
    证据 3：随机森林基尼重要性（逐感官维度训练）
    证据 4：官方 6.csv 交叉相关（44 个化合物，外部证据；缺失时按可用证据平均）
    结构先验：2.csv 共表达图（|r|≥0.5 阈值化）→ 社区发现 → 模块
    集成：各证据先转百分位，再按可用证据取平均
    改进 1：Top-3 施加“模块多样性约束”（同一共表达模块最多 1 个），
            以缓解参考论文自述的共线性导致重要性偏差的问题
    改进 2：稳定性用“按风格分层 5 折×2 次重复”重估全部证据与多样性 Top-3 的入选频率
    改进 3：模块级重要性（模块内 RF 重要性求和）作为补充证据与解释
    化学合理性：阈值表、来源类别（酵母/酒花/麦芽/野生菌群/其他）、官方 6.csv 符号一致性

运行：
    $env:PYTHONPATH = "D:\\github mama\\.work\\pylibs"
    D:\\miniconda\\python.exe 成果\\code\\04_q2_top3.py
输出：
    成果/results/04_q2_Top3.csv
    成果/results/04_q2_模块重要性.csv
    成果/results/04_q2_证据汇总.json
    成果/figures/fig_q2_*.png
"""
import os
import sys
import json
import time
import warnings

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_prep_common import load_data, make_folds, save_json, RANDOM_SEED

warnings.filterwarnings("ignore")
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

TOPK = 3
GRAPH_THRESHOLD = 0.5
N_FOLDS = 5
REPEATS = [RANDOM_SEED, 1024, 2024, 7, 99]
RF_TREES_FULL = 300
RF_TREES_FOLD = 150


def norm(s):
    import re
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def load_mapping():
    mp = pd.read_csv(os.path.join(OUT, "00_数据字典_字段映射.csv"), encoding="utf-8-sig")
    return mp


def build_chem_display():
    thr = pd.read_excel(os.path.join(MAT, "8.xlsx"), sheet_name="Sensory thresholds")
    disp, source, threshold = {}, {}, {}
    for _, r in thr.iterrows():
        keys = [r.get("field_name_R"), r.get("field_name_python")]
        keys = [k for k in keys if isinstance(k, str)]
        if not keys:
            continue
        name = r.get("chemical_name_short") if isinstance(r.get("chemical_name_short"), str) else (
            r.get("chemical_name") if isinstance(r.get("chemical_name"), str) else keys[0])
        src = r.get("origin_paper") if isinstance(r.get("origin_paper"), str) else "未标注"
        unit = r.get("unit") if isinstance(r.get("unit"), str) else ""
        tmean = r.get("threshold_mean")
        try:
            tmean = None if pd.isna(tmean) else float(tmean)
        except Exception:
            tmean = None
        for k in keys:
            disp[norm(k)] = name
            source[norm(k)] = src
            threshold[norm(k)] = {"threshold_mean": tmean, "unit": unit}
    return disp, source, threshold


def build_graph(chem_cols):
    df2 = pd.read_csv(os.path.join(MAT, "2.csv"), index_col=0)
    mp = load_mapping()
    mp2 = mp[mp["source"] == "2.csv"]
    # 去重：多个标签可能映射到同一列（或同一标签重复），只保留第一个且保证列唯一
    label_to_col, used = {}, set()
    for _, r in mp2.iterrows():
        lab, col = r["label"], r["target"]
        if lab in df2.index and col not in used:
            label_to_col[lab] = col
            used.add(col)
    labels = list(label_to_col.keys())
    cols = [label_to_col[l] for l in labels]
    sub = df2.loc[labels, labels].copy()
    sub.index = cols
    sub.columns = cols
    G = nx.Graph()
    G.add_nodes_from(chem_cols)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = sub.loc[a, b]
            if pd.notna(r) and abs(r) >= GRAPH_THRESHOLD:
                G.add_edge(a, b, weight=float(abs(r)))
    communities = list(nx.algorithms.community.greedy_modularity_communities(G, weight="weight"))
    module = {}
    for cid, comm in enumerate(communities):
        for node in comm:
            module[node] = cid
    for c in chem_cols:
        module.setdefault(c, len(communities))  # 无连接节点归入独立模块
    degree = pd.Series({c: G.degree(c, weight="weight") for c in chem_cols}, dtype=float)
    return G, module, degree, communities


def partial_corr_style(X, Y, styles):
    D = pd.get_dummies(styles, drop_first=False).astype(float).values
    D = np.column_stack([np.ones(len(D)), D])
    cX, *_ = np.linalg.lstsq(D, X, rcond=None)
    rX = X - D @ cX
    cY, *_ = np.linalg.lstsq(D, Y, rcond=None)
    rY = Y - D @ cY
    rX = (rX - rX.mean(0)) / (rX.std(0) + 1e-12)
    rY = (rY - rY.mean(0)) / (rY.std(0) + 1e-12)
    return (rX.T @ rY) / len(X)


def percentile_score(values):
    s = pd.Series(values, dtype=float)
    if s.notna().sum() <= 1:
        return np.full(len(s), np.nan)
    ranks = s.rank(ascending=True, method="average")
    n = s.notna().sum()
    return ((ranks - 1) / (n - 1)).values


def compute_evidence(X_imp, Y, styles, chem_cols, official_corr, rf_trees, n_jobs=8):
    """返回 percentile 证据 dict 与原始 RF 重要性矩阵 (n_features, n_targets)。"""
    C = partial_corr_style(X_imp, Y.values, styles)
    ev_partial = np.column_stack([percentile_score(np.abs(C[:, j])) for j in range(Y.shape[1])])

    Xs = StandardScaler().fit_transform(X_imp)
    ridge = Ridge(alpha=10.0).fit(Xs, Y.values)
    coef = np.abs(ridge.coef_)
    ev_ridge = np.column_stack([percentile_score(coef[j]) for j in range(Y.shape[1])])

    ev_rf = np.zeros((X_imp.shape[1], Y.shape[1]))
    raw_imp = np.zeros_like(ev_rf)
    for j in range(Y.shape[1]):
        rf = RandomForestRegressor(n_estimators=rf_trees, max_features="sqrt",
                                   min_samples_leaf=2, n_jobs=n_jobs, random_state=RANDOM_SEED)
        rf.fit(X_imp, Y.iloc[:, j].values)
        imp = rf.feature_importances_
        raw_imp[:, j] = imp
        ev_rf[:, j] = percentile_score(imp)

    ev_official = np.full((X_imp.shape[1], Y.shape[1]), np.nan)
    idx = {c: i for i, c in enumerate(chem_cols)}
    for j, scol in enumerate(Y.columns):
        if scol not in official_corr.columns:
            continue
        for compound, val in official_corr[scol].dropna().items():
            if compound in idx and pd.notna(val):
                ev_official[idx[compound], j] = abs(float(val))
    for j in range(Y.shape[1]):
        col = ev_official[:, j]
        mask = ~np.isnan(col)
        if mask.sum() > 1:
            ev_official[mask, j] = percentile_score(col[mask])
    return {"partial": ev_partial, "ridge": ev_ridge, "rf": ev_rf, "official": ev_official}, raw_imp


def composite_score(ev):
    stack = np.stack([ev[k] for k in ev.keys()], axis=0)
    with np.errstate(all="ignore"):
        return np.nanmean(stack, axis=0)


def module_topk(score, chem_cols, module, k=TOPK):
    """按综合得分从高到低选 Top-k，并施加“同一模块最多 1 个”的多样性约束。"""
    order = np.argsort(-score, axis=0)
    out = []
    for j in range(score.shape[1]):
        chosen, used = [], set()
        for i in order[:, j]:
            v = score[i, j]
            if not np.isfinite(v):
                continue
            c = chem_cols[i]
            m = module.get(c, -1)
            if m in used:
                continue
            chosen.append(c)
            used.add(m)
            if len(chosen) == k:
                break
        if len(chosen) < k:  # 模块数不足时回退为纯得分排序
            chosen = [chem_cols[i] for i in order[:k, j] if np.isfinite(score[i, j])]
        out.append(chosen)
    return out


def plain_topk(score, chem_cols, k=TOPK):
    order = np.argsort(-score, axis=0)
    return [[chem_cols[i] for i in order[:k, j]] for j in range(score.shape[1])]


def method_topk(ev, chem_cols, k=TOPK):
    out = {}
    for key, mat in ev.items():
        order = np.argsort(-mat, axis=0)
        out[key] = [[chem_cols[i] for i in order[:k, j] if np.isfinite(mat[i, j])]
                    for j in range(mat.shape[1])]
    return out


def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return np.nan
    return len(a & b) / len(a | b)


def main():
    t0 = time.time()
    X, Y, styles, ids = load_data()
    chem_cols = list(X.columns)
    targets = list(Y.columns)
    disp, source, threshold = build_chem_display()
    mp = load_mapping()

    # 官方 6.csv 映射
    df6 = pd.read_csv(os.path.join(MAT, "6.csv"), index_col=0)
    m6 = mp[mp["source"] == "6.csv"]
    chem_set, sens_set = set(chem_cols), set(targets)
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

    # 结构先验（全数据，结构固定后用于所有折）
    G, module, degree, communities = build_graph(chem_cols)

    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)
    ev_full, raw_imp = compute_evidence(X_imp, Y, styles, chem_cols, official, RF_TREES_FULL)
    C_raw = partial_corr_style(X_imp, Y.values, styles)
    score_full = composite_score(ev_full)
    top_full = module_topk(score_full, chem_cols, module)
    tops_plain = plain_topk(score_full, chem_cols)
    tops_by_method = method_topk(ev_full, chem_cols)

    # 模块级重要性（RF 重要性按模块求和）
    mod_ids = sorted(set(module.values()))
    mod_imp = np.zeros((len(mod_ids), len(targets)))
    for mi, mid in enumerate(mod_ids):
        idxs = [i for i, c in enumerate(chem_cols) if module.get(c) == mid]
        mod_imp[mi, :] = raw_imp[idxs, :].sum(axis=0)
    mod_df = pd.DataFrame(mod_imp, index=[f"module_{m}" for m in mod_ids], columns=targets)
    mod_df.to_csv(os.path.join(OUT, "04_q2_模块重要性.csv"), encoding="utf-8-sig")

    # ---------------- 稳定性：重复分层 CV 重估证据 + 多样性 Top-3 ----------------
    freq = np.zeros((len(chem_cols), len(targets)))
    freq_plain = np.zeros_like(freq)
    score_sum = np.zeros_like(score_full)
    module_freq = np.zeros((len(mod_ids), len(targets)))
    mod_index = {mid: i for i, mid in enumerate(mod_ids)}
    n_runs = 0
    for seed in REPEATS:
        for tr, va in make_folds(styles, N_FOLDS, seed):
            imp_tr = SimpleImputer(strategy="median").fit(X.iloc[tr])
            X_tr = imp_tr.transform(X.iloc[tr])
            ev, _ = compute_evidence(X_tr, Y.iloc[tr], styles.iloc[tr], chem_cols, official, RF_TREES_FOLD)
            score = composite_score(ev)
            score_sum += score
            top = module_topk(score, chem_cols, module)
            top_plain = plain_topk(score, chem_cols)
            for j, comps in enumerate(top):
                for c in comps:
                    freq[chem_cols.index(c), j] += 1
            for j, comps in enumerate(top_plain):
                for c in comps:
                    freq_plain[chem_cols.index(c), j] += 1
            # 模块级入选频次（折内模块综合得分前 3）
            # 模块得分取模块内最高化合物得分（与“每模块选 1 个代表化合物”的最终选择口径一致）
            mod_scores = np.array([score[[i for i, c in enumerate(chem_cols) if module.get(c) == mid], :].max(axis=0)
                                   for mid in mod_ids])
            for j in range(len(targets)):
                order = np.argsort(-mod_scores[:, j])[:TOPK]
                for mi in order:
                    module_freq[mi, j] += 1
            n_runs += 1
    freq /= n_runs
    freq_plain /= n_runs
    module_freq /= n_runs

    # 最终排序用“全量证据 + 25 次折内证据”的 bagged 得分（减少单次划分的噪声）
    score_bag = (score_sum + score_full) / (n_runs + 1)
    top_full = module_topk(score_bag, chem_cols, module)
    tops_plain = plain_topk(score_bag, chem_cols)

    # ---------------- 逐目标 Top-3 输出 ----------------
    rows, jaccards = [], []
    for j, target in enumerate(targets):
        top = top_full[j]
        pairs = []
        for a, b in [("partial", "ridge"), ("partial", "rf"), ("ridge", "rf"),
                     ("partial", "official"), ("rf", "official")]:
            ja, jb = tops_by_method[a][j], tops_by_method[b][j]
            if len(ja) >= TOPK and len(jb) >= TOPK:
                pairs.append(jaccard(ja, jb))
        mj = float(np.nanmean(pairs)) if pairs else np.nan
        jaccards.append(mj)
        mods_chosen = [module.get(c, -1) for c in top]
        for rk, c in enumerate(top, 1):
            i = chem_cols.index(c)
            rows.append(dict(
                target=target, rank=rk, compound=c,
                display_name=disp.get(norm(c), c),
                composite_score=float(score_bag[i, j]),
                bagged_score=float(score_bag[i, j]),
                pct_partial=float(ev_full["partial"][i, j]),
                pct_ridge=float(ev_full["ridge"][i, j]),
                pct_rf=float(ev_full["rf"][i, j]),
                pct_official=float(ev_full["official"][i, j]) if np.isfinite(ev_full["official"][i, j]) else np.nan,
                cv_frequency=float(freq[i, j]),
                cv_frequency_plain=float(freq_plain[i, j]),
                method_jaccard=mj,
                module=int(module.get(c, -1)),
                module_size=int(sum(1 for x in chem_cols if module.get(x) == module.get(c, -1))),
                module_diversity=len(set(mods_chosen)),
                module_cv_frequency=float(module_freq[mod_index.get(module.get(c, -1), 0), j]),
                source=source.get(norm(c), "未标注"),
                threshold_mean=threshold.get(norm(c), {}).get("threshold_mean"),
                threshold_unit=threshold.get(norm(c), {}).get("unit"),
            ))
    top3_df = pd.DataFrame(rows)
    top3_df["graph_degree"] = top3_df["compound"].map(degree).astype(float)
    # 官方 6.csv 符号一致性（Top-3 有官方对照时）
    sign_ok = []
    for _, r in top3_df.iterrows():
        val = np.nan
        if r["target"] in official.columns and r["compound"] in official.index:
            val = official.loc[r["compound"], r["target"]]
        raw = C_raw[chem_cols.index(r["compound"]), targets.index(r["target"])] if np.isfinite(val) else np.nan
        sign_ok.append(bool(np.sign(val) == np.sign(raw)) if np.isfinite(val) else np.nan)
    top3_df["official_sign_consistent"] = sign_ok
    top3_df.to_csv(os.path.join(OUT, "04_q2_Top3.csv"), index=False, encoding="utf-8-sig")

    # ---------------- 汇总 ----------------
    stable_all = 0
    for j in range(len(targets)):
        if min(freq[chem_cols.index(c), j] for c in top_full[j]) >= 0.8:
            stable_all += 1
    div = top3_df.groupby("target")["module_diversity"].first()
    top_counter = pd.Series([c for comps in top_full for c in comps]).value_counts()
    n_with_source = int((top3_df["source"] != "未标注").sum())
    n_with_threshold = int(top3_df["threshold_mean"].notna().sum())
    sign_vals = top3_df["official_sign_consistent"].dropna()
    summary = {
        "config": {"topk": TOPK, "graph_threshold": GRAPH_THRESHOLD,
                   "n_folds": N_FOLDS, "repeats": REPEATS, "stability_runs": n_runs,
                   "evidence": ["partial (style-controlled)", "ridge", "rf", "official 6.csv"],
                   "selection": "各证据百分位平均 → 模块多样性约束（同一模块最多 1 个）",
                   "rf_trees": {"full": RF_TREES_FULL, "fold": RF_TREES_FOLD}},
        "graph": {"n_nodes": int(G.number_of_nodes()), "n_edges": int(G.number_of_edges()),
                  "n_communities": int(len(communities)),
                  "community_sizes": sorted([len(c) for c in communities], reverse=True)[:10],
                  "mean_degree": float(degree.mean()), "max_degree": float(degree.max())},
        "stability": {
            "mean_top3_cv_frequency": float(np.mean([freq[chem_cols.index(c), j]
                                                    for j in range(len(targets)) for c in top_full[j]])),
            "min_target_mean_frequency": float(np.min([np.mean([freq[chem_cols.index(c), j] for c in top_full[j]])
                                                       for j in range(len(targets))])),
            "mean_target_min_frequency": float(np.mean([np.min([freq[chem_cols.index(c), j] for c in top_full[j]])
                                                        for j in range(len(targets))])),
            "mean_top3_cv_frequency_plain": float(np.mean([freq_plain[chem_cols.index(c), j]
                                                          for j in range(len(targets)) for c in tops_plain[j]])),
            "targets_all3_stable_ge0p8": int(sum(1 for j in range(len(targets))
                                                 if min(freq[chem_cols.index(c), j] for c in top_full[j]) >= 0.8)),
            "targets_min_freq_ge0p6": int(sum(1 for j in range(len(targets))
                                              if min(freq[chem_cols.index(c), j] for c in top_full[j]) >= 0.6)),
            "targets_min_freq_ge0p5": int(sum(1 for j in range(len(targets))
                                              if min(freq[chem_cols.index(c), j] for c in top_full[j]) >= 0.5)),
            "mean_top1_cv_frequency": float(np.mean([freq[chem_cols.index(top_full[j][0]), j] for j in range(len(targets))])),
            "targets_top1_ge0p8": int(sum(1 for j in range(len(targets)) if freq[chem_cols.index(top_full[j][0]), j] >= 0.8)),
            "mean_stable_core_size_ge0p5": float(np.mean([sum(1 for c in top_full[j] if freq[chem_cols.index(c), j] >= 0.5) for j in range(len(targets))])),
            "mean_module_cv_frequency": float(np.mean([module_freq[mod_index[module.get(c, -1)], j]
                                                       for j in range(len(targets)) for c in top_full[j]])),
            "n_targets": int(len(targets)),
            "mean_method_jaccard": float(np.nanmean(jaccards)),
            "mean_module_diversity": float(div.mean()),
            "selection_note": "最终 Top-3 由 bagged 得分（全量+25 折）排序；频率为 25 次折内重估的入选频率",
        },
        "chemical_plausibility": {
            "top3_with_known_source": n_with_source,
            "top3_with_threshold": n_with_threshold,
            "n_top3_entries": int(len(top3_df)),
            "official_sign_consistent_share": float(np.nanmean(sign_ok)) if len(sign_vals) else None,
            "official_sign_checked": int(len(sign_vals)),
        },
        "top_compounds_overall": top_counter.head(15).to_dict(),
        "official_coverage": {"compounds": int(official.shape[0]), "targets": int(official.shape[1])},
        "runtime_seconds": round(time.time() - t0, 1),
        "env": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__},
    }
    save_json(summary, os.path.join(OUT, "04_q2_证据汇总.json"))

    # ---------------- 图 ----------------
    freqs = np.array([freq[chem_cols.index(c), j] for j in range(len(targets)) for c in top_full[j]])
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    axes[0].hist(freqs, bins=np.linspace(0, 1, 21), color="#4C72B0", edgecolor="white")
    axes[0].set_xlabel("Top-3 入选频率（重复分层 CV）")
    axes[0].set_ylabel("化合物-维度组合数")
    axes[0].set_title("Top-3 稳定性分布")
    axes[1].bar(["全部 3 个稳定", "其余"], [stable_all, len(targets) - stable_all],
                color=["#55A868", "#C44E52"])
    axes[1].set_ylabel("感官维度数")
    axes[1].set_title("Top-3 全部稳定的维度数（频率 ≥0.8）")
    axes[2].hist(div.values, bins=np.arange(0.5, 4.5, 1), color="#8172B2", edgecolor="white")
    axes[2].set_xlabel("每个维度的不同模块数")
    axes[2].set_ylabel("感官维度数")
    axes[2].set_title("Top-3 模块多样性")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_q2_stability.png"), dpi=200)
    plt.close(fig)

    methods = ["partial", "ridge", "rf"]
    mat = np.zeros((3, 3))
    for a, ma in enumerate(methods):
        for b, mb in enumerate(methods):
            mat[a, b] = np.nanmean([jaccard(tops_by_method[ma][j], tops_by_method[mb][j])
                                    for j in range(len(targets))])
    fig, ax = plt.subplots(figsize=(5.8, 5))
    im = ax.imshow(mat, cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(range(3)); ax.set_xticklabels(["偏相关", "Ridge", "随机森林"])
    ax.set_yticks(range(3)); ax.set_yticklabels(["偏相关", "Ridge", "随机森林"])
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=11)
    ax.set_title("不同方法 Top-3 的平均 Jaccard 系数")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_q2_method_jaccard.png"), dpi=200)
    plt.close(fig)

    try:
        pos = nx.spring_layout(G, seed=RANDOM_SEED, k=0.25, iterations=60)
        top_compounds = set(c for comps in top_full for c in comps)
        fig, ax = plt.subplots(figsize=(11, 9))
        edges = list(G.edges(data=True))
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.08,
                               width=[0.3 + e[2].get("weight", 0.5) for e in edges])
        for cid in sorted(set(module.values())):
            nodes = [n for n, m in module.items() if m == cid]
            nx.draw_networkx_nodes(G, pos, nodelist=nodes, ax=ax, node_size=22,
                                   cmap=plt.cm.tab20, node_color=[cid] * len(nodes), alpha=0.75)
        nx.draw_networkx_nodes(G, pos, nodelist=[n for n in top_compounds if n in pos], ax=ax,
                               node_color="red", node_size=70, alpha=0.9)
        for n in top_compounds:
            if n in pos:
                ax.annotate(disp.get(norm(n), n), pos[n], fontsize=5, alpha=0.8)
        ax.set_title("化合物共表达图（|r|≥0.5）：颜色为社区，红点为进入任一维度 Top-3 的化合物")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "fig_q2_graph_modules.png"), dpi=180)
        plt.close(fig)
    except Exception as e:
        print("graph figure skipped:", e)

    # Top-3 高频化合物
    topc = top_counter.head(20)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.barh(list(topc.index)[::-1], list(topc.values)[::-1], color="#C44E52")
    ax.set_xlabel("进入 Top-3 的感官维度数（50 维）")
    ax.set_title("出现频率最高的 Top-3 化合物")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_q2_top_compounds.png"), dpi=200)
    plt.close(fig)

    # 官方交叉相关 vs 风格控制偏相关一致性
    common = official.stack().dropna()
    if len(common):
        C = partial_corr_style(X_imp, Y.values, styles)
        vals = [C[chem_cols.index(c), targets.index(s)] for (c, s) in common.index]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(common.values, vals, s=8, alpha=0.4, color="#4C72B0")
        ax.axhline(0, color="gray", lw=0.8); ax.axvline(0, color="gray", lw=0.8)
        ax.set_xlabel("官方 6.csv 交叉相关")
        ax.set_ylabel("本文偏相关（含风格控制）")
        ax.set_title("官方交叉相关与风格控制偏相关的一致性")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "fig_q2_consistency.png"), dpi=200)
        plt.close(fig)
        summary["consistency"] = {"n_pairs": int(len(common)),
                                  "pearson_official_vs_partial": float(np.corrcoef(common.values, vals)[0, 1])}
        save_json(summary, os.path.join(OUT, "04_q2_证据汇总.json"))

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
