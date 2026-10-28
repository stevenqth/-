# -*- coding: utf-8 -*-

# 沙箱环境：禁止 OpenMP/线程池创建管道（避免 WinError 5）
import os as _os
for _k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    _os.environ.setdefault(_k, '1')
"""04_q2_top3.py — 问题二：每个感官维度的 Top-3 关键化合物（结构先验 + 多方法集成 + 稳定性）。

方法对应方案文档“问题二”：
    证据 1：风格控制偏相关（对 22 种风格哑变量做残差化后计算）
    证据 2：Ridge 稀疏化系数（标准化特征上的线性监督重要性）
    证据 3：随机森林基尼重要性（逐感官维度训练）
    证据 4：官方 6.csv 交叉相关（44 个化合物，外部证据，缺失时按可用证据加权）
    结构先验：2.csv 共表达图（阈值化后社区发现，用于解释与模块一致性检查）
    集成：各证据先转百分位，再按可用证据取加权平均，按均值排序取 Top-3
    稳定性：按风格分层 5 折 × 2 次重复重采样下 Top-3 入选频率 + 跨方法 Jaccard 系数

运行：
    $env:PYTHONPATH = "D:\\github mama\\.work\\pylibs"
    D:\\miniconda\\python.exe 成果\\code\\04_q2_top3.py

输出：
    成果/results/04_q2_Top3.csv
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
from data_prep_common import (load_data, classify_transforms, make_folds, save_json, RANDOM_SEED)

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

META = ["beer", "beer_id", "tasting_category_fine"]
N_FOLDS = 5
REPEATS = [RANDOM_SEED, 1024]
GRAPH_THRESHOLD = 0.5
TOPK = 3


def norm(s):
    import re
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def load_mapping():
    """读取 00_数据字典_字段映射.csv，构造 6.csv 行（化合物）与列（感官）的映射。"""
    mp = pd.read_csv(os.path.join(OUT, "00_数据字典_字段映射.csv"), encoding="utf-8-sig")
    m6 = mp[mp["source"] == "6.csv"]
    # 6.csv 的行与列都在这份映射里；按 label 能否在 4.csv 感官列集合中区分
    # 统一：target 为 1.csv 化合物列名 或 4.csv 感官列名
    return m6


def build_chem_display():
    """1.csv 列名 -> 化学显示名 / 来源类别 / 阈值信息。

    8.xlsx 的 field_name_R 与 1.csv 列名同源（如 X2.3.butanedione），
    field_name_python 为论文侧命名（如 2,3-butanedione），两者都登记。
    """
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


def partial_corr_style(X, Y, styles):
    """控制风格后的偏相关（对哑变量回归取残差后计算 Pearson 相关）。"""
    D = pd.get_dummies(styles, drop_first=False).astype(float).values
    D = np.column_stack([np.ones(len(D)), D])
    # 最小二乘残差
    coefX, *_ = np.linalg.lstsq(D, X, rcond=None)
    rX = X - D @ coefX
    coefY, *_ = np.linalg.lstsq(D, Y, rcond=None)
    rY = Y - D @ coefY
    # 标准化后相关
    rX = (rX - rX.mean(0)) / (rX.std(0) + 1e-12)
    rY = (rY - rY.mean(0)) / (rY.std(0) + 1e-12)
    C = (rX.T @ rY) / len(X)
    return C  # (n_features, n_targets)


def percentile_score(values):
    """把“越大越重要”的分数转成 [0,1] 百分位（NaN 保持 NaN）。"""
    s = pd.Series(values, dtype=float)
    if s.notna().sum() <= 1:
        return np.full(len(s), np.nan)
    ranks = s.rank(ascending=True, method="average")
    n = s.notna().sum()
    pct = (ranks - 1) / (n - 1) if n > 1 else ranks
    return pct.values


def compute_evidence(X_imp, Y, styles, chem_cols, official_corr, ridge_alpha=10.0, rf_trees=300):
    """在给定训练数据上计算四类证据的百分位分数。

    official_corr: DataFrame (44 x 50)，列为 4.csv 感官列，行为 1.csv 化合物列。
    返回 dict: name -> (n_features, n_targets) percentile array（官方证据对未覆盖化合物为 NaN）。
    """
    # 偏相关
    C_open = partial_corr_style(X_imp, Y.values, styles)
    ev_partial = np.zeros_like(C_open, dtype=float)
    for j in range(C_open.shape[1]):
        ev_partial[:, j] = percentile_score(np.abs(C_open[:, j]))

    # Ridge 系数
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_imp)
    ridge = Ridge(alpha=ridge_alpha)
    ridge.fit(Xs, Y.values)
    coef = np.abs(ridge.coef_)  # (n_targets, n_features)
    ev_ridge = np.zeros((X_imp.shape[1], Y.shape[1]), dtype=float)
    for j in range(Y.shape[1]):
        ev_ridge[:, j] = percentile_score(coef[j])

    # 随机森林基尼重要性
    ev_rf = np.zeros((X_imp.shape[1], Y.shape[1]), dtype=float)
    rf_models = []
    for j, col in enumerate(Y.columns):
        rf = RandomForestRegressor(n_estimators=rf_trees, max_features="sqrt",
                                   min_samples_leaf=2, n_jobs=1, random_state=RANDOM_SEED)
        rf.fit(X_imp, Y.iloc[:, j].values)
        rf_models.append(rf)
        ev_rf[:, j] = percentile_score(rf.feature_importances_)

    # 官方 6.csv 交叉相关
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
            sub = pd.Series(col[mask])
            ev_official[mask, j] = percentile_score(sub.values)

    return {"partial": ev_partial, "ridge": ev_ridge, "rf": ev_rf, "official": ev_official}, rf_models


def composite_rank(ev):
    """对可用证据取平均百分位；返回 score (n_features, n_targets) 与各方法贡献。"""
    keys = list(ev.keys())
    stack = np.stack([ev[k] for k in keys], axis=0)  # (methods, feat, target)
    with np.errstate(all="ignore"):
        score = np.nanmean(stack, axis=0)
    return score


def rank_top3(score, chem_cols):
    order = np.argsort(-score, axis=0)
    top = []
    for j in range(score.shape[1]):
        top.append([chem_cols[i] for i in order[:TOPK, j]])
    return top


def method_top3(ev, chem_cols):
    out = {}
    for k, mat in ev.items():
        order = np.argsort(-mat, axis=0)
        top = []
        for j in range(mat.shape[1]):
            vals = mat[order[:TOPK, j], j]
            top.append([chem_cols[i] for i, v in zip(order[:TOPK, j], vals) if not np.isnan(v)])
        out[k] = top
    return out


def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return np.nan
    return len(a & b) / len(a | b)


def build_graph(X_imp, chem_cols):
    df2 = pd.read_csv(os.path.join(MAT, "2.csv"), index_col=0)
    mp = pd.read_csv(os.path.join(OUT, "00_数据字典_字段映射.csv"), encoding="utf-8-sig")
    mp2 = mp[mp["source"] == "2.csv"]
    label_to_col = dict(zip(mp2["label"], mp2["target"]))
    labels = [l for l in df2.index if label_to_col.get(l)]
    cols = [label_to_col[l] for l in labels]
    sub = df2.loc[labels, labels].copy()
    sub.index = cols
    sub.columns = cols
    G = nx.Graph()
    G.add_nodes_from(cols)
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
    degree = pd.Series(dict(G.degree(weight="weight")))
    return G, module, degree, communities


def main():
    t0 = time.time()
    X, Y, styles, ids = load_data()
    chem_cols = list(X.columns)
    transforms = classify_transforms()
    disp, source, threshold = build_chem_display()

    # 6.csv 映射到 1.csv 化合物列与 4.csv 感官列
    df6 = pd.read_csv(os.path.join(MAT, "6.csv"), index_col=0)
    mp = pd.read_csv(os.path.join(OUT, "00_数据字典_字段映射.csv"), encoding="utf-8-sig")
    m6 = mp[mp["source"] == "6.csv"]
    # 行标签 -> 1.csv 列；列标签 -> 4.csv 列；按“目标是否在 X/Y 列中”区分
    chem_set, sens_set = set(chem_cols), set(Y.columns)
    row_map, col_map = {}, {}
    for _, r in m6.iterrows():
        if r["target"] in chem_set:
            row_map[r["label"]] = r["target"]
        elif r["target"] in sens_set:
            col_map[r["label"]] = r["target"]
    official = df6.rename(index=row_map, columns=col_map)
    official = official.loc[[i for i in official.index if i in chem_set],
                            [c for c in official.columns if c in sens_set]]

    # 全量数据：中位数插补（仅用于证据计算）
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)

    ev_full, rf_models = compute_evidence(X_imp, Y, styles, chem_cols, official)
    score_full = composite_rank(ev_full)
    top_full = rank_top3(score_full, chem_cols)
    tops_by_method = method_top3(ev_full, chem_cols)

    # ---------- 稳定性：按风格分层重复 K 折，重估证据与 Top-3 ----------
    freq = np.zeros((len(chem_cols), len(Y.columns)))  # 入选频次
    n_runs = 0
    fold_top3 = []
    for rep, seed in enumerate(REPEATS):
        folds = make_folds(styles, N_FOLDS, seed)
        for tr, va in folds:
            imp_tr = SimpleImputer(strategy="median").fit(X.iloc[tr])
            X_tr = imp_tr.transform(X.iloc[tr])
            ev, _ = compute_evidence(X_tr, Y.iloc[tr], styles.iloc[tr], chem_cols, official)
            score = composite_rank(ev)
            top = rank_top3(score, chem_cols)
            fold_top3.append(top)
            for j, comps in enumerate(top):
                for c in comps:
                    freq[chem_cols.index(c), j] += 1
            n_runs += 1
    freq /= n_runs

    # 逐目标 Top-3 输出
    rows = []
    jaccards = []
    for j, target in enumerate(Y.columns):
        top = top_full[j]
        # 跨方法 Jaccard（官方证据仅在可用时参与）
        pairs = []
        for a, b in [(("partial", "ridge")), ("partial", "rf"), ("ridge", "rf"),
                     ("partial", "official"), ("rf", "official")]:
            ja = tops_by_method[a][j]
            jb = tops_by_method[b][j]
            if len(jb) >= TOPK and len(ja) >= TOPK:
                pairs.append(jaccard(ja, jb))
        mj = float(np.nanmean(pairs)) if pairs else np.nan
        jaccards.append(mj)
        for rk, c in enumerate(top, 1):
            i = chem_cols.index(c)
            rows.append(dict(
                target=target, rank=rk, compound=c,
                display_name=disp.get(norm(c), c),
                composite_score=float(score_full[i, j]),
                pct_partial=float(ev_full["partial"][i, j]),
                pct_ridge=float(ev_full["ridge"][i, j]),
                pct_rf=float(ev_full["rf"][i, j]),
                pct_official=float(ev_full["official"][i, j]) if pd.notna(ev_full["official"][i, j]) else np.nan,
                cv_frequency=float(freq[i, j]),
                method_jaccard=mj,
                source=source.get(norm(c), "未标注"),
                threshold_mean=threshold.get(norm(c), {}).get("threshold_mean"),
                threshold_unit=threshold.get(norm(c), {}).get("unit"),
            ))
    top3_df = pd.DataFrame(rows)
    top3_df.to_csv(os.path.join(OUT, "04_q2_Top3.csv"), index=False, encoding="utf-8-sig")

    # 结构先验：共表达图与社区
    G, module, degree, communities = build_graph(X_imp, chem_cols)
    for j, target in enumerate(Y.columns):
        pass
    top3_df["graph_module"] = top3_df["compound"].map(module)
    top3_df["graph_degree"] = top3_df["compound"].map(degree).astype(float)
    top3_df.to_csv(os.path.join(OUT, "04_q2_Top3.csv"), index=False, encoding="utf-8-sig")

    # 稳定性汇总
    stable_all = 0
    for j in range(len(Y.columns)):
        tf = [freq[chem_cols.index(c), j] for c in top_full[j]]
        if min(tf) >= 0.8:
            stable_all += 1
    summary = {
        "config": {
            "topk": TOPK, "graph_threshold": GRAPH_THRESHOLD,
            "n_folds": N_FOLDS, "repeats": REPEATS, "stability_runs": n_runs,
            "evidence": ["partial(风格控制偏相关)", "ridge(标准化系数)", "rf(基尼重要性)", "official(6.csv 交叉相关)"],
            "aggregation": "各证据百分位取可用证据平均",
        },
        "graph": {
            "n_nodes": int(G.number_of_nodes()), "n_edges": int(G.number_of_edges()),
            "n_communities": int(len(communities)),
            "community_sizes": sorted([len(c) for c in communities], reverse=True)[:10],
            "mean_degree": float(degree.mean()), "max_degree": float(degree.max()),
        },
        "stability": {
            "mean_top3_cv_frequency": float(np.mean([freq[chem_cols.index(c), j]
                                                    for j in range(len(Y.columns)) for c in top_full[j]])),
            "targets_all3_stable_ge0p8": int(stable_all),
            "n_targets": int(len(Y.columns)),
            "mean_method_jaccard": float(np.nanmean(jaccards)),
        },
        "evidence_coverage": {
            "official_compounds": int(official.shape[0]),
            "official_targets": int(official.shape[1]),
        },
        "runtime_seconds": round(time.time() - t0, 1),
        "env": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__},
    }
    save_json(summary, os.path.join(OUT, "04_q2_证据汇总.json"))

    # ---------- 图 ----------
    # 1) Top-3 入选频率分布
    freqs = np.array([freq[chem_cols.index(c), j] for j in range(len(Y.columns)) for c in top_full[j]])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].hist(freqs, bins=np.linspace(0, 1, 21), color="#4C72B0", edgecolor="white")
    axes[0].set_xlabel("Top-3 入选频率（重复分层交叉验证）")
    axes[0].set_ylabel("化合物-维度组合数")
    axes[0].set_title("Top-3 稳定性分布")
    axes[1].bar(["全部 3 个稳定", "其余"], [stable_all, len(Y.columns) - stable_all],
                color=["#55A868", "#C44E52"])
    axes[1].set_ylabel("感官维度数")
    axes[1].set_title("Top-3 全部稳定的维度数（频率 ≥0.8）")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_q2_stability.png"), dpi=200)
    plt.close(fig)

    # 2) 跨方法 Jaccard 热图
    methods = ["partial", "ridge", "rf"]
    mat = np.zeros((len(methods), len(methods)))
    for a, ma in enumerate(methods):
        for b, mb in enumerate(methods):
            vals = [jaccard(tops_by_method[ma][j], tops_by_method[mb][j]) for j in range(len(Y.columns))]
            mat[a, b] = np.nanmean(vals)
    fig, ax = plt.subplots(figsize=(5.8, 5))
    im = ax.imshow(mat, cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(["偏相关", "Ridge", "随机森林"])
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(["偏相关", "Ridge", "随机森林"])
    for i in range(len(methods)):
        for j in range(len(methods)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=11)
    ax.set_title("不同方法 Top-3 的平均 Jaccard 系数")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_q2_method_jaccard.png"), dpi=200)
    plt.close(fig)

    # 3) 共表达图模块与高频 Top-3 化合物
    try:
        pos = nx.spring_layout(G, seed=RANDOM_SEED, k=0.25, iterations=80)
        top_compounds = set()
        for j in range(len(Y.columns)):
            top_compounds.update(top_full[j])
        fig, ax = plt.subplots(figsize=(11, 9))
        edge_list = list(G.edges(data=True))
        weights = [e[2].get("weight", 0.5) for e in edge_list]
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.08, width=[0.3 + w for w in weights])
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

    # 4) 官方交叉相关与自算偏相关的一致性（验证外部证据可用性）
    common = official.stack().dropna()
    pairs = []
    for (c, s), v in common.items():
        j = list(Y.columns).index(s)
        i = chem_cols.index(c)
        pairs.append(dict(official=v, partial=ev_full["partial"][i, j],
                          raw_partial_corr=0.0))
    if pairs:
        pdf = pd.DataFrame(pairs)
        # 用偏相关原始值重算展示（上面 percentile 仅用于排序，这里重新算原值）
        C_open = partial_corr_style(X_imp, Y.values, styles)
        vals = []
        for (c, s), v in common.items():
            vals.append(C_open[chem_cols.index(c), list(Y.columns).index(s)])
        pdf["raw_partial_corr"] = vals
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(pdf["official"], pdf["raw_partial_corr"], s=8, alpha=0.4, color="#4C72B0")
        ax.axhline(0, color="gray", lw=0.8)
        ax.axvline(0, color="gray", lw=0.8)
        ax.set_xlabel("官方 6.csv 交叉相关")
        ax.set_ylabel("本文偏相关（含风格控制）")
        ax.set_title("官方交叉相关与风格控制偏相关的一致性")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "fig_q2_consistency.png"), dpi=200)
        plt.close(fig)
        summary["consistency"] = {
            "n_pairs": int(len(pdf)),
            "pearson_official_vs_partial": float(np.corrcoef(pdf["official"], pdf["raw_partial_corr"])[0, 1]),
        }
        save_json(summary, os.path.join(OUT, "04_q2_证据汇总.json"))

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
