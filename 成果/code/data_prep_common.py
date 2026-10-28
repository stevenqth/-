# -*- coding: utf-8 -*-
"""data_prep_common.py — 问题一/二/三共享的数据准备函数。

被 02_data_prep.py（元信息审计）与 03_q1_regression.py 等脚本导入。
"""
import os
import json

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold

ROOT = r"D:\github mama"
MAT = os.path.join(ROOT, "资料", "附件")
OUT = os.path.join(ROOT, "成果", "results")
FIG = os.path.join(ROOT, "成果", "figures")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

META = ["beer", "beer_id", "tasting_category_fine"]
RANDOM_SEED = 42


def load_data():
    """返回 (X 特征 DataFrame, Y 目标 DataFrame, styles Series, ids DataFrame)。"""
    df1 = pd.read_csv(os.path.join(MAT, "1.csv"))
    df4 = pd.read_csv(os.path.join(MAT, "4.csv"))
    if not (df1["beer"].values == df4["beer"].values).all():
        raise ValueError("1.csv 与 4.csv 的 beer 行序不一致")
    chem = [c for c in df1.columns if c not in META]
    sens = [c for c in df4.columns if c not in META]
    X = df1[chem].copy()
    Y = df4[sens].copy()
    styles = df1["tasting_category_fine"].copy()
    ids = df1[["beer", "beer_id", "tasting_category_fine"]].copy()
    return X, Y, styles, ids


def classify_transforms():
    """逐列判定 1.csv 的数值变换：log10 / raw（8.xlsx 校准浓度为金标准），其余按负值推断。"""
    df1 = pd.read_csv(os.path.join(MAT, "1.csv"))
    cal = pd.read_excel(os.path.join(MAT, "8.xlsx"), sheet_name="Calibrated chemical dataset")
    chem = [c for c in df1.columns if c not in META]
    common = [c for c in chem if c in cal.columns]
    merged = df1[["beer_id"] + common].merge(
        cal[["beer_id"] + common], on="beer_id", suffixes=("_raw", "_cal"))
    info = {}
    for c in chem:
        has_neg = bool((df1[c] < 0).any())
        if c in common:
            v = pd.to_numeric(merged[f"{c}_raw"], errors="coerce")
            cv = pd.to_numeric(merged[f"{c}_cal"], errors="coerce")
            pos = cv > 0
            mad_log = float(np.nanmedian(np.abs(v[pos] - np.log10(cv[pos])))) if pos.sum() >= 5 else np.inf
            mad_raw = float(np.nanmedian(np.abs(v[pos] - cv[pos]))) if pos.sum() >= 5 else np.inf
            if mad_log < 1e-6 and mad_log <= mad_raw:
                info[c] = "log10_verified"
            elif mad_raw < 1e-6:
                info[c] = "raw_verified"
            else:
                info[c] = "log10_inferred" if has_neg else "raw_inferred"
        else:
            info[c] = "log10_inferred" if has_neg else "raw_inferred"
    return info


def fit_imputer(X_train):
    """折内插补器：中位数 + 缺失指示。"""
    imp = SimpleImputer(strategy="median", add_indicator=True)
    imp.fit(X_train)
    return imp


def grade_mapping_fit(y_train, method="minmax"):
    """在训练数据上拟合“连续值 -> 0–6 整数等级”的单调映射。

    minmax   : 线性缩放到 [0,6] 后四舍五入（保留分布形状；主口径）
    quantile : 按经验分位数等频切成 7 级（敏感性口径）
    """
    y = np.asarray(y_train, dtype=float)
    if method == "minmax":
        mn = np.nanmin(y, axis=0)
        mx = np.nanmax(y, axis=0)
        span = np.where(mx > mn, mx - mn, 1.0)
        return {"method": "minmax", "min": mn, "span": span}
    if method == "quantile":
        sorted_cols = []
        for j in range(y.shape[1]):
            col = y[:, j]
            col = col[~np.isnan(col)]
            sorted_cols.append(np.sort(col))
        return {"method": "quantile", "sorted": sorted_cols}
    raise ValueError(method)


def grade_mapping_apply(y, params):
    """把连续值按映射参数转换为 0–6 整数等级。"""
    y = np.asarray(y, dtype=float)
    if params["method"] == "minmax":
        z = (y - params["min"]) / params["span"]
        g = np.round(np.clip(z, 0.0, 1.0) * 6.0)
    else:
        g = np.zeros_like(y, dtype=float)
        for j in range(y.shape[1]):
            s = params["sorted"][j]
            if len(s) == 0:
                continue
            pos = np.searchsorted(s, y[:, j], side="right")
            g[:, j] = np.floor(pos / len(s) * 7.0)
        g = np.clip(g, 0.0, 6.0)
    return g.astype(int)


def make_folds(styles, n_splits=5, seed=RANDOM_SEED):
    """按风格分层的 K 折，返回 [(train_idx, valid_idx), ...]。"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(tr, va) for tr, va in skf.split(np.zeros(len(styles)), styles.values)]


def save_json(obj, path):
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        return str(o)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=default)
