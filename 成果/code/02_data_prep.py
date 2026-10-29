# -*- coding: utf-8 -*-
"""02_data_prep.py — 阶段 2 数据视图元信息（可在建模前先跑，作为数据审计的延续）。

职责：核对 1.csv/4.csv 行序、统计特征/目标/风格、输出逐列数值变换分类与缺失模式。
共享函数在 data_prep_common.py；本脚本只负责生成 02_数据视图_元信息.json。

运行：
    $env:PYTHONPATH = "D:\\github mama\\.work\\pylibs"
    D:\\miniconda\\python.exe 成果\\code\\02_data_prep.py
"""
import os
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_prep_common import load_data, classify_transforms, save_json, RANDOM_SEED, OUT


def main():
    X, Y, styles, ids = load_data()
    transforms = classify_transforms()
    counts = pd.Series(transforms).value_counts().to_dict()
    meta = {
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "n_targets": int(Y.shape[1]),
        "n_styles": int(styles.nunique()),
        "feature_columns": list(X.columns),
        "target_columns": list(Y.columns),
        "transforms": transforms,
        "transform_counts": counts,
        "missing_cells": int(X.isna().sum().sum()),
        "missing_columns": {k: int(v) for k, v in X.isna().sum().items() if v > 0},
        "target_summary": {
            "mean_abs_max": float(Y.mean().abs().max()),
            "std_min": float(Y.std(ddof=1).min()),
            "std_max": float(Y.std(ddof=1).max()),
            "min": float(Y.min().min()),
            "max": float(Y.max().max()),
        },
        "style_counts": {k: int(v) for k, v in styles.value_counts().items()},
        "env": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__},
        "random_seed": RANDOM_SEED,
    }
    save_json(meta, os.path.join(OUT, "02_数据视图_元信息.json"))
    print("saved 02_数据视图_元信息.json")
    print(json.dumps({k: meta[k] for k in ["n_samples", "n_features", "n_targets", "n_styles",
                                           "transform_counts", "missing_cells"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
