# -*- coding: utf-8 -*-
"""00_data_audit.py — 赛道A 附件数据审计（材料清点 / 变量映射 / 变换底数 / 缺失模式 / 相关矩阵复现）。

运行方式（本机无外网，使用工作区内复制的纯 Python openpyxl）：
    set PYTHONPATH=D:\\github mama\\.work\\pylibs
    D:\\miniconda\\python.exe 成果\\code\\00_data_audit.py

输出（成果/results/）：
    00_材料清单.json            文件清单、大小、SHA256
    00_数据字典_字段映射.csv    2/5/6.csv 变量名 -> 1/4.csv 字段名 的映射与依据
    00_对数底数验证.csv         1.csv 与 8.xlsx 校准浓度逐化合物对比（log10 vs ln）
    00_相关矩阵复现对比.json    2/5/6.csv 相关矩阵复现误差
    00_感官维度映射.csv         3.xlsx 品评卡维度 -> 4.csv 字段
    00_审计摘要.json            全部关键结论（供文档引用）
"""
import os
import re
import json
import hashlib
import itertools
from collections import Counter

import numpy as np
import pandas as pd

ROOT = r"D:\github mama"
MAT = os.path.join(ROOT, "资料", "01_赛题原始材料", "附件")
PDF = os.path.join(ROOT, "资料", "01_赛题原始材料", "2026年MathorCup大数据挑战赛-赛道A初赛.pdf")
REF = os.path.join(ROOT, "资料", "02_参考文献", "s41467-024-46346-0.pdf")
OUT = os.path.join(ROOT, "成果", "results")
os.makedirs(OUT, exist_ok=True)

META = ["beer", "beer_id", "tasting_category_fine"]
SUMMARY = {}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm(s):
    s = str(s).lower()
    return re.sub(r"[^a-z0-9]+", "", s)


def fuzzy_map(label, candidate_keys, cutoff=0.85):
    """candidate_keys: iterable of normalized keys; returns the best normalized key or None."""
    import difflib
    cand = difflib.get_close_matches(norm(label), list(candidate_keys), n=1, cutoff=cutoff)
    return cand[0] if cand else None


# ---------------------------------------------------------------- 0. 文件清单
files = []
for name in ["2026年MathorCup大数据挑战赛-赛道A初赛.pdf"]:
    p = os.path.join(os.path.dirname(PDF), name)
    files.append(dict(path=os.path.relpath(p, ROOT), size=os.path.getsize(p), sha256=sha256(p)))
for name in sorted(os.listdir(MAT)):
    p = os.path.join(MAT, name)
    if os.path.isfile(p):
        files.append(dict(path=os.path.relpath(p, ROOT), size=os.path.getsize(p), sha256=sha256(p)))
files.append(dict(path=os.path.relpath(REF, ROOT), size=os.path.getsize(REF), sha256=sha256(REF)))
with open(os.path.join(OUT, "00_材料清单.json"), "w", encoding="utf-8") as f:
    json.dump(files, f, ensure_ascii=False, indent=2)
SUMMARY["files"] = files

# ---------------------------------------------------------------- 1. 读取
df1 = pd.read_csv(os.path.join(MAT, "1.csv"))
df2 = pd.read_csv(os.path.join(MAT, "2.csv"), index_col=0)
df4 = pd.read_csv(os.path.join(MAT, "4.csv"))
df5 = pd.read_csv(os.path.join(MAT, "5.csv"), index_col=0)
df6 = pd.read_csv(os.path.join(MAT, "6.csv"), index_col=0)
df7 = pd.read_csv(os.path.join(MAT, "7.csv"))
x8 = pd.ExcelFile(os.path.join(MAT, "8.xlsx"))
names = x8.parse("Sensory names")
thr = x8.parse("Sensory thresholds")
cal = x8.parse("Calibrated chemical dataset")
gbm = x8.parse("GBM importance MDI")
prices = x8.parse("Beer Prices")

chem1 = [c for c in df1.columns if c not in META]
sens4 = [c for c in df4.columns if c not in META]

SUMMARY["shapes"] = {
    "1.csv": list(df1.shape), "2.csv": list(df2.shape), "3.xlsx": None,
    "4.csv": list(df4.shape), "5.csv": list(df5.shape), "6.csv": list(df6.shape),
    "7.csv": list(df7.shape),
    "8.xlsx": {sn: list(x8.parse(sn).shape) for sn in x8.sheet_names},
}

# ---------------------------------------------------------------- 2. 基础质量
na1 = df1[chem1].isna().sum()
SUMMARY["1_csv"] = {
    "n_rows": int(df1.shape[0]),
    "n_unique_beer": int(df1["beer"].nunique()),
    "n_unique_beer_id": int(df1["beer_id"].nunique()),
    "n_chem_cols": len(chem1),
    "na_total_chem": int(na1.sum()),
    "na_cols": {k: int(v) for k, v in na1[na1 > 0].items()},
    "categories": {k: int(v) for k, v in df1["tasting_category_fine"].value_counts().items()},
    "chem_min": float(df1[chem1].min().min()),
    "chem_max": float(df1[chem1].max().max()),
}
floor = df1[chem1].min()
SUMMARY["1_csv"]["floor_value_counts"] = {k: int(v) for k, v in floor.round(6).value_counts().head(10).items()}

na7 = df7.isna().sum()
SUMMARY["7_csv"] = {
    "na_total": int(na7.sum()),
    "na_cols": {k: int(v) for k, v in na7[na7 > 0].items()},
    "n_descriptor_cols": int(len([c for c in df7.columns if c not in META])),
}

# 4.csv 标准化核查
mu = df4[sens4].mean()
sd = df4[sens4].std()
SUMMARY["4_csv"] = {
    "n_rows": int(df4.shape[0]),
    "n_sens_cols": len(sens4),
    "A_cols": [c for c in sens4 if c.startswith("A_")],
    "F_cols": [c for c in sens4 if c.startswith("F_")],
    "other_cols": [c for c in sens4 if not c.startswith(("A_", "F_"))],
    "col_mean_abs_max": float(mu.abs().max()),
    "col_std_min": float(sd.min()),
    "col_std_max": float(sd.max()),
    "min": float(df4[sens4].min().min()),
    "max": float(df4[sens4].max().max()),
}

# ---------------------------------------------------------------- 3. 数值变换核验（1.csv vs 8.xlsx 校准浓度）
common = [c for c in chem1 if c in cal.columns]
merged = df1[["beer_id"] + common].merge(cal[["beer_id"] + common], on="beer_id", suffixes=("_raw", "_cal"))
rows = []
for c in common:
    v = pd.to_numeric(merged[f"{c}_raw"], errors="coerce")
    cval = pd.to_numeric(merged[f"{c}_cal"], errors="coerce")
    pos = cval > 0
    mad_log = float(np.nanmedian(np.abs(v[pos] - np.log10(cval[pos])))) if pos.sum() >= 5 else np.nan
    mad_raw = float(np.nanmedian(np.abs(v[pos] - cval[pos]))) if pos.sum() >= 5 else np.nan
    if pd.notna(mad_log) and mad_log < 1e-6:
        transform = "log10"
    elif pd.notna(mad_raw) and mad_raw < 1e-6:
        transform = "raw"
    else:
        transform = "unclear"
    zero = ~pos
    zero_vals = v[zero].dropna()
    rows.append(dict(
        compound=c,
        n_positive=int(pos.sum()),
        mad_log10=mad_log,
        mad_raw=mad_raw,
        transform=transform,
        zero_n=int(zero.sum()),
        zero_value_mode=float(zero_vals.mode().iloc[0]) if len(zero_vals) else np.nan,
        zero_unique=int(zero_vals.nunique()),
        floor_share=float((zero_vals == zero_vals.mode().iloc[0]).mean()) if len(zero_vals) else np.nan,
    ))
logdf = pd.DataFrame(rows)
logdf.to_csv(os.path.join(OUT, "00_数值变换核验.csv"), index=False, encoding="utf-8-sig")
SUMMARY["value_transform"] = {
    "n_common_columns": len(common),
    "n_log10": int((logdf["transform"] == "log10").sum()),
    "n_raw": int((logdf["transform"] == "raw").sum()),
    "n_unclear": int((logdf["transform"] == "unclear").sum()),
    "raw_columns": logdf[logdf["transform"] == "raw"]["compound"].tolist(),
    "unclear_columns": {
        r["compound"]: {"mad_log10": r["mad_log10"], "mad_raw": r["mad_raw"]}
        for _, r in logdf[logdf["transform"] == "unclear"].iterrows()
    },
    "log10_median_mad": float(logdf[logdf["transform"] == "log10"]["mad_log10"].median()),
    "zero_replacement": {
        "n_cols_with_zeros": int((logdf["zero_n"] > 0).sum()),
        "n_cols_constant_floor": int(((logdf["zero_n"] > 0) & (logdf["zero_unique"] == 1)).sum()),
        "example_floor_values": logdf[logdf["zero_n"] > 0][["compound", "zero_n", "zero_value_mode", "zero_unique"]].head(10).values.tolist(),
    },
    "conclusion": "70 个可与 8.xlsx 校准浓度对照的列中，37 列为 log10 变换、33 列为原始浓度（逐列精确核验）；其余 161 列无校准对照，需按“负值仅可能来自 log10 列”等规则推断并标注为假设。题面称 ethanol/color/bitterness/pH/CO2 保留原始量纲，但实测这些列在 1.csv 中为 log10 值，题面表述与文件不一致。",
}

# ---------------------------------------------------------------- 4. 变量映射
# 4-1 2.csv 化合物标签 -> 1.csv 列名
bridge = {}
for _, row in thr.iterrows():
    keys = [row.get("field_name_python"), row.get("chemical_name"), row.get("chemical_name_short")]
    keys = [norm(k) for k in keys if isinstance(k, str)]
    for k in keys:
        bridge.setdefault(k, set()).update(keys)

ALIAS = {
    "ammonia": "ammonia.mg.L.",
    "diacetyl": "X2.3.butanedione",
    "cis3hexenol": "X3Z.hexenol",
    "furfurylalcohol": "X2.furanmethanol",
    "methionol": "X3.methylthio.propanol",
    "ebetadamascenone": "X.E..beta.damascenone",
    "ealphaionone": "X.E..alpha.ionone",
    "zbetafarnesene": "X.Z..beta.farnesene",
    "ebetafarnesene": "X.E..beta.farnesene",
    "ebetaionone": "X.E..beta.ionone",
    "epicubenol": "X1.10.di.epi.cubenol",
}
norm1 = {norm(c): c for c in chem1}

# 希腊字母在归一化时会被删除，需按原始标签精确映射
MANUAL_BY_LABEL = {
    "E-β-damascenone": "X.E..beta.damascenone",
    "E-α-ionone": "X.E..alpha.ionone",
    "Z-β-farnesene": "X.Z..beta.farnesene",
    "E-β-farnesene": "X.E..beta.farnesene",
    "E-β-ionone": "X.E..beta.ionone",
}


def map_chem(label):
    n = norm(label)
    if label in MANUAL_BY_LABEL and MANUAL_BY_LABEL[label] in chem1:
        return MANUAL_BY_LABEL[label], "manual"
    if n in norm1:
        return norm1[n], "exact"
    if n in ALIAS and ALIAS[n] in chem1:
        return ALIAS[n], "manual"
    for k in bridge.get(n, set()):
        if k in norm1:
            return norm1[k], "bridge"
    hit = fuzzy_map(label, norm1)
    if hit:
        return norm1[hit], "fuzzy"
    return None, "unmatched"


map2 = []
for lab in df2.index:
    t, how = map_chem(lab)
    map2.append(dict(source="2.csv", label=lab, target=t, method=how))
map6 = []
for lab in df6.index:
    t, how = map_chem(lab)
    map6.append(dict(source="6.csv", label=lab, target=t, method=how))

# 4-2 5.csv 感官标签 -> 4.csv 列名
panel = names[names["dataset"] == "Trained panel"]
pfb = {}
for _, row in panel.iterrows():
    r = row["R_name"]
    for col in ["CleanName_PanelFeedback", "CleanName_Paper", "CleanName_PaperLong"]:
        val = row.get(col)
        if isinstance(val, str):
            pfb.setdefault(norm(val), r)
norm4 = {norm(c): c for c in sens4}
SENS_ALIAS = {"overallappreciationtrainedpanel": "overall"}
map5 = []
for lab in df5.index:
    t = pfb.get(norm(lab))
    how = "names_sheet" if t else ""
    if t is None and norm(lab) in SENS_ALIAS:
        t, how = SENS_ALIAS[norm(lab)], "manual"
    if t is None:
        cand = fuzzy_map(lab, norm4)
        if cand:
            t, how = norm4[cand], "fuzzy"
    map5.append(dict(source="5.csv", label=lab, target=t, method=how if t else "unmatched"))
map6c = []
for lab in df6.columns:
    t = pfb.get(norm(lab))
    how = "names_sheet" if t else ""
    if t is None and norm(lab) in SENS_ALIAS:
        t, how = SENS_ALIAS[norm(lab)], "manual"
    if t is None:
        cand = fuzzy_map(lab, norm4)
        if cand:
            t, how = norm4[cand], "fuzzy"
    map6c.append(dict(source="6.csv", label=lab, target=t, method=how if t else "unmatched"))

mapping = pd.DataFrame(map2 + map5 + map6 + map6c)
mapping.to_csv(os.path.join(OUT, "00_数据字典_字段映射.csv"), index=False, encoding="utf-8-sig")
SUMMARY["mapping"] = {
    s: {
        "n": int((mapping["source"] == s).sum()),
        "matched": int(((mapping["source"] == s) & mapping["target"].notna()).sum()),
        "unmatched": list(mapping[(mapping["source"] == s) & mapping["target"].isna()]["label"]),
    }
    for s in ["2.csv", "5.csv", "6.csv"]
}

# ---------------------------------------------------------------- 5. 相关矩阵复现
def compare_matrix(orig, recomputed, pairs, name):
    """orig/recomputed: square/rect DataFrame indexed by labels; pairs: list of (row_label, col_label)."""
    diffs = []
    for (r, c) in pairs:
        a = orig.loc[r, c]
        b = recomputed.loc[r, c]
        if pd.notna(a) and pd.notna(b):
            diffs.append(abs(float(a) - float(b)))
    diffs = np.array(diffs)
    return {
        "matrix": name,
        "n_pairs": int(len(diffs)),
        "mean_abs_diff": float(np.mean(diffs)) if len(diffs) else None,
        "median_abs_diff": float(np.median(diffs)) if len(diffs) else None,
        "max_abs_diff": float(np.max(diffs)) if len(diffs) else None,
        "frac_lt_0p01": float(np.mean(diffs < 0.01)) if len(diffs) else None,
    }


corr_out = {}
# 2.csv: 231x231 化合物相关性
pairs2 = [(l, l2) for l in df2.index for l2 in df2.columns if l < l2]
targets2 = [m["target"] for m in map2 if m["target"]]
labels2 = [m["label"] for m in map2 if m["target"]]
d1 = df1[targets2].copy()
d1.columns = labels2
spec2 = d1.corr(method="spearman").reindex(index=df2.index, columns=df2.columns)
pear2 = d1.corr(method="pearson").reindex(index=df2.index, columns=df2.columns)
corr_out["2.csv_spearman"] = compare_matrix(df2, spec2, pairs2, "2.csv")
corr_out["2.csv_pearson"] = compare_matrix(df2, pear2, pairs2, "2.csv")

# 5.csv: 50x50 感官相关性
p5 = [(l1, l2) for l1 in df5.index for l2 in df5.columns if l1 < l2]
t5 = [m["target"] for m in map5 if m["target"]]
l5 = [m["label"] for m in map5 if m["target"]]
dsens = df4[t5].copy()
dsens.columns = l5
spec5 = dsens.corr(method="spearman").reindex(index=df5.index, columns=df5.columns)
pear5 = dsens.corr(method="pearson").reindex(index=df5.index, columns=df5.columns)
corr_out["5.csv_spearman"] = compare_matrix(df5, spec5, p5, "5.csv")
corr_out["5.csv_pearson"] = compare_matrix(df5, pear5, p5, "5.csv")

# 6.csv: 44 x 50 化学-感官交叉相关
chem6 = [(m["label"], m["target"]) for m in map6 if m["target"]]
sens6 = [(m["label"], m["target"]) for m in map6c if m["target"]]
d6c = df1[[t for _, t in chem6]].copy()
d6c.columns = [l for l, _ in chem6]
d6s = df4[[t for _, t in sens6]].copy()
d6s.columns = [l for l, _ in sens6]
pairs6 = [(r, c) for r in d6c.columns for c in d6s.columns]
# 复现 6.csv 中实际支持的对（行标签和列标签都在映射内）
pairs6 = [(r, c) for r in df6.index if r in d6c.columns for c in df6.columns if c in d6s.columns]
spec6 = pd.DataFrame(index=d6c.columns, columns=d6s.columns, dtype=float)
pear6 = pd.DataFrame(index=d6c.columns, columns=d6s.columns, dtype=float)
for r in d6c.columns:
    for c in d6s.columns:
        x, y = d6c[r], d6s[c]
        ok = x.notna() & y.notna()
        if ok.sum() > 3:
            spec6.loc[r, c] = x[ok].corr(y[ok], method="spearman")
            pear6.loc[r, c] = x[ok].corr(y[ok], method="pearson")
corr_out["6.csv_spearman"] = compare_matrix(df6, spec6, pairs6, "6.csv")
corr_out["6.csv_pearson"] = compare_matrix(df6, pear6, pairs6, "6.csv")

# 全量 231x50 交叉相关（2.csv 之外、6.csv 的 44 行之外）
all6 = pd.DataFrame(index=[norm(c) for c in chem1], columns=[norm(c) for c in sens4], dtype=float)
SUMMARY["full_cross_corr_shape"] = list(all6.shape)
with open(os.path.join(OUT, "00_相关矩阵复现对比.json"), "w", encoding="utf-8") as f:
    json.dump(corr_out, f, ensure_ascii=False, indent=2)
SUMMARY["corr_replication"] = corr_out

# ---------------------------------------------------------------- 6. 3.xlsx 品评卡维度 -> 4.csv
FORM = {
    "Aroma": {
        "Malt": [("Cereals, flour, grains", "A_malt_grain"), ("Bread crust", "A_malt_bread"),
                 ("Caramel", "A_malt_cara"), ("Smoked, burnt toast", "A_malt_burn"), ("Malt (overall)", "A_malt_all")],
        "Hops": [("Citrus", "A_hops_citrus"), ("Tropical fruit", "A_hops_tropical"),
                 ("Noble hops (dried herbs, dried grass, hay)", "A_hops_noble"),
                 ("Pine, woody, resinous", "A_hops_woody"), ("Hops (overall)", "A_hops_all")],
        "Esters": [("Ethyl acetate, solvent, nail polish", "A_esters_ethac"),
                   ("Banana, pear, candy", "A_esters_isoaa"), ("Flowers, roses", "A_esters_flower"),
                   ("Sweet, fruity, apples", "A_esters_fruity"), ("Esters (overall)", "A_esters_all")],
        "Others": [("Acetaldehyde (green apples, peel)", None), ("Body", "body"), ("Bitterness", "bitternes"),
                   ("Carbonation", "co2"), ("Aftertaste", "aftertaste"),
                   ("DMS (cooked cabbage, corn)", "dms"),
                   ("Trans-2-nonenal (wet cardboard, old attic, papery)", "t2n"),
                   ("Metallic", "metallic"), ("Diacetyl (buttery, caramel/butterscotch)", "diacetyl"),
                   ("4VG (smoky, clove, medicinal, dentist)", "X4vg"),
                   ("Stale hops (cheesy, roquefort)", "stale_hops"),
                   ("Orange", "orange"), ("Coriander", "coriander"), ("Clove/anise", "clove"),
                   ("Overall", "overall")],
    },
    "Flavour": {
        "Malt": [("Cereals, flour, grains", "F_malt_grain"), ("Bread crust", "F_malt_bread"),
                 ("Caramel", "F_malt_cara"), ("Smoked, burnt toast", "F_malt_burn"), ("Malt (overall)", "F_malt_all")],
        "Hops": [("Citrus", "F_hops_citrus"), ("Tropical fruit", "F_hops_tropical"),
                 ("Noble hops (dried herbs, dried grass, hay)", "F_hops_noble"),
                 ("Pine, woody, resinous", "F_hops_woody"), ("Hops (overall)", "F_hops_all")],
        "Esters": [("Ethyl acetate, solvent, nail polish", "F_esters_ethac"),
                   ("Banana, pear, candy", "F_esters_isoaa"), ("Flowers, roses", "F_esters_flower"),
                   ("Sweet, fruity, apples", "F_esters_fruity"), ("Esters (overall)", "F_esters_all")],
        "Others": [("Acetaldehyde (green apples, peel)", None), ("Alcohol", "alcohol"), ("Sweetness", "sweetness"),
                   ("Acidity/sourness", "acidity"),
                   ("DMS (cooked cabbage, corn)", "dms"),
                   ("Trans-2-nonenal (wet cardboard, old attic, papery)", "t2n"),
                   ("Metallic", "metallic"), ("Diacetyl (buttery, caramel/butterscotch)", "diacetyl"),
                   ("4VG (smoky, clove, medicinal, dentist)", "X4vg"),
                   ("Stale hops (cheesy, roquefort)", "stale_hops"),
                   ("Lactic acid", "lactic"), ("Acetic acid", "acetic"),
                   ("Barnyard/horsey", "barnyard")],
    },
}
form_rows = []
for side, groups in FORM.items():
    for g, items in groups.items():
        for label, target in items:
            form_rows.append(dict(side=side, group=g, item=label, target=target,
                                  target_exists=(target in sens4) if target else False))
formdf = pd.DataFrame(form_rows)
formdf.to_csv(os.path.join(OUT, "00_感官维度映射.csv"), index=False, encoding="utf-8-sig")
SUMMARY["form_mapping"] = {
    "n_items": int(len(formdf)),
    "n_targets": int(formdf["target"].notna().sum()),
    "n_missing_target": int(formdf["target"].isna().sum()),
    "missing_items": formdf[formdf["target"].isna()]["item"].tolist(),
    "n_unique_targets": int(formdf["target"].dropna().nunique()),
    "targets_not_in_form": [c for c in sens4 if c not in set(formdf["target"].dropna())],
}

# ---------------------------------------------------------------- 7. 8.xlsx 关键参考表
defect_keys = ["dimethyl sulfide", "diacetyl", "4-vinylguaiacol", "2-nonenal", "stale"]
thr_rows = thr.to_dict(orient="records")
defect_rows = []
for r in thr_rows:
    txt = " ".join(str(r.get(k)) for k in ["chemical_name", "chemical_name_short", "off_flavors", "field_name_python"]).lower()
    if any(k in txt for k in defect_keys):
        defect_rows.append({k: (None if pd.isna(v) else v) for k, v in r.items() if k in
                            ["field_name_python", "chemical_name_short", "unit", "threshold_mean", "threshold_median", "off_flavors"]})
SUMMARY["8_xlsx"] = {
    "sensory_thresholds": defect_rows,
    "gbm_mdi_top10": gbm.head(10).values.tolist(),
    "calibrated_cols": int(cal.shape[1] - 3),
    "prices_cols": list(prices.columns),
    "n_sensory_threshold_rows": int(thr.shape[0]),
    "n_sensory_names": int(names.shape[0]),
}

# 8-1 缺陷化合物“超阈值”基线统计（Q3 约束的现实基线）
DEFECT = {
    "DMS": dict(short="DMS", unit="ug/L", thr_mean=17.25, thr_median=19.0, cal_col="DMS"),
    "diacetyl": dict(short="Diacetyl", unit="mg/L", thr_mean=0.29925, thr_median=0.09, cal_col="X2.3.butanedione"),
    "4VG": dict(short="4-vinylguaiacol", unit="mg/L", thr_mean=0.206333333, thr_median=0.2, cal_col="X4VG"),
}
defect_stats = {}
for k, d in DEFECT.items():
    if d["cal_col"] not in cal.columns:
        continue
    c = pd.to_numeric(cal[d["cal_col"]], errors="coerce")
    defect_stats[k] = {
        "unit": d["unit"],
        "threshold_mean": d["thr_mean"],
        "threshold_median": d["thr_median"],
        "n_positive": int((c > 0).sum()),
        "n_above_mean": int((c > d["thr_mean"]).sum()),
        "n_above_median": int((c > d["thr_median"]).sum()),
        "share_above_mean": float((c > d["thr_mean"]).mean()),
    }
SUMMARY["8_xlsx"]["defect_baseline"] = defect_stats
SUMMARY["8_xlsx"]["defect_threshold_gaps"] = {
    "trans-2-nonenal": "8.xlsx 阈值表中 (E)-2-nonenal 的 threshold_mean/median 为 NA（原论文未给统一阈值）",
    "stale hops": "阈值表中无 stale hops 对应的化学阈值，仅 3.xlsx/4.csv 有感官评分列",
}

# ---------------------------------------------------------------- 8. 写出摘要
with open(os.path.join(OUT, "00_审计摘要.json"), "w", encoding="utf-8") as f:
    json.dump(SUMMARY, f, ensure_ascii=False, indent=2, default=str)

print("=== audit done ===")
print(json.dumps({
    "shapes": SUMMARY["shapes"],
    "value_transform": SUMMARY["value_transform"],
    "corr": {k: {kk: vv for kk, vv in v.items() if kk != "matrix"} for k, v in corr_out.items()},
    "mapping": SUMMARY["mapping"],
    "form": SUMMARY["form_mapping"],
    "defect": SUMMARY["8_xlsx"].get("defect_baseline"),
    "na1": SUMMARY["1_csv"]["na_total_chem"],
    "na7": SUMMARY["7_csv"]["na_total"],
}, ensure_ascii=False, indent=2))
