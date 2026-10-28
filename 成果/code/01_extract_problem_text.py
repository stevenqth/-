# -*- coding: utf-8 -*-
"""01_extract_problem_text.py — 从官方题面 PDF 提取纯文本（无需任何第三方 PDF 库）。

背景：本机无 PDF 解析库、Word 转换失败、外网不可用；题面 PDF 由 WPS 生成，
字体为 Type0/Identity-H，带 ToUnicode CMap。本脚本解析 PDF 对象、解压内容流、
按 ToUnicode 映射还原文字并按坐标重组行序。

运行：
    D:\\miniconda\\python.exe 成果\\code\\01_extract_problem_text.py
输出：
    成果\\00_材料清点\\题面文本.txt
"""
import os
import re
import zlib

ROOT = r"D:\github mama"
PDF = os.path.join(ROOT, "资料", "2026年MathorCup大数据挑战赛-赛道A初赛.pdf")
OUT = os.path.join(ROOT, "成果", "00_材料清点", "题面文本.txt")

data = open(PDF, "rb").read()

# ---------- PDF 对象表 ----------
objs = {}
for m in re.finditer(rb"(\d+)\s+(\d+)\s+obj\b", data):
    num = int(m.group(1))
    start = m.end()
    end = data.find(b"endobj", start)
    body = data[start:end]
    sm = re.search(rb"stream\r?\n", body)
    stream = None
    if sm:
        sstart = start + sm.end()
        eend = data.find(b"endstream", sstart)
        stream = data[sstart:eend]
        if stream.endswith(b"\r\n"):
            stream = stream[:-2]
        elif stream.endswith(b"\n"):
            stream = stream[:-1]
    objs[num] = stream


def stream_of(n):
    return zlib.decompress(objs[n])


# ---------- ToUnicode CMap ----------
def parse_cmap(raw):
    txt = raw.decode("latin-1")
    mapping = {}
    for block in re.findall(r"beginbfchar(.*?)endbfchar", txt, re.S):
        for a, b in re.findall(r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>", block):
            mapping[int(a, 16)] = "".join(chr(int(b[i:i + 4], 16)) for i in range(0, len(b), 4))
    for block in re.findall(r"beginbfrange(.*?)endbfrange", txt, re.S):
        for a, b, c in re.findall(r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>", block):
            lo, hi, base = int(a, 16), int(b, 16), int(c, 16)
            for k in range(lo, hi + 1):
                mapping[k] = chr(base + (k - lo))
    return mapping


cmaps = {8: parse_cmap(stream_of(9)), 14: parse_cmap(stream_of(15)), 19: parse_cmap(stream_of(20))}
fmap = {"FT8": 8, "FT14": 14, "FT19": 19}

tm_re = re.compile(r"([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+Tm")
td_re = re.compile(r"([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+TD")
tf_re = re.compile(r"/(FT\d+)\s+([-+]?\d*\.?\d+)\s+Tf")
tj_hex = re.compile(r"<([0-9A-Fa-f\s]+)>\s*Tj")
tj_lit = re.compile(r"\(((?:\\.|[^\\()])*)\)\s*Tj")


def decode_hex(h, fnum):
    h = re.sub(r"\s+", "", h)
    cm = cmaps[fnum]
    return "".join(cm.get(int(h[i:i + 4], 16), "\uFFFD") for i in range(0, len(h) - 1, 4))


def extract_stream2(num):
    txt = stream_of(num).decode("latin-1")
    glyphs, font = [], None
    a, d, fcur, xcur = 0.05, -0.05, 0.0, 0.0
    for ln in txt.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        mtf = tf_re.search(ln)
        if mtf:
            font = fmap[mtf.group(1)]
        mtm = tm_re.search(ln)
        if mtm:
            a, d = float(mtm.group(1)), float(mtm.group(4))
            xcur, fcur = float(mtm.group(5)), float(mtm.group(6))
        mtd = td_re.search(ln)
        if mtd:
            xcur += float(mtd.group(1)) * a
            fcur += float(mtd.group(2)) * d
        for m in tj_hex.finditer(ln):
            for ch in decode_hex(m.group(1), font):
                glyphs.append((round(xcur, 2), round(841.9 - fcur, 2), ch))
        for m in tj_lit.finditer(ln):
            s = m.group(1).replace("\\n", "\n").replace("\\(", "(").replace("\\)", ")").replace("\\\\", "\\")
            for ch in s:
                glyphs.append((round(xcur, 2), round(841.9 - fcur, 2), ch))
    return glyphs


def render_page(num):
    glyphs = extract_stream2(num)
    glyphs.sort(key=lambda g: (-g[1], g[0]))
    groups = []
    for g in glyphs:
        for grp in groups:
            if abs(grp[0] - g[1]) <= 1.0:
                grp[1].append(g)
                break
        else:
            groups.append([g[1], [g]])
    groups.sort(key=lambda grp: -grp[0])
    lines = []
    for y, items in groups:
        items.sort(key=lambda g: g[0])
        parts, prev_x = [], None
        for g in items:
            if prev_x is not None and g[0] - prev_x > 6.0:
                parts.append("  ")
            parts.append(g[2])
            prev_x = g[0]
        lines.append("".join(parts).strip())
    return lines


pages = [(7,), (25,), (27,)]
all_lines = []
for i, (snum,) in enumerate(pages, 1):
    all_lines.append(f"===== 第 {i} 页 =====")
    all_lines.extend(render_page(snum))
    all_lines.append("")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(all_lines).strip() + "\n")
print(f"written: {OUT}  lines={len(all_lines)}")
