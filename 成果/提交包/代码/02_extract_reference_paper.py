# -*- coding: utf-8 -*-
"""02_extract_reference_paper.py — 从参考论文 PDF（Springer/iText, PDF 1.4）提取全文文本。

论文：Predicting and improving complex beer flavor through machine learning
      Michiel Schreurs 等, Nature Communications (2024) 15:2368, doi:10.1038/s41467-024-46346-0
方法：自写 PDF 解析器（对象表 -> 页面树 -> 内容流 FlateDecode -> Type1 字体宽度与 WinAnsi 解码 -> 按坐标重组行）
用途：为参考论文拆解与引文核对提供可检索文本；不依赖任何第三方 PDF 库。
运行：
    D:\\miniconda\\python.exe 成果\\code\\02_extract_reference_paper.py
输出：
    资料\\参考论文_文本提取.txt
"""
import re, zlib, sys, os

ROOT = r"D:\github mama"
PDF = os.path.join(ROOT, "资料", "s41467-024-46346-0.pdf")
OUT = os.path.join(ROOT, "资料", "参考论文_文本提取.txt")

data = open(PDF, 'rb').read()
objs = {}
for m in re.finditer(rb'(\d+)\s+(\d+)\s+obj\b', data):
    num = int(m.group(1)); start = m.end(); end = data.find(b'endobj', start)
    body = data[start:end]
    sm = re.search(rb'stream\r?\n', body)
    stream = None
    if sm:
        sstart = start + sm.end(); eend = data.find(b'endstream', sstart)
        stream = data[sstart:eend]
        if stream.endswith(b'\r\n'): stream = stream[:-2]
        elif stream.endswith(b'\n'): stream = stream[:-1]
    objs[num] = (body, stream)

def get_stream(num):
    body, stream = objs[num]
    if stream is None:
        return b''
    filt = re.search(rb'/Filter\s*/(\w+)', body)
    if filt and filt.group(1) == b'FlateDecode':
        try:
            return zlib.decompress(stream)
        except Exception:
            try:
                return zlib.decompressobj().decompress(stream)
            except Exception:
                return b''
    return stream

# ---------- page tree ----------
catalog = next(n for n, (d, _) in objs.items() if b'/Type' in d and b'/Catalog' in d)
pages_root = int(re.search(rb'/Pages\s+(\d+)\s+\d+\s+R', objs[catalog][0]).group(1))
page_nums = []

def walk(node):
    d = objs[node][0]
    if re.search(rb'/Type\s*/Pages', d):
        kids = re.search(rb'/Kids\s*\[(.*?)\]', d, re.S)
        for k in re.findall(rb'(\d+)\s+\d+\s+R', kids.group(1)):
            walk(int(k))
    elif re.search(rb'/Type\s*/Page\b', d):
        page_nums.append(node)

walk(pages_root)

# ---------- CMap for fonts that have one ----------
def parse_cmap(raw):
    txt = raw.decode('latin-1')
    mp = {}
    for block in re.findall(r'beginbfchar(.*?)endbfchar', txt, re.S):
        for a, b in re.findall(r'<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>', block):
            mp[int(a, 16)] = ''.join(chr(int(b[i:i+4], 16)) for i in range(0, len(b), 4))
    for block in re.findall(r'beginbfrange(.*?)endbfrange', txt, re.S):
        for a, b, c in re.findall(r'<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>', block):
            lo, hi, base = int(a, 16), int(b, 16), int(c, 16)
            for k in range(lo, hi + 1):
                mp[k] = chr(base + (k - lo))
    return mp

font_cache = {}
def font_info(fnum):
    if fnum in font_cache:
        return font_cache[fnum]
    d = objs[fnum][0]
    sub = re.search(rb'/Subtype\s*/(\w+)', d)
    sub = sub.group(1) if sub else b''
    two_byte = sub in (b'Type0', b'CIDFontType2')
    first = re.search(rb'/FirstChar\s+(\d+)', d)
    first = int(first.group(1)) if first else 0
    widths = re.search(rb'/Widths\s*\[(.*?)\]', d, re.S)
    wlist = [float(x) for x in re.findall(rb'[-+]?\d*\.?\d+', widths.group(1))] if widths else []
    missing = re.search(rb'/MissingWidth\s+(\d+)', d)
    missing = float(missing.group(1)) if missing else 500.0
    cmap = {}
    tu = re.search(rb'/ToUnicode\s+(\d+)\s+\d+\s+R', d)
    if tu:
        cmap = parse_cmap(get_stream(int(tu.group(1))))
    info = {"two_byte": two_byte, "first": first, "widths": wlist, "missing": missing, "cmap": cmap}
    font_cache[fnum] = info
    return info

def char_width(info, code):
    if info["two_byte"]:
        return info["missing"]
    idx = code - info["first"]
    if 0 <= idx < len(info["widths"]):
        w = info["widths"][idx]
        return w if w > 0 else info["missing"]
    return info["missing"]

def decode_bytes(bs, info):
    out = []
    if info["two_byte"]:
        for i in range(0, len(bs) - 1, 2):
            code = (bs[i] << 8) | bs[i + 1]
            out.append(info["cmap"].get(code, '\ufffd'))
    else:
        for b in bs:
            if info["cmap"]:
                out.append(info["cmap"].get(b, chr(b)))
            else:
                out.append(chr(b))
    return ''.join(out)

def unescape(bs):
    def rep(m):
        k = m.group(1)
        return {b'n': b'\n', b'r': b'\r', b't': b'\t', b'b': b'\b', b'f': b'\f',
                b'(': b'(', b')': b')', b'\\': b'\\'}.get(k, k)
    return re.sub(rb'\\([nrtbf()\\])', rep, bs)

token_re = re.compile(
    rb'<[0-9A-Fa-f\s]*>|\((?:\\.|[^\\)])*\)|\[|\]|/[^\s/<>\[\]()]+|[-+]?\d*\.?\d+|'
    rb'BT|ET|Tf|Tm|Td|TD|T\*|TL|Tj|TJ|Tc|Tw|Tz|Ts|Tr|[A-Za-z\'"]+'
)

def parse_page(pnum):
    d = objs[pnum][0]
    res_ref = re.search(rb'/Resources\s+(\d+)\s+\d+\s+R', d)
    res = objs[int(res_ref.group(1))][0] if res_ref else d
    fonts = {}
    fm = re.search(rb'/Font\s*<<(.*?)>>', res, re.S)
    if fm:
        for name, num in re.findall(rb'/(\w+)\s+(\d+)\s+\d+\s+R', fm.group(1)):
            fonts[name.decode()] = int(num)
    content = b''
    cm = re.search(rb'/Contents\s+(\d+)\s+\d+\s+R', d)
    if cm:
        content = get_stream(int(cm.group(1)))
    else:
        arr = re.search(rb'/Contents\s*\[(.*?)\]', d, re.S)
        if arr:
            for r in re.findall(rb'(\d+)\s+\d+\s+R', arr.group(1)):
                content += get_stream(int(r)) + b'\n'

    toks = token_re.findall(content)
    runs = []  # dict(y, x, w_user, text, fontsize_user)
    cur_font = None
    fs_tf = 1.0
    tm = [1, 0, 0, 1, 0, 0]
    tlm = tm[:]
    tc = 0.0
    tw = 0.0
    th = 1.0
    tl = 0.0
    in_text = False

    def show(bs_hex=False, raw=None):
        nonlocal tm
        info = font_info(fonts[cur_font]) if cur_font in fonts else None
        if info is None:
            return
        if bs_hex:
            h = re.sub(rb'\s+', b'', raw[1:-1])
            if len(h) % 2:
                h += b'0'
            bs = bytes.fromhex(h.decode())
        else:
            bs = unescape(raw[1:-1])
        txt = decode_bytes(bs, info)
        scale = tm[0] if tm[0] != 0 else 1.0
        fs_user = fs_tf * abs(scale)
        # advance = sum(width/1000 * fs_tf + tc + tw) * scale  (approx, ignore Th)
        adv_text = 0.0
        if info["two_byte"]:
            codes = [(bs[i] << 8) | bs[i + 1] for i in range(0, len(bs) - 1, 2)]
        else:
            codes = list(bs)
        for code in codes:
            adv_text += (char_width(info, code) / 1000.0 * fs_tf + tc + tw)
        adv_user = adv_text * abs(scale)
        runs.append({"y": tm[5], "x": tm[4], "w": adv_user, "t": txt, "fs": fs_user})
        tm[4] += adv_text * tm[0]

    def move(tx, ty):
        nonlocal tm, tlm
        tlm = [tlm[0], tlm[1], tlm[2], tlm[3],
               tlm[4] + tx * tlm[0] + ty * tlm[2],
               tlm[5] + tx * tlm[1] + ty * tlm[3]]
        tm = tlm[:]

    i = 0
    while i < len(toks):
        t = toks[i]
        if t == b'BT':
            tm = [1, 0, 0, 1, 0, 0]; tlm = tm[:]; in_text = True
        elif t == b'ET':
            in_text = False
        elif t == b'Tf':
            if i >= 2 and toks[i - 2].startswith(b'/'):
                cur_font = toks[i - 2][1:].decode()
                try:
                    fs_tf = float(toks[i - 1])
                except Exception:
                    pass
        elif t == b'Tm':
            nums = [float(x) for x in toks[i - 6:i] if re.fullmatch(rb'[-+]?\d*\.?\d+', x)]
            if len(nums) == 6:
                tm = nums[:]; tlm = tm[:]
        elif t in (b'Td', b'TD'):
            nums = [float(x) for x in toks[i - 2:i] if re.fullmatch(rb'[-+]?\d*\.?\d+', x)]
            if len(nums) == 2:
                if t == b'TD':
                    tl = -nums[1]
                move(nums[0], nums[1])
        elif t == b'T*':
            move(0, -tl if tl else -12)
        elif t == b'TL':
            try:
                tl = float(toks[i - 1])
            except Exception:
                pass
        elif t == b'Tc':
            try:
                tc = float(toks[i - 1])
            except Exception:
                pass
        elif t == b'Tw':
            try:
                tw = float(toks[i - 1])
            except Exception:
                pass
        elif t == b'Tz':
            pass
        elif t == b'Tj':
            raw = toks[i - 1]
            if raw.startswith(b'<'):
                show(True, raw)
            elif raw.startswith(b'('):
                show(False, raw)
        elif t == b'TJ':
            j = i - 1
            depth = 0
            arr = []
            while j >= 0:
                x = toks[j]
                if x == b']':
                    depth += 1
                elif x == b'[':
                    depth -= 1
                    if depth == 0:
                        break
                if depth >= 0:
                    arr.append(x)
                j -= 1
            arr.reverse()
            for x in arr:
                if x.startswith(b'(') or x.startswith(b'<'):
                    if x.startswith(b'('):
                        show(False, x)
                    else:
                        show(True, x)
                elif re.fullmatch(rb'[-+]?\d*\.?\d+', x):
                    adj = float(x)
                    tm[4] -= adj / 1000.0 * fs_tf * tm[0]
                    # 词间空格常以较大负调整量表示（约 -200 ~ -600），小值为字距微调
                    if adj < -180:
                        runs.append({"y": tm[5], "x": tm[4], "w": 0.0, "t": " ", "fs": fs_tf * abs(tm[0])})
        i += 1

    # group runs into lines
    runs.sort(key=lambda r: (-r["y"], r["x"]))
    lines = []
    for r in runs:
        if not r["t"]:
            continue
        if lines and abs(lines[-1]["y"] - r["y"]) <= 2.5:
            lines[-1]["runs"].append(r)
        else:
            lines.append({"y": r["y"], "runs": [r]})
    out = []
    for ln in lines:
        items = sorted(ln["runs"], key=lambda r: r["x"])
        buf = ""
        prev_end = None
        for r in items:
            if prev_end is not None:
                gap = r["x"] - prev_end
                fs = max(r["fs"], 1e-6)
                if gap > 0.25 * fs:
                    if not buf.endswith(" "):
                        buf += " "
            buf += r["t"]
            prev_end = r["x"] + r["w"]
        out.append(re.sub(r" {2,}", " ", buf.strip()))
    return out

all_lines = []
for i, p in enumerate(page_nums, 1):
    all_lines.append(f"\n===== PAGE {i} =====")
    try:
        all_lines.extend(parse_page(p))
    except Exception as e:
        all_lines.append(f"[error: {e}]")
header = (
    "# 参考论文文本提取说明\n"
    "# 论文：Predicting and improving complex beer flavor through machine learning\n"
    "#      Michiel Schreurs 等, Nature Communications (2024) 15:2368, doi:10.1038/s41467-024-46346-0\n"
    "# 来源：资料/s41467-024-46346-0.pdf\n"
    "# 方法：自写 PDF 解析器（对象表 -> 页面树 -> 内容流 FlateDecode -> Type1 字体宽度与 WinAnsi 解码 -> 按坐标重组行）\n"
    "# 注意：部分连字/词间空格在原文中以字形间隙表示，提取可能缺失；精确引用请对照 PDF 原页。\n"
    "# ============================================================\n"
)
with open(OUT, 'w', encoding='utf-8') as f:
    f.write(header + '\n'.join(all_lines) + '\n')
print("pages", len(page_nums), "lines", len(all_lines), file=sys.stderr)
