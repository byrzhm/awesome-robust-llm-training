#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""extract_figures.py — 从论文 PDF 提取图片 / 渲染页面，供阅读报告（图文并茂）使用。

用法（在仓库根目录运行；输出默认到 papers/<id>/figures/）：
    python3 scripts/extract_figures.py papers/<id>/paper.pdf --info
        # 概览：总页数、每页文本首行、图表标题所在页（Figure N / Table N）
    python3 scripts/extract_figures.py papers/<id>/paper.pdf --find "Figure 4"
        # 定位文本出现的页码与坐标（PDF 点）
    python3 scripts/extract_figures.py papers/<id>/paper.pdf --render 1,3-5
        # 整页渲染成 PNG（默认 150 DPI）
    python3 scripts/extract_figures.py papers/<id>/paper.pdf --render 3 --frac 0.07,0.60,0.95,0.98 --dpi 200
        # 按页面宽高比例裁剪渲染（左,上,右,下，取值 0~1）——先整页渲染看一遍再裁
    python3 scripts/extract_figures.py papers/<id>/paper.pdf --render 3 --clip 54,620,558,780
        # 按 PDF 点坐标裁剪渲染
    python3 scripts/extract_figures.py papers/<id>/paper.pdf --images
        # 抽取内嵌位图（--min-size 过滤小图标，默认 64px）

依赖 PyMuPDF（pip3 install pymupdf）。
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    sys.exit("缺少 PyMuPDF，请先: pip3 install pymupdf")

ALLOWED_IMG_EXTS = {"png", "jpg", "jpeg", "jpx", "jp2", "j2k", "gif", "bmp", "tiff"}


def die(msg: str):
    print("[extract_figures] 错误: %s" % msg, file=sys.stderr)
    sys.exit(1)


def parse_pages(spec: str, max_page: int):
    pages = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, _, b = part.partition("-")
            try:
                lo, hi = int(a), int(b)
            except ValueError:
                die("页码格式错误: %s" % part)
            if not (1 <= lo <= hi <= max_page):
                die("页码范围越界: %s（共 %d 页）" % (part, max_page))
            pages.update(range(lo, hi + 1))
        else:
            try:
                n = int(part)
            except ValueError:
                die("页码格式错误: %s" % part)
            if not (1 <= n <= max_page):
                die("页码越界: %d（共 %d 页）" % (n, max_page))
            pages.add(n)
    return sorted(pages)


def parse_frac(spec: str):
    try:
        l, t, r, b = (float(x) for x in spec.split(","))
    except ValueError:
        die("--frac 需要 4 个数字（左,上,右,下），如 0.07,0.60,0.95,0.98")
    if not (0 <= l < r <= 1 and 0 <= t < b <= 1):
        die("--frac 取值需满足 0 <= 左 < 右 <= 1 且 0 <= 上 < 下 <= 1")
    return l, t, r, b


def parse_clip(spec: str):
    try:
        vals = [float(x) for x in spec.split(",")]
    except ValueError:
        die("--clip 需要 4 个数字（x0,y0,x1,y1，PDF 点）")
    if len(vals) != 4 or not (vals[0] < vals[2] and vals[1] < vals[3]):
        die("--clip 需满足 x0 < x1 且 y0 < y1")
    return vals


# ---------------------------------------------------------------- 各动作

def show_info(doc):
    print("总页数: %d" % len(doc))
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        first = re.sub(r"\s+", " ", text.split("\n", 1)[0])[:60] if text else "(无文本)"
        caps = []
        for ln in text.splitlines():
            ln = ln.strip()
            if re.match(r"^(Figure|Fig\.?|Table)\s+\d", ln):
                caps.append(re.sub(r"\s+", " ", ln)[:70])
            if len(caps) >= 3:
                break
        n_img = len(page.get_images(full=True))
        print("  p%02d | %s" % (i + 1, first))
        for c in caps:
            print("       · %s" % c)
        if n_img:
            print("       · [%d 张内嵌图]" % n_img)


def find_text(doc, needle: str):
    hits = 0
    for i, page in enumerate(doc):
        for r in page.search_for(needle):
            print("p%d: rect(%.0f, %.0f, %.0f, %.0f)  (PDF 点，页宽 %.0f × 页高 %.0f)"
                  % (i + 1, r.x0, r.y0, r.x1, r.y1, page.rect.width, page.rect.height))
            hits += 1
    if not hits:
        print("未找到: %s" % needle)


def render_pages(doc, pages, out_dir: Path, dpi: int, frac, clip):
    written = []
    for n in pages:
        page = doc[n - 1]
        rect = page.rect
        if frac:
            l, t, r, b = frac
            area = fitz.Rect(rect.x0 + l * rect.width, rect.y0 + t * rect.height,
                             rect.x0 + r * rect.width, rect.y0 + b * rect.height)
        elif clip:
            area = fitz.Rect(*clip)
        else:
            area = None
        pix = page.get_pixmap(dpi=dpi, clip=area)
        name = "page-%02d%s.png" % (n, "-crop" if (frac or clip) else "")
        path = out_dir / name
        pix.save(path)
        written.append(path)
    return written


def extract_images(doc, out_dir: Path, min_size: int):
    written, seen = [], set()
    for pno in range(len(doc)):
        for img in doc.get_page_images(pno, full=True):
            xref = img[0]
            if xref in seen:
                continue
            seen.add(xref)
            try:
                d = doc.extract_image(xref)
            except Exception:
                continue
            if min(d["width"], d["height"]) < min_size:
                continue
            ext = (d.get("ext") or "").lower()
            if ext not in ALLOWED_IMG_EXTS:
                continue
            name = "img-p%02d-x%d.%s" % (pno + 1, xref, ext)
            path = out_dir / name
            path.write_bytes(d["image"])
            written.append(path)
    return written


# ---------------------------------------------------------------- 主流程

def main():
    ap = argparse.ArgumentParser(
        description="从论文 PDF 提取图片 / 渲染页面（供图文并茂的阅读报告使用）",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdf", help="论文 PDF 路径，如 papers/<id>/paper.pdf")
    ap.add_argument("--out", help="输出目录（默认 <PDF 所在目录>/figures）")
    ap.add_argument("--info", action="store_true", help="概览：每页文本首行与图表标题")
    ap.add_argument("--find", metavar="TEXT", help="定位文本出现的页码与坐标")
    ap.add_argument("--render", metavar="PAGES", help="整页渲染，如 1,3-5")
    ap.add_argument("--frac", metavar="L,T,R,B",
                    help="按页面宽高比例裁剪（0~1，需与 --render 同用）")
    ap.add_argument("--clip", metavar="X0,Y0,X1,Y1", help="按 PDF 点坐标裁剪（需与 --render 同用）")
    ap.add_argument("--dpi", type=int, default=150, help="渲染 DPI（默认 150，裁剪建议 200）")
    ap.add_argument("--images", action="store_true", help="抽取内嵌位图")
    ap.add_argument("--min-size", type=int, default=64, help="内嵌图最小边长过滤（默认 64px）")
    args = ap.parse_args()

    pdf = Path(args.pdf)
    if not pdf.is_file():
        die("文件不存在: %s" % pdf)
    out_dir = Path(args.out) if args.out else pdf.parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf)
    try:
        any_action = args.info or args.find or args.render or args.images
        if not any_action:
            args.info = True  # 默认显示概览

        if args.info:
            show_info(doc)
        if args.find:
            find_text(doc, args.find)
        written = []
        if args.render:
            pages = parse_pages(args.render, len(doc))
            frac = parse_frac(args.frac) if args.frac else None
            clip = parse_clip(args.clip) if args.clip else None
            if frac and clip:
                die("--frac 与 --clip 只能二选一")
            written += render_pages(doc, pages, out_dir, args.dpi, frac, clip)
        if args.images:
            written += extract_images(doc, out_dir, args.min_size)

        if written:
            print("\n已生成 %d 个文件:" % len(written))
            for p in written:
                print("  %s（%.0f KB）" % (p, p.stat().st_size / 1024))
    finally:
        doc.close()


if __name__ == "__main__":
    main()
