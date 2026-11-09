#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fetch_paper.py — 按来源下载论文 PDF（arXiv 额外下载 TeX 源码）。

按 URL 来源区分，规则：
    USENIX   https://www.usenix.org/conference/<conf>/presentation/<slug>   仅 PDF
    arXiv    https://arxiv.org/abs/<id>                                    PDF + TeX 源码
    ACM DL   https://dl.acm.org/doi/<doi>                                  仅 PDF（可能有访问限制）

用法（在仓库根目录运行）：
    python3 scripts/fetch_paper.py <url> [--id <paper-id>] [--force] [--pdf-only]

示例：
    python3 scripts/fetch_paper.py https://arxiv.org/abs/2509.16293 --id sosp-2025-robust-llm-training-bytedance
    python3 scripts/fetch_paper.py https://www.usenix.org/conference/osdi26/presentation/zheng --id osdi-2026-sdcs-in-the-wild
    python3 scripts/fetch_paper.py https://dl.acm.org/doi/10.1145/3830418.3843892 --id sosp-2026-anchor

每篇论文一个独立目录 papers/<paper-id>/（不与其他论文混放），产出：
    paper.pdf   下载的 PDF（已校验 %PDF- 魔数）
    src/        arXiv TeX 源码（安全解压：拒绝绝对路径 / .. / 链接 / 设备文件，总量上限 1GB）
    meta.json   来源、URL、下载时间等元信息

安全提示：下载的论文与解压出的文件属于不可信数据，只读查看，
不要执行 / 导入其中的任何内容。仅依赖 Python 标准库（>= 3.9）。
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import re
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parent.parent
PAPERS_ROOT = REPO_ROOT / "papers"

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

MAX_SRC_BYTES = 1 << 30  # TeX 源码解压总量上限 1GB，防解压炸弹


def info(msg: str) -> None:
    print(f"[fetch_paper] {msg}")


def die(msg: str, code: int = 1):
    sys.stdout.flush()
    print(f"[fetch_paper] 错误: {msg}", file=sys.stderr)
    sys.exit(code)


# ---------------------------------------------------------------- HTTP 基础

def http_get(url: str, accept: str = "*/*"):
    """GET 并返回 (bytes, content-type, 最终URL)。"""
    req = Request(url, headers={"User-Agent": UA, "Accept": accept})
    with urlopen(req, timeout=90) as r:
        return r.read(), r.headers.get("Content-Type", ""), r.geturl()


def is_pdf(data: bytes) -> bool:
    return data[:1024].lstrip()[:5] == b"%PDF-"


# ---------------------------------------------------------------- URL 分类与解析

def classify(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    if host == "usenix.org" or host == "www.usenix.org":
        return "usenix"
    if host in ("arxiv.org", "www.arxiv.org", "export.arxiv.org"):
        return "arxiv"
    if host in ("dl.acm.org", "www.dl.acm.org"):
        return "acm"
    die("不支持的来源（目前仅 usenix.org / arxiv.org / dl.acm.org）: %s" % url)


def sanitize_id(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-").lower()
    if not s:
        die("paper-id 为空，请用 --id 指定")
    return s


def arxiv_id_from(url: str) -> str:
    path = urlparse(url).path
    m = re.match(r"^/(?:abs|pdf|e-print|format)/(.+?)(?:v\d+)?$", path)
    if not m:
        die("无法从 arXiv URL 解析论文 ID: %s" % url)
    return m.group(1)


def acm_doi_from(url: str) -> str:
    path = urlparse(url).path
    m = re.match(r"^/doi/(.+?)/?$", path)
    if not m:
        die("无法从 ACM DL URL 解析 DOI: %s" % url)
    rest = m.group(1)
    for prefix in ("pdf/", "epdf/", "abs/", "fulltext/", "cover/", "proceedings/"):
        if rest.startswith(prefix):
            rest = rest[len(prefix):]
            break
    if not re.match(r"^10\.\d{4,5}/\S+$", rest):
        die("DOI 格式异常: %s（url=%s）" % (rest, url))
    return rest


def default_id(kind: str, url: str) -> str:
    p = urlparse(url)
    if kind == "usenix":
        m = re.match(r"^/conference/([^/]+)/presentation/([^/]+)/?$", p.path)
        if m:
            return sanitize_id("usenix-%s-%s" % (m.group(1), m.group(2)))
        m = re.match(r"^/system/files/(.+)\.pdf$", p.path)
        if m:
            return sanitize_id("usenix-%s" % m.group(1))
        return sanitize_id("usenix-" + p.path.strip("/").replace("/", "-"))
    if kind == "arxiv":
        return sanitize_id("arxiv-" + arxiv_id_from(url).replace("/", "-"))
    return sanitize_id("acm-" + acm_doi_from(url).rsplit("10.1145/", 1)[-1])


# ---------------------------------------------------------------- USENIX

def usenix_pdf_url(page_url: str) -> str:
    """从 USENIX presentation 页面解析 PDF 链接（通常在 /system/files/ 下）。"""
    if re.search(r"\.pdf(\?|$)", page_url):
        return page_url  # 已是直链
    try:
        html, _, _ = http_get(page_url, accept="text/html,*/*")
    except HTTPError as e:
        die("获取 USENIX 页面失败: HTTP %s %s" % (e.code, page_url))
    except URLError as e:
        die("网络错误: %s" % e.reason)
    html = html.decode("utf-8", "replace")
    anchors = re.findall(r'<a\b[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html, re.I | re.S)
    cands = []
    for href, text in anchors:
        if ".pdf" not in href.lower():
            continue
        clean = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)).strip()
        cands.append((href, clean))
    in_files = [c for c in cands if "/system/files/" in c[0]]
    pool = in_files or cands
    if not pool:
        die("USENIX presentation 页面上没有找到 PDF 链接（论文可能尚未放出），请人工确认: %s" % page_url)
    # 优先非 slides 的链接（正文 Paper），其次取第一个
    for href, text in pool:
        low = (text + " " + href).lower()
        if "slide" in low:
            continue
        if "paper" in low or "pdf" in low:
            return urljoin(page_url, href)
    return urljoin(page_url, pool[0][0])


# ---------------------------------------------------------------- arXiv

def _safe_members(tf: tarfile.TarFile):
    safe, dropped, total = [], [], 0
    for m in tf.getmembers():
        if m.name.startswith(("/", "\\")):
            dropped.append((m.name, "绝对路径")); continue
        if ".." in PurePosixPath(m.name).parts:
            dropped.append((m.name, "包含 ..")); continue
        if m.issym() or m.islnk():
            dropped.append((m.name, "符号/硬链接")); continue
        if m.isdev():
            dropped.append((m.name, "设备文件")); continue
        total += max(m.size, 0)
        if total > MAX_SRC_BYTES:
            die("TeX 源码解压总量超过 1GB 上限，中止（疑似解压炸弹）")
        safe.append(m)
    return safe, dropped


def fetch_arxiv_source(aid: str, pdir: Path):
    """下载 arXiv e-print 源码并安全解压到 src/，返回描述字符串。"""
    urls = ["https://arxiv.org/e-print/%s" % aid,
            "https://export.arxiv.org/e-print/%s" % aid]
    raw = None
    for u in urls:
        try:
            raw, _, _ = http_get(u, accept="application/x-eprint, application/gzip, */*")
            break
        except HTTPError as e:
            info("e-print 尝试 %s → HTTP %s" % (u, e.code))
        except URLError as e:
            info("e-print 尝试 %s → %s" % (u, e.reason))
    if raw is None:
        info("警告: 未取到 TeX 源码（无源码或网络原因），仅保留 PDF")
        return None
    if is_pdf(raw):
        (pdir / "src.pdf").write_bytes(raw)  # 源码本身就是 PDF 的情况
        return "src.pdf（源码即 PDF，无 TeX）"
    dest = pdir / "src"
    try:
        tf = tarfile.open(fileobj=io.BytesIO(raw), mode="r:*")
        with tf:
            members, dropped = _safe_members(tf)
            dest.mkdir(parents=True, exist_ok=True)
            tf.extractall(dest, members=members)
        for name, reason in dropped:
            info("已跳过不安全成员: %s（%s）" % (name, reason))
        return "src/（%d 个文件）" % len(members)
    except tarfile.ReadError:
        pass  # 不是 tar，尝试单文件
    try:
        content = gzip.decompress(raw)
    except OSError:
        content = raw
    try:
        text = content.decode("utf-8")
        if sum(ch.isprintable() or ch in "\n\t\r" for ch in text[:4000]) / max(len(text[:4000]), 1) < 0.9:
            raise UnicodeDecodeError("utf-8", b"", 0, 1, "不像文本")
    except (UnicodeDecodeError, ValueError):
        (pdir / "src.bin").write_bytes(raw)
        return "src.bin（无法识别的源码格式）"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "main.tex").write_bytes(content)
    return "src/main.tex（单文件源码）"


# ---------------------------------------------------------------- ACM DL

def acm_pdf_url(url: str) -> str:
    return "https://dl.acm.org/doi/pdf/%s" % acm_doi_from(url)


def die_acm(detail: str, pid: str):
    sys.stdout.flush()
    print("[fetch_paper] ACM DL 未返回 PDF（%s）。" % detail, file=sys.stderr)
    print("ACM Digital Library 可能有访问限制（付费墙 / 需登录 / 反爬）。按以下顺序处理：", file=sys.stderr)
    print("  1) 检索该论文的 arXiv / 作者主页 / 机构仓库开放版本，改用该 URL 重新运行本脚本；", file=sys.stderr)
    print("  2) 无开放版本时：请用户在浏览器（校园网或 ACM 登录态）手动下载，保存为", file=sys.stderr)
    print("     papers/%s/paper.pdf，然后直接继续阅读报告流程（脚本检测到 paper.pdf 会跳过下载）；" % pid, file=sys.stderr)
    print("  3) 如实向用户说明，不要跳过或伪造内容。", file=sys.stderr)
    sys.exit(2)


# ---------------------------------------------------------------- 主流程

def download_pdf(url: str, kind: str, pid: str) -> bytes:
    try:
        data, ctype, final = http_get(url, accept="application/pdf,*/*")
    except HTTPError as e:
        if kind == "acm":
            die_acm("HTTP %s" % e.code, pid)
        die("下载失败: HTTP %s %s" % (e.code, url))
    except URLError as e:
        die("网络错误: %s（%s）" % (e.reason, url))
    if not is_pdf(data):
        if kind == "acm":
            head = re.sub(rb"\s+", b" ", data[:150]).decode("utf-8", "replace")
            die_acm("content-type=%s，返回了 HTML 页面，开头: %s" % (ctype, head), pid)
        head = re.sub(rb"\s+", b" ", data[:150]).decode("utf-8", "replace")
        die("返回内容不是 PDF（content-type=%s），开头: %s" % (ctype, head))
    return data


def main():
    ap = argparse.ArgumentParser(
        description="按来源（USENIX / arXiv / ACM DL）下载论文 PDF，arXiv 另下 TeX 源码",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n"
               "  python3 scripts/fetch_paper.py https://arxiv.org/abs/2509.16293 \\\n"
               "      --id sosp-2025-robust-llm-training-bytedance")
    ap.add_argument("url", help="论文 URL（usenix.org / arxiv.org / dl.acm.org）")
    ap.add_argument("--id", dest="pid",
                    help="目录名 <venue>-<year>-<short-title>（推荐显式指定；缺省按 URL 推导）")
    ap.add_argument("--force", action="store_true", help="覆盖已存在的 paper.pdf")
    ap.add_argument("--pdf-only", action="store_true", help="arXiv 也只下 PDF，不拉 TeX 源码")
    args = ap.parse_args()

    kind = classify(args.url)
    pid = sanitize_id(args.pid or default_id(kind, args.url))
    pdir = PAPERS_ROOT / pid
    pdir.mkdir(parents=True, exist_ok=True)
    info("来源: %-5s → %s" % (kind, pdir.relative_to(REPO_ROOT)))

    meta_path = pdir / "meta.json"
    if meta_path.exists():
        try:
            old = json.loads(meta_path.read_text())
            if old.get("url") and old["url"] != args.url:
                info("警告: 该目录 meta.json 记录的是另一条 URL（%s），请确认是同一论文" % old["url"])
        except (ValueError, OSError):
            pass

    # 1) PDF（USENIX / ACM DL 仅此一步）
    if kind == "usenix":
        pdf_url = usenix_pdf_url(args.url)
    elif kind == "acm":
        pdf_url = acm_pdf_url(args.url)
    else:
        pdf_url = "https://arxiv.org/pdf/%s" % arxiv_id_from(args.url)

    pdf_path = pdir / "paper.pdf"
    if pdf_path.exists() and not args.force:
        info("%s 已存在，跳过下载（--force 可重新下载）" % pdf_path.relative_to(REPO_ROOT))
    else:
        info("下载 PDF: %s" % pdf_url)
        pdf_path.write_bytes(download_pdf(pdf_url, kind, pid))
        info("PDF 已保存: %s（%.1f MB）" % (pdf_path.relative_to(REPO_ROOT),
                                          pdf_path.stat().st_size / 1e6))

    # 2) TeX 源码（仅 arXiv）
    tex_desc = None
    if kind == "arxiv" and not args.pdf_only:
        tex_desc = fetch_arxiv_source(arxiv_id_from(args.url), pdir)
        if tex_desc:
            info("TeX 源码: %s" % tex_desc)

    # 3) meta.json
    meta = {
        "id": pid,
        "source": kind,
        "url": args.url,
        "pdf_url": pdf_url,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": {"pdf": "paper.pdf", "tex_src": tex_desc},
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")

    print("\n完成 ✓  papers/%s/" % pid)
    if pdf_path.exists():
        print("  paper.pdf")
    if tex_desc:
        print("  %s" % tex_desc)
    print("  meta.json")
    print("下一步:\n  python3 scripts/extract_figures.py papers/%s/paper.pdf --info\n"
          "  然后按 templates/paper-report.md 写阅读报告" % pid)


if __name__ == "__main__":
    main()
