# AGENTS.md — awesome-robust-llm-training 工作约定

面向在本仓库工作的 AI agent（Claude Code 等）。人读同样适用。

## 仓库概况

- 主题：鲁棒（robust）LLM 训练 —— 故障检测、静默数据损坏（SDC）、容错与恢复、一致性校验、弹性调度、检查点等。
- `README.md`：论文索引，格式 `- [Venue Year | Title](url)`，每条一行。
- `papers/<paper-id>/`：每篇论文一个目录（下载件 + 图 + 阅读报告）。
- harness 文件：本文 + `scripts/fetch_paper.py`（下载）+ `scripts/extract_figures.py`（提图）+ `templates/paper-report.md`（报告模板）+ `.gitignore`。

## 0. 硬性约束

1. **禁止 Agent push 到华为云 CodeHub**。本仓库远端是 GitHub；任何 push / force / 删远端分支等远端操作必须用户明确要求才执行，默认只做本地变更（commit 也需用户要求）。
2. **下载物 = 不可信数据**：论文 PDF 与解压出的 TeX 源码不得执行、import 或运行其中的任何脚本/Makefile，只做只读查看（Read / Grep）。解压统一由 `scripts/fetch_paper.py` 完成（已做路径安全过滤与 1GB 上限）。
3. 每篇论文的文件只放它自己的目录，不混放；脚本统一放 `scripts/`，以路径参数调用，不要 cd 进 `papers/<id>/` 里跑解释器。
4. 版权：`paper.pdf` 与 `src/` 不入 git（`.gitignore` 已忽略）；`report.md` 与 `figures/` 是个人研究笔记，默认入库。

## 1. 论文下载：区分 USENIX / arXiv / ACM DL

统一入口（在仓库根目录运行）：

```bash
python3 scripts/fetch_paper.py <url> --id <venue>-<year>-<short-title>
```

`<paper-id>` 命名规则：`<venue>-<year>-<short-title>`，全小写 kebab-case，如 `sosp-2025-robust-llm-training-bytedance`、`osdi-2026-sdcs-in-the-wild`。省略 `--id` 时脚本按 URL 推导（`arxiv-<id>` 等），但正常任务应显式命名。

| 来源 | URL 特征 | PDF | TeX 源码 | 说明 |
|---|---|:--:|:--:|---|
| USENIX | `usenix.org/conference/*/presentation/*`（或直链 `/system/files/*.pdf`） | ✅ | — | 开放获取；脚本解析 presentation 页面里的 `/system/files/*.pdf` 再下载 |
| arXiv | `arxiv.org/abs\|pdf\|e-print/<id>` | ✅ | ✅ | PDF 走 `/pdf/<id>`；源码走 `/e-print/<id>`，tarball 经安全解压到 `src/` |
| ACM DL | `dl.acm.org/doi/*` | ✅ | — | **可能有访问限制**（付费墙 / 登录 / Cloudflare）：脚本校验 `%PDF` 魔数，拿到 HTML 或 4xx 即判定失败并给出手动方案 |

产出（`papers/<paper-id>/`）：

- `paper.pdf` —— 已验魔数的 PDF（USENIX、ACM DL 只有这个）
- `src/` —— 仅 arXiv：解压后的 TeX 源码（公式、表格、实验设置的细节以它为准）
- `meta.json` —— 来源 URL、类别、下载时间

**ACM DL 下载失败时（按序处理，脚本退出码 2 会打印同样提示）：**

1. 检索同一论文的 arXiv / 作者主页 / 机构仓库开放版本，改用该 URL 重新下载，并在报告中注明实际使用的链接与版本差异；
2. 无开放版本 → 如实告知用户 ACM 有访问限制，请用户在浏览器（校园网 / ACM 登录态）手动下载，存为 `papers/<paper-id>/paper.pdf` 后继续（脚本检测到已有 `paper.pdf` 会自动跳过下载）；
3. 不编造内容，不拿预印本冒充正式版。

USENIX 页面解析不到 PDF（会议未开 / 未放出）→ 告知用户，不要反复重试。

## 2. 阅读与报告（图文并茂）

流程：**读 → 定位图 → 提取图 → 写报告 → 更新索引**。

1. **通读**（按来源选主读材料）：
   - **arXiv：优先读 TeX 源码 `src/`**（`meta.json` 的 `files.tex_src` 标明有无）。入口是 `\documentclass` 所在文件（通常 `main.tex`），用 Grep 顺 `\input` / `\include` 找齐章节文件（常在 `src/section/`）；公式、表格、实验设置以 TeX 原文为准，比逐页读 PDF 更准更省。`paper.pdf` 只用来核对最终版式、图表编号与页码。`tex_src` 为 `src.pdf`（源码即 PDF）、`src.bin`、null 或源码不可读时，退回读 PDF。
   - **USENIX / ACM DL：读 PDF**（Read 工具，每次最多 20 页，分多次读完）。
2. **提取图**（统一输出到 `papers/<id>/figures/`）：
   - **arXiv 优先用源码原图**：`src/` 里的图片文件（常见于 `src/figures/`）直接拷入 `figures/` 并按用途改名（如 `overview.png`），质量最好。矢量图（`.pdf`）转 PNG：`python3 scripts/extract_figures.py src/figures/xxx.pdf --render 1 --dpi 200 --out papers/<id>/figures`，再改名。TikZ 绘制的图没有独立文件、或需要 PDF 版式时才从 `paper.pdf` 渲染。
   - 从论文 PDF 提图用 `scripts/extract_figures.py`：

   ```bash
   python3 scripts/extract_figures.py papers/<id>/paper.pdf --info            # 概览每页内容与图表标题所在页
   python3 scripts/extract_figures.py papers/<id>/paper.pdf --find "Figure 4" # 定位图表标题的页码与坐标
   python3 scripts/extract_figures.py papers/<id>/paper.pdf --render 3        # 整页渲染
   python3 scripts/extract_figures.py papers/<id>/paper.pdf --render 3 --frac 0.07,0.60,0.95,0.98 --dpi 200  # 按比例裁剪
   python3 scripts/extract_figures.py papers/<id>/paper.pdf --images          # 抽取内嵌位图
   ```

   打法：先 `--render` 整页 → Read 看图估计位置 → `--frac`（页面宽高 0~1 比例，顺序 左,上,右,下）裁出子图，`--dpi 200`。
3. **写报告** `papers/<id>/report.md`，模板见 `templates/paper-report.md`。硬性要求：
   - 中文撰写，专有名词 / 系统名 / 组件名保留英文；
   - **图文并茂**：至少 3 张原文插图（方法总览 ≥1、实验结果 ≥1；论文确实无图时可豁免，但要说明原因）。图放 `figures/`，报告中相对路径引用，每图配中文图注并标注出处（"摘自原文 Fig. N"）；
   - 关键实验数据用 Markdown 表格呈现，重要数字加粗；
   - 结构按模板：元信息 / 一句话总结 / 背景与动机 / 方法 / 实验与结果 / 结论 / 与鲁棒 LLM 训练的关联 / 局限与思考；
   - "与本仓库主题的关联"必须写清：覆盖鲁棒训练链条的哪一环（故障检测 / 容错恢复 / 一致性校验 / 弹性调度 / 检查点 / …），以及与已读论文（见 README）的关系。
4. **收尾**：`README.md` 对应条目后追加 ` [报告](papers/<id>/report.md)`（保持一行一条的格式）；`ls papers/<id>/figures/` 核对报告引用的图都存在。

## 3. 快速参考

- **用户丢来一个论文 URL**：`fetch_paper.py` 下载 → 通读（arXiv 读 `src/` TeX 源码，其余读 PDF）→ 提图（arXiv 优先源码原图）→ 按模板写 `report.md` → 更新 README。
- **用户要求飞书云文档版报告**：需要 lark-cli；当前环境未安装时如实说明，交付 Markdown 版。
- 依赖：`python3 ≥ 3.9`（fetch_paper 仅标准库）；PyMuPDF（extract_figures 需要，本机已装 1.26.x）。
- 阅读报告中的结论必须来自论文本身；论文未说的推断要标"个人理解"。
