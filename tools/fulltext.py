"""全文载入与聚焦视图构建（确定性工具，零 LLM）。

职责：
  - load_fulltext：优先解析 GROBID TEI XML（结构干净），无 XML 时回退 PDF 文本抽取；
  - get_authors：从 TEI teiHeader 取作者姓氏，供 Primary/Secondary 判定自引线索；
  - build_discovery_context：围绕高分候选裁出「聚焦视图」，把有限的 token 预算投给
    真正含数据引用的段落（数据可得性声明、accession 语境），而非整篇平铺截断。

缓存：解析结果按 (split, article_id) 落盘到 CACHE_DIR/fulltext/，避免重复解析 PDF。
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import config

_TEI_NS = {"t": "http://www.tei-c.org/ns/1.0"}


def _paths(article_id: str, split: str) -> tuple[Path, Path]:
    xml_dir = config.TRAIN_XML if split == "train" else config.TEST_XML
    pdf_dir = config.TRAIN_PDF if split == "train" else config.TEST_PDF
    return xml_dir / f"{article_id}.xml", pdf_dir / f"{article_id}.pdf"


def _cache_file(article_id: str, split: str) -> Path:
    d = config.CACHE_DIR / "fulltext"
    d.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^0-9A-Za-z._-]", "_", article_id)
    return d / f"{split}__{safe}.json"


def _parse_tei(xml_path: Path) -> str:
    """从 GROBID TEI 抽取纯文本：标题 + 摘要 + 正文 <body> + 附录/可得性段落。

    保留段落顺序与换行，便于后续正则定位数据可得性声明与参考文献区。"""
    from lxml import etree
    try:
        tree = etree.parse(str(xml_path))
    except Exception:  # noqa: BLE001  解析失败交由上层回退 PDF
        return ""
    root = tree.getroot()
    parts: list[str] = []

    def _text(el) -> str:
        return " ".join(t.strip() for t in el.itertext() if t and t.strip())

    for xp in ("//t:titleStmt/t:title", "//t:abstract"):
        for el in root.xpath(xp, namespaces=_TEI_NS):
            s = _text(el)
            if s:
                parts.append(s)
    # 正文与背面（back：含 data availability / references）逐段落抽取
    for xp in ("//t:body//t:div", "//t:body//t:p", "//t:back//t:div"):
        for el in root.xpath(xp, namespaces=_TEI_NS):
            s = _text(el)
            if s:
                parts.append(s)
    # 去重相邻重复段（div 与其内 p 可能重复覆盖）
    out, prev = [], ""
    for s in parts:
        if s != prev:
            out.append(s)
        prev = s
    return "\n".join(out)


def _parse_pdf(pdf_path: Path) -> str:
    try:
        from pdfminer.high_level import extract_text
        return extract_text(str(pdf_path)) or ""
    except Exception:  # noqa: BLE001
        return ""


@lru_cache(maxsize=512)
def load_fulltext(article_id: str, split: str = "train") -> str:
    """载入全文（TEI 优先，PDF 回退），带磁盘缓存。找不到任何来源返回空串。"""
    cf = _cache_file(article_id, split)
    if cf.exists():
        try:
            return json.loads(cf.read_text(encoding="utf-8")).get("text", "")
        except Exception:  # noqa: BLE001
            pass
    xml_path, pdf_path = _paths(article_id, split)
    text = ""
    if xml_path.exists():
        text = _parse_tei(xml_path)
    if not text and pdf_path.exists():
        text = _parse_pdf(pdf_path)
    try:
        cf.write_text(json.dumps({"text": text}, ensure_ascii=False),
                      encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    return text


@lru_cache(maxsize=512)
def get_authors(article_id: str, split: str = "train") -> tuple[str, ...]:
    """从 TEI teiHeader 取作者姓氏列表（供自引线索）。无 XML 或解析失败返回空元组。"""
    xml_path, _ = _paths(article_id, split)
    if not xml_path.exists():
        return ()
    from lxml import etree
    try:
        root = etree.parse(str(xml_path)).getroot()
    except Exception:  # noqa: BLE001
        return ()
    surnames: list[str] = []
    for el in root.xpath("//t:fileDesc//t:author//t:persName/t:surname",
                         namespaces=_TEI_NS):
        s = (el.text or "").strip()
        if s and s not in surnames:
            surnames.append(s)
    return tuple(surnames)


def evidence_window(text: str, pos: int, half: int | None = None) -> str:
    """取候选位置附近的证据窗口（辩论架构里每条主张都要锚定到这样的片段）。"""
    half = half if half is not None else config.FOCUS_CHARS // 2
    return text[max(0, pos - half): pos + half]


def build_discovery_context(article_id: str, split: str,
                            cands: list[dict]) -> str:
    """围绕高分候选拼「聚焦视图」：每个候选取 ±FOCUS_CHARS/2 片段，按出现顺序拼接，
    去重叠、控总长在 MAX_FULLTEXT_CHARS 内。无候选时回退全文头部截断。"""
    text = load_fulltext(article_id, split)
    if not text:
        return ""
    if not cands:
        return text[:config.MAX_FULLTEXT_CHARS]
    half = config.FOCUS_CHARS // 2
    spans: list[tuple[int, int]] = []
    for c in cands:
        p = int(c.get("pos", 0) or 0)
        spans.append((max(0, p - half), min(len(text), p + half)))
    spans.sort()
    merged: list[list[int]] = []
    for a, b in spans:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    chunks, total = [], 0
    for a, b in merged:
        seg = text[a:b]
        if total + len(seg) > config.MAX_FULLTEXT_CHARS:
            seg = seg[: config.MAX_FULLTEXT_CHARS - total]
        chunks.append(seg)
        total += len(seg)
        if total >= config.MAX_FULLTEXT_CHARS:
            break
    return "\n[...]\n".join(chunks)
