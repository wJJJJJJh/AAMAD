"""TEI/JATS/text/PDF loading without duplicated nested XML paragraphs."""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path
from xml.etree import ElementTree as ET
import config
from tools.candidates import article_key, normalize_doi
from tools.evidence import evidence_window

def _source(article_id: str, split: str) -> Path:
    names = list(dict.fromkeys(re.sub(r"[^A-Za-z0-9._-]", "_", value) for value in
                 [article_id, article_key(article_id).removeprefix("https://doi.org/")]))
    bases = [config.DATA_DIR / split, config.DATA_DIR]
    for base in bases:
        for folder, suffix in [("XML", ".xml"), ("TXT", ".txt"), ("PDF", ".pdf"),
                               ("", ".xml"), ("", ".txt"), ("", ".pdf")]:
            for name in names:
                p = base / folder / (name + suffix)
                if p.is_file():
                    return p
    raise FileNotFoundError(f"No XML, TXT or PDF for {article_id!r} under {config.DATA_DIR / split}")

def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]

def _parse_tei(path: Path) -> str:
    root = ET.parse(path).getroot()
    parts = []
    def walk(node):
        tag = _local(node.tag)
        if tag in {"ref-list", "listBibl"}:
            parts.append("References")
        if tag in {"article-title", "title", "head", "p", "ref", "biblStruct"}:
            text = " ".join(" ".join(node.itertext()).split())
            if text:
                parts.append(text)
            return
        for child in node:
            walk(child)
    walk(root)
    return "\n".join(parts)

def _parse_pdf(path: Path) -> str:
    from pdfminer.high_level import extract_text
    return extract_text(str(path))

def load_fulltext(article_id: str, split: str = "train") -> str:
    source = _source(article_id, split)
    signature = f"v3:{source.resolve()}:{source.stat().st_mtime_ns}:{source.stat().st_size}"
    cache = config.CACHE_DIR / "fulltext" / (hashlib.sha256(signature.encode()).hexdigest() + ".json")
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))["text"]
    if source.suffix.lower() == ".xml":
        text = _parse_tei(source)
    elif source.suffix.lower() == ".pdf":
        text = _parse_pdf(source)
    else:
        text = source.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Empty full text: {source}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"text": text}, ensure_ascii=False), encoding="utf-8")
    return text

def get_authors(article_id: str, split: str = "train") -> tuple[str, ...]:
    source = _source(article_id, split)
    if source.suffix.lower() != ".xml":
        return ()
    root = ET.parse(source).getroot()
    names = []
    headers = [n for n in root.iter() if _local(n.tag) in {"teiHeader", "article-meta"}]
    for header in headers:
        for author in header.iter():
            tag = _local(author.tag)
            if tag != "author" and not (tag == "contrib" and author.get("contrib-type", "author") == "author"):
                continue
            for n in author.iter():
                if _local(n.tag) == "surname" and n.text and n.text.strip() not in names:
                    names.append(n.text.strip())
    return tuple(names)

def get_article_doi(article_id: str, split: str = "train") -> str:
    doi = normalize_doi(article_key(article_id))
    if doi:
        return doi
    source = _source(article_id, split)
    if source.suffix.lower() != ".xml":
        return ""
    root = ET.parse(source).getroot()
    for header in root.iter():
        if _local(header.tag) not in {"teiHeader", "article-meta"}:
            continue
        for node in header.iter():
            kind = node.get("pub-id-type", node.get("type", "")).lower()
            if _local(node.tag) in {"article-id", "idno"} and kind == "doi":
                doi = normalize_doi(node.text or "")
                if doi:
                    return doi
    return ""

def build_discovery_context(article_id: str, split: str, cands: list[dict]) -> str:
    text = load_fulltext(article_id, split)
    return "\n[...]\n".join(dict.fromkeys(evidence_window(text, c["pos"]) for c in cands))[:config.MAX_FULLTEXT_CHARS]
