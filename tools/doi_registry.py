"""DOI 注册表反查（DataCite）：突破纯文本抽取的召回天花板。

两个能力：
  - related_datasets(article_id)：以论文 DOI 反查 DataCite，取 relatedIdentifiers 里
    resourceType=Dataset 的关联 DOI（很多数据集「登记过但不在正文」，只能靠元数据拿到）；
  - search_by_name(name)：按数据集名称/accession 在 DataCite 全文检索候选 DOI，供
    名称→DOI 消歧。

工程约束：
  - 结果按 query 落盘缓存到 CACHE_DIR/datacite/，相同查询不重复打网络；
  - LLM_OFFLINE=1 时缓存未命中直接返回空（绝不打网络），供审稿人离线复现；
  - 任何网络异常都吞掉并返回空列表——注册表是「增召回」的可选增强，不能因外部服务
    抖动而让主流程失败。
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import config

_DATACITE_API = "https://api.datacite.org/dois"
_TIMEOUT = 20


def _cache_get(key: str) -> list[dict] | None:
    d = config.CACHE_DIR / "datacite"
    d.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    f = d / f"{h}.json"
    if f.exists():
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return None
    return None


def _cache_put(key: str, value: list[dict]) -> None:
    d = config.CACHE_DIR / "datacite"
    d.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    try:
        (d / f"{h}.json").write_text(json.dumps(value, ensure_ascii=False),
                                     encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


def _get_json(url: str, params: dict) -> dict | None:
    """带超时的 GET；离线模式或任何异常返回 None（调用方据此回退空结果）。"""
    if config.LLM_OFFLINE:
        return None
    try:
        import requests
        r = requests.get(url, params=params, timeout=_TIMEOUT,
                         headers={"User-Agent": "dataset-matching-research/1.0"})
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:  # noqa: BLE001
        return None


def related_datasets(article_id: str) -> list[dict]:
    """反查与本文 DOI 关联、且 resourceType=Dataset 的候选。

    返回 [{doi, relationType, publisher, title}]。缓存命中直接返回；
    离线且未命中返回 []。"""
    key = f"related::{article_id}"
    cached = _cache_get(key)
    if cached is not None:
        return cached
    data = _get_json(f"{_DATACITE_API}/{article_id}", {})
    out: list[dict] = []
    if data:
        attrs = (data.get("data") or {}).get("attributes") or {}
        for rel in attrs.get("relatedIdentifiers", []) or []:
            if (rel.get("relatedIdentifierType") == "DOI"
                    and str(rel.get("resourceTypeGeneral", "")).lower() == "dataset"):
                doi = rel.get("relatedIdentifier", "")
                if doi:
                    out.append({"doi": _norm(doi),
                                "relationType": rel.get("relationType", ""),
                                "publisher": "", "title": ""})
        # 若关联标识没带 resourceType，再补一次「本文 DOI 被哪些 Dataset 关联」检索
        if not out:
            out = _query_related_by_search(article_id)
    _cache_put(key, out)
    return out


def _query_related_by_search(article_id: str) -> list[dict]:
    data = _get_json(_DATACITE_API, {
        "query": f'relatedIdentifiers.relatedIdentifier:"{article_id}"',
        "resource-type-id": "dataset", "page[size]": 40})
    return _parse_hits(data)


def search_by_name(name: str, rows: int = 20) -> list[dict]:
    """按名称/accession 在 DataCite 检索 Dataset 候选（名称→DOI 消歧用）。

    返回 [{doi, title, publisher, relationType}]。空名/离线未命中返回 []。"""
    q = (name or "").strip()
    if not q:
        return []
    key = f"search::{q.lower()}"
    cached = _cache_get(key)
    if cached is not None:
        return cached
    data = _get_json(_DATACITE_API, {
        "query": q, "resource-type-id": "dataset", "page[size]": rows})
    out = _parse_hits(data)
    _cache_put(key, out)
    return out


def _parse_hits(data: dict | None) -> list[dict]:
    out: list[dict] = []
    if not data:
        return out
    for hit in data.get("data", []) or []:
        attrs = hit.get("attributes") or {}
        doi = attrs.get("doi") or hit.get("id", "")
        if not doi:
            continue
        titles = attrs.get("titles") or []
        title = (titles[0].get("title", "") if titles else "")
        out.append({"doi": _norm(doi), "title": title,
                    "publisher": attrs.get("publisher", "") or "",
                    "relationType": ""})
    return out


def _norm(doi: str) -> str:
    s = (doi or "").strip()
    low = s.lower()
    for pre in ("https://doi.org/", "http://doi.org/", "doi:"):
        if low.startswith(pre):
            s = s[len(pre):]
            break
    m = re.search(r"10\.\d{3,9}/\S+", s)
    core = m.group(0).rstrip(".,;)]}") if m else s
    return "https://doi.org/" + core.lower() if core else ""
