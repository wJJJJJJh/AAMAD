"""Cached DataCite lookup with exact endpoints and preserved relation direction.

A reverse-search hit is usable only if relatedIdentifiers names this article.
Forward and reverse links are both collected; failed HTTP requests are not cached
as successful empty results.
"""
from __future__ import annotations
import hashlib
import json
import logging
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import config
from tools.candidates import article_key, normalize_doi

_API = "https://api.datacite.org/dois"
log = logging.getLogger(__name__)

def _cache_path(key: str):
    return config.CACHE_DIR / "datacite" / (hashlib.sha256(("v2:" + key).encode()).hexdigest() + ".json")

def _cache_get(key: str):
    p = _cache_path(key)
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    except (OSError, ValueError):
        return None

def _cache_put(key: str, value) -> None:
    p = _cache_path(key)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

def _get_json(url: str, params: dict | None = None) -> dict | None:
    if config.LLM_OFFLINE or not config.DATACITE_ENABLED:
        return None
    try:
        if params:
            url += ("&" if "?" in url else "?") + urlencode(params)
        request = Request(url, headers={"User-Agent": "AAMAD-research/1.0", "Accept": "application/json"})
        with urlopen(request, timeout=config.DATACITE_TIMEOUT) as response:
            return json.load(response)
    except HTTPError as exc:
        if exc.code == 404:
            return {}
        log.warning("DataCite lookup unavailable (HTTP %s)", exc.code)
        return None
    except (URLError, TimeoutError, ValueError, OSError) as exc:
        log.warning("DataCite lookup unavailable (%s)", type(exc).__name__)
        return None

def _get_record(doi: str) -> dict:
    norm = normalize_doi(doi)
    if not norm:
        return {}
    key = "doi:" + norm
    cached = _cache_get(key)
    if cached is not None:
        return cached
    response = _get_json(_API + "/" + quote(norm.removeprefix("https://doi.org/"), safe="/"))
    if response is None:
        return {}
    attrs = (response.get("data") or {}).get("attributes") or {}
    _cache_put(key, attrs)
    return attrs

def _brief(attrs: dict, doi: str) -> dict:
    return {
        "doi": normalize_doi(doi),
        "title": "\n".join(str(t.get("title", "")) for t in attrs.get("titles", [])),
        "description": "\n".join(str(d.get("description", "")) for d in attrs.get("descriptions", [])),
        "publisher": attrs.get("publisher", ""),
        "resourceTypeGeneral": (attrs.get("types") or {}).get("resourceTypeGeneral", ""),
        "creators": attrs.get("creators", []),
        "relations": [],
    }

def dataset_metadata(doi: str) -> dict:
    if not config.DATACITE_ENABLED:
        return {}
    attrs = _get_record(doi)
    return _brief(attrs, doi) if attrs else {}

def related_datasets(article_id: str) -> list[dict]:
    if not config.DATACITE_ENABLED:
        return []
    article = article_key(article_id)
    if not normalize_doi(article):
        return []
    candidates: dict[str, dict] = {}
    def add(record: dict, relation: dict):
        key = record["doi"]
        if not key or key == article:
            return
        item = candidates.setdefault(key, record)
        if relation not in item["relations"]:
            item["relations"].append(relation)

    # Relations registered on the article itself.
    attrs = _get_record(article)
    for rel in attrs.get("relatedIdentifiers", []):
        if str(rel.get("relatedIdentifierType", "")).upper() != "DOI":
            continue
        doi = normalize_doi(rel.get("relatedIdentifier", ""))
        if not doi:
            continue
        record = dataset_metadata(doi)
        resource = record.get("resourceTypeGeneral") or rel.get("resourceTypeGeneral", "")
        if str(resource).lower() == "dataset":
            record = record or _brief({}, doi)
            record["resourceTypeGeneral"] = resource
            add(record, dict(subject=article, object=doi, relationType=rel.get("relationType", "")))

    # Also reverse-query when an article DOI is registered with Crossref, or
    # forward links exist. Do not arbitrarily stop after the first 40 records.
    params = {"query": 'relatedIdentifiers.relatedIdentifier:"' +
                       article.removeprefix("https://doi.org/") + '"',
              "resource-type-id": "dataset", "page[size]": 1000}
    key = "reverse:" + article
    cached = _cache_get(key)
    if cached is None:
        hits = []
        url = _API
        successful = True
        while url:
            response = _get_json(url, params)
            if response is None:
                successful = False
                break
            hits.extend(response.get("data") or [])
            next_url = (response.get("links") or {}).get("next")
            url = next_url if isinstance(next_url, str) and next_url.startswith(_API + "?") else ""
            params = None
        if successful and (not config.LLM_OFFLINE and config.DATACITE_ENABLED):
            _cache_put(key, hits)
    else:
        hits = cached
    for hit in hits:
        attributes = hit.get("attributes") or {}
        if str((attributes.get("types") or {}).get("resourceTypeGeneral", "")).lower() != "dataset":
            continue
        doi = normalize_doi(attributes.get("doi") or hit.get("id", ""))
        record = _brief(attributes, doi)
        for rel in attributes.get("relatedIdentifiers", []):
            if (str(rel.get("relatedIdentifierType", "")).upper() == "DOI"
                    and article_key(rel.get("relatedIdentifier", "")) == article):
                add(record, dict(subject=doi, object=article, relationType=rel.get("relationType", "")))
    return list(candidates.values())

def relation_label(article_id: str, dataset_id: str, relations: list[dict]) -> str | None:
    """Operational input/output mapping, after binding both endpoints.

    IsSupplementTo (dataset -> article) is treated as a study-output signal.
    Version, part/whole, shared-author and generic metadata links are undecided.
    """
    article, dataset = article_key(article_id), normalize_doi(dataset_id)
    if not dataset:
        return None
    labels = set()
    dataset_to_article = {
        "issupplementto": "Primary", "isderivedfrom": "Primary",
        "iscitedby": "Secondary", "isreferencedby": "Secondary", "issourceof": "Secondary",
    }
    article_to_dataset = {
        "issupplementedby": "Primary", "issourceof": "Primary",
        "cites": "Secondary", "references": "Secondary", "isderivedfrom": "Secondary",
    }
    for relation in relations:
        endpoints = (article_key(relation.get("subject", "")), article_key(relation.get("object", "")))
        kind = str(relation.get("relationType", "")).casefold()
        mapping = dataset_to_article if endpoints == (dataset, article) else article_to_dataset if endpoints == (article, dataset) else {}
        if kind in mapping:
            labels.add(mapping[kind])
    return next(iter(labels)) if len(labels) == 1 else None
