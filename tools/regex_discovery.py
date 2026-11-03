"""Compatibility entry points using the same patterns as all other variants."""
from collections import Counter
from tools.candidates import DOI_PATTERN, extract_candidates, normalize_doi, rank_candidates

def discover_candidates_regex(fulltext: str) -> list[dict]:
    return extract_candidates(fulltext)

def annotate_context(candidates: list[dict], fulltext: str, window: int = 450) -> list[dict]:
    return rank_candidates(candidates, fulltext)

def get_deterministic_candidates(fulltext: str, with_context: bool = True) -> list[dict]:
    found = extract_candidates(fulltext)
    return rank_candidates(found, fulltext) if with_context else found

def count_by_kind(candidates: list[dict]) -> dict[str, int]:
    return dict(Counter(c["kind"] for c in candidates))
