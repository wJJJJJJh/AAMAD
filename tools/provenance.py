"""Section 3.5: first decisive relation, then local text, then repository prior."""
from __future__ import annotations
import re
from state import AgentLog, Prediction
from tools.candidates import normalize_doi
from tools.doi_registry import relation_label
from tools.evidence import GENERATION, REUSE, DAS, candidate_clauses, pooled_evidence, positive_cue

_OWNERSHIP = re.compile(r"\b(?:this study|our data|our study|we)\b", re.I)
_EXTERNAL = re.compile(r"\bet al\.?|\([^)]*\b(?:19|20)\d{2}\b[^)]*\)", re.I)

def text_label(article_id: str, mention, authors: tuple[str, ...] = ()) -> str | None:
    window = pooled_evidence(mention)
    clauses = candidate_clauses(window, mention.norm)
    labels = set()
    for clause in clauses:
        generation = positive_cue(GENERATION, clause)
        reuse = positive_cue(REUSE, clause)
        if generation and not reuse:
            labels.add("Primary")
        elif reuse and not generation:
            labels.add("Secondary")
        elif generation and reuse:
            own = any(re.search(r"\b" + re.escape(a) + r"\b", window, re.I) for a in authors)
            own = own and bool(_OWNERSHIP.search(clause) or DAS.search(window) or mention.in_das_section)
            external = bool(_EXTERNAL.search(clause))
            if own and not external:
                labels.add("Primary")
            elif external and not own:
                labels.add("Secondary")
    if len(labels) > 1:
        own = any(re.search(r"\b" + re.escape(a) + r"\b", window, re.I) for a in authors)
        own = own and bool(_OWNERSHIP.search(window) or DAS.search(window) or mention.in_das_section)
        external = bool(_EXTERNAL.search(" ".join(clauses)))
        if own and not external:
            return "Primary"
        if external and not own:
            return "Secondary"
    return next(iter(labels)) if len(labels) == 1 else None

def classify(bb, mention, registry=None) -> Prediction:
    relations = list(mention.relations)
    article = bb.article_doi or bb.article_id
    # Text-only DOI candidates also need the highest-priority relation check.
    if registry is not None and normalize_doi(mention.norm) and not relations:
        for record in registry.related_datasets(article):
            if normalize_doi(record.get("doi", "")) == normalize_doi(mention.norm):
                relations.extend(record.get("relations", []))
    label = relation_label(article, mention.norm, relations)
    route = "relation"
    if label is None:
        label = text_label(bb.article_id, mention, bb.authors)
        route = "text"
    if label is None:
        label = "Primary" if normalize_doi(mention.norm) else "Secondary"
        route = "repository_prior"
    bb.trace.append(AgentLog("ProvenanceClassifier",
                            {"dataset_id": mention.norm, "type": label, "source": route}))
    # A deterministic rule label is not a calibrated probability.
    return Prediction(mention.norm, label, route, mention.discovered_by)
