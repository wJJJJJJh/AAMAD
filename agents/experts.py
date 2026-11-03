"""Independent discovery roles with a shared Mention schema and no validation votes."""
from __future__ import annotations
from collections import Counter
import json
import config
from agents.base import BaseAgent
from state import AgentLog, Mention
from tools import doi_registry
from tools.candidates import (normalize_identifier, normalize_doi, article_key, distinctive,
                              repo_class, ENUMERABLE_SUBRECORD_KINDS)
from tools.evidence import (bounded, confidence, grounded, has_data_use, DAS,
                            pooled_evidence, metadata_text)

DISCOVERY_STANDARD = """Discover paper-to-dataset references with high recall.
Data produced, analyzed or reused by the article counts, including database
records that also denote compounds, protein structures or other scientific entities.
Return identifiers only from the supplied candidate list. Copy supporting evidence
verbatim from the supplied context. Do not invent identifiers or quotations.
Reject only clearly established non-data objects or context-free false identifiers.
Multiplicity or being a database accession alone is not a rejection reason.
Do not assign Primary/Secondary labels or take part in validation voting.
For every candidate return a record in mentions with dataset_id, verdict (EMIT or
SUPPRESS), evidence, reason and confidence. Keep uncertain candidates for validation.
"""

def _mention(candidate: dict, source: str) -> Mention:
    evidence = bounded(candidate.get("evidence", ""))
    return Mention(raw=candidate["raw"], norm=candidate["norm"], kind=candidate["kind"],
                   evidence=evidence, score=confidence(candidate.get("score", 0)),
                   repo_class=candidate.get("repo_class", repo_class(candidate["norm"])),
                   discovered_by=source, pos=candidate.get("pos"), contexts=[evidence] if evidence else [],
                   sources=[source], has_data_ctx=has_data_use(evidence, candidate["norm"]),
                   in_das_section=bool(DAS.search(evidence)))

class _DiscoveryExpert(BaseAgent):
    def __init__(self, llm=None, trace=None, registry=None):
        super().__init__(llm, trace)
        self.registry = registry if registry is not None else doi_registry

    def _verify(self, bb, mentions: list[Mention]) -> list[Mention]:
        out = []
        for start in range(0, len(mentions), config.DISCOVERY_BATCH_SIZE):
            batch = mentions[start:start + config.DISCOVERY_BATCH_SIZE]
            payload = [{"dataset_id": m.norm, "kind": m.kind,
                        "evidence": pooled_evidence(m), "metadata": bounded(metadata_text(m)),
                        "relations": m.relations} for m in batch]
            answer = self._decide(json.dumps({"article_id": bb.article_id, "article_doi": bb.article_doi,
                                             "candidates": payload},
                                            ensure_ascii=False), note="Discovery semantic check")
            returned = answer.get("mentions", [])
            if not isinstance(returned, list):
                raise ValueError(f"{self.name}: mentions must be a list")
            by_id = {normalize_identifier(r.get("dataset_id", r.get("dataset_doi", ""))): r
                     for r in returned if isinstance(r, dict)}
            for mention in batch:
                row = by_id.get(mention.norm)
                if row is not None:
                    quote = str(row.get("evidence", ""))
                    source = pooled_evidence(mention) + "\n" + metadata_text(mention)
                    valid_quote = grounded(quote, source)
                    if (str(row.get("verdict", "EMIT")).upper() == "SUPPRESS"
                            and valid_quote and not mention.has_data_ctx):
                        bb.trace.append(AgentLog(self.name, {"dataset_id": mention.norm,
                                               "discovery_rejected": True, "evidence": quote}))
                        continue
                    if valid_quote:
                        mention.evidence = bounded(quote)
                        mention.contexts = list(dict.fromkeys(mention.contexts + [bounded(quote)]))
                    mention.score = confidence(row.get("confidence", 0))
                # Omission is not interpreted as a confident rejection.
                out.append(mention)
        return out

class DOIExpert(_DiscoveryExpert):
    name = "DOIExpert"
    system = DISCOVERY_STANDARD + """
Find repository dataset DOI references. Use supplied DataCite resource types and
repository information to distinguish datasets from articles, software, images and
supplementary documents. Mixed repositories can contain valid research data.
"""
    def discover(self, bb) -> list[Mention]:
        direct, uncertain = [], []
        for candidate in bb.candidates:
            if candidate["kind"] != "DOI" or candidate["norm"] == article_key(bb.article_doi or bb.article_id):
                continue
            mention = _mention(candidate, self.name)
            mention.metadata = self.registry.dataset_metadata(mention.norm)
            resource = str(mention.metadata.get("resourceTypeGeneral", "")).lower()
            if resource and resource != "dataset":
                bb.trace.append(AgentLog(self.name, {"dataset_id": mention.norm,
                    "discovery_rejected": True, "resourceTypeGeneral": resource}))
                continue
            if mention.has_data_ctx and (distinctive(mention.norm, mention.kind) or resource == "dataset"):
                mention.score = 0.95
                direct.append(mention)
            else:
                uncertain.append(mention)
        return direct + self._verify(bb, uncertain)

class AccessionExpert(_DiscoveryExpert):
    name = "AccessionExpert"
    system = DISCOVERY_STANDARD + """
Find database accession references. Distinguish accessions from genes, sample
labels, figure labels, primers and catalog numbers using their local context.
PDB and ChEMBL records are valid when they supply study data. Keep a batch of
runs/samples when it is explicitly deposited or reused as study data.
"""
    def discover(self, bb) -> list[Mention]:
        candidates = [c for c in bb.candidates if c["kind"] != "DOI"]
        counts = Counter(c["kind"] for c in candidates)
        direct, uncertain = [], []
        for candidate in candidates:
            mention = _mention(candidate, self.name)
            mention.bulk_risk = (mention.kind in ENUMERABLE_SUBRECORD_KINDS
                                 and counts[mention.kind] > config.BULK_FILTER_THRESHOLD)
            if mention.bulk_risk and not mention.has_data_ctx and not mention.in_das_section:
                bb.trace.append(AgentLog(self.name, {"dataset_id": mention.norm,
                    "discovery_rejected": True, "reason": "unsupported_subrecord_batch"}))
                continue
            if distinctive(mention.norm, mention.kind) and mention.has_data_ctx:
                mention.score = 0.95
                direct.append(mention)
            else:
                uncertain.append(mention)
        return direct + self._verify(bb, uncertain)

class MetadataExpert(_DiscoveryExpert):
    name = "MetadataExpert"
    system = DISCOVERY_STANDARD + """
Find dataset DOIs registered as related to the exact current article in DataCite.
Preserve relation endpoints and direction. Dataset links absent from article text
can be valid. A shared author, a shared repository, or a version/collection link
alone does not establish citation or provenance. Use metadata and textual anchors
when the registered relationship is ambiguous.
"""
    def discover(self, bb) -> list[Mention]:
        direct, uncertain = [], []
        article = bb.article_doi or bb.article_id
        for record in self.registry.related_datasets(article):
            norm = normalize_doi(record.get("doi", ""))
            if not norm or norm == article_key(article):
                continue
            relations = record.get("relations", [])
            evidence = bounded("\n".join(str(record.get(k, "")) for k in ("title", "description")))
            mention = Mention(raw=norm, norm=norm, kind="DOI", evidence=evidence,
                              repo_class=repo_class(norm), discovered_by=self.name,
                              sources=[self.name], contexts=[evidence] if evidence else [],
                              metadata=dict(record), relations=list(relations))
            if doi_registry.relation_label(article, norm, relations):
                mention.score = 0.95
                direct.append(mention)
            else:
                uncertain.append(mention)
        return direct + self._verify(bb, uncertain)
