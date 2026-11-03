"""Paper-aligned discovery -> selective validation -> provenance classification."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict
import json
import time
import config
from state import Blackboard, Mention, AgentLog
from llm.client import get_llm
from tools import doi_registry
from tools.candidates import (article_key, normalize_identifier, normalize_doi, repo_class,
                              distinctive, extract_candidates, rank_candidates)
from tools.evidence import bounded, has_data_use, pooled_evidence
from tools.fulltext import load_fulltext, get_authors, get_article_doi
from tools.provenance import classify
from agents.experts import DOIExpert, AccessionExpert, MetadataExpert
from agents.debate_loop import debate_emit_suppress

MODES = ("AAMAD", "discovery_only", "conventional_debate", "rule_based",
         "metadata_only", "single_agent", "bare_llm")

class CachedRegistry:
    def __init__(self, backend):
        self.backend = backend
        self.records, self.links = {}, {}
    def dataset_metadata(self, doi):
        key = normalize_doi(doi)
        if key not in self.records:
            self.records[key] = self.backend.dataset_metadata(key)
        return deepcopy(self.records[key])
    def related_datasets(self, article):
        key = article_key(article)
        if key not in self.links:
            self.links[key] = self.backend.related_datasets(article)
        return deepcopy(self.links[key])

def merge_mentions(mentions: list[Mention]) -> list[Mention]:
    groups = {}
    for item in mentions:
        key = normalize_identifier(item.norm)
        if key:
            groups.setdefault(key, []).append(item)
    merged = []
    for key, items in groups.items():
        # Prefer text containing an explicit candidate-specific data-use assertion.
        ordered = sorted(items, key=lambda m: (has_data_use(pooled_evidence(m), key), m.score), reverse=True)
        result = deepcopy(ordered[0])
        result.norm = key
        result.score = max(m.score for m in items)
        result.sources = list(dict.fromkeys(s for m in items for s in (m.sources or [m.discovered_by]) if s))
        result.discovered_by = "+".join(result.sources)
        result.contexts = list(dict.fromkeys(e for m in ordered for e in [m.evidence] + m.contexts if e))
        result.evidence = bounded("\n".join(result.contexts))
        result.relations = list({json.dumps(r, sort_keys=True): r for m in items for r in m.relations}.values())
        result.metadata = {k: v for m in reversed(ordered) for k, v in m.metadata.items() if v}
        result.in_das_section = any(m.in_das_section for m in items)
        result.has_data_ctx = has_data_use(pooled_evidence(result), key)
        merged.append(result)
    return sorted(merged, key=lambda m: (m.pos is None, m.pos or 0, m.norm))

class Orchestrator:
    def __init__(self, llm=None, registry=None):
        self.llm = llm if llm is not None else get_llm()
        self.registry = CachedRegistry(registry if registry is not None else doi_registry)

    def _snapshot(self):
        return self.llm.snapshot() if hasattr(self.llm, "snapshot") else {}

    def prepare_article(self, article_id: str, split: str = "train", *,
                        fulltext: str | None = None, authors: tuple[str, ...] | None = None,
                        mode: str = "AAMAD") -> Blackboard:
        started, before = time.perf_counter(), self._snapshot()
        text = load_fulltext(article_id, split) if fulltext is None else fulltext
        surnames = get_authors(article_id, split) if authors is None and fulltext is None else (authors or ())
        doi = get_article_doi(article_id, split) if fulltext is None else normalize_doi(article_key(article_id))
        bb = Blackboard(article_id=article_id, article_doi=doi, split=split, fulltext=text, authors=surnames)
        bb.candidates = rank_candidates(extract_candidates(text), text)
        bb.context = "\n".join(c["evidence"] for c in bb.candidates)[:config.MAX_FULLTEXT_CHARS]
        if mode in {"rule_based", "single_agent"}:
            mentions = [Mention(raw=c["raw"], norm=c["norm"], kind=c["kind"],
                                evidence=c["evidence"], contexts=[c["evidence"]],
                                repo_class=c["repo_class"], pos=c["pos"],
                                discovered_by="rules", sources=["rules"]) for c in bb.candidates
                        if c["norm"] != article_key(bb.article_doi or article_id)]
            if mode == "rule_based":
                mentions = [m for m in mentions if distinctive(m.norm, m.kind)
                            and has_data_use(m.evidence, m.norm)]
            else:
                for mention in mentions:
                    if mention.kind == "DOI":
                        mention.metadata = self.registry.dataset_metadata(mention.norm)
                mentions += self._metadata_mentions(bb.article_doi or article_id)
        elif mode == "metadata_only":
            mentions = self._metadata_mentions(bb.article_doi or article_id)
        else:
            mentions = []
            for expert_type in (DOIExpert, AccessionExpert, MetadataExpert):
                expert = expert_type(llm=self.llm, trace=bb.trace, registry=self.registry)
                result = expert.discover(bb)
                mentions.extend(result)
                bb.trace.append(AgentLog(expert.name, {"discovery_count": len(result)}))
        bb.mentions = merge_mentions(mentions)
        bb.trace.append(AgentLog("DiscoveryPool", {
            "raw_text_candidates": len({c["norm"] for c in bb.candidates}),
            "discovery_candidates": len(bb.mentions)}))
        bb.usage = {k: v - before.get(k, 0) for k, v in self._snapshot().items()}
        bb.usage["runtime_seconds"] = time.perf_counter() - started
        return bb

    def _metadata_mentions(self, article_id):
        result = []
        for record in self.registry.related_datasets(article_id):
            norm = normalize_doi(record.get("doi", ""))
            if not norm or norm == article_key(article_id):
                continue
            evidence = bounded("\n".join(str(record.get(k, "")) for k in ("title", "description")))
            result.append(Mention(raw=norm, norm=norm, kind="DOI", evidence=evidence,
                contexts=[evidence], repo_class=repo_class(norm), metadata=record,
                relations=list(record.get("relations", [])), discovered_by="metadata", sources=["metadata"]))
        return result

    def validate_classify(self, discovery: Blackboard, mode="AAMAD") -> Blackboard:
        if mode not in MODES:
            raise ValueError(f"Unknown mode: {mode}")
        started, before = time.perf_counter(), self._snapshot()
        bb = deepcopy(discovery)
        if mode in {"AAMAD", "conventional_debate"}:
            bb.mentions = debate_emit_suppress(bb.mentions, bb, llm=self.llm,
                                              aligned=mode == "AAMAD")
        elif mode == "single_agent":
            from agents.single_agent import SingleAgent
            bb.mentions = SingleAgent(llm=self.llm, trace=bb.trace).validate(bb)
        bb.predictions = [classify(bb, mention,
                           registry=None if mode == "rule_based" else self.registry)
                          for mention in bb.mentions]
        for k, v in self._snapshot().items():
            bb.usage[k] = bb.usage.get(k, 0) + v - before.get(k, 0)
        bb.usage["runtime_seconds"] = bb.usage.get("runtime_seconds", 0) + time.perf_counter() - started
        return bb

    def run_article(self, article_id: str, split: str = "train", *, mode: str | None = None,
                    fulltext: str | None = None, authors=None) -> Blackboard:
        if mode is None:
            mode = ("discovery_only" if config.ABLATION.get("SKIP_DEBATE") else
                    "conventional_debate" if config.ABLATION.get("ANNOTATION_ALIGNED") is False else "AAMAD")
        if mode not in MODES:
            raise ValueError(f"Unknown mode: {mode}")
        if mode == "bare_llm":
            from agents.bare_llm import run
            started, before = time.perf_counter(), self._snapshot()
            text = load_fulltext(article_id, split) if fulltext is None else fulltext
            bb = Blackboard(article_id=article_id, split=split, fulltext=text)
            bb.predictions = run(article_id, text, llm=self.llm, trace=bb.trace)
            bb.usage = {k: v - before.get(k, 0) for k, v in self._snapshot().items()}
            bb.usage["runtime_seconds"] = time.perf_counter() - started
            return bb
        discovery = self.prepare_article(article_id, split, fulltext=fulltext, authors=authors, mode=mode)
        return self.validate_classify(discovery, mode)
