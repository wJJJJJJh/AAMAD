"""Tool-augmented single agent using the same tools and provenance classifier."""
import json
import config
from agents.base import BaseAgent
from agents.experts import DISCOVERY_STANDARD
from tools.candidates import normalize_identifier
from tools.evidence import pooled_evidence, metadata_text

class SingleAgent(BaseAgent):
    name = "ToolAugmentedSingleAgent"
    system = DISCOVERY_STANDARD + """
You are the only agent. Use the supplied identifier candidates, local text,
registered metadata relations and repository evidence to validate dataset references.
Return every accepted candidate in mentions with dataset_id and verdict EMIT.
Return rejected candidates with verdict SUPPRESS."""

    def validate(self, bb):
        accepted = []
        for start in range(0, len(bb.mentions), config.DISCOVERY_BATCH_SIZE):
            batch = bb.mentions[start:start + config.DISCOVERY_BATCH_SIZE]
            payload = [{"dataset_id": m.norm, "evidence": pooled_evidence(m),
                        "metadata": m.metadata, "relations": m.relations} for m in batch]
            answer = self._decide(json.dumps({"article_id": bb.article_id, "candidates": payload},
                                            ensure_ascii=False))
            rows = answer.get("mentions", [])
            if not isinstance(rows, list):
                raise ValueError("Single-agent mentions must be a list")
            selected = {normalize_identifier(r.get("dataset_id", "")) for r in rows
                        if isinstance(r, dict) and str(r.get("verdict", "EMIT")).upper() == "EMIT"}
            accepted.extend(m for m in batch if m.norm in selected)
        return accepted
