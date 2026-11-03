"""One sequential Prosecutor -> Defender -> Judge exchange (Section 3.4)."""
from __future__ import annotations
import json
import re
import config
from agents.base import BaseAgent
from tools.evidence import confidence, grounded, pooled_evidence, metadata_text

DATASET_STANDARD = """Annotation standard: a resource supplies valid data if the
article produced, analyzed or reused its data. Database accessions are not invalid
merely because they denote compounds, structures, proteins or other entities.
A large number of accessions alone is not a disqualifying red flag.
Disqualifying red_flag_type values: publication, software, non_data_object,
context_free_identifier. Use none when no such flag is established.
All evidence must be verbatim from the supplied local context or metadata.
"""

def _context(bb, mention) -> dict:
    return {"article_id": bb.article_id, "article_doi": bb.article_doi,
            "dataset_id": mention.norm, "kind": mention.kind,
            "evidence": pooled_evidence(mention), "metadata": mention.metadata,
            "registered_relations": mention.relations}

def evidence_source(mention) -> str:
    return pooled_evidence(mention) + "\n" + metadata_text(mention) + "\n" + json.dumps(mention.relations, ensure_ascii=False)

class Prosecutor(BaseAgent):
    name = "Prosecutor"
    system = DATASET_STANDARD + """
Examine the candidate for a concrete disqualifying red flag. Be adversarial but
concede EMIT if no red flag is supported. confidence always measures confidence
in SUPPRESSION, including when conceding EMIT (then it should be low).
Return JSON: verdict (EMIT or SUPPRESS), red_flag_type, red_flag (description or
none), evidence (verbatim supporting quote), confidence (0 to 1).
"""
    def argue(self, bb, mention, **kwargs) -> dict:
        return self._decide(json.dumps(_context(bb, mention), ensure_ascii=False),
                            max_tokens=config.LLM_MAX_TOKENS, note="Prosecution")

class Defender(BaseAgent):
    name = "Defender"
    system = DATASET_STANDARD + """
Read the same candidate/evidence AND the Prosecutor's complete argument.
Rebut unsupported exclusions using candidate-specific data-generation/reuse
evidence. Concede when an actual non-data object is established.
Return JSON: verdict (EMIT or SUPPRESS), rebuttal, evidence (verbatim), confidence.
"""
    def argue(self, bb, mention, prosecution: dict, **kwargs) -> dict:
        payload = _context(bb, mention)
        payload["prosecution"] = prosecution
        return self._decide(json.dumps(payload, ensure_ascii=False),
                            max_tokens=config.LLM_MAX_TOKENS, note="Defense")

class EmitJudge(BaseAgent):
    name = "Judge"
    system = DATASET_STANDARD + """
Review the candidate, pooled discovery evidence and BOTH arguments. Prioritize
verbatim evidence of study data generation or reuse. Suppression requires an
explicit, supported disqualifying red flag with suppression confidence STRICTLY
ABOVE 0.85. Otherwise EMIT to protect recall. A confidence of exactly 0.85 does
not qualify. Do not use a confidence gap or the largest confidence as a vote.
Uphold a Prosecutor red flag only if its quote really establishes an exclusion,
after considering the Defender's rebuttal.
Return JSON: decision (EMIT or SUPPRESS), red_flag_supported (boolean), reason,
confidence (confidence in the final decision, NOT the red-flag threshold).
"""
    def __init__(self, llm=None, trace=None, aligned=True):
        super().__init__(llm, trace)
        self.aligned = aligned

    def decide(self, bb, mention, prosecution: dict, defense: dict) -> dict:
        payload = _context(bb, mention)
        payload.update(prosecution=prosecution, defense=defense)
        answer = self._decide(json.dumps(payload, ensure_ascii=False),
                              max_tokens=config.LLM_MAX_TOKENS, note="Arbitration")
        decision = str(answer.get("decision", "EMIT")).upper()
        if decision not in {"EMIT", "SUPPRESS"}:
            decision = "EMIT"
        result = dict(answer, model_decision=decision)
        if self.aligned and decision == "SUPPRESS":
            flag_type = str(prosecution.get("red_flag_type", "")).lower()
            flag = str(prosecution.get("red_flag", "")).strip().lower()
            quote = str(prosecution.get("evidence", ""))
            qualifies = (
                str(prosecution.get("verdict", "")).upper() == "SUPPRESS"
                and flag_type in {"publication", "software", "non_data_object", "context_free_identifier"}
                and flag not in {"", "none", "null", "n/a"}
                and confidence(prosecution.get("confidence")) > config.SUPPRESS_CONFIDENCE_THRESHOLD
                and grounded(quote, evidence_source(mention))
                and answer.get("red_flag_supported") is True
            )
            if not qualifies:
                decision = "EMIT"
                result["reason"] = "No supported disqualifying red flag above 0.85. Retained to protect recall."
                result["policy_override"] = True
        result["decision"] = decision
        result["confidence"] = confidence(answer.get("confidence", 0))
        return result

# Conventional Debate intentionally lacks the annotation-aligned definition and
# the asymmetric suppression gate. Its order, model and temperature stay fixed.
CONVENTIONAL_PROSECUTOR = """You are the Prosecutor in adversarial dataset-reference
validation. Examine the candidate under your general understanding of research
datasets and argue for rejection when justified. Return JSON: verdict (EMIT or
SUPPRESS), red_flag_type, red_flag, evidence, confidence."""
CONVENTIONAL_DEFENDER = """You are the Defender in adversarial dataset-reference
validation. Read the Prosecutor argument and defend the candidate or concede.
Return JSON: verdict (EMIT or SUPPRESS), rebuttal, evidence, confidence."""
CONVENTIONAL_JUDGE = """You are the Judge in adversarial dataset-reference validation.
Review both arguments and the candidate evidence using general dataset criteria.
Return JSON: decision (EMIT or SUPPRESS), reason, confidence."""
