"""Candidate-local evidence and a conservative model-context budget."""
from __future__ import annotations
import math
import re
import config
from tools.candidates import normalize_identifier

GENERATION = re.compile(r"\b(?:generat(?:ed|e|ion)|produc(?:ed|e)|sequenc(?:ed|ing)|deposit(?:ed|ion)|collect(?:ed|ion))\b", re.I)
REUSE = re.compile(r"\b(?:download(?:ed)?|retriev(?:ed|al)|obtain(?:ed)?|acquir(?:ed|e)|reus(?:ed|e)|previously published)\b", re.I)
DAS = re.compile(r"\bdata (?:availability|accessibility|access)\b", re.I)
NEGATION = re.compile(r"\b(?:not|never|without|no)\b.{0,20}$", re.I)

def confidence(value) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) and 0 <= number <= 1 else 0.0

def bounded(text: str, limit: int | None = None) -> str:
    """UTF-8 byte cap, a conservative upper bound for byte-level BPE tokens.

    No approximate word count is advertised as an exact DeepSeek token count.
    """
    limit = config.CONTEXT_TOKEN_LIMIT if limit is None else limit
    return str(text or "").encode("utf-8")[:limit].decode("utf-8", errors="ignore")

def evidence_window(text: str, pos: int, half: int | None = None) -> str:
    half = config.FOCUS_CHARS // 2 if half is None else half
    return bounded(text[max(0, pos - half):pos + half])

def compact(text: str) -> str:
    return " ".join(str(text or "").split()).casefold()

def grounded(quote: str, source: str) -> bool:
    return bool(quote and compact(quote) in compact(source))

def candidate_clauses(text: str, identifier: str) -> list[str]:
    aliases = [identifier, normalize_identifier(identifier).removeprefix("https://doi.org/")]
    # Separate "we generated X and downloaded Y" into candidate-bound assertions.
    clauses = re.split(r"[;\n]|\.\s+|\b(?:and|but|whereas|while)\b(?=\s+(?:we\s+)?(?:download|retriev|generat|produc|deposit|obtain))", text, flags=re.I)
    return [s for s in clauses if any(re.search(r"(?<![A-Za-z0-9])" + re.escape(a) + r"(?![A-Za-z0-9])", s, re.I) for a in aliases if a)]

def positive_cue(pattern: re.Pattern, text: str) -> bool:
    return any(not NEGATION.search(text[max(0, m.start()-25):m.start()]) for m in pattern.finditer(text))

def has_data_use(text: str, identifier: str) -> bool:
    return any(positive_cue(GENERATION, s) or positive_cue(REUSE, s)
               or re.search(r"\b(?:analy[sz]ed|screened)\b.*\b(?:data|database|from)\b", s, re.I)
               or (DAS.search(text) and re.search(r"\b(?:data|dataset)\b.*\b(?:available|accession)\b", s, re.I))
               for s in candidate_clauses(text, identifier))

def pooled_evidence(mention) -> str:
    return bounded("\n".join(dict.fromkeys([mention.evidence] + mention.contexts)))

def metadata_text(mention) -> str:
    return "\n".join(str(mention.metadata.get(k, "")) for k in ("title", "description", "publisher"))

def source_text(bb, mention) -> str:
    return bb.fulltext + "\n" + metadata_text(mention)
