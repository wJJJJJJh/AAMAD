"""One identifier vocabulary for discovery, deduplication and evaluation.

Normalization changes spelling, not dataset identity: distinct version DOIs and
numeric DOI suffixes are preserved.
"""
from __future__ import annotations
import re
from urllib.parse import unquote

DOI_PATTERN = re.compile(r"\b10\.\d{4,9}/\s*[^\s<>\"']+", re.I)
ACCESSION_PATTERNS = [
    (r"GSE\d+", "geo_series"), (r"GSM\d+", "geo_sample"), (r"GPL\d+", "geo_platform"),
    (r"[SED]RP\d+", "sra_study"), (r"[SED]RR\d+", "sra_run"),
    (r"[SED]RX\d+", "sra_experiment"), (r"PRJ[NED][A-Z]\d+", "bioproject"),
    (r"SAM[NED][A-Z]?\d+", "biosample"), (r"E-[A-Z]{4}-\d+", "arrayexpress"),
    (r"PXD\d+", "pride"), (r"EMPIAR-\d+", "empiar"), (r"MTBLS\d+", "metabolights"),
    (r"phs\d+(?:\.v\d+\.p\d+)?", "dbgap"), (r"EGA[SD]\d+", "ega"),
    (r"CHEMBL\d+", "chembl"), (r"IPR\d{6}", "interpro"), (r"PF\d{5}", "pfam"),
    (r"ENS[A-Z]{0,3}[GPTR]\d{11}(?:\.\d+)?", "ensembl"),
    (r"(?:NC|NG|NM|NP|NR|NT|NW|NZ|XM|XP|XR|YP|ZP)_\d+(?:\.\d+)?", "refseq"),
    (r"[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2}", "uniprot"),
    (r"rs\d+", "dbsnp"), (r"[A-Z]{1,2}\d{5,8}(?:\.\d+)?", "genbank"),
    (r"[0-9][A-Z0-9]{3}", "pdb"),
]
ENUMERABLE_SUBRECORD_KINDS = {"sra_run", "sra_experiment", "geo_sample", "biosample"}
PARENT_KINDS = {"geo_series", "sra_study", "pride", "empiar", "metabolights",
                "arrayexpress", "bioproject", "dbgap", "ega"}
RISKY_KINDS = {"pdb", "uniprot", "genbank", "refseq", "dbsnp", "ensembl", "pfam", "chembl", "interpro"}
UNAMBIGUOUS_DB_KINDS = {k for _, k in ACCESSION_PATTERNS} - RISKY_KINDS
NATIVE_PREFIXES = ("10.5061/", "10.1594/", "10.15468/", "10.6073/", "10.3886/",
                   "10.7937/", "10.5066/", "10.17882/", "10.5524/", "10.17182/")
MIXED_PREFIXES = ("10.5281/", "10.6084/", "10.7910/", "10.17632/", "10.17605/")
REPO_PREFIXES = {"geo": "GEO", "sra": "SRA", "pdb": "PDB", "chembl": "ChEMBL"}
MAX_CANDIDATES = None  # No silent candidate cap.

def normalize_doi(value: str) -> str:
    s = unquote(str(value or "")).strip().replace("\u00ad", "")
    s = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*|doi\.org/)", "", s, flags=re.I)
    s = re.sub(r"\s+", "", s).rstrip(".,;:")
    for left, right in [("(", ")"), ("[", "]"), ("{", "}")]:
        while s.endswith(right) and s.count(right) > s.count(left):
            s = s[:-1]
    if not re.fullmatch(r"10\.\d{4,9}/\S+", s, re.I):
        return ""
    return "https://doi.org/" + s.lower()

def normalize_identifier(value: str) -> str:
    s = str(value or "").strip()
    doi = normalize_doi(s)
    if doi:
        return doi
    # Accept common repository labels and landing-page URLs without changing
    # arbitrary opaque identifiers or choosing one ID from a multi-ID record.
    if re.match(r"https?://|(?:GEO|PDB|SRA|PRIDE|UniProt|GenBank|ChEMBL)\s*[: ]", s, re.I):
        found = set()
        for pattern, kind in ACCESSION_PATTERNS:
            for match in re.finditer(r"(?<![A-Za-z0-9_])(?:" + pattern + r")(?![A-Za-z0-9_])", s, re.I):
                if kind != "pdb" or not match.group().isdigit():
                    found.add(match.group().upper())
        if len(found) == 1:
            return next(iter(found))
    return re.sub(r"\s+", "", s).strip(".,;:()[]{}").upper()

def article_key(value: str) -> str:
    s = str(value or "").strip()
    # The MDC download uses DOI slashes replaced by an underscore in file names.
    s = re.sub(r"^(10\.\d{4,9})_", r"\1/", s, flags=re.I)
    return normalize_doi(s) or s

def fold_fragment_doi(doi: str, present: set[str] | None = None) -> str:
    """Compatibility wrapper. Version and numeric suffixes must not be erased."""
    return normalize_doi(doi)

def accession_kind(value: str) -> str:
    s = normalize_identifier(value)
    for pattern, kind in ACCESSION_PATTERNS:
        if re.fullmatch(pattern, s, re.I):
            if kind != "pdb" or not s.isdigit():
                return kind
    return "unknown"

def repo_class(value: str) -> str:
    doi = normalize_doi(value)
    if doi:
        core = doi.removeprefix("https://doi.org/")
        if core.startswith(NATIVE_PREFIXES):
            return "dataset_native"
        if core.startswith("10.5517/"):
            return "supplementary"
        return "mixed"
    kind = str(value).lower().replace("-", "_")
    if kind in {k for _, k in ACCESSION_PATTERNS} or accession_kind(value) != "unknown":
        return "reuse_db"
    return "mixed"

def distinctive(value: str, kind: str = "") -> bool:
    if normalize_doi(value):
        return repo_class(value) == "dataset_native"
    canonical = kind.lower().replace("-", "_") or accession_kind(value)
    return canonical in UNAMBIGUOUS_DB_KINDS

def extract_candidates(text: str) -> list[dict]:
    candidates, doi_spans, occupied = [], [], set()
    # Repair whitespace immediately after a DOI slash without joining sentences.
    for match in DOI_PATTERN.finditer(text):
        raw = match.group(0)
        norm = normalize_doi(raw)
        if norm:
            doi_spans.append(match.span())
            candidates.append(dict(raw=raw, norm=norm, kind="DOI", form="doi",
                                   pos=match.start(), score=0.0, repo_class=repo_class(norm)))
    for pattern, kind in ACCESSION_PATTERNS:
        for match in re.finditer(r"(?<![A-Za-z0-9_])(?:" + pattern + r")(?![A-Za-z0-9_])", text, re.I):
            if any(a <= match.start() < b for a, b in doi_spans):
                continue
            if any(pos in occupied for pos in range(*match.span())):
                continue
            raw = match.group()
            if kind == "pdb" and raw.isdigit():
                continue
            occupied.update(range(*match.span()))
            candidates.append(dict(raw=raw, norm=normalize_identifier(raw), kind=kind,
                                   form="accession", pos=match.start(), score=0.0,
                                   repo_class="reuse_db"))
    return sorted(candidates, key=lambda c: (c["pos"], c["norm"]))

def rank_candidates(candidates: list[dict], fulltext: str) -> list[dict]:
    from tools.evidence import evidence_window, has_data_use
    out = []
    for c in candidates:
        c = dict(c)
        c["evidence"] = evidence_window(fulltext, c["pos"])
        c["has_data_ctx"] = has_data_use(c["evidence"], c["norm"])
        out.append(c)
    return out

__all__ = ["normalize_doi", "normalize_identifier", "article_key", "fold_fragment_doi",
           "extract_candidates", "rank_candidates", "accession_kind", "repo_class",
           "distinctive", "ACCESSION_PATTERNS", "ENUMERABLE_SUBRECORD_KINDS",
           "PARENT_KINDS", "RISKY_KINDS", "UNAMBIGUOUS_DB_KINDS", "REPO_PREFIXES"]
