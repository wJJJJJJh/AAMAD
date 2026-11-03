"""Shared evidence. Discovery outputs are not validation votes."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import Any

@dataclass
class AgentLog:
    agent: str
    output: Any
    note: str = ""

@dataclass
class Mention:
    raw: str
    norm: str
    kind: str
    evidence: str = ""
    score: float = 0.0
    repo_class: str = ""
    relation: str = ""
    in_das_section: bool = False
    discovered_by: str = ""
    pos: int | None = None
    contexts: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    relations: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    has_data_ctx: bool = False
    bulk_risk: bool = False

    @property
    def dataset_id(self) -> str:
        return self.norm
    @property
    def confidence(self) -> float:
        return self.score
    @property
    def form(self) -> str:
        return "doi" if self.kind == "DOI" else "accession"

@dataclass
class Prediction:
    dataset_id: str
    type: str
    case: str = ""
    route: str = ""
    confidence: float | None = None

@dataclass
class Blackboard:
    article_id: str
    split: str = "train"
    fulltext: str = ""
    context: str = ""
    authors: tuple[str, ...] = ()
    article_doi: str = ""
    candidates: list[dict] = field(default_factory=list)
    mentions: list[Mention] = field(default_factory=list)
    predictions: list[Prediction] = field(default_factory=list)
    trace: list[AgentLog] = field(default_factory=list)
    usage: dict = field(default_factory=dict)

    def prediction_rows(self) -> list[dict]:
        return [dict(article_id=self.article_id, **asdict(p)) for p in self.predictions]
