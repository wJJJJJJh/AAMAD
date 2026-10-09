"""状态类：Blackboard、Mention、Prediction"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

@dataclass
class AgentLog:
    """智能体执行日志"""
    agent: str
    output: Any
    note: str = ""

@dataclass
class Mention:
    """候选数据集（发现阶段输出）"""
    raw: str                    # 原始形式
    norm: str                   # 归一化形式
    kind: str                   # 类型（DOI, GEO-series, SRA-study等）
    evidence: str = ""          # 上下文证据
    score: float = 0.0          # 初始评分
    repo_class: str = ""        # dataset_native/mixed/supplementary
    relation: str = ""          # DataCite关系类型
    in_das_section: bool = False  # 是否在DAS段落
    discovered_by: str = ""     # 发现来源

@dataclass
class Prediction:
    """最终预测输出"""
    dataset_id: str
    type: str                   # Primary/Secondary
    case: str = ""              # 分类路径
    route: str = ""             # 发现路由
    confidence: float = 0.0

@dataclass
class Blackboard:
    """黑板：共享状态"""
    article_id: str
    split: str = "train"
    fulltext: str = ""
    candidates: list[dict] = field(default_factory=list)
    mentions: list[Mention] = field(default_factory=list)
    predictions: list[Prediction] = field(default_factory=list)
    trace: list[AgentLog] = field(default_factory=list)
