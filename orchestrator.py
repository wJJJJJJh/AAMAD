"""编排器：统一使用regex_discovery作为候选池"""
from __future__ import annotations

import config
from state import Blackboard, Mention, Prediction
from tools.regex_discovery import get_deterministic_candidates
from tools.fulltext import load_fulltext
from agents.debate_loop import debate_emit_suppress


class Orchestrator:
    """编排器：发现 → 辩论 → 分类"""

    def __init__(self):
        pass

    def run_article(self, article_id: str, split: str = "train") -> Blackboard:
        """处理单篇文章"""
        bb = Blackboard(article_id=article_id, split=split)

        # 加载全文
        bb.fulltext = load_fulltext(article_id, split)

        # 使用确定性正则发现候选池
        bb.candidates = get_deterministic_candidates(bb.fulltext)

        # 转换为Mention对象
        bb.mentions = self._candidates_to_mentions(bb.candidates)

        # 辩论过滤（如果未禁用）
        if not config.ABLATION.get("SKIP_DEBATE", False):
            bb.mentions = debate_emit_suppress(bb.mentions, bb)

        # 分类（如果未禁用）
        if not config.ABLATION.get("SKIP_CLASSIFY", False):
            for men in bb.mentions:
                bb.predictions.append(self._classify_type(bb, men))
        else:
            # discovery_only模式：直接输出所有候选为Primary
            for men in bb.mentions:
                bb.predictions.append(Prediction(
                    dataset_id=men.norm,
                    type="Primary",
                    case="discovery_only",
                    route="regex"
                ))

        return bb

    def _candidates_to_mentions(self, candidates: list[dict]) -> list[Mention]:
        """将候选字典转换为Mention对象"""
        mentions = []
        for c in candidates:
            mentions.append(Mention(
                raw=c.get('raw', ''),
                norm=c.get('norm', ''),
                kind=c.get('kind', ''),
                evidence=c.get('context', ''),
                score=c.get('score', 0.0),
                discovered_by='regex'
            ))
        return mentions

    def _classify_type(self, bb: Blackboard, men: Mention) -> Prediction:
        """简单分类：默认Primary"""
        # 简化版：所有候选默认Primary
        # 实际实现需要根据evidence和kind进行判断
        return Prediction(
            dataset_id=men.norm,
            type="Primary",
            case="simple_rule",
            route="regex"
        )
