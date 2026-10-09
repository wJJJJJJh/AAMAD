"""裸 LLM 基线（bare_llm 消融臂）：单次调用直接抽 + 分类，无专家/规则/辩论。

用于消融对比：证明结构化管线（三专家 + 规则门控 + 辩论）vs 单次 LLM 的差距。
该臂跳过整个 Orchestrator 流程，直接把全文喂给一个 prompt，要求返回数据集列表。"""
from __future__ import annotations

from agents.base import BaseAgent
from state import Blackboard, Prediction

_SYSTEM = """You extract dataset citations from a scientific paper and classify each as Primary or Secondary.

Primary: data produced/generated/first released by this paper's authors in this study.
Secondary: data reuse/citation of pre-existing data released by others or previously.

Extract:
- DOIs from data repositories (Dryad, Zenodo, figshare, PANGAEA, etc.)
- Database accession numbers (GEO, SRA, BioProject, PDB, UniProt, GenBank, etc.)

Output a JSON list of datasets found:
{"datasets":[{"dataset_id":"<normalized DOI or accession>","type":"Primary|Secondary","confidence":0.0}]}

If no datasets, return {"datasets":[]}.
"""


class BareLLM(BaseAgent):
    """单次 LLM 调用完成全任务（抽取 + 分类），无任何结构化辅助。"""
    name = "BareLLM"
    system = _SYSTEM

    def run_article(self, bb: Blackboard) -> None:
        """一次调用返回该文章的全部预测（跳过 Orchestrator 三阶段管线）。"""
        if not bb.fulltext:
            return
        # 截断全文到 LLM 上下文预算内（粗暴截断，无聚焦视图）
        import config
        text = bb.fulltext[:config.MAX_FULLTEXT_CHARS * 3]  # 给裸 LLM 更多预算
        user = f"# article_id: {bb.article_id}\n# Full text:\n{text}\n\nExtract datasets."
        out = self._decide(user, max_tokens=4096, note="裸 LLM 单次调用抽取+分类")
        raw = out.get("datasets", []) if isinstance(out, dict) else []
        for item in raw:
            if not isinstance(item, dict):
                continue
            did = str(item.get("dataset_id", "")).strip()
            typ = str(item.get("type", "")).strip()
            if did and typ in ("Primary", "Secondary"):
                bb.predictions.append(Prediction(
                    dataset_id=did, type=typ, case="bare_llm", route="bare_llm",
                    confidence=float(item.get("confidence", 0) or 0)))
