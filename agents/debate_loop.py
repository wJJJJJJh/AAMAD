"""循环辩论系统（方案B：基于双方置信度差距）"""
from __future__ import annotations

import config
from state import Blackboard, Mention
from agents.emit_debate import Prosecutor, Defender, EmitJudge


def debate_emit_suppress(mentions: list[Mention], bb: Blackboard) -> list[Mention]:
    """循环辩论：支持单轮（Conventional）和多轮动态增长（AAMAD）两种模式

    根据config.ABLATION设置选择辩论模式：
    - Conventional: 单轮辩论，固定temperature=0.0, max_tokens=512
    - AAMAD: 多轮辩论，动态增长temperature和max_tokens，最多5轮

    修复版特性：
    1. 收敛时支持平局判断（d_score >= p_score即EMIT，保召回）
    2. 未收敛时检查是否为数据库DOI，优先保留
    3. 增强tie-breaking策略，减少误杀真数据集
    """
    # 根据消融实验配置选择辩论模式
    use_multi_round = config.ABLATION.get("USE_STRUCTURED_DEBATE", False)

    prosecutor = Prosecutor()
    defender = Defender()
    emit_mentions = []

    for men in mentions:
        if use_multi_round:
            # AAMAD模式：多轮动态增长辩论
            decision = _multi_round_debate(prosecutor, defender, bb, men)
        else:
            # Conventional模式：单轮辩论（固定参数）
            decision = _single_round_debate(prosecutor, defender, bb, men)

        if decision == "EMIT":
            emit_mentions.append(men)

    return emit_mentions


def _single_round_debate(prosecutor: Prosecutor, defender: Defender,
                        bb: Blackboard, men: Mention) -> str:
    """单轮辩论（Conventional Debate with Judge）

    使用固定参数：temperature=0.0, max_tokens=512
    流程：Prosecutor → Defender → EmitJudge 裁决
    """
    # 单轮辩论：使用默认参数（temperature=0.0由config.LLM_TEMPERATURE控制）
    prosecution = prosecutor.argue(bb, men, temperature=0.0, max_tokens=512)
    defense = defender.argue(bb, men, prosecution, temperature=0.0, max_tokens=512)

    # 实例化Judge并裁决
    judge = EmitJudge()
    judgment = judge.decide(bb, men, prosecution, defense)

    # 返回Judge的决定
    return judgment.get("decision", "EMIT")  # 默认EMIT保召回


def _multi_round_debate(prosecutor: Prosecutor, defender: Defender,
                       bb: Blackboard, men: Mention) -> str:
    """多轮动态增长辩论（AAMAD）

    - 动态增长temperature: 0.3 → 0.5 → 0.7 → 0.85 → 0.95
    - 动态增长max_tokens: 400 → 500 → 600 → 700 → 800
    - 增量递减策略：前期大步探索，后期精细调整
    - 每轮检查收敛，早收敛早停止
    """
    prosecution = None
    defense = None

    for round_num in range(1, config.DEBATE_MAX_ROUNDS + 1):
        params = config.DEBATE_ROUND_PARAMS[round_num]
        temperature = params["temperature"]
        max_tokens = params["max_tokens"]

        # 本轮辩论（使用动态参数）
        prosecution = prosecutor.argue(bb, men,
                                      temperature=temperature,
                                      max_tokens=max_tokens)
        defense = defender.argue(bb, men, prosecution,
                                temperature=temperature,
                                max_tokens=max_tokens)

        # 统一为"该候选是真实数据集"的0-1置信度
        p_score = 1 - prosecution.get("confidence", 0.5)
        d_score = defense.get("confidence", 0.5)
        gap = abs(p_score - d_score)

        # 收敛判断
        if gap >= config.DEBATE_CONVERGENCE_GAP:
            # 收敛：立即裁决（平局EMIT保召回）
            return "EMIT" if d_score >= p_score else "SUPPRESS"

        # 未收敛：继续下一轮（温度和token都会增加）

    # 达到最大轮次仍未收敛：应用保召回tie-breaking
    return _apply_tie_breaking(men, p_score, d_score)


def _apply_tie_breaking(men: Mention, p_score: float, d_score: float) -> str:
    """应用tie-breaking规则（保召回策略）

    优先级：
    1. 数据仓库DOI：只要p_score >= 0.2就EMIT
    2. 平局：d_score >= p_score则EMIT
    3. 最终保底：根据config.DEBATE_FINAL_TIE_POLICY
    """
    is_database_doi = _is_recognized_data_repository(men)

    if is_database_doi:
        # 数据仓库DOI默认EMIT，除非Prosecutor有压倒性证据
        # p_score是"是数据集"的置信度，越高越应该EMIT
        if p_score >= 0.2:
            return "EMIT"

    if d_score >= p_score:
        # 平局也EMIT（保召回策略）
        return "EMIT"

    if config.DEBATE_FINAL_TIE_POLICY == "EMIT":
        # 最终保底策略
        return "EMIT"

    return "SUPPRESS"


def _is_recognized_data_repository(men: Mention) -> bool:
    """判断候选是否为公认的数据仓库DOI/accession"""

    # 数据仓库DOI特征
    if men.kind == "DOI":
        norm_lower = men.norm.lower()

        # 方法1: 域名关键词匹配
        repo_keywords = [
            "dryad", "zenodo", "figshare", "pangaea",
            "dataverse", "osf.io", "mendeley"
        ]
        if any(kw in norm_lower for kw in repo_keywords):
            return True

        # 方法2: DOI前缀识别（数据仓库专用前缀）
        # 参考: https://www.doi.org/the-identifier/resources/factsheets/doi-resolution-documentation
        data_repo_prefixes = [
            "10.5061",  # Dryad
            "10.5281",  # Zenodo
            "10.6084",  # figshare
            "10.1594",  # PANGAEA
            "10.17882", # SEANOE (海洋数据)
            "10.7910",  # Dataverse (Harvard)
            "10.15146", # DataOne
            "10.5524",  # GigaScience Database
            "10.6019",  # OpenNeuro
        ]
        doi_without_https = norm_lower.replace("https://doi.org/", "").replace("http://doi.org/", "").replace("doi:", "")
        for prefix in data_repo_prefixes:
            if doi_without_https.startswith(prefix.lower()):
                return True

    # 公共数据库accession特征
    db_kinds = {
        "GEO-series", "GEO-sample", "SRA-study", "SRA-experiment",
        "SRA-run", "BioProject", "BioSample", "PDB", "GenBank",
        "RefSeq", "UniProt", "EMBL", "Ensembl"
    }
    if men.kind in db_kinds:
        return True

    return False

