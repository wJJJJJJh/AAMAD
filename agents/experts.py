"""异构发现专家
  - DOIExpert       —— 数据仓库 DOI；dataset_native+data_ctx 规则直采（零 LLM），
                        mixed/supplementary 交 LLM 甄别；纯参考文献 DOI 前置排除；
  - AccessionExpert —— 数据库 accession；无歧义库确定性直采，风险库交 LLM；
  - MetadataExpert  —— DataCite 反查，突破正文抽取召回天花板（可选、离线安全）。

每条产出都是 state.Mention（带 evidence 锚定 + prior_score/tier 成本标注）
"""
from __future__ import annotations

import config
from agents.base import BaseAgent
from state import AgentLog, Mention
from tools import candidates as cand
from tools import doi_registry as reg

_HALF = 300   # evidence 窗口半径


def _mk_mention(c: dict, discovered_by: str, conf: float,
                evidence: str = "") -> Mention:
    """把带标注的候选 dict 折成 Mention（统一锚定证据与成本先验）。"""
    did = c.get("norm") or c.get("raw", "")
    kind = c.get("kind", "")
    form = "doi" if kind == "DOI" else "accession"
    return Mention(
        dataset_id=did, form=form, kind=kind,
        evidence=evidence or c.get("evidence", ""),
        pos=int(c.get("pos", 0) or 0), confidence=conf,
        discovered_by=discovered_by,
        prior_score=float(c.get("score", 0) or 0),
        repo_class=c.get("repo_class", ""),
        risky=kind in cand.RISKY_KINDS,
        has_data_ctx=bool(c.get("has_data_ctx", False)),
        bulk_risk=c.get("bulk_risk"))  # 传递批量风险标注


class DOIExpert(BaseAgent):
    """数据仓库 DOI 专家。

    成本-精度分层：
      - 纯参考文献区 DOI → 前置排除（被引论文，非数据集），零 LLM；
      - dataset_native + 邻近数据语境 → 规则直采（高置信，零 LLM）；
      - 其余（mixed/supplementary/无语境）→ 交 LLM 结合仓库类别甄别。
    """
    name = "DOIExpert"
    system = (
        "You are an expert at detecting **research dataset** citations that appear "
        "as repository DOIs in a scientific paper.\n\n"
        "Repository classes you will be told about per candidate:\n"
        "- dataset_native: repositories that almost exclusively host datasets "
        "(Dryad 10.5061, PANGAEA 10.1594, GBIF 10.15468, EDI 10.6073, ICPSR, "
        "USGS ScienceBase, TCIA, SEANOE). A DOI here is very likely a dataset.\n"
        "- mixed: general repositories that host datasets AND figures, slides, "
        "software, posters, supplementary PDFs (figshare 10.6084, Zenodo 10.5281, "
        "OSF, Dataverse, Mendeley Data). Only accept as a dataset when the text "
        "shows the object is actual research DATA the paper uses or produced; "
        "reject if it is a figure, slide deck, software, or supplementary PDF.\n"
        "- supplementary: domain deposits that are usually supplementary material "
        "rather than an independently citable dataset (e.g. CCDC 10.5517 crystal "
        "structures). Accept only if the paper explicitly frames it as a reused "
        "dataset.\n\n"
        "STRICT EXCLUSIONS (never emit these as datasets):\n"
        "1. DOIs of cited papers / journal articles (references, related work).\n"
        "2. Software / code DOIs, and figure/slide/poster/supplementary-PDF DOIs.\n"
        "3. A DOI mentioned only in passing background with no indication the paper "
        "uses or produced that data.\n\n"
        "Few-shot examples (drawn from real annotated articles):\n"
        "POSITIVE (accept):\n"
        "- 'DATA ACCESSIBILITY The dataset supporting this article are available "
        "in Dryad https://doi.org/10.5061/dryad.r6nq870' → dataset_native repo + "
        "explicit data-availability statement → dataset. ACCEPT.\n"
        "- 'Data accessibility Repository name: PANGAEA ... 10.1594/PANGAEA.949117' "
        "→ dataset_native repo + data availability → dataset. ACCEPT.\n\n"
        "NEGATIVE (reject):\n"
        "- 'Crystal data deposited: CCDC 2012345-2012389' (45 sequential entries) "
        "→ batch crystallographic auto-deposits, not independent datasets. REJECT.\n"
        "- Dozens of near-identical 'https://doi.org/10.5517/ccdc.csd.cc24d9c7' "
        "CCDC/CSD entries → supplementary crystal-structure deposits. REJECT.\n"
        "- 'Additional figures are provided on figshare (10.6084/m9.figshare.xxxx)' "
        "→ figures on a mixed repo, not research data. REJECT.\n"
        "- 'Raw data for Figure 3 available at 10.5281/zenodo.xxxx' → if this is "
        "SOURCE FILES FOR A FIGURE rather than the analyzed measurements, REJECT.\n"
        "- 'Smith et al. (2019) https://doi.org/10.1038/xxxx reported similar "
        "trends' → cited paper in references. REJECT.\n\n"
        "**RECALL PRIORITY (修复增强)**: When evaluating mixed/supplementary repository DOIs, "
        "lean toward ACCEPT unless there is clear evidence it is NOT a dataset (e.g., "
        "explicit 'figure', 'slides', 'code', or cited paper in references section). "
        "Many legitimate datasets live in mixed repositories like Zenodo, figshare, OSF. "
        "Confidence calibration:\n"
        "- Zenodo/figshare/OSF DOI with data-related keywords ('data', 'dataset', 'measurements') "
        "→ confidence ≥0.6 even without explicit Data Availability statement.\n"
        "- Even weak data context ('available at...', 'deposited') → confidence ≥0.5.\n"
        "- Multiple DOIs from the same repository are often legitimate (e.g., 10 Zenodo datasets "
        "from a large study) - do NOT reject just because there are many.\n"
        "- Only reject with high confidence (≥0.8) for clear non-datasets (figures, software, "
        "cited papers).\n\n"
        "Remember: **Missing a dataset is worse than keeping some noise**. The debate layer "
        "will filter false positives.\n\n"
        "For each accepted DOI output: dataset_doi (normalised https://doi.org/... "
        "form), evidence (verbatim supporting snippet from the text), confidence "
        "(0-1). Empty list if none.\n"
        'Output JSON: {"mentions":[{"dataset_doi":"","evidence":"","confidence":0.0}]}'
    )

    def discover(self, bb) -> list[Mention]:
        doi_cands = [c for c in bb.candidates if c.get("kind") == "DOI"]
        if not doi_cands:
            return []

        def _is_bulk_deposit(cands: list[dict]) -> set[int]:
            """识别成批自动沉积的 DOI 噪声（如 CCDC 晶体结构几十条连号）。

            关键修复（对齐 accession 侧教训）：批量排除**仅对 supplementary 生效**。
            dataset_native（Dryad/PANGAEA/GBIF/Zenodo…）与 mixed 一篇合法引用数十个
            数据集是真信号，绝不因"同前缀数量多"整批误杀——它们仍交下游规则直采/LLM
            甄别。此前对 dataset_native 也套 >8 阈值，是 accession bulk 一刀切 bug 的
            DOI 镜像（会误杀引用大量 Dryad/Zenodo 的文章）。"""
            from collections import Counter
            if len(cands) < 5:
                return set()

            prefix_counts = Counter()
            cand_by_prefix = {}
            for c in cands:
                norm = c.get("norm", "")
                if not norm.startswith("https://doi.org/10."):
                    continue
                parts = norm.replace("https://doi.org/", "").split("/")
                if len(parts) >= 1:
                    prefix = parts[0]  # e.g., "10.5517", "10.18150"
                    prefix_counts[prefix] += 1
                    cand_by_prefix.setdefault(prefix, []).append(c)

            # 只有 supplementary（CCDC 等已知批量自动沉积库）且超过阈值才整批排除。
            # 【修复】使用config中的BULK_FILTER_THRESHOLD（默认50），大幅减少误杀
            bulk_ids = set()
            for prefix, cnt in prefix_counts.items():
                is_supp = all(c.get("repo_class") == "supplementary"
                              for c in cand_by_prefix[prefix])
                if is_supp and cnt > config.BULK_FILTER_THRESHOLD:
                    for c in cand_by_prefix[prefix]:
                        bulk_ids.add(id(c))

            return bulk_ids

        bulk_ids = _is_bulk_deposit(doi_cands)
        n_bulk = len(bulk_ids)
        doi_cands = [c for c in doi_cands if id(c) not in bulk_ids]

        # 前置排除：仅见于参考文献区的 DOI（被引论文）
        screened = [c for c in doi_cands if not c.get("in_refs_only", False)]

        # 放宽直接采纳条件：降低score阈值，不强制要求has_data_ctx
        # 让更多候选进入后续流程（规则采纳或LLM确认），辩论层负责最终过滤
        # 【修复】进一步降低阈值：4.5→3.0→2.0，提高召回
        direct = [c for c in screened
                  if c.get("repo_class") == "dataset_native"
                  and c.get("direct_adopt_ok")
                  and float(c.get("score", 0) or 0) >= 2.0]  # 从3.0降到2.0
        direct_ids = {id(c) for c in direct}
        to_llm = [c for c in screened if id(c) not in direct_ids]

        # 【修复】直接采纳的候选初始confidence降低（0.75→0.65），让部分送辩论
        out: list[Mention] = [_mk_mention(c, self.name, 0.65) for c in direct]

        raw: list[dict] = []
        if to_llm:
            lines = []
            for c in to_llm[:60]:
                rc = c.get("repo_class", "other")
                ctx = "yes" if c.get("has_data_ctx") else "no"
                lines.append(f"- {c['raw']} | class={rc} | data-context={ctx}")
            user = (f"# Paper article_id: {bb.article_id}\n"
                    f"# Candidate DOIs (repository class + data-availability context):\n"
                    + "\n".join(lines) +
                    "\n\n# Focused full-text view:\n" + bb.context)
            res = self._decide(user, max_tokens=4096,
                               note="mixed/supplementary DOI 交 LLM 甄别")
            raw = (res or {}).get("mentions", []) if isinstance(res, dict) else []
        by_norm = {cand.normalize_doi(c.get("norm") or c.get("raw", "")): c
                   for c in to_llm}
        for m in raw:
            if not isinstance(m, dict) or not m.get("dataset_doi"):
                continue
            nid = cand.normalize_doi(m["dataset_doi"])
            base = by_norm.get(nid, {"norm": nid, "kind": "DOI"})
            out.append(_mk_mention(base, self.name,
                                   float(m.get("confidence", 0) or 0),
                                   evidence=m.get("evidence", "")))
        bb.trace.append(AgentLog(agent=self.name, output={
            "n_in": len(doi_cands) + n_bulk, "n_bulk_excluded": n_bulk,
            "n_ref_only": len(doi_cands) - len(screened),
            "n_direct": len(direct), "n_llm": len(to_llm), "n_out": len(out)},
            note="DOIExpert：批量沉积前置排除；参考文献DOI排除；原生仓库+强数据语境+高分规则直采，余交LLM"))
        return out


class AccessionExpert(BaseAgent):
    """数据库 accession 专家。accession 本身即可评分 id（form-based 直采）。

    无歧义库（GEO/SRA/BioProject/PRIDE/EMPIAR… 语法唯一）确定性直采；
    风险库（PDB/GenBank/UniProt/Ensembl/Pfam，短/泛前缀）交 LLM 结合语境甄别。
    """
    name = "AccessionExpert"
    system = (
        "You are an expert at detecting database **accession numbers** that denote "
        "datasets. You are given candidate IDs whose **format matches but needs "
        "context confirmation** (mostly short prefixes like PDB/GenBank/UniProt/"
        "Ensembl/Pfam, easily confused with gene names, primers, or figure/table "
        "labels). Keep only IDs the paper actually cites/uses as a dataset; drop "
        "non-dataset IDs and regex false hits.\n\n"
        "Common database kinds you will see:\n"
        "- Genomic/transcriptomic: GEO (GSE*/GSM*), SRA (SRR*/SRP*/SRX*), "
        "BioProject (PRJNA*/PRJEB*), ArrayExpress (E-MTAB-*)\n"
        "- Protein/structure: PDB (4-char alphanumeric), PRIDE (PXD*), EMPIAR (EMPIAR-*)\n"
        "- Sequence: GenBank (often 1-2 letters + 5-6 digits), UniProt (P*/Q*/O*)\n"
        "- Gene/protein families: Ensembl (ENSG*/ENST*), Pfam (PF*), InterPro (IPR*)\n\n"
        "STRICT EXCLUSIONS (never emit these as datasets):\n"
        "1. Bare gene/protein NAMES with no database context (e.g., 'TP53', 'BRCA1').\n"
        "2. Regex false hits: sample codes, figure/table labels, equation variables.\n"
        "3. Primer names, plasmid IDs, or reagent catalog numbers.\n"
        "4. Bulk enumerable subrecords (dozens of SRA runs, BioSample IDs) unless "
        "explicitly framed as the study's deposited data.\n\n"
        "Few-shot examples (drawn from real annotated articles):\n"
        "POSITIVE (keep):\n"
        "- '30 Spanish (IBS) files downloaded from GEO data set GSE67047' → "
        "reused dataset with explicit data-retrieval context. KEEP, confidence=0.95.\n"
        "- 'Raw sequencing data deposited to SRA under BioProject PRJNA123456' → "
        "newly deposited primary data. KEEP, confidence=0.95.\n"
        "- 'Protein structures were retrieved from PDB entries 1A2B, 3XYZ, 4QRS' → "
        "reused structural data cited as analysis source. KEEP, confidence=0.9.\n"
        "- 'ChEMBL database (accession CHEMBL1234567) provided compound bioactivity' → "
        "reused compound dataset. KEEP, confidence=0.9.\n\n"
        "NEGATIVE (drop):\n"
        "- A four-letter token like '1A2B' in 'Figure 1A2B shows...' or as an "
        "equation label with no PDB/database context → not a dataset. DROP.\n"
        "- 'We analyzed the TP53 gene' (bare gene name, no accession) → gene name, "
        "not a dataset ID. DROP.\n"
        "- 'Primers F1/R1 amplified a 500bp fragment' where 'F1' matches a regex "
        "but is clearly a primer name → false hit. DROP.\n"
        "- Paper lists 80 SRA run IDs (SRR123001, SRR123002, ..., SRR123080) with "
        "no context about depositing/reusing them as a collection → likely enumerable "
        "subrecords, not independent datasets. DROP (or flag low confidence=0.3).\n\n"
        "Confidence calibration:\n"
        "- 0.9-1.0: Explicit data-availability statement or reuse declaration.\n"
        "- 0.7-0.9: Clear database context + accession cited as data source.\n"
        "- 0.5-0.7: Accession mentioned with weak context (e.g., 'similar to PDB 1ABC').\n"
        "- <0.5: Ambiguous or bulk subrecord risk; flag for debate.\n\n"
        "**RECALL PRIORITY (修复增强)**: When in doubt between keeping vs dropping a candidate, "
        "ERR ON THE SIDE OF KEEPING. False positives (keeping noise) can be filtered "
        "later in the debate layer, but false negatives (dropping true datasets) are "
        "irreversible. Confidence calibration guidelines:\n"
        "- If the ID has ANY database context (even weak), assign confidence ≥0.5.\n"
        "- If the ID appears in a Data Availability section, assign confidence ≥0.7.\n"
        "- Multiple accessions of the same type (e.g., 15 ChEMBL compounds, 20 PDB structures) "
        "are often LEGITIMATE in biology/chemistry papers - do NOT drop them just because "
        "there are many. Each can be an independent dataset.\n"
        "- Only assign confidence <0.4 for clear false hits (figure labels, gene names "
        "with zero database context).\n\n"
        "Remember: **Recall > Precision at this stage**. The debate layer will handle "
        "precision refinement.\n\n"
        "For each kept ID output: dataset_id (copy the ID verbatim), evidence "
        "(verbatim supporting snippet), confidence (0-1). Empty list if none.\n"
        'Output JSON: {"mentions":[{"dataset_id":"","evidence":"","confidence":0.0}]}'
    )

    def discover(self, bb) -> list[Mention]:
        acc = [c for c in bb.candidates if c.get("kind") != "DOI"]
        if not acc:
            return []

        def _annotate_bulk_risk(cands: list[dict]) -> None:
            """标注批量风险特征，但不排除候选（延迟决策到辩论层）。

            批量判断不再是单个Expert的独裁决策，而是标注风险特征供辩论层协作判断。
            【修复】阈值从6提高到20/30，减少误杀合法的多数据集引用：
              - ENUMERABLE_SUBRECORD（BioSample/SRA-run/GenBank子记录）：阈值30
              - 独立可评分库（EMPIAR/ChEMBL/InterPro等）：阈值20
            辩论层的RecallAdvocate和Judge会结合DAS段位置、语境等综合判断。
            """
            from collections import Counter
            if len(cands) < 5:
                return

            kind_counts = Counter()
            cand_by_kind = {}
            for c in cands:
                k = c.get("kind", "unknown")
                if k != "DOI":
                    kind_counts[k] += 1
                    cand_by_kind.setdefault(k, []).append(c)

            # 标注批量风险（不排除）【修复】大幅提高阈值
            for k, cnt in kind_counts.items():
                # 根据类型使用不同阈值（config.BULK_FILTER_THRESHOLD默认50）
                threshold = config.BULK_FILTER_THRESHOLD if k in cand.ENUMERABLE_SUBRECORD_KINDS else (config.BULK_FILTER_THRESHOLD - 10)
                if cnt > threshold:
                    risk_level = "high" if k in cand.ENUMERABLE_SUBRECORD_KINDS else "low"
                    for c in cand_by_kind[k]:
                        c['bulk_risk'] = {
                            'kind': k,
                            'count': cnt,
                            'risk_level': risk_level,
                            'reason': 'enumerable_subrecord' if risk_level == 'high' else 'high_frequency'
                        }

        _annotate_bulk_risk(acc)
        n_bulk = sum(1 for c in acc if 'bulk_risk' in c)
        # 不再过滤，全部保留

        # 放宽预过滤：不再丢弃无语境的bare_subrecord
        # 让这些候选进入后续流程，由辩论层根据完整语境判断
        n_dropped = 0  # 暂时不删除bare_subrecord

        safe = [c for c in acc if c.get("kind") not in cand.RISKY_KINDS
                and float(c.get("score", 0) or 0) > 0]
        risky = [c for c in acc if c.get("kind") in cand.RISKY_KINDS]

        # 【修复】降低safe候选的初始confidence（0.75→0.60），让更多候选送入辩论层
        # 这样可以减少过早采纳带来的假阳性，同时保持召回
        out: list[Mention] = [_mk_mention(c, self.name, 0.60) for c in safe]

        if risky:
            user = (f"# Paper article_id: {bb.article_id}\n"
                    f"# IDs to adjudicate (raw|kind):\n" +
                    "\n".join(f"- {c['raw']} | {c['kind']}" for c in risky[:60]) +
                    "\n\n# Focused full-text view:\n" + bb.context)
            res = self._decide(user, max_tokens=4096,
                               note="风险库 accession 交 LLM 结合语境甄别")
            raw = (res or {}).get("mentions", []) if isinstance(res, dict) else []
            by_raw = {c.get("raw", "").lower(): c for c in risky}
            for m in raw:
                if not isinstance(m, dict) or not m.get("dataset_id"):
                    continue
                base = by_raw.get(str(m["dataset_id"]).strip().lower(),
                                  {"raw": m["dataset_id"], "norm": m["dataset_id"],
                                   "kind": cand.accession_kind(m["dataset_id"])})
                out.append(_mk_mention(base, self.name,
                                       float(m.get("confidence", 0) or 0),
                                       evidence=m.get("evidence", "")))
        bb.trace.append(AgentLog(agent=self.name, output={
            "n_bulk_annotated": n_bulk,
            "n_safe_direct": len(safe), "n_risky_llm": len(risky), "n_out": len(out)},
            note="AccessionExpert：批量风险标注（不排除），无歧义库直采，风险库交LLM"))
        return out


class MetadataExpert(BaseAgent):
    """外部元数据发现专家：DataCite 反查与本文关联的数据集 DOI。

    突破正文抽取召回天花板（很多数据集登记过但不在正文）。离线/无网络时
    related_datasets 返回空，本专家自然产出空——不影响主流程。
    """
    name = "MetadataExpert"
    system = (
        "You screen dataset candidates obtained by reverse-lookup from DataCite. "
        "The system found DOIs that DataCite marks as resourceType=Dataset and "
        "claims are related to this paper (with relationType and publisher). These "
        "candidates are NOT all genuine dataset citations — some are merely "
        "registered under the same project/author, or only weakly related.\n\n"
        "Keep only well-supported ones; drop registration noise.\n\n"
        "Common relationType values and their implications:\n"
        "- IsSupplementTo / IsReferencedBy: Strong signal (dataset supplements this paper).\n"
        "- IsCitedBy: Moderate signal (paper cites the dataset).\n"
        "- IsPartOf / HasPart: Weak signal (project/collection membership, not citation).\n"
        "- Other relations (IsVersionOf, etc.): Usually not direct citations.\n\n"
        "STRICT EXCLUSIONS (do NOT keep):\n"
        "1. Bulk crystallographic-structure deposits (CCDC 10.5517 / CSD entries).\n"
        "2. Figures, slide decks, posters, software, or supplementary PDFs.\n"
        "3. Candidates with no textual support that the paper used/produced them.\n"
        "4. DOIs with weak relationType (IsPartOf) and no mention in paper text.\n\n"
        "Be wary when dozens of near-identical DOIs from one prefix appear at once "
        "(auto-registered supplementary deposits, not cited datasets).\n\n"
        "Few-shot examples (drawn from real reverse-lookup scenarios):\n"
        "POSITIVE (keep):\n"
        "- DOI: 10.5061/dryad.abc123 | relationType: IsSupplementTo | publisher: Dryad | "
        "title: 'Data from: Impact of climate on bird migration' | Paper text mentions: "
        "'Migration data are available in Dryad (doi:10.5061/dryad.abc123)' → "
        "Strong relation + explicit textual support. KEEP, confidence=0.95.\n"
        "- DOI: 10.5281/zenodo.456789 | relationType: IsReferencedBy | publisher: Zenodo | "
        "title: 'Supplementary dataset for Smith et al 2023' | Paper has Data Availability "
        "section mentioning this DOI → registered supplement with textual anchor. "
        "KEEP, confidence=0.9.\n\n"
        "NEGATIVE (drop):\n"
        "- DOI: 10.5517/ccdc.csd.cc1a2b3c | relationType: IsSupplementTo | publisher: CCDC | "
        "title: 'Crystal structure of compound 47' | Paper text has 50+ similar CCDC DOIs → "
        "bulk crystal deposits, not datasets. DROP.\n"
        "- DOI: 10.6084/m9.figshare.999888 | relationType: IsPartOf | publisher: figshare | "
        "title: 'Figure S2 from Jones 2022' | Candidate is a figure, not data. DROP.\n"
        "- DOI: 10.5281/zenodo.777666 | relationType: IsPartOf | publisher: Zenodo | "
        "title: 'Project XYZ data collection' | Paper text has no mention of this DOI or "
        "project → weak relation, no textual support. DROP.\n"
        "- 30+ DOIs from same prefix (10.1594/PANGAEA.*) all with IsPartOf relation and "
        "no individual mention in text → likely auto-registered collection members. DROP "
        "(or flag very low confidence=0.2).\n\n"
        "Confidence calibration:\n"
        "- 0.9-1.0: Strong relationType (IsSupplementTo) + explicit text mention.\n"
        "- 0.7-0.9: Moderate relationType (IsReferencedBy) + text anchor found.\n"
        "- 0.5-0.7: Weak relationType but clear textual evidence of use.\n"
        "- <0.5: Weak relation + no text mention; flag for debate or drop.\n\n"
        "For each kept candidate output: dataset_doi (normalised https://doi.org/... "
        "form), relation (copy the given relationType), evidence (verbatim text snippet "
        "if found, or 'DataCite relation only'), confidence (0-1). Empty list if none.\n"
        'Output JSON: {"mentions":[{"dataset_doi":"","relation":"","evidence":"","confidence":0.0}]}'
    )

    def discover(self, bb) -> list[Mention]:
        cands = reg.related_datasets(bb.article_id)
        if not cands:
            bb.trace.append(AgentLog(agent=self.name, output={"n_datacite": 0},
                                     note="DataCite 反查无关联数据集（或离线）"))
            return []
        listing = "\n".join(
            f"- {c['doi']} | rel={c.get('relationType','')} | {c.get('publisher','')}"
            f" | {c.get('title','')[:60]}" for c in cands[:40])
        user = (f"# Paper article_id: {bb.article_id}\n"
                f"# Related dataset candidates from DataCite reverse-lookup:\n"
                + listing + "\n\n# Focused full-text view:\n" + bb.context)
        res = self._decide(user, max_tokens=4096, note="DataCite 反查候选甄别")
        raw = (res or {}).get("mentions", []) if isinstance(res, dict) else []
        out: list[Mention] = []
        for m in raw:
            if not isinstance(m, dict) or not m.get("dataset_doi"):
                continue
            nid = cand.normalize_doi(m["dataset_doi"])
            men = Mention(dataset_id=nid, form="doi", kind="DOI",
                          evidence=m.get("evidence", ""), pos=0,
                          confidence=float(m.get("confidence", 0) or 0),
                          discovered_by=self.name,
                          prior_score=1.0, repo_class=cand.repo_class(nid),
                          relation=m.get("relation", ""))
            out.append(men)
        bb.trace.append(AgentLog(agent=self.name, output={
            "n_datacite": len(cands), "n_out": len(out)},
            note="MetadataExpert：DataCite 反查增召回"))
        return out

