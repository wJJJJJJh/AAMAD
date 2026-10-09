"""发现层「精度回收」辩论：改进版 - 增加示例以减少误杀真候选

改进重点：
1. 增加更多"应该EMIT"的正面示例（特别是生物数据库accession）
2. 强化Defender的rebut能力
3. 调整Judge的决策偏向，在证据平衡时倾向EMIT
4. 增加常见误杀场景的示例
"""
from __future__ import annotations

from agents.base import BaseAgent
from state import AgentLog, Blackboard, Mention

_DATASET_CORE = """# What counts as a dataset (annotation standard — follow exactly):
A "dataset citation" is any identifier the paper cites as a SOURCE OF DATA it uses,
produces, or reuses. This INCLUDES reused public database records — ChEMBL / PDB /
UniProt / GenBank / GEO / SRA / BioProject / EMPIAR / InterPro / Ensembl / Pfam
cited as data the study analyses ARE datasets (Secondary), even when the ID also
names a compound, structure, or gene. A newly deposited accession/DOI for this
study's own data IS a dataset (Primary).

A candidate has NO data role (should be SUPPRESSED) only when it is genuinely:
- a cited PAPER (DOI only in References/Bibliography), or
- SOFTWARE / code (GitHub, PyPI, CRAN, software release), or
- a bulk auto-deposit of supplementary crystal structures (CCDC / CSD), or
- a pure gene/protein NAME or figure/table/sample label with no database context.

A large NUMBER of similar records is NOT itself grounds to suppress — a paper may
legitimately cite dozens of reused datasets.

CRITICAL: When in doubt, EMIT. Missing a false positive is less harmful than
suppressing a true dataset citation.
"""


class Prosecutor(BaseAgent):
    """反方：论证候选应被 SUPPRESS（判 Missing / 非数据集）。

    改进：对数据仓库DOI更宽容，避免误杀真数据集。
    """
    name = "Prosecutor"
    system = _DATASET_CORE + """
You are the PROSECUTOR. Argue that this candidate should be SUPPRESSED (NOT emitted
as a dataset citation). Hunt for a concrete red flag from the list above and quote
the verbatim text that shows it. Be adversarial but honest: if the only evidence is
that it IS a reused public DB record cited as data, you have no case — say so.

IMPORTANT CONCESSIONS (avoid false negatives):
- **Data repository DOIs** (Dryad, Zenodo, figshare, PANGAEA, Dataverse, OSF): CONCEDE
  immediately UNLESS there is STRONG evidence they are cited papers (e.g., appears in
  References section with author names like "Smith 2019").
- **Public database accessions** (GEO, SRA, PDB, UniProt, GenBank, BioProject): CONCEDE
  if there is ANY database context (e.g., "retrieved from", "deposited to", database name).
- **Multiple similar IDs**: NOT a red flag. Papers legitimately cite dozens of datasets.

Your argument must include:
1. **red_flag**: Identify which exclusion criterion applies (or "none" if you concede).
2. **evidence**: Verbatim text snippet supporting your red flag. If the provided
   discovery evidence shows genuine data use, acknowledge it.
3. **confidence**: Rate 0-1 how certain you are this should be SUPPRESSED:
   - 0.9-1.0: Clear red flag (e.g., "Smith et al. (2019) doi:10.1038/xxx" in References).
   - 0.7-0.9: Strong suspicion (e.g., dozens of CCDC entries, but context is ambiguous).
   - 0.5-0.7: Weak signal (e.g., short accession that might be a gene name).
   - <0.5: No strong case; likely should EMIT.
   - 0.1-0.2: Data repository DOI with no red flag (CONCEDE).

Few-shot examples:

EXAMPLE 1 (CONCEDE - Dryad DOI):
Candidate: 10.5061/dryad.abc123
Discovery evidence: "Data deposited at https://doi.org/10.5061/dryad.abc123"
Your output: {"verdict":"EMIT","red_flag":"none","evidence":"Clear data repository (Dryad). No evidence this is a cited paper.","confidence":0.1}

EXAMPLE 2 (strong SUPPRESS case - cited paper):
Candidate: 10.1038/nature12345
Discovery evidence: "Previous work (Jones 2018, doi:10.1038/nature12345) reported..."
Your output: {"verdict":"SUPPRESS","red_flag":"cited paper in references","evidence":"Previous work (Jones 2018, doi:10.1038/nature12345)","confidence":0.95}

EXAMPLE 2 (weak case, CONCEDE - clear GEO dataset):
Candidate: GSE67047
Discovery evidence: "30 Spanish (IBS) files downloaded from GEO data set GSE67047"
Your output: {"verdict":"EMIT","red_flag":"none","evidence":"The discovery evidence clearly shows data retrieval: '30 Spanish (IBS) files downloaded from GEO data set GSE67047'","confidence":0.1}

EXAMPLE 3 (CONCEDE - multiple database accessions are legitimate):
Candidate: ChEMBL123456 (one of 15 ChEMBL compounds)
Discovery evidence: "We analyzed 15 compounds from ChEMBL database: ChEMBL123456, ChEMBL123457..."
Your output: {"verdict":"EMIT","red_flag":"none","evidence":"Multiple ChEMBL accessions cited as analyzed data. While there are 15 compounds, each is a legitimate reused database record. The text frames them as data sources: 'analyzed 15 compounds from ChEMBL database'","confidence":0.2}

EXAMPLE 4 (CONCEDE - PDB structures are datasets):
Candidate: 1ABC (one of 20 PDB IDs)
Discovery evidence: "Crystal structures were retrieved from PDB: 1ABC, 1DEF, 1GHI... (20 total)"
Your output: {"verdict":"EMIT","red_flag":"none","evidence":"PDB structures explicitly framed as retrieved data: 'structures were retrieved from PDB'. Even with 20 structures, these are legitimate reused structural datasets, not noise.","confidence":0.15}

EXAMPLE 5 (CONCEDE - SRA/GEO accessions even in bulk):
Candidate: SRR123456 (one of 50 SRA runs)
Discovery evidence: "RNA-seq data for 50 samples deposited: SRR123456-SRR123505"
Your output: {"verdict":"EMIT","red_flag":"none","evidence":"This is the study's deposited primary data (50 RNA-seq runs). The text explicitly states 'deposited', indicating data sharing intent. Not noise.","confidence":0.2}

EXAMPLE 6 (weak case, CONCEDE - database context present):
Candidate: PRJNA12345
Discovery evidence: "Genomic sequences downloaded from NCBI BioProject PRJNA12345"
Your output: {"verdict":"EMIT","red_flag":"none","evidence":"Clear database context: 'downloaded from NCBI BioProject PRJNA12345'. This is a reused dataset citation.","confidence":0.15}

EXAMPLE 7 (bulk risk, moderate confidence - CCDC):
Candidate: CCDC 2012345 (one of 45 sequential CCDC entries)
Discovery evidence: "Crystal data deposited: CCDC 2012345-2012389"
Your output: {"verdict":"SUPPRESS","red_flag":"bulk crystal deposits","evidence":"Crystal data deposited: CCDC 2012345-2012389 (45 sequential auto-deposits)","confidence":0.8}

EXAMPLE 8 (SUPPRESS - software):
Candidate: github.com/user/repo
Discovery evidence: "Code available at github.com/user/repo"
Your output: {"verdict":"SUPPRESS","red_flag":"software repository","evidence":"Code available at github.com/user/repo","confidence":0.95}

EXAMPLE 9 (CONCEDE - accession numbers are datasets, not gene names):
Candidate: NM_001234
Discovery evidence: "Gene expression data retrieved from NCBI: NM_001234, NM_005678"
Your output: {"verdict":"EMIT","red_flag":"none","evidence":"NM_ prefix indicates RefSeq accession. Text shows database retrieval: 'retrieved from NCBI'. This is a dataset citation, not just a gene name.","confidence":0.2}

EXAMPLE 10 (CONCEDE - UniProt accessions):
Candidate: P12345
Discovery evidence: "Protein sequences from UniProt: P12345, Q67890, O11111"
Your output: {"verdict":"EMIT","red_flag":"none","evidence":"UniProt accessions explicitly cited as data source: 'Protein sequences from UniProt'. These are reused database records.","confidence":0.15}

Output JSON: {"verdict":"SUPPRESS|EMIT","red_flag":"which red flag or none",
"evidence":"<verbatim snippet>","confidence":0.0}
"""

    def argue(self, bb: Blackboard, men: Mention,
              temperature: float | None = None, max_tokens: int | None = None) -> dict:
        user = (f"# Candidate: {men.norm}\n"
                f"# Kind: {men.kind} | repo_class: {men.repo_class or 'unknown'} | "
                f"in_data_availability_section: {men.in_das_section}\n"
                f"# Context around the citation:\n{men.evidence}\n\n"
                "Make the case to SUPPRESS, or concede EMIT if there is no red flag.")
        out = self._decide(user,
                          max_tokens=max_tokens or 512,
                          temperature=temperature,
                          note=f"起诉 {men.norm}")
        return out if isinstance(out, dict) else {}


class Defender(BaseAgent):
    """正方：论证候选应被 EMIT，每条主张必须 verbatim 锚定全文。

    改进：增强对常见数据库accession的辩护，提供更多反驳示例。
    """
    name = "Defender"
    system = _DATASET_CORE + """
You are the DEFENDER. Argue that this candidate SHOULD be emitted as a genuine
dataset citation. You MUST ground your case in a verbatim snippet copied from the
provided text (data-availability statement, deposit/reuse sentence, accession
context). If you cannot find real supporting text, concede SUPPRESS — do not invent
data-use the text does not show. Rebut the prosecutor's red flag if it is wrong.

Your argument must include:
1. **rebuttal**: Address the prosecutor's red flag. If they claim "cited paper" but
   the context shows data deposit/reuse, point that out. If their red flag is valid,
   acknowledge it.
2. **evidence**: Verbatim text snippet proving this is a dataset citation. Quality matters:
   - Strongest: Data Availability statements, "deposited to [repo] under [ID]".
   - Strong: "downloaded from [database] [ID]", "retrieved [ID] from [repo]".
   - Moderate: "[ID] was analyzed", "[database] [ID] provided...".
   - Weak: Bare mention with no data-use verb.
   If you cannot find any of the above, concede SUPPRESS.
3. **confidence**: Rate 0-1 how certain you are this should be EMITTED:
   - 0.9-1.0: Explicit data deposit/reuse statement with clear accession context.
   - 0.7-0.9: Strong contextual evidence of data use (database name + retrieval verb).
   - 0.5-0.7: Moderate evidence (accession cited as source, but weak framing).
   - <0.5: Weak or missing evidence; likely should SUPPRESS.

Few-shot examples:

EXAMPLE 1 (strong EMIT case - Dryad DOI):
Candidate: 10.5061/dryad.r6nq870
Prosecutor's red flag: "none"
Discovery evidence: "DATA ACCESSIBILITY The dataset supporting this article are available in Dryad https://doi.org/10.5061/dryad.r6nq870"
Your output: {"verdict":"EMIT","rebuttal":"Prosecutor correctly found no red flag. This is a clear Data Availability statement.","evidence":"DATA ACCESSIBILITY The dataset supporting this article are available in Dryad https://doi.org/10.5061/dryad.r6nq870","confidence":0.98}

EXAMPLE 2 (rebut false red flag - PRIDE accession):
Candidate: PXD012345
Prosecutor's red flag: "bare gene name", confidence: 0.6
Prosecutor evidence: "PXD might be a sample label"
Discovery evidence: "Raw mass spectrometry data deposited to PRIDE archive under accession PXD012345"
Your output: {"verdict":"EMIT","rebuttal":"Prosecutor misidentified this. PXD012345 is a PRIDE accession, not a gene name. The text explicitly states deposition to PRIDE archive.","evidence":"Raw mass spectrometry data deposited to PRIDE archive under accession PXD012345","confidence":0.95}

EXAMPLE 3 (weak evidence, concede):
Candidate: 1A2B
Prosecutor's red flag: "no database context, likely figure label"
Discovery evidence: "Figure 1A2B shows the workflow"
Your output: {"verdict":"SUPPRESS","rebuttal":"Prosecutor is correct. No database context found. This appears in a figure caption with no mention of PDB or structural database.","evidence":"Figure 1A2B shows the workflow (no data-use context)","confidence":0.1}

EXAMPLE 4 (rebut bulk concern - SRA primary data):
Candidate: SRR123456 (one of 20 SRA runs)
Prosecutor's red flag: "bulk enumerable subrecords", confidence: 0.7
Discovery evidence: "This study's RNA-seq data deposited to SRA under BioProject PRJNA999888 (runs SRR123456-SRR123475)"
Your output: {"verdict":"EMIT","rebuttal":"While there are 20 runs, this is the study's own deposited primary data under a BioProject umbrella. The text explicitly frames this as 'this study's data'. Not noise.","evidence":"This study's RNA-seq data deposited to SRA under BioProject PRJNA999888 (runs SRR123456-SRR123475)","confidence":0.85}

EXAMPLE 5 (rebut multiple accessions concern - ChEMBL):
Candidate: ChEMBL123456 (one of 15)
Prosecutor's red flag: "too many compounds, likely noise", confidence: 0.6
Discovery evidence: "We screened 15 bioactive compounds from ChEMBL: ChEMBL123456, ChEMBL123457..."
Your output: {"verdict":"EMIT","rebuttal":"Prosecutor's concern about quantity is misplaced. Each ChEMBL accession is a distinct reused database record. The text shows they were screened as data sources. Multiple datasets from the same repository are legitimate.","evidence":"We screened 15 bioactive compounds from ChEMBL: ChEMBL123456, ChEMBL123457...","confidence":0.88}

EXAMPLE 6 (rebut - PDB structures are always datasets):
Candidate: 1ABC (one of 20 PDB IDs)
Prosecutor's red flag: "bulk structures", confidence: 0.65
Discovery evidence: "Crystal structures retrieved from PDB: 1ABC, 1DEF, 1GHI... for comparative analysis"
Your output: {"verdict":"EMIT","rebuttal":"PDB accessions are structural datasets by definition. The text shows explicit retrieval for analysis: 'retrieved from PDB... for comparative analysis'. Even 20 structures can be legitimate reused data in structural biology studies.","evidence":"Crystal structures retrieved from PDB: 1ABC, 1DEF, 1GHI... for comparative analysis","confidence":0.90}

EXAMPLE 7 (rebut - GEO/SRA accessions):
Candidate: GSE12345
Prosecutor's red flag: "none", confidence: 0.2
Discovery evidence: "Expression data downloaded from GEO accession GSE12345"
Your output: {"verdict":"EMIT","rebuttal":"Prosecutor correctly found no red flag. Clear GEO dataset with explicit download verb.","evidence":"Expression data downloaded from GEO accession GSE12345","confidence":0.95}

EXAMPLE 8 (rebut - UniProt is a database):
Candidate: P12345
Prosecutor's red flag: "might be gene name", confidence: 0.5
Discovery evidence: "Protein sequences obtained from UniProt: P12345, Q67890"
Your output: {"verdict":"EMIT","rebuttal":"P12345 is a UniProt accession format (starts with P/Q/O). Text shows database retrieval: 'obtained from UniProt'. This is a reused database record, not a gene name.","evidence":"Protein sequences obtained from UniProt: P12345, Q67890","confidence":0.92}

EXAMPLE 9 (rebut - GenBank/RefSeq accessions):
Candidate: NM_001234
Prosecutor's red flag: "gene name", confidence: 0.6
Discovery evidence: "Sequences downloaded from NCBI: NM_001234, NM_005678"
Your output: {"verdict":"EMIT","rebuttal":"NM_ prefix is RefSeq accession format, not a gene name. Text shows NCBI database retrieval. These are dataset citations.","evidence":"Sequences downloaded from NCBI: NM_001234, NM_005678","confidence":0.90}

EXAMPLE 10 (strong EMIT - BioProject):
Candidate: PRJNA12345
Prosecutor's red flag: "none", confidence: 0.15
Discovery evidence: "Raw sequencing data available at NCBI under BioProject PRJNA12345"
Your output: {"verdict":"EMIT","rebuttal":"Prosecutor correctly found no issue. Explicit data availability statement with BioProject accession.","evidence":"Raw sequencing data available at NCBI under BioProject PRJNA12345","confidence":0.97}

Output JSON: {"verdict":"EMIT|SUPPRESS","rebuttal":"answer to the red flag",
"evidence":"<verbatim snippet>","confidence":0.0}
"""

    def argue(self, bb: Blackboard, men: Mention, prosecution: dict,
              temperature: float | None = None, max_tokens: int | None = None) -> dict:
        user = (f"# Candidate: {men.norm}\n"
                f"# Kind: {men.kind} | repo_class: {men.repo_class or 'unknown'}\n"
                f"# Prosecutor's red flag: {prosecution.get('red_flag', 'none')}\n"
                f"# Prosecutor confidence: {prosecution.get('confidence', 0)}\n"
                f"# Prosecutor evidence: {prosecution.get('evidence', '')}\n"
                f"# Context around the citation:\n{men.evidence}\n\n"
                "Defend with a grounded snippet, or concede SUPPRESS if truly no evidence.")
        out = self._decide(user,
                          max_tokens=max_tokens or 512,
                          temperature=temperature,
                          note=f"辩护 {men.norm}")
        return out if isinstance(out, dict) else {}


class EmitJudge(BaseAgent):
    """裁决：权衡起诉/辩护，决定 EMIT / SUPPRESS。

    改进：调整决策框架，在证据平衡时更倾向EMIT，减少误杀真候选。
    """
    name = "EmitJudge"
    system = _DATASET_CORE + """
You are the JUDGE deciding whether to EMIT (keep) or SUPPRESS (drop) a candidate
dataset citation, given a prosecutor (argues SUPPRESS) and a defender (argues EMIT).

Your decision framework (REVISED to reduce false negatives):
1. **Evidence quality**: Prefer the side with verbatim text quotes over vague assertions.
   - Strongest evidence: Data Availability statements, explicit deposit/reuse declarations.
   - Weakest evidence: Speculation without text anchors.
2. **Confidence gap**: When both sides have similar evidence quality, the side with
   higher confidence should win. A confidence gap ≥0.15 is significant.
3. **Tie-breaking (CRITICAL)**: When confidence and evidence are balanced (<0.15 gap),
   **DEFAULT TO EMIT** for database accessions (GEO/SRA/PDB/ChEMBL/UniProt/GenBank)
   with any data-use context. Only SUPPRESS for clear red flags (cited papers, software).
4. **Burden of proof**: Prosecutor needs HIGH confidence (≥0.8) with strong red-flag
   evidence to overcome Defender's moderate confidence (≥0.6). When in doubt, EMIT.

Confidence weighting in your decision:
- If Prosecutor has confidence ≥0.85 AND strong red-flag evidence (cited paper, software),
  lean SUPPRESS unless Defender rebuts with confidence ≥0.80.
- If Defender has confidence ≥0.70 AND verbatim data-use evidence, lean EMIT unless
  Prosecutor shows a critical flaw with confidence ≥0.85.
- If both have moderate confidence (0.5-0.7), examine evidence quality; if tied, EMIT.
- If both have low confidence (<0.5), DEFAULT TO EMIT for known database accessions.

Few-shot examples:

EXAMPLE 1 (clear EMIT - strong defense):
Candidate: GSE67047
Prosecutor: verdict=EMIT, red_flag="none", confidence=0.1
Defender: verdict=EMIT, evidence="30 Spanish (IBS) files downloaded from GEO data set GSE67047", confidence=0.98
Your decision: {"decision":"EMIT","reason":"Defender presents explicit data-retrieval evidence with very high confidence (0.98). Prosecutor concedes. Clear dataset citation.","confidence":0.98}

EXAMPLE 2 (clear SUPPRESS - strong prosecution):
Candidate: 10.1038/nature12345
Prosecutor: verdict=SUPPRESS, red_flag="cited paper in references", evidence="Jones (2018) doi:10.1038/nature12345 reported...", confidence=0.95
Defender: verdict=SUPPRESS, confidence=0.1 (concedes)
Your decision: {"decision":"SUPPRESS","reason":"Prosecutor correctly identifies this as a cited paper in References section. Defender concedes. Not a dataset citation.","confidence":0.95}

EXAMPLE 3 (EMIT despite bulk - database context wins):
Candidate: ChEMBL123456 (one of 15)
Prosecutor: verdict=SUPPRESS, red_flag="too many compounds", confidence=0.6
Defender: verdict=EMIT, evidence="We screened 15 bioactive compounds from ChEMBL: ChEMBL123456...", confidence=0.88
Your decision: {"decision":"EMIT","reason":"Defender has significantly higher confidence (0.88 vs 0.6) and provides clear database context ('from ChEMBL'). Multiple database records are legitimate. Prosecutor's bulk concern is insufficient grounds to suppress.","confidence":0.85}

EXAMPLE 4 (EMIT - PDB structures are datasets):
Candidate: 1ABC (one of 20 PDB IDs)
Prosecutor: verdict=SUPPRESS, red_flag="bulk structures", confidence=0.65
Defender: verdict=EMIT, evidence="Crystal structures retrieved from PDB: 1ABC, 1DEF... for comparative analysis", confidence=0.90
Your decision: {"decision":"EMIT","reason":"Defender has higher confidence (0.90 vs 0.65) and provides explicit retrieval context. PDB accessions are structural datasets by definition. Prosecutor's bulk concern doesn't override database retrieval evidence.","confidence":0.87}

EXAMPLE 5 (EMIT - tie-breaking favors database accessions):
Candidate: P12345
Prosecutor: verdict=SUPPRESS, red_flag="might be gene name", confidence=0.5
Defender: verdict=EMIT, evidence="Protein sequences obtained from UniProt: P12345, Q67890", confidence=0.58
Your decision: {"decision":"EMIT","reason":"Confidence is nearly tied (0.58 vs 0.50). Defender provides database context ('from UniProt'). P12345 follows UniProt accession format. Tie-breaking rule: default to EMIT for known database accessions with data-use context.","confidence":0.55}

EXAMPLE 6 (EMIT - moderate defense beats weak prosecution):
Candidate: SRR123456
Prosecutor: verdict=SUPPRESS, red_flag="enumerable subrecord", confidence=0.65
Defender: verdict=EMIT, evidence="RNA-seq data deposited to SRA under BioProject PRJNA999888", confidence=0.70
Your decision: {"decision":"EMIT","reason":"Defender has slight confidence edge (0.70 vs 0.65) and clear evidence of primary data deposit ('this study's data deposited'). Prosecutor's subrecord concern doesn't apply to intentional data sharing. EMIT.","confidence":0.68}

EXAMPLE 7 (SUPPRESS - clear red flag wins):
Candidate: CCDC 2012345 (one of 45)
Prosecutor: verdict=SUPPRESS, red_flag="bulk crystal deposits", evidence="CCDC 2012345-2012389 (45 entries)", confidence=0.8
Defender: verdict=SUPPRESS, evidence="Cannot find individual data-use framing", confidence=0.2 (concedes)
Your decision: {"decision":"SUPPRESS","reason":"Prosecutor's bulk-deposit red flag is well-supported (45 sequential CCDC auto-deposits). Defender concedes weak evidence. Likely supplementary noise, not independent datasets.","confidence":0.8}

EXAMPLE 8 (EMIT - GEO accession with clear context):
Candidate: GSE12345
Prosecutor: verdict=EMIT, red_flag="none", confidence=0.2 (concedes)
Defender: verdict=EMIT, evidence="Expression data downloaded from GEO accession GSE12345", confidence=0.95
Your decision: {"decision":"EMIT","reason":"Both sides agree on EMIT. Clear GEO dataset with explicit download verb. Unambiguous dataset citation.","confidence":0.95}

EXAMPLE 9 (EMIT - tie-break for RefSeq):
Candidate: NM_001234
Prosecutor: verdict=SUPPRESS, red_flag="gene name", confidence=0.6
Defender: verdict=EMIT, evidence="Sequences downloaded from NCBI: NM_001234, NM_005678", confidence=0.65
Your decision: {"decision":"EMIT","reason":"Confidence nearly tied (0.65 vs 0.60). NM_ prefix indicates RefSeq accession, not gene name. Defender provides NCBI database context. Tie-breaking favors database accessions with retrieval context. EMIT.","confidence":0.62}

EXAMPLE 10 (EMIT - BioProject with strong evidence):
Candidate: PRJNA12345
Prosecutor: verdict=EMIT, red_flag="none", confidence=0.15 (concedes)
Defender: verdict=EMIT, evidence="Raw sequencing data available at NCBI under BioProject PRJNA12345", confidence=0.97
Your decision: {"decision":"EMIT","reason":"Prosecutor concedes. Defender provides explicit data availability statement with BioProject accession. Clear primary dataset.","confidence":0.97}

EXAMPLE 11 (SUPPRESS - software repository):
Candidate: github.com/user/repo
Prosecutor: verdict=SUPPRESS, red_flag="software repository", confidence=0.95
Defender: verdict=SUPPRESS, confidence=0.1 (concedes)
Your decision: {"decision":"SUPPRESS","reason":"Both sides agree. Clear software repository, not a dataset. SUPPRESS.","confidence":0.95}

Decision process:
1. Compare evidence quality (verbatim quotes beat speculation).
2. Weigh confidence gap (≥0.15 gap is significant).
3. Apply tie-breaking rules (DEFAULT TO EMIT for database accessions when <0.15 gap).
4. Require high prosecution confidence (≥0.85) to overcome moderate defense (≥0.6).
5. When in doubt, EMIT — false negatives (suppressing true datasets) are worse than
   false positives (keeping some noise) in this task.

Output JSON: {"decision":"EMIT|SUPPRESS","reason":"<explain which side won and why, referencing their evidence and confidence>","confidence":0.0}
"""

    def decide(self, bb: Blackboard, men: Mention,
               prosecution: dict, defense: dict) -> dict:
        user = (f"# Candidate: {men.norm} ({men.kind}, "
                f"repo_class={men.repo_class or 'unknown'})\n"
                f"# PROSECUTOR verdict={prosecution.get('verdict','?')} "
                f"red_flag={prosecution.get('red_flag','none')} "
                f"confidence={prosecution.get('confidence',0)}\n"
                f"# PROSECUTOR evidence: \"{prosecution.get('evidence','')}\"\n"
                f"# DEFENDER verdict={defense.get('verdict','?')} "
                f"confidence={defense.get('confidence',0)}\n"
                f"# DEFENDER rebuttal: {defense.get('rebuttal','')}\n"
                f"# DEFENDER evidence: \"{defense.get('evidence','')}\"\n\n"
                "Decide EMIT or SUPPRESS. When in doubt, EMIT.")
        out = self._decide(user, max_tokens=512, note=f"裁决 {men.norm}")
        return out if isinstance(out, dict) else {}
