"""候选数据集抽取与归一化（完整独立实现）

从全文中抽取可能的数据集标识符：
1. DOI（数据集DOI，通过前缀识别）
2. Accession号（GenBank、GEO、SRA、PDB等公共数据库ID）

归一化规则与竞赛评测口径一致。
"""
from __future__ import annotations
import re
from typing import Any

# ================================================================ DOI 处理
def normalize_doi(doi: str) -> str:
    """归一化 DOI：去前缀、转小写、规范化为 https://doi.org/ 格式"""
    if not doi:
        return ""
    s = doi.strip()
    # 移除 URL 前缀
    s = re.sub(r'^https?://(?:dx\.)?doi\.org/', '', s, flags=re.I)
    s = re.sub(r'^doi:\s*', '', s, flags=re.I)
    # 去除尾部标点
    s = s.rstrip('.,;:)')
    s = s.lower()
    if s.startswith('10.'):
        return f'https://doi.org/{s}'
    return s


def fold_fragment_doi(doi: str, present: set[str] | None = None) -> str:
    """折叠 DOI 版本/分片后缀（.v1, .v2 等）

    Args:
        doi: 待折叠的 DOI
        present: 可选，当前上下文中存在的 DOI 集合。如果提供，只在 concept 版本
                 也存在时才折叠数字尾缀，避免误伤本身以数字结尾的独立 DOI
    """
    normalized = normalize_doi(doi)
    if not normalized.startswith('https://doi.org/'):
        return normalized

    # 移除版本后缀 (.v1, .v2 等)
    normalized = re.sub(r'\.v\d+$', '', normalized)

    # 移除分片后缀 (如 /1, /2)，但只在 present 上下文允许时
    if present is not None:
        # 只在 concept 版本也存在时才折叠数字尾缀
        base = re.sub(r'/\d+$', '', normalized)
        if base != normalized and base in present:
            normalized = base
    else:
        # 无上下文约束时，直接折叠
        normalized = re.sub(r'/\d+$', '', normalized)

    return normalized


# 数据集仓库 DOI 前缀（dataset-native repositories）
_DATASET_DOI_PREFIXES = [
    '10.5061/dryad',      # Dryad
    '10.1594/pangaea',    # PANGAEA
    '10.15468',           # GBIF
    '10.5281/zenodo',     # Zenodo
    '10.6084/m9.figshare', # figshare
    '10.7910/dvn',        # Dataverse
    '10.17632',           # Mendeley Data
    '10.17605/osf.io',    # OSF
    '10.6084',            # figshare (all)
    '10.7937',            # TCIA (The Cancer Imaging Archive)
    '10.25493',           # DataCite test prefix
    '10.5066',            # USGS ScienceBase
    '10.5067',            # USGS (another prefix)
    '10.17182',           # HEPData (High Energy Physics)
    '10.5256',            # CDL (California Digital Library)
    '10.3886',            # ICPSR (Inter-university Consortium)
    '10.18150',           # NCI (National Cancer Institute)
    '10.6073',            # Harvard Dataverse
    '10.6075',            # Harvard Dataverse
    '10.6078',            # Harvard Dataverse
    '10.6096',            # Harvard Dataverse
    '10.17882',           # SEANOE (Marine data)
    '10.5517',            # CCDC (Cambridge Crystallographic)
]

# 排除的DOI前缀（明确非数据集的出版物DOI）
_EXCLUDE_DOI_PREFIXES = [
    '10.1002',  # Wiley期刊
    '10.1007',  # Springer期刊
    '10.1016',  # Elsevier期刊
    '10.1038',  # Nature期刊
    '10.1126',  # Science期刊
    '10.1371',  # PLOS期刊
    '10.1073',  # PNAS期刊
    '10.1093',  # Oxford期刊
]

# DOI 正则：匹配 10.xxxxx/yyyyy 格式
_DOI_PATTERN = re.compile(
    r'\b(10\.\d{4,}(?:\.\d+)?/[^\s<>"\']{2,})', re.I
)


def _is_dataset_doi(doi: str) -> bool:
    """判断是否为数据集DOI（放宽规则）

    策略：
    1. 明确的数据集仓库前缀 → 接受
    2. 明确的期刊前缀 → 拒绝
    3. 其他前缀 → 接受（可能是数据集或其他仓库）
    """
    norm = normalize_doi(doi)

    # 检查是否为已知数据集仓库
    for prefix in _DATASET_DOI_PREFIXES:
        if norm.startswith(f'https://doi.org/{prefix.lower()}'):
            return True

    # 检查是否为已知期刊DOI（排除）
    for prefix in _EXCLUDE_DOI_PREFIXES:
        if norm.startswith(f'https://doi.org/{prefix.lower()}'):
            return False

    # 其他DOI默认接受（可能是数据集）
    return True


# ================================================================ Accession 处理

# Accession 模式定义：(pattern, kind, parent_kind)
ACCESSION_PATTERNS = [
    # GenBank/NCBI
    (r'\b([A-Z]{1,2}_?\d{5,6}(?:\.\d+)?)\b', 'genbank', 'genbank'),
    (r'\b(NC_\d{6})\b', 'refseq', 'genbank'),
    (r'\b(NM_\d{6})\b', 'refseq', 'genbank'),

    # GEO
    (r'\b(GSE\d{3,})\b', 'geo_series', 'geo'),
    (r'\b(GSM\d{3,})\b', 'geo_sample', 'geo'),
    (r'\b(GPL\d{3,})\b', 'geo_platform', 'geo'),

    # SRA
    (r'\b(SRP\d{6,})\b', 'sra_study', 'sra'),
    (r'\b(SRR\d{6,})\b', 'sra_run', 'sra'),
    (r'\b(SRX\d{6,})\b', 'sra_experiment', 'sra'),
    (r'\b(ERP\d{6,})\b', 'ena_study', 'sra'),
    (r'\b(ERR\d{6,})\b', 'ena_run', 'sra'),

    # Protein databases
    (r'\b([A-Z][0-9][A-Z0-9]{3})\b', 'pdb', 'pdb'),  # PDB: 4 chars
    (r'\b([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})\b', 'uniprot', 'uniprot'),

    # ArrayExpress
    (r'\b(E-[A-Z]{4}-\d+)\b', 'arrayexpress', 'arrayexpress'),

    # PRIDE
    (r'\b(PXD\d{6,})\b', 'pride', 'pride'),

    # BioProject/BioSample
    (r'\b(PRJ[NED][A-Z]\d{1,})\b', 'bioproject', 'bioproject'),
    (r'\b(SAM[NED][A-Z]?\d{7,})\b', 'biosample', 'biosample'),

    # ChEMBL
    (r'\b(CHEMBL\d{1,7})\b', 'chembl', 'chembl'),

    # dbSNP
    (r'\b(rs\d{3,})\b', 'dbsnp', 'dbsnp'),

    # EMPIAR
    (r'\b(EMPIAR-\d{5,})\b', 'empiar', 'empiar'),

    # MetaboLights
    (r'\b(MTBLS\d{1,})\b', 'metabolights', 'metabolights'),
]

# 可枚举子记录类型（单个run/sample通常不是研究级数据集）
ENUMERABLE_SUBRECORD_KINDS = {
    'sra_run', 'sra_experiment', 'ena_run',
    'geo_sample', 'genbank', 'biosample'
}

# 父级/项目级类型
PARENT_KINDS = {
    'geo_series', 'sra_study', 'ena_study', 'bioproject',
    'arrayexpress', 'pride', 'empiar', 'metabolights'
}

# 高风险/模糊类型（需要上下文验证）
RISKY_KINDS = {
    'pdb', 'uniprot', 'dbsnp', 'genbank'
}

# 无歧义数据库类型（高质量信号）
UNAMBIGUOUS_DB_KINDS = {
    'geo_series', 'sra_study', 'pride', 'empiar',
    'metabolights', 'arrayexpress', 'bioproject'
}

# 仓库前缀映射
REPO_PREFIXES = {
    'geo': 'GEO',
    'sra': 'SRA',
    'genbank': 'GenBank',
    'pdb': 'PDB',
    'uniprot': 'UniProt',
    'pride': 'PRIDE',
    'chembl': 'ChEMBL',
    'empiar': 'EMPIAR',
    'metabolights': 'MetaboLights',
    'arrayexpress': 'ArrayExpress',
    'bioproject': 'BioProject',
    'biosample': 'BioSample',
}

MAX_CANDIDATES = 1000


def accession_kind(accession: str) -> str:
    """识别 accession 的数据库类型"""
    if not accession:
        return "unknown"
    for pattern, kind, _ in ACCESSION_PATTERNS:
        if re.match(pattern, accession, re.I):
            return kind
    return "unknown"


def repo_class(kind: str) -> str:
    """返回仓库类别：dataset_native, reuse_db, mixed"""
    if kind in {'geo', 'sra', 'pride', 'empiar', 'metabolights', 'arrayexpress'}:
        return 'reuse_db'
    if kind in {'pdb', 'uniprot', 'genbank', 'chembl', 'dbsnp'}:
        return 'reuse_db'
    if kind == 'doi':
        return 'dataset_native'  # 假设是数据集DOI
    return 'mixed'


# ================================================================ 候选提取

def extract_candidates(text: str) -> list[dict[str, Any]]:
    """从全文中提取候选数据集标识符（DOI + Accession）

    返回格式：[{"id": str, "form": "doi"|"accession", "kind": str,
               "pos": int, "confidence": float}, ...]
    """
    if not text:
        return []

    candidates = []
    seen = set()

    # 1. 提取 DOI
    for match in _DOI_PATTERN.finditer(text):
        doi_raw = match.group(1)
        doi_norm = normalize_doi(doi_raw)

        # 只保留数据集DOI
        if not _is_dataset_doi(doi_norm):
            continue

        if doi_norm in seen:
            continue
        seen.add(doi_norm)

        # 判断仓库类别
        repo_cls = "dataset_native"
        for prefix in _DATASET_DOI_PREFIXES:
            if doi_norm.startswith(f'https://doi.org/{prefix.lower()}'):
                repo_cls = "dataset_native"
                break
        else:
            repo_cls = "mixed"  # 其他DOI默认为mixed

        candidates.append({
            "id": doi_norm,
            "norm": doi_norm,
            "form": "doi",
            "kind": "DOI",  # 注意：大写，与DOIExpert匹配
            "pos": match.start(),
            "confidence": 0.95,
            "raw": doi_raw,
            "repo_class": repo_cls,
            "score": 5.0,  # 默认高分
            "direct_adopt_ok": True,
        })

    # 2. 提取 Accession
    for pattern, kind, parent_kind in ACCESSION_PATTERNS:
        for match in re.finditer(pattern, text):
            acc = match.group(1)
            acc_upper = acc.upper()

            if acc_upper in seen:
                continue
            seen.add(acc_upper)

            # 基础置信度
            conf = 0.7

            # 提升父级/项目级ID的置信度
            if kind in PARENT_KINDS:
                conf = 0.9

            # 降低高风险类型的置信度
            if kind in RISKY_KINDS:
                conf = 0.5

            candidates.append({
                "id": acc_upper,
                "form": "accession",
                "kind": kind,
                "parent_kind": parent_kind,
                "pos": match.start(),
                "confidence": conf,
                "raw": acc,
            })

    # 限制候选数量
    if len(candidates) > MAX_CANDIDATES:
        candidates = sorted(candidates, key=lambda x: -x["confidence"])[:MAX_CANDIDATES]

    return candidates


def rank_candidates(candidates: list[dict], fulltext: str) -> list[dict]:
    """对候选进行排序和上下文增强

    排序规则：
    1. 置信度高的优先
    2. DOI 优先于 accession
    3. 父级ID优先于子记录
    """
    if not candidates:
        return []

    def _score(c: dict) -> tuple:
        conf = c.get("confidence", 0.0)
        is_doi = 1 if c.get("form") == "doi" else 0
        is_parent = 1 if c.get("kind") in PARENT_KINDS else 0
        return (is_doi, is_parent, conf)

    ranked = sorted(candidates, key=_score, reverse=True)

    # 添加上下文特征
    for c in ranked:
        pos = c.get("pos", 0)
        # 检查是否在 data availability 语句附近
        window_start = max(0, pos - 200)
        window_end = min(len(fulltext), pos + 200)
        context = fulltext[window_start:window_end].lower()

        c["has_data_ctx"] = bool(
            re.search(r'data\s+availab|accession|deposit|archive|repository', context)
        )

    return ranked


# ================================================================ 导出接口
__all__ = [
    'normalize_doi',
    'fold_fragment_doi',
    'extract_candidates',
    'rank_candidates',
    'accession_kind',
    'repo_class',
    'ACCESSION_PATTERNS',
    'ENUMERABLE_SUBRECORD_KINDS',
    'MAX_CANDIDATES',
    'PARENT_KINDS',
    'REPO_PREFIXES',
    'RISKY_KINDS',
    'UNAMBIGUOUS_DB_KINDS',
]
