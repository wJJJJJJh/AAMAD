"""确定性候选发现（纯正则，零LLM）供消融实验使用

目的：保证三组实验（bare_llm/no_debate/full）使用完全相同的候选池
- 不依赖LLM判断（确定性、可复现）
- 不做批量过滤（保持最大召回）
- 只做形态识别 + 基础排序

与tools/candidates.py的区别：
- candidates.py: 生产环境用，包含LLM甄别 + 批量过滤 + 启发式排序
- regex_discovery.py: 消融实验专用，纯正则 + 位置排序，无任何LLM调用
"""
from __future__ import annotations

import re
from typing import Any

# DOI 正则：捕获 10.xxxxx/yyyyy 格式
DOI_PATTERN = re.compile(
    r'\b(10\.\d{4,}(?:\.\d+)?/[^\s<>"\']{2,})',
    re.IGNORECASE
)

# 数据集仓库 DOI 前缀（dataset-native repositories）
DATASET_DOI_PREFIXES = [
    '10.5061/dryad',      # Dryad
    '10.1594/pangaea',    # PANGAEA
    '10.15468',           # GBIF
    '10.5281/zenodo',     # Zenodo
    '10.6084/m9.figshare', # figshare
    '10.7910/dvn',        # Dataverse
    '10.17632',           # Mendeley Data
    '10.17605/osf.io',    # OSF
    '10.6084',            # figshare (all)
    '10.7937',            # TCIA
    '10.25493',           # DataCite test
    '10.5066',            # USGS ScienceBase
    '10.5067',            # USGS
    '10.17182',           # HEPData
    '10.5256',            # CDL
    '10.3886',            # ICPSR
    '10.18150',           # NCI
    '10.6073',            # Harvard Dataverse
    '10.6075',            # Harvard Dataverse
    '10.6078',            # Harvard Dataverse
    '10.6096',            # Harvard Dataverse
    '10.17882',           # SEANOE
    '10.5517',            # CCDC
]

# 排除的DOI前缀（明确非数据集的出版物DOI）
EXCLUDE_DOI_PREFIXES = [
    '10.1002',  # Wiley期刊
    '10.1007',  # Springer期刊
    '10.1016',  # Elsevier期刊
    '10.1038',  # Nature期刊
    '10.1126',  # Science期刊
    '10.1371',  # PLOS期刊
    '10.1073',  # PNAS期刊
    '10.1093',  # Oxford期刊
    '10.1111',  # Wiley期刊
    '10.1080',  # Taylor & Francis期刊
    '10.1128',  # ASM期刊
    '10.1186',  # BMC期刊
    '10.3354',  # Inter-Research期刊
    '10.4319',  # ASLO期刊
]

# Accession 正则（从 candidates.py 提取核心模式）
ACCESSION_PATTERNS = {
    # 无歧义公共数据库（高置信）
    'GEO-series': re.compile(r'\b(GSE\d{3,})\b'),
    'GEO-platform': re.compile(r'\b(GPL\d{3,})\b'),
    'GEO-sample': re.compile(r'\b(GSM\d{3,})\b'),
    'SRA-study': re.compile(r'\b(SRP\d{6,})\b'),
    'SRA-experiment': re.compile(r'\b(SRX\d{6,})\b'),
    'SRA-run': re.compile(r'\b(SRR\d{6,})\b'),
    'ENA-study': re.compile(r'\b(ERP\d{6,})\b'),
    'ENA-run': re.compile(r'\b(ERR\d{6,})\b'),
    'BioProject': re.compile(r'\b(PRJ[NDE][A-Z]\d+)\b'),
    'BioSample': re.compile(r'\b(SAM[NDE][A-Z]?\d+)\b'),
    'ArrayExpress': re.compile(r'\b(E-[A-Z]{4}-\d+)\b'),
    'PRIDE': re.compile(r'\b(P[XR]D\d{6,})\b'),
    'EMPIAR': re.compile(r'\b(EMPIAR-\d{4,})\b'),
    'MetaboLights': re.compile(r'\b(MTBLS\d+)\b'),
    'dbGaP': re.compile(r'\b(phs\d{6})\b'),
    'EGA': re.compile(r'\b(EGAS\d{11})\b'),

    # 有风险数据库（需要上下文，但仍捕获）
    'PDB': re.compile(r'\b([0-9][A-Za-z0-9]{3})\b'),
    'UniProt': re.compile(r'\b([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})\b'),
    'GenBank': re.compile(r'\b([A-Z]{1,2}_?\d{5,})\b'),
    'RefSeq': re.compile(r'\b((NC|NG|NM|NP|NR|NT|NW|NZ|XM|XP|XR|YP|ZP)_\d+)\b'),
    'Ensembl': re.compile(r'\b(ENS[A-Z]{0,3}[GPTR]\d{11})\b'),
    'Pfam': re.compile(r'\b(PF\d{5})\b'),
    'InterPro': re.compile(r'\b(IPR\d{6})\b'),
    'ChEMBL': re.compile(r'\b(CHEMBL\d+)\b'),
    'dbSNP': re.compile(r'\b(rs\d{5,})\b'),
}

def normalize_doi(raw: str) -> str:
    """归一化 DOI 到 https://doi.org/ 格式"""
    s = raw.strip()
    # 移除常见前缀
    for prefix in ['doi:', 'DOI:', 'https://doi.org/', 'http://dx.doi.org/', 'doi.org/']:
        if s.lower().startswith(prefix.lower()):
            s = s[len(prefix):]
    s = s.strip().lstrip('/')
    if s.startswith('10.'):
        return f"https://doi.org/{s}"
    return raw


def _is_dataset_doi(doi_norm: str) -> bool:
    """判断归一化DOI是否为数据集DOI（应用排除规则）"""
    # 检查是否为已知数据集仓库（直接接受）
    for prefix in DATASET_DOI_PREFIXES:
        if doi_norm.startswith(f'https://doi.org/{prefix.lower()}'):
            return True

    # 检查是否为已知期刊DOI（排除）
    for prefix in EXCLUDE_DOI_PREFIXES:
        if doi_norm.startswith(f'https://doi.org/{prefix.lower()}'):
            return False

    # 其他DOI默认接受（可能是数据集）
    return True


def discover_candidates_regex(fulltext: str) -> list[dict[str, Any]]:
    """纯正则候选发现（零LLM，确定性）

    返回格式：
    [
        {
            'raw': '原始文本',
            'norm': '归一化形式',
            'kind': 'DOI|GEO-series|PDB|...',
            'pos': 文本位置,
            'score': 0.0  # 占位，不做启发式评分
        },
        ...
    ]
    """
    candidates = []
    seen_norm = set()  # 去重（同一归一化ID只保留第一次出现）

    # 1. 提取 DOI（应用期刊排除规则）
    for match in DOI_PATTERN.finditer(fulltext):
        raw = match.group(1)
        norm = normalize_doi(raw)
        pos = match.start()

        # 只保留数据集DOI，排除期刊DOI
        if not _is_dataset_doi(norm):
            continue

        if norm not in seen_norm:
            candidates.append({
                'raw': raw,
                'norm': norm,
                'kind': 'DOI',
                'pos': pos,
                'score': 0.0
            })
            seen_norm.add(norm)

    # 2. 提取 Accession
    for kind, pattern in ACCESSION_PATTERNS.items():
        for match in pattern.finditer(fulltext):
            raw = match.group(1)
            norm = raw.strip().upper()  # accession统一大写
            pos = match.start()

            # 基础过滤：跳过明显的假阳性
            # PDB: 4字符全数字的可能是年份
            if kind == 'PDB' and raw.isdigit():
                continue
            # GenBank: 太短的可能是基因名
            if kind == 'GenBank' and len(raw) < 6:
                continue

            if norm not in seen_norm:
                candidates.append({
                    'raw': raw,
                    'norm': norm,
                    'kind': kind,
                    'pos': pos,
                    'score': 0.0
                })
                seen_norm.add(norm)

    # 按文本位置排序（稳定顺序）
    candidates.sort(key=lambda x: x['pos'])

    return candidates


def annotate_context(candidates: list[dict], fulltext: str, window: int = 300) -> list[dict]:
    """为候选添加上下文窗口（供后续阶段使用）

    Args:
        candidates: 候选列表
        fulltext: 全文
        window: 上下文窗口半径（字符数）

    Returns:
        增强后的候选列表（添加 'evidence' 字段）
    """
    for cand in candidates:
        pos = cand['pos']
        start = max(0, pos - window)
        end = min(len(fulltext), pos + window)
        cand['evidence'] = fulltext[start:end]

        # 简单判断是否在 Data Availability 段附近
        context_lower = cand['evidence'].lower()
        cand['has_data_ctx'] = any(
            keyword in context_lower
            for keyword in ['data availab', 'data accessib', 'deposited',
                           'accession', 'archived at', 'available at']
        )

    return candidates


def get_deterministic_candidates(fulltext: str, with_context: bool = True) -> list[dict]:
    """获取确定性候选池（消融实验入口）

    Args:
        fulltext: 论文全文
        with_context: 是否添加上下文窗口

    Returns:
        候选列表，按文本位置排序
    """
    candidates = discover_candidates_regex(fulltext)

    if with_context:
        candidates = annotate_context(candidates, fulltext)

    return candidates


# 统计函数（用于验证候选池一致性）
def count_by_kind(candidates: list[dict]) -> dict[str, int]:
    """按kind统计候选数量"""
    from collections import Counter
    return dict(Counter(c['kind'] for c in candidates))


if __name__ == '__main__':
    # 测试用例
    test_text = """
    DATA AVAILABILITY
    The dataset is available at Dryad: https://doi.org/10.5061/dryad.abc123

    RNA-seq data deposited to SRA under BioProject PRJNA123456 (runs SRR123001-SRR123010).
    Protein structures retrieved from PDB: 1ABC, 2XYZ.
    Expression data from GEO: GSE67047.

    Previous work (Smith 2019, doi:10.1038/nature12345) reported similar trends.
    """

    cands = get_deterministic_candidates(test_text)
    print(f"Total candidates: {len(cands)}")
    print("\nBy kind:")
    for kind, count in count_by_kind(cands).items():
        print(f"  {kind}: {count}")

    print("\nFirst 5 candidates:")
    for c in cands[:5]:
        print(f"  {c['kind']:15s} {c['norm']}")
