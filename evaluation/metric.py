"""竞赛官方评测（micro-F1）+ 诊断口径。

官方口径（严格）：
  - 计分单元 = (article_id, dataset_id, type)，type∈{Primary,Secondary}；
  - 金标里 type=Missing 的项不计分（既非 TP 目标，命中也不算 FP——见下）；
  - dataset_id 先归一（DOI 折叠版本/分片、accession 大小写）再比对；
  - micro：全体文章 TP/FP/FN 汇总后算一个 P/R/F1。

诊断口径（偏离官方，仅供分析，绝不用于报告主数字）：
  - exclude_missing_matches=True：预测命中了金标中 Missing 的 (article,dataset) 时，
    把该预测置为「中性」（既不计 TP 也不计 FP），用于估计「发现对了但类型判定/计分
    口径导致的损失」，隔离出真正的类型错误。

对外 API（与 run.py 对齐）：
  norm_id, score(...).as_dict(), decomposed, discovery_by_form,
  route_gold_breakdown, bootstrap_ci。
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from tools.candidates import normalize_doi, fold_fragment_doi

_SCORABLE = ("Primary", "Secondary")

def norm_id(dataset_id: str) -> str:
    """归一 dataset_id 用于比对：DOI 走归一+版本/分片折叠；accession 去空白后大写。

    金标与预测都过同一归一，保证「同一数据集的不同写法」判为相等。"""
    s = (dataset_id or "").strip()
    if not s:
        return ""
    low = s.lower()
    if "10." in low and ("doi" in low or low.startswith("10.") or "/" in s):
        n = normalize_doi(s)
        if "doi.org/" in n:
            return fold_fragment_doi(n)
    # accession：大小写不敏感，统一大写去空白
    return s.strip().upper()


def _keyset(rows: list[dict], scorable_only: bool = True) -> set[tuple[str, str, str]]:
    """把行折成 {(article_id, norm_dataset_id, type)}。scorable_only 时仅保留
    type∈{Primary,Secondary}。同一 (article,dataset,type) 天然去重。"""
    out: set[tuple[str, str, str]] = set()
    for r in rows:
        t = str(r.get("type", ""))
        if scorable_only and t not in _SCORABLE:
            continue
        aid = str(r.get("article_id", ""))
        did = norm_id(str(r.get("dataset_id", "")))
        if not did:
            continue
        out.add((aid, did, t))
    return out


def _missing_pairs(gold_rows: list[dict]) -> set[tuple[str, str]]:
    """金标中 type=Missing 的 (article_id, norm_dataset_id) 集合。"""
    out: set[tuple[str, str]] = set()
    for r in gold_rows:
        if str(r.get("type", "")) == "Missing":
            out.add((str(r.get("article_id", "")),
                     norm_id(str(r.get("dataset_id", "")))))
    return out


def _prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4),
            "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn}


@dataclass
class ScoreResult:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int

    def as_dict(self) -> dict:
        return {"precision": round(self.precision, 4),
                "recall": round(self.recall, 4),
                "f1": round(self.f1, 4),
                "tp": self.tp, "fp": self.fp, "fn": self.fn}


def score(pred_rows: list[dict], gold_rows: list[dict],
          exclude_missing_matches: bool = False) -> ScoreResult:
    """官方 micro-F1（type∈{Primary,Secondary}）。

    exclude_missing_matches=True（诊断口径）：命中金标 Missing 的 (article,dataset)
    的预测既不计 TP 也不计 FP（中性化），用于隔离类型判定损失。"""
    gold = _keyset(gold_rows)
    pred = _keyset(pred_rows)
    tp = len(pred & gold)
    fp_set = pred - gold
    fn = len(gold - pred)
    if exclude_missing_matches:
        miss = _missing_pairs(gold_rows)
        fp_set = {k for k in fp_set if (k[0], k[1]) not in miss}
    fp = len(fp_set)
    d = _prf(tp, fp, fn)
    return ScoreResult(d["precision"], d["recall"], d["f1"], tp, fp, fn)


def decomposed(pred_rows: list[dict], gold_rows: list[dict],
               exclude_missing_matches: bool = False) -> dict:
    """把总 F1 拆成两阶段：
      - discovery_f1：只看 (article,dataset) 是否发现（忽略 type）；
      - type_accuracy：在「发现且金标可计分」的交集里，type 判对的比例。
    用于回答「丢分是没发现，还是类型判错」。"""
    gold_pairs = {(a, d) for (a, d, _t) in _keyset(gold_rows)}
    pred_pairs_all = {(str(r.get("article_id", "")),
                       norm_id(str(r.get("dataset_id", ""))))
                      for r in pred_rows if norm_id(str(r.get("dataset_id", "")))}
    if exclude_missing_matches:
        # 发现阶段不惩罚「命中 Missing 的发现」
        miss = _missing_pairs(gold_rows)
        pred_pairs = {p for p in pred_pairs_all if p not in miss}
    else:
        pred_pairs = pred_pairs_all
    dtp = len(pred_pairs & gold_pairs)
    dfp = len(pred_pairs - gold_pairs)
    dfn = len(gold_pairs - pred_pairs)
    disc = _prf(dtp, dfp, dfn)

    # 类型判定准确率：在发现交集上比对 type
    gold_type = {(a, d): t for (a, d, t) in _keyset(gold_rows)}
    pred_type: dict[tuple[str, str], str] = {}
    for r in pred_rows:
        t = str(r.get("type", ""))
        if t not in _SCORABLE:
            continue
        k = (str(r.get("article_id", "")), norm_id(str(r.get("dataset_id", ""))))
        if k[1]:
            pred_type.setdefault(k, t)
    inter = set(gold_type) & set(pred_type)
    correct = sum(1 for k in inter if gold_type[k] == pred_type[k])
    type_acc = correct / len(inter) if inter else 0.0
    return {"discovery_f1": disc,
            "type_accuracy": round(type_acc, 4),
            "type_eval_n": len(inter)}


def _form(dataset_id: str) -> str:
    """把 dataset_id 归到形态桶：doi vs accession（用于阶段A分桶诊断）。"""
    return "doi" if "doi.org/" in norm_id(dataset_id).lower() else "accession"


def discovery_by_form(pred_rows: list[dict], gold_rows: list[dict],
                      exclude_missing_matches: bool = False) -> dict:
    """按 id 形态（DOI / accession）分桶报发现 P/R/F1，定位召回短板在哪种形态。"""
    miss = _missing_pairs(gold_rows) if exclude_missing_matches else set()
    out: dict[str, dict] = {}
    for form in ("doi", "accession"):
        gp = {(a, d) for (a, d, _t) in _keyset(gold_rows) if _form(d) == form}
        pp = {(str(r.get("article_id", "")), norm_id(str(r.get("dataset_id", ""))))
              for r in pred_rows
              if norm_id(str(r.get("dataset_id", "")))
              and _form(str(r.get("dataset_id", ""))) == form}
        if exclude_missing_matches:
            pp = {p for p in pp if p not in miss}
        tp = len(pp & gp)
        fp = len(pp - gp)
        fn = len(gp - pp)
        out[form] = _prf(tp, fp, fn)
    return out


def route_gold_breakdown(pred_rows: list[dict], gold_rows: list[dict]) -> dict:
    """诊断：预测的 route/case × 金标类型交叉表。看每条决策路径命中的是
    Primary/Secondary/Missing/无标（后两者在官方口径下均为 FP）。"""
    gold_lookup: dict[tuple[str, str], str] = {}
    for r in gold_rows:
        gold_lookup[(str(r.get("article_id", "")),
                     norm_id(str(r.get("dataset_id", ""))))] = str(r.get("type", ""))
    table: dict[str, dict[str, int]] = {}
    for r in pred_rows:
        route = str(r.get("route", "") or r.get("case", "") or "unknown")
        k = (str(r.get("article_id", "")), norm_id(str(r.get("dataset_id", ""))))
        gt = gold_lookup.get(k, "NONE")   # 金标里没有 → 无标注
        bucket = table.setdefault(route, {})
        bucket[gt] = bucket.get(gt, 0) + 1
    return table


def bootstrap_ci(pred_rows: list[dict], gold_rows: list[dict],
                 n: int = 1000, seed: int = 20260705) -> dict:
    """按文章自助重采样估计官方 F1 的 95% 置信区间。

    重采样单元是文章（保持文章内 TP/FP/FN 结构），比按行重采样更贴近真实方差。"""
    if n <= 0:
        return {}
    # 按文章聚合
    articles = sorted({str(r.get("article_id", "")) for r in gold_rows}
                      | {str(r.get("article_id", "")) for r in pred_rows})
    pred_by: dict[str, list[dict]] = {a: [] for a in articles}
    gold_by: dict[str, list[dict]] = {a: [] for a in articles}
    for r in pred_rows:
        pred_by.setdefault(str(r.get("article_id", "")), []).append(r)
    for r in gold_rows:
        gold_by.setdefault(str(r.get("article_id", "")), []).append(r)
    rng = random.Random(seed)
    f1s: list[float] = []
    m = len(articles)
    if m == 0:
        return {}
    for _ in range(n):
        sample = [articles[rng.randrange(m)] for _ in range(m)]
        pr, gr = [], []
        for a in sample:
            pr.extend(pred_by.get(a, []))
            gr.extend(gold_by.get(a, []))
        f1s.append(score(pr, gr).f1)
    f1s.sort()
    lo = f1s[int(0.025 * len(f1s))]
    hi = f1s[min(len(f1s) - 1, int(0.975 * len(f1s)))]
    mean = sum(f1s) / len(f1s)
    return {"f1_mean": round(mean, 4), "f1_lo95": round(lo, 4),
            "f1_hi95": round(hi, 4), "n_boot": n}

