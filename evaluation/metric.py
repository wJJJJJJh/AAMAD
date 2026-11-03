"""Manuscript evaluation: annotated-pair MDC micro metrics and DataRef macro metrics.

MDC negative rows are annotated non-references, not neutral observations. Unlabeled
predictions are excluded from the fixed annotated universe. Distinct DOI versions
are never collapsed. All calculations retain precision until presentation.
"""
from __future__ import annotations
from dataclasses import dataclass
import random
from tools.candidates import article_key, normalize_identifier, normalize_doi, accession_kind

_SCORABLE = {"Primary", "Secondary"}
_NEGATIVE = {"missing", "negative", "nonreference", "non-reference", "non_reference", "suppress", "false", "0"}

def norm_id(value: str) -> str:
    raw = str(value or "").strip()
    if raw.casefold() in {"", "missing", "none", "null", "n/a", "nan"}:
        return ""
    return normalize_identifier(raw)

def _pair(row: dict) -> tuple[str, str]:
    return article_key(row.get("article_id", "")), norm_id(row.get("dataset_id", ""))

def _prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return dict(precision=p, recall=r, f1=2*p*r/(p+r) if p+r else 0.0, tp=tp, fp=fp, fn=fn)

def _gold(rows: list[dict], require_provenance=True) -> dict:
    result = {}
    for row in rows:
        pair = _pair(row)
        if not all(pair):
            continue  # Includes Missing entries with no resolvable identifier.
        label = str(row.get("type", "") or "").strip()
        label = label.capitalize()
        flag = row.get("is_reference")
        flag = str(flag).casefold() if flag is not None else None
        if (label.casefold() == "missing" and flag is None
                and not normalize_doi(pair[1]) and accession_kind(pair[1]) == "unknown"):
            continue
        positive = (flag in {"true", "1", "yes"} if flag is not None
                    else label.casefold() not in _NEGATIVE)
        if flag is not None and flag not in {"true", "1", "yes", "false", "0", "no"}:
            raise ValueError(f"Invalid is_reference for {pair}: {flag}")
        if positive and require_provenance and label not in _SCORABLE:
            raise ValueError(f"A positive MDC pair needs Primary/Secondary: {pair}")
        value = label if positive and require_provenance else ("reference" if positive else None)
        if pair in result and result[pair] != value:
            raise ValueError(f"Conflicting gold annotations for {pair}")
        result[pair] = value
    return result

def _predictions(rows: list[dict]) -> dict:
    result = {}
    for row in rows:
        pair = _pair(row)
        if not all(pair):
            continue
        label = str(row.get("type", "") or "").strip().capitalize()
        if pair in result and result[pair] != label:
            raise ValueError(f"Conflicting predicted types for {pair}")
        result[pair] = label
    return result

def mdc_metrics(pred_rows: list[dict], gold_rows: list[dict]) -> dict:
    annotated = _gold(gold_rows)
    positive = {p for p, label in annotated.items() if label is not None}
    negative = set(annotated) - positive
    all_pred = _predictions(pred_rows)
    predicted = set(all_pred) & set(annotated)
    found = predicted & positive
    discovery = _prf(len(found), len(predicted & negative), len(positive - predicted))
    discovery["tn"] = len(negative - predicted)
    correct = sum(all_pred[p] == annotated[p] for p in found)
    overall = _prf(correct, len(predicted) - correct, len(positive) - correct)
    accuracy = correct / len(found) if found else 0.0
    return dict(discovery=discovery, overall=overall, provenance_accuracy=accuracy,
                provenance_correct=correct, provenance_total=len(found),
                evaluation_pairs=len(annotated), positive_pairs=len(positive),
                negative_pairs=len(negative),
                excluded_unlabeled_predictions=len(set(all_pred) - set(annotated)))

@dataclass
class ScoreResult:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    def as_dict(self) -> dict:
        return dict(self.__dict__)

def score(pred_rows, gold_rows, exclude_missing_matches=False) -> ScoreResult:
    if exclude_missing_matches:
        raise ValueError("MDC annotated non-references cannot be neutralized under the manuscript protocol.")
    return ScoreResult(**mdc_metrics(pred_rows, gold_rows)["overall"])

def decomposed(pred_rows, gold_rows, exclude_missing_matches=False) -> dict:
    if exclude_missing_matches:
        raise ValueError("Use the annotated-universe protocol; valid negative IDs are not neutral.")
    result = mdc_metrics(pred_rows, gold_rows)
    return dict(discovery_f1=result["discovery"], type_accuracy=result["provenance_accuracy"],
                type_eval_n=result["provenance_total"], **result)

def macro_discovery(pred_rows: list[dict], gold_rows: list[dict], articles=None) -> dict:
    annotated = _gold(gold_rows, require_provenance=False)
    gold = {p for p, label in annotated.items() if label is not None}
    predicted = set(_predictions(pred_rows))
    cohort = {article_key(a) for a in articles} if articles is not None else {a for a, _ in gold | predicted | set(annotated)}
    per_article = []
    for article in sorted(cohort):
        g = {d for a, d in gold if a == article}
        p = {d for a, d in predicted if a == article}
        m = _prf(len(g & p), len(p - g), len(g - p))
        per_article.append(dict(article_id=article, gold_references=len(g),
                                predictions=len(p), **m))
    precision = sum(r["precision"] for r in per_article)/len(per_article) if per_article else 0.0
    recall_rows = [r for r in per_article if r["gold_references"]]
    recall = sum(r["recall"] for r in recall_rows)/len(recall_rows) if recall_rows else 0.0
    return dict(precision=precision, recall=recall,
                f1=2*precision*recall/(precision+recall) if precision+recall else 0.0,
                articles=len(cohort), recall_articles=len(recall_rows),
                gold_references=sum(r["gold_references"] for r in per_article),
                per_article=per_article)

def discovery_by_form(pred_rows, gold_rows, exclude_missing_matches=False) -> dict:
    if exclude_missing_matches:
        raise ValueError("Annotated negative pairs remain part of evaluation.")
    out = {}
    for form in ("doi", "accession"):
        select = lambda row: ("doi" if normalize_doi(row.get("dataset_id", "")) else "accession") == form
        out[form] = mdc_metrics([r for r in pred_rows if select(r)],
                               [r for r in gold_rows if select(r)])["discovery"]
    return out

def route_gold_breakdown(pred_rows, gold_rows) -> dict:
    gold = _gold(gold_rows)
    result = {}
    for row in pred_rows:
        p = _pair(row)
        label = gold.get(p) if p in gold else "unlabeled"
        bucket = result.setdefault(row.get("route") or row.get("case") or "unknown", {})
        label = label or "non_reference"
        bucket[label] = bucket.get(label, 0) + 1
    return result

def bootstrap_ci(pred_rows, gold_rows, n=1000, seed=20260705) -> dict:
    articles = sorted({article_key(r.get("article_id", "")) for r in gold_rows})
    if n <= 0 or not articles:
        return {}
    rng, values = random.Random(seed), []
    for _ in range(n):
        pred, gold = [], []
        for replica, article in enumerate(rng.choices(articles, k=len(articles))):
            # Resampled copies need distinct IDs, otherwise set dedup erases weights.
            for rows, target in [(pred_rows, pred), (gold_rows, gold)]:
                target.extend(dict(r, article_id=f"replica-{replica}") for r in rows
                              if article_key(r.get("article_id", "")) == article)
        values.append(score(pred, gold).f1)
    values.sort()
    return dict(f1_mean=sum(values)/n, f1_lo95=values[int(.025*n)],
                f1_hi95=values[min(n-1, int(.975*n))], n_boot=n)
