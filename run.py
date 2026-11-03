"""CLI for inference/evaluation; actual data and model access are user supplied."""
from __future__ import annotations
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics
import config
from evaluation.metric import mdc_metrics, macro_discovery
from orchestrator import Orchestrator, MODES
from tools.candidates import article_key

def read_rows(path: Path) -> list[dict]:
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))

def read_articles(path: Path) -> list[str]:
    articles = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len({article_key(a) for a in articles}) != len(articles):
        raise ValueError("The article list contains duplicate identifiers")
    if not articles:
        raise ValueError("The article list is empty")
    return articles

def evaluate(predictions, labels, protocol, articles):
    for row in labels + predictions:
        if not {"article_id", "dataset_id"}.issubset(row):
            raise ValueError("Evaluation rows require article_id and dataset_id columns")
    cohort = {article_key(a) for a in articles}
    gold = [r for r in labels if article_key(r.get("article_id", "")) in cohort]
    pred = [r for r in predictions if article_key(r.get("article_id", "")) in cohort]
    if not gold:
        raise ValueError("No gold rows match the requested article cohort")
    result = mdc_metrics(pred, gold) if protocol == "mdc" else macro_discovery(pred, gold, articles)
    count = result["evaluation_pairs"] if protocol == "mdc" else result["gold_references"]
    if not count:
        raise ValueError("No resolvable annotated evaluation records in this cohort")
    return result

def summarize(boards):
    summary = {}
    for field in ("total_tokens", "recorded_tokens", "calls", "logical_calls", "cache_hits",
                  "network_requests", "runtime_seconds"):
        values = [b.usage.get(field, 0) for b in boards]
        summary[field] = dict(mean=statistics.mean(values) if values else 0,
                              sd=statistics.stdev(values) if len(values) > 1 else 0,
                              minimum=min(values) if values else 0, maximum=max(values) if values else 0)
    return summary

def write_result(output: Path, boards, metrics, mode, articles, labels_path=None):
    output.mkdir(parents=True, exist_ok=True)
    predictions = [row for bb in boards for row in bb.prediction_rows()]
    (output/"predictions.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in predictions), encoding="utf-8")
    report = dict(mode=mode, model=config.LLM_MODEL, temperature=config.LLM_TEMPERATURE,
                  suppression_threshold=config.SUPPRESS_CONFIDENCE_THRESHOLD,
                  threshold_comparison="strictly_greater", articles=articles,
                  actual_article_count=len(articles), metrics=metrics, usage=summarize(boards),
                  offline=config.LLM_OFFLINE, metadata_enabled=config.DATACITE_ENABLED,
                  context_budget="900 UTF-8 bytes, conservative byte-level token bound",
                  labels_sha256=hashlib.sha256(labels_path.read_bytes()).hexdigest() if labels_path else None,
                  per_article=[dict(article_id=b.article_id, usage=b.usage,
                                    discovery_candidates=sum(x.output.get("discovery_candidates", 0)
                                      for x in b.trace if x.agent == "DiscoveryPool"),
                                    retained_candidates=len(b.predictions),
                                    trace=[asdict(x) for x in b.trace]) for b in boards])
    (output/"report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--articles", type=Path, default=Path(__file__).with_name("paper_ids.txt"))
    p.add_argument("--data-dir", type=Path)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--evaluation", choices=("mdc", "dataref"), default="mdc")
    p.add_argument("--split", choices=("train", "test"), default="train")
    p.add_argument("--mode", choices=MODES, default="AAMAD")
    p.add_argument("--output", type=Path, default=config.OUT_DIR/"AAMAD")
    p.add_argument("--score-only", type=Path, help="Existing predictions CSV/JSONL, no inference")
    return p

def main(argv=None):
    args = parser().parse_args(argv)
    if args.data_dir:
        config.DATA_DIR = args.data_dir
    articles, labels = read_articles(args.articles), read_rows(args.labels)
    # Reject invalid/mismatched annotation inputs before making paid requests.
    evaluate([], labels, args.evaluation, articles)
    if args.score_only:
        metrics = evaluate(read_rows(args.score_only), labels, args.evaluation, articles)
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
        return
    orchestrator = Orchestrator()
    boards = [orchestrator.run_article(a, args.split, mode=args.mode) for a in articles]
    metrics = evaluate([r for b in boards for r in b.prediction_rows()], labels, args.evaluation, articles)
    write_result(args.output, boards, metrics, args.mode, articles, args.labels)
    print(json.dumps({"articles": len(articles), "metrics": metrics, "output": str(args.output)}, ensure_ascii=False))

if __name__ == "__main__":
    main()
