"""Three ablations sharing identical discovered candidates and cached responses."""
from __future__ import annotations
from pathlib import Path
import argparse
import json
import config
from orchestrator import Orchestrator
from run import read_rows, read_articles, evaluate, write_result

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--articles", type=Path, default=Path(__file__).with_name("paper_ids.txt"))
    p.add_argument("--data-dir", type=Path)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--evaluation", choices=("mdc", "dataref"), default="mdc")
    p.add_argument("--split", choices=("train", "test"), default="train")
    p.add_argument("--output", type=Path, default=config.OUT_DIR/"ablations")
    args = p.parse_args(argv)
    if args.data_dir:
        config.DATA_DIR = args.data_dir
    articles, labels = read_articles(args.articles), read_rows(args.labels)
    evaluate([], labels, args.evaluation, articles)
    orchestrator = Orchestrator()
    # No default "first 10" subset, and no separate stochastic discovery per mode.
    discovery = [orchestrator.prepare_article(a, args.split) for a in articles]
    pools = [dict(article_id=b.article_id, candidates=[m.norm for m in b.mentions])
             for b in discovery]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"shared_pools.json").write_text(json.dumps(pools, indent=2), encoding="utf-8")
    for mode in ("discovery_only", "conventional_debate", "AAMAD"):
        boards = [orchestrator.validate_classify(b, mode) for b in discovery]
        metrics = evaluate([r for b in boards for r in b.prediction_rows()], labels, args.evaluation, articles)
        write_result(args.output/mode, boards, metrics, mode, articles, args.labels)
        print(json.dumps({"mode": mode, "metrics": metrics}, ensure_ascii=False))

if __name__ == "__main__":
    main()
