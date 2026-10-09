"""消融实验：三组对比实验"""
from __future__ import annotations

import json
import time
from pathlib import Path
from datetime import datetime
import pandas as pd

import config
from orchestrator import Orchestrator
from evaluation.metric import score


def run_ablation_experiment(ablation_name: str, article_ids: list[str], split: str = "train"):
    """运行单个消融实验"""

    print(f"\n{'='*70}")
    print(f"运行消融实验: {ablation_name}")
    print(f"{'='*70}")

    # 应用消融配置
    config.apply_ablation(ablation_name)
    print(f"配置: {config.ABLATION}")

    # 创建orchestrator
    orch = Orchestrator()

    # 处理所有文章
    all_predictions = []
    start_time = time.time()

    for i, article_id in enumerate(article_ids, 1):
        print(f"\n[{i}/{len(article_ids)}] 处理: {article_id}")
        try:
            bb = orch.run_article(article_id, split)

            # 收集预测结果
            for pred in bb.predictions:
                all_predictions.append({
                    "article_id": article_id,
                    "dataset_id": pred.dataset_id,
                    "type": pred.type
                })

            print(f"  候选: {len(bb.candidates)}, 预测: {len(bb.predictions)}")

        except Exception as e:
            print(f"  ❌ 错误: {e}")
            continue

    elapsed = time.time() - start_time

    # 评测
    if all_predictions:
        # 读取金标准
        df_gold = pd.read_csv(config.TRAIN_LABELS)
        gold_rows = df_gold[df_gold['article_id'].isin(article_ids)].to_dict('records')

        # 计算指标
        result = score(all_predictions, gold_rows)

        print(f"\n{'='*70}")
        print(f"实验结果: {ablation_name}")
        print(f"{'='*70}")
        print(f"文章数: {len(article_ids)}")
        print(f"预测数: {len(all_predictions)}")
        print(f"Precision: {result.precision:.3f}")
        print(f"Recall: {result.recall:.3f}")
        print(f"F1: {result.f1:.3f}")
        print(f"TP: {result.tp}, FP: {result.fp}, FN: {result.fn}")
        print(f"耗时: {elapsed:.1f}秒")

        # 保存结果
        output_dir = config.OUT_DIR / "ablation_experiments"
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"{ablation_name}_{timestamp}.json"

        results = {
            "ablation": ablation_name,
            "config": config.ABLATION,
            "num_articles": len(article_ids),
            "num_predictions": len(all_predictions),
            "metrics": {
                "precision": result.precision,
                "recall": result.recall,
                "f1": result.f1,
                "tp": result.tp,
                "fp": result.fp,
                "fn": result.fn
            },
            "elapsed_seconds": elapsed,
            "predictions": all_predictions
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n结果已保存: {output_file}")

        return results
    else:
        print(f"\n❌ 无有效预测结果")
        return None


def main():
    """运行三组消融实验"""

    # 读取训练集文章列表
    df_gold = pd.read_csv(config.TRAIN_LABELS)
    all_articles = sorted(df_gold['article_id'].unique())

    # 选择前10篇作为测试
    test_articles = all_articles[:10]

    print(f"\n{'='*70}")
    print(f"消融实验：三组对比")
    print(f"{'='*70}")
    print(f"测试文章: {len(test_articles)}篇")
    print(f"文章列表: {test_articles}")

    # 三组实验
    ablations = ["discovery_only", "conventional_debate", "AAMAD"]

    results = {}
    for ablation in ablations:
        result = run_ablation_experiment(ablation, test_articles, "train")
        if result:
            results[ablation] = result

    # 生成对比报告
    print(f"\n{'='*70}")
    print(f"三组实验对比")
    print(f"{'='*70}")
    print(f"{'实验组':<25} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print(f"{'-'*70}")

    for ablation in ablations:
        if ablation in results:
            m = results[ablation]['metrics']
            print(f"{ablation:<25} {m['precision']:<12.3f} {m['recall']:<12.3f} {m['f1']:<12.3f}")

    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
