"""Command-line smoke tests using temporary cohorts, never live API requests."""
from pathlib import Path
import csv
import json
import tempfile
import unittest
from unittest.mock import patch
import config
from run import main
from run_ablation import main as run_ablation
from scripts.export_prompts import render, sync_readme, README_BEGIN, README_END
from test_pipeline import FakeLLM, FakeRegistry, ARTICLE, DATASET, mention
from orchestrator import Orchestrator
from tools.provenance import classify
from state import Blackboard
from tools.candidates import extract_candidates
from tools.fulltext import get_article_doi

class CommandLineTests(unittest.TestCase):
    def data(self, directory):
        root = Path(directory)
        (root/"train/TXT").mkdir(parents=True)
        (root/"train/TXT/article.txt").write_text("We downloaded GEO data GSE12345.", encoding="utf-8")
        (root/"articles.txt").write_text("article\n", encoding="utf-8")
        with (root/"gold.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["article_id", "dataset_id", "type"])
            writer.writeheader()
            writer.writerow(dict(article_id="article", dataset_id="GSE12345", type="Secondary"))
        return root

    def test_inference_report_and_score_only_work_without_model_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.data(directory)
            output = root/"result"
            with patch.object(config, "DATA_DIR", root), patch.object(config, "LLM_OFFLINE", True), patch.object(config, "DATACITE_ENABLED", False), \
                 patch.object(config, "CACHE_DIR", root/"cache"), patch("builtins.print"):
                args = ["--articles", str(root/"articles.txt"), "--labels", str(root/"gold.csv"),
                        "--data-dir", str(root), "--output", str(output)]
                main(args)
                report = json.loads((output/"report.json").read_text())
                self.assertEqual(report["actual_article_count"], 1)
                self.assertEqual(report["metrics"]["overall"]["f1"], 1)
                self.assertEqual(report["temperature"], .2)
                main(args + ["--score-only", str(output/"predictions.jsonl")])

    def test_all_three_ablation_outputs_have_same_discovery_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.data(directory)
            output = root/"ablations"
            with patch.object(config, "DATA_DIR", root), patch.object(config, "LLM_OFFLINE", True), patch.object(config, "DATACITE_ENABLED", False), \
                 patch.object(config, "CACHE_DIR", root/"cache"), patch("builtins.print"):
                run_ablation(["--articles", str(root/"articles.txt"), "--labels", str(root/"gold.csv"),
                              "--data-dir", str(root), "--output", str(output)])
            pool = json.loads((output/"shared_pools.json").read_text())
            self.assertEqual(pool[0]["candidates"], ["GSE12345"])
            for mode in ("AAMAD", "discovery_only", "conventional_debate"):
                report = json.loads((output/mode/"report.json").read_text())
                self.assertEqual(report["per_article"][0]["discovery_candidates"], 1)
                self.assertEqual(report["metrics"]["provenance_accuracy"], 1)
                self.assertEqual(report["per_article"][0]["usage"]["runtime_seconds"] >= 0, True)

    def test_runtime_prompts_match_exported_documents(self):
        root = Path(__file__).resolve().parents[1]
        self.assertEqual((root/"PROMPTS.md").read_text(encoding="utf-8"), render())
        readme = (root/"README.md").read_text(encoding="utf-8")
        self.assertEqual(readme, sync_readme(readme))

    def test_prompt_sync_preserves_authored_readme_sections(self):
        prefix = "# Custom introduction\n\n"
        suffix = "\n\n## Custom setup\nKeep these instructions.\n"
        readme = prefix + README_BEGIN + "\nOutdated prompt\n" + README_END + suffix
        updated = sync_readme(readme)
        self.assertTrue(updated.startswith(prefix + README_BEGIN))
        self.assertTrue(updated.endswith(README_END + suffix))
        self.assertNotIn("Outdated prompt", updated)

    def test_invalid_gold_is_rejected_before_constructing_inference_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.data(directory)
            (root/"gold.csv").write_text("article_id,wrong_column\narticle,GSE12345\n")
            with patch("run.Orchestrator") as orchestrator:
                with self.assertRaises(ValueError):
                    main(["--articles", str(root/"articles.txt"), "--labels", str(root/"gold.csv")])
                orchestrator.assert_not_called()

class AdditionalIntegrationTests(unittest.TestCase):
    def test_registry_non_dataset_type_excludes_a_doi_without_model_call(self):
        registry = FakeRegistry(metadata={DATASET: {"resourceTypeGeneral": "Software"}})
        llm = FakeLLM()
        result = Orchestrator(llm, registry).run_article(ARTICLE, fulltext="We downloaded " + DATASET)
        self.assertEqual(result.predictions, [])
        self.assertEqual(llm.history, [])

    def test_batched_discovery_processes_candidates_beyond_first_twenty(self):
        class DiscoveryLLM(FakeLLM):
            def call_json(self, system, user, **kwargs):
                self.history.append((system, json.loads(user), kwargs))
                return {"mentions": [{"dataset_id": r["dataset_id"], "evidence": r["evidence"],
                                       "confidence": .9} for r in json.loads(user)["candidates"]]}
        text = "\n".join(f"We downloaded PDB structures 1A{i:02d}." for i in range(25))
        llm = DiscoveryLLM()
        result = Orchestrator(llm, FakeRegistry()).run_article(ARTICLE, mode="discovery_only", fulltext=text)
        self.assertEqual(len(result.predictions), 25)
        self.assertEqual(len(llm.history), 2)

    def test_text_metadata_duplicate_retains_relation_and_maximum_confidence(self):
        from test_pipeline import link
        record = dict(doi=DATASET, resourceTypeGeneral="Dataset", title="Study data",
                      relations=[link("IsSupplementTo")])
        registry = FakeRegistry([record], {DATASET: {"resourceTypeGeneral": "Dataset"}})
        result = Orchestrator(FakeLLM(), registry).run_article(ARTICLE, mode="discovery_only",
                                fulltext="We downloaded data " + DATASET)
        self.assertEqual(len(result.mentions), 1)
        self.assertEqual(result.predictions[0].type, "Primary")
        self.assertIn("DOIExpert", result.mentions[0].sources)
        self.assertIn("MetadataExpert", result.mentions[0].sources)
        self.assertEqual(len(result.mentions[0].relations), 1)

    def test_mixed_cues_across_sentences_are_resolved_using_article_local_ownership(self):
        bb = Blackboard(ARTICLE, authors=("Lee",))
        text = "Lee generated GSE12345 for our data. GSE12345 was retrieved for analysis."
        self.assertEqual(classify(bb, mention(evidence=text)).type, "Primary")

    def test_wrapped_doi_after_slash_is_repaired(self):
        found = extract_candidates("Measurements at 10.5061/\ndryad.abc.")
        self.assertEqual(found[0]["norm"], "https://doi.org/10.5061/dryad.abc")

    def test_pmc_header_doi_is_used_for_metadata_relation_binding(self):
        from test_pipeline import link
        xml = f"""<article><front><article-meta>
        <article-id pub-id-type="doi">{ARTICLE}</article-id></article-meta></front>
        <body><p>Expression measurements.</p></body><back><ref-list><ref>
        <article-id pub-id-type="doi">10.1234/not-the-article</article-id></ref></ref-list></back></article>"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"train/XML").mkdir(parents=True)
            (root/"train/XML/PMC123.xml").write_text(xml)
            record = dict(doi=DATASET, resourceTypeGeneral="Dataset", title="Study data",
                          relations=[link("IsCitedBy")])
            registry = FakeRegistry([record])
            with patch.object(config, "DATA_DIR", root), patch.object(config, "CACHE_DIR", root/"cache"):
                self.assertEqual(get_article_doi("PMC123"), "https://doi.org/" + ARTICLE)
                result = Orchestrator(FakeLLM(), registry).run_article("PMC123", mode="metadata_only")
                self.assertEqual(result.predictions[0].type, "Secondary")
                self.assertIn(("relations", "https://doi.org/" + ARTICLE), registry.calls)

    def test_single_agent_receives_same_datacite_object_type_information(self):
        llm = FakeLLM([{"mentions": [{"dataset_id": DATASET, "verdict": "SUPPRESS"}]}])
        registry = FakeRegistry(metadata={DATASET: {"resourceTypeGeneral": "Software", "title": "Code"}})
        result = Orchestrator(llm, registry).run_article(ARTICLE, mode="single_agent",
                                                    fulltext="Repository " + DATASET)
        self.assertEqual(result.predictions, [])
        self.assertEqual(llm.history[0][1]["candidates"][0]["metadata"]["resourceTypeGeneral"], "Software")

if __name__ == "__main__":
    unittest.main()
