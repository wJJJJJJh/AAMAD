"""Offline regression cases for the manuscript's behavior, not published scores."""
from __future__ import annotations
from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import config
from state import Mention, Blackboard
from orchestrator import Orchestrator, merge_mentions
from agents.debate_loop import route_candidate
from agents.emit_debate import EmitJudge
from evaluation.metric import mdc_metrics, macro_discovery, bootstrap_ci, norm_id
from tools.candidates import normalize_doi, extract_candidates, article_key
from tools.doi_registry import relation_label
from tools.provenance import classify
from tools.fulltext import load_fulltext, get_authors
from tools.evidence import bounded
from llm.client import LLMClient, _extract_json

ARTICLE = "10.1234/current"
DATASET = "https://doi.org/10.5061/dryad.record"

class FakeLLM:
    def __init__(self, responses=()):
        self.responses = list(responses)
        self.history = []
    def call_json(self, system, user, **kwargs):
        self.history.append((system, json.loads(user), kwargs))
        if not self.responses:
            raise AssertionError("Unexpected model call")
        return deepcopy(self.responses.pop(0))

class FakeRegistry:
    def __init__(self, records=(), metadata=None):
        self.records = list(records)
        self.metadata = metadata or {}
        self.calls = []
    def related_datasets(self, article):
        self.calls.append(("relations", article))
        return deepcopy(self.records)
    def dataset_metadata(self, doi):
        self.calls.append(("metadata", doi))
        return deepcopy(self.metadata.get(doi, {}))

def mention(identifier="GSE12345", evidence="", kind="geo_series", **kwargs):
    return Mention(raw=identifier, norm=normalize_doi(identifier) or identifier.upper(),
                   kind=kind, evidence=evidence, **kwargs)

def row(article, identifier, label="Primary"):
    return dict(article_id=article, dataset_id=identifier, type=label)

def link(kind, reverse=True, article=ARTICLE):
    return dict(subject=DATASET if reverse else article_key(article),
                object=article_key(article) if reverse else DATASET, relationType=kind)

class IdentifierAndRoutingTests(unittest.TestCase):
    def test_doi_spelling_and_real_versions(self):
        self.assertEqual(normalize_doi("HTTP://DX.DOI.ORG/10.1234/ABC."), "https://doi.org/10.1234/abc")
        self.assertEqual(normalize_doi("10.1234/object(test)"), "https://doi.org/10.1234/object(test)")
        self.assertNotEqual(norm_id("10.1234/data.v1"), norm_id("10.1234/data.v2"))
        self.assertNotEqual(norm_id("10.1234/data/1"), norm_id("10.1234/data/2"))

    def test_repository_wrappers_normalize_to_accession(self):
        self.assertEqual(norm_id("PDB: 5VA1"), "5VA1")
        self.assertEqual(norm_id("https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE12345"), "GSE12345")

    def test_patterns_keep_pdb_and_chembl_but_not_years(self):
        found = {c["norm"] for c in extract_candidates("PDB 5va1 and CHEMBL70 in 2026.")}
        self.assertEqual(found, {"5VA1", "CHEMBL70"})

    def test_accession_embedded_in_doi_is_not_second_raw_candidate(self):
        found = extract_candidates("Data: https://doi.org/10.6019/PXD012345")
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["kind"], "DOI")

    def test_repeated_mentions_preserve_the_data_use_context(self):
        text = "GSE12345 is mentioned in background.\nWe downloaded data from GSE12345."
        parts = [mention(evidence=c["evidence"], pos=c["pos"], score=.3)
                 for c in __import__("tools.candidates", fromlist=["rank_candidates"]).rank_candidates(extract_candidates(text), text)]
        merged = merge_mentions(parts)
        self.assertEqual(len(merged), 1)
        self.assertEqual(route_candidate(merged[0]), "ACCEPT")

    def test_routing_requires_both_namespace_and_target_data_use(self):
        cases = [
            (mention(evidence="We downloaded data from GEO GSE12345."), "ACCEPT"),
            (mention(evidence="GSE12345."), "DEBATE"),
            (mention("5VA1", "We downloaded structures from PDB 5VA1.", "pdb"), "DEBATE"),
            (mention("5VA1", "5VA1.", "pdb"), "DEBATE"),
            (mention(evidence="We downloaded GSE99999. GSE12345 appears only in background."), "DEBATE"),
            (mention(evidence="We did not download GSE12345."), "DEBATE"),
        ]
        for item, route in cases:
            with self.subTest(item=item.evidence):
                self.assertEqual(route_candidate(item), route)

    def test_context_byte_budget(self):
        text = "数据" * 2000
        self.assertLessEqual(len(bounded(text).encode("utf-8")), config.CONTEXT_TOKEN_LIMIT)

class ProvenanceTests(unittest.TestCase):
    def classify(self, item, authors=()):
        return classify(Blackboard(ARTICLE, authors=authors), item)

    def test_directional_relations(self):
        for kind, reverse, expected in [
            ("IsSupplementTo", True, "Primary"), ("IsSupplementedBy", False, "Primary"),
            ("IsCitedBy", True, "Secondary"), ("References", False, "Secondary"),
            ("IsSourceOf", True, "Secondary"), ("IsSourceOf", False, "Primary")]:
            with self.subTest(kind=kind, reverse=reverse):
                self.assertEqual(relation_label(ARTICLE, DATASET, [link(kind, reverse)]), expected)

    def test_relation_overrides_conflicting_text(self):
        item = mention(DATASET, f"We downloaded existing data {DATASET}.", "DOI",
                       relations=[link("IsSupplementTo")])
        self.assertEqual(self.classify(item).type, "Primary")

    def test_unrelated_article_and_version_are_not_provenance_proof(self):
        for records in [[link("IsSupplementTo", article="10.1234/other")],
                        [link("IsNewVersionOf")], [link("HasPart")]]:
            item = mention(DATASET, f"We downloaded {DATASET}.", "DOI", relations=records)
            prediction = self.classify(item)
            self.assertEqual((prediction.type, prediction.case), ("Secondary", "text"))

    def test_conflicting_relations_defer_to_text(self):
        item = mention(DATASET, f"We downloaded {DATASET}.", "DOI",
                       relations=[link("IsSupplementTo"), link("IsCitedBy")])
        self.assertEqual(self.classify(item).type, "Secondary")

    def test_text_overrides_both_repository_priors(self):
        self.assertEqual(self.classify(mention(evidence="We generated and deposited data as GSE12345.")).type, "Primary")
        self.assertEqual(self.classify(mention(DATASET, f"We downloaded {DATASET}.", "DOI")).type, "Secondary")

    def test_two_datasets_in_same_passage_receive_different_roles(self):
        text = "We generated GSE11111 and downloaded GSE22222."
        self.assertEqual(self.classify(mention("GSE11111", text)).type, "Primary")
        self.assertEqual(self.classify(mention("GSE22222", text)).type, "Secondary")

    def test_mixed_cues_use_local_authorship_and_ownership(self):
        text = "Lee generated GSE12345 for our data, although it was previously published."
        self.assertEqual(self.classify(mention(evidence=text), ("Lee",)).type, "Primary")

    def test_priors_only_when_no_decision(self):
        self.assertEqual(self.classify(mention(DATASET, "", "DOI")).type, "Primary")
        self.assertEqual(self.classify(mention()).type, "Secondary")

class JudgeTests(unittest.TestCase):
    def judge(self, prosecution, answer=None):
        item = mention(DATASET, "Software implementation is available at " + DATASET, "DOI")
        llm = FakeLLM([answer or dict(decision="SUPPRESS", red_flag_supported=True, confidence=.99)])
        return EmitJudge(llm=llm).decide(Blackboard(ARTICLE), item, prosecution, dict(verdict="EMIT"))

    def prosecution(self, score=.9, **updates):
        value = dict(verdict="SUPPRESS", red_flag_type="software", red_flag="software artifact",
                     evidence="Software implementation is available at " + DATASET, confidence=score)
        return dict(value, **updates)

    def test_strict_threshold_and_final_confidence_are_separate(self):
        for score in [0, .2, .85, -.1, 1.1, float("nan"), True, None]:
            with self.subTest(score=score):
                self.assertEqual(self.judge(self.prosecution(score))["decision"], "EMIT")
        self.assertEqual(self.judge(self.prosecution(.850001))["decision"], "SUPPRESS")

    def test_no_flag_never_suppresses(self):
        self.assertEqual(self.judge(self.prosecution(.99, red_flag="none"))["decision"], "EMIT")

    def test_ungrounded_quote_cannot_suppress(self):
        self.assertEqual(self.judge(self.prosecution(.99, evidence="This quotation is invented."))["decision"], "EMIT")

    def test_bulk_or_entity_prior_is_not_disqualifying(self):
        self.assertEqual(self.judge(self.prosecution(.99, red_flag_type="bulk_compounds"))["decision"], "EMIT")

    def test_judge_must_uphold_red_flag(self):
        answer = dict(decision="SUPPRESS", red_flag_supported=False, confidence=.99)
        self.assertEqual(self.judge(self.prosecution(.99), answer)["decision"], "EMIT")

class EndToEndTests(unittest.TestCase):
    def test_clear_geo_bypasses_all_model_calls_but_is_classified(self):
        llm = FakeLLM()
        result = Orchestrator(llm, FakeRegistry()).run_article(ARTICLE,
                     fulltext="We downloaded data from GEO accession GSE12345.")
        self.assertEqual([(p.dataset_id, p.type) for p in result.predictions], [("GSE12345", "Secondary")])
        self.assertEqual(llm.history, [])

    def test_discovery_handoff_then_one_sequential_exchange(self):
        quote = "We downloaded bioactivity data from ChEMBL CHEMBL70."
        llm = FakeLLM([
            {"mentions": [{"dataset_id": "CHEMBL70", "evidence": quote, "confidence": .9}]},
            {"verdict": "SUPPRESS", "red_flag": "none", "confidence": .2},
            {"verdict": "EMIT", "rebuttal": "Explicit data reuse", "evidence": quote, "confidence": .95},
            {"decision": "EMIT", "red_flag_supported": False, "confidence": .95},
        ])
        result = Orchestrator(llm, FakeRegistry()).run_article(ARTICLE, fulltext=quote)
        self.assertEqual(result.predictions[0].type, "Secondary")
        names = [t.agent for t in result.trace if t.agent in {"Prosecutor", "Defender", "Judge"}]
        self.assertEqual(names, ["Prosecutor", "Defender", "Judge"])
        self.assertEqual(len(llm.history), 4)
        self.assertEqual(llm.history[2][1]["prosecution"]["confidence"], .2)
        self.assertIn("defense", llm.history[3][1])

    def test_metadata_only_candidate_missing_from_article_is_recovered(self):
        record = dict(doi=DATASET, resourceTypeGeneral="Dataset", title="Expression measurements",
                      relations=[link("IsCitedBy")])
        llm = FakeLLM([{"verdict": "EMIT", "red_flag": "none", "confidence": .1},
                       {"verdict": "EMIT", "confidence": .9},
                       {"decision": "EMIT", "confidence": .9}])
        result = Orchestrator(llm, FakeRegistry([record])).run_article(
            ARTICLE, fulltext="The study analyzes expression measurements.")
        self.assertEqual([(p.dataset_id, p.type) for p in result.predictions], [(DATASET, "Secondary")])
        self.assertEqual(len(llm.history), 3)

    def test_no_validation_ablation_keeps_real_provenance_classifier(self):
        result = Orchestrator(FakeLLM(), FakeRegistry()).run_article(
            ARTICLE, mode="discovery_only", fulltext="We downloaded GEO data GSE12345.")
        self.assertEqual(result.predictions[0].type, "Secondary")
        self.assertFalse(any(t.agent == "Judge" for t in result.trace))

    def test_same_discovery_pool_can_be_reused_without_mutation(self):
        orchestrator = Orchestrator(FakeLLM(), FakeRegistry())
        pool = orchestrator.prepare_article(ARTICLE, fulltext="We downloaded GEO data GSE12345.")
        before = deepcopy(pool)
        for mode in ("discovery_only", "conventional_debate", "AAMAD"):
            result = orchestrator.validate_classify(pool, mode)
            self.assertEqual([p.dataset_id for p in result.predictions], ["GSE12345"])
        self.assertEqual(pool, before)

    def test_bare_llm_has_no_metadata_or_expert_calls(self):
        registry = FakeRegistry()
        llm = FakeLLM([{"mentions": [row(ARTICLE, "GSE12345", "Secondary")]}])
        result = Orchestrator(llm, registry).run_article(ARTICLE, mode="bare_llm", fulltext="GSE12345")
        self.assertEqual(result.predictions[0].type, "Secondary")
        self.assertEqual(registry.calls, [])
        self.assertEqual(len(llm.history), 1)

class EvaluationTests(unittest.TestCase):
    def test_annotated_universe_confusion_counts_and_product_identities(self):
        gold = [row(ARTICLE, f"GSE{i}", "Primary") for i in range(1, 271)]
        gold += [row(ARTICLE, f"GSE{i}", "Missing") for i in range(271, 720)]
        predictions = [row(ARTICLE, f"GSE{i}", "Primary" if i <= 170 else "Secondary") for i in range(1, 206)]
        predictions += [row(ARTICLE, f"GSE{i}") for i in range(271, 331)]
        predictions += [row(ARTICLE, "GSE90000"), row(ARTICLE, "GSE90000")]
        result = mdc_metrics(predictions, gold)
        d, o, acc = result["discovery"], result["overall"], result["provenance_accuracy"]
        self.assertEqual((d["tp"], d["fp"], d["tn"], d["fn"]), (205, 60, 389, 65))
        self.assertEqual(result["evaluation_pairs"], 719)
        self.assertEqual(result["excluded_unlabeled_predictions"], 1)
        self.assertEqual(result["provenance_correct"], 170)
        self.assertAlmostEqual(o["precision"], d["precision"] * acc)
        self.assertAlmostEqual(o["recall"], d["recall"] * acc)

    def test_missing_placeholder_is_excluded_but_resolvable_negative_counts(self):
        result = mdc_metrics([row(ARTICLE, "GSE12345")],
                             [row(ARTICLE, "Missing", "Missing"), row(ARTICLE, "GSE12345", "Missing")])
        self.assertEqual(result["evaluation_pairs"], 1)
        self.assertEqual(result["discovery"]["fp"], 1)

    def test_macro_is_average_of_per_paper_rates_not_micro_or_average_f1(self):
        gold = [row("A", f"GSE{i}") for i in range(111, 115)] + [row("B", "GSE222")]
        pred = [row("A", "GSE111"), row("B", "GSE222"), row("B", "GSE999")]
        result = macro_discovery(pred, gold, ["A", "B", "C"])
        self.assertEqual(result["articles"], 3)
        self.assertEqual(result["recall_articles"], 2)
        self.assertAlmostEqual(result["precision"], .5)
        self.assertAlmostEqual(result["recall"], .625)
        self.assertAlmostEqual(result["f1"], 2*.5*.625/1.125)

    def test_conflicting_labels_are_not_silently_resolved(self):
        with self.assertRaises(ValueError):
            mdc_metrics([row(ARTICLE, "GSE12345", "Primary"), row(ARTICLE, "GSE12345", "Secondary")],
                        [row(ARTICLE, "GSE12345")])

    def test_dataref_does_not_drop_unknown_predictions_or_gold_identifiers(self):
        result = macro_discovery([row("A", "opaque-false-positive")],
                                 [row("A", "opaque-gold-reference")], ["A"])
        self.assertEqual(result["gold_references"], 1)
        self.assertEqual(result["per_article"][0]["fp"], 1)
        self.assertEqual(result["per_article"][0]["fn"], 1)

    def test_article_doi_filename_and_url_match(self):
        result = mdc_metrics([row("https://doi.org/10.1234/Current", "gse12345")],
                             [row("10.1234_current", "GSE12345")])
        self.assertEqual(result["overall"]["tp"], 1)

    def test_bootstrap_preserves_duplicate_article_weights(self):
        gold = [row("A", "GSE111"), row("B", "GSE222")]
        pred = [row("A", "GSE111")]
        with patch("evaluation.metric.random.Random") as rng:
            rng.return_value.choices.return_value = ["A", "A", "B"]
            result = bootstrap_ci(pred, gold, n=1)
        self.assertAlmostEqual(result["f1_mean"], .8)

class FullTextAndClientTests(unittest.TestCase):
    def test_jats_text_is_not_duplicated_and_only_article_authors_are_used(self):
        xml = """<article><front><article-meta><title-group><article-title>Study</article-title>
        </title-group><contrib-group><contrib><name><surname>Lee</surname></name></contrib>
        </contrib-group></article-meta></front><body><sec><title>Data availability</title>
        <p>We generated GSE12345.</p></sec></body><back><ref-list><ref><name><surname>Other</surname>
        </name><mixed-citation>Prior paper</mixed-citation></ref></ref-list></back></article>"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path/"train/XML").mkdir(parents=True)
            (path/"train/XML/paper.xml").write_text(xml)
            with patch.object(config, "DATA_DIR", path), patch.object(config, "CACHE_DIR", path/"cache"):
                text = load_fulltext("paper")
                self.assertEqual(text.count("We generated GSE12345."), 1)
                self.assertEqual(get_authors("paper"), ("Lee",))

    def test_missing_source_does_not_silently_make_an_empty_prediction(self):
        with tempfile.TemporaryDirectory() as d, patch.object(config, "DATA_DIR", Path(d)):
            with self.assertRaises(FileNotFoundError):
                load_fulltext("no-such-paper")

    def test_final_json_only_and_no_reasoning_fallback(self):
        self.assertEqual(_extract_json('{"decision":"EMIT"}')["decision"], "EMIT")
        with self.assertRaises(ValueError):
            _extract_json("Here is a reasoning trace without a final JSON answer")

    def test_cache_and_offline_accounting(self):
        class StubClient(LLMClient):
            def _raw_completion(self, *args):
                self.calls += 1
                self.total_tokens += 100
                return '{"mentions":[]}', 100
        with tempfile.TemporaryDirectory() as d, patch.object(config, "CACHE_DIR", Path(d)), \
             patch.object(config, "LLM_CACHE", True), patch.object(config, "LLM_OFFLINE", False):
            client = StubClient()
            client.call_json("system", "user")
            with patch.object(config, "LLM_OFFLINE", True):
                client.call_json("system", "user")
                with self.assertRaises(RuntimeError):
                    client.call_json("system", "new input")
            self.assertEqual((client.calls, client.cache_hits, client.total_tokens, client.recorded_tokens),
                             (1, 1, 100, 200))

if __name__ == "__main__":
    unittest.main()
