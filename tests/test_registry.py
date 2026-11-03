"""DataCite response fixtures verify exact links, pagination and direction."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import config
from tools import doi_registry as reg

ARTICLE = "https://doi.org/10.1234/article"

def hit(doi, related):
    return {"id": doi, "attributes": {"doi": doi, "types": {"resourceTypeGeneral": "Dataset"},
        "titles": [{"title": "Data"}], "relatedIdentifiers": related}}

def rel(article, kind="IsCitedBy"):
    return {"relatedIdentifier": article, "relatedIdentifierType": "DOI", "relationType": kind}

class RegistryTests(unittest.TestCase):
    def test_reverse_lookup_runs_without_a_datacite_article_record_and_checks_exact_link(self):
        first = {"data": [hit("10.5061/data-a", [rel(ARTICLE)]),
                          hit("10.5061/unrelated", [rel("10.1234/another")])],
                 "links": {"next": "https://api.datacite.org/dois?page[number]=2"}}
        second = {"data": [hit("10.5061/data-b", [rel(ARTICLE, "IsSupplementTo")])]}
        with tempfile.TemporaryDirectory() as d, patch.object(config, "CACHE_DIR", Path(d)), \
             patch.object(config, "LLM_OFFLINE", False), patch.object(config, "DATACITE_ENABLED", True), \
             patch.object(reg, "_get_record", return_value={}), \
             patch.object(reg, "_get_json", side_effect=[first, second]) as http:
            result = reg.related_datasets(ARTICLE)
            self.assertEqual(len(result), 2)
            self.assertEqual(http.call_count, 2)
            self.assertEqual(reg.relation_label(ARTICLE, result[0]["doi"], result[0]["relations"]), "Secondary")
            self.assertEqual(reg.relation_label(ARTICLE, result[1]["doi"], result[1]["relations"]), "Primary")

    def test_failed_query_is_not_cached_as_successful_empty_result(self):
        with tempfile.TemporaryDirectory() as d, patch.object(config, "CACHE_DIR", Path(d)), \
             patch.object(config, "LLM_OFFLINE", False), patch.object(config, "DATACITE_ENABLED", True), \
             patch.object(reg, "_get_record", return_value={}), patch.object(reg, "_get_json", return_value=None):
            self.assertEqual(reg.related_datasets(ARTICLE), [])
            self.assertFalse(list(Path(d).rglob("*.json")))

    def test_offline_uncached_queries_never_open_the_network(self):
        with tempfile.TemporaryDirectory() as d, patch.object(config, "CACHE_DIR", Path(d)), \
             patch.object(config, "LLM_OFFLINE", True), patch("tools.doi_registry.urlopen") as http:
            self.assertEqual(reg.related_datasets(ARTICLE), [])
            http.assert_not_called()

if __name__ == "__main__":
    unittest.main()
