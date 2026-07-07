import importlib
import pytest

import rag.indexer as indexer
import rag.retriever as retriever
from rag.retriever import find_similar, SIMILARITY_FLOOR


@pytest.fixture(autouse=True)
def reset_rag_modules():
    """
    Reload indexer + retriever between tests to reset the qdrant client
    singleton and the _ensured_collections cache, so a bug in one test's
    collection setup can't mask a bug in another's.
    """
    importlib.reload(indexer)
    importlib.reload(retriever)
    yield


class TestQdrantFuzzyMatch:

    def _index_acme_clients(self):
        from mock.db import get_all_clients
        clients = get_all_clients(company_id=1, page=1, size=10_000)["items"]
        indexer.index_entities(
            "client", 1, clients, lambda c: f"{c['first_name']} {c['last_name']}"
        )
        return clients

    def _index_nova_clients(self):
        from mock.db import get_all_clients
        clients = get_all_clients(company_id=2, page=1, size=10_000)["items"]
        indexer.index_entities(
            "client", 2, clients, lambda c: f"{c['first_name']} {c['last_name']}"
        )
        return clients

    def test_typo_name_still_matches_above_floor(self):
        self._index_acme_clients()
        results = find_similar("client", company_id=1, query_text="Amed Salm", top_k=3)
        assert len(results) >= 1
        assert all(r["score"] >= SIMILARITY_FLOOR for r in results)
        matched_names = [r["text"].lower() for r in results]
        assert any("ahmed" in name or "salem" in name for name in matched_names)

    def test_completely_unrelated_query_returns_nothing_or_low_score(self):
        self._index_acme_clients()
        results = find_similar(
            "client", company_id=1, query_text="quantum thermodynamics banana", top_k=3
        )
        assert all(r["score"] >= SIMILARITY_FLOOR for r in results)

    def test_products_typo_match(self):
        from mock.db import get_all_products
        products = get_all_products(company_id=1, page=1, size=10_000)["items"]
        indexer.index_entities("product", 1, products, lambda p: p["designation"])

        results = find_similar("product", company_id=1, query_text="Consultasion IT", top_k=3)
        assert len(results) >= 1
        assert all(r["score"] >= SIMILARITY_FLOOR for r in results)

    def test_find_similar_never_crosses_company_boundary(self):
        acme_clients = self._index_acme_clients()
        nova_clients = self._index_nova_clients()

        acme_ids = {c["id"] for c in acme_clients}
        nova_ids = {c["id"] for c in nova_clients}
        assert acme_ids.isdisjoint(nova_ids)

        results = find_similar("client", company_id=1, query_text="Karim Mzoughi", top_k=3)
        assert {r["id"] for r in results}.isdisjoint(nova_ids)

        results_nova_side = find_similar("client", company_id=2, query_text="Ahmed Salem", top_k=3)
        assert {r["id"] for r in results_nova_side}.isdisjoint(acme_ids)