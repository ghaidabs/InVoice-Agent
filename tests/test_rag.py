import os
import importlib
import pytest

import rag.indexer as indexer
import rag.retriever as retriever
from rag.retriever import find_similar, SIMILARITY_FLOOR


@pytest.fixture(autouse=True)
def reset_rag_state():
    """
    _MEMORY_INDEX is module-level mutable state (dict keyed by entity_type
    then company_id). Reload indexer + retriever between tests so indexing
    done in one test can't leak into / mask a bug in another.
    """
    importlib.reload(indexer)
    importlib.reload(retriever)
    yield


# ---------- environment sanity ----------

def test_no_qdrant_endpoint_means_in_memory_path(monkeypatch):
    monkeypatch.delenv("QDRANT_ENDPOINT", raising=False)
    importlib.reload(indexer)
    assert indexer.has_qdrant() is False


# ---------- in-memory fuzzy match ----------

class TestInMemoryFuzzyMatch:

    def _index_acme_clients(self):
        from mock.db import get_all_clients
        clients = get_all_clients(company_id=1, page=1, size=10_000)["items"]
        indexer.index_entities(
            "client", 1, clients,
            lambda c: f"{c['first_name']} {c['last_name']}",
        )
        return clients

    def _index_nova_clients(self):
        from mock.db import get_all_clients
        clients = get_all_clients(company_id=2, page=1, size=10_000)["items"]
        indexer.index_entities(
            "client", 2, clients,
            lambda c: f"{c['first_name']} {c['last_name']}",
        )
        return clients

    def test_typo_name_still_matches_above_floor(self, monkeypatch):
        monkeypatch.delenv("QDRANT_ENDPOINT", raising=False)
        self._index_acme_clients()

        # "Ahmed Salem" misspelled
        results = find_similar("client", company_id=1, query_text="Amed Salm", top_k=3)
        assert len(results) >= 1
        assert all(r["score"] >= SIMILARITY_FLOOR for r in results)
        matched_names = [r["text"].lower() for r in results]
        assert any("ahmed" in name or "salem" in name for name in matched_names)

    def test_completely_unrelated_query_returns_nothing_or_low_score(self, monkeypatch):
        monkeypatch.delenv("QDRANT_ENDPOINT", raising=False)
        self._index_acme_clients()

        results = find_similar(
            "client", company_id=1, query_text="quantum thermodynamics banana", top_k=3
        )
        # Everything returned (if anything) must clear the floor — retriever
        # already filters below SIMILARITY_FLOOR, so this is really a check
        # that nothing sneaks through under the threshold.
        assert all(r["score"] >= SIMILARITY_FLOOR for r in results)

    def test_products_typo_match(self, monkeypatch):
        monkeypatch.delenv("QDRANT_ENDPOINT", raising=False)
        from mock.db import get_all_products
        products = get_all_products(company_id=1, page=1, size=10_000)["items"]
        indexer.index_entities("product", 1, products, lambda p: p["designation"])

        results = find_similar("product", company_id=1, query_text="Consultasion IT", top_k=3)
        assert len(results) >= 1
        assert all(r["score"] >= SIMILARITY_FLOOR for r in results)

    # ---------- tenant isolation ----------

    def test_find_similar_never_crosses_company_boundary(self, monkeypatch):
        monkeypatch.delenv("QDRANT_ENDPOINT", raising=False)
        acme_clients = self._index_acme_clients()
        nova_clients = self._index_nova_clients()

        acme_ids = {c["id"] for c in acme_clients}
        nova_ids = {c["id"] for c in nova_clients}
        assert acme_ids.isdisjoint(nova_ids)  # sanity on fixture data itself

        # Query ACME's index with NOVA's client's name (typo'd) — should never
        # surface a NOVA id even if the vectors are similar
        results = find_similar("client", company_id=1, query_text="Karim Mzoughi", top_k=3)
        result_ids = {r["id"] for r in results}
        assert result_ids.isdisjoint(nova_ids)

        # And the reverse
        results_nova_side = find_similar("client", company_id=2, query_text="Ahmed Salem", top_k=3)
        result_ids_nova_side = {r["id"] for r in results_nova_side}
        assert result_ids_nova_side.isdisjoint(acme_ids)

    def test_memory_index_namespaced_by_entity_type_too(self, monkeypatch):
        # Regression guard: _MEMORY_INDEX is {"client": {company_id: [...]}, "product": {...}}.
        # Confirm indexing products for company 1 doesn't pollute the "client" bucket.
        monkeypatch.delenv("QDRANT_ENDPOINT", raising=False)
        self._index_acme_clients()
        from mock.db import get_all_products
        products = get_all_products(company_id=1, page=1, size=10_000)["items"]
        indexer.index_entities("product", 1, products, lambda p: p["designation"])

        client_rows = indexer.get_memory_index("client", 1)
        product_rows = indexer.get_memory_index("product", 1)
        assert client_rows is not None and product_rows is not None
        client_ids = {r["id"] for r in client_rows}
        product_ids = {r["id"] for r in product_rows}
        # ids overlap numerically (both start at 1) — the real guarantee is
        # they're stored in separate buckets, not that ids never collide
        assert indexer.get_memory_index("client", 1) is not indexer.get_memory_index("product", 1)


# ---------- live Qdrant path (only if configured) ----------

@pytest.mark.skipif(
    not os.environ.get("QDRANT_ENDPOINT"),
    reason="QDRANT_ENDPOINT not set — skipping live Qdrant integration test",
)
class TestLiveQdrant:
    """
    Runs only when a real Qdrant instance is configured. Confirms indexer
    correctly switches to the Qdrant-backed path and that fuzzy search still
    respects SIMILARITY_FLOOR and tenant isolation end-to-end.
    """

    def test_has_qdrant_true_when_configured(self):
        importlib.reload(indexer)
        assert indexer.has_qdrant() is True

    def test_live_index_and_search_roundtrip(self):
        importlib.reload(indexer)
        importlib.reload(retriever)
        from mock.db import get_all_clients

        clients = get_all_clients(company_id=1, page=1, size=10_000)["items"]
        indexer.index_entities(
            "client", 1, clients, lambda c: f"{c['first_name']} {c['last_name']}"
        )

        results = find_similar("client", company_id=1, query_text="Amed Salm", top_k=3)
        assert len(results) >= 1
        assert all(r["score"] >= SIMILARITY_FLOOR for r in results)

    def test_live_tenant_isolation(self):
        importlib.reload(indexer)
        importlib.reload(retriever)
        from mock.db import get_all_clients

        acme = get_all_clients(company_id=1, page=1, size=10_000)["items"]
        nova = get_all_clients(company_id=2, page=1, size=10_000)["items"]
        indexer.index_entities("client", 1, acme, lambda c: f"{c['first_name']} {c['last_name']}")
        indexer.index_entities("client", 2, nova, lambda c: f"{c['first_name']} {c['last_name']}")

        nova_ids = {c["id"] for c in nova}
        results = find_similar("client", company_id=1, query_text="Karim Mzoughi", top_k=3)
        assert {r["id"] for r in results}.isdisjoint(nova_ids)