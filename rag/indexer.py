"""
Per-company vector index for clients/products. Falls back to an in-memory
index when QDRANT_ENDPOINT isn't set, so this works in local/dev/mock
environments without a live Qdrant instance — same call interface either way.
"""
import os
from rag.embeddings import embed

QDRANT_ENDPOINT = os.environ.get("QDRANT_ENDPOINT")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")

_MEMORY_INDEX: dict[str, dict[int, list[dict]]] = {"client": {}, "product": {}}
_qdrant_client = None

def _get_qdrant():
    global _qdrant_client
    if _qdrant_client is None and QDRANT_ENDPOINT:
        from qdrant_client import QdrantClient
        _qdrant_client = QdrantClient(url=QDRANT_ENDPOINT, api_key=QDRANT_API_KEY)
    return _qdrant_client

def has_qdrant() -> bool:
    return _get_qdrant() is not None

def _collection_name(entity_type: str, company_id: int) -> str:
    return f"{entity_type}_{company_id}"

def index_entities(entity_type: str, company_id: int, entities: list[dict], text_fn) -> None:
    texts = [text_fn(e) for e in entities]
    vectors = embed(texts)
    client = _get_qdrant()

    if client is None:
        _MEMORY_INDEX[entity_type][company_id] = [
            {"id": e["id"], "text": t, "vector": v} for e, t, v in zip(entities, texts, vectors)
        ]
        return

    from qdrant_client.models import VectorParams, Distance, PointStruct
    name = _collection_name(entity_type, company_id)
    if name not in {c.name for c in client.get_collections().collections}:
        client.create_collection(name, vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE))
    client.upsert(collection_name=name, points=[
        PointStruct(id=e["id"], vector=v, payload={"text": t}) for e, t, v in zip(entities, texts, vectors)
    ])

def get_memory_index(entity_type: str, company_id: int) -> list[dict] | None:
    return _MEMORY_INDEX.get(entity_type, {}).get(company_id)

def ensure_client_index(company_id: int) -> None:
    if get_memory_index("client", company_id) is not None and not has_qdrant():
        return
    from mock.db import get_all_clients
    clients = get_all_clients(company_id, page=1, size=10_000)["items"]
    index_entities("client", company_id, clients, lambda c: f"{c['first_name']} {c['last_name']}")

def ensure_product_index(company_id: int) -> None:
    if get_memory_index("product", company_id) is not None and not has_qdrant():
        return
    from mock.db import get_all_products
    products = get_all_products(company_id, page=1, size=10_000)["items"]
    index_entities("product", company_id, products, lambda p: p["designation"])