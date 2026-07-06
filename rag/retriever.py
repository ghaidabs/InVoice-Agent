"""
Semantic fallback for client/product resolution — only invoked when
exact/substring matching (mock/db.py) comes up empty. Literal ops stay
the fast, deterministic path for the common case.
"""
import math
from rag.embeddings import embed_one
from rag.indexer import get_memory_index, has_qdrant, _get_qdrant, _collection_name

SIMILARITY_FLOOR = 0.55

def _cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0

def find_similar(entity_type: str, company_id: int, query_text: str, top_k: int = 3) -> list[dict]:
    query_vec = embed_one(query_text)

    if has_qdrant():
        hits = _get_qdrant().search(
            collection_name=_collection_name(entity_type, company_id),
            query_vector=query_vec, limit=top_k,
        )
        results = [{"id": h.id, "text": h.payload.get("text"), "score": h.score} for h in hits]
    else:
        rows = get_memory_index(entity_type, company_id) or []
        scored = [{"id": r["id"], "text": r["text"], "score": _cosine(query_vec, r["vector"])} for r in rows]
        results = sorted(scored, key=lambda r: r["score"], reverse=True)[:top_k]

    return [r for r in results if r["score"] >= SIMILARITY_FLOOR]