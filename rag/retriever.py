import rag.indexer as indexer

SIMILARITY_FLOOR = 0.55


def find_similar(entity_type: str, company_id: int, query_text: str, top_k: int = 3) -> list[dict]:
    from rag.embeddings import embed_one
    query_vec = embed_one(query_text)

    hits = indexer.get_qdrant().search(
        collection_name=indexer.collection_name(entity_type, company_id),
        query_vector=query_vec,
        limit=top_k,
    )
    results = [{"id": h.id, "text": h.payload.get("text"), "score": h.score} for h in hits]

    return [r for r in results if r["score"] >= SIMILARITY_FLOOR]