from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
from mock.db import get_all_clients, get_all_products, get_all_invoices
from .env import QDRANT_endpoint, QDRANT_port

COLLECTION = "tenant_demo"
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
client = QdrantClient(host=QDRANT_endpoint, port=QDRANT_port)

def build_index():
    client.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    points = []
    uid = 0

    for c in get_all_clients():
        full_name = f"{c['first_name']} {c['last_name']}"
        vec = model.encode(full_name).tolist()
        points.append(PointStruct(
            id=uid, vector=vec,
            payload={"type": "client", "data": c, "full_name": full_name}
        ))
        uid += 1

    for p in get_all_products():
        vec = model.encode(p["name"]).tolist()
        points.append(PointStruct(
            id=uid, vector=vec,
            payload={"type": "product", "data": p}
        ))
        uid += 1

    for i in get_all_invoices():
        vec = model.encode(i["description"]).tolist()
        points.append(PointStruct(
            id=uid, vector=vec,
            payload={"type": "invoice", "data": i}
        ))
        uid += 1

    client.upsert(collection_name=COLLECTION, points=points)
    print(f"Indexed {len(points)} items into '{COLLECTION}'")

if __name__ == "__main__":
    build_index()