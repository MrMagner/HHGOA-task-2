"""Retrieval evaluation using REAL MSMARCO-XI indexed records.

Evaluates Dense, BM25, and Hybrid retrieval strategies against
the actual query→passage ground truth from the indexed dataset.

Every query and passage comes from data/msmarco_xi_real.jsonl.
No synthetic data is used.
"""

import json
import csv
import time
import random
from pathlib import Path

import numpy as np

from backend.config.settings import get_settings
from backend.services.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from backend.services.retrieval.vector_store import VectorStore
from backend.services.retrieval.dense import DenseRetriever
from backend.services.retrieval.bm25 import BM25Retriever
from backend.services.retrieval.hybrid import HybridRetriever


def load_ground_truth(data_file: str, max_queries: int = 100) -> list[dict]:
    """Load real query→passage pairs from the JSONL dataset."""
    records = []
    with open(data_file, "r") as f:
        for line in f:
            row = json.loads(line.strip())
            query = row.get("metadata", {}).get("query", "")
            text = row.get("text", "")
            doc_id = row.get("id", "")
            if query and text and text != "कोई उत्तर नहीं मिला।":
                records.append({
                    "doc_id": str(doc_id),
                    "query": query,
                    "passage": text,
                })
    random.seed(42)
    random.shuffle(records)
    return records[:max_queries]


def build_retrievers(settings):
    """Build all retrieval components from the existing index."""
    vector_store = VectorStore(
        collection_name=settings.qdrant_collection,
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        local_path=settings.qdrant_local_path,
        embedding_dimension=settings.embedding_dimension,
    )

    embedding = SentenceTransformerEmbeddings(
        model_name=settings.embedding_model,
        device=settings.embedding_device,
    )

    # Warm up the model
    embedding.embed_query("warmup")

    dense = DenseRetriever(vector_store=vector_store)

    bm25 = BM25Retriever()
    # Load all docs from Qdrant for BM25
    scroll_res = vector_store._client.scroll(
        collection_name=settings.qdrant_collection,
        limit=2000,
        with_payload=True,
    )
    points = scroll_res[0]
    ids = [str(p.id) for p in points]
    texts = [str(p.payload.get("text", "")) for p in points]
    bm25.build_index(ids, texts)

    hybrid = HybridRetriever(
        dense_retriever=dense,
        bm25_retriever=bm25,
        dense_weight=settings.retrieval_dense_weight,
        bm25_weight=settings.retrieval_bm25_weight,
    )

    return embedding, dense, bm25, hybrid, vector_store


def evaluate_retriever(retriever_name, retriever, embedding, queries, top_k=5):
    """Run retrieval evaluation for a single strategy."""
    results = []
    latencies = []

    for q in queries:
        query_text = q["query"]
        expected_passage = q["passage"]

        start = time.perf_counter()
        query_emb = embedding.embed_query(query_text)
        retrieved = retriever.search(query_text, query_emb, top_k=top_k)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

        # Find if the expected passage is in the results
        hit_rank = 0
        for rank, res in enumerate(retrieved, 1):
            if res.text.strip() == expected_passage.strip():
                hit_rank = rank
                break

        results.append({
            "query": query_text,
            "expected_passage": expected_passage[:80],
            "hit_rank": hit_rank,
            "top_retrieved_text": retrieved[0].text[:80] if retrieved else "",
            "top_score": round(retrieved[0].score, 4) if retrieved else 0.0,
            "num_retrieved": len(retrieved),
        })

    # Calculate metrics
    n = len(results)
    hit_at_1 = sum(1 for r in results if r["hit_rank"] == 1) / n
    hit_at_3 = sum(1 for r in results if 1 <= r["hit_rank"] <= 3) / n
    hit_at_5 = sum(1 for r in results if 1 <= r["hit_rank"] <= 5) / n

    # MRR: Mean Reciprocal Rank
    mrr = sum(1.0 / r["hit_rank"] for r in results if r["hit_rank"] > 0) / n

    # Recall@K: Since each query has exactly 1 relevant doc, recall@k = hit@k
    recall_at_1 = hit_at_1
    recall_at_3 = hit_at_3
    recall_at_5 = hit_at_5

    latency_arr = np.array(latencies)

    return {
        "strategy": retriever_name,
        "queries_evaluated": n,
        "hit_at_1": round(hit_at_1, 4),
        "hit_at_3": round(hit_at_3, 4),
        "hit_at_5": round(hit_at_5, 4),
        "recall_at_1": round(recall_at_1, 4),
        "recall_at_3": round(recall_at_3, 4),
        "recall_at_5": round(recall_at_5, 4),
        "mrr": round(mrr, 4),
        "latency_p50_ms": round(float(np.percentile(latency_arr, 50)), 2),
        "latency_p70_ms": round(float(np.percentile(latency_arr, 70)), 2),
        "latency_p90_ms": round(float(np.percentile(latency_arr, 90)), 2),
        "latency_p95_ms": round(float(np.percentile(latency_arr, 95)), 2),
        "latency_p100_ms": round(float(np.percentile(latency_arr, 100)), 2),
        "per_query": results,
    }


def main():
    settings = get_settings()
    data_file = "data/msmarco_xi_real.jsonl"

    print("Loading ground truth queries...")
    queries = load_ground_truth(data_file, max_queries=50)
    print(f"Loaded {len(queries)} evaluation queries with ground truth.")

    print("Building retrievers...")
    embedding, dense, bm25, hybrid, vector_store = build_retrievers(settings)

    # Evaluate each strategy
    strategies = {
        "dense_only": dense,
        "bm25_only": bm25,
        "hybrid_rrf": hybrid,
    }

    all_results = {}
    for name, retriever in strategies.items():
        print(f"\nEvaluating: {name}...")
        result = evaluate_retriever(name, retriever, embedding, queries, top_k=5)
        all_results[name] = result
        print(f"  Hit@1={result['hit_at_1']}, Hit@3={result['hit_at_3']}, "
              f"Hit@5={result['hit_at_5']}, MRR={result['mrr']}, "
              f"P50={result['latency_p50_ms']}ms")

    # Save JSON
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    json_out = results_dir / "retrieval_evaluation.json"
    summary = {name: {k: v for k, v in r.items() if k != "per_query"} for name, r in all_results.items()}
    with open(json_out, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {json_out}")

    # Save CSV
    csv_out = results_dir / "retrieval_evaluation.csv"
    with open(csv_out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Strategy", "Queries", "Hit@1", "Hit@3", "Hit@5",
                         "Recall@1", "Recall@3", "Recall@5", "MRR",
                         "P50_ms", "P70_ms", "P90_ms", "P95_ms", "P100_ms"])
        for name, r in all_results.items():
            writer.writerow([
                r["strategy"], r["queries_evaluated"],
                r["hit_at_1"], r["hit_at_3"], r["hit_at_5"],
                r["recall_at_1"], r["recall_at_3"], r["recall_at_5"], r["mrr"],
                r["latency_p50_ms"], r["latency_p70_ms"],
                r["latency_p90_ms"], r["latency_p95_ms"], r["latency_p100_ms"],
            ])
    print(f"Saved: {csv_out}")

    # Print comparison table
    print("\n" + "=" * 100)
    print("RETRIEVAL EVALUATION RESULTS")
    print("=" * 100)
    print(f"{'Strategy':<16} {'Hit@1':>7} {'Hit@3':>7} {'Hit@5':>7} "
          f"{'Recall@5':>9} {'MRR':>7} {'P50':>8} {'P70':>8} {'P100':>8}")
    print("-" * 100)
    for name, r in all_results.items():
        print(f"{r['strategy']:<16} {r['hit_at_1']:>7.4f} {r['hit_at_3']:>7.4f} "
              f"{r['hit_at_5']:>7.4f} {r['recall_at_5']:>9.4f} {r['mrr']:>7.4f} "
              f"{r['latency_p50_ms']:>7.2f}ms {r['latency_p70_ms']:>7.2f}ms "
              f"{r['latency_p100_ms']:>7.2f}ms")

    vector_store.close()


if __name__ == "__main__":
    main()
