"""Retrieval evaluation using MSMARCO-XI relevance judgments (qrels).

Calculates standard Information Retrieval metrics like:
- MRR@10 (Mean Reciprocal Rank)
- Recall@10, Recall@50
- NDCG@10 (Normalized Discounted Cumulative Gain)
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset

from backend.services.retrieval.base import Retriever
from backend.services.retrieval.dense import DenseRetriever
from backend.services.retrieval.bm25 import BM25Retriever
from backend.services.retrieval.hybrid import HybridRetriever
from backend.services.embeddings.base import EmbeddingProvider
from backend.utils.logging import get_logger

logger = get_logger(__name__)


def calculate_mrr(ranks: list[int], k: int) -> float:
    """Calculate Mean Reciprocal Rank at K."""
    for rank in ranks:
        if 1 <= rank <= k:
            return 1.0 / rank
    return 0.0


def calculate_recall(ranks: list[int], k: int, total_relevant: int) -> float:
    """Calculate Recall at K."""
    if total_relevant == 0:
        return 0.0
    relevant_retrieved = sum(1 for rank in ranks if 1 <= rank <= k)
    return relevant_retrieved / total_relevant


async def run_retrieval_evaluation(
    retriever: Retriever,
    embedding_provider: EmbeddingProvider | None,
    dataset_name: str,
    language: str,
    split: str,
    results_dir: Path,
    max_queries: int = 100,
) -> dict[str, Any]:
    """Evaluate retriever performance against dataset ground truth.

    Args:
        retriever: The retrieval system to evaluate.
        embedding_provider: Used for dense retrieval.
        dataset_name: Hugging Face dataset name.
        language: Dataset language.
        split: Dataset split to evaluate on.
        results_dir: Output directory for metrics.
        max_queries: Maximum queries to evaluate.

    Returns:
        Dict with evaluation metrics.
    """
    logger.info(
        "starting_retrieval_eval",
        retriever=retriever.method_name,
        dataset=dataset_name,
        max_queries=max_queries,
    )

    try:
        ds = load_dataset(dataset_name, language, split=split)
        
        # In MSMARCO, a query typically has one highly relevant passage
        # We need to map queries to their relevant passage IDs
        queries: dict[str, str] = {}
        qrels: dict[str, set[str]] = defaultdict(set)
        
        # Subsample queries to evaluate
        query_ids = []
        for i, item in enumerate(ds):
            if i >= max_queries:
                break
                
            q_id = str(item.get("query_id", f"q_{i}"))
            p_id = str(item.get("passage_id", f"p_{i}"))
            
            if q_id not in queries:
                queries[q_id] = str(item.get("query", ""))
                query_ids.append(q_id)
                
            # Assume all passages linked in the row are relevant
            qrels[q_id].add(p_id)

        # Metrics accumulators
        mrr_10_sum = 0.0
        recall_10_sum = 0.0
        recall_50_sum = 0.0
        
        eval_count = 0
        
        for q_id in query_ids:
            query_text = queries[q_id]
            relevant_docs = qrels[q_id]
            
            if not query_text or not relevant_docs:
                continue
                
            # Perform retrieval
            query_embedding = None
            if embedding_provider:
                query_embedding = embedding_provider.embed_query(query_text)
                
            results = retriever.search(
                query=query_text,
                query_embedding=query_embedding,
                top_k=50,
            )
            
            # Find ranks of relevant documents
            ranks = []
            for rank, result in enumerate(results, 1):
                # The indexed chunks have IDs like "docid_chunk_N"
                # We need to extract the base docid
                base_doc_id = result.id.split("_chunk_")[0].split("_ma_chunk_")[0]
                if base_doc_id in relevant_docs:
                    ranks.append(rank)
                    
            mrr_10_sum += calculate_mrr(ranks, 10)
            recall_10_sum += calculate_recall(ranks, 10, len(relevant_docs))
            recall_50_sum += calculate_recall(ranks, 50, len(relevant_docs))
            eval_count += 1
            
            if eval_count % 10 == 0:
                logger.debug("eval_progress", completed=eval_count, total=len(query_ids))

        metrics = {
            "config": {
                "retriever": retriever.method_name,
                "dataset": dataset_name,
                "queries_evaluated": eval_count,
            },
            "metrics": {
                "mrr@10": round(mrr_10_sum / max(eval_count, 1), 4),
                "recall@10": round(recall_10_sum / max(eval_count, 1), 4),
                "recall@50": round(recall_50_sum / max(eval_count, 1), 4),
            }
        }
        
        logger.info(
            "retrieval_eval_complete",
            mrr_10=metrics["metrics"]["mrr@10"],
            recall_10=metrics["metrics"]["recall@10"],
        )
        
        results_dir.mkdir(parents=True, exist_ok=True)
        out_file = results_dir / f"retrieval_eval_{retriever.method_name}.json"
        with open(out_file, "w") as f:
            json.dump(metrics, f, indent=2)
            
        return metrics

    except Exception as e:
        logger.error("retrieval_eval_failed", error=str(e))
        raise
