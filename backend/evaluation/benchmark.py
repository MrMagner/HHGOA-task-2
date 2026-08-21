"""End-to-end latency benchmarking for the RAG pipeline.

Measures latency percentiles (P50, P90, P95, P99) for all stages
of the pipeline to ensure the system meets real-time requirements
(<200ms for core retrieval).
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset

from backend.pipeline.orchestrator import VoiceRAGPipeline
from backend.utils.logging import get_logger

logger = get_logger(__name__)


async def run_latency_benchmark(
    pipeline: VoiceRAGPipeline,
    queries: list[str],
    results_dir: Path,
    num_runs_per_query: int = 3,
    top_k: int = 5,
    rerank: bool = False,
) -> dict[str, Any]:
    """Run a comprehensive latency benchmark across the pipeline.

    Args:
        pipeline: Initialized VoiceRAGPipeline.
        queries: List of test queries.
        results_dir: Directory to save benchmark results.
        num_runs_per_query: Number of times to run each query.
        top_k: Top-k parameter for retrieval.
        rerank: Whether to enable reranking.

    Returns:
        Dictionary with benchmark metrics.
    """
    logger.info(
        "starting_benchmark",
        queries=len(queries),
        runs_per_query=num_runs_per_query,
        top_k=top_k,
        rerank=rerank,
    )

    # Warmup
    if queries:
        logger.info("running_warmup")
        await pipeline.process_text_query(queries[0], "warmup", top_k=top_k, rerank=rerank)

    latencies: dict[str, list[float]] = defaultdict(list)
    total_queries_run = 0

    for query in queries:
        for _ in range(num_runs_per_query):
            total_queries_run += 1
            request_id = f"bench_{total_queries_run}"
            
            # Text query benchmark
            response = await pipeline.process_text_query(
                query=query,
                request_id=request_id,
                top_k=top_k,
                rerank=rerank,
            )
            
            l = response.latency
            
            # Record component latencies
            if l.embedding_ms is not None:
                latencies["embedding"].append(l.embedding_ms)
            if l.retrieval_ms is not None:
                latencies["retrieval"].append(l.retrieval_ms)
            if l.reranking_ms is not None:
                latencies["reranking"].append(l.reranking_ms)
            if l.generation_ms is not None:
                latencies["generation"].append(l.generation_ms)
            if l.guardrails_ms is not None:
                latencies["guardrails"].append(l.guardrails_ms)
                
            latencies["total_text"].append(l.total_ms)

    # Calculate percentiles
    metrics = {
        "config": {
            "queries": len(queries),
            "runs_per_query": num_runs_per_query,
            "top_k": top_k,
            "rerank": rerank,
        },
        "percentiles": {},
    }
    
    for stage, times in latencies.items():
        if not times:
            continue
            
        metrics["percentiles"][stage] = {
            "mean": round(float(np.mean(times)), 2),
            "p50": round(float(np.percentile(times, 50)), 2),
            "p70": round(float(np.percentile(times, 70)), 2),
            "p90": round(float(np.percentile(times, 90)), 2),
            "p95": round(float(np.percentile(times, 95)), 2),
            "p99": round(float(np.percentile(times, 99)), 2),
            "p100": round(float(np.percentile(times, 100)), 2),
        }

    # Log summary
    logger.info(
        "benchmark_complete",
        total_mean=metrics["percentiles"].get("total_text", {}).get("mean"),
        total_p95=metrics["percentiles"].get("total_text", {}).get("p95"),
        retrieval_mean=metrics["percentiles"].get("retrieval", {}).get("mean"),
    )

    # Save to disk
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / "latency_benchmark.json"
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics
