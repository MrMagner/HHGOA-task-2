"""Evaluation module for benchmarking and retrieval quality metrics."""

from backend.evaluation.benchmark import run_latency_benchmark
from backend.evaluation.retrieval_eval import run_retrieval_evaluation

__all__ = ["run_latency_benchmark", "run_retrieval_evaluation"]
