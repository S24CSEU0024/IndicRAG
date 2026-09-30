"""
Retrieval Metrics for IndicRAG
Implements Recall@K, Precision@K, Hit Rate@K, and Mean Reciprocal Rank (MRR).
"""

from typing import List, Set, Union


def hit_rate_at_k(retrieved: List[Union[int, str]], relevant: Set[Union[int, str]], k: int = 5) -> float:
    """Returns 1.0 if at least one relevant item is in top-K, else 0.0."""
    top_k = retrieved[:k]
    return 1.0 if any(item in relevant for item in top_k) else 0.0


def precision_at_k(retrieved: List[Union[int, str]], relevant: Set[Union[int, str]], k: int = 5) -> float:
    """Fraction of top-K retrieved items that are relevant."""
    if k == 0:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k


def recall_at_k(retrieved: List[Union[int, str]], relevant: Set[Union[int, str]], k: int = 5) -> float:
    """Fraction of all relevant items that appear in top-K."""
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def reciprocal_rank(retrieved: List[Union[int, str]], relevant: Set[Union[int, str]]) -> float:
    """1 / rank of first relevant item retrieved, or 0.0 if not found."""
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def compute_retrieval_metrics(
    all_retrieved: List[List[Union[int, str]]],
    all_relevant: List[Set[Union[int, str]]],
    k_list: List[int] = [1, 3, 5]
) -> dict:
    """
    Computes aggregate Recall@K, Precision@K, HitRate@K, and MRR across queries.
    """
    n = len(all_retrieved)
    if n == 0:
        return {}

    metrics = {}
    for k in k_list:
        metrics[f"Recall@{k}"] = round(sum(recall_at_k(ret, rel, k) for ret, rel in zip(all_retrieved, all_relevant)) / n, 4)
        metrics[f"Precision@{k}"] = round(sum(precision_at_k(ret, rel, k) for ret, rel in zip(all_retrieved, all_relevant)) / n, 4)
        metrics[f"HitRate@{k}"] = round(sum(hit_rate_at_k(ret, rel, k) for ret, rel in zip(all_retrieved, all_relevant)) / n, 4)

    metrics["MRR"] = round(sum(reciprocal_rank(ret, rel) for ret, rel in zip(all_retrieved, all_relevant)) / n, 4)
    return metrics
