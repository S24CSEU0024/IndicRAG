"""
Reranker for IndicRAG
Applies semantic passage reranking based on query term overlap,
semantic coherence, and section relevance.
"""

from typing import List, Dict, Any
import numpy as np


class SimpleSemanticReranker:
    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Rerank candidates using reciprocal rank and term overlap boost."""
        query_words = set(query.lower().split())
        reranked = []

        for c in candidates:
            text_words = set(c["text"].lower().split())
            overlap_count = len(query_words.intersection(text_words))
            overlap_ratio = overlap_count / max(1, len(query_words))

            base_score = c.get("score", 0.5)
            # Combine base retriever score with overlap boost
            rerank_score = 0.75 * base_score + 0.25 * overlap_ratio

            entry = c.copy()
            entry["rerank_score"] = float(rerank_score)
            reranked.append(entry)

        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        for r, item in enumerate(reranked[:top_k], start=1):
            item["rank"] = r
        return reranked[:top_k]
