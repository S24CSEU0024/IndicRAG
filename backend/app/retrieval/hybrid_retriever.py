"""
Hybrid Retriever for IndicRAG
Combines Lexical (BM25 or TF-IDF) and Semantic Dense Retrieval.
Supports Linear Weighted Fusion with Min-Max Score Normalization:
  Score(p) = α·Score_lexical + (1−α)·Score_semantic
and Reciprocal Rank Fusion (RRF).
"""

from typing import List, Dict, Any, Optional
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.dense_retriever import DenseRetriever
from app.config import TOP_K, HYBRID_ALPHA, CANDIDATE_K


class HybridRetriever:
    def __init__(
        self,
        lexical_retriever: BM25Retriever,
        dense_retriever: DenseRetriever,
        alpha: float = HYBRID_ALPHA
    ):
        self.lexical_retriever = lexical_retriever
        self.dense_retriever = dense_retriever
        self.alpha = float(alpha)

    def search(
        self,
        query: str,
        k: int = TOP_K,
        alpha: Optional[float] = None,
        candidate_k: int = CANDIDATE_K,
        fusion_mode: str = "weighted"  # 'weighted' or 'rrf'
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid retrieval combining lexical and dense search.
        """
        curr_alpha = self.alpha if alpha is None else float(alpha)

        # Retrieve candidates from both modalities
        lexical_results = self.lexical_retriever.search(query, k=candidate_k)
        dense_results = self.dense_retriever.search(query, k=candidate_k)

        # Build candidate map keyed by chunk_id
        candidates = {}

        # 1. Process Lexical Scores
        max_lex = max([r.get("score", 0.0) for r in lexical_results], default=1.0)
        min_lex = min([r.get("score", 0.0) for r in lexical_results], default=0.0)
        lex_range = (max_lex - min_lex) if (max_lex - min_lex) > 1e-6 else 1.0

        for r in lexical_results:
            cid = r["chunk_id"]
            norm_lex = (r["score"] - min_lex) / lex_range if max_lex > 0 else 0.0
            candidates[cid] = {
                "passage": r,
                "lexical_score": float(r["score"]),
                "norm_lexical_score": float(norm_lex),
                "lexical_rank": int(r["rank"]),
                "dense_score": 0.0,
                "norm_dense_score": 0.0,
                "dense_rank": candidate_k + 1
            }

        # 2. Process Dense Scores
        max_dense = max([r.get("score", 0.0) for r in dense_results], default=1.0)
        min_dense = min([r.get("score", 0.0) for r in dense_results], default=0.0)
        dense_range = (max_dense - min_dense) if (max_dense - min_dense) > 1e-6 else 1.0

        for r in dense_results:
            cid = r["chunk_id"]
            norm_dense = (r["score"] - min_dense) / dense_range if max_dense > min_dense else r["score"]
            if cid not in candidates:
                candidates[cid] = {
                    "passage": r,
                    "lexical_score": 0.0,
                    "norm_lexical_score": 0.0,
                    "lexical_rank": candidate_k + 1,
                    "dense_score": float(r["score"]),
                    "norm_dense_score": float(norm_dense),
                    "dense_rank": int(r["rank"])
                }
            else:
                candidates[cid]["dense_score"] = float(r["score"])
                candidates[cid]["norm_dense_score"] = float(norm_dense)
                candidates[cid]["dense_rank"] = int(r["rank"])

        # 3. Calculate Hybrid Score
        scored_candidates = []
        rrf_k = 60

        for cid, data in candidates.items():
            if fusion_mode == "rrf":
                score = (
                    curr_alpha * (1.0 / (rrf_k + data["lexical_rank"])) +
                    (1.0 - curr_alpha) * (1.0 / (rrf_k + data["dense_rank"]))
                )
            else:  # 'weighted' linear fusion
                score = (
                    curr_alpha * data["norm_lexical_score"] +
                    (1.0 - curr_alpha) * data["norm_dense_score"]
                )

            entry = data["passage"].copy()
            entry["score"] = float(score)
            entry["hybrid_score"] = float(score)
            entry["lexical_score"] = data["lexical_score"]
            entry["dense_score"] = data["dense_score"]
            entry["retrieval_method"] = "hybrid"
            scored_candidates.append(entry)

        scored_candidates.sort(key=lambda x: x["score"], reverse=True)

        # Set final rank
        for rank, item in enumerate(scored_candidates[:k], start=1):
            item["rank"] = rank

        return scored_candidates[:k]
