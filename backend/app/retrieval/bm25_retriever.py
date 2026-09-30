"""
BM25 Lexical Retriever for IndicRAG
Uses BM25Okapi with multilingual tokenization and index serialization.
"""

import pickle
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from app.config import BM25_DIR


def tokenize(text: str) -> List[str]:
    """Tokenize English, Indic scripts, and alphanumeric tokens."""
    # Matches words in any script (including Devanagari)
    tokens = re.findall(r'[\w]+', text.lower())
    return tokens


class BM25Retriever:
    def __init__(self, passages: Optional[List[Dict[str, Any]]] = None):
        self.passages: List[Dict[str, Any]] = passages or []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None

        if self.passages:
            self.build_index(self.passages)

    def build_index(self, passages: List[Dict[str, Any]]):
        """Build BM25 index on passage list."""
        self.passages = passages
        self.tokenized_corpus = [tokenize(p["text"]) for p in passages]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top-K passages by BM25 relevance score."""
        if not self.bm25 or not self.passages:
            return []

        tokens = tokenize(query)
        if not tokens:
            return []

        raw_scores = self.bm25.get_scores(tokens)
        scored_indices = sorted(
            enumerate(raw_scores),
            key=lambda x: x[1],
            reverse=True
        )

        max_score = scored_indices[0][1] if scored_indices and scored_indices[0][1] > 0 else 1.0

        results = []
        for rank, (idx, score) in enumerate(scored_indices[:k], start=1):
            passage = self.passages[idx].copy()
            passage["score"] = float(score)
            passage["normalized_score"] = float(score / max_score) if max_score > 0 else 0.0
            passage["rank"] = rank
            passage["retrieval_method"] = "bm25"
            results.append(passage)

        return results

    def save(self, directory: Optional[Path] = None):
        """Save index and passages to disk."""
        target_dir = directory or BM25_DIR
        target_dir.mkdir(parents=True, exist_ok=True)
        with open(target_dir / "bm25_model.pkl", "wb") as f:
            pickle.dump({
                "passages": self.passages,
                "tokenized_corpus": self.tokenized_corpus,
                "bm25": self.bm25
            }, f)
        print(f"BM25 index saved to {target_dir}")

    @classmethod
    def load(cls, directory: Optional[Path] = None) -> "BM25Retriever":
        """Load saved BM25 index from disk."""
        target_dir = directory or BM25_DIR
        model_file = target_dir / "bm25_model.pkl"
        if not model_file.exists():
            raise FileNotFoundError(f"BM25 index file not found at {model_file}")

        with open(model_file, "rb") as f:
            data = pickle.load(f)

        retriever = cls()
        retriever.passages = data["passages"]
        retriever.tokenized_corpus = data["tokenized_corpus"]
        retriever.bm25 = data["bm25"]
        return retriever
