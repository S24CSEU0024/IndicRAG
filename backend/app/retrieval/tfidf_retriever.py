"""
TF-IDF Lexical Retriever for IndicRAG
Uses scikit-learn TfidfVectorizer with sublinear TF scaling and cosine similarity.
"""

import pickle
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from app.config import TFIDF_DIR


def tokenize(text: str) -> List[str]:
    return re.findall(r'[\w]+', text.lower())


class TfidfRetriever:
    def __init__(self, passages: Optional[List[Dict[str, Any]]] = None):
        self.passages: List[Dict[str, Any]] = passages or []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None

        if self.passages:
            self.build_index(self.passages)

    def build_index(self, passages: List[Dict[str, Any]]):
        self.passages = passages
        corpus = [p["text"] for p in passages]
        self.vectorizer = TfidfVectorizer(
            tokenizer=tokenize,
            token_pattern=None,
            sublinear_tf=True,
            norm="l2"
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if not self.vectorizer or self.tfidf_matrix is None or not self.passages:
            return []

        tokens = tokenize(query)
        if not tokens:
            return []

        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        scored_indices = sorted(
            enumerate(sims),
            key=lambda x: x[1],
            reverse=True
        )

        results = []
        for rank, (idx, score) in enumerate(scored_indices[:k], start=1):
            passage = self.passages[idx].copy()
            passage["score"] = float(score)
            passage["normalized_score"] = float(score)
            passage["rank"] = rank
            passage["retrieval_method"] = "tfidf"
            results.append(passage)

        return results

    def save(self, directory: Optional[Path] = None):
        target_dir = directory or TFIDF_DIR
        target_dir.mkdir(parents=True, exist_ok=True)
        with open(target_dir / "tfidf_model.pkl", "wb") as f:
            pickle.dump({
                "passages": self.passages,
                "vectorizer": self.vectorizer,
                "tfidf_matrix": self.tfidf_matrix
            }, f)
        print(f"TF-IDF index saved to {target_dir}")

    @classmethod
    def load(cls, directory: Optional[Path] = None) -> "TfidfRetriever":
        target_dir = directory or TFIDF_DIR
        model_file = target_dir / "tfidf_model.pkl"
        if not model_file.exists():
            raise FileNotFoundError(f"TF-IDF index file not found at {model_file}")

        with open(model_file, "rb") as f:
            data = pickle.load(f)

        retriever = cls()
        retriever.passages = data["passages"]
        retriever.vectorizer = data["vectorizer"]
        retriever.tfidf_matrix = data["tfidf_matrix"]
        return retriever
