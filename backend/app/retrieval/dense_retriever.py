"""
Dense Multilingual Semantic Retriever for IndicRAG
Uses FAISS IndexFlatIP with normalized multilingual embeddings.
"""

import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import faiss

from app.config import FAISS_DIR
from app.embeddings.embedding_model import encode_texts, encode_query


class DenseRetriever:
    def __init__(self, passages: Optional[List[Dict[str, Any]]] = None):
        self.passages: List[Dict[str, Any]] = passages or []
        self.index: Optional[faiss.IndexFlatIP] = None
        self.dimension: int = 384

        if self.passages:
            self.build_index(self.passages)

    def build_index(self, passages: List[Dict[str, Any]]):
        self.passages = passages
        texts = [p["text"] for p in passages]
        print(f"Generating dense embeddings for {len(texts)} passages...")
        embeddings = encode_texts(texts)
        self.dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        print(f"Dense FAISS index built with {self.index.ntotal} vectors.")

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if self.index is None or not self.passages:
            return []

        query_emb = encode_query(query)
        scores, indices = self.index.search(query_emb, k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx == -1 or idx >= len(self.passages):
                continue
            passage = self.passages[idx].copy()
            passage["score"] = float(score)
            passage["normalized_score"] = float(max(0.0, score))
            passage["rank"] = rank
            passage["retrieval_method"] = "dense"
            results.append(passage)

        return results

    def save(self, directory: Optional[Path] = None):
        target_dir = directory or FAISS_DIR
        target_dir.mkdir(parents=True, exist_ok=True)
        if self.index is not None:
            faiss.write_index(self.index, str(target_dir / "faiss_index.bin"))
        with open(target_dir / "passages.pkl", "wb") as f:
            pickle.dump(self.passages, f)
        print(f"Dense FAISS index and passages saved to {target_dir}")

    @classmethod
    def load(cls, directory: Optional[Path] = None) -> "DenseRetriever":
        target_dir = directory or FAISS_DIR
        index_file = target_dir / "faiss_index.bin"
        passages_file = target_dir / "passages.pkl"

        if not index_file.exists() or not passages_file.exists():
            raise FileNotFoundError(f"FAISS index or passages not found in {target_dir}")

        retriever = cls()
        retriever.index = faiss.read_index(str(index_file))
        with open(passages_file, "rb") as f:
            retriever.passages = pickle.load(f)
        retriever.dimension = retriever.index.d
        return retriever
