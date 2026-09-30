"""
Multilingual Embedding Model Wrapper for IndicRAG
Uses sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
with Apple Silicon MPS GPU acceleration when available.
"""

from typing import List, Union
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from app.config import EMBEDDING_MODEL

_MODEL_INSTANCE: SentenceTransformer = None


def get_embedding_model() -> SentenceTransformer:
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Loading embedding model: {EMBEDDING_MODEL} on device: {device}...")
        _MODEL_INSTANCE = SentenceTransformer(EMBEDDING_MODEL, device=device)
        print("Embedding model loaded successfully.")
    return _MODEL_INSTANCE


def encode_texts(texts: List[str], batch_size: int = 64) -> np.ndarray:
    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return np.array(embeddings, dtype="float32")


def encode_query(query: str) -> np.ndarray:
    model = get_embedding_model()
    embedding = model.encode(
        [query],
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return np.array(embedding, dtype="float32")
