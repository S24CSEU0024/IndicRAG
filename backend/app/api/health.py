"""
Health & System Status API Router for IndicRAG
"""

from fastapi import APIRouter
import torch
from app.config import FAISS_DIR, BM25_DIR, TFIDF_DIR, PROCESSED_DIR

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health():
    mps_available = torch.backends.mps.is_available()
    cuda_available = torch.cuda.is_available()
    device = "mps" if mps_available else ("cuda" if cuda_available else "cpu")

    faiss_ready = (FAISS_DIR / "faiss_index.bin").exists()
    bm25_ready = (BM25_DIR / "bm25_model.pkl").exists()
    tfidf_ready = (TFIDF_DIR / "tfidf_model.pkl").exists()
    chunks_ready = (PROCESSED_DIR / "chunks.json").exists()

    all_ready = faiss_ready and bm25_ready and chunks_ready

    return {
        "status": "healthy" if all_ready else "initializing",
        "system": "IndicRAG-QA",
        "version": "1.0.0",
        "hardware_acceleration": device,
        "indexes": {
            "faiss": faiss_ready,
            "bm25": bm25_ready,
            "tfidf": tfidf_ready,
            "chunks": chunks_ready
        }
    }
