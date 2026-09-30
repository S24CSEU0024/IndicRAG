"""
Build Dense FAISS Index Script
Reads chunks.json, generates embeddings with multilingual model, and builds FAISS index.
"""

import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.config import PROCESSED_DIR, FAISS_DIR
from app.retrieval.dense_retriever import DenseRetriever


def main():
    print("Building Dense FAISS Index...")
    chunks_file = PROCESSED_DIR / "chunks.json"
    if not chunks_file.exists():
        print(f"Error: {chunks_file} not found. Run ingest.py first.")
        return

    with open(chunks_file, "r", encoding="utf-8") as f:
        passages = json.load(f)

    print(f"Loaded {len(passages)} passages.")
    retriever = DenseRetriever(passages)
    retriever.save(FAISS_DIR)
    print("Dense FAISS index successfully built and saved.")


if __name__ == "__main__":
    main()
