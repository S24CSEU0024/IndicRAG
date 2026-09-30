"""
Build BM25 Index Script
Reads chunks.json and builds serialized BM25 index.
"""

import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.config import PROCESSED_DIR, BM25_DIR
from app.retrieval.bm25_retriever import BM25Retriever


def main():
    print("Building BM25 Index...")
    chunks_file = PROCESSED_DIR / "chunks.json"
    if not chunks_file.exists():
        print(f"Error: {chunks_file} not found. Run ingest.py first.")
        return

    with open(chunks_file, "r", encoding="utf-8") as f:
        passages = json.load(f)

    print(f"Loaded {len(passages)} passages.")
    retriever = BM25Retriever(passages)
    retriever.save(BM25_DIR)
    print("BM25 index successfully built and saved.")


if __name__ == "__main__":
    main()
