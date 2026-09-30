"""
Build TF-IDF Index Script
Reads chunks.json and builds serialized TF-IDF index.
"""

import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.config import PROCESSED_DIR, TFIDF_DIR
from app.retrieval.tfidf_retriever import TfidfRetriever


def main():
    print("Building TF-IDF Index...")
    chunks_file = PROCESSED_DIR / "chunks.json"
    if not chunks_file.exists():
        print(f"Error: {chunks_file} not found. Run ingest.py first.")
        return

    with open(chunks_file, "r", encoding="utf-8") as f:
        passages = json.load(f)

    print(f"Loaded {len(passages)} passages.")
    retriever = TfidfRetriever(passages)
    retriever.save(TFIDF_DIR)
    print("TF-IDF index successfully built and saved.")


if __name__ == "__main__":
    main()
