"""
Document Ingestion Script for IndicRAG
Extracts text from all PDFs, performs OCR when needed, creates passages with identifiers,
and saves chunks.json and metadata.json.
"""

import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.ingestion.pdf_loader import load_pdfs
from app.ingestion.chunker import create_chunks
from app.config import PROCESSED_DIR


def main():
    print("==================================================")
    print("IndicRAG: Document Ingestion & Passage Segmentation")
    print("==================================================")

    # Load PDF pages (with Apple Vision OCR fallback and caching)
    documents = load_pdfs(use_cache=False)
    print(f"Loaded {len(documents)} total pages across documents.")

    if not documents:
        print("No PDF documents found or extracted.")
        return

    # Create retrievable passages
    chunks = create_chunks(
        documents,
        chunk_size=550,
        overlap=120
    )
    print(f"Created {len(chunks)} retrievable passages.")

    # Save chunks
    chunks_file = PROCESSED_DIR / "chunks.json"
    with open(chunks_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    # Save metadata
    metadata = [
        {
            "chunk_id": c["chunk_id"],
            "doc_id": c["doc_id"],
            "source": c["source"],
            "page": c["page"],
            "section_title": c.get("section_title", ""),
            "char_count": c.get("char_count", len(c["text"]))
        }
        for c in chunks
    ]
    metadata_file = PROCESSED_DIR / "metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Passages saved to: {chunks_file}")
    print(f"Metadata saved to: {metadata_file}")
    print("Ingestion completed successfully.")


if __name__ == "__main__":
    main()
