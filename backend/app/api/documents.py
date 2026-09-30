"""
Documents API Router for IndicRAG
Provides information about ingested policy documents, passages, and chunking statistics.
"""

import json
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, Query, HTTPException
from app.config import DOCUMENTS_DIR, PROCESSED_DIR

router = APIRouter(prefix="/api/documents", tags=["Documents"])


@router.get("")
async def get_documents():
    """List all ingested documents and their statistics."""
    pdf_files = sorted(Path(DOCUMENTS_DIR).glob("*.pdf"))

    # Load chunks metadata
    metadata_file = PROCESSED_DIR / "metadata.json"
    chunks_meta = []
    if metadata_file.exists():
        with open(metadata_file, "r", encoding="utf-8") as f:
            chunks_meta = json.load(f)

    # Document stats
    docs = []
    for pdf in pdf_files:
        matching_chunks = [c for c in chunks_meta if c["source"] == pdf.name]
        pages = set(c["page"] for c in matching_chunks)
        docs.append({
            "filename": pdf.name,
            "size_bytes": pdf.stat().st_size,
            "total_passages": len(matching_chunks),
            "total_pages": len(pages),
            "status": "Indexed" if matching_chunks else "Pending"
        })

    return {
        "documents": docs,
        "total_documents": len(docs),
        "total_passages": len(chunks_meta)
    }


@router.get("/passages")
async def get_passages(
    source: Optional[str] = Query(None, description="Filter by document filename"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100)
):
    """Get paginated passages from chunks.json."""
    chunks_file = PROCESSED_DIR / "chunks.json"
    if not chunks_file.exists():
        raise HTTPException(status_code=404, detail="chunks.json not found. Run ingestion first.")

    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if source:
        chunks = [c for c in chunks if c.get("source") == source]

    total = len(chunks)
    start = (page - 1) * page_size
    end = start + page_size
    paginated = chunks[start:end]

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "passages": paginated
    }
