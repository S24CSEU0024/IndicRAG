"""
Passage Chunker for IndicRAG
Segments document pages into semantically cohesive, retrievable passages
with unique passage IDs, source citations, and metadata.
"""

import re
from typing import List, Dict, Any


def clean_text(text: str) -> str:
    """Normalize whitespace and remove repetitive artifacts."""
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def extract_section_title(text: str, default: str = "General Regulations") -> str:
    """Extract first heading or topic from the passage."""
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    for line in lines[:3]:
        if re.match(r'^(?:[0-9]+[\.\)]|[A-Z\s]{4,}|Section|Article|Rule|Annexure|Undertaking|Scholarship|Examination)', line, re.IGNORECASE):
            return line[:100]
    return default


def create_chunks(
    documents: List[Dict[str, Any]],
    chunk_size: int = 500,
    overlap: int = 100
) -> List[Dict[str, Any]]:
    """
    Segment document pages into passages with document/passage identifiers.
    Splits long pages into multi-passage windows with overlap.
    """
    chunks = []
    chunk_counter = 0

    for doc in documents:
        text = clean_text(doc.get("text", ""))
        source = doc.get("source", "unknown")
        page = doc.get("page", 1)

        if not text or len(text) < 30:
            continue

        # If text is already compact enough, keep as single chunk
        if len(text) <= chunk_size + 100:
            chunks.append({
                "chunk_id": chunk_counter,
                "doc_id": source,
                "source": source,
                "page": page,
                "section_title": extract_section_title(text, default=f"{source} - Page {page}"),
                "text": text,
                "char_count": len(text)
            })
            chunk_counter += 1
            continue

        # Split longer text by sentences/paragraphs with sliding window
        start = 0
        while start < len(text):
            end = start + chunk_size

            # If we're not at the very end, try to break at a newline or period
            if end < len(text):
                # Look for sentence boundary or newline in the last 100 chars of window
                window_slice = text[start:end]
                break_point = -1
                for punct in ['\n\n', '\n', '. ', '? ', '; ']:
                    pos = window_slice.rfind(punct)
                    if pos > chunk_size * 0.6:
                        break_point = pos + len(punct)
                        break
                if break_point != -1:
                    end = start + break_point

            chunk_text = text[start:end].strip()
            if len(chunk_text) >= 40:
                chunks.append({
                    "chunk_id": chunk_counter,
                    "doc_id": source,
                    "source": source,
                    "page": page,
                    "section_title": extract_section_title(chunk_text, default=f"{source} - Page {page}"),
                    "text": chunk_text,
                    "char_count": len(chunk_text)
                })
                chunk_counter += 1

            if end >= len(text):
                break
            start = end - overlap

    return chunks