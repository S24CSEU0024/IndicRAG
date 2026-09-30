"""
PDF Document Loader with native text extraction and Apple Vision OCR fallback
for scanned university regulation documents.
"""

from pathlib import Path
import json
import subprocess
import os
import pymupdf

from app.config import DOCUMENTS_DIR, PROCESSED_DIR, BACKEND_DIR


OCR_BINARY = BACKEND_DIR / "scripts" / "ocr_page"
CACHE_FILE = PROCESSED_DIR / "extracted_pages.json"


def run_ocr(pixmap_path: str) -> str:
    """Run compiled Apple Vision OCR tool on image."""
    if not OCR_BINARY.exists():
        # compile on the fly if needed
        swift_file = BACKEND_DIR / "scripts" / "ocr_page.swift"
        if swift_file.exists():
            subprocess.run(
                ["swiftc", "-O", str(swift_file), "-o", str(OCR_BINARY)],
                check=False
            )

    if OCR_BINARY.exists():
        try:
            res = subprocess.run(
                [str(OCR_BINARY), pixmap_path],
                capture_output=True,
                text=True,
                timeout=30
            )
            return res.stdout.strip()
        except Exception as e:
            print(f"OCR binary execution failed: {e}")
            return ""
    return ""


def load_pdfs(use_cache: bool = True):
    """
    Load all PDFs from DOCUMENTS_DIR.
    Uses cached extracted pages if available.
    Returns list of dicts: [{'source': str, 'page': int, 'text': str}]
    """
    if use_cache and CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                cached_docs = json.load(f)
                if cached_docs and len(cached_docs) > 0:
                    print(f"Loaded {len(cached_docs)} pages from cache: {CACHE_FILE}")
                    return cached_docs
        except Exception as e:
            print(f"Cache read error: {e}, re-extracting...")

    documents = []
    pdf_files = sorted(Path(DOCUMENTS_DIR).glob("*.pdf"))

    print(f"Extracting text from {len(pdf_files)} PDF files in {DOCUMENTS_DIR}...")
    temp_img = Path("/tmp/indic_ocr_tmp.png")

    for pdf_path in pdf_files:
        print(f"Processing {pdf_path.name}...")
        try:
            doc = pymupdf.open(pdf_path)
            doc_pages = 0

            for page_number, page in enumerate(doc, start=1):
                raw_text = page.get_text("text") or ""
                clean_raw = raw_text.strip()

                # If page has sufficient native text, use it
                if len(clean_raw) >= 50:
                    text = clean_raw
                else:
                    # Scanned page - use high-accuracy Apple Vision OCR
                    try:
                        pix = page.get_pixmap(dpi=140)
                        pix.save(str(temp_img))
                        ocr_text = run_ocr(str(temp_img))
                        text = ocr_text if len(ocr_text) >= 10 else clean_raw
                    except Exception as err:
                        print(f"OCR error on {pdf_path.name} p.{page_number}: {err}")
                        text = clean_raw

                if text.strip():
                    documents.append({
                        "source": pdf_path.name,
                        "page": page_number,
                        "text": text.strip()
                    })
                    doc_pages += 1

            doc.close()
            print(f"Extracted {doc_pages} pages from {pdf_path.name}")
        except Exception as e:
            print(f"Error reading {pdf_path}: {e}")

    if temp_img.exists():
        temp_img.unlink(missing_ok=True)

    # Cache extracted pages for fast future reloads
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)
        print(f"Successfully cached {len(documents)} pages to {CACHE_FILE}")
    except Exception as e:
        print(f"Failed to cache pages: {e}")

    return documents
