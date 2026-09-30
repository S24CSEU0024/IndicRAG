import fitz
from pathlib import Path


def load_pdf(pdf_path):
    doc = fitz.open(pdf_path)

    pages = []

    for page_number, page in enumerate(doc):
        text = page.get_text()

        if text.strip():
            pages.append({
                "page": page_number + 1,
                "text": text
            })

    return pages


def load_all_documents(folder):
    documents = []

    folder = Path(folder)

    for pdf_file in folder.glob("*.pdf"):

        pages = load_pdf(pdf_file)

        for page in pages:
            documents.append({
                "source": pdf_file.name,
                "page": page["page"],
                "text": page["text"]
            })

    return documents