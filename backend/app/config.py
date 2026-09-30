from pathlib import Path
import os
from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parents[1]

load_dotenv(BACKEND_DIR / ".env")

DATA_DIR = BACKEND_DIR / "data"

DOCUMENTS_DIR = DATA_DIR / "documents"
PROCESSED_DIR = DATA_DIR / "processed"
QUESTIONS_DIR = DATA_DIR / "questions"
EVALUATION_DIR = DATA_DIR / "evaluation"

INDEX_DIR = BACKEND_DIR / "indexes"
FAISS_DIR = INDEX_DIR / "faiss"
BM25_DIR = INDEX_DIR / "bm25"
TFIDF_DIR = INDEX_DIR / "tfidf"

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

TOP_K = int(os.getenv("TOP_K", "5"))
CANDIDATE_K = int(os.getenv("CANDIDATE_K", "20"))
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.5"))
ANSWERABILITY_THRESHOLD = float(
    os.getenv("ANSWERABILITY_THRESHOLD", "0.35")
)

for directory in [
    DOCUMENTS_DIR,
    PROCESSED_DIR,
    QUESTIONS_DIR,
    EVALUATION_DIR,
    FAISS_DIR,
    BM25_DIR,
    TFIDF_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)