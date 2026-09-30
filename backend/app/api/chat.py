"""
Chat API Router for IndicRAG-QA
Handles evidence-grounded multilingual QA queries.
"""

from fastapi import APIRouter, HTTPException
from app.models.schemas import ChatRequest, ChatResponse
from app.generations.rag_generator import IndicRAGPipeline
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.tfidf_retriever import TfidfRetriever
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.config import TOP_K, HYBRID_ALPHA, ANSWERABILITY_THRESHOLD

router = APIRouter(prefix="/api", tags=["Chat"])

# Lazy pipeline singleton
_pipeline_instance = None


def get_pipeline() -> IndicRAGPipeline:
    global _pipeline_instance
    if _pipeline_instance is None:
        try:
            bm25 = BM25Retriever.load()
        except Exception:
            bm25 = None

        try:
            tfidf = TfidfRetriever.load()
        except Exception:
            tfidf = None

        try:
            dense = DenseRetriever.load()
        except Exception:
            dense = None

        hybrid = None
        if bm25 and dense:
            hybrid = HybridRetriever(bm25, dense, alpha=HYBRID_ALPHA)

        _pipeline_instance = IndicRAGPipeline(
            hybrid_retriever=hybrid,
            bm25_retriever=bm25,
            dense_retriever=dense,
            tfidf_retriever=tfidf
        )
    return _pipeline_instance


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    query_text = request.get_query()
    if not query_text:
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")

    pipeline = get_pipeline()
    result = pipeline.process_query(
        raw_query=query_text,
        retrieval_method=request.retrieval_method or "hybrid",
        top_k=request.top_k or TOP_K,
        alpha=request.alpha if request.alpha is not None else HYBRID_ALPHA,
        threshold=request.threshold if request.threshold is not None else ANSWERABILITY_THRESHOLD,
        direct_llm=bool(request.direct_llm)
    )

    return result
