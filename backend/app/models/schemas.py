"""
Pydantic Schemas for IndicRAG-QA API
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: Optional[str] = None
    query: Optional[str] = None
    retrieval_method: Optional[str] = Field(default="hybrid", description="hybrid, dense, bm25, or tfidf")
    top_k: Optional[int] = Field(default=5, ge=1, le=20)
    alpha: Optional[float] = Field(default=0.5, ge=0.0, le=1.0)
    threshold: Optional[float] = Field(default=0.35, ge=0.0, le=1.0)
    direct_llm: Optional[bool] = Field(default=False, description="Direct LLM QA without context")

    def get_query(self) -> str:
        return (self.question or self.query or "").strip()


class SourceDocument(BaseModel):
    chunk_id: Optional[int] = None
    source: Optional[str] = None
    document: Optional[str] = None
    page: Optional[int] = None
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    score: Optional[float] = None
    text: Optional[str] = None


class VerificationResult(BaseModel):
    has_citations: bool = False
    cited_ranks: List[int] = []
    valid_citations: List[int] = []
    invalid_citations: List[int] = []
    citation_validity_ratio: float = 1.0
    all_citations_valid: bool = True


class HallucinationCheck(BaseModel):
    is_grounded: bool = True
    groundedness_score: float = 1.0
    unsupported_entities: List[str] = []
    total_entities_checked: int = 0


class ChatResponse(BaseModel):
    query: str
    detected_language: str
    is_code_mixed: bool
    answerability: str
    answerable: bool
    confidence: float
    answer: str
    response: str
    explanation: str
    retrieval_method: str
    supporting_passages: List[SourceDocument] = []
    sources: List[SourceDocument] = []
    evidence_quality: str
    verification: Optional[VerificationResult] = None
    hallucination_check: Optional[HallucinationCheck] = None


class EvaluationRequest(BaseModel):
    modules: Optional[List[int]] = Field(default=[1, 2, 3, 4, 5, 6])
    sample_size: Optional[int] = Field(default=50, ge=5, le=500)
