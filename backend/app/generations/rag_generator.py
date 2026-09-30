"""
Complete RAG Generation Pipeline for IndicRAG
Coordinates:
  1. Language detection & query normalization / code-mixed expansion
  2. Hybrid / Dense / Lexical Retrieval
  3. Answerability Detection & Confidence Evaluation
  4. Context-Grounded Answer Generation with Citations
  5. Citation & Hallucination Verification
"""

from typing import Dict, Any, List, Optional
from app.language.detector import detect_language
from app.language.normalizer import normalize_query
from app.language.codemix import expand_codemixed_query
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.tfidf_retriever import TfidfRetriever
from app.answerability.detector import AnswerabilityDetector
from app.generations.prompt import build_rag_prompt, build_direct_llm_prompt
from app.generations.llm import LLMClient
from app.verifications.citation_checker import CitationChecker
from app.verifications.evidence_checker import EvidenceChecker
from app.config import TOP_K, HYBRID_ALPHA, ANSWERABILITY_THRESHOLD


class IndicRAGPipeline:
    def __init__(
        self,
        hybrid_retriever: Optional[HybridRetriever] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
        dense_retriever: Optional[DenseRetriever] = None,
        tfidf_retriever: Optional[TfidfRetriever] = None
    ):
        self.hybrid_retriever = hybrid_retriever
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.tfidf_retriever = tfidf_retriever
        self.detector = AnswerabilityDetector(threshold=ANSWERABILITY_THRESHOLD)
        self.llm = LLMClient()

    def process_query(
        self,
        raw_query: str,
        retrieval_method: str = "hybrid",  # "hybrid", "dense", "bm25", "tfidf"
        top_k: int = TOP_K,
        alpha: float = HYBRID_ALPHA,
        threshold: float = ANSWERABILITY_THRESHOLD,
        direct_llm: bool = False
    ) -> Dict[str, Any]:
        """
        Executes end-to-end evidence-grounded cross-lingual RAG.
        """
        # 1. Query Processing & Language Detection
        lang_info = detect_language(raw_query)
        detected_language = lang_info["language"]
        is_code_mixed = lang_info["is_code_mixed"]

        normalized_query = normalize_query(raw_query)
        expanded_query = normalized_query
        codemix_concepts = []

        if is_code_mixed:
            expanded_query, codemix_concepts = expand_codemixed_query(normalized_query)

        # Module 4: Direct LLM baseline without retrieval
        if direct_llm:
            direct_prompt = build_direct_llm_prompt(raw_query)
            direct_answer = self.llm.generate(direct_prompt)
            return {
                "query": raw_query,
                "detected_language": detected_language,
                "is_code_mixed": is_code_mixed,
                "answerability": "ANSWERABLE",
                "answer": direct_answer,
                "confidence": 0.50,
                "supporting_passages": [],
                "sources": [],
                "retrieval_method": "direct_llm",
                "explanation": "Answer generated directly by LLM internal knowledge without retrieval."
            }

        # 2. Retrieval according to requested method
        search_query = expanded_query if (is_code_mixed and retrieval_method in ["bm25", "tfidf", "hybrid"]) else normalized_query

        if retrieval_method == "bm25" and self.bm25_retriever:
            passages = self.bm25_retriever.search(search_query, k=top_k)
        elif retrieval_method == "tfidf" and self.tfidf_retriever:
            passages = self.tfidf_retriever.search(search_query, k=top_k)
        elif retrieval_method == "dense" and self.dense_retriever:
            passages = self.dense_retriever.search(normalized_query, k=top_k)
        elif self.hybrid_retriever:
            passages = self.hybrid_retriever.search(
                search_query,
                k=top_k,
                alpha=alpha
            )
        elif self.dense_retriever:
            passages = self.dense_retriever.search(normalized_query, k=top_k)
        else:
            passages = []

        # 3. Answerability Detection
        answerability_res = self.detector.detect(
            raw_query,
            passages,
            custom_threshold=threshold
        )
        is_answerable = answerability_res["answerable"]
        confidence = answerability_res["confidence"]

        # 4. Answer Generation & Grounding
        if not is_answerable:
            answer = "Insufficient information available in the provided documents."
            explanation = (
                f"Question rejected as UNANSWERABLE (Confidence: {confidence:.2f} < Threshold: {threshold:.2f}). "
                f"Evidence quality was evaluated as {answerability_res['evidence_quality']}."
            )
            verification = {
                "has_citations": False,
                "all_citations_valid": True,
                "citation_validity_ratio": 1.0
            }
            hallucination_check = {
                "is_grounded": True,
                "groundedness_score": 1.0,
                "unsupported_entities": []
            }
        else:
            prompt = build_rag_prompt(
                query=raw_query,
                passages=passages,
                query_language=detected_language
            )
            answer = self.llm.generate(prompt)

            # If the generator determined context was insufficient
            if "insufficient information" in answer.lower():
                is_answerable = False
                confidence = min(confidence, threshold - 0.05)
                explanation = "Generator verified that retrieved passages do not answer the specific query."
            else:
                top_doc = passages[0]["source"] if passages else "Unknown"
                top_page = passages[0]["page"] if passages else "N/A"
                explanation = f"Answer grounded in retrieved evidence from {top_doc} (Page {top_page}) with {confidence * 100:.1f}% confidence."

            # 5. Verification
            verification = CitationChecker.verify_citations(answer, passages)
            hallucination_check = EvidenceChecker.check_hallucination(answer, passages)

        # Format sources for UI & API
        formatted_sources = [
            {
                "chunk_id": p.get("chunk_id"),
                "source": p.get("source"),
                "document": p.get("source"),
                "page": p.get("page"),
                "page_number": p.get("page"),
                "section_title": p.get("section_title", ""),
                "score": round(float(p.get("score", 0.0)), 4),
                "text": p.get("text", "")
            }
            for p in passages
        ]

        return {
            "query": raw_query,
            "normalized_query": normalized_query,
            "detected_language": detected_language,
            "is_code_mixed": is_code_mixed,
            "codemix_concepts": codemix_concepts,
            "answerability": "ANSWERABLE" if is_answerable else "UNANSWERABLE",
            "answerable": is_answerable,
            "confidence": confidence,
            "answer": answer,
            "response": answer,
            "explanation": explanation,
            "retrieval_method": retrieval_method,
            "supporting_passages": formatted_sources,
            "sources": formatted_sources,
            "evidence_quality": answerability_res.get("evidence_quality", "Medium"),
            "verification": verification,
            "hallucination_check": hallucination_check
        }
