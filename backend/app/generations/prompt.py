"""
Prompt Templates for Evidence-Grounded Cross-Lingual RAG
Ensures strict contextual grounding, passage citation, and unanswerability handling.
"""

from typing import List, Dict, Any


def format_evidence_passages(passages: List[Dict[str, Any]]) -> str:
    """Format passages with citation indices and source metadata."""
    formatted_blocks = []
    for i, p in enumerate(passages, start=1):
        source = p.get("source", "Document")
        page = p.get("page", 1)
        text = p.get("text", "").strip()
        formatted_blocks.append(
            f"[Passage {i}] (Source: {source}, Page: {page})\n{text}"
        )
    return "\n\n".join(formatted_blocks)


def build_rag_prompt(
    query: str,
    passages: List[Dict[str, Any]],
    query_language: str = "English"
) -> str:
    """
    Builds strict evidence-grounded prompt for RAG answer generation.
    """
    evidence_text = format_evidence_passages(passages)

    language_instruction = ""
    if query_language == "Indic":
        language_instruction = "The query is in an Indian language (e.g., Hindi). Answer clearly in Hindi or English, grounded strictly in the provided documents."
    elif query_language == "Code-Mixed":
        language_instruction = "The query is in Code-Mixed format (e.g., Hinglish). Answer clearly, addressing the user's intent, grounded strictly in the provided documents."

    prompt = f"""You are IndicRAG, an authoritative university regulation and policy question-answering assistant.
Your task is to answer the user's question based EXCLUSIVELY on the retrieved evidence passages provided below.

CRITICAL INSTRUCTIONS:
1. Ground every claim directly in the provided passages. Cite the passage index (e.g., [Passage 1] or [1]) for each piece of information.
2. If the provided passages do NOT contain sufficient information to answer the question accurately, you MUST reply:
   "Insufficient information available in the provided documents."
3. Do NOT hallucinate, guess, or use external knowledge not present in the context.
4. Keep the answer clear, structured, and factual.
{language_instruction}

RETRIEVED EVIDENCE PASSAGES:
{evidence_text}

USER QUESTION:
{query}

GROUNDED ANSWER (with citations):"""
    return prompt


def build_direct_llm_prompt(query: str) -> str:
    """
    Builds direct prompt WITHOUT retrieved context for Module 4 comparative analysis.
    """
    prompt = f"""You are an AI assistant. Answer the following question about Bennett University regulations and policies based on your general internal knowledge.

USER QUESTION:
{query}

ANSWER:"""
    return prompt
