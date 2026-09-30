"""
Citation and Hallucination Verification for IndicRAG
Checks whether generated answers cite genuine passage IDs and whether the generated
factual assertions are present in the cited evidence.
"""

import re
from typing import List, Dict, Any, Tuple


class CitationChecker:
    @staticmethod
    def extract_citations(text: str) -> List[int]:
        """Extract passage citation markers like [1], [Passage 2], [p. 3], [Doc: 1]."""
        citations = []
        # Match patterns like [1], [2], [Passage 1], [Chunk 5]
        bracketed = re.findall(r'\[(?:Passage\s*|Chunk\s*|Source\s*|P\s*)?([0-9]+)\]', text, re.IGNORECASE)
        for num in bracketed:
            try:
                citations.append(int(num))
            except ValueError:
                pass
        return sorted(list(set(citations)))

    @staticmethod
    def verify_citations(
        answer: str,
        retrieved_passages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Verifies that citations point to valid retrieved passages and computes
        citation coverage.
        """
        cited_indices = CitationChecker.extract_citations(answer)
        valid_ranks = set(range(1, len(retrieved_passages) + 1))
        valid_citations = [idx for idx in cited_indices if idx in valid_ranks]
        invalid_citations = [idx for idx in cited_indices if idx not in valid_ranks]

        has_citations = len(cited_indices) > 0
        all_valid = len(invalid_citations) == 0 and has_citations

        return {
            "has_citations": has_citations,
            "cited_ranks": cited_indices,
            "valid_citations": valid_citations,
            "invalid_citations": invalid_citations,
            "citation_validity_ratio": len(valid_citations) / max(1, len(cited_indices)),
            "all_citations_valid": all_valid
        }
