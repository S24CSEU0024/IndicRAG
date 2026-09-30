"""
Evidence Consistency and Hallucination Checker for IndicRAG
Checks whether terms and key numerical values in the generated answer
actually exist in the retrieved evidence passages.
"""

import re
from typing import List, Dict, Any


class EvidenceChecker:
    @staticmethod
    def extract_numbers_and_entities(text: str) -> List[str]:
        """Extract numbers, percentages, CGPA values, and proper entities."""
        # Find percentages, decimals, years, currency, thresholds
        patterns = [
            r'\b[0-9]+(?:\.[0-9]+)?%',  # 75%, 8.5%
            r'\b(?:cgpa|sgpa)\s*(?:of\s*)?[0-9]+(?:\.[0-9]+)?\b',
            r'\b[0-9]+(?:\.[0-9]+)?\b',  # raw numbers
            r'\b(?:₹|rs\.?)\s*[0-9,]+',  # monetary
        ]
        entities = []
        for p in patterns:
            matches = re.findall(p, text, re.IGNORECASE)
            entities.extend(matches)
        return list(set(entities))

    @staticmethod
    def check_hallucination(
        answer: str,
        passages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Flags potential numerical or factual hallucinations not grounded in passages.
        """
        combined_evidence = " ".join([p.get("text", "") for p in passages]).lower()
        answer_entities = EvidenceChecker.extract_numbers_and_entities(answer)

        unsupported_entities = []
        for entity in answer_entities:
            clean_ent = entity.strip().lower()
            if clean_ent not in combined_evidence:
                unsupported_entities.append(entity)

        is_grounded = len(unsupported_entities) == 0
        groundedness_score = 1.0 - (len(unsupported_entities) / max(1, len(answer_entities)))

        return {
            "is_grounded": is_grounded,
            "groundedness_score": round(max(0.0, groundedness_score), 3),
            "unsupported_entities": unsupported_entities,
            "total_entities_checked": len(answer_entities)
        }
