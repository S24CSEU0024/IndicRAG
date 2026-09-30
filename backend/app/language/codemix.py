"""
Code-Mixed (Hinglish) Processing & Query Expansion for IndicRAG
Extracts intent, maps Romanized Indic functional words to English semantic equivalents,
and provides enriched query representations to boost cross-lingual retrieval.
"""

import re
from typing import Dict, List, Tuple
from app.language.detector import HINGLISH_MARKERS


# Dictionary mapping common Romanized Indic intent and domain keywords to English glosses
HINGLISH_TO_ENGLISH_MAP: Dict[str, str] = {
    "kya": "what is",
    "kaise": "how to",
    "kab": "when",
    "kahan": "where",
    "kaha": "where",
    "kitna": "how much minimum",
    "kitni": "how much",
    "kitne": "how many",
    "kyun": "why",
    "chahiye": "required criteria",
    "milega": "awarded eligible receive",
    "milegi": "awarded eligible receive",
    "milta": "awarded granted",
    "niyam": "rules regulations",
    "shartein": "terms conditions",
    "patrata": "eligibility criteria",
    "pariksha": "examination",
    "ank": "marks score",
    "dakhila": "admission",
    "chhatravritti": "scholarship",
    "fees": "fee structure",
    "paisa": "fee refund",
    "pachees": "25",
    "pachas": "50",
    "sau": "100",
    "hoga": "will be",
    "hogi": "will be",
    "padega": "required compulsory",
    "dena": "submit provide",
    "kaat": "deduction withdrawal",
    "chhut": "concession discount rebate",
    "bhai": "sibling",
    "behen": "sibling sister",
    "fauji": "defence personnel ward",
    "army": "defence personnel ward"
}


def expand_codemixed_query(query: str) -> Tuple[str, List[str]]:
    """
    Expands a code-mixed query with translated English concepts while preserving
    original query tokens for lexical matching.
    Returns:
      (expanded_query_str, detected_concepts)
    """
    tokens = query.lower().split()
    detected_concepts = []

    # Identify English content tokens (not Hindi markers)
    english_content_words = [
        t for t in tokens
        if t not in HINGLISH_MARKERS and re.match(r'^[a-z0-9\-\.]+$', t)
    ]

    # Map Hinglish semantic triggers
    glosses = []
    for token in tokens:
        clean_tok = re.sub(r'[^a-z]', '', token)
        if clean_tok in HINGLISH_TO_ENGLISH_MAP:
            glosses.append(HINGLISH_TO_ENGLISH_MAP[clean_tok])
            detected_concepts.append(f"{clean_tok} -> {HINGLISH_TO_ENGLISH_MAP[clean_tok]}")

    # Build hybrid query for retriever
    # Preserve original tokens first, append translated keywords
    expanded_parts = [query]
    if glosses:
        expanded_parts.append(" ".join(glosses))
    if english_content_words:
        expanded_parts.append(" ".join(english_content_words))

    expanded_query = " ".join(expanded_parts)
    return expanded_query, detected_concepts
