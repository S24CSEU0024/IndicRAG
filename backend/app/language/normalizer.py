"""
Text & Query Normalizer for IndicRAG
Handles spelling variation, transliteration variations, slang normalization,
and query cleaning for English, Indic, and Code-Mixed queries.
"""

import re
from typing import Dict

# Common Romanized spelling variations and phonetic slang in Indian English/Hinglish
SPELLING_NORMALIZATION: Dict[str, str] = {
    # Spelling errors & abbreviations
    "schlrshp": "scholarship",
    "skolrship": "scholarship",
    "skolerchip": "scholarship",
    "scholarshil": "scholarship",
    "scholarshipp": "scholarship",
    "eligiblity": "eligibility",
    "elegibility": "eligibility",
    "elegiblity": "eligibility",
    "elgibility": "eligibility",
    "critaria": "criteria",
    "criterias": "criteria",
    "crteria": "criteria",
    "attendence": "attendance",
    "attendence": "attendance",
    "atendance": "attendance",
    "attn": "attendance",
    "exm": "examination",
    "exam": "examination",
    "exams": "examinations",
    "docmnt": "document",
    "documnts": "documents",
    "docs": "documents",
    "univ": "university",
    "clg": "college",
    "cgpa": "CGPA",
    "sgpa": "SGPA",
    "ufm": "unfair means",
    "cheating": "unfair means",
    "plag": "plagiarism",
    "min": "minimum",
    "max": "maximum",
    "pct": "percentage",
    "percent": "percentage",
    "req": "required",
    "reqd": "required",
    "cert": "certificate",
    # Hinglish common words normalization
    "kya": "kya",
    "kiya": "kiya",
    "kaise": "kaise",
    "kese": "kaise",
    "chahiye": "chahiye",
    "chaheye": "chahiye",
    "chahye": "chahiye",
    "milega": "milega",
    "milga": "milega",
    "kitna": "kitna",
    "ktna": "kitna",
    "batao": "batao",
    "btaye": "bataiye",
    "bataiye": "bataiye",
    "btao": "batao"
}


def normalize_query(query: str) -> str:
    """
    Cleans punctuation, normalizes common slang/typos, and standardizes spacing.
    """
    cleaned = query.strip()
    if not cleaned:
        return ""

    # Replace multiple question marks/exclamations
    cleaned = re.sub(r'[\?!]+', ' ? ', cleaned)
    cleaned = re.sub(r'[,;:]+', ' ', cleaned)

    # Word-level replacement
    tokens = cleaned.split()
    normalized_tokens = []
    for token in tokens:
        lower_token = token.lower()
        if lower_token in SPELLING_NORMALIZATION:
            normalized_tokens.append(SPELLING_NORMALIZATION[lower_token])
        else:
            normalized_tokens.append(token)

    result = " ".join(normalized_tokens)
    result = re.sub(r'\s+', ' ', result).strip()
    return result
