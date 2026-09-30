"""
Language Detector for IndicRAG
Classifies queries into:
  - 'English'
  - 'Indic' (e.g. Hindi in Devanagari script)
  - 'Code-Mixed' (Indic-English romanized code-mixing / Hinglish)
"""

import re
from typing import Dict, Any

# Extensive vocabulary of Romanized Hindi / Hinglish markers and grammatical particles
HINGLISH_MARKERS = {
    "kya", "hai", "hain", "ke", "ki", "ka", "ko", "se", "mein", "me", "aur", "bhi",
    "liye", "hota", "hote", "hoti", "hoga", "hogi", "honge", "kaise", "kab", "kaha",
    "kahan", "kitna", "kitni", "kitne", "kyun", "kyu", "wala", "wali", "wale", "kare",
    "karna", "chahiye", "chaheye", "milega", "milegi", "milta", "milti", "sakte", "sakti",
    "sakta", "nhi", "nahi", "nahin", "agr", "agar", "toh", "to", "par", "pe", "unka",
    "inka", "apna", "apni", "apne", "krna", "karein", "dene", "lena", "diya", "baad",
    "pehle", "sirf", "batao", "bataiye", "bata", "kuch", "sabse", "kam", "jyada", "zyada",
    "aise", "waise", "hume", "hame", "humko", "muje", "mujhe", "mera", "meri", "mere",
    "hota", "hoti", "kr", "kar", "rakha", "rakh", "rha", "raha", "rahi", "rahe", "thha", "tha", "thi", "the"
}

DEVANAGARI_REGEX = re.compile(r'[\u0900-\u097F]')


def detect_language(query: str) -> Dict[str, Any]:
    """
    Detects whether the query is English, Indic (Devanagari), or Code-Mixed (Hinglish).
    Returns dict:
      {
        "language": "English" | "Indic" | "Code-Mixed",
        "script": "Latin" | "Devanagari" | "Mixed",
        "confidence": float,
        "is_code_mixed": bool
      }
    """
    cleaned = query.strip()
    if not cleaned:
        return {
            "language": "English",
            "script": "Latin",
            "confidence": 1.0,
            "is_code_mixed": False
        }

    # Check for Devanagari characters
    devanagari_chars = len(DEVANAGARI_REGEX.findall(cleaned))
    total_letters = len(re.findall(r'[\w]', cleaned))

    if total_letters > 0 and (devanagari_chars / total_letters) > 0.25:
        # Check if it has a mix of Latin and Devanagari
        latin_chars = len(re.findall(r'[a-zA-Z]', cleaned))
        if latin_chars > 2 and devanagari_chars > 2:
            return {
                "language": "Code-Mixed",
                "script": "Mixed",
                "confidence": 0.92,
                "is_code_mixed": True
            }
        return {
            "language": "Indic",
            "script": "Devanagari",
            "confidence": 0.98,
            "is_code_mixed": False
        }

    # Lowercase tokenized words
    tokens = re.findall(r'[a-zA-Z]+', cleaned.lower())
    if not tokens:
        return {
            "language": "English",
            "script": "Latin",
            "confidence": 0.8,
            "is_code_mixed": False
        }

    # Count Hinglish marker matches
    hinglish_matches = [t for t in tokens if t in HINGLISH_MARKERS]
    match_ratio = len(hinglish_matches) / len(tokens)

    if len(hinglish_matches) >= 1 and (match_ratio >= 0.15 or len(hinglish_matches) >= 2):
        return {
            "language": "Code-Mixed",
            "script": "Latin",
            "confidence": min(0.99, 0.70 + match_ratio * 0.3),
            "is_code_mixed": True,
            "matched_markers": hinglish_matches
        }

    # Check for common phonetic code-mixed patterns (e.g. -ing, -ly with Hindi stem or verb suffixes)
    for t in tokens:
        if t.endswith(("karoge", "karenge", "dega", "degi", "aayega", "aayegi")):
            return {
                "language": "Code-Mixed",
                "script": "Latin",
                "confidence": 0.88,
                "is_code_mixed": True,
                "matched_markers": [t]
            }

    return {
        "language": "English",
        "script": "Latin",
        "confidence": 0.95,
        "is_code_mixed": False
    }
