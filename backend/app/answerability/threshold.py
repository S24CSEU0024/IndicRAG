"""
Threshold Configuration and Dynamic Calibration for Answerability Detection.
"""

from app.config import ANSWERABILITY_THRESHOLD

# Calibrated confidence thresholds
DEFAULT_CONFIDENCE_THRESHOLD = ANSWERABILITY_THRESHOLD  # 0.35
LEXICAL_OVERLAP_MIN = 0.08
HIGH_CONFIDENCE_THRESHOLD = 0.65


def is_confident_enough(
    max_retrieval_score: float,
    semantic_entailment_score: float,
    lexical_overlap: float,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
) -> bool:
    """
    Combined confidence rule:
    Requires both semantic relevance and minimum query concept coverage.
    """
    composite_confidence = (
        0.55 * max_retrieval_score +
        0.30 * semantic_entailment_score +
        0.15 * lexical_overlap
    )
    return composite_confidence >= threshold
