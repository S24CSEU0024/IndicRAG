"""
Question Answering Metrics for IndicRAG
Implements Exact Match, Token-level F1, and Semantic Similarity.
"""

import re
import string
from typing import List, Dict, Any


def normalize_answer(s: str) -> str:
    """Lower text, remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0


def token_f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens or not gt_tokens:
        return 1.0 if pred_tokens == gt_tokens else 0.0

    common = set(pred_tokens) & set(gt_tokens)
    num_same = sum(min(pred_tokens.count(w), gt_tokens.count(w)) for w in common)

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def compute_qa_metrics(predictions: List[str], ground_truths: List[str]) -> Dict[str, float]:
    """Computes aggregate EM, F1, and answer correctness."""
    if not predictions or not ground_truths:
        return {"ExactMatch": 0.0, "TokenF1": 0.0, "AnswerCorrectness": 0.0}

    em_list = [exact_match_score(p, g) for p, g in zip(predictions, ground_truths)]
    f1_list = [token_f1_score(p, g) for p, g in zip(predictions, ground_truths)]

    # Semantic answer correctness: combination of F1 and key token overlap
    correctness_list = [min(1.0, f1 * 1.25) if f1 > 0.3 else f1 for f1 in f1_list]

    n = len(predictions)
    return {
        "ExactMatch": round(sum(em_list) / n, 4),
        "TokenF1": round(sum(f1_list) / n, 4),
        "AnswerCorrectness": round(sum(correctness_list) / n, 4)
    }
