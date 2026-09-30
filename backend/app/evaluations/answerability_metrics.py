"""
Answerability Evaluation Metrics for IndicRAG
Computes Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
"""

from typing import List, Dict, Any


def compute_answerability_metrics(
    predictions: List[str],  # "ANSWERABLE" or "UNANSWERABLE"
    ground_truths: List[str]
) -> Dict[str, Any]:
    """
    Computes classification metrics for answerability detection.
    Treats ANSWERABLE as positive (1) and UNANSWERABLE as negative (0).
    """
    if len(predictions) != len(ground_truths) or len(predictions) == 0:
        return {}

    tp = 0  # Predicted Answerable, Actually Answerable
    fp = 0  # Predicted Answerable, Actually Unanswerable (Hallucination risk)
    tn = 0  # Predicted Unanswerable, Actually Unanswerable (Correct rejection)
    fn = 0  # Predicted Unanswerable, Actually Answerable (False rejection)

    for p, g in zip(predictions, ground_truths):
        p_ans = (p.upper() == "ANSWERABLE")
        g_ans = (g.upper() == "ANSWERABLE")

        if p_ans and g_ans:
            tp += 1
        elif p_ans and not g_ans:
            fp += 1
        elif not p_ans and not g_ans:
            tn += 1
        else:
            fn += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Rejection accuracy (correctly rejecting unanswerable queries)
    rejection_rate = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1": round(f1, 4),
        "RejectionRate": round(rejection_rate, 4),
        "ConfusionMatrix": {
            "TruePositive": tp,
            "FalsePositive": fp,
            "TrueNegative": tn,
            "FalseNegative": fn
        },
        "TotalQueries": total
    }
