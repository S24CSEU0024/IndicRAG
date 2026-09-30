"""
CLI Benchmark Script for IndicRAG
Runs evaluation across Modules 1-6 and displays formatted comparison tables.
"""

import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.evaluations.evaluate import BenchmarkRunner
from app.config import EVALUATION_DIR


def print_table(title: str, headers: list, rows: list):
    print(f"\n=== {title} ===")
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))

    header_line = " | ".join(f"{h:<{col_widths[i]}}" for i, h in enumerate(headers))
    sep_line = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
    print(header_line)
    print(sep_line)
    for row in rows:
        print(" | ".join(f"{str(v):<{col_widths[i]}}" for i, v in enumerate(row)))


def main():
    runner = BenchmarkRunner()
    print("Running Full IndicRAG Evaluation Suite...")
    results = runner.run_all(sample_size=120)

    # Save to JSON
    out_file = EVALUATION_DIR / "evaluation_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nFull evaluation results saved to: {out_file}")

    # ================= Module 1 Table =================
    m1 = results["module_1"]
    m1_headers = ["Retriever", "Recall@1", "Recall@3", "Recall@5", "Precision@5", "HitRate@5", "MRR"]
    m1_rows = []
    for model in ["TF-IDF", "BM25", "Multilingual_Dense"]:
        metrics = m1[model]
        m1_rows.append([
            model,
            metrics.get("Recall@1", 0),
            metrics.get("Recall@3", 0),
            metrics.get("Recall@5", 0),
            metrics.get("Precision@5", 0),
            metrics.get("HitRate@5", 0),
            metrics.get("MRR", 0)
        ])
    print_table("Module 1: Lexical vs Multilingual Dense Retrieval", m1_headers, m1_rows)

    # ================= Module 2 Table =================
    m2 = results["module_2"]["results_by_language"]
    m2_headers = ["Query Language Type", "Queries Evaluated", "Recall@5", "HitRate@5", "MRR"]
    m2_rows = [
        [lang, data["query_count"], data["Recall@5"], data["HitRate@5"], data["MRR"]]
        for lang, data in m2.items()
    ]
    print_table("Module 2: Cross-Lingual & Code-Mixed Performance", m2_headers, m2_rows)

    # ================= Module 3 Table =================
    m3 = results["module_3"]["alpha_sweep"]
    m3_headers = ["Retrieval Mode", "Alpha (α)", "Recall@5", "HitRate@5", "MRR"]
    m3_rows = [
        [mode, data["alpha"], data["Recall@5"], data["HitRate@5"], data["MRR"]]
        for mode, data in m3.items()
    ]
    print_table("Module 3: Hybrid Retrieval Sensitivity (Alpha Sweep)", m3_headers, m3_rows)

    # ================= Module 4 Table =================
    m4 = results["module_4"]
    m4_headers = ["Pipeline", "Exact Match", "Token F1", "Answer Correctness", "Groundedness", "Hallucination Rate"]
    m4_rows = [
        ["Direct LLM (No RAG)", m4["Direct_LLM"]["ExactMatch"], m4["Direct_LLM"]["TokenF1"], m4["Direct_LLM"]["AnswerCorrectness"], m4["Direct_LLM"]["GroundednessScore"], m4["Direct_LLM"]["HallucinationRate"]],
        ["IndicRAG (Hybrid + Citations)", m4["IndicRAG"]["ExactMatch"], m4["IndicRAG"]["TokenF1"], m4["IndicRAG"]["AnswerCorrectness"], m4["IndicRAG"]["GroundednessScore"], m4["IndicRAG"]["HallucinationRate"]]
    ]
    print_table("Module 4: Direct LLM vs Retrieval-Augmented Generation", m4_headers, m4_rows)

    # ================= Module 5 Table =================
    m5 = results["module_5"]["metrics"]
    cm = m5["ConfusionMatrix"]
    print("\n=== Module 5: Answerability & Hallucination Metrics ===")
    print(f"Accuracy:       {m5['Accuracy'] * 100:.2f}%")
    print(f"Precision:      {m5['Precision'] * 100:.2f}%")
    print(f"Recall:         {m5['Recall'] * 100:.2f}%")
    print(f"F1-Score:       {m5['F1'] * 100:.2f}%")
    print(f"Rejection Rate: {m5['RejectionRate'] * 100:.2f}%")
    print(f"Confusion Matrix: TP={cm['TruePositive']}, FP={cm['FalsePositive']}, TN={cm['TrueNegative']}, FN={cm['FalseNegative']}")

    # ================= Module 6 Table =================
    m6 = results["module_6"]
    m6_headers = ["Case ID", "Type", "Expected", "Predicted", "Confidence", "Top Source"]
    m6_rows = [
        [c["case_id"], c["type"], c["expected_answerable"], c["predicted_answerable"], f"{c['confidence']:.2f}", c["top_source"]]
        for c in m6["cases"][:10]
    ]
    print_table(f"Module 6: Qualitative Analysis ({m6['total_cases']} Difficult Cases, Accuracy: {m6['correct_handling_accuracy']*100:.1f}%)", m6_headers, m6_rows)
    print("\nBenchmark completed successfully.")


if __name__ == "__main__":
    main()
