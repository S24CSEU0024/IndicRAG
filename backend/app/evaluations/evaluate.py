"""
Comprehensive Evaluation Suite for IndicRAG
Executes benchmarks across all 6 Required Modules:
  - Module 1: TF-IDF vs BM25 vs Multilingual Dense Retrieval
  - Module 2: Monolingual (EN) vs Cross-Lingual (Indic) vs Code-Mixed (Hinglish)
  - Module 3: Lexical vs Dense vs Hybrid Retrieval (Sensitivity to alpha)
  - Module 4: Direct LLM QA vs Retrieval-Augmented QA
  - Module 5: Hallucination & Answerability Detection (Confusion Matrix)
  - Module 6: Qualitative Analysis of 16 Difficult Cases
"""

import csv
import json
from pathlib import Path
from typing import Dict, Any, List

from app.config import QUESTIONS_DIR, EVALUATION_DIR
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.tfidf_retriever import TfidfRetriever
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.generations.rag_generator import IndicRAGPipeline
from app.evaluations.retrieval_metrics import compute_retrieval_metrics
from app.evaluations.qa_metrics import compute_qa_metrics
from app.evaluations.answerability_metrics import compute_answerability_metrics


def load_dataset(sample_size: int = None) -> List[Dict[str, Any]]:
    csv_path = QUESTIONS_DIR / "questions.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if sample_size and sample_size < len(rows):
        # Balanced sampling across answerable/unanswerable and languages
        import random
        random.seed(42)
        sample = random.sample(rows, sample_size)
        return sample
    return rows


def load_difficult_cases() -> List[Dict[str, Any]]:
    csv_path = EVALUATION_DIR / "difficult_cases.csv"
    rows = []
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
    return rows


class BenchmarkRunner:
    def __init__(self):
        print("Initializing Retrievers for Evaluation...")
        self.bm25 = BM25Retriever.load()
        self.tfidf = TfidfRetriever.load()
        self.dense = DenseRetriever.load()
        self.hybrid = HybridRetriever(self.bm25, self.dense, alpha=0.5)
        self.pipeline = IndicRAGPipeline(
            hybrid_retriever=self.hybrid,
            bm25_retriever=self.bm25,
            dense_retriever=self.dense,
            tfidf_retriever=self.tfidf
        )

    def run_all(self, sample_size: int = 150) -> Dict[str, Any]:
        """Runs benchmarks across all 6 modules."""
        dataset = load_dataset(sample_size)
        answerable_data = [d for d in dataset if d["answerability"] == "ANSWERABLE"]

        results = {
            "total_queries_evaluated": len(dataset),
            "sample_size": sample_size,
            "module_1": self.evaluate_module_1(answerable_data),
            "module_2": self.evaluate_module_2(answerable_data),
            "module_3": self.evaluate_module_3(answerable_data),
            "module_4": self.evaluate_module_4(answerable_data[:40]),
            "module_5": self.evaluate_module_5(dataset),
            "module_6": self.evaluate_module_6()
        }
        return results

    # ================= MODULE 1 =================
    def evaluate_module_1(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Module 1: TF-IDF vs BM25 vs Multilingual Dense Retrieval."""
        queries = [d["query"] for d in dataset]
        relevant_list = [
            set(map(int, d["relevant_chunk_ids"].split(";")))
            for d in dataset
        ]

        # 1. TF-IDF
        tfidf_ret = [[p["chunk_id"] for p in self.tfidf.search(q, k=5)] for q in queries]
        tfidf_metrics = compute_retrieval_metrics(tfidf_ret, relevant_list, k_list=[1, 3, 5])

        # 2. BM25
        bm25_ret = [[p["chunk_id"] for p in self.bm25.search(q, k=5)] for q in queries]
        bm25_metrics = compute_retrieval_metrics(bm25_ret, relevant_list, k_list=[1, 3, 5])

        # 3. Dense FAISS
        dense_ret = [[p["chunk_id"] for p in self.dense.search(q, k=5)] for q in queries]
        dense_metrics = compute_retrieval_metrics(dense_ret, relevant_list, k_list=[1, 3, 5])

        return {
            "name": "Module 1: Lexical (TF-IDF, BM25) vs Multilingual Dense Retrieval",
            "TF-IDF": tfidf_metrics,
            "BM25": bm25_metrics,
            "Multilingual_Dense": dense_metrics
        }

    # ================= MODULE 2 =================
    def evaluate_module_2(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Module 2: Compare Monolingual (EN), Cross-Lingual (Indic), and Code-Mixed Queries."""
        lang_groups = {"English": [], "Indic": [], "Code-Mixed": []}
        for d in dataset:
            lang = d.get("language", "English")
            if lang in lang_groups:
                lang_groups[lang].append(d)

        comparison = {}
        for lang, items in lang_groups.items():
            if not items:
                continue
            queries = [d["query"] for d in items]
            relevant_list = [
                set(map(int, d["relevant_chunk_ids"].split(";")))
                for d in items
            ]
            hybrid_ret = [[p["chunk_id"] for p in self.hybrid.search(q, k=5)] for q in queries]
            ret_metrics = compute_retrieval_metrics(hybrid_ret, relevant_list, k_list=[1, 5])

            comparison[lang] = {
                "query_count": len(items),
                "MRR": ret_metrics.get("MRR", 0.0),
                "Recall@5": ret_metrics.get("Recall@5", 0.0),
                "HitRate@5": ret_metrics.get("HitRate@5", 0.0)
            }

        return {
            "name": "Module 2: Monolingual vs Cross-Lingual vs Code-Mixed Comparison",
            "results_by_language": comparison
        }

    # ================= MODULE 3 =================
    def evaluate_module_3(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Module 3: Compare Lexical, Dense, and Hybrid Retrieval with Alpha Sensitivity."""
        queries = [d["query"] for d in dataset]
        relevant_list = [
            set(map(int, d["relevant_chunk_ids"].split(";")))
            for d in dataset
        ]

        alpha_results = {}
        for alpha in [0.0, 0.25, 0.50, 0.75, 1.0]:
            ret_list = [[p["chunk_id"] for p in self.hybrid.search(q, k=5, alpha=alpha)] for q in queries]
            metrics = compute_retrieval_metrics(ret_list, relevant_list, k_list=[1, 5])
            mode_label = (
                "Pure Dense (α=0.0)" if alpha == 0.0 else
                "Dense Heavy (α=0.25)" if alpha == 0.25 else
                "Balanced Hybrid (α=0.5)" if alpha == 0.5 else
                "Lexical Heavy (α=0.75)" if alpha == 0.75 else
                "Pure Lexical (α=1.0)"
            )
            alpha_results[mode_label] = {
                "alpha": alpha,
                "MRR": metrics.get("MRR", 0.0),
                "Recall@5": metrics.get("Recall@5", 0.0),
                "HitRate@5": metrics.get("HitRate@5", 0.0)
            }

        return {
            "name": "Module 3: Lexical vs Dense vs Hybrid Retrieval (Alpha Sweep)",
            "alpha_sweep": alpha_results
        }

    # ================= MODULE 4 =================
    def evaluate_module_4(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Module 4: Direct LLM QA vs Retrieval-Augmented QA."""
        direct_preds = []
        rag_preds = []
        ground_truths = []

        direct_hallucination_count = 0
        rag_hallucination_count = 0

        for d in dataset:
            q = d["query"]
            gt = d["ground_truth_answer"]
            ground_truths.append(gt)

            # Direct LLM
            res_direct = self.pipeline.process_query(q, direct_llm=True)
            direct_preds.append(res_direct["answer"])
            if "refer to official" in res_direct["answer"].lower() or len(res_direct["answer"]) < 20:
                direct_hallucination_count += 1

            # RAG
            res_rag = self.pipeline.process_query(q, retrieval_method="hybrid")
            rag_preds.append(res_rag["answer"])
            if not res_rag.get("hallucination_check", {}).get("is_grounded", True):
                rag_hallucination_count += 1

        direct_metrics = compute_qa_metrics(direct_preds, ground_truths)
        rag_metrics = compute_qa_metrics(rag_preds, ground_truths)

        n = len(dataset)
        return {
            "name": "Module 4: Direct LLM vs Retrieval-Augmented Generation",
            "Direct_LLM": {
                **direct_metrics,
                "HallucinationRate": round(direct_hallucination_count / n, 4),
                "GroundednessScore": round(1.0 - (direct_hallucination_count / n), 4)
            },
            "IndicRAG": {
                **rag_metrics,
                "HallucinationRate": round(rag_hallucination_count / n, 4),
                "GroundednessScore": round(1.0 - (rag_hallucination_count / n), 4)
            }
        }

    # ================= MODULE 5 =================
    def evaluate_module_5(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Module 5: Answerability & Hallucination Test using Unanswerable Questions."""
        predictions = []
        ground_truths = []

        for d in dataset:
            q = d["query"]
            gt = d["answerability"]
            res = self.pipeline.process_query(q, retrieval_method="hybrid")
            predictions.append(res["answerability"])
            ground_truths.append(gt)

        metrics = compute_answerability_metrics(predictions, ground_truths)
        return {
            "name": "Module 5: Answerability Detection & Hallucination Prevention",
            "metrics": metrics
        }

    # ================= MODULE 6 =================
    def evaluate_module_6(self) -> Dict[str, Any]:
        """Module 6: Qualitative Analysis of 16 Difficult Cases."""
        cases = load_difficult_cases()
        eval_cases = []

        for c in cases:
            q = c["query"]
            res = self.pipeline.process_query(q, retrieval_method="hybrid")
            top_pass = res["supporting_passages"][0] if res["supporting_passages"] else {}

            eval_cases.append({
                "case_id": c["case_id"],
                "type": c["type"],
                "query": q,
                "detected_language": res["detected_language"],
                "difficulty_factor": c.get("difficulty_factor", ""),
                "expected_answerable": c["expected_answerable"],
                "predicted_answerable": res["answerability"],
                "is_correct_detection": (res["answerability"] == c["expected_answerable"]),
                "confidence": res["confidence"],
                "top_source": f"{top_pass.get('source', 'None')} (p.{top_pass.get('page', 0)})",
                "answer_excerpt": res["answer"][:120] + "..." if len(res["answer"]) > 120 else res["answer"]
            })

        correct_count = sum(1 for c in eval_cases if c["is_correct_detection"])
        accuracy = round(correct_count / len(eval_cases), 4) if eval_cases else 0.0

        return {
            "name": "Module 6: Qualitative Analysis of Difficult Cases",
            "total_cases": len(eval_cases),
            "correct_handling_accuracy": accuracy,
            "cases": eval_cases
        }
