"""
Evaluation API Router for IndicRAG
Serves benchmark results, ablation tables, and difficult qualitative cases.
"""

import json
from fastapi import APIRouter, HTTPException, Query
from app.config import EVALUATION_DIR
from app.evaluations.evaluate import BenchmarkRunner, load_difficult_cases

router = APIRouter(prefix="/api/evaluate", tags=["Evaluation"])


@router.get("")
async def get_evaluation_results():
    """Retrieve the latest benchmark results for all 6 modules."""
    eval_file = EVALUATION_DIR / "evaluation_results.json"
    if not eval_file.exists():
        # Run a fast benchmark if not generated yet
        runner = BenchmarkRunner()
        results = runner.run_all(sample_size=80)
        with open(eval_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        return results

    with open(eval_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


@router.post("/run")
async def trigger_evaluation(sample_size: int = Query(100, ge=10, le=350)):
    """Run evaluation benchmark on the dataset."""
    try:
        runner = BenchmarkRunner()
        results = runner.run_all(sample_size=sample_size)
        eval_file = EVALUATION_DIR / "evaluation_results.json"
        with open(eval_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        return {
            "status": "success",
            "message": f"Evaluated {sample_size} queries across all 6 modules.",
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/difficult-cases")
async def get_difficult_cases():
    """Retrieve qualitative difficult cases (Module 6)."""
    cases = load_difficult_cases()
    return {
        "total_cases": len(cases),
        "cases": cases
    }
