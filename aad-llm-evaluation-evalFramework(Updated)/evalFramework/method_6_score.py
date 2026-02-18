import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI


def _try_import_deepeval():
    """
    DeepEval is an optional dependency in this repo.
    We import lazily so the rest of the pipeline can still run without it.
    """
    try:
        # Common path (DeepEval 1.x+)
        from deepeval.metrics import GEval
    except Exception:
        # Some versions expose it here
        from deepeval.metrics.g_eval import GEval

    try:
        from deepeval.test_case import LLMTestCase, LLMTestCaseParams
    except Exception as e:
        raise RuntimeError(
            "DeepEval is installed but could not import required symbols "
            "(LLMTestCase / LLMTestCaseParams). Please check your DeepEval version."
        ) from e

    return GEval, LLMTestCase, LLMTestCaseParams


def _normalize_score(score: Any) -> Optional[float]:
    """
    DeepEval scores are commonly 0-1, but some setups can return 0-10.
    Normalize to 0-1 when needed.
    """
    try:
        s = float(score)
    except Exception:
        return None

    if s > 1.0:
        return max(0.0, min(1.0, s / 10.0))
    return max(0.0, min(1.0, s))


def run(
    canonical_json: str,
    results_json: str,
    out_results_json: str,
    out_results_jsonl: Optional[str] = None,
    threshold: float = 0.80,
    judge_model: str = "gpt-4o-mini",
) -> str:
    """
    Method #6: DeepEval Scoring
    - Reads `results.json` (array) produced by Method #4 (embeddings scoring).
    - Adds `deepeval_score` (0..1) and `deepeval_passed` fields per row.
    - Writes updated JSON (and optional JSONL).

    NOTE: This uses DeepEval's GEval (LLM-as-a-judge) against the Expected Response.
    """
    GEval, LLMTestCase, LLMTestCaseParams = _try_import_deepeval()

    p = Path(results_json)
    if not p.exists():
        raise FileNotFoundError(f"Results JSON not found: {results_json}")

    # Load scenarios (for prompt/reference fallback)
    with open(canonical_json, "r", encoding="utf-8") as fh:
        scenarios = {s["id"]: s for s in json.load(fh)["scenarios"]}

    with open(p, "r", encoding="utf-8") as fh:
        rows: List[Dict[str, Any]] = json.load(fh)

    # Configure judge metric
    metric = GEval(
        name="DeepEval",
        criteria=(
            "Evaluate whether the actual response correctly and completely "
            "matches the expected response."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        threshold=threshold,
        model=judge_model,  # <-- FIX: use judge_model, not undefined 'model'
    )

    # Keep OpenAI client instantiation consistent with other methods
    _ = OpenAI()

    updated: List[Dict[str, Any]] = []
    jsonl_lines: List[str] = []

    for r in rows:
        sid = r.get("id", "")
        prompt = r.get("prompt") or scenarios.get(sid, {}).get("prompt", "")
        answer = r.get("answer") or ""
        reference = r.get("reference") or scenarios.get(sid, {}).get("reference")

        deepeval_score = None
        deepeval_passed = None

        if reference:
            tc = LLMTestCase(
                input=prompt,
                actual_output=answer,
                expected_output=reference,
            )
            try:
                metric.measure(tc)
                deepeval_score = _normalize_score(getattr(metric, "score", None))
                deepeval_passed = (
                    deepeval_score is not None and deepeval_score >= float(threshold)
                )
            except Exception:
                deepeval_score = None
                deepeval_passed = None

        row2 = {
            **r,
            "deepeval_score": deepeval_score,
            "deepeval_threshold": threshold,
            "deepeval_passed": deepeval_passed,
        }
        updated.append(row2)
        jsonl_lines.append(json.dumps(row2, ensure_ascii=False))

    with open(out_results_json, "w", encoding="utf-8") as f:
        json.dump(updated, f, indent=2, ensure_ascii=False)

    if out_results_jsonl:
        with open(out_results_jsonl, "w", encoding="utf-8") as f:
            f.write("\n".join(jsonl_lines) + "\n")

    return out_results_json


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical-json", required=True)
    ap.add_argument("--results-json", required=True)
    ap.add_argument("--out-results-json", required=True)
    ap.add_argument("--out-results-jsonl", default=None)
    ap.add_argument("--threshold", type=float, default=0.80)
    ap.add_argument("--judge-model", default="gpt-4o-mini")
    args = ap.parse_args()

    run(
        canonical_json=args.canonical_json,
        results_json=args.results_json,
        out_results_json=args.out_results_json,
        out_results_jsonl=args.out_results_jsonl,
        threshold=args.threshold,
        judge_model=args.judge_model,
    )
    print(f"[DONE] Wrote DeepEval-scored results: {args.out_results_json}")
