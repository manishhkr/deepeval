import json
import re
from typing import Any, Dict, List, Optional

import pandas as pd


def _norm(s: Any) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\r", " ").replace("\n", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s


# Columns we will merge from the metrics XLSX (business-friendly names)
METRIC_COLUMNS = [
    # Scores
    "Contextual Precision_Score",
    "Contextual Recall_Score",
    "Contextual Relevancy_Score",
    "Answer Relevancy_Score",
    "Faithfulness_Score",
    "Hallucination_Score",
    "metric_1",
    "Traceability (GEval)_Score",
    # Success
    "Contextual Precision_Success",
    "Contextual Recall_Success",
    "Contextual Relevancy_Success",
    "Answer Relevancy_Success",
    "Faithfulness_Success",
    "Hallucination_Success",
    "Traceability (GEval)_Success",
    # Reasons
    "Contextual Precision_Reason",
    "Contextual Recall_Reason",
    "Contextual Relevancy_Reason",
    "Answer Relevancy_Reason",
    "Faithfulness_Reason",
    "Hallucination_Reason",
    "Traceability (GEval)_Reason",
]


def _to_key(col: str) -> str:
    """Convert the XLSX column name to a JSON-friendly key."""
    key = col.lower()
    key = key.replace(" (geval)", "_geval")
    key = key.replace(" ", "_")
    key = key.replace("__", "_")
    key = key.replace("(", "").replace(")", "")
    key = key.replace("-", "_")
    return key


def run(
    canonical_json: str,
    in_results_json: str,
    metrics_xlsx: str,
    out_results_json: str,
    out_results_jsonl: str,
    prompt_col: str = "Input",
) -> None:
    """Merge precomputed RAG-quality metrics (from XLSX) into results.json/results.jsonl.

    Matching logic:
      - Joins by normalized prompt text: XLSX[prompt_col] <-> scenarios[id].prompt OR results[].prompt

    This avoids needing additional API calls for contextual metrics.
    """
    with open(canonical_json, "r", encoding="utf-8") as fh:
        scenarios = {s["id"]: s for s in json.load(fh)["scenarios"]}

    with open(in_results_json, "r", encoding="utf-8") as fh:
        rows: List[Dict[str, Any]] = json.load(fh)

    df = pd.read_excel(metrics_xlsx)

    if prompt_col not in df.columns:
        raise ValueError(f"prompt_col='{prompt_col}' not found in XLSX. Columns: {list(df.columns)}")

    # Build lookup: normalized prompt -> metric dict
    lookup: Dict[str, Dict[str, Any]] = {}
    for _, r in df.iterrows():
        p = _norm(r.get(prompt_col))
        if not p:
            continue

        payload: Dict[str, Any] = {}
        for c in METRIC_COLUMNS:
            if c in df.columns:
                payload[_to_key(c)] = r.get(c)
        lookup[p] = payload

    # Merge into each result row
    merged = 0
    for row in rows:
        sid = row.get("id")
        sc = scenarios.get(sid, {}) if sid else {}
        prompt = _norm(row.get("prompt") or sc.get("prompt") or "")
        payload = lookup.get(prompt)

        if not payload:
            continue

        row.update(payload)
        merged += 1

    # Write output
    with open(out_results_json, "w", encoding="utf-8") as fo:
        json.dump(rows, fo, indent=2, ensure_ascii=False)

    with open(out_results_jsonl, "w", encoding="utf-8") as fo:
        for r in rows:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"✅ Merged RAG-quality metrics for {merged}/{len(rows)} prompts from: {metrics_xlsx}")
