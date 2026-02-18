"""
AI Evaluation Framework

Author: Gaurav Mittal
Organization: Thermo Fisher Scientific
Created: 2026-02
Description:
  Orchestrates data-driven evaluation of AI responses
  across multiple providers (OpenAI, DataIKU).
"""

import argparse
import os
import datetime

from method_1_xlsx_to_json import run as step1
from method_2_json_to_jsonl import run as step2
from method_3_run_model import run as step3
from method_4_score_embeddings import run as step4
from method_6_score import run as step6
from method_5_offline_report import run as step5
from method_7_hallucination_tracebility import run as step7

from sendEmail import send_report_email


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True, help="Input XLSX file containing prompts and expected responses")
    ap.add_argument("--sheet", default="SVA-Mini")
    ap.add_argument("--reference-col", default="Expected Response")
    ap.add_argument("--model", default="gpt-5-mini")
    ap.add_argument("--project-folder",default=None,help="Path to project config folder (e.g. ./configs/MCP, ./configs/DataIKU)")
    ap.add_argument("--threshold", type=float, default=0.80)
    ap.add_argument("--deepeval-threshold", type=float, default=None, help="Pass threshold for DeepEval (defaults to --threshold)")
    ap.add_argument("--deepeval-model", default="gpt-4o-mini", help="LLM judge model used by DeepEval GEval")
    ap.add_argument("--out-dir", default="output", help="Base output folder (date subfolder will be created)")
    ap.add_argument("--env-file", default=".env", help="Path to .env containing SMTP settings")
    args = ap.parse_args()

    print("🚀 Starting Evaluation Pipeline")

    # Inputs
    xlsx = args.xlsx

    # Output folder (date-wise)
    run_date = datetime.datetime.now().strftime("%Y-%m-%d")
    out_dir = os.path.join(args.out_dir, run_date)
    os.makedirs(out_dir, exist_ok=True)
    print(f"📁 Output directory: {out_dir}")

    # Output files
    json_path = os.path.join(out_dir, "scenarios.json")
    jsonl_path = os.path.join(out_dir, "scenarios.jsonl")
    answers_jsonl = os.path.join(out_dir, "responses.jsonl")
    answers_json = os.path.join(out_dir, "responses.json")
    results_jsonl = os.path.join(out_dir, "results.jsonl")
    results_json = os.path.join(out_dir, "results.json")
    report_html = os.path.join(out_dir, "report_offline.html")

    # Method 1: XLSX → JSON
    print("🔹 Method 1: XLSX → JSON")
    step1(xlsx, json_path, sheet=args.sheet, reference_col=args.reference_col)
    print(f"   ✔ Wrote {json_path}")

    # Method 2: JSON → JSONL
    print("🔹 Method 2: JSON → JSONL")
    step2(json_path, jsonl_path)
    print(f"   ✔ Wrote {jsonl_path}")

    # orchestrator.py (snippet)
    
    # Method 3: Model Invocation
    print("🔹 Method 3: Model Invocation")
    step3(
        canonical_json=json_path,
        provider=None,
        out_answers_jsonl=answers_jsonl,
        out_answers_json=answers_json,
        model=args.model,
        project_folder=args.project_folder,
    )
    print(f"   ✔ Wrote {answers_jsonl} and {answers_json}")

    # Method 4: Embedding Similarity
    print("🔹 Method 4: Embedding Similarity (OpenAI)")
    step4(
        canonical_json=json_path,
        answers_jsonl=answers_jsonl,
        out_results_jsonl=results_jsonl,
        out_results_json=results_json,
        threshold=args.threshold,
    )
    print(f"   ✔ Wrote {results_json}")


    # Method 6: DeepEval (LLM-judge scoring)
    print("🔹 Method 6: DeepEval Scoring")
    deepeval_threshold = args.deepeval_threshold if args.deepeval_threshold is not None else args.threshold
    step6(
        canonical_json=json_path,
        results_json=results_json,
        out_results_json=results_json,
        out_results_jsonl=results_jsonl,
        threshold=deepeval_threshold,
        judge_model=args.deepeval_model,
    )
    print(f"   ✔ Updated {results_json} with DeepEval scores")
    
    print("🔹 Method 7: Hallucination + Traceability")
    step7(
        results_json=results_json,
        out_results_json=results_json,
        out_results_jsonl=results_jsonl,
        judge_model="gpt-4o-mini",
    )

    # Method 5: Offline Report
    print("🔹 Method 5: Offline KPI Report")
    step5(results_json, report_html)
    print(f"   ✔ Wrote {report_html}")

    print("✅ Evaluation pipeline completed")

    # Always send email with report attachment (today’s output folder)
    print("📧 Sending email with offline report attachment...")
    send_report_email(
        subject_suffix="REPORT",
        out_base_dir=args.out_dir,
        env_file=args.env_file,
        body_lines=[
            "AI Evaluation Framework - Report Notification",
            "",
            f"Timestamp : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Model     : {args.model}",
            f"Threshold : {args.threshold}",
            f"Output    : {out_dir}",
            "",
            "Attachment: report_offline.html (offline HTML report)",
        ],
    )


if __name__ == "__main__":
    main()
